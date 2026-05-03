"""Tracer Provider -- Span creation and trace management (v19).

Central instance for distributed tracing:
  - Span creation with automatic parent linking
  - Context propagation (W3C traceparent)
  - Sampling (AlwaysOn, RateBased, Probabilistic)
  - In-memory trace store with retention
  - Export interface for OTLP/JSON/Console

Usage:
    tracer = TracerProvider(service_name="jarvis")
    with tracer.start_span("handle_request") as span:
        span.set_attribute("user.id", "u123")
        with tracer.start_span("call_llm") as child:
            child.set_attribute("model", "claude")
        span.set_ok()
"""

from __future__ import annotations

import contextvars
import time
from collections import deque
from typing import Any

from cognithor.telemetry.types import (
    Span,
    SpanContext,
    SpanKind,
    StatusCode,
    Trace,
)
from cognithor.utils.logging import get_logger

log = get_logger(__name__)

# Context variable for active span
_current_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "_current_span", default=None
)


# ── Sampler ──────────────────────────────────────────────────────


class Sampler:
    """Base sampler -- decides whether a trace is recorded."""

    def should_sample(self, trace_id: str, name: str) -> bool:
        return True


class AlwaysOnSampler(Sampler):
    """Records every trace."""

    pass


class AlwaysOffSampler(Sampler):
    """Records no traces."""

    def should_sample(self, trace_id: str, name: str) -> bool:
        return False


class ProbabilisticSampler(Sampler):
    """Records a fraction of traces."""

    def __init__(self, ratio: float = 1.0) -> None:
        self._ratio = max(0.0, min(1.0, ratio))

    def should_sample(self, trace_id: str, name: str) -> bool:
        # Deterministic: based on trace_id
        hash_val = int(trace_id[:8], 16) if trace_id else 0
        return (hash_val / 0x100000000) < self._ratio


class RateBasedSampler(Sampler):
    """Limits to N traces per second."""

    def __init__(self, max_per_second: float = 10.0) -> None:
        self._max = max_per_second
        self._count = 0.0
        self._last_reset = time.monotonic()

    def should_sample(self, trace_id: str, name: str) -> bool:
        now = time.monotonic()
        elapsed = now - self._last_reset
        if elapsed >= 1.0:
            self._count = 0.0
            self._last_reset = now
        if self._count < self._max:
            self._count += 1
            return True
        return False


# ── Span Processor ───────────────────────────────────────────────


class SpanProcessor:
    """Processes spans after completion."""

    def on_start(self, span: Span) -> None:
        pass

    def on_end(self, span: Span) -> None:
        pass

    def shutdown(self) -> None:
        pass


class InMemoryProcessor(SpanProcessor):
    """Stores spans in memory (for tests and local debugging)."""

    def __init__(self, max_spans: int = 10000) -> None:
        self._spans: deque[Span] = deque(maxlen=max_spans)

    def on_end(self, span: Span) -> None:
        self._spans.append(span)

    @property
    def spans(self) -> list[Span]:
        return list(self._spans)

    def clear(self) -> None:
        self._spans.clear()


class ConsoleProcessor(SpanProcessor):
    """Outputs spans to the console."""

    def on_end(self, span: Span) -> None:
        status = (
            "OK"
            if span.status_code == StatusCode.OK
            else ("ERROR" if span.status_code == StatusCode.ERROR else "UNSET")
        )
        log.info(
            "span_ended",
            name=span.name,
            trace_id=span.trace_id[:8],
            span_id=span.span_id[:8],
            duration_ms=round(span.duration_ms, 2),
            status=status,
        )


class BatchProcessor(SpanProcessor):
    """Collects spans and exports them in batches."""

    def __init__(
        self,
        exporter: SpanExporter | None = None,
        max_batch_size: int = 100,
        flush_interval_seconds: float = 5.0,
    ) -> None:
        self._exporter = exporter
        self._batch: list[Span] = []
        self._max_batch = max_batch_size
        self._flush_interval = flush_interval_seconds
        self._last_flush = time.monotonic()

    def on_end(self, span: Span) -> None:
        self._batch.append(span)
        now = time.monotonic()
        if len(self._batch) >= self._max_batch or now - self._last_flush >= self._flush_interval:
            self.flush()

    def flush(self) -> None:
        if self._batch and self._exporter:
            self._exporter.export(list(self._batch))
        self._batch.clear()
        self._last_flush = time.monotonic()

    def shutdown(self) -> None:
        self.flush()


# ── Span Exporter ────────────────────────────────────────────────


class SpanExporter:
    """Base interface for span export."""

    def export(self, spans: list[Span]) -> bool:
        return True

    def shutdown(self) -> None:
        pass


class OTLPJsonExporter(SpanExporter):
    """Exports spans in OTLP JSON format (local or remote)."""

    def __init__(self, endpoint: str = "", file_path: str = "") -> None:
        self._endpoint = endpoint
        self._file_path = file_path
        self._exported_count = 0

    def export(self, spans: list[Span]) -> bool:
        payload = self._build_payload(spans)

        if self._file_path:
            try:
                import json
                from pathlib import Path

                path = Path(self._file_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, default=str) + "\n")
            except Exception as exc:
                log.warning("otlp_file_export_error", error=str(exc))
                return False

        self._exported_count += len(spans)
        return True

    def _build_payload(self, spans: list[Span]) -> dict[str, Any]:
        """Builds OTLP ExportTraceServiceRequest."""
        resource_spans: dict[str, list[dict[str, Any]]] = {}
        for span in spans:
            svc = span.service_name
            if svc not in resource_spans:
                resource_spans[svc] = []
            resource_spans[svc].append(span.to_otlp())

        return {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": {"stringValue": svc}},
                        ],
                    },
                    "scopeSpans": [
                        {
                            "scope": {"name": "cognithor.telemetry", "version": "1.0"},
                            "spans": otlp_spans,
                        }
                    ],
                }
                for svc, otlp_spans in resource_spans.items()
            ],
        }

    @property
    def exported_count(self) -> int:
        return self._exported_count


# ── TracerProvider ───────────────────────────────────────────────


class TracerProvider:
    """Central instance for distributed tracing.

    Creates spans, manages context propagation and coordinates
    processors/exporters.
    """

    def __init__(
        self,
        service_name: str = "jarvis",
        sampler: Sampler | None = None,
        processors: list[SpanProcessor] | None = None,
    ) -> None:
        self._service_name = service_name
        self._sampler = sampler or AlwaysOnSampler()
        self._processors = processors or [InMemoryProcessor()]
        self._traces: dict[str, Trace] = {}
        self._active_spans: dict[str, Span] = {}
        self._span_count = 0
        self._dropped_count = 0
        self._resource_attributes: dict[str, str] = {
            "service.name": service_name,
            "telemetry.sdk.name": "jarvis-otel",
            "telemetry.sdk.version": "1.0",
        }

    @property
    def service_name(self) -> str:
        return self._service_name

    # ── Span Creation ────────────────────────────────────────────

    def start_span(
        self,
        name: str,
        *,
        kind: SpanKind = SpanKind.INTERNAL,
        parent: SpanContext | None = None,
        attributes: dict[str, Any] | None = None,
        links: list[SpanContext] | None = None,
    ) -> SpanContextManager:
        """Starts a new span.

        Automatic parent linking via ContextVar.
        Returns SpanContextManager for with-statement.
        """
        # Determine parent
        parent_span = _current_span.get()
        if parent:
            context = parent.child_context()
            parent_span_id = parent.span_id
        elif parent_span:
            context = parent_span.context.child_context()
            parent_span_id = parent_span.span_id
        else:
            context = SpanContext()
            parent_span_id = ""

        # Sampling
        if not self._sampler.should_sample(context.trace_id, name):
            self._dropped_count += 1
            return SpanContextManager(None, self)

        span = Span(
            name=name,
            context=context,
            parent_span_id=parent_span_id,
            kind=kind,
            attributes=attributes or {},
            service_name=self._service_name,
            resource_attributes=dict(self._resource_attributes),
        )

        # Links
        if links:
            for link_ctx in links:
                span.add_link(link_ctx)

        # Notify processors
        for proc in self._processors:
            proc.on_start(span)

        # Tracking
        self._active_spans[span.span_id] = span
        self._span_count += 1

        # Assign to trace
        if context.trace_id not in self._traces:
            self._traces[context.trace_id] = Trace(
                trace_id=context.trace_id,
                service_name=self._service_name,
            )
        self._traces[context.trace_id].add_span(span)

        return SpanContextManager(span, self)

    def _end_span(self, span: Span) -> None:
        """Ends a span (internal)."""
        span.end()
        self._active_spans.pop(span.span_id, None)
        for proc in self._processors:
            proc.on_end(span)

    # ── Context Propagation ──────────────────────────────────────

    def extract_context(self, headers: dict[str, str]) -> SpanContext | None:
        """Extracts SpanContext from HTTP headers."""
        traceparent = headers.get("traceparent", "")
        if traceparent:
            return SpanContext.from_traceparent(traceparent)
        return None

    def inject_context(self, span: Span, headers: dict[str, str]) -> None:
        """Injects SpanContext into HTTP headers."""
        headers["traceparent"] = span.context.to_traceparent()

    # ── Trace Access ─────────────────────────────────────────────

    def get_trace(self, trace_id: str) -> Trace | None:
        return self._traces.get(trace_id)

    def get_recent_traces(self, limit: int = 20) -> list[Trace]:
        traces = sorted(self._traces.values(), key=lambda t: t.started_at, reverse=True)
        return traces[:limit]

    def get_current_span(self) -> Span | None:
        return _current_span.get()

    # ── Lifecycle ────────────────────────────────────────────────

    def shutdown(self) -> None:
        for proc in self._processors:
            proc.shutdown()

    def add_processor(self, processor: SpanProcessor) -> None:
        self._processors.append(processor)

    def set_resource_attribute(self, key: str, value: str) -> None:
        self._resource_attributes[key] = value

    # ── Cleanup ──────────────────────────────────────────────────

    def cleanup(self, max_traces: int = 1000) -> int:
        """Removes old traces."""
        if len(self._traces) <= max_traces:
            return 0
        sorted_traces = sorted(self._traces.items(), key=lambda x: x[1].started_at)
        to_remove = len(self._traces) - max_traces
        for trace_id, _ in sorted_traces[:to_remove]:
            del self._traces[trace_id]
        return to_remove

    # ── Stats ────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        return {
            "service_name": self._service_name,
            "total_spans": self._span_count,
            "active_spans": len(self._active_spans),
            "traces": len(self._traces),
            "dropped": self._dropped_count,
            "processors": len(self._processors),
        }


# ── SpanContextManager ──────────────────────────────────────────


class SpanContextManager:
    """Context manager for automatic span lifecycle."""

    def __init__(self, span: Span | None, provider: TracerProvider) -> None:
        self._span = span
        self._provider = provider
        self._token: contextvars.Token | None = None  # type: ignore[type-arg]

    @property
    def span(self) -> Span | None:
        return self._span

    def __enter__(self) -> Span | _NoOpSpan:
        if self._span:
            self._token = _current_span.set(self._span)
            return self._span
        return _NoOpSpan()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._span:
            if exc_type and self._span.status_code == StatusCode.UNSET:
                self._span.set_error(str(exc_val) if exc_val else exc_type.__name__)
            elif self._span.status_code == StatusCode.UNSET:
                self._span.set_ok()
            self._provider._end_span(self._span)
            if self._token:
                _current_span.reset(self._token)

    async def __aenter__(self) -> Span | _NoOpSpan:
        return self.__enter__()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)


class _NoOpSpan:
    """Dummy span when sampling discards the span."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_attributes(self, attrs: dict[str, Any]) -> None:
        pass

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        pass

    def set_status(self, code: StatusCode, message: str = "") -> None:
        pass

    def set_ok(self) -> None:
        pass

    def set_error(self, message: str = "") -> None:
        pass

    def end(self) -> None:
        pass

    @property
    def context(self) -> SpanContext:
        return SpanContext()
