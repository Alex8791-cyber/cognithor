"""Instrumentation -- Auto-instrumentation for Cognithor modules (v19).

Automatically instruments:
  - Gateway:     Request spans, latency histograms, error counters
  - GraphEngine: Workflow spans, node spans, checkpoint events
  - A2A:         Agent-to-agent spans with cross-trace propagation
  - Browser:     Navigation spans, action spans
  - LLM:         Model calls with token metrics

All instrumentations are optional (graceful degradation).

Usage:
    from cognithor.telemetry import TelemetryHub
    hub = TelemetryHub()
    hub.instrument_gateway(gateway)
"""

from __future__ import annotations

import asyncio
import functools
import time
from typing import TYPE_CHECKING, Any

from cognithor.telemetry.metrics import MetricsProvider
from cognithor.telemetry.tracer import SpanContextManager, TracerProvider
from cognithor.telemetry.types import SpanKind
from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

log = get_logger(__name__)


# ── Decorator: Trace Function ───────────────────────────────────


def trace(
    tracer: TracerProvider,
    name: str = "",
    *,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    """Decorator that wraps a function with a span.

    Usage:
        @trace(tracer, "process_message")
        async def handle(msg):
            ...
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        span_name = name or fn.__qualname__

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.start_span(span_name, kind=kind, attributes=attributes) as span:
                try:
                    result = await fn(*args, **kwargs)
                    span.set_ok()
                    return result
                except Exception as exc:
                    span.set_error(str(exc))
                    raise

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.start_span(span_name, kind=kind, attributes=attributes) as span:
                try:
                    result = fn(*args, **kwargs)
                    span.set_ok()
                    return result
                except Exception as exc:
                    span.set_error(str(exc))
                    raise

        if asyncio.iscoroutinefunction(fn):
            return async_wrapper
        return sync_wrapper

    return decorator


# ── Decorator: Measure Latency ───────────────────────────────────


def measure(
    metrics: MetricsProvider,
    histogram_name: str,
    counter_name: str = "",
    **labels: str,
) -> Callable[..., Any]:
    """Decorator that writes latency to a histogram.

    Usage:
        @measure(metrics, "llm_latency_ms", "llm_calls_total", model="claude")
        async def call_llm(prompt):
            ...
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            try:
                result = await fn(*args, **kwargs)
                elapsed = (time.monotonic() - start) * 1000
                metrics.histogram(histogram_name, elapsed, **labels)
                if counter_name:
                    metrics.counter(counter_name, 1, status="ok", **labels)
                return result
            except Exception:
                elapsed = (time.monotonic() - start) * 1000
                metrics.histogram(histogram_name, elapsed, **labels)
                if counter_name:
                    metrics.counter(counter_name, 1, status="error", **labels)
                raise

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            try:
                result = fn(*args, **kwargs)
                elapsed = (time.monotonic() - start) * 1000
                metrics.histogram(histogram_name, elapsed, **labels)
                if counter_name:
                    metrics.counter(counter_name, 1, status="ok", **labels)
                return result
            except Exception:
                elapsed = (time.monotonic() - start) * 1000
                metrics.histogram(histogram_name, elapsed, **labels)
                if counter_name:
                    metrics.counter(counter_name, 1, status="error", **labels)
                raise

        if asyncio.iscoroutinefunction(fn):
            return async_wrapper
        return sync_wrapper

    return decorator


# ── TelemetryHub ─────────────────────────────────────────────────


class TelemetryHub:
    """Central telemetry instance -- connects tracer + metrics.

    Provides standard metrics and instrumentation.
    """

    def __init__(
        self,
        service_name: str = "jarvis",
        tracer: TracerProvider | None = None,
        metrics: MetricsProvider | None = None,
    ) -> None:
        self.tracer = tracer or TracerProvider(service_name=service_name)
        self.metrics = metrics or MetricsProvider(service_name=service_name)
        self._service_name = service_name
        self._setup_default_metrics()

    def _setup_default_metrics(self) -> None:
        """Registers standard metric descriptions."""
        self.metrics.describe("requests_total", "Total requests processed", "1")
        self.metrics.describe("request_latency_ms", "Request latency", "ms")
        self.metrics.describe("errors_total", "Total errors", "1")
        self.metrics.describe("llm_calls_total", "Total LLM calls", "1")
        self.metrics.describe("llm_latency_ms", "LLM call latency", "ms")
        self.metrics.describe("llm_tokens_total", "Total tokens processed", "1")
        self.metrics.describe("graph_executions_total", "Graph workflow executions", "1")
        self.metrics.describe("graph_execution_latency_ms", "Graph execution latency", "ms")
        self.metrics.describe("browser_actions_total", "Browser actions performed", "1")
        self.metrics.describe("a2a_messages_total", "A2A protocol messages", "1")
        self.metrics.describe("active_sessions", "Currently active sessions", "1")

    # ── Request Tracing ──────────────────────────────────────────

    def trace_request(
        self, method: str, path: str, headers: dict[str, str] | None = None
    ) -> SpanContextManager:
        """Starts a request span (SERVER).

        Extracts parent context from headers if available.
        """
        parent = None
        if headers:
            parent = self.tracer.extract_context(headers)

        return self.tracer.start_span(
            f"{method} {path}",
            kind=SpanKind.SERVER,
            parent=parent,
            attributes={"http.method": method, "http.path": path},
        )

    def trace_llm_call(self, model: str, prompt_length: int = 0) -> SpanContextManager:
        """Starts an LLM call span."""
        self.metrics.counter("llm_calls_total", 1, model=model)
        return self.tracer.start_span(
            f"llm.{model}",
            kind=SpanKind.CLIENT,
            attributes={
                "llm.model": model,
                "llm.prompt_length": prompt_length,
            },
        )

    def trace_tool_call(self, tool_name: str) -> SpanContextManager:
        """Starts a tool call span."""
        return self.tracer.start_span(
            f"tool.{tool_name}",
            kind=SpanKind.CLIENT,
            attributes={"tool.name": tool_name},
        )

    def trace_graph_execution(self, graph_name: str) -> SpanContextManager:
        """Starts a graph execution span."""
        self.metrics.counter("graph_executions_total", 1, graph=graph_name)
        return self.tracer.start_span(
            f"graph.{graph_name}",
            kind=SpanKind.INTERNAL,
            attributes={"graph.name": graph_name},
        )

    def trace_a2a_message(
        self, remote_agent: str, direction: str = "outbound"
    ) -> SpanContextManager:
        """Starts an A2A message span."""
        kind = SpanKind.CLIENT if direction == "outbound" else SpanKind.SERVER
        self.metrics.counter("a2a_messages_total", 1, agent=remote_agent, direction=direction)
        return self.tracer.start_span(
            f"a2a.{direction}.{remote_agent}",
            kind=kind,
            attributes={
                "a2a.remote_agent": remote_agent,
                "a2a.direction": direction,
            },
        )

    def trace_browser_action(self, action: str, url: str = "") -> SpanContextManager:
        """Starts a browser action span."""
        self.metrics.counter("browser_actions_total", 1, action=action)
        return self.tracer.start_span(
            f"browser.{action}",
            kind=SpanKind.INTERNAL,
            attributes={"browser.action": action, "browser.url": url},
        )

    # ── Metric Shortcuts ─────────────────────────────────────────

    def record_request(self, method: str, status: int, latency_ms: float) -> None:
        """Records request metrics."""
        self.metrics.counter("requests_total", 1, method=method, status=str(status))
        self.metrics.histogram("request_latency_ms", latency_ms, method=method)
        if status >= 400:
            self.metrics.counter("errors_total", 1, method=method, status=str(status))

    def record_llm_usage(
        self, model: str, latency_ms: float, input_tokens: int = 0, output_tokens: int = 0
    ) -> None:
        """Records LLM usage."""
        self.metrics.histogram("llm_latency_ms", latency_ms, model=model)
        if input_tokens:
            self.metrics.counter("llm_tokens_total", input_tokens, model=model, direction="input")
        if output_tokens:
            self.metrics.counter("llm_tokens_total", output_tokens, model=model, direction="output")

    def record_graph_execution(
        self, graph_name: str, latency_ms: float, status: str = "completed"
    ) -> None:
        """Records graph execution metrics."""
        self.metrics.histogram("graph_execution_latency_ms", latency_ms, graph=graph_name)
        self.metrics.counter("graph_executions_total", 1, graph=graph_name, status=status)

    def set_active_sessions(self, count: int) -> None:
        self.metrics.gauge("active_sessions", count)

    # ── Dashboard ────────────────────────────────────────────────

    def dashboard_snapshot(self) -> dict[str, Any]:
        """Snapshot for dashboard."""
        return {
            "service": self._service_name,
            "tracer": self.tracer.stats(),
            "metrics": self.metrics.snapshot(),
            "recent_traces": [t.to_dict() for t in self.tracer.get_recent_traces(10)],
        }

    # ── Lifecycle ────────────────────────────────────────────────

    def shutdown(self) -> None:
        self.tracer.shutdown()

    def stats(self) -> dict[str, Any]:
        return {
            "tracer": self.tracer.stats(),
            "metrics": self.metrics.stats(),
        }
