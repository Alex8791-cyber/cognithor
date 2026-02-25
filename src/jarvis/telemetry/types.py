"""OpenTelemetry Types — v19.

OTLP-kompatible Datenmodelle für Distributed Tracing und Metrics.
Folgt der OpenTelemetry Specification (Traces + Metrics).

Kern-Konzepte:
  - TraceId / SpanId:  128/64-bit Hex-IDs
  - SpanContext:       Propagation-Kontext (TraceId + SpanId + Flags)
  - Span:              Einzelne Operation mit Timing, Status, Events
  - SpanKind:          CLIENT, SERVER, INTERNAL, PRODUCER, CONSUMER
  - Trace:             Baum von Spans (vollständige Request-Kette)
  - Metric:            Counter, Histogram, Gauge (OTLP-kompatibel)
"""

from __future__ import annotations

import os
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any


# ── ID Generation ────────────────────────────────────────────────

def generate_trace_id() -> str:
    """Generiert eine 32-Hex-Char Trace-ID (128 bit)."""
    return uuid.uuid4().hex

def generate_span_id() -> str:
    """Generiert eine 16-Hex-Char Span-ID (64 bit)."""
    return os.urandom(8).hex()


# ── Enums ────────────────────────────────────────────────────────

class SpanKind(IntEnum):
    """Art des Spans (OpenTelemetry Spec)."""
    INTERNAL = 0   # Default: Interne Operation
    SERVER = 1     # Eingehender Request
    CLIENT = 2     # Ausgehender Request
    PRODUCER = 3   # Nachricht senden (async)
    CONSUMER = 4   # Nachricht empfangen (async)


class StatusCode(IntEnum):
    """Status eines Spans (OpenTelemetry Spec)."""
    UNSET = 0
    OK = 1
    ERROR = 2


class MetricKind(str, Enum):
    """Art einer Metrik."""
    COUNTER = "counter"           # Monoton steigend
    UP_DOWN_COUNTER = "up_down"   # Kann steigen/fallen
    HISTOGRAM = "histogram"       # Verteilung
    GAUGE = "gauge"               # Aktueller Wert


# ── SpanContext ──────────────────────────────────────────────────

@dataclass
class SpanContext:
    """Propagation-Kontext für Distributed Tracing.

    Wird über HTTP-Header (traceparent) oder gRPC-Metadata propagiert.
    Format: 00-{trace_id}-{span_id}-{trace_flags}
    """
    trace_id: str = ""
    span_id: str = ""
    trace_flags: int = 1  # 1 = sampled
    is_remote: bool = False

    def __post_init__(self) -> None:
        if not self.trace_id:
            self.trace_id = generate_trace_id()
        if not self.span_id:
            self.span_id = generate_span_id()

    @property
    def is_valid(self) -> bool:
        return bool(self.trace_id and self.span_id)

    @property
    def is_sampled(self) -> bool:
        return bool(self.trace_flags & 0x01)

    def to_traceparent(self) -> str:
        """W3C Trace Context Header (traceparent)."""
        return f"00-{self.trace_id}-{self.span_id}-{self.trace_flags:02x}"

    @classmethod
    def from_traceparent(cls, header: str) -> SpanContext:
        """Parst W3C traceparent Header."""
        parts = header.split("-")
        if len(parts) != 4:
            return cls()  # Invalid → neuer Kontext
        return cls(
            trace_id=parts[1],
            span_id=parts[2],
            trace_flags=int(parts[3], 16),
            is_remote=True,
        )

    def child_context(self) -> SpanContext:
        """Erstellt Kind-Kontext (gleiche Trace-ID, neue Span-ID)."""
        return SpanContext(
            trace_id=self.trace_id,
            span_id=generate_span_id(),
            trace_flags=self.trace_flags,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "trace_flags": self.trace_flags,
        }


# ── Span Event ───────────────────────────────────────────────────

@dataclass
class SpanEvent:
    """Ein zeitgestempeltes Event innerhalb eines Spans."""
    name: str
    timestamp_ns: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.timestamp_ns:
            self.timestamp_ns = time.time_ns()

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "timestamp_ns": self.timestamp_ns,
        }
        if self.attributes:
            d["attributes"] = self.attributes
        return d


# ── Span Link ────────────────────────────────────────────────────

@dataclass
class SpanLink:
    """Verknüpfung zu einem anderen Span (z.B. für Batch-Jobs)."""
    context: SpanContext
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "attributes": self.attributes,
        }


# ── Span ─────────────────────────────────────────────────────────

@dataclass
class Span:
    """Eine einzelne Operation im Distributed Trace.

    Lebenszyklus: start() → add_event() → set_status() → end()
    """
    name: str
    context: SpanContext = field(default_factory=SpanContext)
    parent_span_id: str = ""
    kind: SpanKind = SpanKind.INTERNAL
    start_time_ns: int = 0
    end_time_ns: int = 0
    status_code: StatusCode = StatusCode.UNSET
    status_message: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)
    links: list[SpanLink] = field(default_factory=list)
    # Metadata
    service_name: str = "jarvis"
    resource_attributes: dict[str, str] = field(default_factory=dict)
    _ended: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if not self.start_time_ns:
            self.start_time_ns = time.time_ns()

    @property
    def trace_id(self) -> str:
        return self.context.trace_id

    @property
    def span_id(self) -> str:
        return self.context.span_id

    @property
    def duration_ns(self) -> int:
        if self.end_time_ns:
            return self.end_time_ns - self.start_time_ns
        return time.time_ns() - self.start_time_ns

    @property
    def duration_ms(self) -> float:
        return self.duration_ns / 1_000_000

    @property
    def is_ended(self) -> bool:
        return self._ended

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_attributes(self, attrs: dict[str, Any]) -> None:
        self.attributes.update(attrs)

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> SpanEvent:
        event = SpanEvent(name=name, attributes=attributes or {})
        self.events.append(event)
        return event

    def add_link(self, context: SpanContext, attributes: dict[str, Any] | None = None) -> None:
        self.links.append(SpanLink(context=context, attributes=attributes or {}))

    def set_status(self, code: StatusCode, message: str = "") -> None:
        self.status_code = code
        self.status_message = message

    def set_ok(self) -> None:
        self.set_status(StatusCode.OK)

    def set_error(self, message: str = "") -> None:
        self.set_status(StatusCode.ERROR, message)

    def end(self) -> None:
        if not self._ended:
            self.end_time_ns = time.time_ns()
            self._ended = True

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "kind": self.kind.name,
            "start_time_ns": self.start_time_ns,
            "end_time_ns": self.end_time_ns,
            "duration_ms": round(self.duration_ms, 3),
            "status": {
                "code": self.status_code.name,
                "message": self.status_message,
            },
            "attributes": self.attributes,
            "events": [e.to_dict() for e in self.events],
            "service": self.service_name,
        }
        if self.links:
            d["links"] = [l.to_dict() for l in self.links]
        return d

    def to_otlp(self) -> dict[str, Any]:
        """OTLP-kompatibles Format (für Export)."""
        return {
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "parentSpanId": self.parent_span_id,
            "name": self.name,
            "kind": self.kind.value + 1,  # OTLP ist 1-basiert
            "startTimeUnixNano": str(self.start_time_ns),
            "endTimeUnixNano": str(self.end_time_ns),
            "attributes": [
                {"key": k, "value": _otlp_value(v)}
                for k, v in self.attributes.items()
            ],
            "events": [
                {
                    "name": e.name,
                    "timeUnixNano": str(e.timestamp_ns),
                    "attributes": [
                        {"key": k, "value": _otlp_value(v)}
                        for k, v in e.attributes.items()
                    ],
                }
                for e in self.events
            ],
            "status": {
                "code": self.status_code.value,
                "message": self.status_message,
            },
        }


# ── Trace ────────────────────────────────────────────────────────

@dataclass
class Trace:
    """Vollständiger Trace — Baum von Spans."""
    trace_id: str = ""
    spans: list[Span] = field(default_factory=list)
    service_name: str = "jarvis"
    started_at: str = ""

    def __post_init__(self) -> None:
        if not self.trace_id:
            self.trace_id = generate_trace_id()
        if not self.started_at:
            self.started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def add_span(self, span: Span) -> None:
        self.spans.append(span)

    @property
    def root_span(self) -> Span | None:
        for s in self.spans:
            if not s.parent_span_id:
                return s
        return self.spans[0] if self.spans else None

    @property
    def duration_ms(self) -> float:
        root = self.root_span
        return root.duration_ms if root else 0.0

    @property
    def span_count(self) -> int:
        return len(self.spans)

    @property
    def error_count(self) -> int:
        return sum(1 for s in self.spans if s.status_code == StatusCode.ERROR)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "service": self.service_name,
            "span_count": self.span_count,
            "duration_ms": round(self.duration_ms, 3),
            "error_count": self.error_count,
            "started_at": self.started_at,
            "spans": [s.to_dict() for s in self.spans],
        }


# ── Metric Types ─────────────────────────────────────────────────

@dataclass
class MetricDataPoint:
    """Ein einzelner Metrik-Datenpunkt."""
    timestamp_ns: int = 0
    value: float = 0.0
    attributes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.timestamp_ns:
            self.timestamp_ns = time.time_ns()

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ns": self.timestamp_ns,
            "value": self.value,
            "attributes": self.attributes,
        }


@dataclass
class HistogramDataPoint:
    """Datenpunkt für Histogram-Metriken."""
    timestamp_ns: int = 0
    count: int = 0
    total: float = 0.0
    min_value: float = float("inf")
    max_value: float = float("-inf")
    bucket_counts: list[int] = field(default_factory=list)
    bucket_boundaries: list[float] = field(default_factory=lambda: [
        5, 10, 25, 50, 75, 100, 250, 500, 750, 1000, 2500, 5000, 10000,
    ])
    attributes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.timestamp_ns:
            self.timestamp_ns = time.time_ns()
        if not self.bucket_counts:
            self.bucket_counts = [0] * (len(self.bucket_boundaries) + 1)

    def record(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)
        for i, boundary in enumerate(self.bucket_boundaries):
            if value <= boundary:
                self.bucket_counts[i] += 1
                return
        self.bucket_counts[-1] += 1

    @property
    def average(self) -> float:
        return self.total / self.count if self.count else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "sum": self.total,
            "min": self.min_value if self.count else 0,
            "max": self.max_value if self.count else 0,
            "avg": round(self.average, 3),
            "bucket_counts": self.bucket_counts,
            "bucket_boundaries": self.bucket_boundaries,
        }


@dataclass
class MetricDefinition:
    """Definition einer Metrik."""
    name: str
    kind: MetricKind
    description: str = ""
    unit: str = ""
    data_points: list[MetricDataPoint] = field(default_factory=list)
    histogram: HistogramDataPoint | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind.value,
            "description": self.description,
            "unit": self.unit,
        }
        if self.kind == MetricKind.HISTOGRAM and self.histogram:
            d["histogram"] = self.histogram.to_dict()
        elif self.data_points:
            d["latest"] = self.data_points[-1].value
            d["points_count"] = len(self.data_points)
        return d


# ── OTLP Helper ──────────────────────────────────────────────────

def _otlp_value(v: Any) -> dict[str, Any]:
    """Konvertiert Python-Wert zu OTLP AnyValue."""
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    if isinstance(v, (list, tuple)):
        return {"arrayValue": {"values": [_otlp_value(x) for x in v]}}
    return {"stringValue": str(v)}
