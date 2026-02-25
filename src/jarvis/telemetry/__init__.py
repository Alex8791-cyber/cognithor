"""Jarvis OpenTelemetry v19 -- Distributed Tracing & Metrics.

OTLP-kompatibles Observability-Framework:
  - Distributed Tracing (W3C Trace Context)
  - Metrics (Counter, Histogram, Gauge)
  - Auto-Instrumentierung (Gateway, Graph, A2A, Browser, LLM)
  - Sampling (AlwaysOn, Probabilistic, RateBased)
  - Export (OTLP-JSON, Console, InMemory)

Usage:
    from jarvis.telemetry import TelemetryHub

    hub = TelemetryHub(service_name="jarvis")
    with hub.trace_request("POST", "/chat") as span:
        span.set_attribute("user.id", "u123")
        hub.record_request("POST", 200, 42.5)
"""

from jarvis.telemetry.types import (
    SpanKind,
    StatusCode,
    MetricKind,
    SpanContext,
    SpanEvent,
    SpanLink,
    Span,
    Trace,
    MetricDataPoint,
    HistogramDataPoint,
    MetricDefinition,
    generate_trace_id,
    generate_span_id,
)
from jarvis.telemetry.tracer import (
    TracerProvider,
    SpanContextManager,
    Sampler,
    AlwaysOnSampler,
    AlwaysOffSampler,
    ProbabilisticSampler,
    RateBasedSampler,
    SpanProcessor,
    InMemoryProcessor,
    ConsoleProcessor,
    BatchProcessor,
    SpanExporter,
    OTLPJsonExporter,
)
from jarvis.telemetry.metrics import MetricsProvider
from jarvis.telemetry.instrumentation import (
    TelemetryHub,
    trace,
    measure,
)

__all__ = [
    # Types
    "SpanKind", "StatusCode", "MetricKind",
    "SpanContext", "SpanEvent", "SpanLink", "Span", "Trace",
    "MetricDataPoint", "HistogramDataPoint", "MetricDefinition",
    "generate_trace_id", "generate_span_id",
    # Tracer
    "TracerProvider", "SpanContextManager",
    "Sampler", "AlwaysOnSampler", "AlwaysOffSampler",
    "ProbabilisticSampler", "RateBasedSampler",
    "SpanProcessor", "InMemoryProcessor", "ConsoleProcessor", "BatchProcessor",
    "SpanExporter", "OTLPJsonExporter",
    # Metrics
    "MetricsProvider",
    # Instrumentation
    "TelemetryHub", "trace", "measure",
]
