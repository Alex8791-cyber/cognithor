"""Tests: OpenTelemetry v19.

Tests für alle v19-Module: Types, TracerProvider, MetricsProvider,
Instrumentation/TelemetryHub, Integration.
"""

import asyncio
import json
import tempfile
import time
import pytest
from pathlib import Path
from typing import Any

from jarvis.telemetry.types import (
    SpanKind, StatusCode, MetricKind,
    SpanContext, SpanEvent, SpanLink, Span, Trace,
    MetricDataPoint, HistogramDataPoint, MetricDefinition,
    generate_trace_id, generate_span_id, _otlp_value,
)
from jarvis.telemetry.tracer import (
    TracerProvider, SpanContextManager,
    AlwaysOnSampler, AlwaysOffSampler, ProbabilisticSampler, RateBasedSampler,
    InMemoryProcessor, ConsoleProcessor, BatchProcessor,
    SpanExporter, OTLPJsonExporter, _NoOpSpan,
)
from jarvis.telemetry.metrics import MetricsProvider
from jarvis.telemetry.instrumentation import (
    TelemetryHub, trace, measure,
)


# ============================================================================
# ID Generation Tests
# ============================================================================

class TestIdGeneration:
    def test_trace_id_format(self):
        tid = generate_trace_id()
        assert len(tid) == 32
        int(tid, 16)  # Must be valid hex

    def test_span_id_format(self):
        sid = generate_span_id()
        assert len(sid) == 16
        int(sid, 16)

    def test_ids_unique(self):
        ids = {generate_trace_id() for _ in range(100)}
        assert len(ids) == 100


# ============================================================================
# SpanContext Tests
# ============================================================================

class TestSpanContext:
    def test_auto_generation(self):
        ctx = SpanContext()
        assert ctx.is_valid
        assert len(ctx.trace_id) == 32
        assert len(ctx.span_id) == 16

    def test_is_sampled(self):
        assert SpanContext(trace_flags=1).is_sampled
        assert not SpanContext(trace_flags=0).is_sampled

    def test_traceparent_roundtrip(self):
        ctx = SpanContext()
        header = ctx.to_traceparent()
        restored = SpanContext.from_traceparent(header)
        assert restored.trace_id == ctx.trace_id
        assert restored.span_id == ctx.span_id
        assert restored.is_remote

    def test_traceparent_format(self):
        ctx = SpanContext(trace_id="a" * 32, span_id="b" * 16, trace_flags=1)
        assert ctx.to_traceparent() == f"00-{'a' * 32}-{'b' * 16}-01"

    def test_invalid_traceparent(self):
        ctx = SpanContext.from_traceparent("invalid")
        assert ctx.is_valid  # Gets new IDs

    def test_child_context(self):
        parent = SpanContext()
        child = parent.child_context()
        assert child.trace_id == parent.trace_id
        assert child.span_id != parent.span_id

    def test_to_dict(self):
        ctx = SpanContext()
        d = ctx.to_dict()
        assert "trace_id" in d
        assert "span_id" in d


# ============================================================================
# SpanEvent / SpanLink Tests
# ============================================================================

class TestSpanEvent:
    def test_basic(self):
        e = SpanEvent(name="error", attributes={"code": 500})
        assert e.name == "error"
        assert e.timestamp_ns > 0

    def test_to_dict(self):
        e = SpanEvent(name="retry", attributes={"attempt": 2})
        d = e.to_dict()
        assert d["name"] == "retry"
        assert d["attributes"]["attempt"] == 2


class TestSpanLink:
    def test_basic(self):
        ctx = SpanContext()
        link = SpanLink(context=ctx, attributes={"batch": "b1"})
        d = link.to_dict()
        assert d["trace_id"] == ctx.trace_id


# ============================================================================
# Span Tests
# ============================================================================

class TestSpan:
    def test_basic(self):
        span = Span(name="test_op")
        assert span.name == "test_op"
        assert span.trace_id
        assert span.span_id
        assert span.start_time_ns > 0
        assert not span.is_ended

    def test_set_attribute(self):
        span = Span(name="op")
        span.set_attribute("key", "value")
        assert span.attributes["key"] == "value"

    def test_set_attributes(self):
        span = Span(name="op")
        span.set_attributes({"a": 1, "b": "x"})
        assert span.attributes["a"] == 1

    def test_add_event(self):
        span = Span(name="op")
        event = span.add_event("checkpoint", {"step": 3})
        assert len(span.events) == 1
        assert event.name == "checkpoint"

    def test_add_link(self):
        span = Span(name="op")
        ctx = SpanContext()
        span.add_link(ctx, {"reason": "batch"})
        assert len(span.links) == 1

    def test_set_status(self):
        span = Span(name="op")
        span.set_ok()
        assert span.status_code == StatusCode.OK
        span.set_error("oops")
        assert span.status_code == StatusCode.ERROR
        assert span.status_message == "oops"

    def test_end(self):
        span = Span(name="op")
        assert not span.is_ended
        span.end()
        assert span.is_ended
        assert span.end_time_ns > 0

    def test_double_end(self):
        span = Span(name="op")
        span.end()
        end1 = span.end_time_ns
        span.end()
        assert span.end_time_ns == end1  # No change

    def test_duration(self):
        span = Span(name="op")
        time.sleep(0.01)
        span.end()
        assert span.duration_ms >= 5  # At least some ms

    def test_to_dict(self):
        span = Span(name="op", kind=SpanKind.SERVER)
        span.set_attribute("http.method", "GET")
        span.add_event("start")
        span.set_ok()
        span.end()
        d = span.to_dict()
        assert d["name"] == "op"
        assert d["kind"] == "SERVER"
        assert d["status"]["code"] == "OK"
        assert len(d["events"]) == 1

    def test_to_otlp(self):
        span = Span(name="op", kind=SpanKind.CLIENT)
        span.set_attribute("model", "claude")
        span.end()
        otlp = span.to_otlp()
        assert otlp["name"] == "op"
        assert otlp["kind"] == 3  # CLIENT = 2, OTLP 1-based = 3
        assert len(otlp["attributes"]) == 1

    def test_parent_span_id(self):
        span = Span(name="child", parent_span_id="abc123")
        assert span.parent_span_id == "abc123"


# ============================================================================
# Trace Tests
# ============================================================================

class TestTrace:
    def test_basic(self):
        t = Trace()
        assert t.trace_id
        assert t.span_count == 0

    def test_add_spans(self):
        t = Trace()
        t.add_span(Span(name="root"))
        t.add_span(Span(name="child", parent_span_id="x"))
        assert t.span_count == 2

    def test_root_span(self):
        t = Trace()
        root = Span(name="root")
        t.add_span(root)
        t.add_span(Span(name="child", parent_span_id=root.span_id))
        assert t.root_span.name == "root"

    def test_error_count(self):
        t = Trace()
        s1 = Span(name="ok")
        s1.set_ok()
        s2 = Span(name="err")
        s2.set_error("fail")
        t.add_span(s1)
        t.add_span(s2)
        assert t.error_count == 1

    def test_to_dict(self):
        t = Trace()
        s = Span(name="root")
        s.end()
        t.add_span(s)
        d = t.to_dict()
        assert d["span_count"] == 1
        assert len(d["spans"]) == 1


# ============================================================================
# Metric Types Tests
# ============================================================================

class TestMetricTypes:
    def test_data_point(self):
        dp = MetricDataPoint(value=42.0, attributes={"env": "prod"})
        assert dp.value == 42.0
        d = dp.to_dict()
        assert d["value"] == 42.0

    def test_histogram_data_point(self):
        h = HistogramDataPoint()
        h.record(5)
        h.record(15)
        h.record(150)
        assert h.count == 3
        assert h.min_value == 5
        assert h.max_value == 150
        assert h.average == pytest.approx((5 + 15 + 150) / 3)

    def test_histogram_buckets(self):
        h = HistogramDataPoint()
        h.record(3)   # bucket [0, 5]
        h.record(8)   # bucket (5, 10]
        h.record(99)  # bucket (75, 100]
        assert h.bucket_counts[0] == 1  # <=5
        assert h.bucket_counts[1] == 1  # <=10

    def test_histogram_to_dict(self):
        h = HistogramDataPoint()
        h.record(42)
        d = h.to_dict()
        assert d["count"] == 1
        assert d["sum"] == 42

    def test_metric_definition(self):
        m = MetricDefinition(name="requests", kind=MetricKind.COUNTER,
                             description="Total requests")
        d = m.to_dict()
        assert d["name"] == "requests"
        assert d["kind"] == "counter"


class TestOTLPValue:
    def test_string(self):
        assert _otlp_value("hello") == {"stringValue": "hello"}

    def test_int(self):
        assert _otlp_value(42) == {"intValue": "42"}

    def test_float(self):
        assert _otlp_value(3.14) == {"doubleValue": 3.14}

    def test_bool(self):
        assert _otlp_value(True) == {"boolValue": True}

    def test_list(self):
        result = _otlp_value([1, "two"])
        assert "arrayValue" in result


# ============================================================================
# Sampler Tests
# ============================================================================

class TestSamplers:
    def test_always_on(self):
        s = AlwaysOnSampler()
        assert s.should_sample("abc", "test")

    def test_always_off(self):
        s = AlwaysOffSampler()
        assert not s.should_sample("abc", "test")

    def test_probabilistic_full(self):
        s = ProbabilisticSampler(ratio=1.0)
        assert all(s.should_sample(generate_trace_id(), "t") for _ in range(10))

    def test_probabilistic_zero(self):
        s = ProbabilisticSampler(ratio=0.0)
        assert not any(s.should_sample(generate_trace_id(), "t") for _ in range(10))

    def test_probabilistic_partial(self):
        s = ProbabilisticSampler(ratio=0.5)
        results = [s.should_sample(generate_trace_id(), "t") for _ in range(1000)]
        ratio = sum(results) / len(results)
        assert 0.3 < ratio < 0.7  # Rough check

    def test_rate_based(self):
        s = RateBasedSampler(max_per_second=3)
        results = [s.should_sample("tid", "t") for _ in range(10)]
        assert sum(results) == 3


# ============================================================================
# Processor Tests
# ============================================================================

class TestProcessors:
    def test_in_memory(self):
        proc = InMemoryProcessor(max_spans=5)
        for i in range(10):
            span = Span(name=f"s{i}")
            span.end()
            proc.on_end(span)
        assert len(proc.spans) == 5

    def test_in_memory_clear(self):
        proc = InMemoryProcessor()
        proc.on_end(Span(name="test"))
        proc.clear()
        assert len(proc.spans) == 0

    def test_console_no_crash(self):
        proc = ConsoleProcessor()
        span = Span(name="test")
        span.set_ok()
        span.end()
        proc.on_end(span)  # Should not raise

    def test_batch_processor(self):
        exported = []

        class MockExporter(SpanExporter):
            def export(self, spans):
                exported.extend(spans)
                return True

        proc = BatchProcessor(exporter=MockExporter(), max_batch_size=3)
        for i in range(5):
            s = Span(name=f"s{i}")
            s.end()
            proc.on_end(s)
        # 3 exported on batch, 2 remaining
        assert len(exported) == 3
        proc.flush()
        assert len(exported) == 5

    def test_batch_shutdown(self):
        exported = []

        class MockExporter(SpanExporter):
            def export(self, spans):
                exported.extend(spans)
                return True

        proc = BatchProcessor(exporter=MockExporter(), max_batch_size=100)
        proc.on_end(Span(name="s1"))
        proc.shutdown()
        assert len(exported) == 1


# ============================================================================
# OTLP Exporter Tests
# ============================================================================

class TestOTLPExporter:
    def test_export_to_file(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        exporter = OTLPJsonExporter(file_path=path)
        span = Span(name="test", service_name="jarvis")
        span.set_ok()
        span.end()
        assert exporter.export([span])
        assert exporter.exported_count == 1

        lines = [l for l in Path(path).read_text().strip().splitlines() if l.strip()]
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert "resourceSpans" in data
        Path(path).unlink()

    def test_export_empty(self):
        exporter = OTLPJsonExporter()
        assert exporter.export([])


# ============================================================================
# TracerProvider Tests
# ============================================================================

class TestTracerProvider:
    def test_start_span(self):
        tracer = TracerProvider()
        with tracer.start_span("test_op") as span:
            assert span.name == "test_op"
            assert span.trace_id
        assert span.is_ended

    def test_auto_parent(self):
        tracer = TracerProvider()
        with tracer.start_span("parent") as parent:
            with tracer.start_span("child") as child:
                assert child.parent_span_id == parent.span_id
                assert child.trace_id == parent.trace_id

    def test_nested_three_levels(self):
        tracer = TracerProvider()
        with tracer.start_span("root") as root:
            with tracer.start_span("mid") as mid:
                with tracer.start_span("leaf") as leaf:
                    assert leaf.parent_span_id == mid.span_id
                    assert mid.parent_span_id == root.span_id
                    assert leaf.trace_id == root.trace_id

    def test_explicit_parent(self):
        tracer = TracerProvider()
        parent_ctx = SpanContext()
        with tracer.start_span("child", parent=parent_ctx) as span:
            assert span.trace_id == parent_ctx.trace_id
            assert span.parent_span_id == parent_ctx.span_id

    def test_auto_ok_status(self):
        tracer = TracerProvider()
        with tracer.start_span("op") as span:
            pass
        assert span.status_code == StatusCode.OK

    def test_auto_error_status(self):
        tracer = TracerProvider()
        with pytest.raises(ValueError):
            with tracer.start_span("op") as span:
                raise ValueError("boom")
        assert span.status_code == StatusCode.ERROR
        assert "boom" in span.status_message

    def test_sampling_off(self):
        tracer = TracerProvider(sampler=AlwaysOffSampler())
        with tracer.start_span("op") as span:
            assert isinstance(span, _NoOpSpan)

    def test_get_trace(self):
        tracer = TracerProvider()
        with tracer.start_span("op") as span:
            trace_id = span.trace_id
        t = tracer.get_trace(trace_id)
        assert t is not None
        assert t.span_count >= 1

    def test_recent_traces(self):
        tracer = TracerProvider()
        for i in range(5):
            with tracer.start_span(f"op_{i}"):
                pass
        traces = tracer.get_recent_traces(3)
        assert len(traces) <= 5

    def test_extract_inject_context(self):
        tracer = TracerProvider()
        with tracer.start_span("server") as span:
            headers: dict[str, str] = {}
            tracer.inject_context(span, headers)
            assert "traceparent" in headers

            extracted = tracer.extract_context(headers)
            assert extracted is not None
            assert extracted.trace_id == span.trace_id

    def test_cleanup(self):
        tracer = TracerProvider()
        for i in range(20):
            with tracer.start_span(f"s{i}"):
                pass
        removed = tracer.cleanup(max_traces=5)
        assert removed == 15

    def test_stats(self):
        tracer = TracerProvider()
        with tracer.start_span("x"):
            pass
        stats = tracer.stats()
        assert stats["total_spans"] >= 1
        assert stats["active_spans"] == 0

    def test_processor_on_end(self):
        proc = InMemoryProcessor()
        tracer = TracerProvider(processors=[proc])
        with tracer.start_span("test"):
            pass
        assert len(proc.spans) == 1

    def test_resource_attributes(self):
        tracer = TracerProvider(service_name="my_svc")
        tracer.set_resource_attribute("env", "prod")
        with tracer.start_span("op") as span:
            assert span.resource_attributes["env"] == "prod"

    def test_span_with_links(self):
        tracer = TracerProvider()
        link_ctx = SpanContext()
        with tracer.start_span("op", links=[link_ctx]) as span:
            assert len(span.links) == 1


# ============================================================================
# MetricsProvider Tests
# ============================================================================

class TestMetricsProvider:
    def test_counter(self):
        m = MetricsProvider()
        m.counter("requests", 1)
        m.counter("requests", 1)
        assert m.get_counter("requests") == 2.0

    def test_counter_with_labels(self):
        m = MetricsProvider()
        m.counter("req", 1, method="GET")
        m.counter("req", 1, method="POST")
        assert m.get_counter("req", method="GET") == 1.0
        assert m.get_counter("req", method="POST") == 1.0

    def test_gauge(self):
        m = MetricsProvider()
        m.gauge("connections", 42)
        assert m.get_gauge("connections") == 42
        m.gauge("connections", 30)
        assert m.get_gauge("connections") == 30

    def test_histogram(self):
        m = MetricsProvider()
        m.histogram("latency", 10)
        m.histogram("latency", 20)
        m.histogram("latency", 30)
        h = m.get_histogram("latency")
        assert h is not None
        assert h.count == 3
        assert h.average == pytest.approx(20.0)

    def test_describe(self):
        m = MetricsProvider()
        m.describe("req", "Total requests", "1")
        m.counter("req", 5)
        metrics = m.get_all_metrics()
        found = [md for md in metrics if md.name == "req"]
        assert len(found) == 1
        assert found[0].description == "Total requests"

    def test_history(self):
        m = MetricsProvider()
        for i in range(10):
            m.counter("clicks", 1)
        history = m.get_history("clicks", last_n=5)
        assert len(history) == 5

    def test_snapshot(self):
        m = MetricsProvider()
        m.counter("a", 1)
        m.gauge("b", 2)
        snap = m.snapshot()
        assert "counters" in snap
        assert "gauges" in snap

    def test_to_otlp(self):
        m = MetricsProvider()
        m.counter("req", 5)
        m.gauge("cpu", 0.75)
        m.histogram("latency", 42)
        otlp = m.to_otlp()
        assert "resourceMetrics" in otlp
        scope = otlp["resourceMetrics"][0]["scopeMetrics"][0]
        assert len(scope["metrics"]) == 3

    def test_reset(self):
        m = MetricsProvider()
        m.counter("x", 5)
        m.reset()
        assert m.get_counter("x") == 0.0

    def test_stats(self):
        m = MetricsProvider()
        m.counter("a", 1)
        m.gauge("b", 2)
        m.histogram("c", 3)
        stats = m.stats()
        assert stats["total_metrics"] == 3


# ============================================================================
# Instrumentation Decorator Tests
# ============================================================================

class TestDecorators:
    @pytest.mark.asyncio
    async def test_trace_decorator_async(self):
        tracer = TracerProvider()
        proc = InMemoryProcessor()
        tracer.add_processor(proc)

        @trace(tracer, "my_func")
        async def my_func(x: int) -> int:
            return x * 2

        result = await my_func(5)
        assert result == 10
        assert any(s.name == "my_func" for s in proc.spans)

    @pytest.mark.asyncio
    async def test_trace_decorator_error(self):
        tracer = TracerProvider()
        proc = InMemoryProcessor()
        tracer.add_processor(proc)

        @trace(tracer, "failing")
        async def failing():
            raise RuntimeError("oops")

        with pytest.raises(RuntimeError):
            await failing()

        error_spans = [s for s in proc.spans if s.status_code == StatusCode.ERROR]
        assert len(error_spans) == 1

    def test_trace_decorator_sync(self):
        tracer = TracerProvider()
        proc = InMemoryProcessor()
        tracer.add_processor(proc)

        @trace(tracer, "sync_func")
        def sync_func() -> str:
            return "ok"

        assert sync_func() == "ok"
        assert any(s.name == "sync_func" for s in proc.spans)

    @pytest.mark.asyncio
    async def test_measure_decorator(self):
        metrics = MetricsProvider()

        @measure(metrics, "func_latency_ms", "func_calls", label="test")
        async def my_func():
            return 42

        result = await my_func()
        assert result == 42
        assert metrics.get_counter("func_calls", status="ok", label="test") == 1
        h = metrics.get_histogram("func_latency_ms", label="test")
        assert h is not None
        assert h.count == 1


# ============================================================================
# TelemetryHub Tests
# ============================================================================

class TestTelemetryHub:
    def test_creation(self):
        hub = TelemetryHub(service_name="test")
        assert hub.tracer.service_name == "test"

    def test_trace_request(self):
        hub = TelemetryHub()
        with hub.trace_request("POST", "/chat") as span:
            span.set_attribute("user.id", "u1")
        assert span.status_code == StatusCode.OK

    def test_trace_request_with_propagation(self):
        hub = TelemetryHub()
        parent_ctx = SpanContext()
        headers = {"traceparent": parent_ctx.to_traceparent()}
        with hub.trace_request("GET", "/health", headers=headers) as span:
            assert span.trace_id == parent_ctx.trace_id

    def test_trace_llm_call(self):
        hub = TelemetryHub()
        with hub.trace_llm_call("claude", prompt_length=100) as span:
            pass
        assert hub.metrics.get_counter("llm_calls_total", model="claude") == 1

    def test_trace_tool_call(self):
        hub = TelemetryHub()
        with hub.trace_tool_call("web_search") as span:
            span.set_attribute("query", "test")

    def test_trace_graph_execution(self):
        hub = TelemetryHub()
        with hub.trace_graph_execution("pipeline") as span:
            pass
        assert hub.metrics.get_counter("graph_executions_total", graph="pipeline") >= 1

    def test_trace_a2a_message(self):
        hub = TelemetryHub()
        with hub.trace_a2a_message("agent-2", "outbound") as span:
            pass
        assert hub.metrics.get_counter("a2a_messages_total",
                                        agent="agent-2", direction="outbound") == 1

    def test_trace_browser_action(self):
        hub = TelemetryHub()
        with hub.trace_browser_action("navigate", "https://example.com") as span:
            pass
        assert hub.metrics.get_counter("browser_actions_total", action="navigate") == 1

    def test_record_request(self):
        hub = TelemetryHub()
        hub.record_request("POST", 200, 42.5)
        hub.record_request("POST", 500, 100.0)
        assert hub.metrics.get_counter("requests_total", method="POST", status="200") == 1
        assert hub.metrics.get_counter("errors_total", method="POST", status="500") == 1

    def test_record_llm_usage(self):
        hub = TelemetryHub()
        hub.record_llm_usage("claude", 150.0, input_tokens=100, output_tokens=200)
        assert hub.metrics.get_counter("llm_tokens_total",
                                        model="claude", direction="input") == 100

    def test_record_graph_execution(self):
        hub = TelemetryHub()
        hub.record_graph_execution("etl", 500.0, status="completed")
        h = hub.metrics.get_histogram("graph_execution_latency_ms", graph="etl")
        assert h is not None and h.count == 1

    def test_set_active_sessions(self):
        hub = TelemetryHub()
        hub.set_active_sessions(12)
        assert hub.metrics.get_gauge("active_sessions") == 12

    def test_dashboard_snapshot(self):
        hub = TelemetryHub()
        with hub.trace_request("GET", "/test"):
            pass
        hub.record_request("GET", 200, 10.0)
        snap = hub.dashboard_snapshot()
        assert "tracer" in snap
        assert "metrics" in snap
        assert "recent_traces" in snap

    def test_shutdown(self):
        hub = TelemetryHub()
        hub.shutdown()  # Should not raise

    def test_stats(self):
        hub = TelemetryHub()
        stats = hub.stats()
        assert "tracer" in stats
        assert "metrics" in stats


# ============================================================================
# Integration Tests
# ============================================================================

class TestTelemetryIntegration:
    def test_full_request_trace(self):
        """Simuliert vollständigen Request mit verschachtelten Spans."""
        hub = TelemetryHub(service_name="jarvis")
        proc = InMemoryProcessor()
        hub.tracer.add_processor(proc)

        with hub.trace_request("POST", "/chat") as request_span:
            request_span.set_attribute("user.id", "u123")

            with hub.trace_llm_call("claude", prompt_length=500) as llm_span:
                llm_span.set_attribute("llm.temperature", 0.7)
                llm_span.add_event("tokens_received", {"count": 150})
                hub.record_llm_usage("claude", 200.0, input_tokens=500, output_tokens=150)

            with hub.trace_tool_call("web_search") as tool_span:
                tool_span.set_attribute("query", "jarvis ai")
                tool_span.add_event("results_found", {"count": 5})

        hub.record_request("POST", 200, request_span.duration_ms)

        # Verify trace structure
        trace = hub.tracer.get_trace(request_span.trace_id)
        assert trace is not None
        assert trace.span_count == 3
        assert trace.error_count == 0

        # All spans have same trace_id
        assert all(s.trace_id == request_span.trace_id for s in trace.spans)

        # Verify parent-child
        llm_spans = [s for s in trace.spans if "llm" in s.name]
        assert llm_spans[0].parent_span_id == request_span.span_id

    def test_cross_service_propagation(self):
        """Simuliert Cross-Service-Propagation via HTTP-Headers."""
        service_a = TelemetryHub(service_name="gateway")
        service_b = TelemetryHub(service_name="agent")

        # Service A startet Request
        with service_a.trace_request("POST", "/process") as span_a:
            headers: dict[str, str] = {}
            service_a.tracer.inject_context(span_a, headers)

            # Service B empfängt Request
            with service_b.trace_request("POST", "/execute", headers=headers) as span_b:
                assert span_b.trace_id == span_a.trace_id
                assert span_b.parent_span_id == span_a.span_id

    def test_metrics_otlp_export(self):
        """Prüft OTLP-Metrik-Export."""
        hub = TelemetryHub()
        hub.record_request("GET", 200, 15.0)
        hub.record_request("POST", 201, 50.0)
        hub.record_request("POST", 500, 100.0)
        hub.record_llm_usage("claude", 200.0, input_tokens=1000)

        otlp = hub.metrics.to_otlp()
        assert "resourceMetrics" in otlp
        scope = otlp["resourceMetrics"][0]["scopeMetrics"][0]
        assert len(scope["metrics"]) > 0

    def test_full_trace_otlp_export(self):
        """Prüft OTLP-Trace-Export."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        exporter = OTLPJsonExporter(file_path=path)
        proc = BatchProcessor(exporter=exporter, max_batch_size=100)
        tracer = TracerProvider(processors=[proc])

        with tracer.start_span("root") as root:
            with tracer.start_span("child1"):
                pass
            with tracer.start_span("child2"):
                pass

        proc.shutdown()
        assert exporter.exported_count == 3

        data = json.loads(Path(path).read_text())
        assert len(data) >= 1
        Path(path).unlink()

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Prüft async with für Spans."""
        tracer = TracerProvider()
        async with tracer.start_span("async_op") as span:
            span.set_attribute("async", True)
            await asyncio.sleep(0.01)
        assert span.is_ended
        assert span.status_code == StatusCode.OK

    def test_high_volume_spans(self):
        """Stress-Test: Viele Spans."""
        proc = InMemoryProcessor(max_spans=100)
        tracer = TracerProvider(processors=[proc])

        for i in range(200):
            with tracer.start_span(f"op_{i}"):
                pass

        assert len(proc.spans) == 100  # Capped
        assert tracer.stats()["total_spans"] == 200

    def test_noop_span_safety(self):
        """NoOp-Span darf keine Exceptions werfen."""
        noop = _NoOpSpan()
        noop.set_attribute("key", "val")
        noop.set_attributes({"a": 1})
        noop.add_event("test")
        noop.set_ok()
        noop.set_error("err")
        noop.end()
        assert noop.context.is_valid
