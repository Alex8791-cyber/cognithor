"""Tests für Live-Monitoring-System.

Testet: EventBus, MetricCollector, AuditTrailViewer,
HeartbeatMonitor, MonitoringHub, SSE-Streaming.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from jarvis.gateway.monitoring import (
    AuditEntry,
    AuditTrailViewer,
    EventBus,
    EventType,
    HeartbeatMonitor,
    HeartbeatRun,
    MetricCollector,
    MetricPoint,
    MonitoringHub,
    SystemEvent,
)


# ============================================================================
# EventBus
# ============================================================================


class TestEventBus:
    def test_publish_and_history(self) -> None:
        bus = EventBus()
        bus.publish(SystemEvent(event_type=EventType.MESSAGE_RECEIVED, source="test"))
        assert bus.event_count == 1

    def test_subscribe_and_receive(self) -> None:
        bus = EventBus()
        received: list[SystemEvent] = []
        bus.subscribe(EventType.MESSAGE_RECEIVED, handler=lambda e: received.append(e))

        bus.publish(SystemEvent(event_type=EventType.MESSAGE_RECEIVED))
        bus.publish(SystemEvent(event_type=EventType.ERROR))  # Sollte nicht empfangen werden

        assert len(received) == 1
        assert received[0].event_type == EventType.MESSAGE_RECEIVED

    def test_wildcard_subscriber(self) -> None:
        bus = EventBus()
        received: list[SystemEvent] = []
        bus.subscribe(None, handler=lambda e: received.append(e))

        bus.publish(SystemEvent(event_type=EventType.MESSAGE_RECEIVED))
        bus.publish(SystemEvent(event_type=EventType.ERROR))

        assert len(received) == 2

    def test_recent_events_filter(self) -> None:
        bus = EventBus()
        bus.publish(SystemEvent(event_type=EventType.ERROR, severity="error"))
        bus.publish(SystemEvent(event_type=EventType.MESSAGE_RECEIVED, severity="info"))
        bus.publish(SystemEvent(event_type=EventType.WARNING, severity="warning"))

        errors = bus.recent_events(severity="error")
        assert len(errors) == 1
        assert errors[0].severity == "error"

    def test_recent_events_by_type(self) -> None:
        bus = EventBus()
        bus.publish(SystemEvent(event_type=EventType.TOOL_EXECUTED))
        bus.publish(SystemEvent(event_type=EventType.TOOL_BLOCKED))
        bus.publish(SystemEvent(event_type=EventType.TOOL_EXECUTED))

        results = bus.recent_events(event_type=EventType.TOOL_EXECUTED)
        assert len(results) == 2

    def test_max_history(self) -> None:
        bus = EventBus(max_history=5)
        for i in range(10):
            bus.publish(SystemEvent(event_type=EventType.METRIC, data={"i": i}))
        assert bus.event_count == 5

    def test_sse_stream_creation(self) -> None:
        bus = EventBus()
        queue = bus.create_sse_stream()
        assert bus.sse_consumer_count == 1

        bus.publish(SystemEvent(event_type=EventType.MESSAGE_RECEIVED))
        assert not queue.empty()

        bus.remove_sse_stream(queue)
        assert bus.sse_consumer_count == 0

    def test_handler_error_doesnt_crash(self) -> None:
        bus = EventBus()
        bus.subscribe(None, handler=lambda e: 1 / 0)  # Will raise
        bus.publish(SystemEvent(event_type=EventType.ERROR))  # Should not crash
        assert bus.event_count == 1


class TestSystemEvent:
    def test_to_dict(self) -> None:
        event = SystemEvent(
            event_type=EventType.AGENT_SELECTED,
            source="gateway",
            agent_id="coder",
            severity="info",
        )
        d = event.to_dict()
        assert d["event_type"] == "agent_selected"
        assert d["source"] == "gateway"
        assert "timestamp" in d

    def test_to_sse(self) -> None:
        event = SystemEvent(event_type=EventType.ERROR, source="test")
        sse = event.to_sse()
        assert sse.startswith("event: error")
        assert "data: " in sse
        assert sse.endswith("\n\n")


# ============================================================================
# MetricCollector
# ============================================================================


class TestMetricCollector:
    def test_gauge(self) -> None:
        mc = MetricCollector()
        mc.gauge("cpu_percent", 45.2)
        assert mc.get_gauge("cpu_percent") == 45.2

    def test_gauge_overwrite(self) -> None:
        mc = MetricCollector()
        mc.gauge("mem", 100)
        mc.gauge("mem", 200)
        assert mc.get_gauge("mem") == 200

    def test_counter_increment(self) -> None:
        mc = MetricCollector()
        mc.increment("requests")
        mc.increment("requests")
        mc.increment("requests", delta=5)
        assert mc.get_counter("requests") == 7.0

    def test_history(self) -> None:
        mc = MetricCollector()
        mc.gauge("cpu", 10)
        mc.gauge("cpu", 20)
        mc.gauge("cpu", 30)
        history = mc.get_history("cpu", last_n=2)
        assert len(history) == 2
        assert history[-1]["value"] == 30

    def test_snapshot(self) -> None:
        mc = MetricCollector()
        mc.gauge("a", 1)
        mc.increment("b")
        snap = mc.snapshot()
        assert snap["gauges"]["a"] == 1
        assert snap["counters"]["b"] == 1
        assert snap["series_count"] == 2

    def test_all_metric_names(self) -> None:
        mc = MetricCollector()
        mc.gauge("z_gauge", 1)
        mc.increment("a_counter")
        names = mc.all_metric_names()
        assert "a_counter" in names
        assert "z_gauge" in names
        # Sorted
        assert names[0] == "a_counter"

    def test_max_points_per_metric(self) -> None:
        mc = MetricCollector(max_points_per_metric=3)
        for i in range(10):
            mc.gauge("test", float(i))
        assert len(mc.get_history("test")) == 3

    def test_labels(self) -> None:
        mc = MetricCollector()
        mc.gauge("latency", 150.0, endpoint="/api/v1/chat")
        history = mc.get_history("latency")
        assert history[0]["labels"]["endpoint"] == "/api/v1/chat"

    def test_get_nonexistent(self) -> None:
        mc = MetricCollector()
        assert mc.get_gauge("nope") == 0.0
        assert mc.get_counter("nope") == 0.0
        assert mc.get_history("nope") == []


# ============================================================================
# AuditTrailViewer
# ============================================================================


class TestAuditTrailViewer:
    def test_record(self) -> None:
        at = AuditTrailViewer()
        entry = at.record("login", "user_1", "system")
        assert entry.action == "login"
        assert at.entry_count == 1

    def test_search_by_action(self) -> None:
        at = AuditTrailViewer()
        at.record("credential_access", "user_1", "api_key")
        at.record("config_change", "admin", "heartbeat")
        at.record("credential_access", "user_2", "oauth_token")

        results = at.search(action="credential")
        assert len(results) == 2

    def test_search_by_actor(self) -> None:
        at = AuditTrailViewer()
        at.record("action1", "alice", "target1")
        at.record("action2", "bob", "target2")

        results = at.search(actor="alice")
        assert len(results) == 1

    def test_search_by_severity(self) -> None:
        at = AuditTrailViewer()
        at.record("ok", "sys", "t", severity="info")
        at.record("warning", "sys", "t", severity="warning")
        at.record("error", "sys", "t", severity="error")

        errors = at.search(severity="error")
        assert len(errors) == 1

    def test_search_by_time_range(self) -> None:
        at = AuditTrailViewer()
        at.record("old", "sys", "t")
        at.record("new", "sys", "t")

        now = datetime.now(UTC)
        results = at.search(since=now - timedelta(seconds=5))
        assert len(results) == 2

        results = at.search(until=now - timedelta(hours=1))
        assert len(results) == 0

    def test_severity_counts(self) -> None:
        at = AuditTrailViewer()
        at.record("a", "s", "t", severity="info")
        at.record("b", "s", "t", severity="info")
        at.record("c", "s", "t", severity="error")

        counts = at.severity_counts()
        assert counts["info"] == 2
        assert counts["error"] == 1

    def test_max_entries(self) -> None:
        at = AuditTrailViewer(max_entries=3)
        for i in range(5):
            at.record(f"action_{i}", "sys", "t")
        assert at.entry_count == 3

    def test_recent(self) -> None:
        at = AuditTrailViewer()
        for i in range(10):
            at.record(f"a{i}", "sys", "t")
        recent = at.recent(3)
        assert len(recent) == 3
        assert recent[-1].action == "a9"


class TestAuditEntry:
    def test_to_dict(self) -> None:
        entry = AuditEntry(
            timestamp=datetime.now(UTC),
            action="tool_execute",
            actor="agent_coder",
            target="shell.run",
            severity="warning",
        )
        d = entry.to_dict()
        assert d["action"] == "tool_execute"
        assert d["actor"] == "agent_coder"
        assert "timestamp" in d


# ============================================================================
# HeartbeatMonitor
# ============================================================================


class TestHeartbeatMonitor:
    def test_start_and_complete_run(self) -> None:
        hm = HeartbeatMonitor()
        run = hm.start_run(channel="cli")
        hm.complete_run(run, success=True, tasks_found=3, tasks_executed=2)

        assert run.success is True
        assert run.tasks_found == 3
        assert run.duration_ms >= 0
        assert run.completed_at is not None

    def test_last_run(self) -> None:
        hm = HeartbeatMonitor()
        assert hm.last_run() is None

        hm.start_run()
        assert hm.last_run() is not None

    def test_recent_runs(self) -> None:
        hm = HeartbeatMonitor()
        for _ in range(5):
            run = hm.start_run()
            hm.complete_run(run, success=True)
        assert len(hm.recent_runs(3)) == 3

    def test_stats_empty(self) -> None:
        hm = HeartbeatMonitor()
        s = hm.stats()
        assert s["total_runs"] == 0
        assert s["success_rate"] == 0.0

    def test_stats_with_runs(self) -> None:
        hm = HeartbeatMonitor()
        hm.set_schedule(enabled=True, interval_minutes=15)

        run1 = hm.start_run()
        hm.complete_run(run1, success=True)

        run2 = hm.start_run()
        hm.complete_run(run2, success=False, error="Timeout")

        s = hm.stats()
        assert s["total_runs"] == 2
        assert s["success_rate"] == 50.0
        assert s["enabled"] is True
        assert s["interval_minutes"] == 15

    def test_failed_run(self) -> None:
        hm = HeartbeatMonitor()
        run = hm.start_run()
        hm.complete_run(run, success=False, error="Connection refused")
        assert run.error == "Connection refused"

    def test_max_history(self) -> None:
        hm = HeartbeatMonitor(max_history=3)
        for _ in range(5):
            hm.start_run()
        assert len(hm.recent_runs(10)) == 3


class TestHeartbeatRun:
    def test_to_dict(self) -> None:
        run = HeartbeatRun(
            run_id=1,
            started_at=datetime.now(UTC),
            success=True,
            channel="cli",
        )
        d = run.to_dict()
        assert d["run_id"] == 1
        assert d["success"] is True


# ============================================================================
# MonitoringHub
# ============================================================================


class TestMonitoringHub:
    def test_emit(self) -> None:
        hub = MonitoringHub()
        event = hub.emit(EventType.MESSAGE_RECEIVED, source="test")
        assert event.event_type == EventType.MESSAGE_RECEIVED
        assert hub.events.event_count == 1
        assert hub.metrics.get_counter("events.message_received") == 1

    def test_emit_error_increments_severity(self) -> None:
        hub = MonitoringHub()
        hub.emit(EventType.ERROR, severity="error")
        assert hub.metrics.get_counter("severity.error") == 1

    def test_dashboard_snapshot(self) -> None:
        hub = MonitoringHub()
        hub.emit(EventType.TOOL_EXECUTED, source="shell")
        hub.audit.record("test", "user", "target")
        run = hub.heartbeat.start_run()
        hub.heartbeat.complete_run(run, success=True)

        snap = hub.dashboard_snapshot()
        assert snap["events"]["total"] == 1
        assert snap["audit"]["total"] == 1
        assert snap["heartbeat"]["total_runs"] == 1
        assert "metrics" in snap

    def test_all_subsystems_accessible(self) -> None:
        hub = MonitoringHub()
        assert hub.events is not None
        assert hub.metrics is not None
        assert hub.audit is not None
        assert hub.heartbeat is not None

    def test_emit_with_data(self) -> None:
        hub = MonitoringHub()
        event = hub.emit(
            EventType.TOOL_EXECUTED,
            source="shell",
            agent_id="coder",
            tool_name="run",
            duration_ms=150,
        )
        assert event.data["tool_name"] == "run"
        assert event.data["duration_ms"] == 150

    def test_multiple_events_tracked(self) -> None:
        hub = MonitoringHub()
        for _ in range(10):
            hub.emit(EventType.MESSAGE_RECEIVED)
        assert hub.metrics.get_counter("events.message_received") == 10
