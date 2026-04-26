"""TraceBus — in-process pub/sub for crew audit events."""

from __future__ import annotations

from cognithor.crew.trace_bus import TraceBus, get_trace_bus


def test_get_trace_bus_returns_singleton() -> None:
    bus1 = get_trace_bus()
    bus2 = get_trace_bus()
    assert bus1 is bus2


def test_publish_does_not_raise_with_no_subscribers() -> None:
    bus = TraceBus()
    bus.publish({"event_type": "crew_kickoff_started", "trace_id": "abc"})
