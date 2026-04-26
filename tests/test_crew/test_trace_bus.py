"""TraceBus — in-process pub/sub for crew audit events."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from cognithor.crew.trace_bus import TraceBus, get_trace_bus


def test_get_trace_bus_returns_singleton() -> None:
    bus1 = get_trace_bus()
    bus2 = get_trace_bus()
    assert bus1 is bus2


def test_publish_does_not_raise_with_no_subscribers() -> None:
    bus = TraceBus()
    bus.publish({"event_type": "crew_kickoff_started", "trace_id": "abc"})


@pytest.mark.asyncio
async def test_subscribe_lifecycle_returns_handle() -> None:
    bus = TraceBus()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=10)
    handle = bus.subscribe_lifecycle(queue)
    assert handle is not None
    assert handle.topic == "__lifecycle__"
    bus.unsubscribe(handle)


@pytest.mark.asyncio
async def test_lifecycle_event_routes_to_lifecycle_subscriber() -> None:
    bus = TraceBus()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=10)
    bus.subscribe_lifecycle(queue)
    bus.publish({"event_type": "crew_kickoff_started", "trace_id": "abc", "n_tasks": 4})
    received = await asyncio.wait_for(queue.get(), timeout=0.5)
    assert received["event_type"] == "crew_kickoff_started"
    assert received["trace_id"] == "abc"


@pytest.mark.asyncio
async def test_non_lifecycle_event_does_not_reach_lifecycle_subscriber() -> None:
    bus = TraceBus()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=10)
    bus.subscribe_lifecycle(queue)
    bus.publish({"event_type": "crew_task_started", "trace_id": "abc", "task_id": "t1"})
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(queue.get(), timeout=0.1)


@pytest.mark.asyncio
async def test_subscribe_topic_routes_per_trace_events() -> None:
    bus = TraceBus()
    q1: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=10)
    q2: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=10)
    bus.subscribe("trace-1", q1)
    bus.subscribe("trace-2", q2)

    bus.publish({"event_type": "crew_task_started", "trace_id": "trace-1", "task_id": "t1"})

    rec1 = await asyncio.wait_for(q1.get(), timeout=0.5)
    assert rec1["task_id"] == "t1"
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(q2.get(), timeout=0.1)


@pytest.mark.asyncio
async def test_unsubscribe_removes_from_routing() -> None:
    bus = TraceBus()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=10)
    handle = bus.subscribe("trace-x", queue)
    bus.unsubscribe(handle)
    bus.publish({"event_type": "crew_task_completed", "trace_id": "trace-x"})
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(queue.get(), timeout=0.1)


@pytest.mark.asyncio
async def test_lifecycle_event_also_reaches_topic_subscriber() -> None:
    """A crew_kickoff_started event has trace_id; topic-subscribers want it too."""
    bus = TraceBus()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=10)
    bus.subscribe("trace-y", queue)
    bus.publish({"event_type": "crew_kickoff_started", "trace_id": "trace-y", "n_tasks": 2})
    rec = await asyncio.wait_for(queue.get(), timeout=0.5)
    assert rec["event_type"] == "crew_kickoff_started"


@pytest.mark.asyncio
async def test_queue_full_drops_oldest_and_increments_counter() -> None:
    bus = TraceBus()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=2)
    handle = bus.subscribe("trace-z", queue)

    bus.publish({"event_type": "crew_task_started", "trace_id": "trace-z", "task_id": "1"})
    bus.publish({"event_type": "crew_task_started", "trace_id": "trace-z", "task_id": "2"})
    bus.publish({"event_type": "crew_task_started", "trace_id": "trace-z", "task_id": "3"})

    # Queue should have items 2 and 3 (oldest "1" was dropped).
    first = await asyncio.wait_for(queue.get(), timeout=0.5)
    second = await asyncio.wait_for(queue.get(), timeout=0.5)
    assert first["task_id"] == "2"
    assert second["task_id"] == "3"
    assert handle.dropped_count == 1


def test_publish_hot_path_under_1ms_with_many_subscribers() -> None:
    """publish() should be <1ms even with ~50 active subscribers."""
    import time as _time

    bus = TraceBus()
    queues = [asyncio.Queue(maxsize=100) for _ in range(50)]
    handles = [bus.subscribe(f"trace-{i}", q) for i, q in enumerate(queues)]
    bus.subscribe_lifecycle(asyncio.Queue(maxsize=100))

    record = {"event_type": "crew_task_started", "trace_id": "trace-25", "task_id": "perf"}

    start = _time.perf_counter()
    for _ in range(1000):
        bus.publish(record)
    elapsed = _time.perf_counter() - start
    avg_us = (elapsed / 1000) * 1_000_000
    assert avg_us < 1000, f"publish too slow: avg {avg_us:.1f}µs (target <1000µs)"

    for h in handles:
        bus.unsubscribe(h)
