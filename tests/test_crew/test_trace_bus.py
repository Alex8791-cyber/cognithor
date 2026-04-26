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
