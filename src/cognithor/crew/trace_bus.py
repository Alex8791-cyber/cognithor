"""TraceBus — in-process pub/sub for cognithor.crew audit events.

Hooks `compiler.append_audit()` to live-broadcast crew_* events to
WebSocket subscribers without changing the JSONL persistence path.

Lifecycle events (`crew_kickoff_started`, `crew_kickoff_completed`,
`crew_kickoff_failed`) fan out to ALL lifecycle-subscribers (Dashboard
view). Per-trace events fan out to topic-subscribers keyed by `trace_id`.

Backpressure: each subscriber gets a bounded `asyncio.Queue(maxsize=1000)`.
On overflow → drop oldest, increment dropped-counter, rate-limited warn-log.
JSONL persistence is independent and lossless.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

LIFECYCLE_EVENTS: frozenset[str] = frozenset(
    {"crew_kickoff_started", "crew_kickoff_completed", "crew_kickoff_failed"}
)
LIFECYCLE_TOPIC = "__lifecycle__"
DEFAULT_QUEUE_MAXSIZE = 1000
_DROP_LOG_INTERVAL_SEC = 60.0


@dataclass
class SubscriptionHandle:
    """Opaque handle returned by subscribe(); pass to unsubscribe()."""

    topic: str
    queue: asyncio.Queue[dict[str, Any]] = field(repr=False)
    dropped_count: int = 0
    last_drop_log_ts: float = 0.0


class TraceBus:
    """In-process pub/sub for crew audit events."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # topic → list of handles
        self._subscribers: dict[str, list[SubscriptionHandle]] = {}

    def subscribe_lifecycle(self, queue: asyncio.Queue[dict[str, Any]]) -> SubscriptionHandle:
        """Subscribe to lifecycle-only events (crew_kickoff_*)."""
        handle = SubscriptionHandle(topic=LIFECYCLE_TOPIC, queue=queue)
        with self._lock:
            self._subscribers.setdefault(LIFECYCLE_TOPIC, []).append(handle)
        return handle

    def unsubscribe(self, handle: SubscriptionHandle) -> None:
        """Remove a subscription."""
        with self._lock:
            subs = self._subscribers.get(handle.topic, [])
            try:
                subs.remove(handle)
            except ValueError:
                return
            if not subs:
                self._subscribers.pop(handle.topic, None)

    def subscribe(self, trace_id: str, queue: asyncio.Queue[dict[str, Any]]) -> SubscriptionHandle:
        """Subscribe to all events for a single trace_id."""
        handle = SubscriptionHandle(topic=trace_id, queue=queue)
        with self._lock:
            self._subscribers.setdefault(trace_id, []).append(handle)
        return handle

    def publish(self, record: dict[str, Any]) -> None:
        """Broadcast an audit record. Hot-path; must be <1ms."""
        event_type = record.get("event_type") or record.get("event")
        trace_id = record.get("trace_id") or record.get("session_id")
        if event_type in LIFECYCLE_EVENTS:
            self._fanout(LIFECYCLE_TOPIC, record)
        if trace_id:
            self._fanout(trace_id, record)

    def _fanout(self, topic: str, record: dict[str, Any]) -> None:
        with self._lock:
            subs = list(self._subscribers.get(topic, ()))
        for handle in subs:
            self._enqueue(handle, record)

    def _enqueue(self, handle: SubscriptionHandle, record: dict[str, Any]) -> None:
        try:
            handle.queue.put_nowait(record)
        except asyncio.QueueFull:
            with contextlib.suppress(asyncio.QueueEmpty):
                handle.queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                handle.queue.put_nowait(record)
            handle.dropped_count += 1
            now = time.monotonic()
            if now - handle.last_drop_log_ts > _DROP_LOG_INTERVAL_SEC:
                log.warning(
                    "trace_bus_drop topic=%s dropped_total=%d",
                    handle.topic,
                    handle.dropped_count,
                )
                handle.last_drop_log_ts = now


_singleton: TraceBus | None = None
_singleton_lock = threading.Lock()


def get_trace_bus() -> TraceBus:
    """Process-wide singleton accessor."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = TraceBus()
    return _singleton
