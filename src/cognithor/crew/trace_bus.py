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

import threading
from typing import Any


class TraceBus:
    """In-process pub/sub for crew audit events."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def publish(self, record: dict[str, Any]) -> None:
        """Broadcast an audit record. Currently a no-op stub."""
        # Real fan-out lands in Task 3.
        return None


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
