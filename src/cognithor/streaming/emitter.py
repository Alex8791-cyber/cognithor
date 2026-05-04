"""EventEmitter — single Producer fanning out to multiple Sinks (D5 + H4).

The Producer side of Sprint-27's streaming architecture. Knows
nothing about the transport (JSONL, WebSocket, future SQLite
sink, etc.) — it just calls :meth:`Sink.offer` on every
registered sink and reports aggregate fanout success.

Per H4 backpressure semantics, ``EventEmitter.emit`` blocks
*only* when:

* The event is critical (terminal or ``sink_dropped``) AND
  every sink's reserve pool is exhausted.

For non-critical events, ``emit`` is non-blocking — slow sinks
simply drop and surface a :class:`SinkDropped` notice next time
they catch up. This keeps the agent loop's hot path free of
sink-side latency.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from cognithor.streaming.events import (
    CRITICAL_EVENT_TYPES,
    StreamEvent,
)

if TYPE_CHECKING:
    from cognithor.streaming.sinks.base import Sink

log = logging.getLogger(__name__)


class EventEmitter:
    """Producer that fans an event out to a fixed set of sinks.

    Sinks are added once at startup (typically by the agent-run
    CLI bootstrapping a :class:`JsonlSink` and/or
    :class:`WebSocketSink`) and remain for the lifetime of the
    emitter. There is no removal API in PR-A — Sprint-28+ may add
    one if a use case appears.
    """

    def __init__(self) -> None:
        self._sinks: list[Sink] = []

    def add_sink(self, sink: Sink) -> None:
        """Register a sink for fanout. Order is preserved for tests."""

        self._sinks.append(sink)

    @property
    def sinks(self) -> tuple[Sink, ...]:
        """Read-only view of registered sinks."""

        return tuple(self._sinks)

    def emit(self, event: StreamEvent) -> int:
        """Fan ``event`` out to every registered sink.

        Returns the number of sinks that successfully accepted the
        event. For critical events, a return value < ``len(sinks)``
        is logged at ERROR level — the caller can decide whether
        to surface it as a process-level failure.
        """

        accepted = 0
        for sink in self._sinks:
            if sink.is_faulted:
                continue
            if sink.offer(event):
                accepted += 1

        if event.EVENT_TYPE in CRITICAL_EVENT_TYPES and accepted < len(self._sinks):
            # Don't include faulted sinks in the "missed" count —
            # they're a separate already-logged failure mode.
            live = sum(1 for s in self._sinks if not s.is_faulted)
            if accepted < live:
                log.error(
                    "emitter: critical event %s dropped by %d/%d live sinks",
                    event.EVENT_TYPE,
                    live - accepted,
                    live,
                )

        return accepted

    async def start(self) -> None:
        """Spin up every registered sink's consumer task."""

        for sink in self._sinks:
            await sink.start()

    async def stop(self) -> None:
        """Drain + close every sink. Best-effort across all sinks."""

        for sink in self._sinks:
            try:
                await sink.stop()
            except Exception:
                log.exception("emitter: sink %s failed to stop cleanly", sink.name)
