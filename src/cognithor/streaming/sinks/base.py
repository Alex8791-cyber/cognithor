"""Sink ABC + bounded-buffer + critical-event-bypass machinery (H4).

Every concrete sink (:class:`JsonlSink`, :class:`WebSocketSink`,
or any future sink that lands in Sprint-28+) inherits from
:class:`Sink` and implements :meth:`Sink._write` (the actual
transport). Backpressure semantics are enforced by the base
class so concrete sinks cannot drift away from them:

* Each sink owns a single bounded ``asyncio.Queue`` of capacity
  ``normal_capacity + critical_reserve`` (defaults 1000 + 16).
  Events are accepted in **strict insertion order**; the dual
  capacity is enforced via two counters tracked at offer-time.
* Non-critical events use ``normal_capacity`` slots. When full,
  ``offer()`` returns ``False`` and the sink emits a
  :class:`SinkDropped` notice into the same stream the next time
  it has slack.
* Critical events (terminal lifecycle + ``sink_dropped``) draw
  from the ``critical_reserve`` pool; reserve exhaustion logs at
  ERROR and ``offer()`` returns ``False``. Reserve exhaustion is
  treated by the caller as a fatal fanout error.
* Insertion order is preserved on the wire — critical events
  emitted *after* a backlog of normal events still appear after
  them in the output stream. This is what the JSONL extension
  consumer needs (it switches on the LAST event to render the
  terminal banner).

All method-level errors in ``_write`` are caught and logged;
sinks self-quarantine (``self._faulted = True``) on the first
exception so a runaway transport can't block the producer.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Final

from cognithor.streaming.events import (
    CRITICAL_EVENT_TYPES,
    SinkDropped,
    StreamEvent,
)

log = logging.getLogger(__name__)

# Default buffer dimensions. Sinks may override via SinkBufferConfig.
_DEFAULT_NORMAL_CAPACITY: Final[int] = 1000
_DEFAULT_CRITICAL_RESERVE: Final[int] = 16


@dataclass(frozen=True, slots=True)
class SinkBufferConfig:
    """Per-sink buffer parameters."""

    normal_capacity: int = _DEFAULT_NORMAL_CAPACITY
    critical_reserve: int = _DEFAULT_CRITICAL_RESERVE

    def __post_init__(self) -> None:
        if self.normal_capacity < 1:
            msg = "normal_capacity must be >= 1"
            raise ValueError(msg)
        if self.critical_reserve < 1:
            msg = "critical_reserve must be >= 1"
            raise ValueError(msg)


class Sink(ABC):
    """Base class for every streaming sink (JSONL, WebSocket, ...).

    Concrete sinks override :meth:`_write` and pick a unique
    :attr:`name` (used in :class:`SinkDropped` notices and logs).
    The producer fans out to multiple sinks via
    :meth:`EventEmitter.emit`; each call invokes :meth:`offer` on
    every registered sink.
    """

    #: Short identifier surfaced in ``sink_dropped`` events and logs.
    name: str = "sink"

    def __init__(self, *, buffer: SinkBufferConfig | None = None) -> None:
        self._buffer = buffer or SinkBufferConfig()
        # Single FIFO queue preserves insertion order. Capacity is
        # ``normal_capacity + critical_reserve`` so a terminal event
        # can still slot in when the normal pool is saturated.
        max_size = self._buffer.normal_capacity + self._buffer.critical_reserve
        self._queue: asyncio.Queue[StreamEvent] = asyncio.Queue(maxsize=max_size)
        self._normal_occupancy = 0
        self._critical_occupancy = 0
        self._dropped_count = 0
        self._faulted = False
        self._stop = asyncio.Event()
        self._consumer_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Producer-side surface
    # ------------------------------------------------------------------

    def offer(self, event: StreamEvent) -> bool:
        """Best-effort enqueue. Returns ``True`` on success.

        Critical events (terminal lifecycle + ``sink_dropped``) are
        admitted to the reserved-slot pool and ``offer`` only
        returns ``False`` if the reserve is also exhausted, which
        is treated by the caller as a fatal fanout error.
        """

        if self._faulted:
            return False

        is_critical = event.EVENT_TYPE in CRITICAL_EVENT_TYPES
        if is_critical:
            if self._critical_occupancy >= self._buffer.critical_reserve:
                log.error(
                    "sink %s: critical-event reserve exhausted, dropping %s",
                    self.name,
                    event.EVENT_TYPE,
                )
                return False
            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:  # pragma: no cover — should not happen
                log.error("sink %s: queue full when putting critical event", self.name)
                return False
            self._critical_occupancy += 1
            return True

        if self._normal_occupancy >= self._buffer.normal_capacity:
            self._dropped_count += 1
            return False
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:  # pragma: no cover — should not happen
            self._dropped_count += 1
            return False
        self._normal_occupancy += 1
        return True

    # ------------------------------------------------------------------
    # Consumer-side surface
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Spin up the consumer task. Idempotent."""

        if self._consumer_task is not None and not self._consumer_task.done():
            return
        await self._open()
        self._consumer_task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Signal the consumer to drain everything queued + exit.

        Graceful-drain semantics: every event already enqueued at
        the moment ``stop()`` is called is delivered before the
        consumer exits. New events offered after ``stop()`` may or
        may not make it depending on the race with the consumer
        loop, but they are explicitly best-effort.
        """

        self._stop.set()
        if self._consumer_task is not None:
            try:
                await asyncio.wait_for(self._consumer_task, timeout=5.0)
            except TimeoutError:
                log.warning("sink %s: consumer did not stop within 5s", self.name)
                self._consumer_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await self._consumer_task
            self._consumer_task = None
        await self._close()

    async def _run(self) -> None:
        """Drain loop in strict insertion order.

        On stop signal: drain whatever is already queued, then
        exit. Does not block waiting for new events once the stop
        signal is set.
        """

        while True:
            stopping = self._stop.is_set()

            event: StreamEvent
            if stopping:
                if self._queue.empty():
                    return
                event = self._queue.get_nowait()
            else:
                try:
                    next_event = await self._next_event()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("sink %s: drain loop error", self.name)
                    self._faulted = True
                    return
                if next_event is None:
                    # _next_event woke us up because stop got set.
                    continue
                event = next_event

            # Decrement the occupancy counter that this event used.
            if event.EVENT_TYPE in CRITICAL_EVENT_TYPES:
                self._critical_occupancy = max(0, self._critical_occupancy - 1)
            else:
                self._normal_occupancy = max(0, self._normal_occupancy - 1)

            try:
                await self._write(event)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception(
                    "sink %s: _write raised on event %s — quarantining sink",
                    self.name,
                    event.EVENT_TYPE,
                )
                self._faulted = True
                return

            # Surface backpressure drops to the consumer next time
            # the queue is quiet enough to accept the notice.
            if self._dropped_count > 0 and self._queue.empty():
                dropped = self._dropped_count
                self._dropped_count = 0
                notice = SinkDropped(
                    run_id=event.run_id,
                    sink=self.name,
                    dropped_count=dropped,
                )
                if not self.offer(notice):
                    # Reserve full; restore counter for next cycle.
                    self._dropped_count += dropped

    async def _next_event(self) -> StreamEvent | None:
        """Pull the next event, racing the stop signal.

        Returns ``None`` if the stop signal fired before an event
        arrived (caller loops back to the ``stopping`` branch).
        """

        get_task = asyncio.create_task(self._queue.get())
        stop_task = asyncio.create_task(self._stop.wait())
        try:
            done, _pending = await asyncio.wait(
                {get_task, stop_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            for t in (get_task, stop_task):
                if not t.done():
                    t.cancel()

        if get_task in done:
            return get_task.result()
        return None

    # ------------------------------------------------------------------
    # Concrete-sink hooks
    # ------------------------------------------------------------------

    @abstractmethod
    async def _write(self, event: StreamEvent) -> None:
        """Transport-level write. Concrete sinks implement only this."""

    async def _open(self) -> None:
        """Hook for transport-level setup (open file, bind socket)."""

    async def _close(self) -> None:
        """Hook for transport-level teardown."""

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def is_faulted(self) -> bool:
        """``True`` iff a previous ``_write`` raised and the sink quarantined itself."""

        return self._faulted

    @property
    def dropped_count(self) -> int:
        """Pending un-reported drops, for tests + telemetry."""

        return self._dropped_count
