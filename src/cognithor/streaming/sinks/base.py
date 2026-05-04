"""Sink ABC + bounded-buffer + critical-event-bypass machinery (H4).

Every concrete sink (:class:`JsonlSink`, :class:`WebSocketSink`,
or any future sink that lands in Sprint-28+) inherits from
:class:`Sink` and implements :meth:`Sink._write` (the actual
transport). Backpressure semantics are enforced by the base
class so concrete sinks cannot drift away from them:

* Each sink owns a bounded ``asyncio.Queue`` of capacity
  ``SinkBufferConfig.normal_capacity`` (default 1000) plus a
  reserved-slot pool of ``SinkBufferConfig.critical_reserve``
  events (default 16). The reserve is for terminal events
  (:data:`CRITICAL_EVENT_TYPES`).
* :meth:`Sink.offer` is the producer-side entry point. Returns
  ``True`` if the event was queued, ``False`` if the sink is full
  AND the event is non-critical — in which case the sink emits a
  :class:`SinkDropped` notice into its OWN stream the next time
  it has slack.
* Critical events bypass the normal-capacity gate by drawing from
  the reserve pool. Reserve exhaustion is treated as a process-
  level failure (the sink raises) rather than silently losing a
  terminal event.
* :meth:`Sink.run` is the consumer-side coroutine — pulls events
  off the queue and calls :meth:`_write`. Concrete sinks override
  ``_write`` only.

All method-level errors in ``_write`` are caught and logged to
``cognithor.streaming.sinks.base``'s logger; sinks self-quarantine
(``self._faulted = True``) on the first exception so a runaway
``_write`` failure cannot block the producer.
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
        # Two queues: normal events use ``_queue``, critical events
        # bypass into ``_critical_queue``. The consumer drains
        # ``_critical_queue`` first on every iteration.
        self._queue: asyncio.Queue[StreamEvent] = asyncio.Queue(
            maxsize=self._buffer.normal_capacity,
        )
        self._critical_queue: asyncio.Queue[StreamEvent] = asyncio.Queue(
            maxsize=self._buffer.critical_reserve,
        )
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
            try:
                self._critical_queue.put_nowait(event)
                return True
            except asyncio.QueueFull:
                log.error(
                    "sink %s: critical-event reserve exhausted, dropping %s",
                    self.name,
                    event.EVENT_TYPE,
                )
                return False

        try:
            self._queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            self._dropped_count += 1
            return False

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
        """Drain loop: critical queue first, then normal queue.

        On stop signal: drain whatever is already in both queues,
        then exit. Does not block waiting for new events once the
        stop signal is set.
        """

        while True:
            stopping = self._stop.is_set()

            if stopping:
                event = self._drain_one_nowait()
                if event is None:
                    return  # both queues empty + stop signalled → exit
            else:
                try:
                    event = await self._next_event()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("sink %s: drain loop error", self.name)
                    self._faulted = True
                    return
                if event is None:
                    # _next_event woke us up because stop got set.
                    continue

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
            # the queues are quiet enough to accept the notice.
            if self._dropped_count > 0 and self._queue.qsize() == 0:
                dropped = self._dropped_count
                self._dropped_count = 0
                notice = SinkDropped(
                    run_id=event.run_id,
                    sink=self.name,
                    dropped_count=dropped,
                )
                try:
                    self._critical_queue.put_nowait(notice)
                except asyncio.QueueFull:
                    # Reserve is full; restore counter for next cycle.
                    self._dropped_count += dropped

    def _drain_one_nowait(self) -> StreamEvent | None:
        """Pull the next queued event without awaiting. ``None`` if both empty."""

        if not self._critical_queue.empty():
            return self._critical_queue.get_nowait()
        if not self._queue.empty():
            return self._queue.get_nowait()
        return None

    async def _next_event(self) -> StreamEvent | None:
        """Pull the next event, prioritising the critical queue.

        Returns ``None`` only on stop-signal-during-wait (so the
        caller loops back to the ``self._stop.is_set()`` check).
        """

        # Fast path: critical queue has work.
        if not self._critical_queue.empty():
            return await self._critical_queue.get()

        # Race the normal queue against the stop signal so a quiet
        # sink doesn't block forever on shutdown.
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
