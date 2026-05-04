"""Tests for `cognithor.streaming.emitter` + `sinks.base`.

Covers H4 (backpressure: bounded buffer per sink, fan-out async,
critical events bypass drop) end-to-end with a small recording
:class:`_RecordingSink` that lets us assert ordering and counts.
"""

from __future__ import annotations

import asyncio

import pytest

from cognithor.streaming.emitter import EventEmitter
from cognithor.streaming.events import (
    PlanStep,
    RunComplete,
    RunStarted,
    SinkDropped,
    StreamEvent,
)
from cognithor.streaming.sinks.base import Sink, SinkBufferConfig


class _RecordingSink(Sink):
    """In-process sink that records every event ``_write`` receives."""

    name = "recording"

    def __init__(
        self,
        *,
        buffer: SinkBufferConfig | None = None,
        write_delay: float = 0.0,
    ) -> None:
        super().__init__(buffer=buffer)
        self.received: list[StreamEvent] = []
        self._write_delay = write_delay

    async def _write(self, event: StreamEvent) -> None:
        if self._write_delay > 0:
            await asyncio.sleep(self._write_delay)
        self.received.append(event)


class _ExplodingSink(Sink):
    """Sink whose ``_write`` always raises; used to test self-quarantine."""

    name = "exploding"

    async def _write(self, event: StreamEvent) -> None:
        raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# Basic fanout
# ---------------------------------------------------------------------------


class TestFanout:
    @pytest.mark.asyncio
    async def test_emit_to_two_sinks(self) -> None:
        a = _RecordingSink()
        b = _RecordingSink()
        emitter = EventEmitter()
        emitter.add_sink(a)
        emitter.add_sink(b)
        await emitter.start()
        try:
            evt = RunStarted(run_id="r1", plan_path="plan.json", step_count=1)
            accepted = emitter.emit(evt)
            assert accepted == 2
            await asyncio.sleep(0.05)
            assert len(a.received) == 1
            assert len(b.received) == 1
            assert a.received[0].run_id == "r1"
        finally:
            await emitter.stop()

    @pytest.mark.asyncio
    async def test_emit_returns_zero_with_no_sinks(self) -> None:
        emitter = EventEmitter()
        evt = PlanStep(run_id="r", step=0, action={"tool": "noop"})
        assert emitter.emit(evt) == 0

    @pytest.mark.asyncio
    async def test_sinks_property_is_readonly_view(self) -> None:
        emitter = EventEmitter()
        emitter.add_sink(_RecordingSink())
        view = emitter.sinks
        assert isinstance(view, tuple)
        assert len(view) == 1


# ---------------------------------------------------------------------------
# H4 — backpressure: non-critical events drop, critical events bypass
# ---------------------------------------------------------------------------


class TestBackpressure:
    @pytest.mark.asyncio
    async def test_non_critical_drops_when_buffer_full(self) -> None:
        # Tiny buffer + slow consumer → producer fills and drops.
        sink = _RecordingSink(
            buffer=SinkBufferConfig(normal_capacity=2, critical_reserve=4),
            write_delay=0.05,
        )
        emitter = EventEmitter()
        emitter.add_sink(sink)
        await emitter.start()
        try:
            for i in range(20):
                evt = PlanStep(run_id="r", step=i, action={"tool": "x"})
                emitter.emit(evt)
            await asyncio.sleep(2.0)  # let consumer drain whatever it can
        finally:
            await emitter.stop()

        # Some non-critical events MUST have been dropped.
        assert len(sink.received) < 20
        # And the sink must have noticed (sink_dropped is in received
        # because the consumer drains it from the critical queue).
        drops = [e for e in sink.received if isinstance(e, SinkDropped)]
        assert drops, "sink must surface a SinkDropped notice"
        assert drops[0].sink == "recording"
        assert drops[0].dropped_count >= 1

    @pytest.mark.asyncio
    async def test_critical_event_bypasses_normal_capacity(self) -> None:
        # Fill the normal queue to its limit with the consumer paused
        # (we never call ``start``), then enqueue a terminal event —
        # it MUST land in the reserve, not drop.
        sink = _RecordingSink(
            buffer=SinkBufferConfig(normal_capacity=2, critical_reserve=4),
        )
        # Manually fill the normal queue (no consumer running yet).
        for i in range(2):
            assert sink.offer(PlanStep(run_id="r", step=i, action={"tool": "x"}))
        # 3rd non-critical → drop.
        assert sink.offer(PlanStep(run_id="r", step=99, action={"tool": "x"})) is False
        # Critical event bypasses → still accepted.
        assert sink.offer(
            RunComplete(run_id="r", receipt={}),
        )

    @pytest.mark.asyncio
    async def test_critical_reserve_exhausted_returns_false_and_logs(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        sink = _RecordingSink(
            buffer=SinkBufferConfig(normal_capacity=1, critical_reserve=1),
        )
        # Fill the reserve.
        assert sink.offer(RunComplete(run_id="r", receipt={}))
        # Reserve is now full.
        with caplog.at_level("ERROR", logger="cognithor.streaming.sinks.base"):
            ok = sink.offer(RunComplete(run_id="r", receipt={}))
        assert ok is False
        assert any("reserve exhausted" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Sink quarantine on _write failure
# ---------------------------------------------------------------------------


class TestQuarantine:
    @pytest.mark.asyncio
    async def test_exploding_sink_quarantines_after_first_write(self) -> None:
        bad = _ExplodingSink()
        good = _RecordingSink()
        emitter = EventEmitter()
        emitter.add_sink(bad)
        emitter.add_sink(good)
        await emitter.start()
        try:
            emitter.emit(RunStarted(run_id="r", plan_path="p", step_count=0))
            # Give the consumer task time to fail + quarantine itself.
            await asyncio.sleep(0.2)
            # After quarantine, further emits must skip the bad sink.
            emitter.emit(PlanStep(run_id="r", step=0, action={"tool": "noop"}))
            await asyncio.sleep(0.1)
            assert bad.is_faulted is True
            # Good sink keeps receiving.
            assert len(good.received) == 2
        finally:
            await emitter.stop()


# ---------------------------------------------------------------------------
# Stop semantics
# ---------------------------------------------------------------------------


class TestStop:
    @pytest.mark.asyncio
    async def test_stop_drains_critical_events(self) -> None:
        sink = _RecordingSink()
        emitter = EventEmitter()
        emitter.add_sink(sink)
        await emitter.start()
        emitter.emit(RunStarted(run_id="r", plan_path="p", step_count=0))
        emitter.emit(RunComplete(run_id="r", receipt={}))
        await emitter.stop()
        # Both events must have been delivered before stop returned.
        types = [e.EVENT_TYPE for e in sink.received]
        assert "run_started" in types
        assert "run_complete" in types

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self) -> None:
        sink = _RecordingSink()
        emitter = EventEmitter()
        emitter.add_sink(sink)
        await emitter.start()
        await emitter.start()
        await emitter.stop()
