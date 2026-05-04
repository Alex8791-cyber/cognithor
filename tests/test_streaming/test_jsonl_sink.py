"""Tests for `cognithor.streaming.sinks.jsonl_sink`."""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from cognithor.streaming import EventEmitter
from cognithor.streaming.events import (
    PlanStep,
    RunComplete,
    RunStarted,
)
from cognithor.streaming.sinks import JsonlSink, SinkBufferConfig


class TestJsonlSinkBasic:
    @pytest.mark.asyncio
    async def test_writes_one_line_per_event(self) -> None:
        buf = io.StringIO()
        sink = JsonlSink(stream=buf)
        emitter = EventEmitter()
        emitter.add_sink(sink)
        await emitter.start()
        try:
            emitter.emit(RunStarted(run_id="r", plan_path="p.json", step_count=1))
            emitter.emit(PlanStep(run_id="r", step=0, action={"tool": "noop"}))
            emitter.emit(RunComplete(run_id="r", receipt={"trace_id": "r"}))
        finally:
            await emitter.stop()

        lines = [line for line in buf.getvalue().splitlines() if line]
        assert len(lines) == 3
        events = [json.loads(line) for line in lines]
        assert events[0]["event"] == "run_started"
        assert events[1]["event"] == "plan_step"
        assert events[2]["event"] == "run_complete"
        # Each line must be self-contained valid JSON.
        for line, evt in zip(lines, events, strict=True):
            assert json.loads(line) == evt

    @pytest.mark.asyncio
    async def test_keys_are_sorted_for_deterministic_diffs(self) -> None:
        buf = io.StringIO()
        sink = JsonlSink(stream=buf)
        emitter = EventEmitter()
        emitter.add_sink(sink)
        await emitter.start()
        try:
            emitter.emit(RunStarted(run_id="r", plan_path="p", step_count=1))
        finally:
            await emitter.stop()
        line = buf.getvalue().strip()
        # sort_keys=True must produce a deterministic byte-stable
        # representation. Re-encoding the parsed payload with the
        # same flags must round-trip exactly.
        payload = json.loads(line)
        recompiled = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        assert line == recompiled


class TestJsonlSinkToPath:
    @pytest.mark.asyncio
    async def test_to_path_writes_to_file(self, tmp_path: Path) -> None:
        out = tmp_path / "events.jsonl"
        sink = JsonlSink.to_path(out)
        emitter = EventEmitter()
        emitter.add_sink(sink)
        await emitter.start()
        try:
            emitter.emit(RunStarted(run_id="r", plan_path="p", step_count=0))
            emitter.emit(RunComplete(run_id="r", receipt={}))
        finally:
            await emitter.stop()

        content = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(content) == 2
        first = json.loads(content[0])
        assert first["event"] == "run_started"

    @pytest.mark.asyncio
    async def test_to_path_appends(self, tmp_path: Path) -> None:
        out = tmp_path / "events.jsonl"
        out.write_text('{"existing": true}\n', encoding="utf-8")
        sink = JsonlSink.to_path(out)
        emitter = EventEmitter()
        emitter.add_sink(sink)
        await emitter.start()
        try:
            emitter.emit(RunComplete(run_id="r", receipt={}))
        finally:
            await emitter.stop()
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"existing": True}


class TestJsonlSinkFaultHandling:
    @pytest.mark.asyncio
    async def test_closed_stream_quarantines_sink(self) -> None:
        buf = io.StringIO()
        buf.close()
        sink = JsonlSink(
            stream=buf,
            buffer=SinkBufferConfig(normal_capacity=4, critical_reserve=4),
        )
        emitter = EventEmitter()
        emitter.add_sink(sink)
        await emitter.start()
        try:
            emitter.emit(RunStarted(run_id="r", plan_path="p", step_count=0))
            # Give the consumer time to attempt the write and quarantine itself.
            import asyncio

            await asyncio.sleep(0.1)
        finally:
            await emitter.stop()

        assert sink.is_faulted is True
