"""Tests for `cognithor.streaming.sinks.ws_sink` + `cognithor.streaming.auth`.

H3 — token auto-gen + 0o600, frame format stability, multi-event
ordering on the wire, sink quarantine on send-failure.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import pytest

from cognithor.streaming import EventEmitter
from cognithor.streaming.auth import auth_token_path, load_or_create_token
from cognithor.streaming.events import (
    PlanStep,
    RunComplete,
    RunStarted,
)
from cognithor.streaming.sinks import WebSocketSink, encode_event_frame

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Auth — load_or_create_token
# ---------------------------------------------------------------------------


class TestAuthToken:
    def test_creates_token_with_64_hex_chars(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("COGNITHOR_HOME", str(tmp_path))
        token = load_or_create_token()
        assert re.fullmatch(r"[0-9a-f]{64}", token)
        # File written.
        assert auth_token_path().exists()

    def test_preserves_existing_token(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("COGNITHOR_HOME", str(tmp_path))
        first = load_or_create_token()
        second = load_or_create_token()
        assert first == second

    def test_strips_whitespace_from_existing_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("COGNITHOR_HOME", str(tmp_path))
        path = auth_token_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("  abc123\n  ", encoding="utf-8")
        assert load_or_create_token() == "abc123"

    def test_regenerates_when_file_is_blank(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("COGNITHOR_HOME", str(tmp_path))
        path = auth_token_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n", encoding="utf-8")
        token = load_or_create_token()
        assert token  # non-empty
        assert path.read_text(encoding="utf-8").strip() == token


# ---------------------------------------------------------------------------
# WebSocketSink
# ---------------------------------------------------------------------------


class _FakeWS:
    """In-process recording WebSocket double."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.fail_after: int | None = None

    async def send(self, message: str) -> None:
        if self.fail_after is not None and len(self.sent) >= self.fail_after:
            raise ConnectionError("peer reset")
        self.sent.append(message)


class TestWebSocketSink:
    @pytest.mark.asyncio
    async def test_sends_one_frame_per_event(self) -> None:
        ws = _FakeWS()
        sink = WebSocketSink(ws)
        emitter = EventEmitter()
        emitter.add_sink(sink)
        await emitter.start()
        try:
            emitter.emit(RunStarted(run_id="r", plan_path="p", step_count=1))
            emitter.emit(PlanStep(run_id="r", step=0, action={"tool": "noop"}))
            emitter.emit(RunComplete(run_id="r", receipt={}))
        finally:
            await emitter.stop()

        assert len(ws.sent) == 3
        # Each frame is exactly one JSON line + trailing newline.
        for frame in ws.sent:
            assert frame.endswith("\n")
            payload = json.loads(frame.rstrip("\n"))
            assert "event" in payload

    @pytest.mark.asyncio
    async def test_preserves_event_order(self) -> None:
        ws = _FakeWS()
        sink = WebSocketSink(ws)
        emitter = EventEmitter()
        emitter.add_sink(sink)
        await emitter.start()
        try:
            for i in range(5):
                emitter.emit(PlanStep(run_id="r", step=i, action={"tool": f"t{i}"}))
            emitter.emit(RunComplete(run_id="r", receipt={}))
        finally:
            await emitter.stop()

        types = [json.loads(f.rstrip("\n"))["event"] for f in ws.sent]
        # 5 plan_step in order, then run_complete LAST.
        assert types == ["plan_step"] * 5 + ["run_complete"]
        steps = [
            json.loads(f.rstrip("\n")).get("step")
            for f in ws.sent
            if json.loads(f.rstrip("\n"))["event"] == "plan_step"
        ]
        assert steps == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_send_failure_quarantines_sink(self) -> None:
        ws = _FakeWS()
        ws.fail_after = 1  # accept one event, then raise.
        sink = WebSocketSink(ws)
        emitter = EventEmitter()
        emitter.add_sink(sink)
        await emitter.start()
        try:
            emitter.emit(RunStarted(run_id="r", plan_path="p", step_count=2))
            emitter.emit(PlanStep(run_id="r", step=0, action={"tool": "noop"}))
            import asyncio as _a

            await _a.sleep(0.1)  # let consumer catch the raise
        finally:
            await emitter.stop()
        assert sink.is_faulted is True


# ---------------------------------------------------------------------------
# encode_event_frame helper
# ---------------------------------------------------------------------------


class TestEncodeEventFrame:
    def test_appends_newline(self) -> None:
        frame = encode_event_frame({"event": "noop", "schema_version": 1})
        assert frame.endswith("\n")

    def test_sort_keys_for_determinism(self) -> None:
        a = encode_event_frame({"b": 2, "a": 1})
        b = encode_event_frame({"a": 1, "b": 2})
        assert a == b
