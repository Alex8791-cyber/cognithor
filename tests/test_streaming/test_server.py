"""Tests for the WebSocket server in `cognithor.streaming.server`.

H3 — auth header parsing, bad-token rejection, run_plan request
parsing (path + inline). Full async-server boot/connect E2E is
covered in PR-K's smoke test; these tests exercise the
unit-level seams.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import pytest

from cognithor.streaming.server import (
    WS_CLOSE_POLICY_VIOLATION,
    _extract_bearer_token,
    _handle_request_payload,
    _serve_connection,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Bearer-token header extraction
# ---------------------------------------------------------------------------


class TestExtractBearerToken:
    def test_dict_like_headers(self) -> None:
        headers = {"Authorization": "Bearer abc123"}
        assert _extract_bearer_token(headers) == "abc123"

    def test_iterable_headers(self) -> None:
        headers = [("X-Other", "y"), ("Authorization", "Bearer xyz789")]
        assert _extract_bearer_token(headers) == "xyz789"

    def test_iterable_headers_case_insensitive(self) -> None:
        headers = [("authorization", "Bearer secret-lower")]
        assert _extract_bearer_token(headers) == "secret-lower"

    def test_missing_header(self) -> None:
        assert _extract_bearer_token({}) is None

    def test_non_bearer_scheme(self) -> None:
        assert _extract_bearer_token({"Authorization": "Basic dXNlcjpwYXNz"}) is None

    def test_empty_token(self) -> None:
        assert _extract_bearer_token({"Authorization": "Bearer "}) is None


# ---------------------------------------------------------------------------
# run_plan request payload parsing
# ---------------------------------------------------------------------------


class TestHandleRequestPayload:
    @pytest.mark.asyncio
    async def test_path_form(self, tmp_path: Path) -> None:
        plan_path = tmp_path / "plan.json"
        plan_path.write_text(
            json.dumps(
                {
                    "run_id": "r1",
                    "steps": [{"tool": "noop", "params": {}}],
                },
            ),
            encoding="utf-8",
        )
        msg = json.dumps({"type": "run_plan", "plan_path": str(plan_path)})
        plan, kind = await _handle_request_payload(msg)
        assert kind == "path"
        assert plan.run_id == "r1"  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_inline_form(self) -> None:
        msg = json.dumps(
            {
                "type": "run_plan",
                "plan": {
                    "run_id": "inline-r",
                    "steps": [{"tool": "noop", "params": {}}],
                },
            },
        )
        plan, kind = await _handle_request_payload(msg)
        assert kind == "inline"
        assert plan.plan_path == "<inline>"  # type: ignore[attr-defined]
        assert plan.run_id == "inline-r"  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_wrong_type_raises(self) -> None:
        msg = json.dumps({"type": "wrong"})
        with pytest.raises(ValueError, match="run_plan"):
            await _handle_request_payload(msg)

    @pytest.mark.asyncio
    async def test_neither_path_nor_inline(self) -> None:
        msg = json.dumps({"type": "run_plan"})
        with pytest.raises(ValueError, match="plan"):
            await _handle_request_payload(msg)


# ---------------------------------------------------------------------------
# _serve_connection — bad-token and happy-path
# ---------------------------------------------------------------------------


class _FakeWSConnection:
    """In-process double for `websockets.asyncio.server.ServerConnection`."""

    def __init__(self, *, token: str | None, first_message: str | None = None) -> None:
        if token is None:
            self.request_headers: dict[str, str] = {}
        else:
            self.request_headers = {"Authorization": f"Bearer {token}"}
        self._first_message = first_message
        self.sent: list[str] = []
        self.closed_with: tuple[int | None, str | None] = (None, None)
        self._first_consumed = False

    async def recv(self) -> str:
        if self._first_consumed or self._first_message is None:
            await asyncio.sleep(3600)  # never returns; tests timeout instead
            raise RuntimeError("recv called twice")
        self._first_consumed = True
        return self._first_message

    async def send(self, message: str) -> None:
        self.sent.append(message)

    async def close(
        self,
        code: int | None = None,
        reason: str | None = None,
    ) -> None:
        self.closed_with = (code, reason)


class TestServeConnection:
    @pytest.mark.asyncio
    async def test_rejects_missing_token(self) -> None:
        ws = _FakeWSConnection(token=None)
        await _serve_connection(ws, expected_token="server-secret")  # type: ignore[arg-type]
        assert ws.closed_with[0] == WS_CLOSE_POLICY_VIOLATION
        assert ws.sent == []

    @pytest.mark.asyncio
    async def test_rejects_wrong_token(self) -> None:
        ws = _FakeWSConnection(token="client-wrong-secret")
        await _serve_connection(ws, expected_token="server-secret")  # type: ignore[arg-type]
        assert ws.closed_with[0] == WS_CLOSE_POLICY_VIOLATION

    @pytest.mark.asyncio
    async def test_accepts_correct_token_and_runs_plan(self) -> None:
        plan_msg: dict[str, Any] = {
            "type": "run_plan",
            "plan": {
                "run_id": "ws-r",
                "steps": [{"tool": "noop", "params": {}}],
            },
        }
        ws = _FakeWSConnection(token="shared", first_message=json.dumps(plan_msg))
        await _serve_connection(ws, expected_token="shared")  # type: ignore[arg-type]

        # Streamed events landed.
        assert len(ws.sent) > 0
        types = [json.loads(f.rstrip("\n"))["event"] for f in ws.sent]
        assert types[0] == "run_started"
        assert types[-1] == "run_complete"
        # Connection closed cleanly after run.
        assert ws.closed_with[0] is None or ws.closed_with[0] != WS_CLOSE_POLICY_VIOLATION

    @pytest.mark.asyncio
    async def test_rejects_bad_first_frame(self) -> None:
        ws = _FakeWSConnection(token="shared", first_message="not json at all")
        await _serve_connection(ws, expected_token="shared")  # type: ignore[arg-type]
        assert ws.closed_with[0] == WS_CLOSE_POLICY_VIOLATION
