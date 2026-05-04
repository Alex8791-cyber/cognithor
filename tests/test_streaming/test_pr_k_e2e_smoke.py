"""Sprint-27 PR-K — end-to-end smoke for the Plan→Gate→Execute roundtrip.

Boots the real `cognithor agent ws` server (PR-C) on a random
port, connects with a websockets client that mirrors the VS-Code
extension's wire protocol (PR-F), sends a synthetic plan, and
verifies the canonical event stream lands in the right order.

The runner falls back to a deterministic mock tool when
``execute_tool`` is not provided, so this smoke does not need
a live MCP environment — it exercises the Sprint-27 streaming
+ auth + protocol surface end-to-end without the MCP tools.

This test deliberately does NOT pull in the VS Code extension's
TypeScript code; the wire-level invariants the extension binds
against (auth header, first-frame schema, event ordering) are
the same here, so a Python smoke is sufficient for CI.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
from typing import TYPE_CHECKING, Any

import pytest

from cognithor.streaming import server as ws_server

if TYPE_CHECKING:
    from pathlib import Path


def _free_localhost_port() -> int:
    """Bind to :0 to ask the OS for an unused port, then release it."""

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


async def _boot_server(*, port: int, token: str) -> asyncio.Task[Any]:
    """Run ``cognithor agent ws`` until the test cancels the task.

    Yields control with a short sleep so the websockets serve() body
    has a chance to bind the listener before the caller connects. We
    deliberately do NOT raw-socket-probe the port — the server logs
    a "did not receive a valid HTTP request" error on those connects,
    which pollutes the test output.
    """

    task = asyncio.create_task(
        ws_server.serve(bind="127.0.0.1", port=port, token=token),
    )
    # Cooperative yield is enough on every OS we test on; if the
    # listener isn't ready by the time the caller's WS handshake
    # fires, the websockets client retries on its own.
    await asyncio.sleep(0.25)
    return task


async def _drain_events(websocket: Any, *, timeout: float = 5.0) -> list[dict[str, Any]]:
    """Read every JSON frame the server emits until it closes the connection."""

    events: list[dict[str, Any]] = []
    while True:
        try:
            payload = await asyncio.wait_for(websocket.recv(), timeout=timeout)
        except TimeoutError as exc:  # pragma: no cover - failure path
            msg = f"server did not close within {timeout} s; got {len(events)} events"
            raise AssertionError(msg) from exc
        except Exception:
            # ConnectionClosed[OK|Error] from websockets — server done.
            break
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        events.append(json.loads(payload))
    return events


@pytest.mark.asyncio
async def test_plan_gate_execute_roundtrip_via_ws(tmp_path: Path) -> None:
    websockets = pytest.importorskip("websockets")

    token = "smoke-token-" + "a" * 48
    port = _free_localhost_port()

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "run_id": "run-pr-k-smoke-1",
                "agent_id": "smoke-agent",
                "goal": "verify the streaming roundtrip",
                "steps": [
                    {"tool": "read_file", "params": {"path": "README.md"}},
                    {"tool": "list_dir", "params": {"path": "."}},
                ],
            },
        ),
        encoding="utf-8",
    )

    server_task = await _boot_server(port=port, token=token)
    try:
        async with websockets.connect(  # type: ignore[attr-defined]
            f"ws://127.0.0.1:{port}/",
            additional_headers={"Authorization": f"Bearer {token}"},
        ) as ws:
            await ws.send(
                json.dumps({"type": "run_plan", "plan_path": str(plan_path)}),
            )
            events = await _drain_events(ws)
    finally:
        server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await server_task

    # ---- Assertions on the event stream ----
    types = [e.get("event") for e in events]

    assert events, "no events received"
    assert types[0] == "run_started", f"first event should be run_started, got {types[:3]}"
    assert types[-1] in {"run_complete", "run_error", "run_cancelled"}, types[-3:]

    # Two plan steps → at least two `plan_step` + two `tool_result` events.
    assert types.count("plan_step") == 2
    assert types.count("tool_result") == 2

    # Every event carries the same run_id.
    run_ids = {e.get("run_id") for e in events}
    assert run_ids == {"run-pr-k-smoke-1"}, run_ids

    # H1 invariant: every event carries its own schema_version.
    assert all("schema_version" in e for e in events), (
        "schema_version missing on at least one frame"
    )

    # Auth-failure variant — same wiring should reject a bad token.
    bad_token_events: list[dict[str, Any]] = []
    server_task = await _boot_server(port=port, token=token)
    try:
        with contextlib.suppress(Exception):
            # ConnectionClosed (1008) is the expected outcome on bad token.
            async with websockets.connect(  # type: ignore[attr-defined]
                f"ws://127.0.0.1:{port}/",
                additional_headers={"Authorization": "Bearer wrong"},
            ) as ws:
                await ws.send(
                    json.dumps({"type": "run_plan", "plan_path": str(plan_path)}),
                )
                async for frame in ws:  # pragma: no cover - should not run
                    bad_token_events.append(json.loads(frame))
    finally:
        server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await server_task

    # No events on the auth-rejected connection.
    assert bad_token_events == [], bad_token_events
