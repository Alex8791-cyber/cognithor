"""WebSocket server backing ``cognithor agent ws`` (Sprint-27 PR-C).

H3 security defaults locked in:

* Bind ``127.0.0.1`` by default. ``--bind 0.0.0.0`` is allowed
  but emits an explicit warning AND **still requires the token**
  — no auth-bypass-on-bind escape hatch. Port defaults to 8742.
* Bearer-token auth via ``Authorization: Bearer <token>`` header
  on the WS upgrade. Token from
  :func:`cognithor.streaming.auth.load_or_create_token`. Mismatch
  closes the connection with code ``1008`` (policy violation).
* Per-connection request schema (newline-or-frame-delimited JSON):

  .. code-block:: json

      {"type": "run_plan", "plan": <ActionPlan-shape>}
      {"type": "run_plan", "plan_path": "/abs/or/cwd-relative/path.json"}

  After the request lands, the server runs the plan, streams
  events through a :class:`WebSocketSink` for that single
  connection, then closes the connection cleanly.

The server is intentionally simple — Phase-2's VS-Code
extension is the primary consumer; external orchestrators that
want long-lived connections can be added in Sprint-28+.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from cognithor.streaming.auth import load_or_create_token
from cognithor.streaming.emitter import EventEmitter
from cognithor.streaming.runner import parse_plan_file, run_plan
from cognithor.streaming.sinks.ws_sink import WebSocketSink

if TYPE_CHECKING:
    from collections.abc import Iterable

log = logging.getLogger(__name__)

DEFAULT_PORT = 8742
DEFAULT_BIND = "127.0.0.1"

WS_CLOSE_POLICY_VIOLATION = 1008
WS_CLOSE_INTERNAL_ERROR = 1011


def _extract_bearer_token(headers: Iterable[tuple[str, str]] | object) -> str | None:
    """Pull ``Authorization: Bearer <token>`` from a websockets request.

    The websockets library exposes request headers as a mapping-
    or list-like object depending on version; we duck-type both.
    """

    auth_header: str | None = None

    get_attr = getattr(headers, "get", None)
    if callable(get_attr):
        auth_header = get_attr("Authorization")
    if auth_header is None and hasattr(headers, "__iter__"):
        for k, v in headers:
            if k.lower() == "authorization":
                auth_header = v
                break

    if not auth_header:
        return None
    parts = auth_header.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


async def _handle_request_payload(payload: str) -> tuple[object, str]:
    """Parse the first message a client sends. Returns ``(plan, kind)``.

    ``plan`` is a parsed :class:`_ParsedPlan` from
    :mod:`cognithor.streaming.runner`. ``kind`` reports how the
    plan was supplied ("inline" or "path") for the
    ``run_started`` event's ``plan_path`` field.
    """

    msg = json.loads(payload)
    if not isinstance(msg, dict) or msg.get("type") != "run_plan":
        msg_text = "first frame must be a JSON object with type='run_plan'"
        raise ValueError(msg_text)

    plan_path = msg.get("plan_path")
    inline = msg.get("plan")

    if plan_path:
        return parse_plan_file(Path(plan_path)), "path"
    if isinstance(inline, dict):
        # Marshal the inline plan dict through the same parser so
        # validation + canonicalisation are consistent.
        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            json.dump(inline, tmp)
            tmp_path = Path(tmp.name)
        try:
            plan = parse_plan_file(tmp_path)
            # Replace the synthetic temp-path with a sentinel for
            # the run_started event so consumers know the source
            # was inline.
            return dataclasses.replace(plan, plan_path="<inline>"), "inline"
        finally:
            tmp_path.unlink(missing_ok=True)

    msg_text = "run_plan request must carry either 'plan' (inline) or 'plan_path'"
    raise ValueError(msg_text)


async def _serve_connection(
    ws: object,
    *,
    expected_token: str,
) -> None:
    """Per-connection handler. Auth + parse + run + close."""

    # Extract Authorization header from whichever websockets API
    # version we're running against.
    headers = getattr(ws, "request_headers", None) or getattr(
        getattr(ws, "request", None),
        "headers",
        None,
    )
    supplied = _extract_bearer_token(headers) if headers is not None else None
    if supplied != expected_token:
        log.warning("ws: rejecting connection — invalid or missing bearer token")
        close = getattr(ws, "close", None)
        if callable(close):
            await close(code=WS_CLOSE_POLICY_VIOLATION, reason="invalid token")
        return

    # First frame must be the run_plan request.
    recv = getattr(ws, "recv", None)
    if not callable(recv):
        log.error("ws: connection object has no recv()")
        return
    payload = await recv()
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8", errors="replace")

    try:
        plan, _kind = await _handle_request_payload(payload)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        log.warning("ws: bad run_plan request: %s", exc)
        close = getattr(ws, "close", None)
        if callable(close):
            await close(code=WS_CLOSE_POLICY_VIOLATION, reason=str(exc)[:120])
        return

    sink = WebSocketSink(ws)  # type: ignore[arg-type]
    emitter = EventEmitter()
    emitter.add_sink(sink)
    await emitter.start()
    try:
        await run_plan(plan, emitter)  # type: ignore[arg-type]
    finally:
        await emitter.stop()
        close = getattr(ws, "close", None)
        if callable(close):
            await close()


async def serve(
    *,
    bind: str = DEFAULT_BIND,
    port: int = DEFAULT_PORT,
    token: str | None = None,
) -> None:
    """Start the WebSocket server and run forever (Ctrl-C to stop)."""

    expected = token or load_or_create_token()
    if bind != DEFAULT_BIND:
        log.warning(
            "cognithor agent ws: binding to %s (NOT localhost). "
            "The bearer-token check is still enforced, but a "
            "mis-rotated token here is a worse exposure than on "
            "127.0.0.1. Make sure that's what you want.",
            bind,
        )

    # Late import so unit tests don't pay the websockets dep cost
    # unless they exercise the server path.
    from websockets.asyncio.server import serve as ws_serve

    async def handler(ws: object) -> None:
        await _serve_connection(ws, expected_token=expected)

    log.info("cognithor agent ws: listening on ws://%s:%d", bind, port)
    async with ws_serve(handler, bind, port):
        await asyncio.Future()  # run forever
