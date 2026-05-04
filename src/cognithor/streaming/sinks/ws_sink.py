"""WebSocketSink — stream events as JSON frames over WebSocket (Sprint-27 PR-C).

Second concrete consumer of :class:`Sink`. Wraps an open
WebSocket connection (``websockets.asyncio.server.ServerConnection``
or any compatible duck-typed object that supports ``await
ws.send(text)`` and a ``close()`` coroutine). One sink ↔ one
connection is the standard usage from ``cognithor agent ws``,
but the class is happy to be reused once another part of the
codebase wants to broadcast to many clients.

Wire format: each event is ``json.dumps(event.to_dict()) + "\\n"``
sent as a single text frame. The trailing newline is intentional
so a client that buffers frame text into a stream-style reader
can still split on ``"\\n"`` like the JSONL transport. Frame-level
boundaries also work — clients are free to use either.

Auth is handled at the server layer (see
:mod:`cognithor.streaming.auth` and the ``cognithor agent ws``
command in ``__main__.py``); this sink trusts that the caller
supplied an already-authenticated connection.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Protocol

from cognithor.streaming.sinks.base import Sink, SinkBufferConfig

if TYPE_CHECKING:
    from cognithor.streaming.events import StreamEvent

log = logging.getLogger(__name__)


class _SendableWebSocket(Protocol):
    """Minimal duck-typing for the WS connection object.

    Both the modern ``websockets.asyncio.server.ServerConnection``
    and the legacy ``WebSocketServerProtocol`` satisfy this. Tests
    use a small in-process fake.
    """

    async def send(self, message: str) -> None: ...


class WebSocketSink(Sink):
    """Sink that pushes JSON events to one WebSocket connection.

    The connection lifecycle (accept, auth, close) is owned by
    the calling server — this sink only writes. On a send-side
    failure (socket closed mid-stream, peer reset, etc.) the sink
    self-quarantines so subsequent events are not retried; the
    server typically tears the connection down at that point.
    """

    name = "websocket"

    def __init__(
        self,
        ws: _SendableWebSocket,
        *,
        buffer: SinkBufferConfig | None = None,
    ) -> None:
        super().__init__(buffer=buffer)
        self._ws = ws

    async def _write(self, event: StreamEvent) -> None:
        line = json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True)
        await self._ws.send(line + "\n")


def encode_event_frame(event_dict: dict[str, Any]) -> str:
    """Public helper for tests + non-Sink callers that need the wire shape."""

    return json.dumps(event_dict, ensure_ascii=False, sort_keys=True) + "\n"
