"""Tests für den WebUI Channel.

Testet WebSocket-Kommunikation, Streaming, Tool-Events,
Approval-Flow, und Verbindungsmanagement.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.channels.webui import (
    WebUIChannel,
    WSMessageType,
)
from jarvis.models import IncomingMessage, OutgoingMessage, PlannedAction

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def channel() -> WebUIChannel:
    return WebUIChannel(host="127.0.0.1", port=8742)


@pytest.fixture
def auth_channel() -> WebUIChannel:
    return WebUIChannel(host="127.0.0.1", port=8742, api_token="ws-secret")


@pytest.fixture
def mock_handler() -> AsyncMock:
    handler = AsyncMock()
    handler.return_value = OutgoingMessage(
        text="WebUI-Antwort",
        session_id="ws-session",
        channel="webui",
    )
    return handler


# ============================================================================
# WSMessageType
# ============================================================================


class TestWSMessageType:
    def test_client_message_types(self) -> None:
        assert WSMessageType.USER_MESSAGE == "user_message"
        assert WSMessageType.APPROVAL_RESPONSE == "approval_response"
        assert WSMessageType.PING == "ping"

    def test_server_message_types(self) -> None:
        assert WSMessageType.ASSISTANT_MESSAGE == "assistant_message"
        assert WSMessageType.STREAM_TOKEN == "stream_token"
        assert WSMessageType.STREAM_END == "stream_end"
        assert WSMessageType.TOOL_START == "tool_start"
        assert WSMessageType.TOOL_RESULT == "tool_result"
        assert WSMessageType.ERROR == "error"
        assert WSMessageType.PONG == "pong"


# ============================================================================
# Channel Basics
# ============================================================================


class TestWebUIChannelBasics:
    def test_channel_name(self, channel: WebUIChannel) -> None:
        assert channel.name == "webui"

    @pytest.mark.asyncio
    async def test_start(self, channel: WebUIChannel, mock_handler: AsyncMock) -> None:
        await channel.start(mock_handler)
        assert channel._handler is mock_handler
        assert channel._app is not None

    @pytest.mark.asyncio
    async def test_stop_closes_connections(self, channel: WebUIChannel) -> None:
        # Mock WebSocket
        mock_ws = AsyncMock()
        channel._connections["s1"] = mock_ws
        channel._connections["s2"] = AsyncMock()

        future: asyncio.Future[bool] = asyncio.get_event_loop().create_future()
        channel._pending_approvals["req-1"] = future

        await channel.stop()

        assert len(channel._connections) == 0
        assert len(channel._pending_approvals) == 0
        assert future.done()
        mock_ws.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_handles_close_error(self, channel: WebUIChannel) -> None:
        mock_ws = AsyncMock()
        mock_ws.close.side_effect = RuntimeError("Already closed")
        channel._connections["s1"] = mock_ws

        await channel.stop()  # Sollte nicht crashen
        assert len(channel._connections) == 0

    def test_active_connections(self, channel: WebUIChannel) -> None:
        assert channel.active_connections == 0
        channel._connections["s1"] = MagicMock()
        assert channel.active_connections == 1

    def test_app_property(self, channel: WebUIChannel) -> None:
        app = channel.app
        assert app is not None
        assert channel.app is app


# ============================================================================
# WebSocket Message Handling
# ============================================================================


class TestWSMessageHandling:
    @pytest.mark.asyncio
    async def test_ping_returns_pong(self, channel: WebUIChannel) -> None:
        mock_ws = AsyncMock()
        await channel._handle_ws_message(mock_ws, "session-1", {"type": "ping"})
        mock_ws.send_text.assert_called_once()
        sent = json.loads(mock_ws.send_text.call_args[0][0])
        assert sent["type"] == "pong"

    @pytest.mark.asyncio
    async def test_user_message_calls_handler(
        self, channel: WebUIChannel, mock_handler: AsyncMock
    ) -> None:
        channel._handler = mock_handler
        mock_ws = AsyncMock()

        await channel._handle_ws_message(
            mock_ws,
            "session-1",
            {"type": "user_message", "text": "Hallo Jarvis"},
        )

        mock_handler.assert_called_once()
        call_arg = mock_handler.call_args[0][0]
        assert isinstance(call_arg, IncomingMessage)
        assert call_arg.text == "Hallo Jarvis"
        assert call_arg.channel == "webui"

    @pytest.mark.asyncio
    async def test_user_message_sends_response_and_stream_end(
        self, channel: WebUIChannel, mock_handler: AsyncMock
    ) -> None:
        channel._handler = mock_handler
        mock_ws = AsyncMock()

        await channel._handle_ws_message(
            mock_ws,
            "s1",
            {"type": "user_message", "text": "Test"},
        )

        # Sollte 2 Nachrichten senden: assistant_message + stream_end
        assert mock_ws.send_text.call_count == 2
        msg1 = json.loads(mock_ws.send_text.call_args_list[0][0][0])
        msg2 = json.loads(mock_ws.send_text.call_args_list[1][0][0])
        assert msg1["type"] == "assistant_message"
        assert msg1["text"] == "WebUI-Antwort"
        assert msg2["type"] == "stream_end"

    @pytest.mark.asyncio
    async def test_empty_message_returns_error(
        self, channel: WebUIChannel, mock_handler: AsyncMock
    ) -> None:
        channel._handler = mock_handler
        mock_ws = AsyncMock()

        await channel._handle_ws_message(
            mock_ws,
            "s1",
            {"type": "user_message", "text": ""},
        )

        sent = json.loads(mock_ws.send_text.call_args[0][0])
        assert sent["type"] == "error"

    @pytest.mark.asyncio
    async def test_message_without_handler_returns_error(self, channel: WebUIChannel) -> None:
        channel._handler = None
        mock_ws = AsyncMock()

        await channel._handle_ws_message(
            mock_ws,
            "s1",
            {"type": "user_message", "text": "Hallo"},
        )

        sent = json.loads(mock_ws.send_text.call_args[0][0])
        assert sent["type"] == "error"

    @pytest.mark.asyncio
    async def test_unknown_type_returns_error(self, channel: WebUIChannel) -> None:
        mock_ws = AsyncMock()
        await channel._handle_ws_message(mock_ws, "s1", {"type": "unknown_type"})

        sent = json.loads(mock_ws.send_text.call_args[0][0])
        assert sent["type"] == "error"
        assert "unknown_type" in sent["error"]

    @pytest.mark.asyncio
    async def test_handler_error_sends_error_msg(self, channel: WebUIChannel) -> None:
        handler = AsyncMock(side_effect=ValueError("Boom"))
        channel._handler = handler
        mock_ws = AsyncMock()

        await channel._handle_ws_message(
            mock_ws,
            "s1",
            {"type": "user_message", "text": "Test"},
        )

        sent = json.loads(mock_ws.send_text.call_args[0][0])
        assert sent["type"] == "error"
        assert "Verarbeitungsfehler" in sent["error"]


# ============================================================================
# Approval via WebSocket
# ============================================================================


class TestWSApproval:
    @pytest.mark.asyncio
    async def test_approval_response_resolves_future(self, channel: WebUIChannel) -> None:
        future: asyncio.Future[bool] = asyncio.get_event_loop().create_future()
        channel._pending_approvals["req-1"] = future
        mock_ws = AsyncMock()

        await channel._handle_ws_message(
            mock_ws,
            "s1",
            {
                "type": "approval_response",
                "request_id": "req-1",
                "approved": True,
            },
        )

        assert future.done()
        assert future.result() is True

    @pytest.mark.asyncio
    async def test_approval_no_connection_returns_false(self, channel: WebUIChannel) -> None:
        action = PlannedAction(
            tool="shell_exec",
            params={"command": "ls"},
            rationale="test",
        )
        result = await channel.request_approval("no-conn", action, "test")
        assert result is False

    @pytest.mark.asyncio
    async def test_approval_sends_request_to_ws(self, channel: WebUIChannel) -> None:
        mock_ws = AsyncMock()
        channel._connections["s1"] = mock_ws

        action = PlannedAction(
            tool="file_delete",
            params={"path": "/tmp/test.txt"},
            rationale="Cleanup",
        )

        # Starte Approval in Background, beantworte sofort
        async def respond_after_delay() -> None:
            await asyncio.sleep(0.05)
            # Finde den request_id
            for _req_id, future in channel._pending_approvals.items():
                if not future.done():
                    future.set_result(True)

        task = asyncio.create_task(respond_after_delay())
        result = await channel.request_approval("s1", action, "Cleanup")
        await task

        assert result is True
        # Prüfe dass Approval-Request gesendet wurde
        sent = json.loads(mock_ws.send_text.call_args[0][0])
        assert sent["type"] == "approval_request"
        assert sent["tool"] == "file_delete"


# ============================================================================
# Streaming & Tool Events
# ============================================================================


class TestStreaming:
    @pytest.mark.asyncio
    async def test_send_streaming_token(self, channel: WebUIChannel) -> None:
        mock_ws = AsyncMock()
        channel._connections["s1"] = mock_ws

        await channel.send_streaming_token("s1", "Hallo")

        sent = json.loads(mock_ws.send_text.call_args[0][0])
        assert sent["type"] == "stream_token"
        assert sent["token"] == "Hallo"

    @pytest.mark.asyncio
    async def test_send_streaming_no_connection(self, channel: WebUIChannel) -> None:
        # Kein Crash bei fehlender Verbindung
        await channel.send_streaming_token("nonexistent", "Token")

    @pytest.mark.asyncio
    async def test_send_tool_event(self, channel: WebUIChannel) -> None:
        mock_ws = AsyncMock()
        channel._connections["s1"] = mock_ws

        await channel.send_tool_event(
            "s1",
            WSMessageType.TOOL_START,
            "web_search",
            {"query": "Python tutorials"},
        )

        sent = json.loads(mock_ws.send_text.call_args[0][0])
        assert sent["type"] == "tool_start"
        assert sent["tool"] == "web_search"
        assert sent["data"]["query"] == "Python tutorials"

    @pytest.mark.asyncio
    async def test_send_message_via_ws(self, channel: WebUIChannel) -> None:
        mock_ws = AsyncMock()
        channel._connections["s1"] = mock_ws

        msg = OutgoingMessage(
            text="Proaktive Nachricht",
            session_id="s1",
            channel="webui",
        )
        await channel.send(msg)

        sent = json.loads(mock_ws.send_text.call_args[0][0])
        assert sent["type"] == "assistant_message"
        assert sent["text"] == "Proaktive Nachricht"


# ============================================================================
# WS Send Error Handling
# ============================================================================


class TestWSSendErrors:
    @pytest.mark.asyncio
    async def test_ws_send_handles_error(self, channel: WebUIChannel) -> None:
        mock_ws = AsyncMock()
        mock_ws.send_text.side_effect = RuntimeError("Connection lost")

        # Sollte nicht crashen
        await channel._ws_send(mock_ws, {"type": "test"})
