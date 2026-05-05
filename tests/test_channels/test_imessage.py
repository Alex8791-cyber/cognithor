"""Tests fuer den IMessageChannel.

Testet: Lifecycle, Message-Handling (native + BlueBubbles),
Approval-Workflow, Streaming, Hilfsfunktionen.
Alle externen APIs (AppleScript, SQLite, httpx) werden gemockt.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognithor.channels.imessage import (
    MAX_MESSAGE_LENGTH,
    IMessageChannel,
    _escape_applescript,
    _split_message,
)
from cognithor.models import IncomingMessage, OutgoingMessage, PlannedAction

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def native_ch() -> IMessageChannel:
    """IMessageChannel im native-Modus."""
    return IMessageChannel(
        mode="native",
        allowed_handles=["+491234567"],
        polling_interval=0.1,
    )


@pytest.fixture
def bb_ch() -> IMessageChannel:
    """IMessageChannel im BlueBubbles-Modus."""
    return IMessageChannel(
        mode="bluebubbles",
        bb_url="http://localhost:1234",
        bb_password="test-pass",
        allowed_handles=["+491234567"],
        polling_interval=0.1,
    )


@pytest.fixture
def handler() -> AsyncMock:
    h = AsyncMock()
    h.return_value = OutgoingMessage(channel="imessage", text="Antwort", session_id="s1")
    return h


# ============================================================================
# Properties
# ============================================================================


class TestIMessageProperties:
    def test_name(self, native_ch: IMessageChannel) -> None:
        assert native_ch.name == "imessage"

    def test_auto_mode_non_darwin(self) -> None:
        ch = IMessageChannel(mode="auto")
        if sys.platform != "darwin":
            assert ch._mode == "bluebubbles"
        else:
            assert ch._mode == "native"

    def test_explicit_mode(self) -> None:
        ch = IMessageChannel(mode="native")
        assert ch._mode == "native"
        ch2 = IMessageChannel(mode="bluebubbles")
        assert ch2._mode == "bluebubbles"

    def test_allowed_handles_stored(self, native_ch: IMessageChannel) -> None:
        assert "+491234567" in native_ch._allowed_handles

    def test_bb_url_stripped(self, bb_ch: IMessageChannel) -> None:
        assert not bb_ch._bb_url.endswith("/")


# ============================================================================
# Start/Stop Lifecycle
# ============================================================================


class TestIMessageLifecycle:
    @pytest.mark.asyncio
    async def test_start_native_not_darwin(
        self, native_ch: IMessageChannel, handler: AsyncMock
    ) -> None:
        """Native-Modus auf nicht-Darwin gibt Fehler und startet nicht (sofern nicht macOS)."""
        if sys.platform == "darwin":
            pytest.skip("Test nur fuer nicht-macOS")
        await native_ch.start(handler)
        # Auf nicht-Darwin wird _running nicht auf True gesetzt (early return)
        assert native_ch._running is False

    @pytest.mark.asyncio
    async def test_start_bb_no_url(self, handler: AsyncMock) -> None:
        """BlueBubbles ohne URL startet nicht."""
        ch = IMessageChannel(mode="bluebubbles", bb_url="")
        await ch.start(handler)
        assert ch._running is False

    @pytest.mark.asyncio
    async def test_start_bb_with_mock_httpx(
        self, bb_ch: IMessageChannel, handler: AsyncMock
    ) -> None:
        """BlueBubbles-Start mit gemocktem httpx."""
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": {"server_version": "1.0"}}
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("cognithor.channels.imessage.httpx", create=True) as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client
            # Patch httpx import inside start()

            (__builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__)

            # Mock import httpx
            with patch.dict(
                "sys.modules", {"httpx": MagicMock(AsyncClient=MagicMock(return_value=mock_client))}
            ):
                await bb_ch.start(handler)

        assert bb_ch._running is True
        # Cleanup
        bb_ch._running = False
        if bb_ch._poll_task:
            bb_ch._poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await bb_ch._poll_task

    @pytest.mark.asyncio
    async def test_stop_cancels_poll_task(self, bb_ch: IMessageChannel) -> None:
        """stop() beendet den Poll-Task."""
        bb_ch._running = True
        bb_ch._poll_task = asyncio.create_task(asyncio.sleep(100))

        await bb_ch.stop()

        assert bb_ch._running is False
        assert bb_ch._poll_task is None

    @pytest.mark.asyncio
    async def test_stop_resolves_approvals(self, bb_ch: IMessageChannel) -> None:
        """stop() setzt pending approvals auf False."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        bb_ch._pending_approvals["test-session"] = future
        bb_ch._running = True

        await bb_ch.stop()

        assert future.done()
        assert future.result() is False

    @pytest.mark.asyncio
    async def test_stop_closes_http(self, bb_ch: IMessageChannel) -> None:
        """stop() schliesst den httpx-Client."""
        mock_client = AsyncMock()
        bb_ch._http = mock_client
        bb_ch._running = True

        await bb_ch.stop()

        mock_client.aclose.assert_called_once()
        assert bb_ch._http is None


# ============================================================================
# Polling Loop
# ============================================================================


class TestIMessagePolling:
    @pytest.mark.asyncio
    async def test_polling_loop_stops_on_cancel(self, bb_ch: IMessageChannel) -> None:
        """Polling-Loop bricht bei Cancel sauber ab."""
        bb_ch._running = True
        task = asyncio.create_task(bb_ch._polling_loop())
        await asyncio.sleep(0.05)
        bb_ch._running = False
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_poll_native_delegates(self, native_ch: IMessageChannel) -> None:
        """_poll_native ruft _query_new_messages auf."""
        native_ch._handler = AsyncMock()
        with patch.object(native_ch, "_query_new_messages", return_value=[]):
            await native_ch._poll_native()

    @pytest.mark.asyncio
    async def test_poll_bluebubbles_no_http(self, bb_ch: IMessageChannel) -> None:
        """_poll_bluebubbles ohne http-Client ist noop."""
        bb_ch._http = None
        await bb_ch._poll_bluebubbles()  # Kein Crash


# ============================================================================
# Native Message Processing
# ============================================================================


class TestNativeMessages:
    @pytest.mark.asyncio
    async def test_process_native_empty_handle(
        self, native_ch: IMessageChannel, handler: AsyncMock
    ) -> None:
        native_ch._handler = handler
        await native_ch._process_native_message(
            {"handle": "", "text": "hi", "rowid": 1, "has_attachments": False}
        )
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_native_empty_text(
        self, native_ch: IMessageChannel, handler: AsyncMock
    ) -> None:
        native_ch._handler = handler
        await native_ch._process_native_message(
            {"handle": "+491234567", "text": "", "rowid": 1, "has_attachments": False}
        )
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_native_blocked_handle(
        self, native_ch: IMessageChannel, handler: AsyncMock
    ) -> None:
        native_ch._handler = handler
        await native_ch._process_native_message(
            {"handle": "+490000000", "text": "hi", "rowid": 1, "has_attachments": False}
        )
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_native_allowed_handle(
        self, native_ch: IMessageChannel, handler: AsyncMock
    ) -> None:
        native_ch._handler = handler
        with patch.object(native_ch, "_send_native", new_callable=AsyncMock):
            await native_ch._process_native_message(
                {"handle": "+491234567", "text": "Hallo", "rowid": 2, "has_attachments": False}
            )
        handler.assert_called_once()
        incoming: IncomingMessage = handler.call_args[0][0]
        assert incoming.text == "Hallo"
        assert incoming.channel == "imessage"

    @pytest.mark.asyncio
    async def test_process_native_approval_yes(self, native_ch: IMessageChannel) -> None:
        """'ja' loest Approval auf."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        session_id = "sess-123"
        native_ch._sessions["+491234567"] = session_id
        native_ch._pending_approvals[session_id] = future
        # PASS-3 SEC-HIGH: bind the requester handle so the approval
        # check passes the new ownership gate.
        native_ch._approval_handles[session_id] = "+491234567"

        with patch.object(native_ch, "_send_native", new_callable=AsyncMock):
            await native_ch._process_native_message(
                {"handle": "+491234567", "text": "ja", "rowid": 3, "has_attachments": False}
            )

        assert future.done()
        assert future.result() is True

    @pytest.mark.asyncio
    async def test_process_native_approval_nein(self, native_ch: IMessageChannel) -> None:
        """'nein' lehnt Approval ab."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        session_id = "sess-456"
        native_ch._sessions["+491234567"] = session_id
        native_ch._pending_approvals[session_id] = future
        # PASS-3 SEC-HIGH: bind the requester handle so the approval
        # check passes the new ownership gate.
        native_ch._approval_handles[session_id] = "+491234567"

        with patch.object(native_ch, "_send_native", new_callable=AsyncMock):
            await native_ch._process_native_message(
                {"handle": "+491234567", "text": "nein", "rowid": 4, "has_attachments": False}
            )

        assert future.done()
        assert future.result() is False

    @pytest.mark.asyncio
    async def test_process_native_handler_error(self, native_ch: IMessageChannel) -> None:
        """Handler-Fehler sendet Fehlermeldung."""
        handler = AsyncMock(side_effect=RuntimeError("Boom"))
        native_ch._handler = handler
        with patch.object(native_ch, "_send_native", new_callable=AsyncMock) as send:
            await native_ch._process_native_message(
                {"handle": "+491234567", "text": "test", "rowid": 5, "has_attachments": False}
            )
            send.assert_called()
            # Letzter Aufruf = Fehlermeldung
            last_text = send.call_args[0][1]
            assert "Fehler" in last_text


# ============================================================================
# BlueBubbles Message Processing
# ============================================================================


class TestBBMessages:
    @pytest.mark.asyncio
    async def test_process_bb_from_me(self, bb_ch: IMessageChannel) -> None:
        """Eigene Nachrichten werden ignoriert."""
        handler = AsyncMock()
        bb_ch._handler = handler
        await bb_ch._process_bb_message({"isFromMe": True, "text": "x"})
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_bb_empty_handle(self, bb_ch: IMessageChannel) -> None:
        handler = AsyncMock()
        bb_ch._handler = handler
        await bb_ch._process_bb_message(
            {"isFromMe": False, "text": "hi", "handle": {}, "dateCreated": 100}
        )
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_bb_allowed_handle(
        self, bb_ch: IMessageChannel, handler: AsyncMock
    ) -> None:
        bb_ch._handler = handler
        with patch.object(bb_ch, "_send_bb", new_callable=AsyncMock):
            await bb_ch._process_bb_message(
                {
                    "isFromMe": False,
                    "text": "Hallo BB",
                    "handle": {"address": "+491234567"},
                    "dateCreated": 200,
                    "guid": "guid-1",
                }
            )
        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_bb_blocked_handle(
        self, bb_ch: IMessageChannel, handler: AsyncMock
    ) -> None:
        bb_ch._handler = handler
        await bb_ch._process_bb_message(
            {
                "isFromMe": False,
                "text": "hi",
                "handle": {"address": "+490000000"},
                "dateCreated": 300,
            }
        )
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_bb_updates_timestamp(
        self, bb_ch: IMessageChannel, handler: AsyncMock
    ) -> None:
        bb_ch._handler = handler
        with patch.object(bb_ch, "_send_bb", new_callable=AsyncMock):
            await bb_ch._process_bb_message(
                {
                    "isFromMe": False,
                    "text": "timestamp test",
                    "handle": {"address": "+491234567"},
                    "dateCreated": 999,
                }
            )
        assert bb_ch._last_bb_timestamp == 999

    @pytest.mark.asyncio
    async def test_process_bb_approval_approve(self, bb_ch: IMessageChannel) -> None:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        bb_ch._sessions["+491234567"] = "sess-bb"
        bb_ch._pending_approvals["sess-bb"] = future
        bb_ch._approval_handles["sess-bb"] = "+491234567"

        with patch.object(bb_ch, "_send_bb", new_callable=AsyncMock):
            await bb_ch._process_bb_message(
                {
                    "isFromMe": False,
                    "text": "ok",
                    "handle": {"address": "+491234567"},
                    "dateCreated": 400,
                }
            )
        assert future.done()
        assert future.result() is True

    @pytest.mark.asyncio
    async def test_process_bb_approval_reject(self, bb_ch: IMessageChannel) -> None:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        bb_ch._sessions["+491234567"] = "sess-bb2"
        bb_ch._pending_approvals["sess-bb2"] = future
        bb_ch._approval_handles["sess-bb2"] = "+491234567"

        with patch.object(bb_ch, "_send_bb", new_callable=AsyncMock):
            await bb_ch._process_bb_message(
                {
                    "isFromMe": False,
                    "text": "reject",
                    "handle": {"address": "+491234567"},
                    "dateCreated": 500,
                }
            )
        assert future.done()
        assert future.result() is False

    @pytest.mark.asyncio
    async def test_poll_bb_success(self, bb_ch: IMessageChannel) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": []}
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_resp)
        bb_ch._http = mock_http
        await bb_ch._poll_bluebubbles()
        mock_http.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_poll_bb_non_200(self, bb_ch: IMessageChannel) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_resp)
        bb_ch._http = mock_http
        await bb_ch._poll_bluebubbles()  # Kein Crash


# ============================================================================
# Send
# ============================================================================


class TestIMessageSend:
    @pytest.mark.asyncio
    async def test_send_not_running(self, bb_ch: IMessageChannel) -> None:
        bb_ch._running = False
        msg = OutgoingMessage(channel="imessage", text="noop", session_id="s1")
        await bb_ch.send(msg)  # Kein Crash

    @pytest.mark.asyncio
    async def test_send_no_handle(self, bb_ch: IMessageChannel) -> None:
        bb_ch._running = True
        msg = OutgoingMessage(channel="imessage", text="lost", session_id="unknown")
        await bb_ch.send(msg)  # Kein Crash (warning only)

    @pytest.mark.asyncio
    async def test_send_via_session(self, bb_ch: IMessageChannel) -> None:
        bb_ch._running = True
        bb_ch._mode = "bluebubbles"
        bb_ch._sessions["+491234567"] = "sess-x"
        with patch.object(bb_ch, "_send_bb", new_callable=AsyncMock) as send_bb:
            msg = OutgoingMessage(channel="imessage", text="Hallo", session_id="sess-x")
            await bb_ch.send(msg)
            send_bb.assert_called_once_with("+491234567", "Hallo")

    @pytest.mark.asyncio
    async def test_send_via_metadata(self, bb_ch: IMessageChannel) -> None:
        bb_ch._running = True
        bb_ch._mode = "bluebubbles"
        with patch.object(bb_ch, "_send_bb", new_callable=AsyncMock) as send_bb:
            msg = OutgoingMessage(
                channel="imessage",
                text="Hi",
                session_id="unknown",
                metadata={"handle": "+491234567"},
            )
            await bb_ch.send(msg)
            send_bb.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_native_mode(self, native_ch: IMessageChannel) -> None:
        native_ch._running = True
        native_ch._sessions["+491234567"] = "sess-n"
        with patch.object(native_ch, "_send_native", new_callable=AsyncMock) as send_n:
            msg = OutgoingMessage(channel="imessage", text="Test", session_id="sess-n")
            await native_ch.send(msg)
            send_n.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_bb_no_http(self, bb_ch: IMessageChannel) -> None:
        bb_ch._http = None
        await bb_ch._send_bb("+491234567", "noop")  # Kein Crash

    @pytest.mark.asyncio
    async def test_send_bb_success(self, bb_ch: IMessageChannel) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_resp)
        bb_ch._http = mock_http
        await bb_ch._send_bb("+491234567", "Hallo")
        mock_http.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_bb_error(self, bb_ch: IMessageChannel) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_resp)
        bb_ch._http = mock_http
        await bb_ch._send_bb("+491234567", "Fehler")  # Kein Crash

    @pytest.mark.asyncio
    async def test_send_bb_exception(self, bb_ch: IMessageChannel) -> None:
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(side_effect=RuntimeError("Network"))
        bb_ch._http = mock_http
        await bb_ch._send_bb("+491234567", "Netzwerkfehler")  # Kein Crash


# ============================================================================
# Approval Workflow
# ============================================================================


class TestIMessageApproval:
    @pytest.mark.asyncio
    async def test_approval_no_handle(self, bb_ch: IMessageChannel) -> None:
        action = PlannedAction(tool="delete", params={})
        result = await bb_ch.request_approval("unknown-session", action, "Test")
        assert result is False

    @pytest.mark.asyncio
    async def test_approval_timeout(self, bb_ch: IMessageChannel) -> None:
        bb_ch._sessions["+491234567"] = "sess-timeout"
        bb_ch._mode = "bluebubbles"
        with patch.object(bb_ch, "_send_bb", new_callable=AsyncMock):
            with patch("cognithor.channels.imessage.APPROVAL_TIMEOUT", 0.05):
                action = PlannedAction(tool="rm", params={})
                result = await bb_ch.request_approval("sess-timeout", action, "Danger")
        assert result is False

    @pytest.mark.asyncio
    async def test_resolve_approval_approved(self, bb_ch: IMessageChannel) -> None:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        bb_ch._pending_approvals["sess-r"] = future
        with patch.object(bb_ch, "_send_bb", new_callable=AsyncMock):
            await bb_ch._resolve_approval("sess-r", approved=True, handle="+491234567")
        assert future.result() is True


# ============================================================================
# Streaming
# ============================================================================


class TestIMessageStreaming:
    @pytest.mark.asyncio
    async def test_streaming_token(self, bb_ch: IMessageChannel) -> None:
        bb_ch._running = True
        bb_ch._sessions["+491234567"] = "sess-stream"
        bb_ch._mode = "bluebubbles"
        with patch.object(bb_ch, "send", new_callable=AsyncMock):
            await bb_ch.send_streaming_token("sess-stream", "Token ")
            await asyncio.sleep(0.6)  # Wait for buffer flush


# ============================================================================
# Hilfsfunktionen
# ============================================================================


class TestIMessageHelpers:
    def test_get_or_create_session(self, bb_ch: IMessageChannel) -> None:
        s1 = bb_ch._get_or_create_session("+491234567")
        s2 = bb_ch._get_or_create_session("+491234567")
        assert s1 == s2
        s3 = bb_ch._get_or_create_session("+490000000")
        assert s3 != s1

    def test_handle_for_session(self, bb_ch: IMessageChannel) -> None:
        bb_ch._sessions["+491234567"] = "sess-h"
        assert bb_ch._handle_for_session("sess-h") == "+491234567"
        assert bb_ch._handle_for_session("unknown") is None

    def test_split_message_short(self) -> None:
        assert _split_message("Hello") == ["Hello"]

    def test_split_message_long(self) -> None:
        long_text = "A" * (MAX_MESSAGE_LENGTH + 100)
        chunks = _split_message(long_text)
        assert len(chunks) >= 2
        assert all(len(c) <= MAX_MESSAGE_LENGTH for c in chunks)

    def test_split_message_with_spaces(self) -> None:
        text = " ".join(["word"] * 600)  # ~3000 chars
        chunks = _split_message(text)
        assert len(chunks) >= 2

    def test_escape_applescript(self) -> None:
        assert _escape_applescript('Say "hello"') == 'Say \\"hello\\"'
        assert _escape_applescript("line1\nline2") == "line1\\nline2"
        assert _escape_applescript("back\\slash") == "back\\\\slash"

    def test_query_new_messages_no_db(self, native_ch: IMessageChannel) -> None:
        """Ohne Messages-DB gibt leere Liste zurueck."""
        result = native_ch._query_new_messages()
        assert result == []
