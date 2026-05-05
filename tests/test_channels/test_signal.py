"""Tests fuer den SignalChannel.

Testet: Lifecycle, Webhook/Polling, Message-Handling,
Approval-Workflow, Streaming, Attachment-Handling, Voice-Transkription.
Alle externen APIs (signal-cli-rest-api, aiohttp, whisper) werden gemockt.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognithor.channels.signal import MAX_MESSAGE_LENGTH, SignalChannel, _split_message
from cognithor.models import IncomingMessage, OutgoingMessage, PlannedAction

if TYPE_CHECKING:
    from pathlib import Path

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def signal_ch(tmp_path: Path) -> SignalChannel:
    return SignalChannel(
        api_url="http://localhost:8080",
        phone_number="+491111111",
        allowed_numbers=["+491234567"],
        use_polling=True,
        polling_interval=0.1,
        workspace_dir=tmp_path / "signal",
    )


@pytest.fixture
def signal_webhook(tmp_path: Path) -> SignalChannel:
    return SignalChannel(
        api_url="http://localhost:8080",
        phone_number="+491111111",
        use_polling=False,
        workspace_dir=tmp_path / "signal",
    )


@pytest.fixture
def handler() -> AsyncMock:
    h = AsyncMock()
    h.return_value = OutgoingMessage(channel="signal", text="Antwort", session_id="s1")
    return h


# ============================================================================
# Properties
# ============================================================================


class TestSignalProperties:
    def test_name(self, signal_ch: SignalChannel) -> None:
        assert signal_ch.name == "signal"

    def test_api_url_stored(self, signal_ch: SignalChannel) -> None:
        assert signal_ch._api_url == "http://localhost:8080"

    def test_allowed_numbers(self, signal_ch: SignalChannel) -> None:
        assert "+491234567" in signal_ch._allowed_numbers


# ============================================================================
# Lifecycle
# ============================================================================


class TestSignalLifecycle:
    @pytest.mark.asyncio
    async def test_start_polling(self, signal_ch: SignalChannel, handler: AsyncMock) -> None:
        mock_http = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"versions": {"signal-cli": "1.0"}}
        mock_http.get = AsyncMock(return_value=mock_resp)

        with patch("httpx.AsyncClient", return_value=mock_http):
            await signal_ch.start(handler)

        assert signal_ch._running is True
        assert signal_ch._poll_task is not None

        # Cleanup
        await signal_ch.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_poll(self, signal_ch: SignalChannel) -> None:
        signal_ch._running = True
        signal_ch._poll_task = asyncio.create_task(asyncio.sleep(100))
        signal_ch._http = AsyncMock()

        await signal_ch.stop()

        assert signal_ch._running is False
        assert signal_ch._poll_task is None

    @pytest.mark.asyncio
    async def test_stop_resolves_approvals(self, signal_ch: SignalChannel) -> None:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        signal_ch._pending_approvals["sess"] = future
        signal_ch._running = True
        signal_ch._http = AsyncMock()

        await signal_ch.stop()

        assert future.done()
        assert future.result() is False

    @pytest.mark.asyncio
    async def test_stop_closes_webhook(self, signal_ch: SignalChannel) -> None:
        signal_ch._running = True
        mock_runner = AsyncMock()
        signal_ch._webhook_runner = mock_runner
        signal_ch._http = AsyncMock()

        await signal_ch.stop()

        mock_runner.cleanup.assert_called_once()
        assert signal_ch._webhook_runner is None

    @pytest.mark.asyncio
    async def test_stop_closes_http(self, signal_ch: SignalChannel) -> None:
        mock_http = AsyncMock()
        signal_ch._http = mock_http
        signal_ch._running = True

        await signal_ch.stop()

        mock_http.aclose.assert_called_once()
        assert signal_ch._http is None

    @pytest.mark.asyncio
    async def test_setup_webhook_refuses_non_localhost_without_secret(self) -> None:
        """Non-localhost bind ohne webhook_secret muss fail-closed sein."""
        ch = SignalChannel(
            phone_number="+491234567",
            webhook_host="0.0.0.0",
            webhook_port=8090,
            webhook_secret="",
        )
        with pytest.raises(RuntimeError, match="webhook_secret"):
            await ch._setup_webhook()

    @pytest.mark.asyncio
    async def test_setup_webhook_allows_non_localhost_with_secret(self) -> None:
        """Non-localhost bind MIT webhook_secret darf starten."""
        ch = SignalChannel(
            phone_number="+491234567",
            webhook_host="0.0.0.0",
            webhook_port=18099,
            webhook_secret="s3cret",
        )
        try:
            await ch._setup_webhook()
        finally:
            if ch._webhook_runner is not None:
                await ch._webhook_runner.cleanup()


# ============================================================================
# Outbound: Send
# ============================================================================


class TestSignalSend:
    @pytest.mark.asyncio
    async def test_send_text_success(self, signal_ch: SignalChannel) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_resp)
        signal_ch._http = mock_http

        await signal_ch._send_text("+491234567", "Hallo")
        mock_http.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_text_error(self, signal_ch: SignalChannel) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Error"
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_resp)
        signal_ch._http = mock_http

        await signal_ch._send_text("+491234567", "Error")  # Kein Crash

    @pytest.mark.asyncio
    async def test_send_text_no_http(self, signal_ch: SignalChannel) -> None:
        signal_ch._http = None
        await signal_ch._send_text("+491234567", "noop")  # Kein Crash

    @pytest.mark.asyncio
    async def test_send_not_running(self, signal_ch: SignalChannel) -> None:
        signal_ch._running = False
        msg = OutgoingMessage(channel="signal", text="noop", session_id="s1")
        await signal_ch.send(msg)  # Kein Crash

    @pytest.mark.asyncio
    async def test_send_no_phone(self, signal_ch: SignalChannel) -> None:
        signal_ch._running = True
        msg = OutgoingMessage(channel="signal", text="lost", session_id="unknown")
        await signal_ch.send(msg)  # Kein Crash

    @pytest.mark.asyncio
    async def test_send_via_session(self, signal_ch: SignalChannel) -> None:
        signal_ch._running = True
        signal_ch._sessions["+491234567"] = "sess-x"
        with patch.object(signal_ch, "_send_text", new_callable=AsyncMock) as send:
            msg = OutgoingMessage(channel="signal", text="Test", session_id="sess-x")
            await signal_ch.send(msg)
            send.assert_called_once_with("+491234567", "Test")

    @pytest.mark.asyncio
    async def test_send_via_metadata(self, signal_ch: SignalChannel) -> None:
        signal_ch._running = True
        with patch.object(signal_ch, "_send_text", new_callable=AsyncMock) as send:
            msg = OutgoingMessage(
                channel="signal",
                text="Hi",
                session_id="unknown",
                metadata={"phone_number": "+491234567"},
            )
            await signal_ch.send(msg)
            send.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_attachment_success(self, signal_ch: SignalChannel, tmp_path: Path) -> None:
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_resp)
        signal_ch._http = mock_http
        await signal_ch._send_attachment("+491234567", test_file, "hier")

    @pytest.mark.asyncio
    async def test_send_attachment_no_http(self, signal_ch: SignalChannel, tmp_path: Path) -> None:
        signal_ch._http = None
        await signal_ch._send_attachment("+491234567", tmp_path / "x.txt")


# ============================================================================
# Inbound: Message Processing
# ============================================================================


class TestSignalIncoming:
    @pytest.mark.asyncio
    async def test_process_payload_no_source(self, signal_ch: SignalChannel) -> None:
        await signal_ch._process_webhook_payload({"envelope": {}})  # Kein Crash

    @pytest.mark.asyncio
    async def test_process_payload_blocked_number(
        self, signal_ch: SignalChannel, handler: AsyncMock
    ) -> None:
        signal_ch._handler = handler
        await signal_ch._process_webhook_payload(
            {
                "envelope": {
                    "sourceNumber": "+490000000",
                    "dataMessage": {"message": "hi"},
                },
            }
        )
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_payload_no_data_msg(
        self, signal_ch: SignalChannel, handler: AsyncMock
    ) -> None:
        signal_ch._handler = handler
        await signal_ch._process_webhook_payload(
            {
                "envelope": {"sourceNumber": "+491234567"},
            }
        )
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_payload_text_message(
        self, signal_ch: SignalChannel, handler: AsyncMock
    ) -> None:
        signal_ch._handler = handler
        with patch.object(signal_ch, "_send_text", new_callable=AsyncMock):
            await signal_ch._process_webhook_payload(
                {
                    "envelope": {
                        "sourceNumber": "+491234567",
                        "sourceName": "Alex",
                        "dataMessage": {
                            "message": "Hallo Signal",
                            "timestamp": 12345,
                        },
                    },
                }
            )
        handler.assert_called_once()
        incoming: IncomingMessage = handler.call_args[0][0]
        assert incoming.text == "Hallo Signal"
        assert incoming.channel == "signal"
        assert incoming.metadata["phone_number"] == "+491234567"

    @pytest.mark.asyncio
    async def test_process_payload_empty_text_with_attachment(
        self, signal_ch: SignalChannel, handler: AsyncMock
    ) -> None:
        signal_ch._handler = handler
        with patch.object(signal_ch, "_send_text", new_callable=AsyncMock):
            await signal_ch._process_webhook_payload(
                {
                    "envelope": {
                        "sourceNumber": "+491234567",
                        "dataMessage": {
                            "message": "",
                            "attachments": [
                                {"id": "att1", "contentType": "image/png", "filename": "photo.png"}
                            ],
                        },
                    },
                }
            )
        handler.assert_called_once()
        incoming: IncomingMessage = handler.call_args[0][0]
        assert "Attachment" in incoming.text

    @pytest.mark.asyncio
    async def test_process_payload_empty_text_no_attachment(
        self, signal_ch: SignalChannel, handler: AsyncMock
    ) -> None:
        signal_ch._handler = handler
        await signal_ch._process_webhook_payload(
            {
                "envelope": {
                    "sourceNumber": "+491234567",
                    "dataMessage": {"message": ""},
                },
            }
        )
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_payload_quote_approval(self, signal_ch: SignalChannel) -> None:
        """Quote mit 'Genehmigung erforderlich' loest Approval aus."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        signal_ch._sessions["+491234567"] = "sess-appr"
        signal_ch._pending_approvals["sess-appr"] = future

        with patch.object(signal_ch, "_send_text", new_callable=AsyncMock):
            await signal_ch._process_webhook_payload(
                {
                    "envelope": {
                        "sourceNumber": "+491234567",
                        "dataMessage": {
                            "message": "ja",
                            "quote": {"text": "Genehmigung erforderlich - Tool: xyz"},
                        },
                    },
                }
            )
        assert future.done()
        assert future.result() is True

    @pytest.mark.asyncio
    async def test_process_payload_handler_error(self, signal_ch: SignalChannel) -> None:
        handler = AsyncMock(side_effect=RuntimeError("Boom"))
        signal_ch._handler = handler
        with patch.object(signal_ch, "_send_text", new_callable=AsyncMock) as send:
            await signal_ch._process_webhook_payload(
                {
                    "envelope": {
                        "sourceNumber": "+491234567",
                        "dataMessage": {"message": "crash"},
                    },
                }
            )
            # Fehlermeldung gesendet
            assert send.call_count >= 1

    @pytest.mark.asyncio
    async def test_process_with_voice_transcription(
        self, signal_ch: SignalChannel, handler: AsyncMock
    ) -> None:
        signal_ch._handler = handler
        mock_whisper = MagicMock()
        mock_segments = [MagicMock(text="Transkription")]
        mock_whisper.transcribe.return_value = (mock_segments, None)
        signal_ch._whisper = mock_whisper

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"audio-data"
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        signal_ch._http = mock_http

        with patch.object(signal_ch, "_send_text", new_callable=AsyncMock):
            await signal_ch._process_webhook_payload(
                {
                    "envelope": {
                        "sourceNumber": "+491234567",
                        "dataMessage": {
                            "message": "",
                            "attachments": [
                                {
                                    "id": "voice1",
                                    "contentType": "audio/ogg",
                                    "filename": "voice.ogg",
                                }
                            ],
                        },
                    },
                }
            )

        handler.assert_called_once()
        incoming: IncomingMessage = handler.call_args[0][0]
        assert "Transkription" in incoming.text


# ============================================================================
# Attachment Download
# ============================================================================


class TestSignalAttachments:
    @pytest.mark.asyncio
    async def test_download_success(self, signal_ch: SignalChannel) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"file-bytes"
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        signal_ch._http = mock_http

        result = await signal_ch._download_attachment("att-123")
        assert result == b"file-bytes"

    @pytest.mark.asyncio
    async def test_download_failure(self, signal_ch: SignalChannel) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        signal_ch._http = mock_http

        result = await signal_ch._download_attachment("att-404")
        assert result is None

    @pytest.mark.asyncio
    async def test_download_no_http(self, signal_ch: SignalChannel) -> None:
        signal_ch._http = None
        result = await signal_ch._download_attachment("att-x")
        assert result is None


# ============================================================================
# Transcription
# ============================================================================


class TestSignalTranscription:
    @pytest.mark.asyncio
    async def test_transcribe_no_whisper(self, signal_ch: SignalChannel) -> None:
        signal_ch._whisper = None
        result = await signal_ch._transcribe_audio(b"audio")
        assert "nicht verfuegbar" in result

    @pytest.mark.asyncio
    async def test_transcribe_success(self, signal_ch: SignalChannel) -> None:
        mock_whisper = MagicMock()
        mock_segments = [MagicMock(text="Hello World")]
        mock_whisper.transcribe.return_value = (mock_segments, None)
        signal_ch._whisper = mock_whisper

        result = await signal_ch._transcribe_audio(b"audio-data")
        assert result == "Hello World"

    @pytest.mark.asyncio
    async def test_transcribe_error(self, signal_ch: SignalChannel) -> None:
        mock_whisper = MagicMock()
        mock_whisper.transcribe.side_effect = RuntimeError("Whisper fail")
        signal_ch._whisper = mock_whisper

        result = await signal_ch._transcribe_audio(b"audio-data")
        assert "fehlgeschlagen" in result


# ============================================================================
# Approval Workflow
# ============================================================================


class TestSignalApproval:
    @pytest.mark.asyncio
    async def test_approval_no_phone(self, signal_ch: SignalChannel) -> None:
        action = PlannedAction(tool="delete", params={})
        result = await signal_ch.request_approval("unknown-sess", action, "Test")
        assert result is False

    @pytest.mark.asyncio
    async def test_approval_timeout(self, signal_ch: SignalChannel) -> None:
        signal_ch._sessions["+491234567"] = "sess-t"
        with patch.object(signal_ch, "_send_text", new_callable=AsyncMock):
            with patch("cognithor.channels.signal.APPROVAL_TIMEOUT", 0.05):
                action = PlannedAction(tool="rm", params={})
                result = await signal_ch.request_approval("sess-t", action, "Danger")
        assert result is False

    @pytest.mark.asyncio
    async def test_handle_approval_reply_approve(self, signal_ch: SignalChannel) -> None:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        signal_ch._sessions["+491234567"] = "sess-a"
        signal_ch._pending_approvals["sess-a"] = future

        with patch.object(signal_ch, "_send_text", new_callable=AsyncMock):
            await signal_ch._handle_approval_reply("+491234567", "ja")
        assert future.result() is True

    @pytest.mark.asyncio
    async def test_handle_approval_reply_reject(self, signal_ch: SignalChannel) -> None:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        signal_ch._sessions["+491234567"] = "sess-r"
        signal_ch._pending_approvals["sess-r"] = future

        with patch.object(signal_ch, "_send_text", new_callable=AsyncMock):
            await signal_ch._handle_approval_reply("+491234567", "nein")
        assert future.result() is False

    @pytest.mark.asyncio
    async def test_handle_approval_reply_no_future(self, signal_ch: SignalChannel) -> None:
        """Ohne pending Future kein Crash."""
        signal_ch._sessions["+491234567"] = "sess-none"
        await signal_ch._handle_approval_reply("+491234567", "ja")


# ============================================================================
# Streaming
# ============================================================================


class TestSignalStreaming:
    @pytest.mark.asyncio
    async def test_streaming_token(self, signal_ch: SignalChannel) -> None:
        signal_ch._running = True
        signal_ch._sessions["+491234567"] = "sess-stream"
        with patch.object(signal_ch, "send", new_callable=AsyncMock):
            await signal_ch.send_streaming_token("sess-stream", "Token ")
            await asyncio.sleep(0.6)


# ============================================================================
# Helpers
# ============================================================================


class TestSignalHelpers:
    def test_get_or_create_session(self, signal_ch: SignalChannel) -> None:
        s1 = signal_ch._get_or_create_session("+491234567")
        s2 = signal_ch._get_or_create_session("+491234567")
        assert s1 == s2

    def test_phone_for_session(self, signal_ch: SignalChannel) -> None:
        signal_ch._sessions["+491234567"] = "sess-p"
        assert signal_ch._phone_for_session("sess-p") == "+491234567"
        assert signal_ch._phone_for_session("unknown") is None

    def test_split_message_short(self) -> None:
        assert _split_message("Hi") == ["Hi"]

    def test_split_message_long(self) -> None:
        long_text = "X" * (MAX_MESSAGE_LENGTH + 50)
        chunks = _split_message(long_text)
        assert len(chunks) >= 2

    @pytest.mark.asyncio
    async def test_polling_loop_stops(self, signal_ch: SignalChannel) -> None:
        signal_ch._running = True
        signal_ch._http = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []
        signal_ch._http.get = AsyncMock(return_value=mock_resp)

        task = asyncio.create_task(signal_ch._polling_loop())
        await asyncio.sleep(0.15)
        signal_ch._running = False
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_webhook_health(self, signal_webhook: SignalChannel) -> None:
        """_handle_health gibt OK zurueck."""
        request = MagicMock()
        mock_web = MagicMock()
        mock_web.json_response = MagicMock(return_value={"status": "ok"})
        mock_aiohttp = MagicMock()
        mock_aiohttp.web = mock_web
        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp, "aiohttp.web": mock_web}):
            await signal_webhook._handle_health(request)
