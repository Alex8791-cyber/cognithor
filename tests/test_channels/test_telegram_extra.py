"""Extra tests for TelegramChannel -- covers remaining gaps.

Focus: _polish_vision_text, _transcribe_audio paths, circuit breaker in send,
request_approval timeout, session mapping persistence.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.channels.telegram import TelegramChannel
from jarvis.models import OutgoingMessage, PlannedAction


@pytest.fixture
def ch() -> TelegramChannel:
    return TelegramChannel(token="test-token")


class TestPolishVisionText:
    @pytest.mark.asyncio
    async def test_polish_success(self, ch: TelegramChannel) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": "Polished text"}}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await ch._polish_vision_text("Raw text")
        assert result == "Polished text"

    @pytest.mark.asyncio
    async def test_polish_error_returns_raw(self, ch: TelegramChannel) -> None:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=RuntimeError("fail"))

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await ch._polish_vision_text("Raw stays")
        assert result == "Raw stays"

    @pytest.mark.asyncio
    async def test_polish_empty_returns_raw(self, ch: TelegramChannel) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": "  "}}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await ch._polish_vision_text("Raw text")
        assert result == "Raw text"

    @pytest.mark.asyncio
    async def test_polish_non_200(self, ch: TelegramChannel) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 500

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await ch._polish_vision_text("Keep raw")
        assert result == "Keep raw"


class TestTranscribeAudioPaths:
    @pytest.mark.asyncio
    async def test_whisper_model_reuse(self, ch: TelegramChannel) -> None:
        mock_model = MagicMock()
        seg = MagicMock()
        seg.text = "Hello"
        mock_model.transcribe.return_value = ([seg], MagicMock())
        ch._whisper_model = mock_model

        mock_fw = MagicMock()
        with patch.dict("sys.modules", {"faster_whisper": mock_fw}):
            result = await ch._transcribe_audio(Path(tempfile.gettempdir()) / "x.ogg")
        assert result == "Hello"
        mock_fw.WhisperModel.assert_not_called()  # model reused

    @pytest.mark.asyncio
    async def test_whisper_model_lazy_init(self, ch: TelegramChannel) -> None:
        ch._whisper_model = None
        mock_model = MagicMock()
        seg = MagicMock()
        seg.text = "World"
        mock_model.transcribe.return_value = ([seg], MagicMock())

        mock_fw = MagicMock()
        mock_fw.WhisperModel.return_value = mock_model

        with patch.dict("sys.modules", {"faster_whisper": mock_fw}):
            result = await ch._transcribe_audio(Path(tempfile.gettempdir()) / "x.ogg")
        assert result == "World"
        mock_fw.WhisperModel.assert_called_once()

    @pytest.mark.asyncio
    async def test_whisper_exception(self, ch: TelegramChannel) -> None:
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("crash")
        ch._whisper_model = mock_model

        mock_fw = MagicMock()
        with patch.dict("sys.modules", {"faster_whisper": mock_fw}):
            result = await ch._transcribe_audio(Path(tempfile.gettempdir()) / "x.ogg")
        assert result is None

    @pytest.mark.asyncio
    async def test_whisper_import_error(self, ch: TelegramChannel) -> None:
        with patch.dict("sys.modules", {"faster_whisper": None}):
            result = await ch._transcribe_audio(Path(tempfile.gettempdir()) / "x.ogg")
        assert result is None


class TestSendCircuitBreaker:
    @pytest.mark.asyncio
    async def test_send_circuit_breaker_open(self, ch: TelegramChannel) -> None:
        from jarvis.utils.circuit_breaker import CircuitBreakerOpen

        ch._app = MagicMock()
        ch._circuit_breaker = MagicMock()
        ch._circuit_breaker.call = AsyncMock(side_effect=CircuitBreakerOpen("telegram_api", 60.0))

        msg = OutgoingMessage(channel="telegram", text="hello", metadata={"chat_id": "42"})
        await ch.send(msg)  # should not raise

    @pytest.mark.asyncio
    async def test_send_both_markdown_and_plain_fail(self, ch: TelegramChannel) -> None:
        ch._app = MagicMock()
        ch._app.bot.send_message = AsyncMock(side_effect=Exception("always fails"))

        msg = OutgoingMessage(channel="telegram", text="test", metadata={"chat_id": "42"})
        await ch.send(msg)  # should not raise (logs exception)


class TestApprovalWorkflow:
    @pytest.mark.asyncio
    async def test_approval_approved(self, ch: TelegramChannel) -> None:
        ch._app = MagicMock()
        ch._app.bot.send_message = AsyncMock()
        ch._session_chat_map["s1"] = 100

        action = PlannedAction(tool="test", params={})

        async def _fake_wait_for(coro, *, timeout):
            # Simulate immediate approval
            approval_id = list(ch._approval_events.keys())[0]
            ch._approval_results[approval_id] = True
            ch._approval_events[approval_id].set()
            await coro
            return True

        with patch("jarvis.channels.telegram.asyncio.wait_for", side_effect=_fake_wait_for):
            result = await ch.request_approval("s1", action, "test")
        assert result is True

    @pytest.mark.asyncio
    async def test_approval_import_error(self, ch: TelegramChannel) -> None:
        ch._app = MagicMock()
        ch._session_chat_map["s1"] = 100

        action = PlannedAction(tool="test", params={})
        with patch.dict("sys.modules", {"telegram": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                result = await ch.request_approval("s1", action, "test")
        assert result is False


class TestWebhookConfig:
    def test_webhook_properties(self) -> None:
        ch = TelegramChannel(
            token="t",
            use_webhook=True,
            webhook_url="https://example.com/tg",
            webhook_port=8443,
            ssl_certfile="cert.pem",
            ssl_keyfile="key.pem",
        )
        assert ch._use_webhook is True
        assert ch._webhook_url == "https://example.com/tg"
        assert ch._webhook_port == 8443
        assert ch._ssl_certfile == "cert.pem"


class TestStopWebhook:
    @pytest.mark.asyncio
    async def test_stop_webhook_mode(self, ch: TelegramChannel) -> None:
        ch._running = True
        ch._use_webhook = True
        ch._app = MagicMock()
        ch._app.bot.delete_webhook = AsyncMock()
        ch._app.stop = AsyncMock()
        ch._app.shutdown = AsyncMock()
        runner = MagicMock()
        runner.cleanup = AsyncMock()
        ch._webhook_runner = runner

        await ch.stop()
        assert ch._running is False
        runner.cleanup.assert_called_once()
        assert ch._webhook_runner is None

    @pytest.mark.asyncio
    async def test_stop_exception_graceful(self, ch: TelegramChannel) -> None:
        ch._running = True
        ch._use_webhook = False
        ch._app = MagicMock()
        ch._app.updater = MagicMock()
        ch._app.updater.stop = AsyncMock(side_effect=RuntimeError("error"))
        ch._app.stop = AsyncMock(side_effect=RuntimeError("error"))
        ch._app.shutdown = AsyncMock(side_effect=RuntimeError("error"))

        await ch.stop()  # should not raise
        assert ch._running is False
