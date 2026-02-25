"""Tests für die erweiterten Telegram-Channel-Features.

Testet: Voice-Handling, Foto-Empfang, Dokument-Empfang,
Typing-Indicator, Datei-Versand, zentrale Nachrichtenverarbeitung.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.channels.telegram import TelegramChannel, _split_message


class TestTelegramChannelInit:
    def test_default_init(self) -> None:
        ch = TelegramChannel(token="test-token")
        assert ch.token == "test-token"
        assert ch.allowed_users == set()
        assert ch._running is False
        assert ch._typing_tasks == {}

    def test_init_with_allowed_users(self) -> None:
        ch = TelegramChannel(token="t", allowed_users=[123, 456])
        assert ch.allowed_users == {123, 456}

    def test_init_with_workspace(self, tmp_path: Path) -> None:
        ch = TelegramChannel(token="t", workspace_dir=tmp_path)
        assert ch._workspace_dir == tmp_path

    def test_init_reconnect_config(self) -> None:
        ch = TelegramChannel(token="t", max_reconnect_attempts=10)
        assert ch._max_reconnect == 10

    def test_channel_name(self) -> None:
        ch = TelegramChannel(token="t")
        assert ch.name == "telegram"


class TestProcessIncoming:
    """Tests für die zentrale _process_incoming Methode."""

    @pytest.mark.asyncio
    async def test_process_incoming_no_handler(self) -> None:
        ch = TelegramChannel(token="t")
        ch._handler = None

        mock_update = MagicMock()
        mock_update.effective_message.reply_text = AsyncMock()

        await ch._process_incoming(123, 456, "Hallo", mock_update)

        mock_update.effective_message.reply_text.assert_awaited_once()
        call_text = mock_update.effective_message.reply_text.call_args[0][0]
        assert "nicht bereit" in call_text

    @pytest.mark.asyncio
    async def test_process_incoming_success(self) -> None:
        ch = TelegramChannel(token="t")
        ch._app = MagicMock()
        ch._app.bot.send_chat_action = AsyncMock()
        ch._app.bot.send_message = AsyncMock()

        from jarvis.models import OutgoingMessage

        mock_response = OutgoingMessage(
            channel="telegram",
            text="Antwort",
            session_id="sess-1",
            metadata={},
        )
        ch._handler = AsyncMock(return_value=mock_response)

        mock_update = MagicMock()
        mock_update.effective_message.reply_text = AsyncMock()

        await ch._process_incoming(100, 200, "Test", mock_update)

        ch._handler.assert_awaited_once()
        incoming = ch._handler.call_args[0][0]
        assert incoming.text == "Test"
        assert incoming.channel == "telegram"
        assert incoming.user_id == "200"

    @pytest.mark.asyncio
    async def test_process_incoming_handler_error(self) -> None:
        ch = TelegramChannel(token="t")
        ch._app = MagicMock()
        ch._app.bot.send_chat_action = AsyncMock()
        ch._handler = AsyncMock(side_effect=RuntimeError("Boom"))

        mock_update = MagicMock()
        mock_update.effective_message.reply_text = AsyncMock()

        await ch._process_incoming(100, 200, "Test", mock_update)

        # Sollte Fehlermeldung senden
        mock_update.effective_message.reply_text.assert_awaited()
        error_text = mock_update.effective_message.reply_text.call_args[0][0]
        assert "Fehler" in error_text


class TestVoiceMessage:
    @pytest.mark.asyncio
    async def test_voice_whitelist_rejected(self) -> None:
        ch = TelegramChannel(token="t", allowed_users=[999])

        mock_update = MagicMock()
        mock_update.effective_user.id = 123  # Nicht erlaubt
        mock_update.effective_message = MagicMock()

        # Should return without processing
        await ch._on_voice_message(mock_update, None)

    @pytest.mark.asyncio
    async def test_voice_no_voice_object(self) -> None:
        ch = TelegramChannel(token="t")

        mock_update = MagicMock()
        mock_update.effective_user.id = 1
        mock_update.effective_chat.id = 1
        mock_update.effective_message.voice = None
        mock_update.effective_message.audio = None

        # Should return gracefully
        await ch._on_voice_message(mock_update, None)


class TestPhotoMessage:
    @pytest.mark.asyncio
    async def test_photo_whitelist_rejected(self) -> None:
        ch = TelegramChannel(token="t", allowed_users=[999])

        mock_update = MagicMock()
        mock_update.effective_user.id = 123
        mock_update.effective_message = MagicMock()

        await ch._on_photo_message(mock_update, None)

    @pytest.mark.asyncio
    async def test_photo_no_photos(self) -> None:
        ch = TelegramChannel(token="t")

        mock_update = MagicMock()
        mock_update.effective_user.id = 1
        mock_update.effective_chat.id = 1
        mock_update.effective_message.photo = []

        await ch._on_photo_message(mock_update, None)


class TestDocumentMessage:
    @pytest.mark.asyncio
    async def test_document_whitelist_rejected(self) -> None:
        ch = TelegramChannel(token="t", allowed_users=[999])

        mock_update = MagicMock()
        mock_update.effective_user.id = 123
        mock_update.effective_message = MagicMock()

        await ch._on_document_message(mock_update, None)

    @pytest.mark.asyncio
    async def test_document_none(self) -> None:
        ch = TelegramChannel(token="t")

        mock_update = MagicMock()
        mock_update.effective_user.id = 1
        mock_update.effective_chat.id = 1
        mock_update.effective_message.document = None

        await ch._on_document_message(mock_update, None)


class TestTranscribeAudio:
    @pytest.mark.asyncio
    async def test_transcribe_import_error(self, tmp_path: Path) -> None:
        ch = TelegramChannel(token="t")
        audio_path = tmp_path / "test.ogg"
        audio_path.write_bytes(b"fake audio")

        with patch.dict("sys.modules", {"faster_whisper": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                result = await ch._transcribe_audio(audio_path)
                assert result is None


class TestTypingIndicator:
    def test_start_typing_no_app(self) -> None:
        ch = TelegramChannel(token="t")
        ch._app = None
        task = ch._start_typing(123)
        assert task is None

    def test_stop_typing_no_task(self) -> None:
        ch = TelegramChannel(token="t")
        # Should not raise
        ch._stop_typing(123, None)

    def test_stop_typing_cancels_task(self) -> None:
        ch = TelegramChannel(token="t")
        mock_task = MagicMock()
        ch._typing_tasks[123] = mock_task

        ch._stop_typing(123, mock_task)

        mock_task.cancel.assert_called_once()
        assert 123 not in ch._typing_tasks

    def test_stop_typing_with_different_task(self) -> None:
        ch = TelegramChannel(token="t")
        old_task = MagicMock()
        new_task = MagicMock()
        ch._typing_tasks[123] = old_task

        ch._stop_typing(123, new_task)

        new_task.cancel.assert_called_once()
        old_task.cancel.assert_called_once()


class TestSendFile:
    @pytest.mark.asyncio
    async def test_send_file_no_app(self) -> None:
        ch = TelegramChannel(token="t")
        ch._app = None
        result = await ch.send_file(123, Path("/tmp/test.txt"))
        assert result is False

    @pytest.mark.asyncio
    async def test_send_file_image(self, tmp_path: Path) -> None:
        ch = TelegramChannel(token="t")
        ch._app = MagicMock()
        ch._app.bot.send_photo = AsyncMock()

        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"fake png")

        result = await ch.send_file(123, img_path, caption="Bild")

        assert result is True
        ch._app.bot.send_photo.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_file_document(self, tmp_path: Path) -> None:
        ch = TelegramChannel(token="t")
        ch._app = MagicMock()
        ch._app.bot.send_document = AsyncMock()

        pdf_path = tmp_path / "report.pdf"
        pdf_path.write_bytes(b"fake pdf")

        result = await ch.send_file(123, pdf_path, caption="Report")

        assert result is True
        ch._app.bot.send_document.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_file_error(self, tmp_path: Path) -> None:
        ch = TelegramChannel(token="t")
        ch._app = MagicMock()
        ch._app.bot.send_document = AsyncMock(side_effect=Exception("Send failed"))

        txt_path = tmp_path / "test.txt"
        txt_path.write_bytes(b"hello")

        result = await ch.send_file(123, txt_path)

        assert result is False


class TestSplitMessage:
    def test_short_message(self) -> None:
        assert _split_message("Hello") == ["Hello"]

    def test_empty_message(self) -> None:
        assert _split_message("") == [""]

    def test_long_message_splits_at_newline(self) -> None:
        msg = ("A" * 4000) + "\n" + ("B" * 100)
        chunks = _split_message(msg)
        assert len(chunks) == 2
        assert chunks[0] == "A" * 4000
        assert chunks[1] == "B" * 100

    def test_very_long_message(self) -> None:
        msg = "X" * 10000
        chunks = _split_message(msg)
        assert len(chunks) >= 2
        assert all(len(c) <= 4096 for c in chunks)


class TestTelegramSend:
    @pytest.mark.asyncio
    async def test_send_not_running(self) -> None:
        ch = TelegramChannel(token="t")
        ch._app = None

        from jarvis.models import OutgoingMessage

        msg = OutgoingMessage(channel="telegram", text="Test", metadata={"chat_id": "123"})
        # Should not raise
        await ch.send(msg)

    @pytest.mark.asyncio
    async def test_send_no_chat_id(self) -> None:
        ch = TelegramChannel(token="t")
        ch._app = MagicMock()

        from jarvis.models import OutgoingMessage

        msg = OutgoingMessage(channel="telegram", text="Test", metadata={})
        # Should not raise, just log warning
        await ch.send(msg)

    @pytest.mark.asyncio
    async def test_send_success(self) -> None:
        ch = TelegramChannel(token="t")
        ch._app = MagicMock()
        ch._app.bot.send_message = AsyncMock()

        from jarvis.models import OutgoingMessage

        msg = OutgoingMessage(channel="telegram", text="Hallo!", metadata={"chat_id": "123"})
        await ch.send(msg)

        ch._app.bot.send_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_markdown_fallback(self) -> None:
        ch = TelegramChannel(token="t")
        ch._app = MagicMock()
        # First call with Markdown fails, second without succeeds
        ch._app.bot.send_message = AsyncMock(
            side_effect=[Exception("Bad markdown"), None]
        )

        from jarvis.models import OutgoingMessage

        msg = OutgoingMessage(
            channel="telegram", text="*broken markdown", metadata={"chat_id": "123"}
        )
        await ch.send(msg)

        assert ch._app.bot.send_message.await_count == 2


class TestRequestApproval:
    @pytest.mark.asyncio
    async def test_approval_no_app(self) -> None:
        ch = TelegramChannel(token="t")
        ch._app = None

        from jarvis.models import PlannedAction

        action = PlannedAction(tool="shell", params={"cmd": "rm -rf /"})
        result = await ch.request_approval("sess-1", action, "Gefährlich")
        assert result is False

    @pytest.mark.asyncio
    async def test_approval_no_chat_id(self) -> None:
        ch = TelegramChannel(token="t")
        ch._app = MagicMock()

        from jarvis.models import PlannedAction

        action = PlannedAction(tool="shell", params={"cmd": "ls"})
        result = await ch.request_approval("unknown-session", action, "Test")
        assert result is False

    def test_extract_chat_id(self) -> None:
        ch = TelegramChannel(token="t")
        ch._session_chat_map["sess-1"] = 42
        assert ch._extract_chat_id_from_session("sess-1") == 42
        assert ch._extract_chat_id_from_session("unknown") is None


class TestStop:
    @pytest.mark.asyncio
    async def test_stop_not_running(self) -> None:
        ch = TelegramChannel(token="t")
        ch._running = False
        # Should not raise
        await ch.stop()

    @pytest.mark.asyncio
    async def test_stop_running(self) -> None:
        ch = TelegramChannel(token="t")
        ch._running = True
        ch._app = MagicMock()
        ch._app.updater = MagicMock()
        ch._app.updater.stop = AsyncMock()
        ch._app.stop = AsyncMock()
        ch._app.shutdown = AsyncMock()

        await ch.stop()

        assert ch._running is False
        assert ch._app is None
