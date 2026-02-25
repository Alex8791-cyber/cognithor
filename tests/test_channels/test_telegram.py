"""Tests für den Telegram-Channel.

Testet Message-Splitting, Whitelist, Approval-Workflow und Handler-Integration.
Alle Telegram-API-Aufrufe werden gemockt.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.channels.telegram import (
    MAX_MESSAGE_LENGTH,
    TelegramChannel,
    _split_message,
)
from jarvis.models import IncomingMessage, OutgoingMessage

# ============================================================================
# _split_message
# ============================================================================


class TestSplitMessage:
    """Tests für die Nachrichten-Splitter-Funktion."""

    def test_short_message_not_split(self) -> None:
        result = _split_message("Hello World")
        assert result == ["Hello World"]

    def test_exactly_max_length(self) -> None:
        text = "a" * MAX_MESSAGE_LENGTH
        result = _split_message(text)
        assert len(result) == 1
        assert result[0] == text

    def test_long_message_split_at_newline(self) -> None:
        line1 = "a" * (MAX_MESSAGE_LENGTH - 10)
        line2 = "b" * 100
        text = f"{line1}\n{line2}"

        result = _split_message(text)
        assert len(result) == 2
        assert result[0] == line1
        assert result[1] == line2

    def test_long_message_split_at_space(self) -> None:
        word = "abcdefghij"
        # Erstelle Text der nur Leerzeichen als Trennzeichen hat
        words = [word] * (MAX_MESSAGE_LENGTH // len(word) + 50)
        text = " ".join(words)

        result = _split_message(text)
        assert len(result) >= 2
        for chunk in result:
            assert len(chunk) <= MAX_MESSAGE_LENGTH

    def test_long_message_hard_split(self) -> None:
        # Text ohne Leerzeichen oder Zeilenumbrüche
        text = "x" * (MAX_MESSAGE_LENGTH + 100)
        result = _split_message(text)
        assert len(result) == 2
        assert len(result[0]) == MAX_MESSAGE_LENGTH

    def test_empty_message(self) -> None:
        result = _split_message("")
        assert result == [""]

    def test_multiple_splits(self) -> None:
        text = "\n".join(["x" * 2000] * 10)  # ~20000 chars
        result = _split_message(text)
        assert len(result) >= 3
        for chunk in result:
            assert len(chunk) <= MAX_MESSAGE_LENGTH


# ============================================================================
# TelegramChannel Initialization
# ============================================================================


class TestTelegramChannelInit:
    """Tests für die TelegramChannel-Initialisierung."""

    def test_name(self) -> None:
        ch = TelegramChannel(token="test-token")
        assert ch.name == "telegram"

    def test_allowed_users_set(self) -> None:
        ch = TelegramChannel(token="t", allowed_users=[123, 456])
        assert ch.allowed_users == {123, 456}

    def test_allowed_users_none(self) -> None:
        ch = TelegramChannel(token="t", allowed_users=None)
        assert ch.allowed_users == set()

    def test_initial_state(self) -> None:
        ch = TelegramChannel(token="t")
        assert ch._running is False
        assert ch._app is None
        assert ch._handler is None


# ============================================================================
# TelegramChannel.send
# ============================================================================


class TestTelegramChannelSend:
    """Tests für das Senden von Nachrichten."""

    @pytest.mark.asyncio
    async def test_send_without_app(self) -> None:
        ch = TelegramChannel(token="t")
        msg = OutgoingMessage(
            channel="telegram",
            text="Hello",
            metadata={"chat_id": "123"},
        )
        # Should not crash
        await ch.send(msg)

    @pytest.mark.asyncio
    async def test_send_without_chat_id(self) -> None:
        ch = TelegramChannel(token="t")
        ch._app = MagicMock()
        msg = OutgoingMessage(channel="telegram", text="Hello")
        # Should not crash
        await ch.send(msg)

    @pytest.mark.asyncio
    async def test_send_calls_bot(self) -> None:
        ch = TelegramChannel(token="t")
        ch._app = MagicMock()
        ch._app.bot.send_message = AsyncMock()

        msg = OutgoingMessage(
            channel="telegram",
            text="Hello Jarvis",
            metadata={"chat_id": "42"},
        )
        await ch.send(msg)

        ch._app.bot.send_message.assert_called_once_with(
            chat_id=42,
            text="Hello Jarvis",
            parse_mode="Markdown",
        )

    @pytest.mark.asyncio
    async def test_send_markdown_fallback(self) -> None:
        ch = TelegramChannel(token="t")
        ch._app = MagicMock()

        # Erste Sendung schlägt fehl (Markdown-Fehler)
        ch._app.bot.send_message = AsyncMock(side_effect=[Exception("Markdown error"), None])

        msg = OutgoingMessage(
            channel="telegram",
            text="*bad markdown",
            metadata={"chat_id": "42"},
        )
        await ch.send(msg)

        # Zweiter Aufruf ohne parse_mode
        assert ch._app.bot.send_message.call_count == 2
        second_call = ch._app.bot.send_message.call_args_list[1]
        assert "parse_mode" not in second_call.kwargs

    @pytest.mark.asyncio
    async def test_send_splits_long_message(self) -> None:
        ch = TelegramChannel(token="t")
        ch._app = MagicMock()
        ch._app.bot.send_message = AsyncMock()

        long_text = "x" * (MAX_MESSAGE_LENGTH + 100)
        msg = OutgoingMessage(
            channel="telegram",
            text=long_text,
            metadata={"chat_id": "42"},
        )
        await ch.send(msg)

        assert ch._app.bot.send_message.call_count == 2


# ============================================================================
# TelegramChannel._on_telegram_message
# ============================================================================


class TestTelegramOnMessage:
    """Tests für eingehende Telegram-Nachrichten."""

    def _make_update(self, user_id: int = 1, chat_id: int = 1, text: str = "Hi") -> MagicMock:
        """Erzeugt ein Mock-Update-Objekt."""
        update = MagicMock()
        update.effective_user.id = user_id
        update.effective_chat.id = chat_id
        update.effective_message.text = text
        update.effective_message.reply_text = AsyncMock()
        return update

    @pytest.mark.asyncio
    async def test_whitelist_blocks_unauthorized(self) -> None:
        ch = TelegramChannel(token="t", allowed_users=[999])
        ch._handler = AsyncMock()

        update = self._make_update(user_id=123)
        await ch._on_telegram_message(update, MagicMock())

        ch._handler.assert_not_called()
        update.effective_message.reply_text.assert_called_once()
        assert "verweigert" in update.effective_message.reply_text.call_args[0][0]

    @pytest.mark.asyncio
    async def test_whitelist_allows_authorized(self) -> None:
        ch = TelegramChannel(token="t", allowed_users=[123])
        response = OutgoingMessage(channel="telegram", text="Response")
        ch._handler = AsyncMock(return_value=response)
        ch._app = MagicMock()
        ch._app.bot.send_message = AsyncMock()

        update = self._make_update(user_id=123)
        await ch._on_telegram_message(update, MagicMock())

        ch._handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_whitelist_allows_all(self) -> None:
        ch = TelegramChannel(token="t", allowed_users=None)
        response = OutgoingMessage(channel="telegram", text="OK")
        ch._handler = AsyncMock(return_value=response)
        ch._app = MagicMock()
        ch._app.bot.send_message = AsyncMock()

        update = self._make_update(user_id=42)
        await ch._on_telegram_message(update, MagicMock())

        ch._handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_handler_replies_not_ready(self) -> None:
        ch = TelegramChannel(token="t")
        ch._handler = None

        update = self._make_update()
        await ch._on_telegram_message(update, MagicMock())

        update.effective_message.reply_text.assert_called_once()
        assert "nicht bereit" in update.effective_message.reply_text.call_args[0][0]

    @pytest.mark.asyncio
    async def test_creates_incoming_message(self) -> None:
        ch = TelegramChannel(token="t")
        response = OutgoingMessage(channel="telegram", text="OK")
        ch._handler = AsyncMock(return_value=response)
        ch._app = MagicMock()
        ch._app.bot.send_message = AsyncMock()

        update = self._make_update(user_id=42, text="Hallo Jarvis")
        await ch._on_telegram_message(update, MagicMock())

        msg = ch._handler.call_args[0][0]
        assert isinstance(msg, IncomingMessage)
        assert msg.channel == "telegram"
        assert msg.user_id == "42"
        assert msg.text == "Hallo Jarvis"

    @pytest.mark.asyncio
    async def test_handler_exception_sends_error(self) -> None:
        ch = TelegramChannel(token="t")
        ch._handler = AsyncMock(side_effect=RuntimeError("boom"))

        update = self._make_update()
        await ch._on_telegram_message(update, MagicMock())

        update.effective_message.reply_text.assert_called_once()
        assert "Fehler" in update.effective_message.reply_text.call_args[0][0]

    @pytest.mark.asyncio
    async def test_null_message_ignored(self) -> None:
        ch = TelegramChannel(token="t")
        ch._handler = AsyncMock()

        update = MagicMock()
        update.effective_message = None
        await ch._on_telegram_message(update, MagicMock())
        ch._handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_null_user_ignored(self) -> None:
        ch = TelegramChannel(token="t")
        ch._handler = AsyncMock()

        update = MagicMock()
        update.effective_message = MagicMock()
        update.effective_user = None
        await ch._on_telegram_message(update, MagicMock())
        ch._handler.assert_not_called()


# ============================================================================
# TelegramChannel._on_approval_callback
# ============================================================================


class TestApprovalCallback:
    """Tests für Approval-Inline-Keyboard-Callbacks."""

    @pytest.mark.asyncio
    async def test_approve_sets_event(self) -> None:
        ch = TelegramChannel(token="t")
        approval_id = "approval-sess1-read_file"

        event = asyncio.Event()
        ch._approval_events[approval_id] = event
        ch._approval_results[approval_id] = False

        query = MagicMock()
        query.answer = AsyncMock()
        query.data = f"approve:{approval_id}"
        query.message.text = "Test"
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query

        await ch._on_approval_callback(update, MagicMock())

        assert event.is_set()
        assert ch._approval_results[approval_id] is True

    @pytest.mark.asyncio
    async def test_deny_sets_event(self) -> None:
        ch = TelegramChannel(token="t")
        approval_id = "approval-sess1-delete_file"

        event = asyncio.Event()
        ch._approval_events[approval_id] = event
        ch._approval_results[approval_id] = True  # Default True

        query = MagicMock()
        query.answer = AsyncMock()
        query.data = f"deny:{approval_id}"
        query.message.text = "Test"
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query

        await ch._on_approval_callback(update, MagicMock())

        assert event.is_set()
        assert ch._approval_results[approval_id] is False

    @pytest.mark.asyncio
    async def test_expired_approval(self) -> None:
        ch = TelegramChannel(token="t")

        query = MagicMock()
        query.answer = AsyncMock()
        query.data = "approve:nonexistent"
        query.message.text = "Old question"
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query

        await ch._on_approval_callback(update, MagicMock())

        query.edit_message_text.assert_called_once()
        assert "Abgelaufen" in query.edit_message_text.call_args[0][0]

    @pytest.mark.asyncio
    async def test_invalid_callback_data(self) -> None:
        ch = TelegramChannel(token="t")

        query = MagicMock()
        query.answer = AsyncMock()
        query.data = "nodatahere"

        update = MagicMock()
        update.callback_query = query

        # Should not crash
        await ch._on_approval_callback(update, MagicMock())

    @pytest.mark.asyncio
    async def test_null_query_ignored(self) -> None:
        ch = TelegramChannel(token="t")

        update = MagicMock()
        update.callback_query = None

        await ch._on_approval_callback(update, MagicMock())


# ============================================================================
# TelegramChannel.send_streaming_token
# ============================================================================


class TestStreamingToken:
    """Tests für send_streaming_token (Telegram-Limitation)."""

    @pytest.mark.asyncio
    async def test_streaming_noop(self) -> None:
        ch = TelegramChannel(token="t")
        # Should not crash – Telegram doesn't support streaming
        await ch.send_streaming_token("session-1", "Hello")


# ============================================================================
# TelegramChannel Stop
# ============================================================================


class TestTelegramStop:
    """Tests für den Stop-Mechanismus."""

    @pytest.mark.asyncio
    async def test_stop_without_start(self) -> None:
        ch = TelegramChannel(token="t")
        await ch.stop()  # Should not crash
        assert ch._running is False
