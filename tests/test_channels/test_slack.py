"""Tests für bidirektionalen SlackChannel.

Testet: Eingehende Nachrichten, Approvals via Block Kit, Streaming,
Bot-Filter, Mention-Handling, Send-Only-Fallback.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.channels.slack import SlackChannel
from jarvis.models import IncomingMessage, OutgoingMessage, PlannedAction


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def slack() -> SlackChannel:
    """SlackChannel mit Dummy-Tokens."""
    return SlackChannel(
        token="xoxb-test-token",
        app_token="xapp-test-token",
        default_channel="C12345",
    )


@pytest.fixture
def slack_send_only() -> SlackChannel:
    """SlackChannel ohne App-Token (Send-Only)."""
    return SlackChannel(token="xoxb-test-token", default_channel="C12345")


@pytest.fixture
def handler() -> AsyncMock:
    """Mock-MessageHandler der eine Antwort zurückgibt."""
    h = AsyncMock()
    h.return_value = OutgoingMessage(channel="slack", text="Antwort", session_id="s1")
    return h


# ============================================================================
# 1. Grundlegende Eigenschaften
# ============================================================================


class TestSlackProperties:
    def test_name(self, slack: SlackChannel) -> None:
        assert slack.name == "slack"

    def test_not_bidirectional_initially(self, slack: SlackChannel) -> None:
        assert slack.is_bidirectional is False

    def test_send_only_mode(self, slack_send_only: SlackChannel) -> None:
        assert slack_send_only.app_token == ""
        assert slack_send_only.is_bidirectional is False


# ============================================================================
# 2. Eingehende Nachrichten
# ============================================================================


class TestSlackIncoming:
    @pytest.mark.asyncio
    async def test_on_message_calls_handler(self, slack: SlackChannel, handler: AsyncMock) -> None:
        """Eingehende Nachricht wird an Handler weitergeleitet."""
        slack._handler = handler
        slack._client = AsyncMock()
        slack._bot_user_id = "U_BOT"

        event = {
            "user": "U_HUMAN",
            "text": "Hallo Jarvis",
            "channel": "C12345",
            "ts": "123.456",
        }
        await slack._on_message(event)

        handler.assert_called_once()
        incoming: IncomingMessage = handler.call_args[0][0]
        assert incoming.text == "Hallo Jarvis"
        assert incoming.user_id == "U_HUMAN"
        assert incoming.channel == "slack"
        assert incoming.metadata["channel_id"] == "C12345"

    @pytest.mark.asyncio
    async def test_on_message_responds_in_thread(self, slack: SlackChannel, handler: AsyncMock) -> None:
        """Antwort wird im gleichen Thread gesendet."""
        slack._handler = handler
        slack._client = AsyncMock()
        slack._bot_user_id = "U_BOT"

        event = {
            "user": "U_HUMAN",
            "text": "Frage",
            "channel": "C12345",
            "ts": "100.1",
            "thread_ts": "100.0",
        }
        await slack._on_message(event)

        slack._client.chat_postMessage.assert_called_once()
        call_kwargs = slack._client.chat_postMessage.call_args[1]
        assert call_kwargs["thread_ts"] == "100.0"

    @pytest.mark.asyncio
    async def test_ignores_own_messages(self, slack: SlackChannel, handler: AsyncMock) -> None:
        """Bot-eigene Nachrichten werden ignoriert."""
        slack._handler = handler
        slack._bot_user_id = "U_BOT"

        event = {"user": "U_BOT", "text": "echo", "channel": "C12345", "ts": "1"}
        await slack._on_message(event)

        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_ignores_bot_messages(self, slack: SlackChannel, handler: AsyncMock) -> None:
        """Nachrichten von anderen Bots werden ignoriert."""
        slack._handler = handler
        slack._bot_user_id = "U_BOT"

        event = {"user": "U_OTHER_BOT", "bot_id": "B123", "text": "ping", "channel": "C12345", "ts": "1"}
        await slack._on_message(event)

        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_strips_mention(self, slack: SlackChannel, handler: AsyncMock) -> None:
        """Bot-Mention (<@U_BOT>) wird aus Text entfernt."""
        slack._handler = handler
        slack._client = AsyncMock()
        slack._bot_user_id = "U_BOT"

        event = {
            "user": "U_HUMAN",
            "text": "<@U_BOT> Was ist 2+2?",
            "channel": "C12345",
            "ts": "1",
        }
        await slack._on_message(event)

        incoming: IncomingMessage = handler.call_args[0][0]
        assert incoming.text == "Was ist 2+2?"
        assert "<@U_BOT>" not in incoming.text

    @pytest.mark.asyncio
    async def test_ignores_empty_messages(self, slack: SlackChannel, handler: AsyncMock) -> None:
        """Leere Nachrichten werden ignoriert."""
        slack._handler = handler
        slack._bot_user_id = "U_BOT"

        event = {"user": "U_HUMAN", "text": "", "channel": "C12345", "ts": "1"}
        await slack._on_message(event)

        handler.assert_not_called()


# ============================================================================
# 3. Senden
# ============================================================================


class TestSlackSend:
    @pytest.mark.asyncio
    async def test_send_to_default_channel(self, slack: SlackChannel) -> None:
        """Nachricht geht an default_channel wenn kein Metadata-Channel."""
        slack._client = AsyncMock()
        msg = OutgoingMessage(channel="slack", text="Hello", session_id="s1")
        await slack.send(msg)

        slack._client.chat_postMessage.assert_called_once_with(
            channel="C12345", text="Hello"
        )

    @pytest.mark.asyncio
    async def test_send_to_metadata_channel(self, slack: SlackChannel) -> None:
        """Metadata-Channel überschreibt default_channel."""
        slack._client = AsyncMock()
        msg = OutgoingMessage(
            channel="slack", text="Hi",
            session_id="s1",
            metadata={"channel_id": "C99999"},
        )
        await slack.send(msg)

        call_kwargs = slack._client.chat_postMessage.call_args[1]
        assert call_kwargs["channel"] == "C99999"

    @pytest.mark.asyncio
    async def test_send_in_thread(self, slack: SlackChannel) -> None:
        """Thread-TS wird an chat.postMessage übergeben."""
        slack._client = AsyncMock()
        msg = OutgoingMessage(
            channel="slack", text="reply",
            session_id="s1",
            metadata={"thread_ts": "100.0"},
        )
        await slack.send(msg)

        call_kwargs = slack._client.chat_postMessage.call_args[1]
        assert call_kwargs["thread_ts"] == "100.0"

    @pytest.mark.asyncio
    async def test_send_without_client_logs_warning(self, slack: SlackChannel) -> None:
        """Ohne Client wird nur gewarnt, kein Crash."""
        slack._client = None
        msg = OutgoingMessage(channel="slack", text="noop", session_id="s1")
        await slack.send(msg)  # Sollte nicht crashen

    @pytest.mark.asyncio
    async def test_send_without_channel_logs_warning(self, slack: SlackChannel) -> None:
        """Ohne Channel-ID wird gewarnt."""
        slack._client = AsyncMock()
        slack.default_channel = None
        msg = OutgoingMessage(channel="slack", text="lost", session_id="s1")
        await slack.send(msg)  # Sollte nicht crashen
        slack._client.chat_postMessage.assert_not_called()


# ============================================================================
# 4. Approval via Block Kit Buttons
# ============================================================================


class TestSlackApproval:
    @pytest.mark.asyncio
    async def test_approval_returns_false_without_bidirectional(self, slack: SlackChannel) -> None:
        """Ohne Socket Mode ist Approval nicht möglich."""
        slack._bidirectional = False
        action = PlannedAction(tool="delete_file", params={"path": "/tmp/x"})
        result = await slack.request_approval("s1", action, "Gefährlich")
        assert result is False

    @pytest.mark.asyncio
    async def test_approval_sends_block_kit(self, slack: SlackChannel) -> None:
        """Block Kit Buttons werden korrekt gesendet."""
        slack._bidirectional = True
        slack._client = AsyncMock()

        action = PlannedAction(tool="delete_file", params={"path": "/tmp/x"})

        # Simuliere sofortige Genehmigung
        async def resolve_approval() -> None:
            await asyncio.sleep(0.05)
            # Finde den approval_id aus den Futures
            for aid, fut in list(slack._approval_futures.items()):
                if not fut.done():
                    fut.set_result(True)

        asyncio.get_running_loop().create_task(resolve_approval())
        result = await slack.request_approval("s1", action, "Gefährlich")

        assert result is True
        # Block Kit wurde gesendet
        slack._client.chat_postMessage.assert_called_once()
        call_kwargs = slack._client.chat_postMessage.call_args[1]
        assert call_kwargs["channel"] == "C12345"
        assert "blocks" in call_kwargs
        assert len(call_kwargs["blocks"]) == 2  # Section + Actions

    @pytest.mark.asyncio
    async def test_approval_rejected(self, slack: SlackChannel) -> None:
        """Ablehnung per Button gibt False zurück."""
        slack._bidirectional = True
        slack._client = AsyncMock()

        action = PlannedAction(tool="rm_rf", params={})

        async def reject() -> None:
            await asyncio.sleep(0.05)
            for aid, fut in list(slack._approval_futures.items()):
                if not fut.done():
                    fut.set_result(False)

        asyncio.get_running_loop().create_task(reject())
        result = await slack.request_approval("s1", action, "Sehr gefährlich")
        assert result is False

    @pytest.mark.asyncio
    async def test_on_approval_resolves_future(self, slack: SlackChannel) -> None:
        """_on_approval löst die richtige Future auf."""
        slack._client = AsyncMock()

        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        slack._approval_futures["appr_test"] = future

        body = {
            "actions": [{"value": "appr_test"}],
            "user": {"name": "alex"},
            "channel": {"id": "C12345"},
            "message": {"ts": "100.0"},
        }
        await slack._on_approval(body, approved=True)

        assert future.done()
        assert future.result() is True

    @pytest.mark.asyncio
    async def test_on_approval_updates_message(self, slack: SlackChannel) -> None:
        """Approval-Nachricht wird nach Klick aktualisiert (Buttons entfernt)."""
        slack._client = AsyncMock()

        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        slack._approval_futures["appr_update"] = future

        body = {
            "actions": [{"value": "appr_update"}],
            "user": {"name": "alex"},
            "channel": {"id": "C12345"},
            "message": {"ts": "200.0"},
        }
        await slack._on_approval(body, approved=False)

        slack._client.chat_update.assert_called_once()
        update_kwargs = slack._client.chat_update.call_args[1]
        assert "Abgelehnt" in update_kwargs["text"]
        assert update_kwargs["blocks"] == []  # Buttons entfernt


# ============================================================================
# 5. Streaming
# ============================================================================


class TestSlackStreaming:
    @pytest.mark.asyncio
    async def test_streaming_buffers_tokens(self, slack: SlackChannel) -> None:
        """Tokens werden gebuffert und als eine Nachricht gesendet."""
        slack._client = AsyncMock()

        # Mehrere Tokens schnell hintereinander senden
        await asyncio.gather(
            slack.send_streaming_token("s1", "Hallo "),
            slack.send_streaming_token("s1", "Welt"),
        )

        # Short sleep to allow the background buffer-flush task to complete
        await asyncio.sleep(0.05)
        # Mindestens ein chat_postMessage-Aufruf
        assert slack._client.chat_postMessage.call_count >= 1


# ============================================================================
# 6. Stop
# ============================================================================


class TestSlackLifecycle:
    @pytest.mark.asyncio
    async def test_stop_resets_state(self, slack: SlackChannel) -> None:
        """stop() setzt alle State-Variablen zurück."""
        slack._running = True
        slack._bidirectional = True
        slack._client = AsyncMock()

        await slack.stop()

        assert slack._running is False
        assert slack._bidirectional is False
        assert slack._client is None
