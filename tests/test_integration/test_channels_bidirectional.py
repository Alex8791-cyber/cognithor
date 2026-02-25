"""Tests für Schwäche 2: Slack & Discord bidirektional.

Beweist:
  - Bidirektionaler Modus (Socket Mode / Gateway)
  - Eingehende Nachrichten → handler Callback
  - Interaktive Approvals (Buttons / Reactions)
  - Streaming-Buffer
  - Backward-Compatibility (Send-Only ohne App-Token)
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.models import IncomingMessage, OutgoingMessage, PlannedAction


# ============================================================================
# 1. Slack-Channel Tests
# ============================================================================


class TestSlackChannelBidirectional:
    """SlackChannel im bidirektionalen Modus."""

    def test_slack_has_bidirectional_property(self) -> None:
        """SlackChannel hat is_bidirectional Property."""
        from jarvis.channels.slack import SlackChannel

        ch = SlackChannel(token="xoxb-test", app_token="xapp-test")
        assert ch.is_bidirectional is False  # Vor start()

    def test_slack_accepts_app_token(self) -> None:
        """Konstruktor akzeptiert app_token für Socket Mode."""
        from jarvis.channels.slack import SlackChannel

        ch = SlackChannel(
            token="xoxb-test",
            app_token="xapp-test",
            default_channel="C12345",
        )
        assert ch.app_token == "xapp-test"
        assert ch.default_channel == "C12345"

    def test_slack_backward_compatible_without_app_token(self) -> None:
        """Ohne app_token funktioniert SlackChannel wie bisher."""
        from jarvis.channels.slack import SlackChannel

        ch = SlackChannel(token="xoxb-test")
        assert ch.app_token == ""
        assert ch.is_bidirectional is False

    @pytest.mark.asyncio
    async def test_slack_on_message_ignores_bot(self) -> None:
        """Eigene Bot-Nachrichten werden ignoriert."""
        from jarvis.channels.slack import SlackChannel

        ch = SlackChannel(token="xoxb-test")
        ch._bot_user_id = "U_BOT"
        ch._handler = AsyncMock()
        ch._client = AsyncMock()

        # Nachricht vom Bot selbst
        await ch._on_message({"user": "U_BOT", "text": "Hallo", "channel": "C1"})
        ch._handler.assert_not_called()

        # Nachricht von einem anderen Bot
        await ch._on_message({"bot_id": "B123", "text": "Hi", "channel": "C1"})
        ch._handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_slack_on_message_forwards_to_handler(self) -> None:
        """Eingehende User-Nachrichten werden an handler weitergeleitet."""
        from jarvis.channels.slack import SlackChannel

        ch = SlackChannel(token="xoxb-test")
        ch._bot_user_id = "U_BOT"
        ch._client = AsyncMock()

        mock_response = OutgoingMessage(channel="slack", text="Antwort", session_id="s1")
        ch._handler = AsyncMock(return_value=mock_response)

        event = {
            "user": "U_USER1",
            "text": "Was gibt es Neues?",
            "channel": "C_GENERAL",
            "ts": "1234.5678",
        }
        await ch._on_message(event)

        ch._handler.assert_called_once()
        call_arg = ch._handler.call_args[0][0]
        assert isinstance(call_arg, IncomingMessage)
        assert call_arg.text == "Was gibt es Neues?"
        assert call_arg.user_id == "U_USER1"

    @pytest.mark.asyncio
    async def test_slack_on_message_strips_mention(self) -> None:
        """Bot-Mention wird aus dem Text entfernt."""
        from jarvis.channels.slack import SlackChannel

        ch = SlackChannel(token="xoxb-test")
        ch._bot_user_id = "U_BOT"
        ch._client = AsyncMock()
        ch._handler = AsyncMock(
            return_value=OutgoingMessage(channel="slack", text="OK", session_id="s1")
        )

        await ch._on_message({
            "user": "U_USER",
            "text": "<@U_BOT> Zeig mir den Status",
            "channel": "C1",
            "ts": "1.0",
        })

        call_arg = ch._handler.call_args[0][0]
        assert call_arg.text == "Zeig mir den Status"

    @pytest.mark.asyncio
    async def test_slack_approval_buttons_structure(self) -> None:
        """Approval sendet Block Kit mit Buttons."""
        from jarvis.channels.slack import SlackChannel

        ch = SlackChannel(token="xoxb-test", app_token="xapp-x", default_channel="C1")
        ch._client = AsyncMock()
        ch._bidirectional = True

        action = PlannedAction(tool="delete_file", params={"path": "/tmp/data"})

        # Starte Approval in Background (wird wegen fehlendem Button-Klick timeouten)
        task = asyncio.create_task(
            ch.request_approval("sess1", action, "Gefährliche Aktion")
        )
        await asyncio.sleep(0.05)

        # Prüfe dass postMessage mit Blocks aufgerufen wurde
        ch._client.chat_postMessage.assert_called_once()
        call_kwargs = ch._client.chat_postMessage.call_args[1]
        assert call_kwargs["channel"] == "C1"
        blocks = call_kwargs["blocks"]
        assert len(blocks) == 2
        assert blocks[1]["type"] == "actions"
        buttons = blocks[1]["elements"]
        assert buttons[0]["action_id"] == "jarvis_approve"
        assert buttons[1]["action_id"] == "jarvis_reject"

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_slack_approval_resolves_on_click(self) -> None:
        """Approval-Future wird aufgelöst wenn Button geklickt wird."""
        from jarvis.channels.slack import SlackChannel

        ch = SlackChannel(token="xoxb-test", app_token="xapp-x", default_channel="C1")
        ch._client = AsyncMock()
        ch._bidirectional = True

        # Manuell einen Approval-Future registrieren
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        ch._approval_futures["test_approval_42"] = future

        # Simuliere Button-Klick
        body = {
            "actions": [{"value": "test_approval_42"}],
            "user": {"name": "alexander"},
            "channel": {"id": "C1"},
            "message": {"ts": "1.0"},
        }
        await ch._on_approval(body, approved=True)

        assert future.result() is True
        assert "test_approval_42" not in ch._approval_futures

    @pytest.mark.asyncio
    async def test_slack_send_only_without_app_token(self) -> None:
        """Ohne Socket Mode: request_approval gibt False zurück."""
        from jarvis.channels.slack import SlackChannel

        ch = SlackChannel(token="xoxb-test")  # kein app_token
        ch._client = AsyncMock()
        # _bidirectional bleibt False

        action = PlannedAction(tool="read_file", params={})
        result = await ch.request_approval("s1", action, "Test")
        assert result is False

    @pytest.mark.asyncio
    async def test_slack_send_uses_thread(self) -> None:
        """send() antwortet im Thread wenn thread_ts in Metadata."""
        from jarvis.channels.slack import SlackChannel

        ch = SlackChannel(token="xoxb-test")
        ch._client = AsyncMock()
        ch._running = True

        msg = OutgoingMessage(
            channel="slack",
            text="Antwort",
            session_id="s1",
            metadata={"channel_id": "C1", "thread_ts": "123.456"},
        )
        await ch.send(msg)

        kwargs = ch._client.chat_postMessage.call_args[1]
        assert kwargs["thread_ts"] == "123.456"


# ============================================================================
# 2. Discord-Channel Tests
# ============================================================================


class TestDiscordChannelBidirectional:
    """DiscordChannel im bidirektionalen Modus."""

    def test_discord_has_bidirectional_property(self) -> None:
        """DiscordChannel hat is_bidirectional Property."""
        from jarvis.channels.discord import DiscordChannel

        ch = DiscordChannel(token="discord-token", channel_id=123456)
        assert ch.is_bidirectional is False

    @pytest.mark.asyncio
    async def test_discord_on_message_ignores_own(self) -> None:
        """Eigene Nachrichten werden ignoriert."""
        from jarvis.channels.discord import DiscordChannel

        ch = DiscordChannel(token="t", channel_id=100)
        bot_user = MagicMock()
        ch._client = MagicMock()
        ch._client.user = bot_user
        ch._handler = AsyncMock()

        msg = MagicMock()
        msg.author = bot_user  # Nachricht vom Bot selbst
        msg.content = "Echo"

        await ch._on_message(msg)
        ch._handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_discord_on_message_ignores_other_bots(self) -> None:
        """Bot-Nachrichten werden ignoriert."""
        from jarvis.channels.discord import DiscordChannel

        ch = DiscordChannel(token="t", channel_id=100)
        ch._client = MagicMock()
        ch._client.user = MagicMock()
        ch._handler = AsyncMock()

        msg = MagicMock()
        msg.author = MagicMock()
        msg.author.bot = True
        msg.content = "Bot-Text"

        await ch._on_message(msg)
        ch._handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_discord_on_message_from_target_channel(self) -> None:
        """Nachrichten im konfigurierten Channel werden verarbeitet."""
        from jarvis.channels.discord import DiscordChannel

        ch = DiscordChannel(token="t", channel_id=100)
        ch._client = MagicMock()
        ch._client.user = MagicMock()
        ch._client.user.id = 999

        mock_resp = OutgoingMessage(channel="discord", text="OK", session_id="s1")
        ch._handler = AsyncMock(return_value=mock_resp)

        msg = MagicMock()
        msg.author = MagicMock()
        msg.author.bot = False
        msg.author.id = 42
        msg.author.__str__ = lambda self: "alex#1234"
        msg.content = "Zeig mir den Status"
        msg.channel = MagicMock()
        msg.channel.id = 100  # Target-Channel
        msg.channel.send = AsyncMock()
        msg.guild = MagicMock()
        msg.guild.id = 1
        msg.id = 555
        msg.mentions = []

        await ch._on_message(msg)

        ch._handler.assert_called_once()
        call_arg = ch._handler.call_args[0][0]
        assert call_arg.text == "Zeig mir den Status"
        assert call_arg.channel == "discord"

    @pytest.mark.asyncio
    async def test_discord_on_message_ignores_other_channels(self) -> None:
        """Nachrichten in anderen Channels werden ignoriert (keine DM, kein Mention)."""
        from jarvis.channels.discord import DiscordChannel

        ch = DiscordChannel(token="t", channel_id=100)
        ch._client = MagicMock()
        ch._client.user = MagicMock()
        ch._handler = AsyncMock()

        msg = MagicMock()
        msg.author = MagicMock()
        msg.author.bot = False
        msg.content = "Random message"
        msg.channel = MagicMock()
        msg.channel.id = 999  # Anderer Channel
        msg.guild = MagicMock()  # Nicht DM
        msg.mentions = []  # Kein Mention

        await ch._on_message(msg)
        ch._handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_discord_on_message_handles_dm(self) -> None:
        """DMs werden immer verarbeitet (guild=None)."""
        from jarvis.channels.discord import DiscordChannel

        ch = DiscordChannel(token="t", channel_id=100)
        ch._client = MagicMock()
        ch._client.user = MagicMock()
        ch._client.user.id = 999

        mock_resp = OutgoingMessage(channel="discord", text="DM Antwort", session_id="s1")
        ch._handler = AsyncMock(return_value=mock_resp)

        msg = MagicMock()
        msg.author = MagicMock()
        msg.author.bot = False
        msg.author.id = 42
        msg.author.__str__ = lambda self: "alex#1234"
        msg.content = "DM Frage"
        msg.channel = MagicMock()
        msg.channel.id = 555
        msg.channel.send = AsyncMock()
        msg.guild = None  # DM
        msg.id = 666
        msg.mentions = []

        await ch._on_message(msg)
        ch._handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_discord_approval_reaction(self) -> None:
        """Approval-Future wird durch ✅ Reaction aufgelöst."""
        from jarvis.channels.discord import DiscordChannel

        ch = DiscordChannel(token="t", channel_id=100)
        ch._client = MagicMock()
        ch._client.user = MagicMock()

        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        requester_id = 99999
        ch._approval_messages[42] = (future, requester_id)

        # Simuliere Reaction ✅
        reaction = MagicMock()
        reaction.message = MagicMock()
        reaction.message.id = 42
        reaction.emoji = "✅"

        user = MagicMock()
        user.id = requester_id  # Richtiger User
        await ch._on_reaction(reaction, user)

        assert future.result() is True

    @pytest.mark.asyncio
    async def test_discord_approval_reject(self) -> None:
        """Approval-Future wird durch ❌ Reaction abgelehnt."""
        from jarvis.channels.discord import DiscordChannel

        ch = DiscordChannel(token="t", channel_id=100)
        ch._client = MagicMock()
        ch._client.user = MagicMock()

        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        requester_id = 99999
        ch._approval_messages[99] = (future, requester_id)

        reaction = MagicMock()
        reaction.message = MagicMock()
        reaction.message.id = 99
        reaction.emoji = "❌"

        user = MagicMock()
        user.id = requester_id
        await ch._on_reaction(reaction, user)

        assert future.result() is False

    @pytest.mark.asyncio
    async def test_discord_approval_ignores_bot_reaction(self) -> None:
        """Bot-eigene Reactions werden bei Approvals ignoriert."""
        from jarvis.channels.discord import DiscordChannel

        ch = DiscordChannel(token="t", channel_id=100)
        bot_user = MagicMock()
        ch._client = MagicMock()
        ch._client.user = bot_user

        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        ch._approval_messages[42] = (future, 99999)

        reaction = MagicMock()
        reaction.message = MagicMock()
        reaction.message.id = 42
        reaction.emoji = "✅"

        # Reaction vom Bot selbst
        await ch._on_reaction(reaction, bot_user)
        assert not future.done()  # Nicht aufgelöst


# ============================================================================
# 3. Channel Interface Compliance
# ============================================================================


class TestChannelInterfaceCompliance:
    """Beide Channels erfüllen das Channel-Interface vollständig."""

    def test_slack_implements_channel(self) -> None:
        from jarvis.channels.base import Channel
        from jarvis.channels.slack import SlackChannel

        assert issubclass(SlackChannel, Channel)
        ch = SlackChannel(token="t")
        assert ch.name == "slack"

    def test_discord_implements_channel(self) -> None:
        from jarvis.channels.base import Channel
        from jarvis.channels.discord import DiscordChannel

        assert issubclass(DiscordChannel, Channel)
        ch = DiscordChannel(token="t", channel_id=1)
        assert ch.name == "discord"

    def test_both_channels_have_required_methods(self) -> None:
        """Beide Channels haben alle abstrakten Methoden."""
        from jarvis.channels.discord import DiscordChannel
        from jarvis.channels.slack import SlackChannel

        for cls in (SlackChannel, DiscordChannel):
            for method in ("start", "stop", "send", "request_approval", "send_streaming_token"):
                assert hasattr(cls, method), f"{cls.__name__} fehlt {method}"
