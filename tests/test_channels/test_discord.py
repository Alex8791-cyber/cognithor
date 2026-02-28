"""Tests für bidirektionalen DiscordChannel.

Testet: Eingehende Nachrichten, Approvals via Reactions, Streaming,
Bot-Filter, Mention-Handling, DM-Handling.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from jarvis.channels.discord import DiscordChannel
from jarvis.models import IncomingMessage, OutgoingMessage, PlannedAction


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def discord_ch() -> DiscordChannel:
    """DiscordChannel mit Dummy-Token."""
    return DiscordChannel(token="discord-test-token", channel_id=123456789)


@pytest.fixture
def handler() -> AsyncMock:
    """Mock-MessageHandler."""
    h = AsyncMock()
    h.return_value = OutgoingMessage(channel="discord", text="Antwort", session_id="s1")
    return h


def _make_discord_message(
    content: str,
    author_id: int = 99999,
    author_bot: bool = False,
    channel_id: int = 123456789,
    guild_id: int | None = 12345,
    mentions: list[Any] | None = None,
) -> MagicMock:
    """Erstellt ein Mock-discord.Message-Objekt."""
    msg = MagicMock()
    msg.content = content
    msg.author.id = author_id
    msg.author.bot = author_bot
    msg.author.__eq__ = lambda self, other: self is other
    msg.channel.id = channel_id
    msg.channel.send = AsyncMock()
    msg.id = 555

    if guild_id:
        msg.guild = MagicMock()
        msg.guild.id = guild_id
    else:
        msg.guild = None  # DM

    msg.mentions = mentions or []
    return msg


# ============================================================================
# 1. Grundlegende Eigenschaften
# ============================================================================


class TestDiscordProperties:
    def test_name(self, discord_ch: DiscordChannel) -> None:
        assert discord_ch.name == "discord"

    def test_not_bidirectional_initially(self, discord_ch: DiscordChannel) -> None:
        assert discord_ch.is_bidirectional is False

    def test_channel_id_stored(self, discord_ch: DiscordChannel) -> None:
        assert discord_ch.channel_id == 123456789


# ============================================================================
# 2. Eingehende Nachrichten
# ============================================================================


class TestDiscordIncoming:
    @pytest.mark.asyncio
    async def test_on_message_calls_handler(
        self, discord_ch: DiscordChannel, handler: AsyncMock,
    ) -> None:
        """Nachricht im Ziel-Channel wird an Handler weitergeleitet."""
        discord_ch._handler = handler
        discord_ch._running = True
        discord_ch._client = MagicMock()
        discord_ch._client.user = MagicMock()
        discord_ch._client.user.id = 88888  # Bot-ID

        msg = _make_discord_message("Hallo Jarvis", channel_id=123456789)
        await discord_ch._on_message(msg)

        handler.assert_called_once()
        incoming: IncomingMessage = handler.call_args[0][0]
        assert incoming.text == "Hallo Jarvis"
        assert incoming.channel == "discord"
        assert incoming.user_id == "99999"

    @pytest.mark.asyncio
    async def test_on_message_responds_in_channel(
        self, discord_ch: DiscordChannel, handler: AsyncMock,
    ) -> None:
        """Antwort wird im gleichen Channel gesendet."""
        discord_ch._handler = handler
        discord_ch._client = MagicMock()
        discord_ch._client.user = MagicMock()
        discord_ch._client.user.id = 88888

        msg = _make_discord_message("Frage", channel_id=123456789)
        await discord_ch._on_message(msg)

        msg.channel.send.assert_called_once_with("Antwort")

    @pytest.mark.asyncio
    async def test_ignores_own_messages(
        self, discord_ch: DiscordChannel, handler: AsyncMock,
    ) -> None:
        """Bot-eigene Nachrichten werden ignoriert."""
        discord_ch._handler = handler
        bot_user = MagicMock()
        discord_ch._client = MagicMock()
        discord_ch._client.user = bot_user

        msg = _make_discord_message("echo")
        msg.author = bot_user  # Gleicher User = Bot
        await discord_ch._on_message(msg)

        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_ignores_other_bot_messages(
        self, discord_ch: DiscordChannel, handler: AsyncMock,
    ) -> None:
        """Nachrichten von anderen Bots werden ignoriert."""
        discord_ch._handler = handler
        discord_ch._client = MagicMock()
        discord_ch._client.user = MagicMock()
        discord_ch._client.user.id = 88888

        msg = _make_discord_message("bot msg", author_bot=True)
        await discord_ch._on_message(msg)

        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_strips_mention(
        self, discord_ch: DiscordChannel, handler: AsyncMock,
    ) -> None:
        """Bot-Mention wird aus Text entfernt."""
        discord_ch._handler = handler
        discord_ch._client = MagicMock()
        discord_ch._client.user = MagicMock()
        discord_ch._client.user.id = 88888

        # Mention im Ziel-Channel
        bot_user = discord_ch._client.user
        msg = _make_discord_message(
            "<@88888> Was ist Python?",
            channel_id=123456789,
            mentions=[bot_user],
        )
        await discord_ch._on_message(msg)

        incoming: IncomingMessage = handler.call_args[0][0]
        assert "88888" not in incoming.text
        assert incoming.text == "Was ist Python?"

    @pytest.mark.asyncio
    async def test_ignores_messages_in_other_channels(
        self, discord_ch: DiscordChannel, handler: AsyncMock,
    ) -> None:
        """Nachrichten in anderen Channels (ohne Mention) werden ignoriert."""
        discord_ch._handler = handler
        discord_ch._client = MagicMock()
        discord_ch._client.user = MagicMock()
        discord_ch._client.user.id = 88888

        msg = _make_discord_message("random chat", channel_id=999999)
        msg.mentions = []
        await discord_ch._on_message(msg)

        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_dm_always_handled(
        self, discord_ch: DiscordChannel, handler: AsyncMock,
    ) -> None:
        """DMs (guild=None) werden immer verarbeitet."""
        discord_ch._handler = handler
        discord_ch._client = MagicMock()
        discord_ch._client.user = MagicMock()
        discord_ch._client.user.id = 88888

        msg = _make_discord_message("DM Nachricht", channel_id=777, guild_id=None)
        await discord_ch._on_message(msg)

        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_ignores_empty_messages(
        self, discord_ch: DiscordChannel, handler: AsyncMock,
    ) -> None:
        """Leere Nachrichten werden ignoriert."""
        discord_ch._handler = handler
        discord_ch._client = MagicMock()
        discord_ch._client.user = MagicMock()

        msg = _make_discord_message("", channel_id=123456789)
        await discord_ch._on_message(msg)

        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_metadata_includes_guild_and_author(
        self, discord_ch: DiscordChannel, handler: AsyncMock,
    ) -> None:
        """Metadata enthält guild_id und author_name."""
        discord_ch._handler = handler
        discord_ch._client = MagicMock()
        discord_ch._client.user = MagicMock()
        discord_ch._client.user.id = 88888

        msg = _make_discord_message("test", channel_id=123456789)
        await discord_ch._on_message(msg)

        incoming: IncomingMessage = handler.call_args[0][0]
        assert "guild_id" in incoming.metadata
        assert "author_name" in incoming.metadata
        assert "message_id" in incoming.metadata


# ============================================================================
# 3. Senden
# ============================================================================


class TestDiscordSend:
    @pytest.mark.asyncio
    async def test_send_without_client_logs_warning(self, discord_ch: DiscordChannel) -> None:
        """Ohne Client wird nicht gecrasht."""
        discord_ch._client = None
        msg = OutgoingMessage(channel="discord", text="noop", session_id="s1")
        await discord_ch.send(msg)  # Kein Crash

    @pytest.mark.asyncio
    async def test_send_without_running_logs_warning(self, discord_ch: DiscordChannel) -> None:
        """Wenn nicht running, wird nicht gesendet."""
        discord_ch._client = MagicMock()
        discord_ch._running = False
        msg = OutgoingMessage(channel="discord", text="noop", session_id="s1")
        await discord_ch.send(msg)  # Kein Crash


# ============================================================================
# 4. Approvals via Reactions
# ============================================================================


class TestDiscordApproval:
    @pytest.mark.asyncio
    async def test_approval_returns_false_without_bidirectional(
        self, discord_ch: DiscordChannel,
    ) -> None:
        """Ohne Verbindung ist Approval nicht möglich."""
        discord_ch._bidirectional = False
        action = PlannedAction(tool="delete", params={})
        result = await discord_ch.request_approval("s1", action, "Gefährlich")
        assert result is False

    @pytest.mark.asyncio
    async def test_on_reaction_approves(self, discord_ch: DiscordChannel) -> None:
        """✅ Reaction vom richtigen User löst Approval aus."""
        bot_user = MagicMock()
        discord_ch._client = MagicMock()
        discord_ch._client.user = bot_user

        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        requester_id = 99999
        discord_ch._approval_messages[555] = (future, requester_id)

        reaction = MagicMock()
        reaction.message.id = 555
        reaction.emoji = "✅"

        user = MagicMock()
        user.id = requester_id  # Richtiger User
        await discord_ch._on_reaction(reaction, user)

        assert future.done()
        assert future.result() is True

    @pytest.mark.asyncio
    async def test_on_reaction_rejects(self, discord_ch: DiscordChannel) -> None:
        """❌ Reaction lehnt ab."""
        bot_user = MagicMock()
        discord_ch._client = MagicMock()
        discord_ch._client.user = bot_user

        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        requester_id = 99999
        discord_ch._approval_messages[666] = (future, requester_id)

        reaction = MagicMock()
        reaction.message.id = 666
        reaction.emoji = "❌"

        user = MagicMock()
        user.id = requester_id
        await discord_ch._on_reaction(reaction, user)

        assert future.done()
        assert future.result() is False

    @pytest.mark.asyncio
    async def test_on_reaction_ignores_own(self, discord_ch: DiscordChannel) -> None:
        """Bot-eigene Reactions werden ignoriert."""
        bot_user = MagicMock()
        discord_ch._client = MagicMock()
        discord_ch._client.user = bot_user

        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        discord_ch._approval_messages[777] = (future, 99999)

        reaction = MagicMock()
        reaction.message.id = 777
        reaction.emoji = "✅"

        await discord_ch._on_reaction(reaction, bot_user)  # Bot = User

        assert not future.done()  # Nicht aufgelöst

    @pytest.mark.asyncio
    async def test_on_reaction_ignores_wrong_user(self, discord_ch: DiscordChannel) -> None:
        """Reactions von fremden Usern werden ignoriert."""
        bot_user = MagicMock()
        discord_ch._client = MagicMock()
        discord_ch._client.user = bot_user

        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        discord_ch._approval_messages[888] = (future, 99999)  # Requester ist 99999

        reaction = MagicMock()
        reaction.message.id = 888
        reaction.emoji = "✅"

        wrong_user = MagicMock()
        wrong_user.id = 77777  # Anderer User
        await discord_ch._on_reaction(reaction, wrong_user)

        assert not future.done()  # Nicht aufgelöst — falscher User

    @pytest.mark.asyncio
    async def test_on_reaction_ignores_unknown_emoji(self, discord_ch: DiscordChannel) -> None:
        """Andere Emojis werden ignoriert."""
        bot_user = MagicMock()
        discord_ch._client = MagicMock()
        discord_ch._client.user = bot_user

        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        discord_ch._approval_messages[888] = (future, 99999)

        reaction = MagicMock()
        reaction.message.id = 888
        reaction.emoji = "👍"

        user = MagicMock()
        user.id = 99999
        await discord_ch._on_reaction(reaction, user)

        assert not future.done()  # Nicht aufgelöst

    @pytest.mark.asyncio
    async def test_on_reaction_ignores_unknown_message(self, discord_ch: DiscordChannel) -> None:
        """Reactions auf nicht-Approval-Nachrichten werden ignoriert."""
        bot_user = MagicMock()
        discord_ch._client = MagicMock()
        discord_ch._client.user = bot_user

        reaction = MagicMock()
        reaction.message.id = 99999  # Kein Approval pending
        reaction.emoji = "✅"

        user = MagicMock()
        user.id = 12345
        await discord_ch._on_reaction(reaction, user)  # Kein Crash


# ============================================================================
# 5. Streaming
# ============================================================================


class TestDiscordStreaming:
    @pytest.mark.asyncio
    async def test_streaming_buffers_tokens(self, discord_ch: DiscordChannel) -> None:
        """Tokens werden gebuffert."""
        discord_ch._client = MagicMock()
        discord_ch._client.is_ready.return_value = True
        discord_ch._running = True

        channel_mock = MagicMock()
        channel_mock.send = AsyncMock()
        discord_ch._client.get_channel.return_value = channel_mock

        await asyncio.gather(
            discord_ch.send_streaming_token("s1", "Teil "),
            discord_ch.send_streaming_token("s1", "1"),
        )
        # Short sleep to allow the background buffer-flush task to complete
        await asyncio.sleep(0.05)

        # Mindestens ein send-Aufruf
        assert channel_mock.send.call_count >= 1


# ============================================================================
# 6. Lifecycle
# ============================================================================


class TestDiscordLifecycle:
    @pytest.mark.asyncio
    async def test_stop_resets_state(self, discord_ch: DiscordChannel) -> None:
        """stop() setzt alle State-Variablen zurück."""
        discord_ch._running = True
        discord_ch._bidirectional = True
        discord_ch._client = AsyncMock()

        await discord_ch.stop()

        assert discord_ch._running is False
        assert discord_ch._bidirectional is False
        assert discord_ch._client is None
