"""Tests für Twitch Channel."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.channels.twitch import TwitchChannel
from jarvis.models import OutgoingMessage, PlannedAction


class TestTwitchChannel:
    """Tests für TwitchChannel."""

    def test_name(self) -> None:
        ch = TwitchChannel()
        assert ch.name == "twitch"

    @pytest.mark.asyncio
    async def test_start_without_config(self) -> None:
        ch = TwitchChannel()
        handler = AsyncMock()
        await ch.start(handler)
        assert not ch._running

    @pytest.mark.asyncio
    async def test_stop(self) -> None:
        ch = TwitchChannel()
        ch._running = True
        await ch.stop()
        assert not ch._running

    @pytest.mark.asyncio
    async def test_handle_ping(self) -> None:
        ch = TwitchChannel(channel="test")
        ch._writer = MagicMock()
        ch._writer.write = MagicMock()
        ch._writer.drain = AsyncMock()

        await ch._handle_line("PING :tmi.twitch.tv")
        ch._writer.write.assert_called_once()
        sent = ch._writer.write.call_args[0][0].decode("utf-8")
        assert "PONG" in sent

    @pytest.mark.asyncio
    async def test_privmsg_with_command(self) -> None:
        """Nachrichten mit !jarvis Prefix werden verarbeitet."""
        ch = TwitchChannel(nick="jarvisbot", channel="test")
        response_msg = OutgoingMessage(channel="twitch", text="Reply", session_id="s1")
        ch._handler = AsyncMock(return_value=response_msg)
        ch._writer = MagicMock()
        ch._writer.write = MagicMock()
        ch._writer.drain = AsyncMock()

        line = ":user!user@user.tmi.twitch.tv PRIVMSG #test :!jarvis what time is it?"
        await ch._on_privmsg(line, {})
        ch._handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_privmsg_without_command(self) -> None:
        """Nachrichten ohne !jarvis Prefix werden ignoriert."""
        ch = TwitchChannel(nick="jarvisbot", channel="test")
        ch._handler = AsyncMock()

        line = ":user!user@user.tmi.twitch.tv PRIVMSG #test :Hello everyone"
        await ch._on_privmsg(line, {})
        ch._handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_user_whitelist_blocks(self) -> None:
        """Nicht-erlaubte User werden blockiert."""
        ch = TwitchChannel(
            nick="jarvisbot",
            channel="test",
            allowed_users=["admin"],
        )
        ch._handler = AsyncMock()

        line = ":randomuser!user@user.tmi.twitch.tv PRIVMSG #test :!jarvis help"
        await ch._on_privmsg(line, {})
        ch._handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_user_whitelist_allows(self) -> None:
        """Erlaubte User werden durchgelassen."""
        ch = TwitchChannel(
            nick="jarvisbot",
            channel="test",
            allowed_users=["admin"],
        )
        response_msg = OutgoingMessage(channel="twitch", text="Reply", session_id="s1")
        ch._handler = AsyncMock(return_value=response_msg)
        ch._writer = MagicMock()
        ch._writer.write = MagicMock()
        ch._writer.drain = AsyncMock()

        line = ":admin!admin@admin.tmi.twitch.tv PRIVMSG #test :!jarvis status"
        await ch._on_privmsg(line, {})
        ch._handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_ignores_self(self) -> None:
        ch = TwitchChannel(nick="jarvisbot", channel="test")
        ch._handler = AsyncMock()

        line = ":jarvisbot!bot@bot.tmi.twitch.tv PRIVMSG #test :!jarvis test"
        await ch._on_privmsg(line, {})
        ch._handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_tags_parsing(self) -> None:
        ch = TwitchChannel(nick="jarvisbot", channel="test")
        tags: dict[str, str] = {}

        line = "@display-name=TestUser;mod=1;subscriber=1 :user!user@user.tmi.twitch.tv PRIVMSG #test :!jarvis help"
        # Parse tags
        tag_str, rest = line.split(" ", 1)
        for tag in tag_str[1:].split(";"):
            if "=" in tag:
                k, v = tag.split("=", 1)
                tags[k] = v

        assert tags["display-name"] == "TestUser"
        assert tags["mod"] == "1"
        assert tags["subscriber"] == "1"

    @pytest.mark.asyncio
    async def test_send_without_writer(self) -> None:
        ch = TwitchChannel()
        msg = OutgoingMessage(channel="twitch", text="Test", session_id="s1")
        await ch.send(msg)  # Should warn, not raise

    @pytest.mark.asyncio
    async def test_approval_not_supported(self) -> None:
        ch = TwitchChannel()
        action = PlannedAction(tool="test", params={})
        result = await ch.request_approval("s1", action, "reason")
        assert result is False

    @pytest.mark.asyncio
    async def test_empty_command_ignored(self) -> None:
        """Nur !jarvis ohne Text wird ignoriert."""
        ch = TwitchChannel(nick="jarvisbot", channel="test")
        ch._handler = AsyncMock()

        line = ":user!user@user.tmi.twitch.tv PRIVMSG #test :!jarvis"
        await ch._on_privmsg(line, {})
        ch._handler.assert_not_called()
