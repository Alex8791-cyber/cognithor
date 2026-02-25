"""Tests für IRC Channel."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.channels.irc import IRCChannel
from jarvis.models import OutgoingMessage, PlannedAction


class TestIRCChannel:
    """Tests für IRCChannel."""

    def test_name(self) -> None:
        ch = IRCChannel()
        assert ch.name == "irc"

    @pytest.mark.asyncio
    async def test_start_without_server(self) -> None:
        ch = IRCChannel()
        handler = AsyncMock()
        await ch.start(handler)
        assert not ch._running

    @pytest.mark.asyncio
    async def test_stop(self) -> None:
        ch = IRCChannel()
        ch._running = True
        await ch.stop()
        assert not ch._running

    @pytest.mark.asyncio
    async def test_handle_ping(self) -> None:
        ch = IRCChannel()
        ch._writer = MagicMock()
        ch._writer.write = MagicMock()
        ch._writer.drain = AsyncMock()

        await ch._handle_line("PING :server.example.com")
        ch._writer.write.assert_called_once()
        sent = ch._writer.write.call_args[0][0].decode("utf-8")
        assert "PONG" in sent

    @pytest.mark.asyncio
    async def test_handle_welcome_joins_channels(self) -> None:
        ch = IRCChannel(channels=["#test", "#dev"])
        ch._writer = MagicMock()
        ch._writer.write = MagicMock()
        ch._writer.drain = AsyncMock()

        # Simulate 001 (welcome)
        await ch._handle_line(":server 001 JarvisBot :Welcome")
        # Should have sent JOIN for each channel
        calls = ch._writer.write.call_args_list
        sent_data = "".join(c[0][0].decode("utf-8") for c in calls)
        assert "JOIN #test" in sent_data
        assert "JOIN #dev" in sent_data

    @pytest.mark.asyncio
    async def test_privmsg_channel_requires_nick(self) -> None:
        """Nachrichten in Channels werden nur verarbeitet wenn Bot angesprochen wird."""
        ch = IRCChannel(nick="JarvisBot")
        ch._handler = AsyncMock()
        ch._writer = MagicMock()
        ch._writer.write = MagicMock()
        ch._writer.drain = AsyncMock()

        # Nachricht ohne Nick-Mention → ignoriert
        await ch._on_privmsg(
            ":user!u@host", [":user!u@host", "PRIVMSG", "#test", ":Hello world"]
        )
        ch._handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_privmsg_with_nick_mention(self) -> None:
        """Nachrichten mit Nick-Mention werden verarbeitet."""
        ch = IRCChannel(nick="JarvisBot")
        response_msg = OutgoingMessage(channel="irc", text="Reply", session_id="s1")
        ch._handler = AsyncMock(return_value=response_msg)
        ch._writer = MagicMock()
        ch._writer.write = MagicMock()
        ch._writer.drain = AsyncMock()

        await ch._on_privmsg(
            ":user!u@host",
            [":user!u@host", "PRIVMSG", "#test", ":JarvisBot: what is the time?"],
        )
        ch._handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_privmsg_private_message(self) -> None:
        """Private Nachrichten werden immer verarbeitet."""
        ch = IRCChannel(nick="JarvisBot")
        response_msg = OutgoingMessage(channel="irc", text="Reply", session_id="s1")
        ch._handler = AsyncMock(return_value=response_msg)
        ch._writer = MagicMock()
        ch._writer.write = MagicMock()
        ch._writer.drain = AsyncMock()

        await ch._on_privmsg(
            ":user!u@host",
            [":user!u@host", "PRIVMSG", "JarvisBot", ":Hello"],
        )
        ch._handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_privmsg_ignores_self(self) -> None:
        ch = IRCChannel(nick="JarvisBot")
        ch._handler = AsyncMock()

        await ch._on_privmsg(
            ":JarvisBot!u@host",
            [":JarvisBot!u@host", "PRIVMSG", "#test", ":JarvisBot: test"],
        )
        ch._handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_without_target(self) -> None:
        ch = IRCChannel()
        msg = OutgoingMessage(channel="irc", text="Test", session_id="s1")
        await ch.send(msg)  # Should warn, not raise

    @pytest.mark.asyncio
    async def test_send_uses_first_channel(self) -> None:
        ch = IRCChannel(channels=["#main"])
        ch._writer = MagicMock()
        ch._writer.write = MagicMock()
        ch._writer.drain = AsyncMock()

        msg = OutgoingMessage(channel="irc", text="Test", session_id="s1")
        await ch.send(msg)
        sent = ch._writer.write.call_args[0][0].decode("utf-8")
        assert "PRIVMSG #main" in sent

    @pytest.mark.asyncio
    async def test_approval_not_supported(self) -> None:
        ch = IRCChannel()
        action = PlannedAction(tool="test", params={})
        result = await ch.request_approval("s1", action, "reason")
        assert result is False
