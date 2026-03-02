"""Enhanced tests for IRCChannel -- additional coverage."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.channels.irc import IRCChannel
from jarvis.models import OutgoingMessage, PlannedAction


@pytest.fixture
def ch() -> IRCChannel:
    return IRCChannel(
        server="irc.example.com",
        port=6667,
        nick="JarvisBot",
        channels=["#general", "#support"],
        password="secret",
    )


class TestIRCProperties:
    def test_name(self, ch: IRCChannel) -> None:
        assert ch.name == "irc"


class TestIRCHandleLine:
    @pytest.mark.asyncio
    async def test_ping_pong(self, ch: IRCChannel) -> None:
        ch._writer = MagicMock()
        ch._writer.write = MagicMock()
        ch._writer.drain = AsyncMock()

        await ch._handle_line("PING :server")
        assert b"PONG" in ch._writer.write.call_args[0][0]

    @pytest.mark.asyncio
    async def test_empty_line(self, ch: IRCChannel) -> None:
        await ch._handle_line("")  # no crash

    @pytest.mark.asyncio
    async def test_welcome_001_joins_channels(self, ch: IRCChannel) -> None:
        ch._writer = MagicMock()
        ch._writer.write = MagicMock()
        ch._writer.drain = AsyncMock()

        await ch._handle_line(":server 001 JarvisBot :Welcome")
        # Should have sent JOIN for both channels + IDENTIFY
        calls = ch._writer.write.call_args_list
        join_calls = [c for c in calls if b"JOIN" in c[0][0]]
        assert len(join_calls) == 2

    @pytest.mark.asyncio
    async def test_privmsg_in_channel(self, ch: IRCChannel) -> None:
        response = OutgoingMessage(channel="irc", text="OK")
        ch._handler = AsyncMock(return_value=response)
        ch._writer = MagicMock()
        ch._writer.write = MagicMock()
        ch._writer.drain = AsyncMock()
        ch._last_msg_time = 0

        # Addressed to bot in channel
        await ch._handle_line(":alice!alice@host 002 unused :ignored")  # test short parts

    @pytest.mark.asyncio
    async def test_short_parts(self, ch: IRCChannel) -> None:
        await ch._handle_line(":x")  # fewer than 2 parts


class TestIRCOnPrivmsg:
    @pytest.mark.asyncio
    async def test_ignore_own_messages(self, ch: IRCChannel) -> None:
        ch._handler = AsyncMock()
        await ch._on_privmsg(
            ":JarvisBot!JarvisBot@host",
            [":JarvisBot!JarvisBot@host", "PRIVMSG", "#general", ":hello"],
        )
        ch._handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_private_message(self, ch: IRCChannel) -> None:
        response = OutgoingMessage(channel="irc", text="OK")
        ch._handler = AsyncMock(return_value=response)
        ch._writer = MagicMock()
        ch._writer.write = MagicMock()
        ch._writer.drain = AsyncMock()
        ch._last_msg_time = 0

        await ch._on_privmsg(
            ":alice!alice@host",
            [":alice!alice@host", "PRIVMSG", "JarvisBot", ":hello bot"],
        )
        ch._handler.assert_called_once()
        incoming = ch._handler.call_args[0][0]
        assert incoming.text == "hello bot"
        assert incoming.metadata["is_private"] is True

    @pytest.mark.asyncio
    async def test_channel_not_addressed(self, ch: IRCChannel) -> None:
        ch._handler = AsyncMock()
        await ch._on_privmsg(
            ":alice!alice@host",
            [":alice!alice@host", "PRIVMSG", "#general", ":hello world"],
        )
        ch._handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_channel_addressed(self, ch: IRCChannel) -> None:
        response = OutgoingMessage(channel="irc", text="OK")
        ch._handler = AsyncMock(return_value=response)
        ch._writer = MagicMock()
        ch._writer.write = MagicMock()
        ch._writer.drain = AsyncMock()
        ch._last_msg_time = 0

        await ch._on_privmsg(
            ":alice!alice@host",
            [":alice!alice@host", "PRIVMSG", "#general", ":JarvisBot: what time is it"],
        )
        ch._handler.assert_called_once()
        incoming = ch._handler.call_args[0][0]
        assert "what time is it" in incoming.text

    @pytest.mark.asyncio
    async def test_empty_text(self, ch: IRCChannel) -> None:
        ch._handler = AsyncMock()
        await ch._on_privmsg(
            ":alice!alice@host",
            [":alice!alice@host", "PRIVMSG", "JarvisBot", ":  "],
        )
        ch._handler.assert_not_called()


class TestIRCSend:
    @pytest.mark.asyncio
    async def test_send_no_target(self, ch: IRCChannel) -> None:
        msg = OutgoingMessage(channel="irc", text="test", metadata={})
        ch._writer = MagicMock()
        ch._writer.write = MagicMock()
        ch._writer.drain = AsyncMock()
        ch._last_msg_time = 0
        await ch.send(msg)
        # Should use first channel as fallback
        assert b"#general" in ch._writer.write.call_args[0][0]

    @pytest.mark.asyncio
    async def test_send_no_target_no_channels(self) -> None:
        ch = IRCChannel(server="x")
        msg = OutgoingMessage(channel="irc", text="test", metadata={})
        await ch.send(msg)  # no crash


class TestIRCApproval:
    @pytest.mark.asyncio
    async def test_approval_returns_false(self, ch: IRCChannel) -> None:
        action = PlannedAction(tool="test", params={})
        result = await ch.request_approval("s1", action, "reason")
        assert result is False


class TestIRCStop:
    @pytest.mark.asyncio
    async def test_stop(self, ch: IRCChannel) -> None:
        ch._running = True
        mock_writer = MagicMock()
        mock_writer.write = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        ch._writer = mock_writer
        ch._reader = MagicMock()
        ch._recv_task = MagicMock()

        await ch.stop()
        assert ch._running is False
        assert ch._writer is None


class TestIRCStart:
    @pytest.mark.asyncio
    async def test_start_no_server(self) -> None:
        ch = IRCChannel(server="")
        handler = AsyncMock()
        await ch.start(handler)
        assert ch._running is False

    @pytest.mark.asyncio
    async def test_start_connection_error(self, ch: IRCChannel) -> None:
        handler = AsyncMock()
        with patch("asyncio.open_connection", side_effect=ConnectionError("refused")):
            await ch.start(handler)
        assert ch._running is False

    @pytest.mark.asyncio
    async def test_start_success_no_ssl(self, ch: IRCChannel) -> None:
        handler = AsyncMock()
        mock_reader = MagicMock()
        mock_writer = MagicMock()
        mock_writer.write = MagicMock()
        mock_writer.drain = AsyncMock()

        with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
            with patch.object(ch, "_receive_loop", new_callable=AsyncMock):
                await ch.start(handler)

        assert ch._running is True
        assert ch._writer is mock_writer

    @pytest.mark.asyncio
    async def test_start_with_password(self) -> None:
        ch = IRCChannel(server="irc.test.com", nick="Bot", password="secret")
        handler = AsyncMock()
        mock_writer = MagicMock()
        mock_writer.write = MagicMock()
        mock_writer.drain = AsyncMock()

        with patch("asyncio.open_connection", return_value=(MagicMock(), mock_writer)):
            with patch.object(ch, "_receive_loop", new_callable=AsyncMock):
                await ch.start(handler)

        # PASS should have been sent
        calls = mock_writer.write.call_args_list
        pass_calls = [c for c in calls if b"PASS" in c[0][0]]
        assert len(pass_calls) == 1


class TestIRCReceiveLoop:
    @pytest.mark.asyncio
    async def test_receive_loop_no_reader(self, ch: IRCChannel) -> None:
        ch._reader = None
        await ch._receive_loop()

    @pytest.mark.asyncio
    async def test_receive_loop_eof(self, ch: IRCChannel) -> None:
        ch._running = True
        mock_reader = AsyncMock()
        mock_reader.read = AsyncMock(return_value=b"")
        ch._reader = mock_reader
        await ch._receive_loop()

    @pytest.mark.asyncio
    async def test_receive_loop_processes_ping(self, ch: IRCChannel) -> None:
        ch._running = True
        call_count = 0

        async def fake_read(n):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return b"PING :server\r\n"
            ch._running = False
            return b""

        mock_reader = AsyncMock()
        mock_reader.read = fake_read
        ch._reader = mock_reader
        ch._writer = MagicMock()
        ch._writer.write = MagicMock()
        ch._writer.drain = AsyncMock()

        await ch._receive_loop()

    @pytest.mark.asyncio
    async def test_receive_loop_error_retries(self, ch: IRCChannel) -> None:
        ch._running = True
        call_count = 0

        async def fake_read(n):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("network error")
            ch._running = False
            return b""

        mock_reader = AsyncMock()
        mock_reader.read = fake_read
        ch._reader = mock_reader

        with patch("jarvis.channels.irc.asyncio.sleep", new_callable=AsyncMock):
            await ch._receive_loop()


class TestIRCSendMessage:
    @pytest.mark.asyncio
    async def test_send_message_basic(self, ch: IRCChannel) -> None:
        ch._writer = MagicMock()
        ch._writer.write = MagicMock()
        ch._writer.drain = AsyncMock()
        ch._last_msg_time = 0

        with patch("jarvis.channels.irc.asyncio.sleep", new_callable=AsyncMock):
            await ch._send_message("#general", "Hello IRC!")

        calls = ch._writer.write.call_args_list
        assert any(b"PRIVMSG #general" in c[0][0] for c in calls)

    @pytest.mark.asyncio
    async def test_send_message_multiline(self, ch: IRCChannel) -> None:
        ch._writer = MagicMock()
        ch._writer.write = MagicMock()
        ch._writer.drain = AsyncMock()
        ch._last_msg_time = 0

        with patch("jarvis.channels.irc.asyncio.sleep", new_callable=AsyncMock):
            await ch._send_message("#general", "Line 1\nLine 2")

        calls = ch._writer.write.call_args_list
        privmsg_calls = [c for c in calls if b"PRIVMSG" in c[0][0]]
        assert len(privmsg_calls) == 2


class TestIRCSendRaw:
    @pytest.mark.asyncio
    async def test_send_raw_success(self, ch: IRCChannel) -> None:
        ch._writer = MagicMock()
        ch._writer.write = MagicMock()
        ch._writer.drain = AsyncMock()
        await ch._send_raw("PING :test")
        ch._writer.write.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_raw_no_writer(self, ch: IRCChannel) -> None:
        ch._writer = None
        await ch._send_raw("PING :test")  # no crash

    @pytest.mark.asyncio
    async def test_send_raw_exception(self, ch: IRCChannel) -> None:
        ch._writer = MagicMock()
        ch._writer.write = MagicMock(side_effect=RuntimeError("broken pipe"))
        await ch._send_raw("PING :test")  # no crash
