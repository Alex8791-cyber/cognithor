"""Enhanced tests for TwitchChannel -- additional coverage."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.channels.twitch import TwitchChannel
from jarvis.models import OutgoingMessage, PlannedAction


@pytest.fixture
def ch() -> TwitchChannel:
    return TwitchChannel(
        token="oauth:test",
        channel="testchannel",
        nick="JarvisBot",
        allowed_users=["alice", "bob"],
        command_prefix="!jarvis",
    )


class TestTwitchProperties:
    def test_name(self, ch: TwitchChannel) -> None:
        assert ch.name == "twitch"

    def test_channel_lowercase(self, ch: TwitchChannel) -> None:
        assert ch._channel == "testchannel"

    def test_nick_lowercase(self, ch: TwitchChannel) -> None:
        assert ch._nick == "jarvisbot"


class TestTwitchHandleLine:
    @pytest.mark.asyncio
    async def test_ping_pong(self, ch: TwitchChannel) -> None:
        ch._writer = MagicMock()
        ch._writer.write = MagicMock()
        ch._writer.drain = AsyncMock()

        await ch._handle_line("PING :tmi.twitch.tv")
        ch._writer.write.assert_called_once()
        assert b"PONG" in ch._writer.write.call_args[0][0]

    @pytest.mark.asyncio
    async def test_empty_line(self, ch: TwitchChannel) -> None:
        await ch._handle_line("")  # no crash

    @pytest.mark.asyncio
    async def test_privmsg_line(self, ch: TwitchChannel) -> None:
        response = OutgoingMessage(channel="twitch", text="OK")
        ch._handler = AsyncMock(return_value=response)
        ch._writer = MagicMock()
        ch._writer.write = MagicMock()
        ch._writer.drain = AsyncMock()
        ch._last_msg_time = 0

        line = "@display-name=Alice;mod=0;subscriber=0 :alice!alice@alice.tmi.twitch.tv PRIVMSG #testchannel :!jarvis what is up"
        await ch._handle_line(line)
        ch._handler.assert_called_once()
        incoming = ch._handler.call_args[0][0]
        assert incoming.text == "what is up"

    @pytest.mark.asyncio
    async def test_privmsg_no_tags(self, ch: TwitchChannel) -> None:
        response = OutgoingMessage(channel="twitch", text="OK")
        ch._handler = AsyncMock(return_value=response)
        ch._writer = MagicMock()
        ch._writer.write = MagicMock()
        ch._writer.drain = AsyncMock()
        ch._last_msg_time = 0

        line = ":alice!alice@alice.tmi.twitch.tv PRIVMSG #testchannel :!jarvis hi"
        await ch._handle_line(line)
        ch._handler.assert_called_once()


class TestTwitchOnPrivmsg:
    @pytest.mark.asyncio
    async def test_ignore_own_messages(self, ch: TwitchChannel) -> None:
        ch._handler = AsyncMock()
        line = ":jarvisbot!jarvisbot@jarvisbot.tmi.twitch.tv PRIVMSG #testchannel :!jarvis test"
        await ch._on_privmsg(line, {})
        ch._handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_whitelist_blocks(self, ch: TwitchChannel) -> None:
        ch._handler = AsyncMock()
        line = ":charlie!charlie@charlie.tmi.twitch.tv PRIVMSG #testchannel :!jarvis test"
        await ch._on_privmsg(line, {})
        ch._handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_prefix_ignored(self, ch: TwitchChannel) -> None:
        ch._handler = AsyncMock()
        line = ":alice!alice@alice.tmi.twitch.tv PRIVMSG #testchannel :hello world"
        await ch._on_privmsg(line, {})
        ch._handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_prefix_only_ignored(self, ch: TwitchChannel) -> None:
        ch._handler = AsyncMock()
        line = ":alice!alice@alice.tmi.twitch.tv PRIVMSG #testchannel :!jarvis"
        await ch._on_privmsg(line, {})
        ch._handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_metadata_tags(self, ch: TwitchChannel) -> None:
        response = OutgoingMessage(channel="twitch", text="OK")
        ch._handler = AsyncMock(return_value=response)
        ch._writer = MagicMock()
        ch._writer.write = MagicMock()
        ch._writer.drain = AsyncMock()
        ch._last_msg_time = 0

        tags = {
            "display-name": "Alice",
            "mod": "1",
            "subscriber": "1",
            "badges": "broadcaster/1",
        }
        line = ":alice!alice@alice.tmi.twitch.tv PRIVMSG #testchannel :!jarvis check"
        await ch._on_privmsg(line, tags)
        incoming = ch._handler.call_args[0][0]
        assert incoming.metadata["is_mod"] is True
        assert incoming.metadata["is_sub"] is True
        assert incoming.metadata["is_broadcaster"] is True


class TestTwitchSend:
    @pytest.mark.asyncio
    async def test_send_no_writer(self, ch: TwitchChannel) -> None:
        ch._writer = None
        msg = OutgoingMessage(channel="twitch", text="test")
        await ch.send(msg)  # no crash


class TestTwitchApproval:
    @pytest.mark.asyncio
    async def test_approval_returns_false(self, ch: TwitchChannel) -> None:
        action = PlannedAction(tool="test", params={})
        result = await ch.request_approval("s1", action, "reason")
        assert result is False


class TestTwitchStop:
    @pytest.mark.asyncio
    async def test_stop(self, ch: TwitchChannel) -> None:
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
        assert ch._reader is None
        assert ch._recv_task is None


class TestTwitchSendRaw:
    @pytest.mark.asyncio
    async def test_send_raw_no_writer(self, ch: TwitchChannel) -> None:
        ch._writer = None
        await ch._send_raw("test")  # no crash

    @pytest.mark.asyncio
    async def test_send_raw_exception(self, ch: TwitchChannel) -> None:
        ch._writer = MagicMock()
        ch._writer.write = MagicMock(side_effect=RuntimeError("broken"))
        await ch._send_raw("test")  # no crash

    @pytest.mark.asyncio
    async def test_send_raw_success(self, ch: TwitchChannel) -> None:
        ch._writer = MagicMock()
        ch._writer.write = MagicMock()
        ch._writer.drain = AsyncMock()
        await ch._send_raw("PASS oauth:test")
        ch._writer.write.assert_called_once()
        assert b"PASS oauth:test\r\n" == ch._writer.write.call_args[0][0]


class TestTwitchStart:
    @pytest.mark.asyncio
    async def test_start_no_token(self) -> None:
        ch = TwitchChannel(token="", channel="test")
        handler = AsyncMock()
        await ch.start(handler)
        assert ch._running is False

    @pytest.mark.asyncio
    async def test_start_no_channel(self) -> None:
        ch = TwitchChannel(token="oauth:test", channel="")
        handler = AsyncMock()
        await ch.start(handler)
        assert ch._running is False

    @pytest.mark.asyncio
    async def test_start_connection_error(self, ch: TwitchChannel) -> None:
        handler = AsyncMock()
        with patch("asyncio.open_connection", side_effect=ConnectionError("refused")):
            await ch.start(handler)
        assert ch._running is False

    @pytest.mark.asyncio
    async def test_start_success(self, ch: TwitchChannel) -> None:
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


class TestTwitchReceiveLoop:
    @pytest.mark.asyncio
    async def test_receive_loop_no_reader(self, ch: TwitchChannel) -> None:
        ch._reader = None
        await ch._receive_loop()

    @pytest.mark.asyncio
    async def test_receive_loop_eof(self, ch: TwitchChannel) -> None:
        ch._running = True
        mock_reader = AsyncMock()
        mock_reader.read = AsyncMock(return_value=b"")
        ch._reader = mock_reader

        await ch._receive_loop()

    @pytest.mark.asyncio
    async def test_receive_loop_processes_lines(self, ch: TwitchChannel) -> None:
        ch._running = True
        call_count = 0

        async def fake_read(n):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return b"PING :tmi.twitch.tv\r\n"
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
    async def test_receive_loop_error_retries(self, ch: TwitchChannel) -> None:
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

        with patch("jarvis.channels.twitch.asyncio.sleep", new_callable=AsyncMock):
            await ch._receive_loop()


class TestTwitchSendChat:
    @pytest.mark.asyncio
    async def test_send_chat_basic(self, ch: TwitchChannel) -> None:
        ch._writer = MagicMock()
        ch._writer.write = MagicMock()
        ch._writer.drain = AsyncMock()
        ch._last_msg_time = 0

        with patch("jarvis.channels.twitch.asyncio.sleep", new_callable=AsyncMock):
            await ch._send_chat("Hello viewers!")

        # Should have sent PRIVMSG
        calls = ch._writer.write.call_args_list
        assert any(b"PRIVMSG" in c[0][0] for c in calls)

    @pytest.mark.asyncio
    async def test_send_chat_multiline(self, ch: TwitchChannel) -> None:
        ch._writer = MagicMock()
        ch._writer.write = MagicMock()
        ch._writer.drain = AsyncMock()
        ch._last_msg_time = 0

        with patch("jarvis.channels.twitch.asyncio.sleep", new_callable=AsyncMock):
            await ch._send_chat("Line 1\nLine 2")

        # Should send multiple PRIVMSGs
        calls = ch._writer.write.call_args_list
        privmsg_calls = [c for c in calls if b"PRIVMSG" in c[0][0]]
        assert len(privmsg_calls) == 2

    @pytest.mark.asyncio
    async def test_send_with_writer(self, ch: TwitchChannel) -> None:
        ch._writer = MagicMock()
        ch._writer.write = MagicMock()
        ch._writer.drain = AsyncMock()
        ch._last_msg_time = 0

        msg = OutgoingMessage(channel="twitch", text="test msg")
        with patch("jarvis.channels.twitch.asyncio.sleep", new_callable=AsyncMock):
            await ch.send(msg)

        calls = ch._writer.write.call_args_list
        assert any(b"PRIVMSG" in c[0][0] for c in calls)


class TestTwitchStreaming:
    @pytest.mark.asyncio
    async def test_streaming_token(self, ch: TwitchChannel) -> None:
        ch._writer = MagicMock()
        ch._writer.write = MagicMock()
        ch._writer.drain = AsyncMock()
        ch._last_msg_time = 0

        with patch("jarvis.channels.twitch.asyncio.sleep", new_callable=AsyncMock):
            await ch.send_streaming_token("s1", "hello")
