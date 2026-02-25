"""Tests für Mattermost Channel."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.channels.mattermost import MattermostChannel
from jarvis.models import IncomingMessage, OutgoingMessage, PlannedAction


class TestMattermostChannel:
    """Tests für MattermostChannel."""

    def test_name(self) -> None:
        ch = MattermostChannel()
        assert ch.name == "mattermost"

    def test_api_url(self) -> None:
        ch = MattermostChannel(url="https://mm.example.com")
        assert ch.api_url == "https://mm.example.com/api/v4"

    def test_api_url_trailing_slash(self) -> None:
        ch = MattermostChannel(url="https://mm.example.com/")
        assert ch.api_url == "https://mm.example.com/api/v4"

    @pytest.mark.asyncio
    async def test_start_without_config(self) -> None:
        ch = MattermostChannel()
        handler = AsyncMock()
        await ch.start(handler)
        assert not ch._running

    @pytest.mark.asyncio
    async def test_stop(self) -> None:
        ch = MattermostChannel()
        ch._running = True
        await ch.stop()
        assert not ch._running

    @pytest.mark.asyncio
    async def test_on_message_ignores_bot(self) -> None:
        ch = MattermostChannel()
        ch._bot_user_id = "bot123"
        ch._handler = AsyncMock()

        post = {"user_id": "bot123", "message": "test", "channel_id": "ch1"}
        await ch._on_message(post)
        ch._handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_message_processes_user(self) -> None:
        ch = MattermostChannel()
        ch._bot_user_id = "bot123"
        ch._http_client = MagicMock()

        response_msg = OutgoingMessage(channel="mattermost", text="Reply", session_id="s1")
        ch._handler = AsyncMock(return_value=response_msg)

        # Mock _create_post
        ch._create_post = AsyncMock(return_value="post_id")

        post = {
            "user_id": "user456",
            "message": "Hello Jarvis",
            "channel_id": "ch1",
            "id": "msg1",
            "root_id": "",
        }
        await ch._on_message(post)
        ch._handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_message_ignores_empty(self) -> None:
        ch = MattermostChannel()
        ch._handler = AsyncMock()

        post = {"user_id": "user1", "message": "", "channel_id": "ch1"}
        await ch._on_message(post)
        ch._handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_reaction_approve(self) -> None:
        ch = MattermostChannel()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        ch._approval_futures["post_123"] = future

        reaction = {
            "emoji_name": "white_check_mark",
            "post_id": "post_123",
        }
        await ch._on_reaction(reaction)
        assert future.result() is True

    @pytest.mark.asyncio
    async def test_on_reaction_reject(self) -> None:
        ch = MattermostChannel()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        ch._approval_futures["post_456"] = future

        reaction = {
            "emoji_name": "x",
            "post_id": "post_456",
        }
        await ch._on_reaction(reaction)
        assert future.result() is False

    @pytest.mark.asyncio
    async def test_on_reaction_ignores_unknown_emoji(self) -> None:
        ch = MattermostChannel()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        ch._approval_futures["post_789"] = future

        reaction = {
            "emoji_name": "smile",
            "post_id": "post_789",
        }
        await ch._on_reaction(reaction)
        assert not future.done()

    @pytest.mark.asyncio
    async def test_send_without_channel(self) -> None:
        ch = MattermostChannel()
        msg = OutgoingMessage(channel="mattermost", text="Test", session_id="s1")
        await ch.send(msg)  # Should warn, not raise

    @pytest.mark.asyncio
    async def test_request_approval_without_client(self) -> None:
        ch = MattermostChannel()
        action = PlannedAction(tool="test_tool", params={})
        result = await ch.request_approval("s1", action, "test reason")
        assert result is False
