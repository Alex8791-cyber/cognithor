"""Tests für Google Chat Channel."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.channels.google_chat import GoogleChatChannel
from jarvis.models import IncomingMessage, OutgoingMessage, PlannedAction


class TestGoogleChatChannel:
    """Tests für GoogleChatChannel."""

    def test_name(self) -> None:
        ch = GoogleChatChannel()
        assert ch.name == "google_chat"

    @pytest.mark.asyncio
    async def test_start_without_credentials(self) -> None:
        ch = GoogleChatChannel()
        handler = AsyncMock()
        await ch.start(handler)
        assert not ch._running

    @pytest.mark.asyncio
    async def test_stop(self) -> None:
        ch = GoogleChatChannel()
        ch._running = True
        await ch.stop()
        assert not ch._running

    @pytest.mark.asyncio
    async def test_space_whitelist_empty_allows_all(self) -> None:
        ch = GoogleChatChannel(allowed_spaces=[])
        assert ch._is_space_allowed("spaces/any")

    @pytest.mark.asyncio
    async def test_space_whitelist_rejects(self) -> None:
        ch = GoogleChatChannel(allowed_spaces=["spaces/allowed"])
        assert not ch._is_space_allowed("spaces/denied")
        assert ch._is_space_allowed("spaces/allowed")

    @pytest.mark.asyncio
    async def test_handle_webhook_message(self) -> None:
        ch = GoogleChatChannel()
        response_msg = OutgoingMessage(channel="google_chat", text="Reply", session_id="s1")
        ch._handler = AsyncMock(return_value=response_msg)

        payload = {
            "type": "MESSAGE",
            "space": {"name": "spaces/test"},
            "message": {
                "text": "Hello Jarvis",
                "argumentText": "Hello Jarvis",
                "name": "spaces/test/messages/123",
                "thread": {"name": "spaces/test/threads/456"},
                "sender": {
                    "name": "users/user1",
                    "displayName": "Test User",
                },
            },
        }

        result = await ch.handle_webhook(payload)
        assert result is not None
        assert result["text"] == "Reply"
        ch._handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_webhook_empty_text(self) -> None:
        ch = GoogleChatChannel()
        ch._handler = AsyncMock()

        payload = {
            "type": "MESSAGE",
            "space": {"name": "spaces/test"},
            "message": {
                "text": "",
                "sender": {"name": "users/user1"},
            },
        }

        result = await ch.handle_webhook(payload)
        assert result is None
        ch._handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_webhook_added_to_space(self) -> None:
        ch = GoogleChatChannel()
        payload = {
            "type": "ADDED_TO_SPACE",
            "space": {"name": "spaces/new"},
        }
        result = await ch.handle_webhook(payload)
        assert result is None

    @pytest.mark.asyncio
    async def test_send_without_client(self) -> None:
        ch = GoogleChatChannel()
        msg = OutgoingMessage(channel="google_chat", text="Test", session_id="s1")
        await ch.send(msg)  # Should not raise

    @pytest.mark.asyncio
    async def test_send_without_space(self) -> None:
        ch = GoogleChatChannel()
        ch._http_client = MagicMock()
        ch._credentials = MagicMock()
        msg = OutgoingMessage(
            channel="google_chat", text="Test", session_id="s1",
            metadata={},
        )
        await ch.send(msg)  # Should warn, not raise

    @pytest.mark.asyncio
    async def test_card_click_approval(self) -> None:
        ch = GoogleChatChannel()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        approval_id = "test_approval"
        ch._approval_futures[approval_id] = future

        payload = {
            "action": {
                "actionMethodName": "jarvis_approve",
                "parameters": [{"key": "approval_id", "value": approval_id}],
            },
        }
        await ch._handle_card_click(payload)
        assert future.result() is True

    @pytest.mark.asyncio
    async def test_card_click_rejection(self) -> None:
        ch = GoogleChatChannel()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        approval_id = "test_reject"
        ch._approval_futures[approval_id] = future

        payload = {
            "action": {
                "actionMethodName": "jarvis_reject",
                "parameters": [{"key": "approval_id", "value": approval_id}],
            },
        }
        await ch._handle_card_click(payload)
        assert future.result() is False

    @pytest.mark.asyncio
    async def test_streaming_buffer(self) -> None:
        ch = GoogleChatChannel()
        # Don't set client/credentials — send() will just warn and return
        # The buffer gets populated but then flushed after sleep
        # We verify no error is raised
        await ch.send_streaming_token("s1", "Hello")
        # After await, buffer was already flushed (sleep 0.5s completed)
        # Just verify it didn't crash
