"""Tests für Feishu/Lark Channel."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.channels.feishu import FeishuChannel
from jarvis.models import OutgoingMessage, PlannedAction


class TestFeishuChannel:
    """Tests für FeishuChannel."""

    def test_name(self) -> None:
        ch = FeishuChannel()
        assert ch.name == "feishu"

    @pytest.mark.asyncio
    async def test_start_without_config(self) -> None:
        ch = FeishuChannel()
        handler = AsyncMock()
        await ch.start(handler)
        assert not ch._running

    @pytest.mark.asyncio
    async def test_stop(self) -> None:
        ch = FeishuChannel()
        ch._running = True
        await ch.stop()
        assert not ch._running
        assert ch._tenant_token == ""

    @pytest.mark.asyncio
    async def test_handle_event_challenge(self) -> None:
        ch = FeishuChannel()
        payload = {"challenge": "test_challenge_token"}
        result = await ch.handle_event(payload)
        assert result == {"challenge": "test_challenge_token"}

    @pytest.mark.asyncio
    async def test_handle_event_message(self) -> None:
        ch = FeishuChannel()
        response_msg = OutgoingMessage(channel="feishu", text="Reply", session_id="s1")
        ch._handler = AsyncMock(return_value=response_msg)
        ch._send_text = AsyncMock()

        payload = {
            "header": {"event_type": "im.message.receive_v1"},
            "event": {
                "message": {
                    "message_type": "text",
                    "content": '{"text": "Hello Jarvis"}',
                    "chat_id": "oc_test",
                    "message_id": "om_test",
                },
                "sender": {
                    "sender_id": {
                        "user_id": "user1",
                        "open_id": "ou_test",
                    },
                },
            },
        }
        await ch.handle_event(payload)
        ch._handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_event_non_text_message(self) -> None:
        ch = FeishuChannel()
        ch._handler = AsyncMock()

        payload = {
            "header": {"event_type": "im.message.receive_v1"},
            "event": {
                "message": {
                    "message_type": "image",
                    "content": "{}",
                    "chat_id": "oc_test",
                    "message_id": "om_test",
                },
                "sender": {"sender_id": {"user_id": "u1"}},
            },
        }
        await ch.handle_event(payload)
        ch._handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_card_action_approve(self) -> None:
        ch = FeishuChannel()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        approval_id = "test_approval"
        ch._approval_futures[approval_id] = future

        event = {
            "action": {
                "tag": "button",
                "value": {"approval_id": approval_id, "action": "approve"},
            },
        }
        await ch._on_card_action(event)
        assert future.result() is True

    @pytest.mark.asyncio
    async def test_handle_card_action_reject(self) -> None:
        ch = FeishuChannel()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        approval_id = "test_reject"
        ch._approval_futures[approval_id] = future

        event = {
            "action": {
                "tag": "button",
                "value": {"approval_id": approval_id, "action": "reject"},
            },
        }
        await ch._on_card_action(event)
        assert future.result() is False

    @pytest.mark.asyncio
    async def test_send_without_chat_id(self) -> None:
        ch = FeishuChannel()
        msg = OutgoingMessage(channel="feishu", text="Test", session_id="s1")
        await ch.send(msg)  # Should warn, not raise

    def test_verify_event_challenge(self) -> None:
        ch = FeishuChannel()
        assert ch.verify_event({"challenge": "token"}) is True

    def test_verify_event_no_key(self) -> None:
        ch = FeishuChannel()
        assert ch.verify_event({"header": {}}) is True
