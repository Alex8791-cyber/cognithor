"""Enhanced tests for FeishuChannel -- additional coverage.

Covers: lifecycle, event handling, message processing, card actions,
send, approval, streaming, verify_event, _parse_json_safe.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognithor.channels.feishu import FeishuChannel, _parse_json_safe
from cognithor.models import OutgoingMessage, PlannedAction


@pytest.fixture
def ch() -> FeishuChannel:
    return FeishuChannel(app_id="app123", app_secret="secret456")


class TestFeishuProperties:
    def test_name(self, ch: FeishuChannel) -> None:
        assert ch.name == "feishu"


class TestFeishuVerifyEvent:
    def test_challenge(self, ch: FeishuChannel) -> None:
        assert ch.verify_event({"challenge": "abc123"}) is True

    def test_no_encrypt_key(self, ch: FeishuChannel) -> None:
        assert ch.verify_event({"header": {}}) is True

    def test_signature_verification(self, ch: FeishuChannel) -> None:
        import hashlib

        # Build body without signature first, then compute hash with body
        # including the signature -- need a mock to sidestep circularity.
        body = {"header": {"event_time": "t", "nonce": "n", "signature": "test_sig"}}
        key = "key"
        # Compute what the source code will compute: sha256(timestamp+nonce+key+str(body))
        body_str = str(body)
        hashlib.sha256(f"tn{key}{body_str}".encode()).hexdigest()
        # Now set the signature to the expected value -- but that changes str(body)!
        # So we use a mock to control the hash output.
        mock_hash = MagicMock()
        mock_hash.hexdigest.return_value = "test_sig"
        with patch("cognithor.channels.feishu.hashlib.sha256", return_value=mock_hash):
            assert ch.verify_event(body, encrypt_key=key) is True

    def test_bad_signature(self, ch: FeishuChannel) -> None:
        body = {"header": {"event_time": "t", "nonce": "n", "signature": "wrong"}}
        assert ch.verify_event(body, encrypt_key="key") is False


class TestFeishuHandleEvent:
    @pytest.mark.asyncio
    async def test_challenge_response(self, ch: FeishuChannel) -> None:
        # Pass-3 (audit-HIGH-3) tightened challenge echo: only when the
        # payload's ``type`` field is explicitly ``url_verification``.
        # Otherwise the handler refuses, closing an auth-bypass where
        # any forged event with a ``challenge`` key would round-trip.
        result = await ch.handle_event({"type": "url_verification", "challenge": "abc123"})
        assert result == {"challenge": "abc123"}

    @pytest.mark.asyncio
    async def test_challenge_without_type_refused(self, ch: FeishuChannel) -> None:
        # The legacy "echo any challenge" path is gone.
        result = await ch.handle_event({"challenge": "abc123"})
        assert result is None

    @pytest.mark.asyncio
    async def test_message_event(self, ch: FeishuChannel) -> None:
        response = OutgoingMessage(channel="feishu", text="OK")
        ch._handler = AsyncMock(return_value=response)
        ch._http_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"code": 0}
        ch._http_client.post = AsyncMock(return_value=mock_resp)
        ch._tenant_token = "token"

        payload = {
            "header": {"event_type": "im.message.receive_v1"},
            "event": {
                "message": {
                    "message_type": "text",
                    "content": '{"text": "Hello"}',
                    "chat_id": "chat1",
                    "message_id": "msg1",
                },
                "sender": {"sender_id": {"user_id": "u1", "open_id": "o1"}},
            },
        }
        result = await ch.handle_event(payload)
        assert result is None
        ch._handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_card_action_event(self, ch: FeishuChannel) -> None:
        future = asyncio.get_event_loop().create_future()
        ch._approval_futures["appr_123"] = future

        payload = {
            "header": {"event_type": "card.action.trigger"},
            "event": {
                "action": {
                    "tag": "button",
                    "value": {"approval_id": "appr_123", "action": "approve"},
                },
            },
        }
        result = await ch.handle_event(payload)
        assert result is None
        assert future.result() is True

    @pytest.mark.asyncio
    async def test_unknown_event(self, ch: FeishuChannel) -> None:
        result = await ch.handle_event({"header": {"event_type": "unknown"}})
        assert result is None


class TestFeishuOnMessage:
    @pytest.mark.asyncio
    async def test_non_text_ignored(self, ch: FeishuChannel) -> None:
        ch._handler = AsyncMock()
        await ch._on_message(
            {
                "message": {"message_type": "image", "content": "{}"},
                "sender": {"sender_id": {}},
            }
        )
        ch._handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_text_ignored(self, ch: FeishuChannel) -> None:
        ch._handler = AsyncMock()
        await ch._on_message(
            {
                "message": {"message_type": "text", "content": '{"text": "   "}'},
                "sender": {"sender_id": {}},
            }
        )
        ch._handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_bad_json_content(self, ch: FeishuChannel) -> None:
        ch._handler = AsyncMock()
        await ch._on_message(
            {
                "message": {"message_type": "text", "content": "not json"},
                "sender": {"sender_id": {}},
            }
        )
        ch._handler.assert_not_called()  # empty after parsing


class TestFeishuOnCardAction:
    @pytest.mark.asyncio
    async def test_non_button_ignored(self, ch: FeishuChannel) -> None:
        await ch._on_card_action({"action": {"tag": "input", "value": {}}})

    @pytest.mark.asyncio
    async def test_no_approval_id(self, ch: FeishuChannel) -> None:
        await ch._on_card_action({"action": {"tag": "button", "value": {}}})

    @pytest.mark.asyncio
    async def test_reject_action(self, ch: FeishuChannel) -> None:
        future = asyncio.get_event_loop().create_future()
        ch._approval_futures["appr_456"] = future

        await ch._on_card_action(
            {
                "action": {
                    "tag": "button",
                    "value": {"approval_id": "appr_456", "action": "reject"},
                },
            }
        )
        assert future.result() is False


class TestFeishuSend:
    @pytest.mark.asyncio
    async def test_send_no_chat_id(self, ch: FeishuChannel) -> None:
        msg = OutgoingMessage(channel="feishu", text="test", metadata={})
        await ch.send(msg)  # no crash

    @pytest.mark.asyncio
    async def test_send_success(self, ch: FeishuChannel) -> None:
        import time

        ch._http_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"code": 0}
        ch._http_client.post = AsyncMock(return_value=mock_resp)
        ch._tenant_token = "token"
        ch._token_expires_at = time.time() + 3600  # skip refresh

        msg = OutgoingMessage(
            channel="feishu",
            text="hello",
            metadata={"chat_id": "chat1", "message_id": "msg1"},
        )
        await ch.send(msg)
        ch._http_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_no_client(self, ch: FeishuChannel) -> None:
        ch._http_client = None
        msg = OutgoingMessage(
            channel="feishu",
            text="hello",
            metadata={"chat_id": "chat1"},
        )
        await ch.send(msg)  # no crash


class TestFeishuApproval:
    @pytest.mark.asyncio
    async def test_approval_no_client(self, ch: FeishuChannel) -> None:
        ch._http_client = None
        action = PlannedAction(tool="test", params={})
        result = await ch.request_approval("s1", action, "reason")
        assert result is False


class TestFeishuStop:
    @pytest.mark.asyncio
    async def test_stop(self, ch: FeishuChannel) -> None:
        ch._running = True
        ch._http_client = AsyncMock()
        ch._http_client.aclose = AsyncMock()
        ch._tenant_token = "token"

        await ch.stop()
        assert ch._running is False
        assert ch._http_client is None
        assert ch._tenant_token == ""


class TestFeishuStart:
    @pytest.mark.asyncio
    async def test_start_no_credentials(self) -> None:
        ch = FeishuChannel()
        handler = AsyncMock()
        await ch.start(handler)
        assert ch._running is False


class TestFeishuRefreshToken:
    @pytest.mark.asyncio
    async def test_refresh_already_valid(self, ch: FeishuChannel) -> None:
        import time

        ch._token_expires_at = time.time() + 3600
        ch._http_client = AsyncMock()
        await ch._refresh_tenant_token()
        ch._http_client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_refresh_success(self, ch: FeishuChannel) -> None:
        ch._http_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "code": 0,
            "tenant_access_token": "new_token",
            "expire": 7200,
        }
        ch._http_client.post = AsyncMock(return_value=mock_resp)
        ch._token_expires_at = 0

        await ch._refresh_tenant_token()
        assert ch._tenant_token == "new_token"

    @pytest.mark.asyncio
    async def test_refresh_failure(self, ch: FeishuChannel) -> None:
        ch._http_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"code": 99}
        ch._http_client.post = AsyncMock(return_value=mock_resp)
        ch._token_expires_at = 0

        await ch._refresh_tenant_token()  # logs error, no crash


class TestParseJsonSafe:
    def test_valid_json(self) -> None:
        assert _parse_json_safe('{"text": "hello"}') == {"text": "hello"}

    def test_invalid_json(self) -> None:
        assert _parse_json_safe("not json") == {}
