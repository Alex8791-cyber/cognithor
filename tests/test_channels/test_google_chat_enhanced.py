"""Enhanced tests for GoogleChatChannel -- additional coverage."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.channels.google_chat import GoogleChatChannel
from jarvis.models import OutgoingMessage, PlannedAction


@pytest.fixture
def ch() -> GoogleChatChannel:
    return GoogleChatChannel(credentials_path="creds.json", allowed_spaces=["spaces/abc"])


class TestGoogleChatProperties:
    def test_name(self, ch: GoogleChatChannel) -> None:
        assert ch.name == "google_chat"


class TestIsSpaceAllowed:
    def test_allowed(self, ch: GoogleChatChannel) -> None:
        assert ch._is_space_allowed("spaces/abc") is True

    def test_not_allowed(self, ch: GoogleChatChannel) -> None:
        assert ch._is_space_allowed("spaces/xyz") is False

    def test_no_whitelist(self) -> None:
        ch = GoogleChatChannel()
        assert ch._is_space_allowed("any_space") is True


class TestHandleWebhook:
    @pytest.mark.asyncio
    async def test_message_event(self, ch: GoogleChatChannel) -> None:
        response = OutgoingMessage(channel="google_chat", text="OK")
        ch._handler = AsyncMock(return_value=response)

        payload = {
            "type": "MESSAGE",
            "space": {"name": "spaces/abc"},
            "message": {
                "argumentText": "Hello",
                "text": "Hello",
                "sender": {"name": "users/123", "displayName": "Alice"},
                "name": "msg1",
                "thread": {"name": "thread1"},
            },
        }
        result = await ch.handle_webhook(payload)
        assert result == {"text": "OK"}
        ch._handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_message_not_allowed_space(self, ch: GoogleChatChannel) -> None:
        ch._handler = AsyncMock()
        payload = {
            "type": "MESSAGE",
            "space": {"name": "spaces/xyz"},
            "message": {"argumentText": "Hello", "sender": {}},
        }
        result = await ch.handle_webhook(payload)
        assert result is None
        ch._handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_card_clicked(self, ch: GoogleChatChannel) -> None:
        future = asyncio.get_event_loop().create_future()
        ch._approval_futures["appr_1"] = future

        payload = {
            "type": "CARD_CLICKED",
            "action": {
                "actionMethodName": "jarvis_approve",
                "parameters": [{"key": "approval_id", "value": "appr_1"}],
            },
        }
        await ch.handle_webhook(payload)
        assert future.result() is True

    @pytest.mark.asyncio
    async def test_card_clicked_reject(self, ch: GoogleChatChannel) -> None:
        future = asyncio.get_event_loop().create_future()
        ch._approval_futures["appr_2"] = future

        payload = {
            "type": "CARD_CLICKED",
            "action": {
                "actionMethodName": "jarvis_reject",
                "parameters": [{"key": "approval_id", "value": "appr_2"}],
            },
        }
        await ch.handle_webhook(payload)
        assert future.result() is False

    @pytest.mark.asyncio
    async def test_added_to_space(self, ch: GoogleChatChannel) -> None:
        payload = {"type": "ADDED_TO_SPACE", "space": {"name": "spaces/new"}}
        result = await ch.handle_webhook(payload)
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_text_ignored(self, ch: GoogleChatChannel) -> None:
        ch._handler = AsyncMock()
        payload = {
            "type": "MESSAGE",
            "space": {"name": "spaces/abc"},
            "message": {"argumentText": "  ", "sender": {}},
        }
        result = await ch.handle_webhook(payload)
        assert result is None


class TestGoogleChatSend:
    @pytest.mark.asyncio
    async def test_send_no_client(self, ch: GoogleChatChannel) -> None:
        ch._http_client = None
        msg = OutgoingMessage(channel="google_chat", text="test", metadata={"space_name": "s"})
        await ch.send(msg)

    @pytest.mark.asyncio
    async def test_send_no_space(self, ch: GoogleChatChannel) -> None:
        ch._http_client = AsyncMock()
        ch._credentials = MagicMock()
        msg = OutgoingMessage(channel="google_chat", text="test", metadata={})
        await ch.send(msg)

    @pytest.mark.asyncio
    async def test_send_no_auth_headers(self, ch: GoogleChatChannel) -> None:
        ch._http_client = AsyncMock()
        ch._credentials = None
        msg = OutgoingMessage(
            channel="google_chat", text="test",
            metadata={"space_name": "spaces/abc"},
        )
        await ch.send(msg)  # no crash since _get_auth_headers returns {}


class TestGoogleChatApproval:
    @pytest.mark.asyncio
    async def test_approval_no_client(self, ch: GoogleChatChannel) -> None:
        ch._http_client = None
        action = PlannedAction(tool="test", params={})
        result = await ch.request_approval("s1", action, "reason")
        assert result is False


class TestGoogleChatStop:
    @pytest.mark.asyncio
    async def test_stop(self, ch: GoogleChatChannel) -> None:
        ch._running = True
        ch._http_client = AsyncMock()
        ch._http_client.aclose = AsyncMock()
        ch._credentials = MagicMock()

        await ch.stop()
        assert ch._running is False
        assert ch._http_client is None
        assert ch._credentials is None


class TestGoogleChatStart:
    @pytest.mark.asyncio
    async def test_start_no_credentials(self) -> None:
        ch = GoogleChatChannel()
        handler = AsyncMock()
        await ch.start(handler)
        assert ch._running is False

    @pytest.mark.asyncio
    async def test_start_google_auth_not_installed(self, tmp_path) -> None:
        creds = tmp_path / "creds.json"
        creds.write_text("{}")
        ch = GoogleChatChannel(credentials_path=str(creds))
        handler = AsyncMock()
        with patch.dict("sys.modules", {
            "google": None,
            "google.oauth2": None,
            "google.oauth2.service_account": None,
        }):
            await ch.start(handler)
        assert ch._running is False

    @pytest.mark.asyncio
    async def test_start_creds_file_not_found(self) -> None:
        ch = GoogleChatChannel(credentials_path="/nonexistent/creds.json")
        handler = AsyncMock()

        mock_sa = MagicMock()
        with patch.dict("sys.modules", {
            "google": MagicMock(),
            "google.oauth2": MagicMock(),
            "google.oauth2.service_account": mock_sa,
        }):
            await ch.start(handler)
        assert ch._running is False

    @pytest.mark.asyncio
    async def test_start_success(self, tmp_path) -> None:
        creds = tmp_path / "creds.json"
        creds.write_text("{}")
        ch = GoogleChatChannel(credentials_path=str(creds))
        handler = AsyncMock()

        mock_creds = MagicMock()
        mock_sa = MagicMock()
        mock_sa.Credentials.from_service_account_file.return_value = mock_creds
        mock_oauth2 = MagicMock()
        mock_oauth2.service_account = mock_sa

        with patch.dict("sys.modules", {
            "google": MagicMock(),
            "google.oauth2": mock_oauth2,
            "google.oauth2.service_account": mock_sa,
        }):
            await ch.start(handler)

        assert ch._running is True
        assert ch._credentials is mock_creds

    @pytest.mark.asyncio
    async def test_start_auth_exception(self, tmp_path) -> None:
        creds = tmp_path / "creds.json"
        creds.write_text("{}")
        ch = GoogleChatChannel(credentials_path=str(creds))
        handler = AsyncMock()

        mock_sa = MagicMock()
        mock_sa.Credentials.from_service_account_file.side_effect = RuntimeError("bad key")
        mock_oauth2 = MagicMock()
        mock_oauth2.service_account = mock_sa

        with patch.dict("sys.modules", {
            "google": MagicMock(),
            "google.oauth2": mock_oauth2,
            "google.oauth2.service_account": mock_sa,
        }):
            await ch.start(handler)

        assert ch._running is False


class TestGetAuthHeaders:
    @pytest.mark.asyncio
    async def test_no_credentials(self, ch: GoogleChatChannel) -> None:
        ch._credentials = None
        headers = await ch._get_auth_headers()
        assert headers == {}

    @pytest.mark.asyncio
    async def test_refresh_success(self, ch: GoogleChatChannel) -> None:
        ch._credentials = MagicMock()
        ch._credentials.token = "test-token"

        mock_request = MagicMock()
        with patch.dict("sys.modules", {
            "google": MagicMock(),
            "google.auth": MagicMock(),
            "google.auth.transport": MagicMock(),
            "google.auth.transport.requests": MagicMock(Request=mock_request),
        }):
            headers = await ch._get_auth_headers()
        assert headers["Authorization"] == "Bearer test-token"

    @pytest.mark.asyncio
    async def test_refresh_failure(self, ch: GoogleChatChannel) -> None:
        ch._credentials = MagicMock()
        ch._credentials.refresh.side_effect = RuntimeError("refresh failed")

        mock_request = MagicMock()
        with patch.dict("sys.modules", {
            "google": MagicMock(),
            "google.auth": MagicMock(),
            "google.auth.transport": MagicMock(),
            "google.auth.transport.requests": MagicMock(Request=mock_request),
        }):
            headers = await ch._get_auth_headers()
        assert headers == {}


class TestGoogleChatSendSuccess:
    @pytest.mark.asyncio
    async def test_send_success(self, ch: GoogleChatChannel) -> None:
        ch._http_client = AsyncMock()
        ch._credentials = MagicMock()
        ch._credentials.token = "token123"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        ch._http_client.post = AsyncMock(return_value=mock_resp)

        mock_request = MagicMock()
        with patch.dict("sys.modules", {
            "google": MagicMock(),
            "google.auth": MagicMock(),
            "google.auth.transport": MagicMock(),
            "google.auth.transport.requests": MagicMock(Request=mock_request),
        }):
            msg = OutgoingMessage(
                channel="google_chat", text="Hello",
                metadata={"space_name": "spaces/abc"},
            )
            await ch.send(msg)
        ch._http_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_with_thread(self, ch: GoogleChatChannel) -> None:
        ch._http_client = AsyncMock()
        ch._credentials = MagicMock()
        ch._credentials.token = "token123"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        ch._http_client.post = AsyncMock(return_value=mock_resp)

        mock_request = MagicMock()
        with patch.dict("sys.modules", {
            "google": MagicMock(),
            "google.auth": MagicMock(),
            "google.auth.transport": MagicMock(),
            "google.auth.transport.requests": MagicMock(Request=mock_request),
        }):
            msg = OutgoingMessage(
                channel="google_chat", text="Reply",
                metadata={"space_name": "spaces/abc", "thread_name": "thread1"},
            )
            await ch.send(msg)
        call_kwargs = ch._http_client.post.call_args[1]
        assert "thread" in call_kwargs["json"]

    @pytest.mark.asyncio
    async def test_send_error_response(self, ch: GoogleChatChannel) -> None:
        ch._http_client = AsyncMock()
        ch._credentials = MagicMock()
        ch._credentials.token = "token123"

        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Forbidden"
        ch._http_client.post = AsyncMock(return_value=mock_resp)

        mock_request = MagicMock()
        with patch.dict("sys.modules", {
            "google": MagicMock(),
            "google.auth": MagicMock(),
            "google.auth.transport": MagicMock(),
            "google.auth.transport.requests": MagicMock(Request=mock_request),
        }):
            msg = OutgoingMessage(
                channel="google_chat", text="Hello",
                metadata={"space_name": "spaces/abc"},
            )
            await ch.send(msg)  # logs error but no crash

    @pytest.mark.asyncio
    async def test_send_exception(self, ch: GoogleChatChannel) -> None:
        ch._http_client = AsyncMock()
        ch._credentials = MagicMock()
        ch._credentials.token = "token123"
        ch._http_client.post = AsyncMock(side_effect=RuntimeError("network error"))

        mock_request = MagicMock()
        with patch.dict("sys.modules", {
            "google": MagicMock(),
            "google.auth": MagicMock(),
            "google.auth.transport": MagicMock(),
            "google.auth.transport.requests": MagicMock(Request=mock_request),
        }):
            msg = OutgoingMessage(
                channel="google_chat", text="Hello",
                metadata={"space_name": "spaces/abc"},
            )
            await ch.send(msg)  # no crash


class TestGoogleChatApprovalFlow:
    @pytest.mark.asyncio
    async def test_approval_timeout(self, ch: GoogleChatChannel) -> None:
        ch._http_client = AsyncMock()
        ch._credentials = MagicMock()

        action = PlannedAction(tool="test", params={})
        with patch("jarvis.channels.google_chat.asyncio.wait_for", side_effect=asyncio.TimeoutError):
            result = await ch.request_approval("s1", action, "reason")
        assert result is False

    @pytest.mark.asyncio
    async def test_approval_success(self, ch: GoogleChatChannel) -> None:
        ch._http_client = AsyncMock()
        ch._credentials = MagicMock()

        action = PlannedAction(tool="email", params={"to": "test@test.com"})

        async def resolve_future():
            await asyncio.sleep(0.05)
            async with ch._approval_lock:
                for aid, future in ch._approval_futures.items():
                    if not future.done():
                        future.set_result(True)
                        break

        task = asyncio.create_task(resolve_future())
        result = await ch.request_approval("s1", action, "reason")
        await task
        assert result is True
