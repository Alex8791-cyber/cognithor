"""Enhanced tests for MattermostChannel -- additional coverage.

Covers: lifecycle, _on_message, _on_reaction, _handle_ws_event,
_create_post, send, request_approval, streaming.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.channels.mattermost import MattermostChannel
from jarvis.models import OutgoingMessage, PlannedAction


@pytest.fixture
def ch() -> MattermostChannel:
    return MattermostChannel(url="https://mm.example.com", token="test-token", default_channel="ch1")


class TestMattermostProperties:
    def test_name(self, ch: MattermostChannel) -> None:
        assert ch.name == "mattermost"

    def test_api_url(self, ch: MattermostChannel) -> None:
        assert ch.api_url == "https://mm.example.com/api/v4"

    def test_headers(self, ch: MattermostChannel) -> None:
        h = ch._headers()
        assert "Bearer" in h["Authorization"]

    def test_token_no_token(self) -> None:
        ch = MattermostChannel(url="http://x")
        assert ch._token == ""


class TestMattermostOnMessage:
    @pytest.mark.asyncio
    async def test_ignore_own_messages(self, ch: MattermostChannel) -> None:
        ch._bot_user_id = "bot1"
        ch._handler = AsyncMock()
        await ch._on_message({"user_id": "bot1", "message": "hello"})
        ch._handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_ignore_empty(self, ch: MattermostChannel) -> None:
        ch._handler = AsyncMock()
        await ch._on_message({"user_id": "u1", "message": "   "})
        ch._handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_message(self, ch: MattermostChannel) -> None:
        response = OutgoingMessage(channel="mattermost", text="OK")
        ch._handler = AsyncMock(return_value=response)
        ch._http_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {"id": "post1"}
        ch._http_client.post = AsyncMock(return_value=mock_resp)

        await ch._on_message({
            "user_id": "u1",
            "message": "Hello",
            "channel_id": "ch1",
            "id": "p1",
            "root_id": "",
        })
        ch._handler.assert_called_once()


class TestMattermostOnReaction:
    @pytest.mark.asyncio
    async def test_approve_reaction(self, ch: MattermostChannel) -> None:
        future = asyncio.get_event_loop().create_future()
        ch._approval_futures["post1"] = future

        await ch._on_reaction({"emoji_name": "white_check_mark", "post_id": "post1"})
        assert future.result() is True

    @pytest.mark.asyncio
    async def test_deny_reaction(self, ch: MattermostChannel) -> None:
        future = asyncio.get_event_loop().create_future()
        ch._approval_futures["post1"] = future

        await ch._on_reaction({"emoji_name": "x", "post_id": "post1"})
        assert future.result() is False

    @pytest.mark.asyncio
    async def test_irrelevant_reaction(self, ch: MattermostChannel) -> None:
        await ch._on_reaction({"emoji_name": "smile", "post_id": "post1"})  # no crash


class TestMattermostHandleWsEvent:
    @pytest.mark.asyncio
    async def test_posted_event(self, ch: MattermostChannel) -> None:
        ch._handler = AsyncMock(
            return_value=OutgoingMessage(channel="mattermost", text="OK")
        )
        ch._http_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {"id": "p1"}
        ch._http_client.post = AsyncMock(return_value=mock_resp)

        post_data = json.dumps({
            "user_id": "u1",
            "message": "test",
            "channel_id": "ch1",
            "id": "p1",
        })
        await ch._handle_ws_event({"event": "posted", "data": {"post": post_data}})
        ch._handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_reaction_added_event(self, ch: MattermostChannel) -> None:
        future = asyncio.get_event_loop().create_future()
        ch._approval_futures["p1"] = future

        reaction = json.dumps({"emoji_name": "white_check_mark", "post_id": "p1"})
        await ch._handle_ws_event({"event": "reaction_added", "data": {"reaction": reaction}})
        assert future.result() is True

    @pytest.mark.asyncio
    async def test_unknown_event(self, ch: MattermostChannel) -> None:
        await ch._handle_ws_event({"event": "unknown"})  # no crash


class TestMattermostCreatePost:
    @pytest.mark.asyncio
    async def test_create_post_success(self, ch: MattermostChannel) -> None:
        ch._http_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {"id": "p123"}
        ch._http_client.post = AsyncMock(return_value=mock_resp)

        result = await ch._create_post("ch1", "hello", "root1")
        assert result == "p123"

    @pytest.mark.asyncio
    async def test_create_post_failure(self, ch: MattermostChannel) -> None:
        ch._http_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        ch._http_client.post = AsyncMock(return_value=mock_resp)

        result = await ch._create_post("ch1", "hello")
        assert result == ""

    @pytest.mark.asyncio
    async def test_create_post_no_client(self, ch: MattermostChannel) -> None:
        ch._http_client = None
        result = await ch._create_post("ch1", "hello")
        assert result == ""

    @pytest.mark.asyncio
    async def test_create_post_exception(self, ch: MattermostChannel) -> None:
        ch._http_client = AsyncMock()
        ch._http_client.post = AsyncMock(side_effect=RuntimeError("err"))
        result = await ch._create_post("ch1", "hello")
        assert result == ""


class TestMattermostSend:
    @pytest.mark.asyncio
    async def test_send_no_channel(self, ch: MattermostChannel) -> None:
        ch._default_channel = ""
        msg = OutgoingMessage(channel="mattermost", text="test", metadata={})
        await ch.send(msg)  # no crash

    @pytest.mark.asyncio
    async def test_send_with_metadata(self, ch: MattermostChannel) -> None:
        ch._http_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {"id": "p1"}
        ch._http_client.post = AsyncMock(return_value=mock_resp)

        msg = OutgoingMessage(
            channel="mattermost", text="hello",
            metadata={"channel_id": "ch2", "root_id": "r1"},
        )
        await ch.send(msg)


class TestMattermostApproval:
    @pytest.mark.asyncio
    async def test_approval_no_channel(self, ch: MattermostChannel) -> None:
        ch._default_channel = ""
        action = PlannedAction(tool="test", params={})
        result = await ch.request_approval("s1", action, "reason")
        assert result is False

    @pytest.mark.asyncio
    async def test_approval_post_failed(self, ch: MattermostChannel) -> None:
        ch._http_client = AsyncMock()
        ch._http_client.post = AsyncMock(return_value=MagicMock(status_code=400))

        action = PlannedAction(tool="test", params={})
        result = await ch.request_approval("s1", action, "reason")
        assert result is False


class TestMattermostStop:
    @pytest.mark.asyncio
    async def test_stop(self, ch: MattermostChannel) -> None:
        ch._running = True
        ch._ws_task = MagicMock()
        ch._http_client = AsyncMock()
        ch._http_client.aclose = AsyncMock()

        await ch.stop()
        assert ch._running is False
        assert ch._ws_task is None
        assert ch._http_client is None


class TestMattermostStreaming:
    @pytest.mark.asyncio
    async def test_streaming_token(self, ch: MattermostChannel) -> None:
        ch._http_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {"id": "p1"}
        ch._http_client.post = AsyncMock(return_value=mock_resp)

        with patch("jarvis.channels.mattermost.asyncio.sleep", new_callable=AsyncMock):
            await ch.send_streaming_token("s1", "hello")


class TestMattermostStart:
    @pytest.mark.asyncio
    async def test_start_no_url(self) -> None:
        ch = MattermostChannel(url="", token="t")
        handler = AsyncMock()
        await ch.start(handler)
        assert ch._running is False

    @pytest.mark.asyncio
    async def test_start_no_token(self) -> None:
        ch = MattermostChannel(url="http://x", token="")
        handler = AsyncMock()
        await ch.start(handler)
        assert ch._running is False

    @pytest.mark.asyncio
    async def test_start_httpx_not_installed(self) -> None:
        ch = MattermostChannel(url="http://x", token="t")
        handler = AsyncMock()
        with patch.dict("sys.modules", {"httpx": None}):
            await ch.start(handler)
        assert ch._running is False

    @pytest.mark.asyncio
    async def test_start_success(self) -> None:
        ch = MattermostChannel(url="https://mm.test.com", token="valid-token")
        handler = AsyncMock()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "bot1", "username": "JarvisBot"}
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)

        with patch("jarvis.channels.mattermost.httpx.AsyncClient", return_value=mock_http):
            with patch.object(ch, "_websocket_loop", new_callable=AsyncMock):
                await ch.start(handler)

        assert ch._running is True
        assert ch._bot_user_id == "bot1"

    @pytest.mark.asyncio
    async def test_start_auth_failure(self) -> None:
        ch = MattermostChannel(url="https://mm.test.com", token="bad-token")
        handler = AsyncMock()

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)

        with patch("jarvis.channels.mattermost.httpx.AsyncClient", return_value=mock_http):
            with patch.object(ch, "_websocket_loop", new_callable=AsyncMock):
                await ch.start(handler)

        assert ch._running is True  # continues even with auth failure

    @pytest.mark.asyncio
    async def test_start_auth_exception(self) -> None:
        ch = MattermostChannel(url="https://mm.test.com", token="token")
        handler = AsyncMock()

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(side_effect=RuntimeError("connection refused"))

        with patch("jarvis.channels.mattermost.httpx.AsyncClient", return_value=mock_http):
            with patch.object(ch, "_websocket_loop", new_callable=AsyncMock):
                await ch.start(handler)

        assert ch._running is True  # continues with warning


class TestMattermostWebsocketLoop:
    @pytest.mark.asyncio
    async def test_websocket_loop_no_websockets(self, ch: MattermostChannel) -> None:
        ch._running = True
        with patch.dict("sys.modules", {"websockets": None}):
            await ch._websocket_loop()

    @pytest.mark.asyncio
    async def test_websocket_loop_connection_error(self, ch: MattermostChannel) -> None:
        ch._running = True
        call_count = 0

        mock_ws_module = MagicMock()

        async def fake_connect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                ch._running = False
            raise ConnectionError("refused")

        mock_ws_module.connect = fake_connect

        with patch.dict("sys.modules", {"websockets": mock_ws_module}):
            with patch("jarvis.channels.mattermost.asyncio.sleep", new_callable=AsyncMock):
                await ch._websocket_loop()

        assert ch._running is False


class TestMattermostApprovalFlow:
    @pytest.mark.asyncio
    async def test_approval_success_flow(self, ch: MattermostChannel) -> None:
        ch._http_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {"id": "post123"}
        ch._http_client.post = AsyncMock(return_value=mock_resp)

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

    @pytest.mark.asyncio
    async def test_approval_timeout(self, ch: MattermostChannel) -> None:
        ch._http_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {"id": "post456"}
        ch._http_client.post = AsyncMock(return_value=mock_resp)

        action = PlannedAction(tool="test", params={})
        with patch("jarvis.channels.mattermost.asyncio.wait_for", side_effect=asyncio.TimeoutError):
            result = await ch.request_approval("s1", action, "reason")
        assert result is False
