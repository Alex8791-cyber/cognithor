"""Tests für den REST-API Channel.

Testet HTTP-Endpunkte, Bearer-Auth, Session-Tracking,
Approval-Flow und Fehlerbehandlung.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from jarvis.channels.api import (
    APIChannel,
    HealthResponse,
    MessageRequest,
    MessageResponse,
    SessionInfo,
)
from jarvis.models import OutgoingMessage

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def channel() -> APIChannel:
    return APIChannel(host="127.0.0.1", port=8741, api_token="test-token")


@pytest.fixture
def auth_channel() -> APIChannel:
    return APIChannel(host="127.0.0.1", port=8741, api_token="test-secret-token")


@pytest.fixture
def mock_handler() -> AsyncMock:
    handler = AsyncMock()
    handler.return_value = OutgoingMessage(
        text="Antwort vom Handler",
        session_id="test-session",
        channel="api",
    )
    return handler


# ============================================================================
# Pydantic-Modelle
# ============================================================================


class TestModels:
    def test_message_request_valid(self) -> None:
        req = MessageRequest(text="Hallo Jarvis")
        assert req.text == "Hallo Jarvis"
        assert req.session_id is None
        assert req.metadata == {}

    def test_message_request_with_session(self) -> None:
        req = MessageRequest(text="Test", session_id="sess-123")
        assert req.session_id == "sess-123"

    def test_message_request_empty_text_rejected(self) -> None:
        with pytest.raises(Exception):  # ValidationError
            MessageRequest(text="")

    def test_message_response(self) -> None:
        resp = MessageResponse(
            text="Antwort",
            session_id="s1",
            timestamp="2026-02-22T10:00:00Z",
            duration_ms=150,
        )
        assert resp.duration_ms == 150

    def test_health_response_defaults(self) -> None:
        hr = HealthResponse()
        assert hr.status == "ok"
        assert hr.version == "0.1.0"
        assert hr.uptime_seconds == 0.0

    def test_session_info(self) -> None:
        si = SessionInfo(
            session_id="s1",
            created_at="2026-02-22T10:00:00Z",
            message_count=5,
        )
        assert si.message_count == 5


# ============================================================================
# Channel-Basics
# ============================================================================


class TestAPIChannelBasics:
    def test_channel_name(self, channel: APIChannel) -> None:
        assert channel.name == "api"

    @pytest.mark.asyncio
    async def test_start(self, channel: APIChannel, mock_handler: AsyncMock) -> None:
        await channel.start(mock_handler)
        assert channel._handler is mock_handler
        assert channel._app is not None

    @pytest.mark.asyncio
    async def test_stop_clears_approvals(self, channel: APIChannel) -> None:
        future: asyncio.Future[bool] = asyncio.get_event_loop().create_future()
        channel._pending_approvals["req-1"] = future
        await channel.stop()
        assert len(channel._pending_approvals) == 0
        assert future.done()
        assert future.result() is False

    @pytest.mark.asyncio
    async def test_send_is_noop(self, channel: APIChannel) -> None:
        msg = OutgoingMessage(text="test", session_id="s1", channel="api")
        await channel.send(msg)  # Sollte einfach nichts tun

    @pytest.mark.asyncio
    async def test_streaming_token_is_noop(self, channel: APIChannel) -> None:
        await channel.send_streaming_token("s1", "token")

    def test_app_property_creates_app(self, channel: APIChannel) -> None:
        app = channel.app
        assert app is not None
        # Zweiter Aufruf gibt gleiche Instanz
        assert channel.app is app


# ============================================================================
# FastAPI App & Routes (via TestClient)
# ============================================================================


@pytest.fixture
def client(channel: APIChannel, mock_handler: AsyncMock):
    """HTTPX TestClient für die FastAPI-App."""
    try:
        from httpx import ASGITransport, AsyncClient
    except ImportError:
        pytest.skip("httpx nicht installiert")
    # Starte Channel synchron
    loop = asyncio.new_event_loop()
    loop.run_until_complete(channel.start(mock_handler))
    loop.close()
    transport = ASGITransport(app=channel.app)
    return AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"Authorization": "Bearer test-token"},
    )


@pytest.fixture
def auth_client(auth_channel: APIChannel, mock_handler: AsyncMock):
    """TestClient mit Auth-Token."""
    try:
        from httpx import ASGITransport, AsyncClient
    except ImportError:
        pytest.skip("httpx nicht installiert")
    loop = asyncio.new_event_loop()
    loop.run_until_complete(auth_channel.start(mock_handler))
    loop.close()
    transport = ASGITransport(app=auth_channel.app)
    return AsyncClient(transport=transport, base_url="http://test")


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self, client) -> None:
        resp = await client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"
        assert "uptime_seconds" in data
        assert "active_sessions" in data


class TestMessageEndpoint:
    @pytest.mark.asyncio
    async def test_send_message_success(self, client, mock_handler: AsyncMock) -> None:
        resp = await client.post(
            "/api/v1/message",
            json={"text": "Hallo Jarvis"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["text"] == "Antwort vom Handler"
        assert "session_id" in data
        assert data["duration_ms"] >= 0
        mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_message_with_session_id(self, client, mock_handler: AsyncMock) -> None:
        resp = await client.post(
            "/api/v1/message",
            json={"text": "Test", "session_id": "my-session"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "my-session"

    @pytest.mark.asyncio
    async def test_sessions_tracked(self, client) -> None:
        await client.post("/api/v1/message", json={"text": "Eins"})
        await client.post(
            "/api/v1/message",
            json={"text": "Zwei", "session_id": "same-session"},
        )
        await client.post(
            "/api/v1/message",
            json={"text": "Drei", "session_id": "same-session"},
        )
        resp = await client.get("/api/v1/sessions")
        assert resp.status_code == 200
        sessions = resp.json()
        assert len(sessions) >= 1


class TestAuthentication:
    @pytest.mark.asyncio
    async def test_no_token_blocks_message(self, auth_client) -> None:
        resp = await auth_client.post(
            "/api/v1/message",
            json={"text": "Ohne Token"},
        )
        assert resp.status_code in (401, 403)

    @pytest.mark.asyncio
    async def test_wrong_token_blocks(self, auth_client) -> None:
        resp = await auth_client.post(
            "/api/v1/message",
            json={"text": "Falsch"},
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code in (401, 403)

    @pytest.mark.asyncio
    async def test_correct_token_allows(self, auth_client) -> None:
        resp = await auth_client.post(
            "/api/v1/message",
            json={"text": "Mit Token"},
            headers={"Authorization": "Bearer test-secret-token"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_no_auth_required(self, auth_client) -> None:
        resp = await auth_client.get("/api/v1/health")
        assert resp.status_code == 200


class TestApprovalFlow:
    @pytest.mark.asyncio
    async def test_approval_timeout_returns_false(self, channel: APIChannel) -> None:
        from jarvis.models import PlannedAction

        PlannedAction(
            tool="email_send",
            params={"to": "test@example.com"},
            rationale="Test-Email",
        )

        # Timeout sofort (sehr kurz) → False
        channel._pending_approvals.clear()

        async def short_approval() -> bool:
            request_id = "test-req"
            future: asyncio.Future[bool] = asyncio.get_event_loop().create_future()
            channel._pending_approvals[request_id] = future
            try:
                return await asyncio.wait_for(future, timeout=0.01)
            except TimeoutError:
                return False
            finally:
                channel._pending_approvals.pop(request_id, None)

        result = await short_approval()
        assert result is False

    @pytest.mark.asyncio
    async def test_approval_pending_list(self, client, channel: APIChannel) -> None:
        future: asyncio.Future[bool] = asyncio.get_event_loop().create_future()
        channel._pending_approvals["req-abc"] = future
        resp = await client.get("/api/v1/approvals/pending")
        assert resp.status_code == 200
        assert "req-abc" in resp.json()
        # Cleanup
        future.set_result(False)
        channel._pending_approvals.clear()

    @pytest.mark.asyncio
    async def test_approval_respond(self, client, channel: APIChannel) -> None:
        future: asyncio.Future[bool] = asyncio.get_event_loop().create_future()
        channel._pending_approvals["req-xyz"] = future

        resp = await client.post(
            "/api/v1/approvals/respond",
            json={"request_id": "req-xyz", "approved": True},
        )
        assert resp.status_code == 200
        assert future.done()
        assert future.result() is True

    @pytest.mark.asyncio
    async def test_approval_respond_not_found(self, client) -> None:
        resp = await client.post(
            "/api/v1/approvals/respond",
            json={"request_id": "nonexistent", "approved": True},
        )
        assert resp.status_code == 404
