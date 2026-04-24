"""Task 70 — sevDesk MCP connector tests."""

from unittest.mock import AsyncMock, patch

import pytest

from cognithor.mcp.sevdesk.client import SevdeskAuthError, SevdeskClient
from cognithor.mcp.sevdesk.tools import sevdesk_get_invoice, sevdesk_list_contacts


@pytest.fixture
def _api_key(monkeypatch):
    monkeypatch.setenv("SEVDESK_API_KEY", "test-key-123")


def test_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("SEVDESK_API_KEY", raising=False)
    with pytest.raises(SevdeskAuthError, match="SEVDESK_API_KEY"):
        SevdeskClient()


async def test_list_contacts_happy_path(_api_key):
    fake_resp = {"objects": [{"id": "1", "name": "Acme GmbH"}]}
    with patch("cognithor.mcp.sevdesk.client.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.json = lambda: fake_resp
        mock_resp.raise_for_status = lambda: None
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_cls.return_value = mock_client

        result = await sevdesk_list_contacts(limit=10)

    assert result == [{"id": "1", "name": "Acme GmbH"}]


async def test_get_invoice_happy_path(_api_key):
    fake_resp = {"objects": [{"id": "42", "invoiceNumber": "2026-001"}]}
    with patch("cognithor.mcp.sevdesk.client.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.json = lambda: fake_resp
        mock_resp.raise_for_status = lambda: None
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_cls.return_value = mock_client

        result = await sevdesk_get_invoice(invoice_id="42")

    assert result == {"id": "42", "invoiceNumber": "2026-001"}


async def test_401_raises_auth_error(_api_key):
    with patch("cognithor.mcp.sevdesk.client.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_resp = AsyncMock()
        mock_resp.status_code = 401
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_cls.return_value = mock_client

        with pytest.raises(SevdeskAuthError, match="rejected"):
            await sevdesk_list_contacts()
