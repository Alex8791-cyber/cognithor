"""sevDesk REST API client (DACH accounting SaaS).

https://my.sevdesk.de/api/v1/Api.html — API-key authentication,
OAuth-free. All requests go through httpx.AsyncClient.
"""

from __future__ import annotations

import os
from typing import Any

import httpx


class SevdeskAuthError(Exception):
    """API key missing or rejected."""


class SevdeskClient:
    """Minimal sevDesk REST client — list contacts, fetch invoice.

    Configure via environment:
      * ``SEVDESK_API_KEY`` — account API token
      * ``SEVDESK_BASE_URL`` — overrides the default production endpoint
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._api_key = api_key or os.getenv("SEVDESK_API_KEY", "")
        if not self._api_key:
            raise SevdeskAuthError("SEVDESK_API_KEY not set. Get a key at https://my.sevdesk.de/")
        self._base_url = (
            base_url or os.getenv("SEVDESK_BASE_URL") or "https://my.sevdesk.de/api/v1"
        ).rstrip("/")
        self._timeout = timeout

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        headers = {"Authorization": self._api_key, "Accept": "application/json"}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(f"{self._base_url}{path}", headers=headers, params=params)
            if resp.status_code == 401:
                raise SevdeskAuthError("sevDesk rejected the API key")
            resp.raise_for_status()
            return resp.json()

    async def list_contacts(self, limit: int = 50) -> list[dict]:
        data = await self._get("/Contact", params={"limit": limit})
        return data.get("objects", [])

    async def get_invoice(self, invoice_id: str) -> dict:
        data = await self._get(f"/Invoice/{invoice_id}")
        objs = data.get("objects") or [{}]
        return objs[0] if objs else {}
