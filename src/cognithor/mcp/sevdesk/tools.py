"""MCP tool wrappers around the sevDesk client.

Functions here are decorated with ``@mcp_tool`` so they get picked up by the
integrations-catalog scanner (``scripts/generate_integrations_catalog.py``).
The decorator itself is a no-op marker — actual runtime registration with
the Cognithor MCP server happens via the existing ``register_*`` convention
in sibling modules; hook those up when the connector is wired into a live
Gateway.
"""

from __future__ import annotations

from typing import Any

from cognithor.mcp.sevdesk.client import SevdeskClient


def mcp_tool(fn):
    """Marker decorator — no-op. The integrations-catalog generator finds
    functions tagged with this decorator via AST parsing.
    """
    fn._is_mcp_tool = True
    return fn


@mcp_tool
async def sevdesk_list_contacts(limit: int = 50) -> list[dict[str, Any]]:
    """DACH accounting: list sevDesk contacts (Kunden/Lieferanten).

    Environment: SEVDESK_API_KEY must be set.
    """
    client = SevdeskClient()
    return await client.list_contacts(limit=limit)


@mcp_tool
async def sevdesk_get_invoice(invoice_id: str) -> dict[str, Any]:
    """DACH accounting: fetch a single sevDesk invoice (Rechnung) by id.

    Environment: SEVDESK_API_KEY must be set.
    """
    client = SevdeskClient()
    return await client.get_invoice(invoice_id=invoice_id)
