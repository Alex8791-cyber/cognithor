"""Web collector — uses existing search_and_read MCP tool."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from jarvis.osint.collectors.base import BaseCollector
from jarvis.osint.models import Evidence
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class WebCollector(BaseCollector):
    source_name = "web"

    def __init__(self, mcp_client: Any = None, language: str = "en") -> None:
        self._mcp = mcp_client
        self._language = language

    def is_available(self) -> bool:
        return self._mcp is not None

    async def collect(self, target: str, claims: list[str]) -> list[Evidence]:
        if not self._mcp:
            return []
        evidence: list[Evidence] = []
        now = datetime.now(timezone.utc)

        # Build claim-specific queries
        queries = [f'"{target}"']
        for claim in claims[:5]:
            queries.append(f"{target} {claim}")

        for query in queries[:6]:
            try:
                result = await self._mcp.call_tool(
                    "search_and_read",
                    {"query": query[:150], "num_results": 3, "language": self._language},
                )
                if result and not result.is_error and result.content:
                    text = result.content[:3000]
                    evidence.append(
                        Evidence(
                            source=f"web_search:{query[:50]}",
                            source_type="web",
                            content=text,
                            confidence=0.6,
                            collected_at=now,
                        )
                    )
            except Exception:
                log.debug("web_collector_query_failed", query=query[:40], exc_info=True)
        return evidence
