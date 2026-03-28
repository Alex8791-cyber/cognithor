"""arXiv collector — official API, no key required."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import httpx

from jarvis.osint.collectors.base import BaseCollector
from jarvis.osint.models import Evidence
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

_ARXIV_API = "http://export.arxiv.org/api/query"
_ATOM_NS = "{http://www.w3.org/2005/Atom}"


class ArxivCollector(BaseCollector):
    source_name = "arxiv"

    def is_available(self) -> bool:
        return True

    async def collect(self, target: str, claims: list[str]) -> list[Evidence]:
        evidence: list[Evidence] = []
        now = datetime.now(timezone.utc)
        query = f'au:"{target}"'
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.get(
                    _ARXIV_API,
                    params={"search_query": query, "max_results": 10, "sortBy": "submittedDate"},
                )
                resp.raise_for_status()
            root = ET.fromstring(resp.text)
            for entry in root.findall(f"{_ATOM_NS}entry"):
                title = (entry.findtext(f"{_ATOM_NS}title") or "").strip()
                summary = (entry.findtext(f"{_ATOM_NS}summary") or "").strip()[:300]
                published = entry.findtext(f"{_ATOM_NS}published") or ""
                entry_id = entry.findtext(f"{_ATOM_NS}id") or ""
                authors = [
                    a.findtext(f"{_ATOM_NS}name") or ""
                    for a in entry.findall(f"{_ATOM_NS}author")
                ]
                evidence.append(Evidence(
                    source=f"arxiv:{entry_id.split('/')[-1]}",
                    source_type="arxiv",
                    content=(
                        f"Paper: {title} | "
                        f"Authors: {', '.join(authors[:5])} | "
                        f"Published: {published[:10]} | "
                        f"Abstract: {summary}"
                    ),
                    confidence=0.85,
                    collected_at=now,
                    url=entry_id,
                ))
        except Exception:
            log.debug("arxiv_collector_failed", exc_info=True)
        return evidence
