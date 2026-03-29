"""Google Scholar collector — STUB (Phase 2)."""

from __future__ import annotations

from jarvis.osint.collectors.base import BaseCollector
from jarvis.osint.models import Evidence


class ScholarCollector(BaseCollector):
    source_name = "scholar"

    def is_available(self) -> bool:
        return False

    async def collect(self, target: str, claims: list[str]) -> list[Evidence]:
        return []
