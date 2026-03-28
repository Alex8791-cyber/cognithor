"""Base collector with retry logic."""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

import httpx

from jarvis.osint.models import Evidence
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class CollectorError(Exception):
    """Raised when a collector exhausts retries."""


class BaseCollector(ABC):
    source_name: str = "base"
    max_requests_per_minute: int = 30

    @abstractmethod
    async def collect(self, target: str, claims: list[str]) -> list[Evidence]:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...

    async def _fetch_with_retry(
        self, url: str, headers: dict[str, str] | None = None, max_retries: int = 3
    ) -> Any:
        """GET url with exponential backoff. Raises CollectorError on exhaustion."""
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=20) as client:
                    resp = await client.get(url, headers=headers or {})
                    resp.raise_for_status()
                    return resp.json()
            except (httpx.HTTPError, Exception) as e:
                if attempt == max_retries - 1:
                    raise CollectorError(f"Failed after {max_retries} retries: {e}") from e
                wait = 2 ** attempt
                log.debug("collector_retry", source=self.source_name, attempt=attempt, wait=wait)
                await asyncio.sleep(wait)
        raise CollectorError("Unreachable")
