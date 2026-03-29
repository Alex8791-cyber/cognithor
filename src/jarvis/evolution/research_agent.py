"""ResearchAgent — web fetching with multiple strategies (Phase 5B)."""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urljoin, urlparse

from jarvis.evolution.models import SourceSpec
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["FetchResult", "ResearchAgent"]


@dataclass
class FetchResult:
    """Single fetched document."""

    url: str
    text: str
    title: Optional[str] = None
    source_type: Optional[str] = None
    error: Optional[str] = None


class ResearchAgent:
    """Fetches web content via MCP tools using pluggable strategies."""

    def __init__(
        self,
        mcp_client,
        idle_detector=None,
        rate_limit_seconds: float = 2.0,
        max_retries: int = 3,
    ) -> None:
        self._mcp = mcp_client
        self._idle = idle_detector
        self._rate_limit = rate_limit_seconds
        self._max_retries = max_retries
        self._last_fetch_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_source(self, source: SourceSpec) -> list[FetchResult]:
        """Dispatch to the right strategy based on *source.fetch_strategy*."""
        strategy = (source.fetch_strategy or "full_page").lower()
        dispatch = {
            "full_page": self._fetch_full_page,
            "sitemap_crawl": self._fetch_sitemap_crawl,
            "rss": self._fetch_rss,
        }
        handler = dispatch.get(strategy, self._fetch_full_page)
        try:
            return await handler(source)
        except Exception:
            log.exception("fetch_source failed for %s", source.url)
            return []

    def extract_links(self, html: str, base_url: str) -> list[str]:
        """Extract href links from *html*, resolve relative URLs, deduplicate.

        Anchor-only links (``#fragment``) are excluded.
        """
        raw = re.findall(r'href=["\']([^"\']+)["\']', html)
        seen: set[str] = set()
        result: list[str] = []
        for href in raw:
            href = href.strip()
            if not href or href.startswith("#"):
                continue
            absolute = urljoin(base_url, href)
            # Strip fragment
            absolute = absolute.split("#")[0]
            if absolute not in seen:
                seen.add(absolute)
                result.append(absolute)
        return result

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    async def _fetch_full_page(self, source: SourceSpec) -> list[FetchResult]:
        """Fetch a page and auto-follow internal sub-links for depth.

        If the page contains links to sub-pages on the same domain
        (e.g. law paragraphs, chapter pages), follow them automatically
        up to max_depth_pages. Also handles PDF links via read_pdf.
        """
        text = await self._web_fetch(source.url)
        if text is None:
            return []

        results = [
            FetchResult(
                url=source.url,
                text=text,
                title=source.title,
                source_type=source.source_type,
            )
        ]

        # Auto-follow: extract sub-links on the same domain
        from urllib.parse import urlparse

        base_domain = urlparse(source.url).netloc
        sub_links = self.extract_links(text, source.url)
        # Filter: same domain, not the same page, not anchors
        sub_links = [l for l in sub_links if urlparse(l).netloc == base_domain and l != source.url]

        # Heuristic: if page has many links (>10), it's likely a TOC → follow them
        max_depth_pages = min(len(sub_links), 15)
        if max_depth_pages < 3:
            return results  # Not a TOC page, just return the single page

        log.info(
            "research_auto_follow",
            url=source.url[:50],
            sub_links=len(sub_links),
            fetching=max_depth_pages,
        )

        for link in sub_links[:max_depth_pages]:
            if self._idle is not None and not self._idle.is_idle:
                break

            # Handle PDF links
            if link.lower().endswith(".pdf"):
                pdf_text = await self._fetch_pdf(link)
                if pdf_text:
                    results.append(
                        FetchResult(
                            url=link,
                            text=pdf_text,
                            title=f"{source.title} (PDF)",
                            source_type=source.source_type,
                        )
                    )
                continue

            sub_text = await self._web_fetch(link)
            if sub_text:
                results.append(
                    FetchResult(
                        url=link,
                        text=sub_text,
                        title=source.title,
                        source_type=source.source_type,
                    )
                )
            await asyncio.sleep(self._rate_limit)

        return results

    async def _fetch_pdf(self, url: str) -> str:
        """Fetch and parse a PDF via MCP read_pdf tool."""
        try:
            # First download PDF to temp location
            result = await self._mcp.call_tool(
                "web_fetch", {"url": url, "extract_text": True, "max_chars": 50000}
            )
            if result and not result.is_error and result.content:
                return result.content
        except Exception:
            log.debug("research_pdf_fetch_failed", url=url[:60], exc_info=True)
        return ""

    async def _fetch_sitemap_crawl(self, source: SourceSpec) -> list[FetchResult]:
        # Idle gate -- only crawl when the system is idle
        if self._idle is not None and not self._idle.is_idle:
            log.info("Skipping sitemap crawl -- system not idle")
            return []

        index_html = await self._web_fetch(source.url)
        if index_html is None:
            return []

        links = self.extract_links(index_html, source.url)

        # Filter to same domain
        base_domain = urlparse(source.url).netloc
        same_domain = [l for l in links if urlparse(l).netloc == base_domain]

        max_pages = source.max_pages or 50
        results: list[FetchResult] = []

        for link in same_domain[:max_pages]:
            # Re-check idle between pages
            if self._idle is not None and not self._idle.is_idle:
                log.info("Idle check failed mid-crawl, stopping")
                break

            text = await self._web_fetch(link)
            if text is not None:
                results.append(
                    FetchResult(
                        url=link,
                        text=text,
                        title=None,
                        source_type=source.source_type,
                    )
                )

        return results

    async def _fetch_rss(self, source: SourceSpec) -> list[FetchResult]:
        xml = await self._web_fetch(source.url)
        if xml is None:
            return []

        # Simple regex extraction of <link> tags from RSS/Atom
        links = re.findall(r"<link[^>]*>([^<]+)</link>", xml)
        if not links:
            # Try href attribute style (Atom)
            links = re.findall(r'<link[^>]+href=["\']([^"\']+)["\']', xml)

        max_pages = source.max_pages or 20
        results: list[FetchResult] = []

        for link in links[:max_pages]:
            link = link.strip()
            if not link or not link.startswith("http"):
                continue
            text = await self._web_fetch(link)
            if text is not None:
                results.append(
                    FetchResult(
                        url=link,
                        text=text,
                        title=None,
                        source_type=source.source_type,
                    )
                )

        return results

    # ------------------------------------------------------------------
    # Low-level fetch with retry + rate limit
    # ------------------------------------------------------------------

    async def _web_fetch(self, url: str) -> Optional[str]:
        """Fetch a URL via MCP ``web_fetch`` with retry and backoff."""
        backoffs = [5, 10, 15]

        for attempt in range(self._max_retries):
            # Rate limiting
            elapsed = time.monotonic() - self._last_fetch_time
            if elapsed < self._rate_limit:
                await asyncio.sleep(self._rate_limit - elapsed)

            try:
                result = await self._mcp.call_tool(
                    "web_fetch",
                    {"url": url, "extract_text": True, "max_chars": 50000},
                )
                self._last_fetch_time = time.monotonic()

                if result.is_error:
                    log.warning(
                        "web_fetch error (attempt %d/%d) for %s: %s",
                        attempt + 1,
                        self._max_retries,
                        url,
                        result.content[:200],
                    )
                    if attempt < self._max_retries - 1:
                        await asyncio.sleep(backoffs[attempt])
                    continue

                return result.content

            except Exception:
                log.exception(
                    "web_fetch exception (attempt %d/%d) for %s",
                    attempt + 1,
                    self._max_retries,
                    url,
                )
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(backoffs[attempt])

        return None
