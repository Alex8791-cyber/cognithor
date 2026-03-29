"""Tests for ResearchAgent — Phase 5B web fetching with multiple strategies."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from jarvis.evolution.models import SourceSpec
from jarvis.evolution.research_agent import FetchResult, ResearchAgent


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


@dataclass
class _MockToolResult:
    content: str = ""
    is_error: bool = False


def _make_agent(
    call_tool_side_effect=None,
    idle: bool = True,
    rate_limit: float = 0.0,
) -> tuple[ResearchAgent, AsyncMock, MagicMock]:
    """Build a ResearchAgent with mocked MCP client and idle detector."""
    mcp = AsyncMock()
    if call_tool_side_effect is not None:
        mcp.call_tool.side_effect = call_tool_side_effect
    idle_det = MagicMock()
    type(idle_det).is_idle = PropertyMock(return_value=idle)
    agent = ResearchAgent(
        mcp_client=mcp,
        idle_detector=idle_det,
        rate_limit_seconds=rate_limit,
    )
    return agent, mcp, idle_det


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_full_page():
    """Single full-page fetch returns FetchResult with url and text."""
    agent, mcp, _ = _make_agent(
        call_tool_side_effect=[
            _MockToolResult(content="Hello from the page", is_error=False),
        ],
    )
    source = SourceSpec(
        url="https://example.com/article",
        source_type="web",
        fetch_strategy="full_page",
    )
    results = await agent.fetch_source(source)

    assert len(results) == 1
    assert results[0].url == "https://example.com/article"
    assert "Hello from the page" in results[0].text
    mcp.call_tool.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_full_page_error():
    """MCP returning an error results in an empty list."""
    agent, mcp, _ = _make_agent(
        call_tool_side_effect=[
            _MockToolResult(content="404 not found", is_error=True),
            _MockToolResult(content="404 not found", is_error=True),
            _MockToolResult(content="404 not found", is_error=True),
        ],
    )
    source = SourceSpec(
        url="https://example.com/missing",
        source_type="web",
        fetch_strategy="full_page",
    )
    results = await agent.fetch_source(source)

    assert results == []


@pytest.mark.asyncio
async def test_fetch_sitemap_crawl():
    """Sitemap crawl fetches index, extracts links, fetches individual pages."""
    index_html = (
        "<html><body>"
        '<a href="https://example.com/page1">P1</a>'
        '<a href="https://example.com/page2">P2</a>'
        '<a href="https://other.com/nope">External</a>'
        "</body></html>"
    )
    agent, mcp, _ = _make_agent(
        call_tool_side_effect=[
            # First call: fetch the index page
            _MockToolResult(content=index_html, is_error=False),
            # Second call: fetch page1
            _MockToolResult(content="Page 1 content", is_error=False),
            # Third call: fetch page2
            _MockToolResult(content="Page 2 content", is_error=False),
        ],
    )
    source = SourceSpec(
        url="https://example.com/sitemap",
        source_type="web",
        fetch_strategy="sitemap_crawl",
        max_pages=10,
    )
    results = await agent.fetch_source(source)

    # Should have results for page1 and page2 (not the external link)
    assert len(results) == 2
    urls = {r.url for r in results}
    assert "https://example.com/page1" in urls
    assert "https://example.com/page2" in urls
    assert "https://other.com/nope" not in urls


@pytest.mark.asyncio
async def test_idle_check_aborts():
    """When idle_detector.is_idle is False, sitemap_crawl makes no MCP calls."""
    agent, mcp, _ = _make_agent(
        call_tool_side_effect=[],
        idle=False,
    )
    source = SourceSpec(
        url="https://example.com/sitemap",
        source_type="web",
        fetch_strategy="sitemap_crawl",
    )
    results = await agent.fetch_source(source)

    assert results == []
    mcp.call_tool.assert_not_called()


@pytest.mark.asyncio
async def test_rate_limiting():
    """Two sequential fetches both succeed (rate limiter does not block)."""
    call_results = [
        _MockToolResult(content="First page", is_error=False),
        _MockToolResult(content="Second page", is_error=False),
    ]
    agent, mcp, _ = _make_agent(
        call_tool_side_effect=call_results,
        rate_limit=0.0,  # no delay for test speed
    )

    r1 = await agent.fetch_source(
        SourceSpec(
            url="https://example.com/a",
            source_type="web",
            fetch_strategy="full_page",
        )
    )
    r2 = await agent.fetch_source(
        SourceSpec(
            url="https://example.com/b",
            source_type="web",
            fetch_strategy="full_page",
        )
    )

    assert len(r1) == 1
    assert len(r2) == 1
    assert r1[0].text == "First page"
    assert r2[0].text == "Second page"
    assert mcp.call_tool.call_count == 2


def test_extract_links():
    """extract_links resolves relative URLs and excludes anchor-only links."""
    html = (
        '<a href="/about">About</a>'
        '<a href="https://example.com/faq">FAQ</a>'
        '<a href="#section">Anchor</a>'
        '<a href="page2">Relative</a>'
    )
    agent, _, _ = _make_agent()
    links = agent.extract_links(html, "https://example.com/docs/index.html")

    assert "https://example.com/about" in links
    assert "https://example.com/faq" in links
    assert "https://example.com/docs/page2" in links
    # Anchor-only links must be excluded
    for link in links:
        assert not link.startswith("#")
    # No duplicates
    assert len(links) == len(set(links))


def test_fetch_result_dataclass():
    """FetchResult fields are correctly populated."""
    fr = FetchResult(
        url="https://example.com",
        text="content",
        title="Example",
        source_type="web",
        error=None,
    )
    assert fr.url == "https://example.com"
    assert fr.text == "content"
    assert fr.title == "Example"
    assert fr.source_type == "web"
    assert fr.error is None

    fr_err = FetchResult(
        url="https://fail.com",
        text="",
        title=None,
        source_type="web",
        error="timeout",
    )
    assert fr_err.error == "timeout"
