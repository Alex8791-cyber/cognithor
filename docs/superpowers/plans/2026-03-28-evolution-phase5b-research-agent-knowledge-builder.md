# Evolution Engine Phase 5B — ResearchAgent + KnowledgeBuilder

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** SubGoals from a LearningPlan are actually executed — sources fetched from the web, content parsed, and knowledge triple-written to Vault + Memory + Knowledge Graph.

**Architecture:** `ResearchAgent` fetches web content using multiple strategies (full_page, sitemap_crawl, rss). `KnowledgeBuilder` takes raw text and writes it to all three storage systems. `DeepLearner` orchestrates the Research→Build cycle per SubGoal. All operations are interruptible (idle check), checkpointed, and rate-limited.

**Tech Stack:** Python 3.12+ (asyncio, json, re), MCP tools (web_fetch, vault_save, save_to_memory, add_entity, add_relation), pytest

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/jarvis/evolution/research_agent.py` | Fetch web content: full_page, sitemap_crawl, rss strategies |
| Create | `src/jarvis/evolution/knowledge_builder.py` | Triple-write: Vault + Memory (chunked) + Graph (entity extraction) |
| Modify | `src/jarvis/evolution/deep_learner.py` | Add `run_subgoal()` orchestrating Research→Build per SubGoal |
| Create | `tests/unit/test_research_agent.py` | Tests for ResearchAgent |
| Create | `tests/unit/test_knowledge_builder.py` | Tests for KnowledgeBuilder |

---

### Task 1: ResearchAgent — Web Fetching with Multiple Strategies

**Files:**
- Create: `src/jarvis/evolution/research_agent.py`
- Create: `tests/unit/test_research_agent.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/test_research_agent.py
"""Tests fuer ResearchAgent."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.evolution.research_agent import ResearchAgent, FetchResult


@dataclass
class _MockToolResult:
    content: str = ""
    is_error: bool = False


@pytest.fixture()
def mock_mcp():
    """MCP client that returns page content."""
    client = AsyncMock()
    client.call_tool = AsyncMock(return_value=_MockToolResult(
        content="<h1>VVG §1</h1>\nDer Versicherer schuldet...",
    ))
    return client


@pytest.fixture()
def mock_idle():
    idle = MagicMock()
    idle.is_idle = True
    return idle


@pytest.fixture()
def agent(mock_mcp, mock_idle):
    return ResearchAgent(mcp_client=mock_mcp, idle_detector=mock_idle)


class TestResearchAgent:
    @pytest.mark.asyncio
    async def test_fetch_full_page(self, agent, mock_mcp):
        """full_page Strategie fetcht eine URL."""
        from jarvis.evolution.models import SourceSpec
        source = SourceSpec(
            url="https://example.com/vvg",
            source_type="law",
            title="VVG",
            fetch_strategy="full_page",
            update_frequency="once",
        )
        results = await agent.fetch_source(source)
        assert len(results) == 1
        assert results[0].url == "https://example.com/vvg"
        assert "Versicherer" in results[0].text
        mock_mcp.call_tool.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_full_page_error(self, agent, mock_mcp):
        """Fehler bei web_fetch → leere Liste."""
        mock_mcp.call_tool.return_value = _MockToolResult(content="", is_error=True)
        from jarvis.evolution.models import SourceSpec
        source = SourceSpec(url="https://fail.com", source_type="reference",
                           title="Fail", fetch_strategy="full_page", update_frequency="once")
        results = await agent.fetch_source(source)
        assert results == []

    @pytest.mark.asyncio
    async def test_fetch_sitemap_crawl(self, agent, mock_mcp):
        """sitemap_crawl extrahiert Links und fetcht jede Seite."""
        # First call: index page with links
        # Subsequent calls: individual pages
        call_count = 0
        async def _multi_response(tool_name, params):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _MockToolResult(
                    content='<a href="/vvg/p1">§1</a>\n<a href="/vvg/p2">§2</a>\n<a href="/vvg/p3">§3</a>'
                )
            return _MockToolResult(content=f"Content of page {call_count}")
        mock_mcp.call_tool = AsyncMock(side_effect=_multi_response)

        from jarvis.evolution.models import SourceSpec
        source = SourceSpec(url="https://example.com/vvg/", source_type="law",
                           title="VVG", fetch_strategy="sitemap_crawl",
                           update_frequency="once", max_pages=10)
        results = await agent.fetch_source(source)
        assert len(results) >= 2  # Index + at least some sub-pages

    @pytest.mark.asyncio
    async def test_idle_check_aborts(self, agent, mock_idle, mock_mcp):
        """User kommt zurueck → fetch stoppt sofort."""
        mock_idle.is_idle = False
        from jarvis.evolution.models import SourceSpec
        source = SourceSpec(url="https://example.com", source_type="reference",
                           title="Test", fetch_strategy="full_page", update_frequency="once")
        results = await agent.fetch_source(source)
        assert results == []
        mock_mcp.call_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_rate_limiting(self, agent, mock_mcp):
        """Mehrere Fetches respektieren Rate-Limit."""
        from jarvis.evolution.models import SourceSpec
        source = SourceSpec(url="https://example.com/", source_type="law",
                           title="Test", fetch_strategy="full_page", update_frequency="once")
        # Fetch twice in sequence
        await agent.fetch_source(source)
        await agent.fetch_source(source)
        assert mock_mcp.call_tool.call_count == 2

    def test_extract_links(self, agent):
        """Link-Extraktion aus HTML."""
        html = '''
        <a href="/vvg/p1">§1 Vertragstypische Pflichten</a>
        <a href="/vvg/p2">§2 Rückwärtsversicherung</a>
        <a href="https://other.com/ads">Werbung</a>
        <a href="#anchor">Intern</a>
        '''
        links = agent.extract_links(html, base_url="https://example.com")
        assert "https://example.com/vvg/p1" in links
        assert "https://example.com/vvg/p2" in links
        # Anchors and external links optionally included
        assert all(l.startswith("http") for l in links)

    def test_fetch_result_dataclass(self):
        """FetchResult hat alle Felder."""
        r = FetchResult(url="https://example.com", text="content", title="Test")
        assert r.url == "https://example.com"
        assert r.text == "content"
```

- [ ] **Step 2: Run tests — verify fail**

Run: `cd "D:\Jarvis\jarvis complete v20" && python -m pytest tests/unit/test_research_agent.py -v`

- [ ] **Step 3: Implement research_agent.py**

```python
# src/jarvis/evolution/research_agent.py
"""ResearchAgent — fetches web content for SubGoals using multiple strategies."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlparse

from jarvis.evolution.models import SourceSpec
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["FetchResult", "ResearchAgent"]


@dataclass
class FetchResult:
    """Result of fetching a single page."""

    url: str = ""
    text: str = ""
    title: str = ""
    source_type: str = ""
    error: str = ""


class ResearchAgent:
    """Fetches web content for evolution SubGoals.

    Strategies:
    - full_page: Single page fetch via web_fetch
    - sitemap_crawl: Fetch index page, extract links, fetch each
    - rss: Parse RSS feed, fetch new entries
    """

    def __init__(
        self,
        mcp_client: Any,
        idle_detector: Any = None,
        rate_limit_seconds: float = 2.0,
        max_retries: int = 3,
    ) -> None:
        self._mcp = mcp_client
        self._idle = idle_detector
        self._rate_limit = rate_limit_seconds
        self._max_retries = max_retries
        self._last_fetch_time: float = 0.0

    async def fetch_source(self, source: SourceSpec) -> list[FetchResult]:
        """Fetch content from a source using its configured strategy."""
        if self._idle and not self._idle.is_idle:
            return []

        strategy = source.fetch_strategy
        if strategy == "full_page":
            return await self._fetch_full_page(source)
        elif strategy == "sitemap_crawl":
            return await self._fetch_sitemap_crawl(source)
        elif strategy == "rss":
            return await self._fetch_rss(source)
        else:
            log.warning("unknown_fetch_strategy", strategy=strategy)
            return await self._fetch_full_page(source)

    # -- Strategies -------------------------------------------------------

    async def _fetch_full_page(self, source: SourceSpec) -> list[FetchResult]:
        """Fetch a single page."""
        text = await self._web_fetch(source.url)
        if not text:
            return []
        return [FetchResult(
            url=source.url,
            text=text,
            title=source.title,
            source_type=source.source_type,
        )]

    async def _fetch_sitemap_crawl(self, source: SourceSpec) -> list[FetchResult]:
        """Fetch index page, extract links, fetch each sub-page."""
        results: list[FetchResult] = []

        # Fetch index page
        index_html = await self._web_fetch(source.url)
        if not index_html:
            return results
        results.append(FetchResult(
            url=source.url, text=index_html,
            title=f"{source.title} (Index)", source_type=source.source_type,
        ))

        # Extract links from index
        links = self.extract_links(index_html, base_url=source.url)
        # Filter to same domain
        base_domain = urlparse(source.url).netloc
        links = [l for l in links if urlparse(l).netloc == base_domain]
        # Limit pages
        links = links[: source.max_pages]

        log.info("research_crawl_links", url=source.url[:60], links_found=len(links))

        for link in links:
            if self._idle and not self._idle.is_idle:
                log.info("research_crawl_interrupted", fetched=len(results))
                break
            text = await self._web_fetch(link)
            if text:
                results.append(FetchResult(
                    url=link, text=text, title=source.title,
                    source_type=source.source_type,
                ))
            await asyncio.sleep(self._rate_limit)

        return results

    async def _fetch_rss(self, source: SourceSpec) -> list[FetchResult]:
        """Fetch RSS feed, extract entry URLs, fetch each new entry."""
        results: list[FetchResult] = []
        feed_text = await self._web_fetch(source.url)
        if not feed_text:
            return results

        # Simple RSS/Atom link extraction
        urls = re.findall(r'<link[^>]*>([^<]+)</link>', feed_text)
        if not urls:
            urls = re.findall(r'<link[^>]*href=["\']([^"\']+)["\']', feed_text)
        urls = [u.strip() for u in urls if u.strip().startswith("http")]
        urls = urls[: source.max_pages]

        for url in urls:
            if self._idle and not self._idle.is_idle:
                break
            text = await self._web_fetch(url)
            if text:
                results.append(FetchResult(
                    url=url, text=text, title=source.title,
                    source_type=source.source_type,
                ))
            await asyncio.sleep(self._rate_limit)

        return results

    # -- Helpers -----------------------------------------------------------

    async def _web_fetch(self, url: str) -> str:
        """Fetch a URL via MCP web_fetch tool. Returns text or empty on error."""
        for attempt in range(self._max_retries):
            try:
                result = await self._mcp.call_tool(
                    "web_fetch",
                    {"url": url, "extract_text": True, "max_chars": 50000},
                )
                if result.is_error or not result.content:
                    if attempt < self._max_retries - 1:
                        await asyncio.sleep(5 * (attempt + 1))
                        continue
                    return ""
                return result.content
            except Exception:
                log.debug("research_fetch_failed", url=url[:80], exc_info=True)
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(5 * (attempt + 1))
        return ""

    def extract_links(self, html: str, base_url: str = "") -> list[str]:
        """Extract all href links from HTML, resolve relative URLs."""
        links: list[str] = []
        for match in re.finditer(r'href=["\']([^"\'#][^"\']*)["\']', html):
            href = match.group(1).strip()
            if href.startswith("http"):
                links.append(href)
            elif base_url and href.startswith("/"):
                links.append(urljoin(base_url, href))
        # Deduplicate preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for link in links:
            if link not in seen:
                seen.add(link)
                unique.append(link)
        return unique
```

- [ ] **Step 4: Run tests — verify pass**

Run: `cd "D:\Jarvis\jarvis complete v20" && python -m pytest tests/unit/test_research_agent.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/jarvis/evolution/research_agent.py tests/unit/test_research_agent.py
git commit -m "feat(evolution): add ResearchAgent — web fetching with full_page, sitemap_crawl, rss strategies"
```

---

### Task 2: KnowledgeBuilder — Triple-Write to Vault + Memory + Graph

**Files:**
- Create: `src/jarvis/evolution/knowledge_builder.py`
- Create: `tests/unit/test_knowledge_builder.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/test_knowledge_builder.py
"""Tests fuer KnowledgeBuilder."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.evolution.knowledge_builder import KnowledgeBuilder, BuildResult
from jarvis.evolution.research_agent import FetchResult


@dataclass
class _MockToolResult:
    content: str = ""
    is_error: bool = False


@pytest.fixture()
def mock_mcp():
    client = AsyncMock()
    client.call_tool = AsyncMock(return_value=_MockToolResult(content="OK"))
    return client


@pytest.fixture()
def mock_llm():
    """LLM that returns entity extraction JSON."""
    async def _llm(prompt: str) -> str:
        import json
        return json.dumps({
            "entities": [
                {"name": "VVG", "type": "law", "attributes": {"full_name": "Versicherungsvertragsgesetz"}},
                {"name": "Widerrufsrecht", "type": "concept", "attributes": {}},
            ],
            "relations": [
                {"source": "VVG", "relation": "regelt", "target": "Widerrufsrecht"},
            ],
        })
    return _llm


@pytest.fixture()
def builder(mock_mcp, mock_llm):
    return KnowledgeBuilder(
        mcp_client=mock_mcp,
        llm_fn=mock_llm,
        goal_slug="versicherungsrecht",
    )


class TestKnowledgeBuilder:
    @pytest.mark.asyncio
    async def test_build_from_fetch_result(self, builder, mock_mcp):
        """FetchResult → Vault + Memory + Graph."""
        fr = FetchResult(url="https://example.com/vvg", text="Der Versicherer schuldet dem Versicherungsnehmer...", title="VVG §1")
        result = await builder.build(fr)
        assert isinstance(result, BuildResult)
        assert result.vault_path != ""
        assert result.chunks_created > 0
        assert result.entities_created >= 1
        assert result.relations_created >= 1
        # vault_save called
        vault_calls = [c for c in mock_mcp.call_tool.call_args_list if c[0][0] == "vault_save"]
        assert len(vault_calls) >= 1
        # save_to_memory called for chunks
        mem_calls = [c for c in mock_mcp.call_tool.call_args_list if c[0][0] == "save_to_memory"]
        assert len(mem_calls) >= 1

    @pytest.mark.asyncio
    async def test_chunking(self, builder):
        """Langer Text wird in Chunks aufgeteilt."""
        long_text = "Wort " * 2000  # ~2000 Woerter
        chunks = builder.chunk_text(long_text, max_tokens=512, overlap_tokens=64)
        assert len(chunks) > 1
        # Each chunk should be smaller than max
        for chunk in chunks:
            assert len(chunk.split()) <= 600  # Some tolerance

    @pytest.mark.asyncio
    async def test_chunking_short_text(self, builder):
        """Kurzer Text → ein Chunk."""
        chunks = builder.chunk_text("Kurzer Satz.", max_tokens=512, overlap_tokens=64)
        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_entity_extraction(self, builder, mock_llm):
        """LLM extrahiert Entities + Relations aus Text."""
        entities, relations = await builder.extract_entities("VVG regelt Widerrufsrecht")
        assert len(entities) == 2
        assert entities[0]["name"] == "VVG"
        assert len(relations) == 1
        assert relations[0]["source"] == "VVG"

    @pytest.mark.asyncio
    async def test_entity_extraction_llm_failure(self, builder):
        """LLM gibt kein JSON → leere Listen."""
        async def _bad_llm(prompt):
            return "Not valid JSON"
        builder._llm_fn = _bad_llm
        entities, relations = await builder.extract_entities("Test text")
        assert entities == []
        assert relations == []

    @pytest.mark.asyncio
    async def test_vault_folder_uses_goal_slug(self, builder, mock_mcp):
        """Vault-Ordner ist wissen/{goal_slug}."""
        fr = FetchResult(url="https://example.com", text="Content", title="Test")
        await builder.build(fr)
        vault_call = [c for c in mock_mcp.call_tool.call_args_list if c[0][0] == "vault_save"][0]
        params = vault_call[0][1]
        assert "versicherungsrecht" in params["folder"]

    @pytest.mark.asyncio
    async def test_memory_uses_goal_slug_as_topic(self, builder, mock_mcp):
        """Memory-Chunks werden mit topic=goal_slug gespeichert."""
        fr = FetchResult(url="https://example.com", text="Content for memory", title="Test")
        await builder.build(fr)
        mem_calls = [c for c in mock_mcp.call_tool.call_args_list if c[0][0] == "save_to_memory"]
        for call in mem_calls:
            params = call[0][1]
            assert params.get("tier") == "semantic"

    @pytest.mark.asyncio
    async def test_build_result_accumulates(self, builder, mock_mcp):
        """Mehrere FetchResults → kumulative Ergebnisse."""
        results = []
        for i in range(3):
            fr = FetchResult(url=f"https://example.com/{i}", text=f"Content {i}", title=f"Page {i}")
            results.append(await builder.build(fr))
        total_chunks = sum(r.chunks_created for r in results)
        assert total_chunks >= 3

    def test_build_result_dataclass(self):
        r = BuildResult()
        assert r.vault_path == ""
        assert r.chunks_created == 0
        assert r.entities_created == 0
```

- [ ] **Step 2: Run tests — verify fail**

Run: `cd "D:\Jarvis\jarvis complete v20" && python -m pytest tests/unit/test_knowledge_builder.py -v`

- [ ] **Step 3: Implement knowledge_builder.py**

```python
# src/jarvis/evolution/knowledge_builder.py
"""KnowledgeBuilder — triple-writes fetched content to Vault + Memory + Knowledge Graph."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from jarvis.evolution.research_agent import FetchResult
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["BuildResult", "KnowledgeBuilder"]

_ENTITY_EXTRACTION_PROMPT = """\
Extrahiere die wichtigsten Entitaeten und deren Beziehungen aus folgendem Text.

Text:
\"\"\"
{text}
\"\"\"

Antworte NUR mit JSON in diesem Format:
{{
  "entities": [
    {{"name": "Name", "type": "person|law|concept|organization|product|event", "attributes": {{}}}}
  ],
  "relations": [
    {{"source": "Name1", "relation": "relation_type", "target": "Name2"}}
  ]
}}

Regeln:
- Maximal 10 Entitaeten und 10 Relationen
- Nur die WICHTIGSTEN extrahieren
- Relation-Typen: regelt, teil_von, gehoert_zu, definiert, referenziert, widerspricht, ergaenzt
- Antworte NUR mit JSON, kein anderer Text
"""


@dataclass
class BuildResult:
    """Result of building knowledge from one FetchResult."""

    vault_path: str = ""
    chunks_created: int = 0
    entities_created: int = 0
    relations_created: int = 0
    errors: list[str] = field(default_factory=list)


class KnowledgeBuilder:
    """Writes fetched content to Vault + Memory + Knowledge Graph.

    Triple-write pipeline per document:
    1. Vault: Full text as Markdown note (folder: wissen/{goal_slug}/)
    2. Memory: Chunked text → semantic memory with topic tag
    3. Graph: LLM-extracted entities + relations
    """

    def __init__(
        self,
        mcp_client: Any,
        llm_fn: Callable[[str], Coroutine[Any, Any, str]] | None = None,
        goal_slug: str = "",
    ) -> None:
        self._mcp = mcp_client
        self._llm_fn = llm_fn
        self._goal_slug = goal_slug

    async def build(self, fetch_result: FetchResult) -> BuildResult:
        """Process a FetchResult into all three storage systems."""
        result = BuildResult()
        text = fetch_result.text
        if not text or not text.strip():
            return result

        # 1. Vault: save full text
        try:
            vault_response = await self._mcp.call_tool(
                "vault_save",
                {
                    "title": fetch_result.title or fetch_result.url.split("/")[-1] or "Untitled",
                    "content": text,
                    "tags": f"{self._goal_slug}, {fetch_result.source_type}, auto-indexed",
                    "folder": f"wissen/{self._goal_slug}" if self._goal_slug else "wissen",
                    "sources": fetch_result.url,
                },
            )
            if not vault_response.is_error:
                result.vault_path = vault_response.content
        except Exception as exc:
            result.errors.append(f"vault: {exc}")
            log.debug("knowledge_builder_vault_failed", exc_info=True)

        # 2. Memory: chunk and save to semantic memory
        chunks = self.chunk_text(text, max_tokens=512, overlap_tokens=64)
        for chunk in chunks:
            try:
                await self._mcp.call_tool(
                    "save_to_memory",
                    {
                        "content": chunk,
                        "tier": "semantic",
                        "source_path": f"evolution/{self._goal_slug}/{fetch_result.title or 'auto'}.md",
                    },
                )
                result.chunks_created += 1
            except Exception:
                log.debug("knowledge_builder_memory_chunk_failed", exc_info=True)

        # 3. Graph: extract entities + relations via LLM
        if self._llm_fn:
            try:
                entities, relations = await self.extract_entities(text[:3000])
                domain_attr = json.dumps({"domain": self._goal_slug})
                for ent in entities:
                    attrs = ent.get("attributes", {})
                    attrs["domain"] = self._goal_slug
                    await self._mcp.call_tool(
                        "add_entity",
                        {
                            "name": ent["name"],
                            "entity_type": ent.get("type", "concept"),
                            "attributes": json.dumps(attrs),
                            "source_file": fetch_result.url,
                        },
                    )
                    result.entities_created += 1
                for rel in relations:
                    await self._mcp.call_tool(
                        "add_relation",
                        {
                            "source_name": rel["source"],
                            "relation_type": rel["relation"],
                            "target_name": rel["target"],
                            "attributes": domain_attr,
                        },
                    )
                    result.relations_created += 1
            except Exception:
                result.errors.append("graph: entity extraction failed")
                log.debug("knowledge_builder_graph_failed", exc_info=True)

        log.info(
            "knowledge_built",
            url=fetch_result.url[:60],
            vault=result.vault_path[:40] if result.vault_path else "",
            chunks=result.chunks_created,
            entities=result.entities_created,
            relations=result.relations_created,
        )
        return result

    def chunk_text(
        self,
        text: str,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
    ) -> list[str]:
        """Split text into overlapping chunks by word count (approximation of tokens)."""
        words = text.split()
        if len(words) <= max_tokens:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(words):
            end = min(start + max_tokens, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += max_tokens - overlap_tokens
        return chunks

    async def extract_entities(
        self, text: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Extract entities and relations from text via LLM."""
        if not self._llm_fn:
            return [], []
        try:
            prompt = _ENTITY_EXTRACTION_PROMPT.format(text=text[:3000])
            raw = await self._llm_fn(prompt)
            # Extract JSON from response
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not json_match:
                return [], []
            data = json.loads(json_match.group())
            entities = data.get("entities", [])
            relations = data.get("relations", [])
            return entities, relations
        except (json.JSONDecodeError, Exception):
            log.debug("entity_extraction_failed", exc_info=True)
            return [], []
```

- [ ] **Step 4: Run tests — verify pass**

Run: `cd "D:\Jarvis\jarvis complete v20" && python -m pytest tests/unit/test_knowledge_builder.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/jarvis/evolution/knowledge_builder.py tests/unit/test_knowledge_builder.py
git commit -m "feat(evolution): add KnowledgeBuilder — triple-write Vault + Memory + Graph"
```

---

### Task 3: DeepLearner — Wire Research→Build Cycle

**Files:**
- Modify: `src/jarvis/evolution/deep_learner.py`

- [ ] **Step 1: Add ResearchAgent + KnowledgeBuilder imports and init**

In `deep_learner.py`, add to imports:
```python
from jarvis.evolution.research_agent import ResearchAgent
from jarvis.evolution.knowledge_builder import KnowledgeBuilder
```

In `__init__`, after `self._strategy_planner = ...`, add:
```python
        self._research_agent = ResearchAgent(
            mcp_client=mcp_client,
            idle_detector=idle_detector,
        ) if mcp_client else None
```

- [ ] **Step 2: Add run_subgoal method**

```python
    async def run_subgoal(self, plan_id: str, subgoal_id: str) -> bool:
        """Execute Research→Build for a single SubGoal."""
        plan = self.get_plan(plan_id)
        if not plan:
            return False
        subgoal = next((sg for sg in plan.sub_goals if sg.id == subgoal_id), None)
        if not subgoal:
            return False

        subgoal.status = "researching"
        plan.save(self._plans_dir)

        # Find sources for this subgoal
        sources = [s for s in plan.sources if s.status == "pending"]
        if not sources:
            # Use web_search to find sources for the subgoal topic
            sources_for_topic = await self._discover_sources(subgoal.title)
            sources = sources_for_topic

        # Research phase
        builder = KnowledgeBuilder(
            mcp_client=self._mcp_client,
            llm_fn=self._llm_fn,
            goal_slug=plan.goal_slug,
        )

        for source in sources:
            if self._idle and not self._idle.is_idle:
                log.info("deep_learner_subgoal_interrupted", subgoal=subgoal.title[:40])
                plan.save(self._plans_dir)
                return False

            log.info("deep_learner_fetching", source=source.url[:60], subgoal=subgoal.title[:40])
            fetch_results = await self._research_agent.fetch_source(source)
            source.status = "done" if fetch_results else "error"
            source.pages_fetched = len(fetch_results)

            # Build phase — process each fetched page
            subgoal.status = "building"
            for fr in fetch_results:
                if self._idle and not self._idle.is_idle:
                    plan.save(self._plans_dir)
                    return False
                build_result = await builder.build(fr)
                subgoal.chunks_created += build_result.chunks_created
                subgoal.entities_created += build_result.entities_created
                if build_result.vault_path:
                    subgoal.vault_entries.append(build_result.vault_path)
                subgoal.sources_fetched.append(fr.url)

        subgoal.status = "testing"  # Ready for QualityAssessor (Phase 5C)
        plan.total_chunks_indexed += subgoal.chunks_created
        plan.total_entities_created += subgoal.entities_created
        plan.total_vault_entries += len(subgoal.vault_entries)
        plan.save(self._plans_dir)

        log.info(
            "deep_learner_subgoal_complete",
            subgoal=subgoal.title[:40],
            chunks=subgoal.chunks_created,
            entities=subgoal.entities_created,
            vault=len(subgoal.vault_entries),
        )
        return True

    async def _discover_sources(self, topic: str) -> list:
        """Use web_search to find sources for a topic when none are specified."""
        from jarvis.evolution.models import SourceSpec
        if not self._mcp_client:
            return []
        try:
            result = await self._mcp_client.call_tool(
                "web_search",
                {"query": topic, "num_results": 5, "language": "de"},
            )
            if result.is_error:
                return []
            # Parse URLs from search results
            import re
            urls = re.findall(r'https?://[^\s<>"\']+', result.content)
            return [
                SourceSpec(
                    url=url,
                    source_type="reference",
                    title=topic,
                    fetch_strategy="full_page",
                    update_frequency="once",
                )
                for url in urls[:5]
            ]
        except Exception:
            return []
```

- [ ] **Step 3: Run all tests**

Run: `cd "D:\Jarvis\jarvis complete v20" && python -m pytest tests/unit/test_deep_learner.py tests/unit/test_research_agent.py tests/unit/test_knowledge_builder.py -v`

- [ ] **Step 4: Commit**

```bash
git add src/jarvis/evolution/deep_learner.py
git commit -m "feat(evolution): wire Research→Build cycle into DeepLearner.run_subgoal()"
```

---

### Task 4: Integration — EvolutionLoop Calls run_subgoal

**Files:**
- Modify: `src/jarvis/evolution/loop.py`

- [ ] **Step 1: Modify _research and _build for deep plan goals**

In the `run_cycle()` method, after setting `result.research_topic` and `result.source`, check if this is a deep plan goal and handle it differently:

In the `_research()` method, add at the top:
```python
        # Deep plan goals are handled by DeepLearner directly
        if hasattr(gap, 'source') and gap.source == "deep_plan" and self._deep_learner:
            query_str = getattr(gap, "query", "")
            # Parse [deep:plan_id:subgoal_id] from query
            import re
            match = re.match(r"\[deep:([^:]+):([^\]]+)\]", query_str)
            if match:
                plan_id, sg_id = match.group(1), match.group(2)
                success = await self._deep_learner.run_subgoal(plan_id, sg_id)
                return f"DeepLearner executed subgoal: {'success' if success else 'interrupted'}"
```

- [ ] **Step 2: Run all evolution tests**

Run: `cd "D:\Jarvis\jarvis complete v20" && python -m pytest tests/unit/test_evolution.py tests/unit/test_evolution_models.py tests/unit/test_strategy_planner.py tests/unit/test_deep_learner.py tests/unit/test_research_agent.py tests/unit/test_knowledge_builder.py -v`
Expected: All tests PASS.

- [ ] **Step 3: Commit + push**

```bash
git add src/jarvis/evolution/loop.py
git commit -m "feat(evolution): deep plan goals trigger DeepLearner.run_subgoal() from EvolutionLoop"
git push origin main
```
