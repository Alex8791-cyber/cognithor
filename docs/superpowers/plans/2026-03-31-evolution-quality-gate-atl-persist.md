# Evolution Quality Gate + ATL Auto-Persist Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two Evolution Engine bugs — garbage content indexing (PDF artifacts) and ATL research results evaporating (0% progress despite 90+ cycles).

**Architecture:** A content quality gate in `KnowledgeBuilder.build()` rejects PDF artifacts and too-short content before the triple-write pipeline. The ATL loop captures `search_and_read` results, synthesizes key findings through the LLM like an intelligent expert taking notes, and persists them through the same `KnowledgeBuilder` pipeline. Both fixes share the quality gate.

**Tech Stack:** Python 3.12+, asyncio, regex, existing MCP tool infrastructure, local qwen LLM.

---

### Task 1: Content Quality Gate — `_is_usable_content`

**Files:**
- Modify: `src/jarvis/evolution/knowledge_builder.py:38-143` (add function after existing constants)
- Test: `tests/unit/test_knowledge_builder.py`

- [ ] **Step 1: Write failing tests for `_is_usable_content`**

Add to `tests/unit/test_knowledge_builder.py`:

```python
class TestContentQualityGate:
    """Tests for _is_usable_content — rejects PDF artifacts and too-short text."""

    def test_rejects_too_short_text(self):
        from jarvis.evolution.knowledge_builder import _is_usable_content

        usable, reason = _is_usable_content("Short.")
        assert usable is False
        assert reason == "too_short"

    def test_rejects_empty_text(self):
        from jarvis.evolution.knowledge_builder import _is_usable_content

        usable, reason = _is_usable_content("")
        assert usable is False
        assert reason == "too_short"

    def test_rejects_whitespace_only(self):
        from jarvis.evolution.knowledge_builder import _is_usable_content

        usable, reason = _is_usable_content("   \n\n\t  \n  ")
        assert usable is False
        assert reason == "too_short"

    def test_rejects_pdf_artifact_text(self):
        from jarvis.evolution.knowledge_builder import _is_usable_content

        pdf_dump = "\n".join([
            "5 0 obj",
            "<< /Type /Page /MediaBox [0 0 612 792] >>",
            "endobj",
            "6 0 obj",
            "<< /Filter /FlateDecode /Length 1528 >>",
            "stream",
            "xref",
            "0 15",
            "trailer",
            "<< /Root 1 0 R /Info 2 0 R >>",
            "%%EOF",
            "Some actual text here.",
        ])
        usable, reason = _is_usable_content(pdf_dump)
        assert usable is False
        assert "pdf_artifacts" in reason

    def test_accepts_real_article(self):
        from jarvis.evolution.knowledge_builder import _is_usable_content

        article = (
            "Das Versicherungsvertragsgesetz (VVG) regelt die Rechtsbeziehungen "
            "zwischen Versicherungsnehmer und Versicherer. Es umfasst allgemeine "
            "Vorschriften ueber den Abschluss und die Durchfuehrung von "
            "Versicherungsvertraegen. Die wichtigsten Paragraphen betreffen "
            "die Anzeigepflicht, das Widerrufsrecht und die Leistungspflicht "
            "des Versicherers bei Eintritt des Versicherungsfalls."
        )
        usable, reason = _is_usable_content(article)
        assert usable is True
        assert reason == "ok"

    def test_accepts_content_at_boundary(self):
        from jarvis.evolution.knowledge_builder import _is_usable_content

        text = "x " * 101  # 202 chars — just above 200 threshold
        usable, reason = _is_usable_content(text)
        assert usable is True

    def test_borderline_garbage_ratio_below_threshold(self):
        from jarvis.evolution.knowledge_builder import _is_usable_content

        # 3 garbage lines + 8 good lines = 27% garbage (below 30%)
        lines = ["endobj", "xref", "trailer"]
        lines += ["Dies ist ein normaler Satz ueber Versicherungsrecht."] * 8
        text = "\n".join(lines)
        usable, reason = _is_usable_content(text)
        assert usable is True

    def test_custom_min_chars(self):
        from jarvis.evolution.knowledge_builder import _is_usable_content

        text = "a " * 60  # 120 chars
        usable_default, _ = _is_usable_content(text)
        usable_low, _ = _is_usable_content(text, min_chars=100)
        assert usable_default is False
        assert usable_low is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_knowledge_builder.py::TestContentQualityGate -v`
Expected: FAIL with `ImportError: cannot import name '_is_usable_content'`

- [ ] **Step 3: Implement `_is_usable_content`**

Add to `src/jarvis/evolution/knowledge_builder.py` after line 143 (`_is_valid_entity` function), before line 146 (`_ENTITY_EXTRACTION_PROMPT`):

```python
# Compiled pattern for detecting PDF structural artifacts in source text.
# Applied per-line: if >30% of lines match, the source is rejected.
_PDF_ARTIFACT_LINE_RE = re.compile(
    r"(\d+ \d+ obj"
    r"|endobj|endstream|xref|trailer"
    r"|stream\s*$"
    r"|/Type\b|/Filter\b|/Length\b|/Pages\b|/Root\b"
    r"|FlateDecode|MediaBox|DeviceRGB|DeviceCMYK"
    r"|%%EOF)",
)


def _is_usable_content(text: str, min_chars: int = 200) -> tuple[bool, str]:
    """Check whether fetched text is worth indexing.

    Returns:
        (usable, reason) — reason is 'ok' on success or a short tag on rejection.
    """
    cleaned = " ".join(text.split())
    if len(cleaned) < min_chars:
        return False, "too_short"

    lines = text.splitlines()
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return False, "too_short"

    garbage_count = sum(1 for l in non_empty if _PDF_ARTIFACT_LINE_RE.search(l))
    ratio = garbage_count / len(non_empty)
    if ratio > 0.3:
        return False, f"pdf_artifacts_{ratio:.0%}"

    return True, "ok"
```

Also add `_is_usable_content` to the module's `__all__` list for test import access. Find line 23:

```python
__all__ = ["BuildResult", "KnowledgeBuilder", "_is_usable_content"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_knowledge_builder.py::TestContentQualityGate -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/jarvis/evolution/knowledge_builder.py tests/unit/test_knowledge_builder.py
git commit -m "feat(evolution): add _is_usable_content quality gate for PDF artifacts"
```

---

### Task 2: Wire Quality Gate into `build()`

**Files:**
- Modify: `src/jarvis/evolution/knowledge_builder.py:220-224` (inside `build()` method)
- Test: `tests/unit/test_knowledge_builder.py`

- [ ] **Step 1: Write failing test for build() rejecting garbage**

Add to `tests/unit/test_knowledge_builder.py`:

```python
class TestBuildRejectsGarbage:
    """build() should skip triple-write when content is unusable."""

    @pytest.mark.asyncio
    async def test_build_skips_pdf_garbage(self):
        from jarvis.evolution.knowledge_builder import BuildResult, KnowledgeBuilder

        mcp = _make_mcp()
        kb = KnowledgeBuilder(mcp_client=mcp, llm_fn=_mock_llm, goal_slug="test")

        pdf_dump = "\n".join([
            "5 0 obj",
            "<< /Type /Page /MediaBox [0 0 612 792] >>",
            "endobj",
            "6 0 obj",
            "<< /Filter /FlateDecode /Length 1528 >>",
            "stream",
            "xref",
            "0 15",
            "trailer",
            "<< /Root 1 0 R /Info 2 0 R >>",
            "%%EOF",
            "Some text.",
        ])
        fr = _make_fetch_result(text=pdf_dump)
        result = await kb.build(fr)

        assert isinstance(result, BuildResult)
        assert result.chunks_created == 0
        assert result.vault_path == ""
        assert len(result.errors) == 1
        assert "Content rejected" in result.errors[0]
        # No MCP calls should have been made
        mcp.call_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_build_skips_too_short(self):
        from jarvis.evolution.knowledge_builder import BuildResult, KnowledgeBuilder

        mcp = _make_mcp()
        kb = KnowledgeBuilder(mcp_client=mcp, goal_slug="test")
        fr = _make_fetch_result(text="Short.")
        result = await kb.build(fr)

        assert result.chunks_created == 0
        assert "Content rejected" in result.errors[0]
        mcp.call_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_build_accepts_good_content(self):
        from jarvis.evolution.knowledge_builder import BuildResult, KnowledgeBuilder

        mcp = _make_mcp()
        kb = KnowledgeBuilder(mcp_client=mcp, llm_fn=_mock_llm, goal_slug="test")
        fr = _make_fetch_result(
            text=(
                "Das Versicherungsvertragsgesetz (VVG) regelt die Rechtsbeziehungen "
                "zwischen Versicherungsnehmer und Versicherer. Es umfasst allgemeine "
                "Vorschriften ueber den Abschluss und die Durchfuehrung von "
                "Versicherungsvertraegen. Die wichtigsten Paragraphen betreffen "
                "die Anzeigepflicht und das Widerrufsrecht."
            )
        )
        result = await kb.build(fr)

        assert result.vault_path != ""
        assert result.chunks_created > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_knowledge_builder.py::TestBuildRejectsGarbage -v`
Expected: FAIL — `test_build_skips_pdf_garbage` and `test_build_skips_too_short` fail because `build()` still processes garbage

- [ ] **Step 3: Add quality gate to `build()`**

In `src/jarvis/evolution/knowledge_builder.py`, modify the `build()` method. After line 224 (the existing `if fetch_result.error or not fetch_result.text` block), add:

```python
        # Content quality gate: reject PDF artifacts and too-short text
        usable, reason = _is_usable_content(fetch_result.text)
        if not usable:
            log.info(
                "content_rejected",
                url=fetch_result.url[:80],
                reason=reason,
            )
            result.errors.append(f"Content rejected: {reason}")
            return result
```

The full block from line 220 should now read:

```python
        result = BuildResult()

        if fetch_result.error or not fetch_result.text:
            result.errors.append(fetch_result.error or "Empty text in FetchResult")
            return result

        # Content quality gate: reject PDF artifacts and too-short text
        usable, reason = _is_usable_content(fetch_result.text)
        if not usable:
            log.info(
                "content_rejected",
                url=fetch_result.url[:80],
                reason=reason,
            )
            result.errors.append(f"Content rejected: {reason}")
            return result

        # 1. Vault save
        try:
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_knowledge_builder.py -v`
Expected: ALL tests pass (including existing tests — verify that `test_build_from_fetch_result` still passes, since its text is >200 chars)

- [ ] **Step 5: Commit**

```bash
git add src/jarvis/evolution/knowledge_builder.py tests/unit/test_knowledge_builder.py
git commit -m "feat(evolution): wire content quality gate into KnowledgeBuilder.build()"
```

---

### Task 3: ATL Goal Matching

**Files:**
- Modify: `src/jarvis/evolution/loop.py` (add `_match_goal_for_action` method)
- Test: `tests/unit/test_evolution.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/unit/test_evolution.py`:

```python
class TestATLGoalMatching:
    """_match_goal_for_action finds the best goal for a research action."""

    def _make_goal(self, title: str, goal_id: str = "") -> Any:
        from dataclasses import dataclass, field

        @dataclass
        class _Goal:
            title: str = ""
            id: str = ""
            progress: float = 0.0
            priority: int = 3

        return _Goal(title=title, id=goal_id or title[:10])

    def test_matches_by_keyword_overlap(self):
        from jarvis.evolution.loop import _match_goal_for_action

        goals = [
            self._make_goal("Werde Experte fuer Cybersecurity und Pentesting"),
            self._make_goal("Werde Experte fuer die deutsche Versicherungswirtschaft"),
        ]
        action = type("A", (), {"rationale": "OWASP Top 10 Cybersecurity Standards", "params": {"query": "OWASP"}})()

        result = _match_goal_for_action(action, goals)
        assert result is not None
        assert "Cybersecurity" in result.title

    def test_matches_query_param_too(self):
        from jarvis.evolution.loop import _match_goal_for_action

        goals = [
            self._make_goal("Werde Experte fuer AI Agent Architektur"),
            self._make_goal("Werde Experte fuer Versicherungsrecht"),
        ]
        action = type("A", (), {"rationale": "Recherche", "params": {"query": "AI Agent Architecture Patterns"}})()

        result = _match_goal_for_action(action, goals)
        assert result is not None
        assert "AI Agent" in result.title

    def test_returns_none_on_no_match(self):
        from jarvis.evolution.loop import _match_goal_for_action

        goals = [
            self._make_goal("Werde Experte fuer Kochen"),
        ]
        action = type("A", (), {"rationale": "quantum physics research", "params": {}})()

        result = _match_goal_for_action(action, goals)
        assert result is None

    def test_explicit_goal_id_in_params(self):
        from jarvis.evolution.loop import _match_goal_for_action

        goals = [
            self._make_goal("Cybersecurity", goal_id="cyber-1"),
            self._make_goal("Versicherung", goal_id="ins-2"),
        ]
        action = type("A", (), {"rationale": "", "params": {"goal_id": "ins-2"}})()

        result = _match_goal_for_action(action, goals)
        assert result is not None
        assert result.id == "ins-2"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_evolution.py::TestATLGoalMatching -v`
Expected: FAIL with `ImportError: cannot import name '_match_goal_for_action'`

- [ ] **Step 3: Implement `_match_goal_for_action`**

Add as a module-level function in `src/jarvis/evolution/loop.py`, after the imports (around line 20, after `log = get_logger(__name__)`):

```python
def _match_goal_for_action(action: Any, goals: list) -> Any | None:
    """Find the goal most relevant to an ATL action.

    Uses explicit goal_id from params if available, otherwise
    word-overlap between action text and goal titles.
    Returns None if no confident match (< 2 word overlap).
    """
    # 1. Explicit goal_id in params
    goal_id = getattr(action, "params", {}).get("goal_id", "")
    if goal_id:
        for g in goals:
            if g.id == goal_id:
                return g

    # 2. Word-overlap heuristic
    action_text = (
        f"{getattr(action, 'rationale', '')} "
        f"{getattr(action, 'params', {}).get('query', '')}"
    ).lower()
    action_words = set(action_text.split())

    best, best_score = None, 0
    for g in goals:
        goal_words = set(g.title.lower().split())
        overlap = len(goal_words & action_words)
        if overlap > best_score:
            best, best_score = g, overlap

    return best if best_score >= 2 else None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_evolution.py::TestATLGoalMatching -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/jarvis/evolution/loop.py tests/unit/test_evolution.py
git commit -m "feat(evolution): add _match_goal_for_action for ATL goal routing"
```

---

### Task 4: ATL Synthesis — Intelligent Note-Taking

**Files:**
- Modify: `src/jarvis/evolution/loop.py` (add `_synthesize_for_goal` method to `EvolutionLoop`)
- Test: `tests/unit/test_evolution.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/unit/test_evolution.py`:

```python
class TestATLSynthesis:
    """_synthesize_for_goal extracts relevant findings like an expert."""

    @pytest.mark.asyncio
    async def test_synthesis_returns_structured_note(self):
        from jarvis.evolution.loop import EvolutionLoop

        async def mock_llm(prompt: str) -> str:
            return (
                "## OWASP Top 10 Updates\n"
                "- SQL Injection bleibt auf Platz 1 (Quelle: owasp.org)\n"
                "- Neue Kategorie: Server-Side Request Forgery\n"
            )

        loop = EvolutionLoop.__new__(EvolutionLoop)
        loop._llm_fn = mock_llm

        result = await loop._synthesize_for_goal(
            research_text="OWASP has updated their top 10 list...",
            goal_title="Werde Experte fuer Cybersecurity",
            query="OWASP Top 10 2024",
        )

        assert result is not None
        assert "OWASP" in result
        assert "##" in result

    @pytest.mark.asyncio
    async def test_synthesis_returns_none_for_irrelevant(self):
        from jarvis.evolution.loop import EvolutionLoop

        async def mock_llm(prompt: str) -> str:
            return "KEINE_RELEVANZ"

        loop = EvolutionLoop.__new__(EvolutionLoop)
        loop._llm_fn = mock_llm

        result = await loop._synthesize_for_goal(
            research_text="This page is about cooking recipes...",
            goal_title="Werde Experte fuer Cybersecurity",
            query="OWASP Top 10",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_synthesis_returns_none_on_llm_error(self):
        from jarvis.evolution.loop import EvolutionLoop

        async def mock_llm(prompt: str) -> str:
            raise RuntimeError("LLM timeout")

        loop = EvolutionLoop.__new__(EvolutionLoop)
        loop._llm_fn = mock_llm

        result = await loop._synthesize_for_goal(
            research_text="Some valid text about security...",
            goal_title="Cybersecurity",
            query="test",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_synthesis_without_llm_returns_none(self):
        from jarvis.evolution.loop import EvolutionLoop

        loop = EvolutionLoop.__new__(EvolutionLoop)
        loop._llm_fn = None

        result = await loop._synthesize_for_goal(
            research_text="Some text...",
            goal_title="Test",
            query="test",
        )

        assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_evolution.py::TestATLSynthesis -v`
Expected: FAIL with `AttributeError: ... has no attribute '_synthesize_for_goal'`

- [ ] **Step 3: Implement `_synthesize_for_goal`**

Add as a method on the `EvolutionLoop` class in `src/jarvis/evolution/loop.py`. Place it after the `__init__` method (after line 107):

```python
    async def _synthesize_for_goal(
        self,
        research_text: str,
        goal_title: str,
        query: str,
    ) -> str | None:
        """Synthesize research findings into a structured note.

        Cognithor processes information like an expert: extracting
        what matters, connecting it to the goal, and discarding noise.
        Returns None if nothing relevant was found or on error.
        """
        if not self._llm_fn:
            return None

        prompt = (
            "Du bist ein Wissensassistent der Recherche-Ergebnisse einordnet.\n\n"
            f"Ziel: {goal_title}\n"
            f"Suchanfrage: {query}\n\n"
            f"Recherche-Ergebnis:\n{research_text[:3000]}\n\n"
            "Aufgabe:\n"
            "1. Extrahiere die 3-5 wichtigsten Fakten die fuer das Ziel relevant sind\n"
            "2. Formuliere eine strukturierte Notiz (deutsch, sachlich)\n"
            "3. Wenn nichts Relevantes gefunden wurde, antworte NUR mit KEINE_RELEVANZ\n\n"
            "Format:\n"
            "## {Thema}\n"
            "- Kernaussage 1 (Quelle: ...)\n"
            "- Kernaussage 2\n"
            "...\n"
        )

        try:
            response = await self._llm_fn(prompt)
            if not response or "KEINE_RELEVANZ" in response:
                return None
            return response.strip()
        except Exception:
            log.debug("atl_synthesis_failed", exc_info=True)
            return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_evolution.py::TestATLSynthesis -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/jarvis/evolution/loop.py tests/unit/test_evolution.py
git commit -m "feat(evolution): add _synthesize_for_goal for intelligent ATL note-taking"
```

---

### Task 5: Wire Auto-Persist into ATL Action Dispatch

**Files:**
- Modify: `src/jarvis/evolution/loop.py:384-436` (action dispatch block in `thinking_cycle`)
- Test: `tests/unit/test_evolution.py`

- [ ] **Step 1: Write failing integration test**

Add to `tests/unit/test_evolution.py`:

```python
class TestATLAutoPersist:
    """ATL auto-persists search_and_read results through KnowledgeBuilder."""

    @pytest.mark.asyncio
    async def test_research_result_persisted(self):
        """search_and_read result flows through synthesis → KnowledgeBuilder."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from jarvis.evolution.loop import EvolutionLoop

        # Build a minimal EvolutionLoop with mocked dependencies
        loop = EvolutionLoop.__new__(EvolutionLoop)
        loop._llm_fn = AsyncMock(return_value=(
            "## ARC-AGI Strategien\n"
            "- CNN+Graph Ansaetze dominieren (Quelle: kaggle.com)\n"
            "- Stochastic Goose erreicht 45% Accuracy\n"
        ))
        loop._mcp_client = AsyncMock()
        loop._mcp_client.call_tool = AsyncMock(return_value="Long research text about ARC-AGI strategies " * 20)
        loop._deep_learner = MagicMock()
        loop._deep_learner._plans_dir = MagicMock()
        loop._deep_learner._plans_dir.parent = MagicMock()

        # Mock goal
        goal = MagicMock()
        goal.title = "Werde Experte fuer ARC-AGI-3 Strategien"
        goal.id = "arc-agi"

        # Track KnowledgeBuilder.build calls
        mock_builder = AsyncMock()
        mock_build_result = MagicMock()
        mock_build_result.errors = []
        mock_build_result.chunks_created = 3
        mock_builder.build = AsyncMock(return_value=mock_build_result)

        loop._atl_knowledge_builders = {"arc-agi": mock_builder}
        loop._atl_persisted_queries = set()

        # Execute the persist logic
        action = MagicMock()
        action.type = "research"
        action.rationale = "ARC-AGI-3 winning strategies"
        action.params = {"query": "ARC-AGI-3 strategies"}

        await loop._persist_research_result(
            tool_result="Long research text about ARC-AGI strategies " * 20,
            action=action,
            goals=[goal],
        )

        # Verify synthesis was called
        assert loop._llm_fn.call_count == 1
        # Verify KnowledgeBuilder.build was called
        mock_builder.build.assert_called_once()
        call_args = mock_builder.build.call_args
        fetch_result = call_args[0][0]
        assert "ARC-AGI" in fetch_result.text  # Synthesized text
        assert fetch_result.source_type == "atl_research"

    @pytest.mark.asyncio
    async def test_dedup_prevents_double_persist(self):
        from unittest.mock import AsyncMock, MagicMock

        from jarvis.evolution.loop import EvolutionLoop

        loop = EvolutionLoop.__new__(EvolutionLoop)
        loop._llm_fn = AsyncMock(return_value="## Note\n- Fact 1\n")
        loop._atl_persisted_queries = {"arc-agi-3 strategies"}  # Already persisted

        goal = MagicMock()
        goal.title = "ARC-AGI-3 Strategien"
        goal.id = "arc"

        mock_builder = AsyncMock()
        loop._atl_knowledge_builders = {"arc": mock_builder}

        action = MagicMock()
        action.rationale = "ARC strategies"
        action.params = {"query": "ARC-AGI-3 strategies"}

        await loop._persist_research_result(
            tool_result="some text " * 50,
            action=action,
            goals=[goal],
        )

        # LLM should not be called — dedup hit
        loop._llm_fn.assert_not_called()
        mock_builder.build.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_persist_when_synthesis_returns_none(self):
        from unittest.mock import AsyncMock, MagicMock

        from jarvis.evolution.loop import EvolutionLoop

        loop = EvolutionLoop.__new__(EvolutionLoop)
        loop._llm_fn = AsyncMock(return_value="KEINE_RELEVANZ")
        loop._mcp_client = AsyncMock()
        loop._atl_persisted_queries = set()

        goal = MagicMock()
        goal.title = "Cybersecurity Experte"
        goal.id = "cyber"

        mock_builder = AsyncMock()
        loop._atl_knowledge_builders = {"cyber": mock_builder}

        action = MagicMock()
        action.rationale = "Cybersecurity research"
        action.params = {"query": "irrelevant cooking recipes"}

        await loop._persist_research_result(
            tool_result="Recipe for chocolate cake " * 50,
            action=action,
            goals=[goal],
        )

        # Synthesis returned None → no build
        mock_builder.build.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_evolution.py::TestATLAutoPersist -v`
Expected: FAIL with `AttributeError: ... has no attribute '_persist_research_result'`

- [ ] **Step 3: Implement `_persist_research_result` and wire into dispatch**

Add two new instance attributes in `__init__` (after line 106 in `loop.py`):

```python
        self._atl_knowledge_builders: dict[str, Any] = {}  # goal_id → KnowledgeBuilder
        self._atl_persisted_queries: set[str] = set()
```

Add `_persist_research_result` method on `EvolutionLoop` (after `_synthesize_for_goal`):

```python
    async def _persist_research_result(
        self,
        tool_result: Any,
        action: Any,
        goals: list,
    ) -> None:
        """Persist a search_and_read result: match goal, dedup, synthesize, build.

        This is the core of Cognithor's intelligent learning during idle time.
        Rather than dumping raw web scrapes, it synthesizes findings like an
        expert taking structured notes.
        """
        result_text = str(tool_result)
        if len(result_text) < 200:
            return

        # Goal matching
        goal = _match_goal_for_action(action, goals)
        if not goal:
            log.debug("atl_persist_no_goal_match", rationale=getattr(action, "rationale", "")[:60])
            return

        # Dedup: skip if this query was already persisted this session
        query_key = getattr(action, "params", {}).get("query", "")[:100].lower().strip()
        if query_key in self._atl_persisted_queries:
            log.debug("atl_persist_dedup_skip", query=query_key[:60])
            return

        # Synthesis: extract relevant findings
        synthesis = await self._synthesize_for_goal(
            research_text=result_text,
            goal_title=goal.title,
            query=query_key,
        )
        if not synthesis:
            log.debug("atl_persist_no_relevance", goal=goal.title[:40], query=query_key[:40])
            return

        # Get or create KnowledgeBuilder for this goal
        builder = self._atl_knowledge_builders.get(goal.id)
        if not builder:
            builder = self._create_builder_for_goal(goal)
            if not builder:
                return
            self._atl_knowledge_builders[goal.id] = builder

        # Build knowledge from synthesized text
        from jarvis.evolution.research_agent import FetchResult

        fetch = FetchResult(
            url=query_key or "atl-research",
            text=synthesis,
            title=f"ATL: {getattr(action, 'rationale', '')[:80]}",
            source_type="atl_research",
        )
        try:
            build_result = await builder.build(fetch, skip_entity_extraction=True)
            if build_result.errors:
                log.debug("atl_persist_build_errors", errors=build_result.errors[:2])
            else:
                self._atl_persisted_queries.add(query_key)
                log.info(
                    "atl_research_persisted",
                    goal=goal.title[:40],
                    chunks=build_result.chunks_created,
                    query=query_key[:40],
                )
        except Exception:
            log.debug("atl_persist_build_failed", exc_info=True)
```

Add `_create_builder_for_goal` helper method:

```python
    def _create_builder_for_goal(self, goal: Any) -> Any | None:
        """Lazily create a KnowledgeBuilder for an ATL goal."""
        if not self._deep_learner or not self._mcp_client:
            return None
        try:
            from jarvis.evolution.goal_index import GoalScopedIndex
            from jarvis.evolution.knowledge_builder import KnowledgeBuilder

            # Use same index base as deep_learner
            index_base = self._deep_learner._plans_dir.parent / "indexes"
            goal_slug = goal.id
            goal_index = GoalScopedIndex(goal_slug=goal_slug, base_dir=index_base)

            return KnowledgeBuilder(
                mcp_client=self._mcp_client,
                llm_fn=self._llm_fn,
                goal_slug=goal_slug,
                goal_index=goal_index,
            )
        except Exception:
            log.debug("atl_builder_creation_failed", exc_info=True)
            return None
```

Now wire `_persist_research_result` into the action dispatch block. In `thinking_cycle()`, modify the dispatch block (around line 430-432). Change:

```python
                try:
                    await self._mcp_client.call_tool(tool_name, params)
                    executed_actions.append(f"[OK] {action.type}: {action.rationale[:60]}")
                    log.info("atl_action_executed", type=action.type, tool=tool_name)
```

To:

```python
                try:
                    tool_result = await self._mcp_client.call_tool(tool_name, params)
                    executed_actions.append(f"[OK] {action.type}: {action.rationale[:60]}")
                    log.info("atl_action_executed", type=action.type, tool=tool_name)

                    # Auto-persist: research results → synthesis → KnowledgeBuilder
                    if tool_name == "search_and_read" and tool_result:
                        try:
                            await self._persist_research_result(tool_result, action, goals)
                        except Exception:
                            log.debug("atl_auto_persist_failed", exc_info=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_evolution.py::TestATLAutoPersist -v`
Expected: All 3 tests PASS

Then run all evolution tests:

Run: `pytest tests/unit/test_evolution.py tests/unit/test_knowledge_builder.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/jarvis/evolution/loop.py tests/unit/test_evolution.py
git commit -m "feat(evolution): ATL auto-persist with intelligent synthesis

Research results from search_and_read are now automatically:
1. Matched to the relevant goal
2. Deduplicated by query
3. Synthesized via LLM into structured expert notes
4. Persisted through KnowledgeBuilder (vault + chunks + goal index)

Cognithor now learns during idle time like an expert taking notes,
not a web crawler dumping raw HTML."
```

---

### Task 6: Run Full Test Suite and Push

**Files:** None (verification only)

- [ ] **Step 1: Run ruff format**

Run: `python -m ruff format src/jarvis/evolution/knowledge_builder.py src/jarvis/evolution/loop.py`

- [ ] **Step 2: Run ruff check**

Run: `python -m ruff check src/jarvis/evolution/knowledge_builder.py src/jarvis/evolution/loop.py`

Fix any new lint issues (pre-existing issues are acceptable).

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/unit/test_knowledge_builder.py tests/unit/test_evolution.py tests/test_evolution_orchestrator.py -v --tb=short`

Expected: All tests PASS

- [ ] **Step 4: Run broader tests to check for regressions**

Run: `pytest tests/ -x -q --tb=short --ignore=tests/test_channels/test_voice_ws_bridge.py -k "not test_arc"`

Expected: No regressions from our changes

- [ ] **Step 5: Push to GitHub**

```bash
git push origin main
```
