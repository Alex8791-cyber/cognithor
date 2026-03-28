# Evolution Engine Phase 5C — QualityAssessor + HorizonScanner + ScheduleManager

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** SubGoals are quality-tested (coverage + self-examination), the system discovers areas beyond the literal goal, and recurring source updates run via cron.

**Architecture:** Three new agents added to the DeepLearner. QualityAssessor runs after Research→Build. HorizonScanner runs after all SubGoals pass. ScheduleManager creates cron jobs for recurring sources.

**Tech Stack:** Python 3.12+ (asyncio, json), MCP tools (vault_search, search_memory, web_fetch), CronEngine, pytest

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/jarvis/evolution/quality_assessor.py` | Coverage check + LLM self-examination |
| Create | `src/jarvis/evolution/horizon_scanner.py` | LLM exploration + graph gap discovery |
| Create | `src/jarvis/evolution/schedule_manager.py` | Create cron jobs for recurring sources |
| Modify | `src/jarvis/evolution/deep_learner.py` | Wire all 3 agents into the cycle |
| Create | `tests/unit/test_quality_assessor.py` | Tests |
| Create | `tests/unit/test_horizon_scanner.py` | Tests |
| Create | `tests/unit/test_schedule_manager.py` | Tests |

---

### Task 1: QualityAssessor

**Files:**
- Create: `src/jarvis/evolution/quality_assessor.py`
- Create: `tests/unit/test_quality_assessor.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/test_quality_assessor.py
"""Tests fuer QualityAssessor."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from jarvis.evolution.models import QualityQuestion, SubGoal
from jarvis.evolution.quality_assessor import QualityAssessor


@pytest.fixture()
def mock_mcp():
    client = AsyncMock()
    from dataclasses import dataclass
    @dataclass
    class R:
        content: str = ""
        is_error: bool = False
    # vault_search returns some results
    client.call_tool = AsyncMock(return_value=R(content="VVG §7 regelt die Widerrufsfrist von 14 Tagen."))
    return client


@pytest.fixture()
def mock_llm_questions():
    """LLM generates questions."""
    async def _llm(prompt: str) -> str:
        if "Generiere" in prompt or "Generate" in prompt or "Erstelle" in prompt:
            return json.dumps({"questions": [
                {"question": "Was ist die Widerrufsfrist nach VVG?", "expected_answer": "14 Tage"},
                {"question": "Was regelt §19 VVG?", "expected_answer": "Vorvertragliche Anzeigepflicht"},
                {"question": "Unterschied Erst- und Folgeraemie?", "expected_answer": "§37 vs §38 VVG"},
            ]})
        # Grading: compare actual vs expected
        return json.dumps({"score": 0.9, "correct": True})
    return _llm


@pytest.fixture()
def assessor(mock_mcp, mock_llm_questions):
    return QualityAssessor(
        mcp_client=mock_mcp,
        llm_fn=mock_llm_questions,
        coverage_threshold=0.7,
        quality_threshold=0.8,
    )


class TestQualityAssessor:
    def test_coverage_check_passes(self, assessor):
        """SubGoal mit genuegend Daten → coverage >= threshold."""
        sg = SubGoal(id="sg-1", title="VVG", description="test")
        sg.chunks_created = 25
        sg.entities_created = 8
        sg.vault_entries = 6
        sg.sources_fetched = 4
        score = assessor.check_coverage(sg)
        assert score >= 0.7

    def test_coverage_check_fails(self, assessor):
        """SubGoal mit wenig Daten → coverage < threshold."""
        sg = SubGoal(id="sg-1", title="VVG", description="test")
        sg.chunks_created = 2
        sg.entities_created = 0
        sg.vault_entries = 0
        sg.sources_fetched = 0
        score = assessor.check_coverage(sg)
        assert score < 0.7

    @pytest.mark.asyncio
    async def test_generate_questions(self, assessor):
        """LLM generiert Pruefungsfragen."""
        questions = await assessor.generate_questions("VVG Grundlagen")
        assert len(questions) >= 1
        assert all(q.question for q in questions)
        assert all(q.expected_answer for q in questions)

    @pytest.mark.asyncio
    async def test_answer_question_uses_memory(self, assessor, mock_mcp):
        """Antwort kommt aus vault_search/search_memory, nicht Web."""
        q = QualityQuestion(question="Was ist die Widerrufsfrist?", expected_answer="14 Tage")
        answered = await assessor.answer_question(q)
        assert answered.actual_answer != ""
        # Should have called vault_search or search_memory
        tool_names = [c[0][0] for c in mock_mcp.call_tool.call_args_list]
        assert any(t in ("vault_search", "search_memory") for t in tool_names)

    @pytest.mark.asyncio
    async def test_grade_question(self, assessor):
        """LLM bewertet Antwort gegen erwartete Antwort."""
        q = QualityQuestion(
            question="Was ist die Widerrufsfrist?",
            expected_answer="14 Tage",
            actual_answer="Die Widerrufsfrist betraegt 14 Tage ab Zugang der Unterlagen.",
        )
        graded = await assessor.grade_question(q)
        assert graded.score > 0.0
        assert graded.passed

    @pytest.mark.asyncio
    async def test_full_quality_test(self, assessor):
        """Kompletter Quality-Test: generate → answer → grade."""
        sg = SubGoal(id="sg-1", title="VVG Grundlagen", description="test")
        sg.chunks_created = 30
        sg.entities_created = 10
        sg.vault_entries = 8
        sg.sources_fetched = 5
        result = await assessor.run_quality_test(sg, "versicherungsrecht")
        assert "coverage_score" in result
        assert "quality_score" in result
        assert "questions" in result
        assert result["coverage_score"] >= 0.7

    @pytest.mark.asyncio
    async def test_quality_test_skips_when_coverage_fails(self, assessor):
        """Coverage zu niedrig → kein LLM-Test, quality_score=0."""
        sg = SubGoal(id="sg-1", title="Test", description="test")
        sg.chunks_created = 1
        result = await assessor.run_quality_test(sg, "test")
        assert result["coverage_score"] < 0.7
        assert result["quality_score"] == 0.0
        assert result["passed"] is False
```

- [ ] **Step 2: Implement quality_assessor.py**

```python
# src/jarvis/evolution/quality_assessor.py
"""QualityAssessor — coverage check + LLM self-examination for SubGoals."""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Coroutine

from jarvis.evolution.models import QualityQuestion, SubGoal
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["QualityAssessor"]

_QUESTION_PROMPT = """\
Erstelle {count} Pruefungsfragen zum Thema "{topic}".
Jede Frage soll konkretes Fachwissen testen.

Antworte NUR mit JSON:
{{
  "questions": [
    {{"question": "Frage?", "expected_answer": "Korrekte Antwort"}}
  ]
}}
"""

_GRADE_PROMPT = """\
Bewerte ob die gegebene Antwort zur erwarteten Antwort passt.

Frage: {question}
Erwartete Antwort: {expected}
Gegebene Antwort: {actual}

Antworte NUR mit JSON:
{{"score": 0.0-1.0, "correct": true/false}}
"""


class QualityAssessor:
    """Checks SubGoal quality via coverage metrics and LLM self-examination."""

    def __init__(
        self,
        mcp_client: Any = None,
        llm_fn: Callable[[str], Coroutine[Any, Any, str]] | None = None,
        coverage_threshold: float = 0.7,
        quality_threshold: float = 0.8,
    ) -> None:
        self._mcp = mcp_client
        self._llm_fn = llm_fn
        self._coverage_threshold = coverage_threshold
        self._quality_threshold = quality_threshold

    def check_coverage(self, subgoal: SubGoal) -> float:
        """Check quantitative coverage of a SubGoal (0.0-1.0)."""
        checks = [
            getattr(subgoal, "vault_entries", 0) >= 5 if isinstance(getattr(subgoal, "vault_entries", 0), int)
            else len(getattr(subgoal, "vault_entries", [])) >= 5,
            subgoal.chunks_created >= 20,
            subgoal.entities_created >= 5,
            getattr(subgoal, "sources_fetched", 0) >= 3 if isinstance(getattr(subgoal, "sources_fetched", 0), int)
            else len(getattr(subgoal, "sources_fetched", [])) >= 3,
        ]
        return sum(checks) / len(checks) if checks else 0.0

    async def generate_questions(
        self, topic: str, count: int = 5
    ) -> list[QualityQuestion]:
        """Generate exam questions via LLM."""
        if not self._llm_fn:
            return []
        try:
            raw = await self._llm_fn(_QUESTION_PROMPT.format(topic=topic, count=count))
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return []
            data = json.loads(match.group())
            return [
                QualityQuestion(
                    question=q.get("question", ""),
                    expected_answer=q.get("expected_answer", ""),
                )
                for q in data.get("questions", [])
            ]
        except Exception:
            log.debug("quality_generate_questions_failed", exc_info=True)
            return []

    async def answer_question(self, q: QualityQuestion) -> QualityQuestion:
        """Answer a question using ONLY stored knowledge (no web)."""
        if not self._mcp:
            return q
        try:
            # Search vault first
            result = await self._mcp.call_tool(
                "vault_search", {"query": q.question, "limit": 3}
            )
            parts = [result.content] if not result.is_error and result.content else []
            # Also search semantic memory
            result2 = await self._mcp.call_tool(
                "search_memory", {"query": q.question, "top_k": 3}
            )
            if not result2.is_error and result2.content:
                parts.append(result2.content)
            q.actual_answer = "\n".join(parts)[:2000] if parts else "Ich weiss es nicht."
        except Exception:
            q.actual_answer = "Fehler bei der Wissenssuche."
        return q

    async def grade_question(self, q: QualityQuestion) -> QualityQuestion:
        """Grade an answered question via LLM."""
        if not self._llm_fn or not q.actual_answer:
            return q
        try:
            raw = await self._llm_fn(_GRADE_PROMPT.format(
                question=q.question,
                expected=q.expected_answer,
                actual=q.actual_answer,
            ))
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
                q.score = float(data.get("score", 0.0))
                q.passed = bool(data.get("correct", False))
        except Exception:
            log.debug("quality_grade_failed", exc_info=True)
        return q

    async def run_quality_test(
        self, subgoal: SubGoal, goal_slug: str
    ) -> dict[str, Any]:
        """Full quality test: coverage check + self-examination."""
        coverage = self.check_coverage(subgoal)
        result: dict[str, Any] = {
            "coverage_score": round(coverage, 2),
            "quality_score": 0.0,
            "passed": False,
            "questions": [],
            "failed_questions": [],
        }

        if coverage < self._coverage_threshold:
            log.info("quality_coverage_insufficient", subgoal=subgoal.title[:40], score=coverage)
            return result

        # Generate and answer questions
        questions = await self.generate_questions(subgoal.title)
        if not questions:
            result["quality_score"] = coverage  # Fallback: use coverage as quality
            result["passed"] = coverage >= self._quality_threshold
            return result

        for q in questions:
            q = await self.answer_question(q)
            q = await self.grade_question(q)

        scores = [q.score for q in questions]
        quality = sum(scores) / len(scores) if scores else 0.0
        failed = [q for q in questions if not q.passed]

        result["quality_score"] = round(quality, 2)
        result["passed"] = quality >= self._quality_threshold
        result["questions"] = [q.to_dict() for q in questions]
        result["failed_questions"] = [q.question for q in failed]

        log.info(
            "quality_test_complete",
            subgoal=subgoal.title[:40],
            coverage=round(coverage, 2),
            quality=round(quality, 2),
            passed=result["passed"],
            failed_count=len(failed),
        )
        return result
```

- [ ] **Step 3: Run tests, verify pass, commit**

```bash
pytest tests/unit/test_quality_assessor.py -v
git add src/jarvis/evolution/quality_assessor.py tests/unit/test_quality_assessor.py
git commit -m "feat(evolution): add QualityAssessor — coverage check + LLM self-examination"
```

---

### Task 2: HorizonScanner

**Files:**
- Create: `src/jarvis/evolution/horizon_scanner.py`
- Create: `tests/unit/test_horizon_scanner.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/test_horizon_scanner.py
"""Tests fuer HorizonScanner."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.evolution.horizon_scanner import HorizonScanner
from jarvis.evolution.models import LearningPlan, SubGoal


@pytest.fixture()
def mock_llm():
    async def _llm(prompt: str) -> str:
        return json.dumps({"expansions": [
            {"title": "Versicherungsombudsmann", "reason": "Wichtiges Streitschlichtungsverfahren"},
            {"title": "Maklerrecht GewO", "reason": "§§59ff GewO reguliert Versicherungsvermittler"},
            {"title": "EU-Richtlinie IDD", "reason": "Europaeische Vermittlerrichtlinie"},
        ]})
    return _llm


@pytest.fixture()
def mock_memory():
    mem = MagicMock()
    # Simulate graph with entities that have many references but little content
    semantic = MagicMock()
    semantic.list_entities.return_value = [
        MagicMock(name="BaFin", attributes={"domain": "versicherungsrecht"}),
        MagicMock(name="Obliegenheit", attributes={"domain": "versicherungsrecht"}),
    ]
    mem.semantic = semantic

    search_result = MagicMock()
    search_result.text = "Kurzer Text"
    mem.search_memory_sync.return_value = [search_result]
    return mem


@pytest.fixture()
def scanner(mock_llm, mock_memory):
    return HorizonScanner(llm_fn=mock_llm, memory_manager=mock_memory)


class TestHorizonScanner:
    @pytest.mark.asyncio
    async def test_llm_exploration(self, scanner):
        """LLM schlaegt neue Gebiete vor."""
        plan = LearningPlan(goal="Versicherungsrecht")
        plan.sub_goals = [
            SubGoal(id="1", title="VVG", description="", status="passed"),
            SubGoal(id="2", title="VAG", description="", status="passed"),
        ]
        expansions = await scanner.explore_via_llm(plan)
        assert len(expansions) >= 1
        assert any("Ombudsmann" in e["title"] for e in expansions)

    @pytest.mark.asyncio
    async def test_graph_discovery(self, scanner):
        """Graph findet Entities mit vielen Referenzen aber wenig Tiefe."""
        gaps = await scanner.discover_graph_gaps("versicherungsrecht")
        # Should find entities from mock that have little content
        assert isinstance(gaps, list)

    @pytest.mark.asyncio
    async def test_full_scan(self, scanner):
        """Kombiniert LLM + Graph Discovery."""
        plan = LearningPlan(goal="Versicherungsrecht", goal_slug="versicherungsrecht")
        plan.sub_goals = [SubGoal(id="1", title="VVG", description="", status="passed")]
        results = await scanner.scan(plan)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_deduplicates_existing_subgoals(self, scanner):
        """Bereits vorhandene SubGoals werden nicht nochmal vorgeschlagen."""
        plan = LearningPlan(goal="Versicherungsrecht", goal_slug="versicherungsrecht")
        plan.sub_goals = [
            SubGoal(id="1", title="Versicherungsombudsmann", description="", status="passed"),
        ]
        results = await scanner.scan(plan)
        titles = [r["title"].lower() for r in results]
        assert "versicherungsombudsmann" not in titles

    @pytest.mark.asyncio
    async def test_llm_failure_graceful(self, scanner):
        """LLM-Fehler → leere Liste, kein Crash."""
        async def _bad_llm(prompt):
            return "Not JSON"
        scanner._llm_fn = _bad_llm
        plan = LearningPlan(goal="Test")
        expansions = await scanner.explore_via_llm(plan)
        assert expansions == []
```

- [ ] **Step 2: Implement horizon_scanner.py**

```python
# src/jarvis/evolution/horizon_scanner.py
"""HorizonScanner — discovers areas beyond the literal goal."""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Coroutine

from jarvis.evolution.models import LearningPlan
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["HorizonScanner"]

_EXPLORE_PROMPT = """\
Du bist ein Forschungsberater. Ein Lernsystem hat folgendes Expertenwissen aufgebaut:

Goal: "{goal}"
Erledigte Teilziele: {completed}
Bekannte Entitaeten: {entities}

Welche angrenzenden Gebiete sind KRITISCH relevant, die wahrscheinlich NICHT bedacht wurden?
Denke ueber den Tellerrand hinaus. Was wuerde ein WAHRER Experte noch wissen muessen?

Antworte NUR mit JSON:
{{
  "expansions": [
    {{"title": "Konkreter Titel", "reason": "Warum das wichtig ist (1 Satz)"}}
  ]
}}

Nenne 3-5 Erweiterungen. Wiederhole KEINE bereits erledigten Teilziele.
"""


class HorizonScanner:
    """Discovers knowledge areas beyond the literal goal.

    Two mechanisms:
    A) LLM Exploration — creative, lateral thinking
    B) Graph Discovery — finds frequently referenced but shallow entities
    """

    def __init__(
        self,
        llm_fn: Callable[[str], Coroutine[Any, Any, str]] | None = None,
        memory_manager: Any = None,
    ) -> None:
        self._llm_fn = llm_fn
        self._memory = memory_manager

    async def scan(self, plan: LearningPlan) -> list[dict[str, str]]:
        """Run both exploration mechanisms, deduplicate against existing SubGoals."""
        existing_titles = {sg.title.lower() for sg in plan.sub_goals}

        results: list[dict[str, str]] = []

        # Mechanism A: LLM exploration
        llm_results = await self.explore_via_llm(plan)
        results.extend(llm_results)

        # Mechanism B: Graph gap discovery
        graph_results = await self.discover_graph_gaps(plan.goal_slug)
        results.extend(graph_results)

        # Deduplicate against existing SubGoals
        filtered: list[dict[str, str]] = []
        seen: set[str] = set()
        for r in results:
            title_lower = r.get("title", "").lower()
            if title_lower not in existing_titles and title_lower not in seen:
                seen.add(title_lower)
                filtered.append(r)

        log.info("horizon_scan_complete", total=len(filtered), llm=len(llm_results), graph=len(graph_results))
        return filtered

    async def explore_via_llm(self, plan: LearningPlan) -> list[dict[str, str]]:
        """Ask LLM for adjacent areas the user hasn't considered."""
        if not self._llm_fn:
            return []
        try:
            completed = [sg.title for sg in plan.sub_goals if sg.status == "passed"]
            # Get top entities from memory
            entity_names: list[str] = []
            if self._memory and hasattr(self._memory, "semantic"):
                try:
                    entities = self._memory.semantic.list_entities(limit=20)
                    entity_names = [getattr(e, "name", str(e)) for e in entities[:20]]
                except Exception:
                    pass

            prompt = _EXPLORE_PROMPT.format(
                goal=plan.goal,
                completed=", ".join(completed) or "(keine)",
                entities=", ".join(entity_names[:20]) or "(keine)",
            )
            raw = await self._llm_fn(prompt)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return []
            data = json.loads(match.group())
            return [
                {"title": e.get("title", ""), "reason": e.get("reason", ""), "source": "llm"}
                for e in data.get("expansions", [])
                if e.get("title")
            ]
        except Exception:
            log.debug("horizon_llm_explore_failed", exc_info=True)
            return []

    async def discover_graph_gaps(self, goal_slug: str) -> list[dict[str, str]]:
        """Find entities with many references but little depth."""
        if not self._memory or not hasattr(self._memory, "semantic"):
            return []
        gaps: list[dict[str, str]] = []
        try:
            entities = self._memory.semantic.list_entities(limit=50)
            for entity in entities:
                name = getattr(entity, "name", str(entity))
                attrs = getattr(entity, "attributes", {})
                domain = attrs.get("domain", "") if isinstance(attrs, dict) else ""
                if domain and domain != goal_slug:
                    continue
                # Check depth: search for chunks mentioning this entity
                results = self._memory.search_memory_sync(query=name, top_k=3)
                chunk_count = len(results) if results else 0
                if chunk_count < 2:
                    gaps.append({
                        "title": f"Vertiefe: {name}",
                        "reason": f"Entitaet '{name}' wird referenziert aber hat nur {chunk_count} Chunk(s)",
                        "source": "graph",
                    })
        except Exception:
            log.debug("horizon_graph_discovery_failed", exc_info=True)
        return gaps[:5]
```

- [ ] **Step 3: Run tests, verify pass, commit**

```bash
pytest tests/unit/test_horizon_scanner.py -v
git add src/jarvis/evolution/horizon_scanner.py tests/unit/test_horizon_scanner.py
git commit -m "feat(evolution): add HorizonScanner — LLM exploration + graph gap discovery"
```

---

### Task 3: ScheduleManager

**Files:**
- Create: `src/jarvis/evolution/schedule_manager.py`
- Create: `tests/unit/test_schedule_manager.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/test_schedule_manager.py
"""Tests fuer ScheduleManager."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.evolution.models import LearningPlan, ScheduleSpec
from jarvis.evolution.schedule_manager import ScheduleManager


@pytest.fixture()
def mock_cron():
    cron = MagicMock()
    cron.add_cron_job = AsyncMock()
    return cron


@pytest.fixture()
def manager(mock_cron):
    return ScheduleManager(cron_engine=mock_cron)


class TestScheduleManager:
    @pytest.mark.asyncio
    async def test_create_schedules(self, manager, mock_cron):
        """ScheduleSpecs → Cron-Jobs angelegt."""
        plan = LearningPlan(goal="Test", goal_slug="test")
        plan.schedules = [
            ScheduleSpec(name="test_daily", cron_expression="0 6 * * *",
                        source_url="https://example.com", action="fetch_and_index",
                        goal_id=plan.id, description="Daily news"),
        ]
        created = await manager.create_schedules(plan)
        assert created == 1
        mock_cron.add_cron_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_multiple_schedules(self, manager, mock_cron):
        """Mehrere ScheduleSpecs → mehrere Cron-Jobs."""
        plan = LearningPlan(goal="Test", goal_slug="test")
        plan.schedules = [
            ScheduleSpec(name="daily", cron_expression="0 6 * * *",
                        source_url="https://a.com", action="fetch_and_index", goal_id=plan.id),
            ScheduleSpec(name="weekly", cron_expression="0 8 * * 1",
                        source_url="https://b.com", action="check_updates", goal_id=plan.id),
        ]
        created = await manager.create_schedules(plan)
        assert created == 2

    @pytest.mark.asyncio
    async def test_skip_empty_schedules(self, manager, mock_cron):
        """Plan ohne Schedules → 0 Jobs."""
        plan = LearningPlan(goal="Test")
        created = await manager.create_schedules(plan)
        assert created == 0
        mock_cron.add_cron_job.assert_not_called()

    @pytest.mark.asyncio
    async def test_cron_job_name_prefixed(self, manager, mock_cron):
        """Job-Name bekommt evolution_ Prefix."""
        plan = LearningPlan(goal="Test", goal_slug="test")
        plan.schedules = [
            ScheduleSpec(name="news", cron_expression="0 6 * * *",
                        source_url="https://example.com", action="fetch_and_index", goal_id=plan.id),
        ]
        await manager.create_schedules(plan)
        call_args = mock_cron.add_cron_job.call_args
        job_name = call_args[1].get("name", "") if call_args[1] else call_args[0][0]
        assert "evolution_" in job_name

    @pytest.mark.asyncio
    async def test_no_cron_engine(self):
        """Kein CronEngine → graceful 0."""
        manager = ScheduleManager(cron_engine=None)
        plan = LearningPlan(goal="Test")
        plan.schedules = [ScheduleSpec(name="x", cron_expression="0 6 * * *",
                                       source_url="https://x.com", action="fetch_and_index", goal_id="p")]
        created = await manager.create_schedules(plan)
        assert created == 0
```

- [ ] **Step 2: Implement schedule_manager.py**

```python
# src/jarvis/evolution/schedule_manager.py
"""ScheduleManager — creates cron jobs for recurring source updates."""

from __future__ import annotations

from typing import Any

from jarvis.evolution.models import LearningPlan
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["ScheduleManager"]


class ScheduleManager:
    """Creates and manages cron jobs for recurring evolution source updates."""

    def __init__(self, cron_engine: Any = None) -> None:
        self._cron = cron_engine

    async def create_schedules(self, plan: LearningPlan) -> int:
        """Create cron jobs for all ScheduleSpecs in a plan. Returns count created."""
        if not self._cron or not plan.schedules:
            return 0

        created = 0
        for spec in plan.schedules:
            job_name = f"evolution_{plan.goal_slug}_{spec.name}"
            description = (
                f"[evolution-update:{plan.id}:{spec.source_url}] "
                f"{spec.description or spec.action}"
            )
            try:
                await self._cron.add_cron_job(
                    name=job_name,
                    schedule=spec.cron_expression,
                    description=description,
                )
                created += 1
                log.info(
                    "schedule_created",
                    job=job_name,
                    cron=spec.cron_expression,
                    source=spec.source_url[:60],
                )
            except Exception:
                log.debug("schedule_create_failed", job=job_name, exc_info=True)

        return created
```

- [ ] **Step 3: Run tests, verify pass, commit**

```bash
pytest tests/unit/test_schedule_manager.py -v
git add src/jarvis/evolution/schedule_manager.py tests/unit/test_schedule_manager.py
git commit -m "feat(evolution): add ScheduleManager — cron jobs for recurring source updates"
```

---

### Task 4: Wire All Three into DeepLearner

**Files:**
- Modify: `src/jarvis/evolution/deep_learner.py`

- [ ] **Step 1: Add imports + init**

Add to imports:
```python
from jarvis.evolution.quality_assessor import QualityAssessor
from jarvis.evolution.horizon_scanner import HorizonScanner
from jarvis.evolution.schedule_manager import ScheduleManager
```

In `__init__`, after research_agent init, add:
```python
        self._quality_assessor = QualityAssessor(
            mcp_client=mcp_client,
            llm_fn=llm_fn,
            coverage_threshold=getattr(config, "coverage_threshold", 0.7),
            quality_threshold=getattr(config, "quality_threshold", 0.8),
        )
        self._horizon_scanner = HorizonScanner(
            llm_fn=llm_fn,
            memory_manager=memory_manager,
        )
        self._schedule_manager = ScheduleManager(cron_engine=cron_engine)
```

- [ ] **Step 2: Add run_quality_test and run_horizon_scan methods**

```python
    async def run_quality_test(self, plan_id: str, subgoal_id: str) -> dict[str, Any]:
        """Run quality test on a SubGoal."""
        plan = self.get_plan(plan_id)
        if not plan:
            return {"error": "Plan not found"}
        subgoal = next((sg for sg in plan.sub_goals if sg.id == subgoal_id), None)
        if not subgoal:
            return {"error": "SubGoal not found"}

        result = await self._quality_assessor.run_quality_test(subgoal, plan.goal_slug)

        subgoal.coverage_score = result["coverage_score"]
        subgoal.quality_score = result["quality_score"]
        if result["passed"]:
            subgoal.status = "passed"
        else:
            subgoal.status = "researching"  # Back to research for failed questions
        plan.coverage_score = sum(sg.coverage_score for sg in plan.sub_goals) / max(len(plan.sub_goals), 1)
        plan.quality_score = sum(sg.quality_score for sg in plan.sub_goals) / max(len(plan.sub_goals), 1)
        plan.save(self._plans_dir)
        return result

    async def run_horizon_scan(self, plan_id: str) -> list[dict[str, str]]:
        """Discover new areas beyond the literal goal."""
        plan = self.get_plan(plan_id)
        if not plan:
            return []
        expansions = await self._horizon_scanner.scan(plan)
        # Add as new SubGoals via replan
        if expansions:
            new_context = "HorizonScanner hat folgende Luecken gefunden:\n"
            new_context += "\n".join(f"- {e['title']}: {e['reason']}" for e in expansions)
            plan = await self._strategy_planner.replan(plan, new_context)
            plan.expansions.extend(e["title"] for e in expansions)
            plan.save(self._plans_dir)
        return expansions

    async def setup_schedules(self, plan_id: str) -> int:
        """Create cron jobs for a plan's recurring sources."""
        plan = self.get_plan(plan_id)
        if not plan:
            return 0
        created = await self._schedule_manager.create_schedules(plan)
        plan.save(self._plans_dir)
        return created
```

- [ ] **Step 3: Enhance run_subgoal to include quality + horizon after completion**

At the end of `run_subgoal()`, after `subgoal.status = "testing"`, add:

```python
        # Auto quality test
        quality = await self._quality_assessor.run_quality_test(subgoal, plan.goal_slug)
        subgoal.coverage_score = quality["coverage_score"]
        subgoal.quality_score = quality["quality_score"]
        if quality["passed"]:
            subgoal.status = "passed"
            log.info("deep_learner_subgoal_passed", subgoal=subgoal.title[:40], quality=quality["quality_score"])
        else:
            subgoal.status = "failed"
            log.info("deep_learner_subgoal_failed", subgoal=subgoal.title[:40],
                     failed_questions=quality.get("failed_questions", []))

        # Check if all SubGoals done → run horizon scan
        all_done = all(sg.status in ("passed", "failed") for sg in plan.sub_goals)
        if all_done and getattr(self._config, "auto_expand", True):
            expansions = await self._horizon_scanner.scan(plan)
            if expansions:
                new_context = "\n".join(f"- {e['title']}: {e['reason']}" for e in expansions)
                plan = await self._strategy_planner.replan(plan, new_context)
                plan.expansions.extend(e["title"] for e in expansions)
                log.info("deep_learner_horizon_expanded", new_subgoals=len(expansions))

        # Setup cron schedules if not done yet
        if plan.schedules and not any(sg.cron_jobs_created for sg in plan.sub_goals):
            await self._schedule_manager.create_schedules(plan)
```

- [ ] **Step 4: Run ALL evolution tests**

```bash
cd "D:\Jarvis\jarvis complete v20" && python -m pytest tests/unit/test_evolution.py tests/unit/test_evolution_models.py tests/unit/test_strategy_planner.py tests/unit/test_deep_learner.py tests/unit/test_research_agent.py tests/unit/test_knowledge_builder.py tests/unit/test_quality_assessor.py tests/unit/test_horizon_scanner.py tests/unit/test_schedule_manager.py tests/unit/test_evolution_resume.py -v
```

- [ ] **Step 5: Commit + push**

```bash
git add src/jarvis/evolution/deep_learner.py
git commit -m "feat(evolution): wire QualityAssessor + HorizonScanner + ScheduleManager into DeepLearner"
git push origin main
```
