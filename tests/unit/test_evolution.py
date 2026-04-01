"""Tests for Evolution Engine — IdleDetector + EvolutionLoop."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestIdleDetector:
    def test_not_idle_initially(self):
        from jarvis.evolution.idle_detector import IdleDetector

        d = IdleDetector(idle_threshold_seconds=5)
        assert d.is_idle is False

    def test_idle_after_threshold(self):
        from jarvis.evolution.idle_detector import IdleDetector

        d = IdleDetector(idle_threshold_seconds=0)
        # Threshold 0 = immediately idle
        assert d.is_idle is True

    def test_activity_resets_idle(self):
        from jarvis.evolution.idle_detector import IdleDetector

        d = IdleDetector(idle_threshold_seconds=0)
        d._last_activity = time.time() - 100  # Force old
        assert d.is_idle is True
        d.notify_activity()
        # After activity with threshold 0, need to wait
        # But with very small threshold it should flip back quickly
        assert d.idle_seconds < 1

    def test_idle_seconds(self):
        from jarvis.evolution.idle_detector import IdleDetector

        d = IdleDetector(idle_threshold_seconds=10)
        d._last_activity = time.time() - 30
        assert d.idle_seconds >= 29


class TestEvolutionLoop:
    @pytest.fixture
    def idle_detector(self):
        from jarvis.evolution.idle_detector import IdleDetector

        d = IdleDetector(idle_threshold_seconds=0)
        d._last_activity = time.time() - 100  # Force idle
        return d

    @pytest.fixture
    def loop(self, idle_detector):
        from jarvis.evolution.loop import EvolutionLoop

        return EvolutionLoop(idle_detector=idle_detector)

    @pytest.mark.asyncio
    async def test_cycle_skips_when_not_idle(self):
        from jarvis.evolution.idle_detector import IdleDetector
        from jarvis.evolution.loop import EvolutionLoop

        d = IdleDetector(idle_threshold_seconds=9999)
        loop = EvolutionLoop(idle_detector=d)
        result = await loop.run_cycle()
        assert result.skipped is True
        assert result.reason == "not_idle"

    @pytest.mark.asyncio
    async def test_cycle_runs_when_idle(self, loop):
        result = await loop.run_cycle()
        # Without curiosity engine, should skip with no_gaps
        assert result.skipped is True
        assert result.reason == "no_gaps"
        assert "scout" in result.steps_completed

    @pytest.mark.asyncio
    async def test_daily_limit(self, loop):
        import time

        loop._cycles_today = 100
        loop._last_cycle_day = time.strftime("%Y-%m-%d")
        assert loop._can_run_cycle() is False

    def test_stats(self, loop):
        stats = loop.stats()
        assert "running" in stats
        assert "total_cycles" in stats
        assert "is_idle" in stats

    @pytest.mark.asyncio
    async def test_start_stop(self, loop):
        await loop.start()
        assert loop._running is True
        loop.stop()
        assert loop._running is False
        await asyncio.sleep(0.1)  # Let cancellation propagate


class TestCooperativeScheduling:
    """Tests fuer ResourceMonitor + Budget Integration im EvolutionLoop."""

    @pytest.fixture
    def idle_detector(self):
        from jarvis.evolution.idle_detector import IdleDetector

        d = IdleDetector(idle_threshold_seconds=0)
        d._last_activity = time.time() - 100
        return d

    @pytest.mark.asyncio
    async def test_cycle_skips_when_system_busy(self, idle_detector):
        from jarvis.evolution.loop import EvolutionLoop
        from jarvis.system.resource_monitor import ResourceMonitor, ResourceSnapshot

        monitor = ResourceMonitor()
        # Inject a busy snapshot
        monitor._last_snapshot = ResourceSnapshot(cpu_percent=95.0, is_busy=True)
        monitor._last_sample_time = time.monotonic()

        loop = EvolutionLoop(idle_detector=idle_detector, resource_monitor=monitor)
        result = await loop.run_cycle()
        assert result.skipped is True
        assert result.reason == "system_busy"

    @pytest.mark.asyncio
    async def test_cycle_proceeds_when_resources_ok(self, idle_detector):
        from jarvis.evolution.loop import EvolutionLoop
        from jarvis.system.resource_monitor import ResourceMonitor, ResourceSnapshot

        monitor = ResourceMonitor()
        monitor._last_snapshot = ResourceSnapshot(cpu_percent=30.0, is_busy=False)
        monitor._last_sample_time = time.monotonic()

        loop = EvolutionLoop(idle_detector=idle_detector, resource_monitor=monitor)
        result = await loop.run_cycle()
        # Should pass resource check but skip at no_gaps (no curiosity engine)
        assert result.reason == "no_gaps"
        assert "scout" in result.steps_completed

    @pytest.mark.asyncio
    async def test_cycle_skips_when_budget_exhausted(self, idle_detector):
        from jarvis.evolution.loop import EvolutionLoop
        from jarvis.models import AgentBudgetStatus

        mock_tracker = MagicMock()
        mock_tracker.check_agent_budget.return_value = AgentBudgetStatus(
            agent_name="scout",
            daily_cost_usd=1.0,
            daily_limit_usd=0.5,
            ok=False,
            warning="Agent 'scout' Tageslimit erreicht",
        )
        mock_config = MagicMock()
        mock_config.agent_budgets = {"scout": 0.5}
        mock_config.max_cycles_per_day = 10
        mock_config.cycle_cooldown_seconds = 300

        loop = EvolutionLoop(
            idle_detector=idle_detector,
            config=mock_config,
            cost_tracker=mock_tracker,
        )
        result = await loop.run_cycle()
        assert result.skipped is True
        assert result.reason == "budget_exhausted"

    @pytest.mark.asyncio
    async def test_cycle_ok_when_budget_available(self, idle_detector):
        from jarvis.evolution.loop import EvolutionLoop
        from jarvis.models import AgentBudgetStatus

        mock_tracker = MagicMock()
        mock_tracker.check_agent_budget.return_value = AgentBudgetStatus(
            agent_name="scout",
            daily_cost_usd=0.1,
            daily_limit_usd=0.5,
            ok=True,
        )
        mock_config = MagicMock()
        mock_config.agent_budgets = {"scout": 0.5}
        mock_config.max_cycles_per_day = 10

        loop = EvolutionLoop(
            idle_detector=idle_detector,
            config=mock_config,
            cost_tracker=mock_tracker,
        )
        result = await loop.run_cycle()
        # Passes budget check, skips at no_gaps
        assert result.reason == "no_gaps"

    def test_stats_includes_resources(self, idle_detector):
        from jarvis.evolution.loop import EvolutionLoop
        from jarvis.system.resource_monitor import ResourceMonitor, ResourceSnapshot

        monitor = ResourceMonitor()
        monitor._last_snapshot = ResourceSnapshot(
            cpu_percent=45.0, ram_percent=62.0, gpu_util_percent=30.0
        )
        loop = EvolutionLoop(idle_detector=idle_detector, resource_monitor=monitor)
        stats = loop.stats()
        assert "resources" in stats
        assert stats["resources"]["cpu_percent"] == 45.0
        assert stats["resources"]["available"] is True

    def test_stats_resources_without_monitor(self, idle_detector):
        from jarvis.evolution.loop import EvolutionLoop

        loop = EvolutionLoop(idle_detector=idle_detector)
        stats = loop.stats()
        assert stats["resources"]["available"] is True


class TestATLGoalMatching:
    """_match_goal_for_action finds the best goal for a research action."""

    def _make_goal(self, title: str, goal_id: str = ""):
        from dataclasses import dataclass

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
        action = type(
            "A",
            (),
            {
                "rationale": "OWASP Top 10 Cybersecurity Pentesting Standards",
                "params": {"query": "OWASP"},
            },
        )()

        result = _match_goal_for_action(action, goals)
        assert result is not None
        assert "Cybersecurity" in result.title

    def test_matches_query_param_too(self):
        from jarvis.evolution.loop import _match_goal_for_action

        goals = [
            self._make_goal("Werde Experte fuer AI Agent Architektur"),
            self._make_goal("Werde Experte fuer Versicherungsrecht"),
        ]
        action = type(
            "A",
            (),
            {
                "rationale": "Recherche",
                "params": {"query": "AI Agent Architecture Patterns"},
            },
        )()

        result = _match_goal_for_action(action, goals)
        assert result is not None
        assert "AI Agent" in result.title

    def test_returns_none_on_no_match(self):
        from jarvis.evolution.loop import _match_goal_for_action

        goals = [self._make_goal("Werde Experte fuer Kochen")]
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


class TestATLAutoPersist:
    """ATL auto-persists search_and_read results through KnowledgeBuilder."""

    @pytest.mark.asyncio
    async def test_research_result_persisted(self):
        """search_and_read result flows through synthesis -> KnowledgeBuilder."""
        from jarvis.evolution.loop import EvolutionLoop

        loop = EvolutionLoop.__new__(EvolutionLoop)
        loop._llm_fn = AsyncMock(
            return_value=(
                "## ARC-AGI Strategien\n"
                "- CNN+Graph Ansaetze dominieren (Quelle: kaggle.com)\n"
                "- Stochastic Goose erreicht 45% Accuracy\n"
            )
        )
        loop._mcp_client = AsyncMock()
        loop._deep_learner = MagicMock()
        loop._deep_learner._plans_dir = MagicMock()
        loop._deep_learner._plans_dir.parent = MagicMock()

        goal = MagicMock()
        goal.title = "Werde Experte fuer ARC-AGI-3 Strategien"
        goal.id = "arc-agi"

        mock_builder = AsyncMock()
        mock_build_result = MagicMock()
        mock_build_result.errors = []
        mock_build_result.chunks_created = 3
        mock_builder.build = AsyncMock(return_value=mock_build_result)

        loop._atl_knowledge_builders = {"arc-agi": mock_builder}
        loop._atl_persisted_queries = set()

        action = MagicMock()
        action.type = "research"
        action.rationale = "ARC-AGI-3 Strategien research"
        action.params = {"query": "ARC-AGI-3 Strategien"}

        await loop._persist_research_result(
            tool_result="Long research text about ARC-AGI strategies " * 20,
            action=action,
            goals=[goal],
        )

        assert loop._llm_fn.call_count == 1
        mock_builder.build.assert_called_once()
        call_args = mock_builder.build.call_args
        fetch_result = call_args[0][0]
        assert "ARC-AGI" in fetch_result.text
        assert fetch_result.source_type == "atl_research"

    @pytest.mark.asyncio
    async def test_dedup_prevents_double_persist(self):
        from jarvis.evolution.loop import EvolutionLoop

        loop = EvolutionLoop.__new__(EvolutionLoop)
        loop._llm_fn = AsyncMock(return_value="## Note\n- Fact 1\n")
        loop._atl_persisted_queries = {"arc-agi-3 strategies"}

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

        loop._llm_fn.assert_not_called()
        mock_builder.build.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_persist_when_synthesis_returns_none(self):
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

        mock_builder.build.assert_not_called()
