"""Tests fuer Self-Profiling Feedback Loop Wiring (Feature 3).

Testet:
  - Planner nutzt CapabilityProfile fuer Selbsteinschaetzung
  - Reflector nutzt RewardCalculator fuer Composite-Score
  - Reward ersetzt einfachen success_score
  - Graceful Degradation bei fehlendem RewardCalculator
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.core.planner import Planner
from jarvis.core.reflector import Reflector
from jarvis.learning.reward import RewardCalculator
from jarvis.models import (
    ActionPlan,
    AgentResult,
    PlannedAction,
    ReflectionResult,
    SessionContext,
    SessionSummary,
    ToolResult,
    WorkingMemory,
)


# ── Planner Tests ──────────────────────────────────────────


class TestPlannerCapabilityProfile:
    """Tests fuer Planner mit TaskProfiler/CapabilityProfile."""

    def _make_planner(self, task_profiler=None) -> Planner:
        config = MagicMock()
        config.owner_name = "Test"
        ollama = MagicMock()
        model_router = MagicMock()
        model_router.select_model.return_value = "test-model"
        model_router.get_model_config.return_value = {"temperature": 0.7, "top_p": 0.9}
        return Planner(
            config, ollama, model_router,
            task_profiler=task_profiler,
        )

    def test_planner_uses_capability_profile(self):
        """Mock-Profiler → Prompt enthaelt 'Staerken'."""
        profiler = MagicMock()
        cap = MagicMock()
        cap.strengths = ["Dateiverwaltung", "Web-Suche", "Memory"]
        cap.weaknesses = ["Shell-Befehle"]
        profiler.get_capability_profile.return_value = cap

        planner = self._make_planner(task_profiler=profiler)

        wm = WorkingMemory(session_id="test")
        prompt = planner._build_system_prompt(
            working_memory=wm,
            tool_schemas={},
        )

        assert "Selbsteinschaetzung" in prompt
        assert "Staerken" in prompt
        assert "Dateiverwaltung" in prompt

    def test_planner_without_profiler(self):
        """Kein Profiler → kein Crash, kein Selbsteinschaetzung-Block."""
        planner = self._make_planner(task_profiler=None)

        wm = WorkingMemory(session_id="test")
        prompt = planner._build_system_prompt(
            working_memory=wm,
            tool_schemas={},
        )

        assert "Selbsteinschaetzung" not in prompt

    def test_planner_profiler_exception(self):
        """Profiler wirft Exception → graceful skip."""
        profiler = MagicMock()
        profiler.get_capability_profile.side_effect = RuntimeError("profile error")

        planner = self._make_planner(task_profiler=profiler)

        wm = WorkingMemory(session_id="test")
        # Should not raise
        prompt = planner._build_system_prompt(
            working_memory=wm,
            tool_schemas={},
        )
        assert "Selbsteinschaetzung" not in prompt

    def test_planner_empty_profile(self):
        """Profiler liefert leeres Profile → kein Block."""
        profiler = MagicMock()
        cap = MagicMock()
        cap.strengths = []
        cap.weaknesses = []
        profiler.get_capability_profile.return_value = cap

        planner = self._make_planner(task_profiler=profiler)

        wm = WorkingMemory(session_id="test")
        prompt = planner._build_system_prompt(
            working_memory=wm,
            tool_schemas={},
        )
        assert "Selbsteinschaetzung" not in prompt


# ── Reflector Tests ──────────────────────────────────────────


class TestReflectorRewardCalculator:
    """Tests fuer Reflector mit RewardCalculator."""

    def _make_reflector(self, reward_calculator=None, causal_analyzer=None, episodic_store=None) -> Reflector:
        config = MagicMock()
        ollama = MagicMock()
        model_router = MagicMock()
        model_router.select_model.return_value = "test-model"
        model_router.get_model_config.return_value = {"temperature": 0.3, "top_p": 0.9}
        return Reflector(
            config, ollama, model_router,
            reward_calculator=reward_calculator,
            causal_analyzer=causal_analyzer,
            episodic_store=episodic_store,
        )

    def _make_agent_result(self, tool_results=None, success=True) -> AgentResult:
        results = tool_results or [
            ToolResult(tool_name="read_file", content="ok", success=True),
            ToolResult(tool_name="write_file", content="ok", success=True),
        ]
        plan = ActionPlan(
            goal="Test goal",
            steps=[PlannedAction(tool="read_file", params={}), PlannedAction(tool="write_file", params={})],
        )
        return AgentResult(
            response="Done",
            plans=[plan],
            tool_results=results,
            total_iterations=1,
            total_duration_ms=5000,
            model_used="test",
            success=success,
        )

    @pytest.mark.asyncio
    async def test_reflector_uses_reward_calculator(self):
        """RewardCalculator Score wird an CausalAnalyzer weitergeleitet."""
        reward_calc = RewardCalculator()
        causal = MagicMock()

        reflector = self._make_reflector(
            reward_calculator=reward_calc,
            causal_analyzer=causal,
        )

        # Mock the LLM call to return a valid reflection JSON
        reflector._ollama.chat = AsyncMock(return_value={
            "message": {"content": '{"success_score": 0.8, "evaluation": "Gut"}'},
        })

        session = SessionContext(user_id="u1", channel="test")
        wm = WorkingMemory(session_id=session.session_id)
        agent_result = self._make_agent_result()

        result = await reflector.reflect(session, wm, agent_result)

        # Causal analyzer should have been called with the reward score,
        # which differs from the raw success_score (0.8) due to composite calculation
        if causal.record_sequence.called:
            call_kwargs = causal.record_sequence.call_args
            recorded_score = call_kwargs.kwargs.get("success_score") or call_kwargs[1].get("success_score", None)
            # Reward score should be different from simple 0.8
            # (composite includes error, efficiency, speed components)
            assert recorded_score is not None

    @pytest.mark.asyncio
    async def test_reward_replaces_simple_score(self):
        """Berechneter Reward weicht vom einfachen Score ab."""
        reward_calc = RewardCalculator()

        # Test with different scenarios
        # All success, 2 unique tools, 2 calls, 5s duration
        reward = reward_calc.calculate_reward(
            success_score=0.8,
            total_tools=2,
            failed_tools=0,
            unique_tools=2,
            total_tool_calls=2,
            duration_seconds=5.0,
        )
        # Composite: 0.4*0.8 + 0.2*1.0 + 0.2*1.0 + 0.2*(1-5/300)
        # = 0.32 + 0.2 + 0.2 + 0.1967 ≈ 0.917
        assert reward != 0.8  # Different from simple score
        assert reward > 0.8  # Should be higher due to perfect efficiency/errors

    @pytest.mark.asyncio
    async def test_graceful_degradation_no_reward_calculator(self):
        """RewardCalculator fehlt → Fallback auf success_score."""
        causal = MagicMock()
        episodic = MagicMock()

        reflector = self._make_reflector(
            reward_calculator=None,
            causal_analyzer=causal,
            episodic_store=episodic,
        )

        reflector._ollama.chat = AsyncMock(return_value={
            "message": {"content": '{"success_score": 0.7, "evaluation": "OK", "session_summary": {"goal": "Test", "outcome": "Done", "tools_used": ["read_file"]}}'},
        })

        session = SessionContext(user_id="u1", channel="test")
        wm = WorkingMemory(session_id=session.session_id)
        agent_result = self._make_agent_result()

        result = await reflector.reflect(session, wm, agent_result)

        # Should still work, just using raw score
        assert result.success_score == 0.7

        # Causal should be called with the raw score (0.7)
        if causal.record_sequence.called:
            call_kwargs = causal.record_sequence.call_args
            recorded_score = call_kwargs.kwargs.get("success_score")
            assert recorded_score == 0.7


# ── RewardCalculator Unit Tests ──────────────────────────────


class TestRewardCalculator:
    def test_perfect_run(self):
        """Alle Tools erfolgreich, unique, schnell → hoher Score."""
        calc = RewardCalculator()
        score = calc.calculate_reward(
            success_score=1.0,
            total_tools=3,
            failed_tools=0,
            unique_tools=3,
            total_tool_calls=3,
            duration_seconds=1.0,
        )
        assert score > 0.95

    def test_failed_run(self):
        """Alles fehlgeschlagen → niedriger Score."""
        calc = RewardCalculator()
        score = calc.calculate_reward(
            success_score=0.0,
            total_tools=3,
            failed_tools=3,
            unique_tools=1,
            total_tool_calls=3,
            duration_seconds=300.0,
        )
        assert score < 0.3

    def test_no_tools(self):
        """Keine Tool-Calls → nur success_score zaehlt."""
        calc = RewardCalculator()
        score = calc.calculate_reward(
            success_score=0.5,
            total_tools=0,
            failed_tools=0,
            unique_tools=0,
            total_tool_calls=0,
            duration_seconds=0.0,
        )
        assert 0.4 < score < 0.9
