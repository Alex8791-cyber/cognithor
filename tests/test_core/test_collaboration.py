"""Tests for the Multi-Agent Collaboration Engine.

Covers agent roles, task board, collaboration patterns (debate, voting,
critic review, pipeline, parallel), consensus, and edge cases.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.core.collaboration import (
    AgentRole,
    BoardEntry,
    CollaborationEngine,
    CollaborationPattern,
    CollaborationResult,
    ROLE_DEFAULTS,
    RoleSpec,
    TaskBoard,
)


# ============================================================================
# Helpers
# ============================================================================


def _role(role: AgentRole, agent: str = "") -> RoleSpec:
    return RoleSpec(role=role, agent_name=agent or role.value)


def _make_runner(responses: dict[str, str] | None = None) -> AsyncMock:
    """Create a mock agent runner that returns predefined responses."""
    default_responses = responses or {}

    async def runner(task: str, agent_name: str) -> MagicMock:
        result = MagicMock()
        result.response = default_responses.get(agent_name, f"Response from {agent_name}")
        return result

    return AsyncMock(side_effect=runner)


def _make_failing_runner() -> AsyncMock:
    async def runner(task: str, agent_name: str) -> MagicMock:
        raise RuntimeError(f"{agent_name} crashed")

    return AsyncMock(side_effect=runner)


# ============================================================================
# AgentRole
# ============================================================================


class TestAgentRole:
    def test_all_roles_exist(self) -> None:
        assert len(AgentRole) == 7

    def test_role_values(self) -> None:
        assert AgentRole.RESEARCHER == "researcher"
        assert AgentRole.CRITIC == "critic"
        assert AgentRole.SYNTHESIZER == "synthesizer"

    def test_role_defaults_for_all_roles(self) -> None:
        for role in AgentRole:
            assert role in ROLE_DEFAULTS
            assert len(ROLE_DEFAULTS[role]) > 10


# ============================================================================
# RoleSpec
# ============================================================================


class TestRoleSpec:
    def test_create_spec(self) -> None:
        spec = RoleSpec(role=AgentRole.RESEARCHER, agent_name="researcher_1")
        assert spec.role == AgentRole.RESEARCHER
        assert spec.agent_name == "researcher_1"
        assert spec.max_iterations == 5

    def test_frozen(self) -> None:
        spec = _role(AgentRole.ANALYST)
        with pytest.raises(Exception):
            spec.max_iterations = 99  # type: ignore[misc]


# ============================================================================
# TaskBoard
# ============================================================================


class TestTaskBoard:
    async def test_post_and_get(self) -> None:
        board = TaskBoard("test")
        await board.post("agent_a", AgentRole.RESEARCHER, "Found results")
        entries = board.get_entries()
        assert len(entries) == 1
        assert entries[0].agent == "agent_a"
        assert entries[0].content == "Found results"

    async def test_filter_by_role(self) -> None:
        board = TaskBoard("test")
        await board.post("a", AgentRole.RESEARCHER, "research")
        await board.post("b", AgentRole.CRITIC, "criticism")
        researchers = board.get_entries(role=AgentRole.RESEARCHER)
        assert len(researchers) == 1
        critics = board.get_entries(role=AgentRole.CRITIC)
        assert len(critics) == 1

    async def test_vote(self) -> None:
        board = TaskBoard("test")
        await board.post("a", AgentRole.RESEARCHER, "output")
        await board.vote("b", "a", 0.8)
        await board.vote("c", "a", 0.6)
        scores = board.get_votes()
        assert scores["a"] == pytest.approx(0.7)

    async def test_vote_clamped(self) -> None:
        board = TaskBoard("test")
        await board.vote("a", "b", 1.5)
        await board.vote("c", "b", -0.5)
        scores = board.get_votes()
        assert scores["b"] == pytest.approx(0.5)  # (1.0 + 0.0) / 2

    async def test_get_winner(self) -> None:
        board = TaskBoard("test")
        await board.post("a", AgentRole.RESEARCHER, "A's output")
        await board.post("b", AgentRole.ANALYST, "B's output")
        await board.vote("c", "a", 0.3)
        await board.vote("c", "b", 0.9)
        winner = board.get_winner()
        assert winner is not None
        assert winner.agent == "b"

    async def test_get_winner_no_votes(self) -> None:
        board = TaskBoard("test")
        await board.post("a", AgentRole.RESEARCHER, "only entry")
        winner = board.get_winner()
        assert winner is not None
        assert winner.agent == "a"

    async def test_get_winner_empty(self) -> None:
        board = TaskBoard("test")
        assert board.get_winner() is None

    async def test_entry_count(self) -> None:
        board = TaskBoard("test")
        assert board.entry_count == 0
        await board.post("a", AgentRole.RESEARCHER, "x")
        assert board.entry_count == 1

    async def test_voter_count(self) -> None:
        board = TaskBoard("test")
        assert board.voter_count == 0
        await board.vote("a", "b", 0.5)
        assert board.voter_count == 1


# ============================================================================
# CollaborationResult
# ============================================================================


class TestCollaborationResult:
    def test_success_with_output(self) -> None:
        r = CollaborationResult(final_output="result")
        assert r.success is True

    def test_failure_with_errors(self) -> None:
        r = CollaborationResult(final_output="result", errors=["fail"])
        assert r.success is False

    def test_failure_no_output(self) -> None:
        r = CollaborationResult()
        assert r.success is False

    def test_to_dict(self) -> None:
        r = CollaborationResult(
            collaboration_id="abc",
            pattern="debate",
            task="test",
            participants=["a", "b"],
            final_output="output",
            consensus_score=0.85,
        )
        d = r.to_dict()
        assert d["collaboration_id"] == "abc"
        assert d["success"] is True
        assert d["consensus_score"] == 0.85


# ============================================================================
# Debate Pattern
# ============================================================================


class TestDebate:
    async def test_successful_debate(self) -> None:
        runner = _make_runner({
            "researcher": "I found important data about market trends.",
            "analyst": "The data shows a clear upward trend with 15% growth.",
        })
        engine = CollaborationEngine(agent_runner=runner)
        result = await engine.debate(
            "Analyze market trends",
            [_role(AgentRole.RESEARCHER), _role(AgentRole.ANALYST)],
            rounds=1,
        )
        assert result.success
        assert result.pattern == CollaborationPattern.DEBATE
        assert len(result.participants) == 2
        assert result.final_output != ""

    async def test_debate_requires_two_participants(self) -> None:
        engine = CollaborationEngine()
        result = await engine.debate("task", [_role(AgentRole.RESEARCHER)])
        assert not result.success
        assert any("at least 2" in e for e in result.errors)

    async def test_debate_multi_round(self) -> None:
        runner = _make_runner()
        engine = CollaborationEngine(agent_runner=runner, max_rounds=2)
        result = await engine.debate(
            "task",
            [_role(AgentRole.RESEARCHER), _role(AgentRole.ANALYST)],
            rounds=2,
        )
        assert result.success
        assert len(result.contributions) >= 4  # 2 agents × 2 rounds

    async def test_debate_no_runner(self) -> None:
        engine = CollaborationEngine()
        result = await engine.debate(
            "task",
            [_role(AgentRole.RESEARCHER), _role(AgentRole.ANALYST)],
            rounds=1,
        )
        # Should still work, just with empty responses
        assert result.pattern == CollaborationPattern.DEBATE

    async def test_debate_records_history(self) -> None:
        engine = CollaborationEngine(agent_runner=_make_runner())
        await engine.debate("t", [_role(AgentRole.RESEARCHER), _role(AgentRole.ANALYST)], rounds=1)
        assert len(engine.history) == 1


# ============================================================================
# Voting Pattern
# ============================================================================


class TestVoting:
    async def test_voting_cross_vote(self) -> None:
        runner = _make_runner({
            "researcher": "Short answer.",
            "analyst": "A much longer and more detailed analysis of the topic with many insights.",
        })
        engine = CollaborationEngine(agent_runner=runner)
        result = await engine.vote(
            "task",
            [_role(AgentRole.RESEARCHER), _role(AgentRole.ANALYST)],
        )
        assert result.success
        assert result.pattern == CollaborationPattern.VOTING
        assert result.winner != ""

    async def test_voting_with_explicit_voter(self) -> None:
        runner = _make_runner({
            "researcher": "Option A",
            "analyst": "Option B",
            "validator": "0.8",
        })
        engine = CollaborationEngine(agent_runner=runner)
        result = await engine.vote(
            "task",
            [_role(AgentRole.RESEARCHER), _role(AgentRole.ANALYST)],
            voter=_role(AgentRole.VALIDATOR),
        )
        assert result.success
        assert len(result.contributions) == 2


# ============================================================================
# Critic Review Pattern
# ============================================================================


class TestCriticReview:
    async def test_basic_review(self) -> None:
        runner = _make_runner({
            "executor": "Draft output",
            "critic": "Gut, korrekt und vollständig.",
        })
        engine = CollaborationEngine(agent_runner=runner)
        result = await engine.critic_review(
            "Write a summary",
            producer=_role(AgentRole.EXECUTOR),
            critic=_role(AgentRole.CRITIC),
        )
        assert result.success
        assert result.pattern == CollaborationPattern.CRITIC_REVIEW
        assert result.winner == "executor"

    async def test_review_with_revision(self) -> None:
        call_count = {"n": 0}

        async def runner(task: str, agent_name: str) -> MagicMock:
            call_count["n"] += 1
            result = MagicMock()
            if agent_name == "critic" and call_count["n"] <= 3:
                result.response = "Needs improvement: more detail required."
            elif agent_name == "critic":
                result.response = "Gut, akzeptiert."
            else:
                result.response = f"Revision {call_count['n']}"
            return result

        engine = CollaborationEngine(agent_runner=AsyncMock(side_effect=runner))
        result = await engine.critic_review(
            "task",
            producer=_role(AgentRole.EXECUTOR),
            critic=_role(AgentRole.CRITIC),
            max_revisions=3,
        )
        assert result.success
        assert len(result.contributions) >= 3  # At least: initial + critique + revision

    async def test_review_max_revisions(self) -> None:
        runner = _make_runner({
            "executor": "Draft",
            "critic": "Nicht gut, verbessern.",  # Never approves
        })
        engine = CollaborationEngine(agent_runner=runner)
        result = await engine.critic_review(
            "task",
            producer=_role(AgentRole.EXECUTOR),
            critic=_role(AgentRole.CRITIC),
            max_revisions=2,
        )
        # Should complete even without approval (max_revisions reached)
        assert result.final_output != ""


# ============================================================================
# Pipeline Pattern
# ============================================================================


class TestPipeline:
    async def test_basic_pipeline(self) -> None:
        runner = _make_runner({
            "researcher": "Raw data collected.",
            "analyst": "Analysis of the raw data.",
            "synthesizer": "Final synthesis.",
        })
        engine = CollaborationEngine(agent_runner=runner)
        result = await engine.pipeline(
            "Analyze topic",
            [
                _role(AgentRole.RESEARCHER),
                _role(AgentRole.ANALYST),
                _role(AgentRole.SYNTHESIZER),
            ],
        )
        assert result.success
        assert result.pattern == CollaborationPattern.PIPELINE
        assert len(result.contributions) == 3

    async def test_pipeline_stops_on_error(self) -> None:
        engine = CollaborationEngine(agent_runner=_make_failing_runner())
        result = await engine.pipeline(
            "task",
            [_role(AgentRole.RESEARCHER), _role(AgentRole.ANALYST)],
        )
        assert not result.success
        assert len(result.errors) >= 1

    async def test_pipeline_empty_stages(self) -> None:
        engine = CollaborationEngine()
        result = await engine.pipeline("task", [])
        assert result.final_output == "task"  # Passthrough


# ============================================================================
# Parallel Pattern
# ============================================================================


class TestParallel:
    async def test_basic_parallel(self) -> None:
        runner = _make_runner({
            "researcher": "Research findings",
            "analyst": "Analysis results",
        })
        engine = CollaborationEngine(agent_runner=runner)
        result = await engine.parallel(
            "Investigate topic",
            [_role(AgentRole.RESEARCHER), _role(AgentRole.ANALYST)],
        )
        assert result.success
        assert result.pattern == CollaborationPattern.PARALLEL
        assert len(result.contributions) == 2
        assert "Research findings" in result.final_output
        assert "Analysis results" in result.final_output

    async def test_parallel_with_failures(self) -> None:
        engine = CollaborationEngine(agent_runner=_make_failing_runner())
        result = await engine.parallel(
            "task",
            [_role(AgentRole.RESEARCHER)],
        )
        # Errors are caught and returned as [Error:...] in content
        assert "[Error:" in result.final_output


# ============================================================================
# Stats
# ============================================================================


class TestStats:
    async def test_stats_empty(self) -> None:
        engine = CollaborationEngine()
        s = engine.stats()
        assert s["total_collaborations"] == 0
        assert s["success_count"] == 0

    async def test_stats_after_collaborations(self) -> None:
        engine = CollaborationEngine(agent_runner=_make_runner())
        await engine.debate("t", [_role(AgentRole.RESEARCHER), _role(AgentRole.ANALYST)], rounds=1)
        await engine.parallel("t", [_role(AgentRole.RESEARCHER)])
        s = engine.stats()
        assert s["total_collaborations"] == 2
        assert s["by_pattern"]["debate"] == 1
        assert s["by_pattern"]["parallel"] == 1

    async def test_duration_positive(self) -> None:
        engine = CollaborationEngine(agent_runner=_make_runner())
        result = await engine.parallel("t", [_role(AgentRole.RESEARCHER)])
        assert result.duration_ms >= 0
