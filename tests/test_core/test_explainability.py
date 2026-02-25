"""Tests: Explainability-Layer.

Prüft DecisionTrail, SourceAttribution, TrustScoreCalculator
und ExplainabilityEngine.
"""

from __future__ import annotations

import pytest

from jarvis.core.explainability import (
    DecisionTrail,
    ExplainabilityEngine,
    SourceAttribution,
    SourceReference,
    SourceType,
    StepType,
    TrailStep,
    TrustLevel,
    TrustScoreCalculator,
)


# ============================================================================
# DecisionTrail
# ============================================================================


class TestDecisionTrail:
    def test_add_step(self) -> None:
        trail = DecisionTrail(trail_id="t1", agent_id="coder")
        step = trail.add_step(StepType.INPUT_RECEIVED, "User fragt nach Code-Review")
        assert trail.step_count == 1
        assert step.step_type == StepType.INPUT_RECEIVED

    def test_tools_used(self) -> None:
        trail = DecisionTrail(trail_id="t1", agent_id="coder")
        trail.add_step(StepType.INPUT_RECEIVED, "Input")
        trail.add_step(StepType.TOOL_CALLED, "shell:run_tests")
        trail.add_step(StepType.TOOL_CALLED, "file:read")
        assert trail.tools_used == ["shell:run_tests", "file:read"]

    def test_agents_involved(self) -> None:
        trail = DecisionTrail(trail_id="t1", agent_id="coder")
        trail.add_step(StepType.INPUT_RECEIVED, "Input")
        trail.add_step(StepType.DELEGATION, "Delegate", agent_id="researcher")
        assert "coder" in trail.agents_involved
        assert "researcher" in trail.agents_involved

    def test_had_approval(self) -> None:
        trail = DecisionTrail(trail_id="t1", agent_id="coder")
        trail.add_step(StepType.APPROVAL_REQUESTED, "Genehmigung angefordert")
        assert trail.had_approval

    def test_no_approval(self) -> None:
        trail = DecisionTrail(trail_id="t1", agent_id="coder")
        trail.add_step(StepType.INPUT_RECEIVED, "Input")
        assert not trail.had_approval

    def test_had_errors(self) -> None:
        trail = DecisionTrail(trail_id="t1", agent_id="coder")
        trail.add_step(StepType.ERROR, "Tool timeout")
        assert trail.had_errors

    def test_complete(self) -> None:
        trail = DecisionTrail(trail_id="t1", agent_id="coder", started_at="2025-01-01T00:00:00Z")
        trail.add_step(StepType.INPUT_RECEIVED, "Input", duration_ms=10)
        trail.add_step(StepType.REASONING, "Thinking", duration_ms=200)
        trail.complete()
        assert trail.completed_at
        assert trail.total_duration_ms == 210

    def test_to_dict(self) -> None:
        trail = DecisionTrail(trail_id="t1", agent_id="coder")
        trail.add_step(StepType.INPUT_RECEIVED, "Input")
        d = trail.to_dict()
        assert d["trail_id"] == "t1"
        assert d["step_count"] == 1

    def test_to_human_readable(self) -> None:
        trail = DecisionTrail(trail_id="t1", agent_id="coder")
        trail.add_step(StepType.INPUT_RECEIVED, "Frage empfangen", duration_ms=5)
        trail.add_step(StepType.TOOL_CALLED, "shell:pytest", duration_ms=150)
        text = trail.to_human_readable()
        assert "📥" in text
        assert "🔧" in text
        assert "shell:pytest" in text


# ============================================================================
# SourceAttribution
# ============================================================================


class TestSourceAttribution:
    def test_add_source(self) -> None:
        attr = SourceAttribution()
        attr.add_source("claim1", SourceReference(
            source_id="s1", source_type=SourceType.MEMORY_SEMANTIC,
            title="User-Profil", relevance_score=0.9,
        ))
        assert attr.claim_count == 1
        assert attr.total_sources == 1

    def test_multiple_sources_per_claim(self) -> None:
        attr = SourceAttribution()
        attr.add_source("claim1", SourceReference(source_id="s1", source_type=SourceType.MEMORY_SEMANTIC, title="A", relevance_score=0.8))
        attr.add_source("claim1", SourceReference(source_id="s2", source_type=SourceType.WEB_SEARCH, title="B", relevance_score=0.7))
        assert attr.claim_count == 1
        assert attr.total_sources == 2
        assert len(attr.get_sources("claim1")) == 2

    def test_source_diversity(self) -> None:
        attr = SourceAttribution()
        attr.add_source("c1", SourceReference(source_id="s1", source_type=SourceType.MEMORY_SEMANTIC, title="A", relevance_score=0.8))
        attr.add_source("c2", SourceReference(source_id="s2", source_type=SourceType.WEB_SEARCH, title="B", relevance_score=0.7))
        attr.add_source("c3", SourceReference(source_id="s3", source_type=SourceType.WEB_SEARCH, title="C", relevance_score=0.6))
        div = attr.source_diversity()
        assert div["memory_semantic"] == 1
        assert div["web_search"] == 2

    def test_to_dict(self) -> None:
        attr = SourceAttribution()
        attr.add_source("c1", SourceReference(source_id="s1", source_type=SourceType.USER_INPUT, title="A", relevance_score=1.0))
        d = attr.to_dict()
        assert d["claim_count"] == 1

    def test_empty(self) -> None:
        attr = SourceAttribution()
        assert attr.get_sources("nonexistent") == []


# ============================================================================
# TrustScoreCalculator
# ============================================================================


class TestTrustScoreCalculator:
    def test_no_sources_low_trust(self) -> None:
        calc = TrustScoreCalculator()
        score, level, _ = calc.calculate()
        assert score < 0.5
        assert level in (TrustLevel.VERY_LOW, TrustLevel.LOW)

    def test_high_trust_multi_source(self) -> None:
        calc = TrustScoreCalculator()
        sources = [
            SourceReference(source_id="s1", source_type=SourceType.USER_INPUT, title="A", relevance_score=0.95),
            SourceReference(source_id="s2", source_type=SourceType.TOOL_OUTPUT, title="B", relevance_score=0.9),
            SourceReference(source_id="s3", source_type=SourceType.MEMORY_SEMANTIC, title="C", relevance_score=0.85),
        ]
        score, level, breakdown = calc.calculate(
            sources=sources,
            human_approved=True,
        )
        assert score >= 0.7
        assert level in (TrustLevel.HIGH, TrustLevel.VERY_HIGH)

    def test_contradictions_lower_trust(self) -> None:
        calc = TrustScoreCalculator()
        _, _, breakdown_clean = calc.calculate(contradictions_found=0)
        _, _, breakdown_dirty = calc.calculate(contradictions_found=3)
        assert breakdown_clean["components"]["consistency"] > breakdown_dirty["components"]["consistency"]

    def test_human_approval_bonus(self) -> None:
        calc = TrustScoreCalculator()
        score_no, _, _ = calc.calculate(human_approved=False)
        score_yes, _, _ = calc.calculate(human_approved=True)
        assert score_yes > score_no

    def test_trail_quality(self) -> None:
        calc = TrustScoreCalculator()
        trail = DecisionTrail(trail_id="t1", agent_id="coder")
        trail.add_step(StepType.TOOL_CALLED, "shell:test")
        trail.add_step(StepType.MEMORY_RETRIEVED, "Recall")
        _, _, breakdown = calc.calculate(trail=trail)
        assert breakdown["components"]["trail_quality"] > 0

    def test_score_clamped(self) -> None:
        calc = TrustScoreCalculator()
        score, _, _ = calc.calculate(
            sources=[SourceReference(source_id="s", source_type=SourceType.USER_INPUT, title="A", relevance_score=1.0)] * 20,
            human_approved=True,
        )
        assert 0 <= score <= 1.0


# ============================================================================
# ExplainabilityEngine
# ============================================================================


class TestExplainabilityEngine:
    def test_start_trail(self) -> None:
        engine = ExplainabilityEngine()
        trail = engine.start_trail("req-001", "coder")
        assert trail.request_id == "req-001"
        assert trail.agent_id == "coder"

    def test_get_trail(self) -> None:
        engine = ExplainabilityEngine()
        trail = engine.start_trail("req-001")
        retrieved = engine.get_trail(trail.trail_id)
        assert retrieved is not None
        assert retrieved.trail_id == trail.trail_id

    def test_complete_trail(self) -> None:
        engine = ExplainabilityEngine()
        trail = engine.start_trail("req-001", "coder")
        trail.add_step(StepType.INPUT_RECEIVED, "Input", duration_ms=5)
        trail.add_step(StepType.TOOL_CALLED, "shell:test", duration_ms=100)

        completed, breakdown = engine.complete_trail(
            trail.trail_id,
            sources=[SourceReference(source_id="s1", source_type=SourceType.TOOL_OUTPUT, title="A", relevance_score=0.8)],
            human_approved=True,
        )
        assert completed is not None
        assert completed.completed_at
        assert breakdown["total_score"] > 0

    def test_complete_nonexistent(self) -> None:
        engine = ExplainabilityEngine()
        trail, breakdown = engine.complete_trail("nope")
        assert trail is None
        assert breakdown == {}

    def test_recent_trails(self) -> None:
        engine = ExplainabilityEngine()
        for i in range(5):
            engine.start_trail(f"req-{i}")
        assert len(engine.recent_trails(limit=3)) == 3

    def test_trails_by_agent(self) -> None:
        engine = ExplainabilityEngine()
        engine.start_trail("r1", "coder")
        engine.start_trail("r2", "researcher")
        engine.start_trail("r3", "coder")
        assert len(engine.trails_by_agent("coder")) == 2

    def test_low_trust_trails(self) -> None:
        engine = ExplainabilityEngine()
        trail = engine.start_trail("r1", "agent")
        trail.add_step(StepType.OUTPUT_GENERATED, "Antwort", confidence=0.2)
        engine.complete_trail(trail.trail_id)
        low = engine.low_trust_trails(threshold=0.5)
        assert len(low) >= 1

    def test_stats(self) -> None:
        engine = ExplainabilityEngine()
        t = engine.start_trail("r1", "coder")
        t.add_step(StepType.INPUT_RECEIVED, "Input")
        engine.complete_trail(t.trail_id)
        stats = engine.stats()
        assert stats["total_requests"] == 1
        assert stats["completed_trails"] == 1
        assert stats["unique_agents"] == 1

    def test_max_trails(self) -> None:
        engine = ExplainabilityEngine(max_trails=5)
        for i in range(10):
            engine.start_trail(f"req-{i}")
        assert len(engine.recent_trails(limit=100)) <= 5
