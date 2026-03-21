"""Tests for GEPA robustness improvements."""
from __future__ import annotations


def test_min_traces_increased():
    """MIN_TRACES should be at least 20 for reliable analysis."""
    from jarvis.learning.evolution_orchestrator import EvolutionOrchestrator

    assert EvolutionOrchestrator.MIN_TRACES >= 20


def test_min_sessions_for_eval_increased():
    """MIN_SESSIONS_FOR_EVAL should be at least 15."""
    from jarvis.learning.evolution_orchestrator import EvolutionOrchestrator

    assert EvolutionOrchestrator.MIN_SESSIONS_FOR_EVAL >= 15


def test_high_impact_types_defined():
    """High-impact proposal types should require review."""
    from jarvis.learning.evolution_orchestrator import EvolutionOrchestrator

    assert hasattr(EvolutionOrchestrator, "HIGH_IMPACT_TYPES")
    assert "prompt_patch" in EvolutionOrchestrator.HIGH_IMPACT_TYPES
    assert "guardrail" in EvolutionOrchestrator.HIGH_IMPACT_TYPES
    assert "strategy_change" in EvolutionOrchestrator.HIGH_IMPACT_TYPES


def test_trace_optimizer_has_llm_method():
    """TraceOptimizer must have _generate_with_llm method."""
    from jarvis.learning.trace_optimizer import TraceOptimizer

    assert hasattr(TraceOptimizer, "_generate_with_llm")
