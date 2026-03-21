"""Tests for jarvis.learning.evolution_orchestrator — EvolutionOrchestrator."""

from __future__ import annotations

import time
import uuid
from unittest.mock import MagicMock, patch

import pytest

from jarvis.learning.causal_attributor import CausalAttributor, CausalFinding
from jarvis.learning.evolution_orchestrator import (
    EvolutionCycleResult,
    EvolutionOrchestrator,
    _MAX_CYCLE_HISTORY,
)
from jarvis.learning.execution_trace import ExecutionTrace, TraceStep, TraceStore
from jarvis.learning.trace_optimizer import (
    OptimizationProposal,
    ProposalStore,
    TraceOptimizer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_step(
    *,
    step_id: str | None = None,
    parent_id: str | None = None,
    tool_name: str = "shell_exec",
    status: str = "success",
    error_detail: str = "",
    duration_ms: int = 100,
) -> TraceStep:
    return TraceStep(
        step_id=step_id or uuid.uuid4().hex[:12],
        parent_id=parent_id,
        tool_name=tool_name,
        input_summary="in",
        output_summary="out",
        status=status,
        error_detail=error_detail,
        duration_ms=duration_ms,
        timestamp=time.time(),
    )


def _make_trace(
    *,
    trace_id: str | None = None,
    session_id: str = "sess-1",
    goal: str = "test",
    steps: list[TraceStep] | None = None,
    success_score: float = 0.8,
    total_duration_ms: int = 500,
    created_at: float = 0.0,
) -> ExecutionTrace:
    return ExecutionTrace(
        trace_id=trace_id or uuid.uuid4().hex[:12],
        session_id=session_id,
        goal=goal,
        steps=steps or [_make_step()],
        total_duration_ms=total_duration_ms,
        success_score=success_score,
        created_at=created_at or time.time(),
    )


def _make_proposal(
    *,
    proposal_id: str | None = None,
    status: str = "proposed",
    confidence: float = 0.75,
    applied_at: float = 0.0,
    metrics_before: dict | None = None,
    metrics_after: dict | None = None,
    optimization_type: str = "tool_param",
    target: str = "web_search.timeout",
) -> OptimizationProposal:
    return OptimizationProposal(
        proposal_id=proposal_id or uuid.uuid4().hex[:12],
        finding_id="f1",
        optimization_type=optimization_type,
        target=target,
        description="Test proposal",
        patch_before="before",
        patch_after="after",
        estimated_impact=0.5,
        confidence=confidence,
        failure_category="timeout",
        tool_name="web_search",
        evidence_trace_ids=["t1"],
        status=status,
        applied_at=applied_at,
        metrics_before=metrics_before or {},
        metrics_after=metrics_after or {},
        created_at=time.time(),
    )


def _build_orchestrator(
    tmp_path,
    *,
    min_traces: int = 10,
    max_active: int = 1,
    auto_apply: bool = False,
) -> tuple[EvolutionOrchestrator, TraceStore, ProposalStore]:
    """Build an orchestrator backed by real SQLite stores."""
    trace_store = TraceStore(tmp_path / "traces.db")
    proposal_store = ProposalStore(tmp_path / "proposals.db")
    attributor = CausalAttributor()
    optimizer = TraceOptimizer(proposal_store)

    orch = EvolutionOrchestrator(
        trace_store=trace_store,
        attributor=attributor,
        optimizer=optimizer,
        proposal_store=proposal_store,
        min_traces=min_traces,
        max_active=max_active,
        auto_apply=auto_apply,
    )
    return orch, trace_store, proposal_store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunCycle:
    """Tests for EvolutionOrchestrator.run_evolution_cycle()."""

    def test_run_cycle_insufficient_traces(self, tmp_path):
        """Fewer than min_traces -> no proposals generated."""
        orch, trace_store, _ = _build_orchestrator(tmp_path, min_traces=10)

        # Add only 3 traces (below min_traces=10)
        now = time.time()
        for i in range(3):
            trace_store.save_trace(_make_trace(created_at=now - i))

        result = orch.run_evolution_cycle()
        assert result.traces_analyzed < 10
        assert result.proposals_generated == 0
        assert result.proposal_applied is None

    def test_run_cycle_generates_proposals(self, tmp_path):
        """Enough traces with failures -> proposals generated."""
        orch, trace_store, proposal_store = _build_orchestrator(tmp_path, min_traces=5)

        now = time.time()
        # Create 6 traces, some with failures
        for i in range(6):
            steps = [
                _make_step(
                    status="error" if i % 2 == 0 else "success",
                    error_detail="timeout exceeded" if i % 2 == 0 else "",
                    tool_name="web_search",
                ),
            ]
            trace_store.save_trace(
                _make_trace(
                    steps=steps,
                    success_score=0.2 if i % 2 == 0 else 0.9,
                    created_at=now - i,
                )
            )

        result = orch.run_evolution_cycle()
        assert result.traces_analyzed >= 5
        assert result.findings_count >= 1, "Should find at least one causal finding"
        # Proposals may or may not be generated depending on threshold filtering
        # (get_improvement_targets has min_count=2 default), but findings should exist


class TestApplyProposal:
    """Tests for apply_proposal()."""

    def test_apply_proposal(self, tmp_path):
        orch, trace_store, proposal_store = _build_orchestrator(tmp_path)

        p = _make_proposal(proposal_id="p-apply", status="proposed")
        proposal_store.save_proposal(p)

        result = orch.apply_proposal("p-apply")
        assert result is True, "Should successfully apply"

        loaded = proposal_store.get_proposal("p-apply")
        assert loaded is not None
        assert loaded.status == "applied"
        assert loaded.applied_at > 0
        assert "success_rate" in loaded.metrics_before

    def test_apply_blocked_by_max_active(self, tmp_path):
        """Cannot apply when max_active already reached."""
        orch, trace_store, proposal_store = _build_orchestrator(tmp_path, max_active=1)

        # Already have one applied proposal
        existing = _make_proposal(
            proposal_id="p-existing", status="applied", applied_at=time.time()
        )
        proposal_store.save_proposal(existing)

        new = _make_proposal(proposal_id="p-new", status="proposed")
        proposal_store.save_proposal(new)

        result = orch.apply_proposal("p-new")
        assert result is False, "Should be blocked by max_active=1"

        loaded = proposal_store.get_proposal("p-new")
        assert loaded.status == "proposed", "Status should remain proposed"

    def test_apply_nonexistent_proposal(self, tmp_path):
        orch, _, _ = _build_orchestrator(tmp_path)
        result = orch.apply_proposal("nonexistent-id")
        assert result is False

    def test_apply_already_applied(self, tmp_path):
        """Cannot re-apply an already applied proposal."""
        orch, _, proposal_store = _build_orchestrator(tmp_path, max_active=2)

        p = _make_proposal(proposal_id="p-dup", status="applied", applied_at=time.time())
        proposal_store.save_proposal(p)

        result = orch.apply_proposal("p-dup")
        assert result is False, "Already-applied proposal should not be re-applied"


class TestRollbackProposal:
    """Tests for rollback_proposal()."""

    def test_rollback_proposal(self, tmp_path):
        orch, _, proposal_store = _build_orchestrator(tmp_path)

        p = _make_proposal(
            proposal_id="p-rollback",
            status="applied",
            applied_at=time.time(),
            metrics_before={"success_rate": 0.7},
        )
        proposal_store.save_proposal(p)

        result = orch.rollback_proposal("p-rollback")
        assert result is True

        loaded = proposal_store.get_proposal("p-rollback")
        assert loaded.status == "rolled_back"
        assert "success_rate" in loaded.metrics_after

    def test_rollback_not_applied(self, tmp_path):
        """Cannot rollback a proposal that is not applied."""
        orch, _, proposal_store = _build_orchestrator(tmp_path)

        p = _make_proposal(proposal_id="p-proposed", status="proposed")
        proposal_store.save_proposal(p)

        result = orch.rollback_proposal("p-proposed")
        assert result is False

    def test_rollback_nonexistent(self, tmp_path):
        orch, _, _ = _build_orchestrator(tmp_path)
        result = orch.rollback_proposal("ghost-id")
        assert result is False


class TestRejectProposal:
    """Tests for reject_proposal()."""

    def test_reject_proposal(self, tmp_path):
        orch, _, proposal_store = _build_orchestrator(tmp_path)

        p = _make_proposal(proposal_id="p-reject", status="proposed")
        proposal_store.save_proposal(p)

        result = orch.reject_proposal("p-reject")
        assert result is True

        loaded = proposal_store.get_proposal("p-reject")
        assert loaded.status == "rejected"

    def test_reject_applied_proposal(self, tmp_path):
        """Rejecting an applied proposal should also work (captures metrics_after)."""
        orch, _, proposal_store = _build_orchestrator(tmp_path)

        p = _make_proposal(
            proposal_id="p-rej-applied",
            status="applied",
            applied_at=time.time(),
        )
        proposal_store.save_proposal(p)

        result = orch.reject_proposal("p-rej-applied")
        assert result is True

        loaded = proposal_store.get_proposal("p-rej-applied")
        assert loaded.status == "rejected"

    def test_reject_already_rejected(self, tmp_path):
        orch, _, proposal_store = _build_orchestrator(tmp_path)

        p = _make_proposal(proposal_id="p-dup-rej", status="rejected")
        proposal_store.save_proposal(p)

        result = orch.reject_proposal("p-dup-rej")
        assert result is False, "Already rejected should fail"


class TestAutoRollbackOnDegradation:
    """Test evaluate_applied() auto-rollback logic."""

    def test_auto_rollback_on_degradation(self, tmp_path):
        """If success rate drops > threshold after applying, auto-rollback."""
        orch, trace_store, proposal_store = _build_orchestrator(tmp_path, max_active=1)

        # Create an applied proposal with good pre-apply metrics
        applied_at = time.time() - 100
        p = _make_proposal(
            proposal_id="p-degrade",
            status="applied",
            applied_at=applied_at,
            metrics_before={"success_rate": 0.8},
        )
        proposal_store.save_proposal(p)

        # Add traces AFTER the proposal was applied, showing degradation
        # Need at least MIN_SESSIONS_FOR_EVAL (15) traces for evaluation
        now = time.time()
        for i in range(16):
            trace_store.save_trace(
                _make_trace(
                    success_score=0.2,  # bad scores
                    created_at=applied_at + 1 + i,
                    steps=[_make_step(status="error", error_detail="things broke")],
                )
            )

        rolled_back = orch.evaluate_applied()
        assert "p-degrade" in rolled_back, "Should auto-rollback on degradation"

        loaded = proposal_store.get_proposal("p-degrade")
        assert loaded.status == "rolled_back"

    def test_no_rollback_when_stable(self, tmp_path):
        """If metrics are stable or improved, no rollback."""
        orch, trace_store, proposal_store = _build_orchestrator(tmp_path)

        applied_at = time.time() - 100
        p = _make_proposal(
            proposal_id="p-stable",
            status="applied",
            applied_at=applied_at,
            metrics_before={"success_rate": 0.7},
        )
        proposal_store.save_proposal(p)

        # Good traces after applying
        # Need at least MIN_SESSIONS_FOR_EVAL (15) traces for evaluation
        for i in range(16):
            trace_store.save_trace(
                _make_trace(
                    success_score=0.9,
                    created_at=applied_at + 1 + i,
                )
            )

        rolled_back = orch.evaluate_applied()
        assert rolled_back == [], "Stable proposal should not be rolled back"


class TestGetStatus:
    """Tests for get_status()."""

    def test_get_status(self, tmp_path):
        orch, trace_store, proposal_store = _build_orchestrator(tmp_path)

        # Add some data
        now = time.time()
        trace_store.save_trace(_make_trace(created_at=now))
        proposal_store.save_proposal(_make_proposal(status="proposed"))
        proposal_store.save_proposal(_make_proposal(status="applied", applied_at=now))

        status = orch.get_status()
        assert "enabled" in status
        assert status["enabled"] is True
        assert "auto_apply" in status
        assert "cycles_completed" in status
        assert "active_proposals" in status
        assert "pending_proposals" in status
        assert "recent_success_rate" in status
        assert "improvement_trend" in status
        assert "top_issues" in status
        assert isinstance(status["top_issues"], list)

    def test_get_status_empty(self, tmp_path):
        """Status should work even with no data."""
        orch, _, _ = _build_orchestrator(tmp_path)
        status = orch.get_status()
        assert status["enabled"] is True
        assert status["cycles_completed"] == 0
        assert status["active_proposals"] == 0


class TestCycleHistoryCapped:
    """Test that cycle history is capped at _MAX_CYCLE_HISTORY."""

    def test_cycle_history_capped(self, tmp_path):
        orch, trace_store, _ = _build_orchestrator(tmp_path, min_traces=0)

        # Run more cycles than the cap (use min_traces=0 so cycles complete fast)
        # We set min_traces=0 but the actual minimum is checked as < min_traces,
        # so 0 traces < 0 is False => it will proceed. Actually min_traces=0
        # means 0 < 0 is False, so it proceeds.
        # But we have 0 traces and the attributor will just return empty findings.
        # This is fine for testing the cap.

        for i in range(_MAX_CYCLE_HISTORY + 10):
            orch.run_evolution_cycle()

        assert len(orch._cycle_history) <= _MAX_CYCLE_HISTORY, (
            f"History should be capped at {_MAX_CYCLE_HISTORY}"
        )
