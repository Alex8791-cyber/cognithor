"""GEPA Evolution Orchestrator — manages the full self-improvement cycle.

Coordinates: collect traces -> analyze root causes -> generate proposals ->
test improvements -> apply or rollback.

Safety constraints:
  - Max 1 active optimization at a time
  - Min 10 traces before proposing
  - Auto-rollback after 5 sessions if no improvement (>10% drop)
  - All changes logged and reversible
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.learning.causal_attributor import CausalAttributor
    from jarvis.learning.execution_trace import TraceStore
    from jarvis.learning.prompt_evolution import PromptEvolutionEngine
    from jarvis.learning.session_analyzer import SessionAnalyzer
    from jarvis.learning.trace_optimizer import ProposalStore, TraceOptimizer

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_CYCLE_HISTORY = 100
_24H_SECONDS = 86400


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class EvolutionCycleResult:
    """Result of one evolution cycle."""

    cycle_id: str
    timestamp: float
    traces_analyzed: int
    findings_count: int
    proposals_generated: int
    proposal_applied: str | None = None  # proposal_id if one was applied
    auto_rollbacks: int = 0
    duration_ms: int = 0


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class EvolutionOrchestrator:
    """Manages the full GEPA self-improvement lifecycle.

    The orchestrator runs evolution cycles either on schedule or on demand.
    Each cycle:
    1. Collects recent execution traces from TraceStore
    2. Runs CausalAttributor to find root causes
    3. Evaluates currently applied proposals (auto-rollback if degraded)
    4. Generates new proposals via TraceOptimizer
    5. Applies the highest-confidence proposal (if above threshold)

    Safety:
    - Only 1 active optimization at a time (max_active_optimizations)
    - Minimum trace count before proposing (min_traces_for_proposal=10)
    - Auto-rollback if success_score drops >10% after applying (rollback_threshold=0.10)
    - All applied changes are logged with before/after metrics
    """

    # Configuration defaults
    MIN_TRACES = 10
    MAX_ACTIVE = 1
    ROLLBACK_THRESHOLD = 0.10  # 10% drop = rollback
    MIN_SESSIONS_FOR_EVAL = 5  # evaluate after 5 sessions
    MIN_CONFIDENCE = 0.6  # minimum confidence to auto-apply

    def __init__(
        self,
        trace_store: TraceStore,
        attributor: CausalAttributor,
        optimizer: TraceOptimizer,
        proposal_store: ProposalStore,
        prompt_evolution: PromptEvolutionEngine | None = None,
        session_analyzer: SessionAnalyzer | None = None,
        *,
        min_traces: int = 10,
        max_active: int = 1,
        rollback_threshold: float = 0.10,
        auto_apply: bool = False,
    ):
        self._trace_store = trace_store
        self._attributor = attributor
        self._optimizer = optimizer
        self._proposals = proposal_store
        self._prompt_evolution = prompt_evolution
        self._session_analyzer = session_analyzer

        self.min_traces = min_traces
        self.max_active = max_active
        self.rollback_threshold = rollback_threshold
        self.auto_apply = auto_apply

        self._last_cycle_time: float = 0
        self._cycle_history: list[EvolutionCycleResult] = []

    # ------------------------------------------------------------------
    # Public API: full cycle
    # ------------------------------------------------------------------

    def run_evolution_cycle(self) -> EvolutionCycleResult:
        """Execute one full evolution cycle.

        Steps:
        1. Get traces since last cycle (or last 24h if first run)
        2. Skip if fewer than min_traces
        3. Evaluate any currently applied proposals
        4. Run causal analysis on recent traces
        5. Get improvement targets
        6. Generate proposals
        7. If auto_apply and best proposal confidence >= MIN_CONFIDENCE, apply it
        8. Return cycle result
        """
        cycle_id = uuid.uuid4().hex[:12]
        start = time.time()
        log.info("evolution_cycle_started", cycle_id=cycle_id)

        traces_analyzed = 0
        findings_count = 0
        proposals_generated = 0
        proposal_applied: str | None = None
        auto_rollbacks = 0

        try:
            # Step 1: Collect traces since last cycle
            since = (
                self._last_cycle_time if self._last_cycle_time > 0 else (time.time() - _24H_SECONDS)
            )
            traces = self._trace_store.get_traces_since(since)
            traces_analyzed = len(traces)

            if traces_analyzed < self.min_traces:
                log.info(
                    "evolution_cycle_skipped_insufficient_traces",
                    cycle_id=cycle_id,
                    traces=traces_analyzed,
                    required=self.min_traces,
                )
                result = EvolutionCycleResult(
                    cycle_id=cycle_id,
                    timestamp=start,
                    traces_analyzed=traces_analyzed,
                    findings_count=0,
                    proposals_generated=0,
                    proposal_applied=None,
                    auto_rollbacks=0,
                    duration_ms=int((time.time() - start) * 1000),
                )
                self._last_cycle_time = time.time()
                self._append_cycle_result(result)
                return result

            # Step 3: Evaluate currently applied proposals (auto-rollback)
            try:
                rolled_back = self.evaluate_applied()
                auto_rollbacks = len(rolled_back)
                if rolled_back:
                    log.info(
                        "evolution_auto_rollbacks",
                        cycle_id=cycle_id,
                        rolled_back=rolled_back,
                    )
            except Exception as exc:
                log.error("evolution_evaluate_applied_failed", cycle_id=cycle_id, error=str(exc))

            # Step 4: Run causal analysis
            try:
                findings = self._attributor.analyze_traces(traces)
                findings_count = len(findings)
                log.info(
                    "evolution_causal_analysis_done",
                    cycle_id=cycle_id,
                    findings=findings_count,
                )
            except Exception as exc:
                log.error("evolution_causal_analysis_failed", cycle_id=cycle_id, error=str(exc))
                findings = []

            # Step 5: Get improvement targets
            try:
                targets = self._attributor.get_improvement_targets(findings)
            except Exception as exc:
                log.error("evolution_get_targets_failed", cycle_id=cycle_id, error=str(exc))
                targets = []

            # Step 6: Generate proposals
            if targets:
                try:
                    new_proposals = self._optimizer.propose_optimizations(
                        targets, self._trace_store
                    )
                    proposals_generated = len(new_proposals)
                    log.info(
                        "evolution_proposals_generated",
                        cycle_id=cycle_id,
                        count=proposals_generated,
                    )
                except Exception as exc:
                    log.error(
                        "evolution_proposal_generation_failed", cycle_id=cycle_id, error=str(exc)
                    )
                    new_proposals = []
            else:
                new_proposals = []

            # Step 7: Auto-apply best proposal if enabled
            if self.auto_apply and new_proposals:
                try:
                    # Sort by confidence descending, pick the best
                    best = max(new_proposals, key=lambda p: p.confidence)
                    if best.confidence >= self.MIN_CONFIDENCE:
                        applied = self.apply_proposal(best.proposal_id)
                        if applied:
                            proposal_applied = best.proposal_id
                            log.info(
                                "evolution_auto_applied",
                                cycle_id=cycle_id,
                                proposal_id=best.proposal_id,
                                confidence=best.confidence,
                            )
                        else:
                            log.info(
                                "evolution_auto_apply_blocked",
                                cycle_id=cycle_id,
                                proposal_id=best.proposal_id,
                                reason="max_active_reached",
                            )
                    else:
                        log.info(
                            "evolution_auto_apply_skipped",
                            cycle_id=cycle_id,
                            best_confidence=best.confidence,
                            required=self.MIN_CONFIDENCE,
                        )
                except Exception as exc:
                    log.error("evolution_auto_apply_failed", cycle_id=cycle_id, error=str(exc))

        except Exception as exc:
            log.error("evolution_cycle_failed", cycle_id=cycle_id, error=str(exc))

        result = EvolutionCycleResult(
            cycle_id=cycle_id,
            timestamp=start,
            traces_analyzed=traces_analyzed,
            findings_count=findings_count,
            proposals_generated=proposals_generated,
            proposal_applied=proposal_applied,
            auto_rollbacks=auto_rollbacks,
            duration_ms=int((time.time() - start) * 1000),
        )
        self._last_cycle_time = time.time()
        self._append_cycle_result(result)

        log.info(
            "evolution_cycle_completed",
            cycle_id=cycle_id,
            traces=traces_analyzed,
            findings=findings_count,
            proposals=proposals_generated,
            applied=proposal_applied,
            rollbacks=auto_rollbacks,
            duration_ms=result.duration_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Public API: evaluate applied proposals
    # ------------------------------------------------------------------

    def evaluate_applied(self) -> list[str]:
        """Evaluate currently applied proposals.

        For each applied proposal:
        1. Get traces since it was applied
        2. If fewer than MIN_SESSIONS_FOR_EVAL traces, skip (too early)
        3. Calculate post-apply success rate
        4. Compare with pre-apply metrics (stored in proposal.metrics_before)
        5. If success rate dropped > rollback_threshold, auto-rollback

        Returns list of rolled-back proposal IDs.
        """
        rolled_back: list[str] = []
        try:
            applied = self._proposals.get_applied()
            if not applied:
                return rolled_back

            for proposal in applied:
                try:
                    # Get traces since proposal was applied
                    if proposal.applied_at <= 0:
                        continue

                    traces_since = self._trace_store.get_traces_since(proposal.applied_at)
                    if len(traces_since) < self.MIN_SESSIONS_FOR_EVAL:
                        log.debug(
                            "evolution_eval_too_early",
                            proposal_id=proposal.proposal_id,
                            traces=len(traces_since),
                            required=self.MIN_SESSIONS_FOR_EVAL,
                        )
                        continue

                    # Calculate post-apply metrics
                    post_metrics = self._compute_metrics_from_traces(traces_since)
                    pre_success = proposal.metrics_before.get("success_rate", 0.0)
                    post_success = post_metrics.get("success_rate", 0.0)

                    # Check for degradation
                    delta = post_success - pre_success
                    if delta < -self.rollback_threshold:
                        log.warning(
                            "evolution_auto_rollback",
                            proposal_id=proposal.proposal_id,
                            pre_success=pre_success,
                            post_success=post_success,
                            delta=round(delta, 4),
                            threshold=-self.rollback_threshold,
                        )
                        if self.rollback_proposal(proposal.proposal_id):
                            rolled_back.append(proposal.proposal_id)
                    else:
                        log.debug(
                            "evolution_proposal_stable",
                            proposal_id=proposal.proposal_id,
                            pre_success=pre_success,
                            post_success=post_success,
                            delta=round(delta, 4),
                        )
                except Exception as exc:
                    log.error(
                        "evolution_eval_proposal_failed",
                        proposal_id=proposal.proposal_id,
                        error=str(exc),
                    )
        except Exception as exc:
            log.error("evolution_evaluate_applied_failed", error=str(exc))

        return rolled_back

    # ------------------------------------------------------------------
    # Public API: apply / rollback / reject proposals
    # ------------------------------------------------------------------

    def apply_proposal(self, proposal_id: str) -> bool:
        """Apply a specific proposal.

        1. Check max_active limit
        2. Get current metrics as baseline (metrics_before)
        3. Mark proposal as "applied" with timestamp
        4. Log the application

        Returns True if applied, False if blocked (too many active).

        NOTE: The actual application of changes (e.g., injecting prompt patches
        into the planner) is done by the gateway, which reads applied proposals
        and applies them at runtime. This method just marks the proposal.
        """
        try:
            # Check max active limit
            currently_applied = self._proposals.get_applied()
            if len(currently_applied) >= self.max_active:
                log.warning(
                    "evolution_apply_blocked",
                    proposal_id=proposal_id,
                    active=len(currently_applied),
                    max_active=self.max_active,
                )
                return False

            # Verify proposal exists and is in proposed state
            proposal = self._proposals.get_proposal(proposal_id)
            if proposal is None:
                log.warning("evolution_apply_not_found", proposal_id=proposal_id)
                return False

            if proposal.status != "proposed":
                log.warning(
                    "evolution_apply_wrong_status",
                    proposal_id=proposal_id,
                    status=proposal.status,
                )
                return False

            # Capture current metrics as baseline
            metrics_before = self._get_current_metrics()

            # Mark as applied
            now = time.time()
            self._proposals.update_status(
                proposal_id,
                "applied",
                applied_at=now,
                metrics_before=metrics_before,
            )

            log.info(
                "evolution_proposal_applied",
                proposal_id=proposal_id,
                optimization_type=proposal.optimization_type,
                target=proposal.target,
                confidence=proposal.confidence,
                metrics_before=metrics_before,
            )
            return True

        except Exception as exc:
            log.error("evolution_apply_failed", proposal_id=proposal_id, error=str(exc))
            return False

    def rollback_proposal(self, proposal_id: str) -> bool:
        """Rollback an applied proposal.

        1. Mark as "rolled_back"
        2. Record metrics_after
        3. Log the rollback

        Returns True if rolled back, False if not found or not applied.
        """
        try:
            proposal = self._proposals.get_proposal(proposal_id)
            if proposal is None:
                log.warning("evolution_rollback_not_found", proposal_id=proposal_id)
                return False

            if proposal.status != "applied":
                log.warning(
                    "evolution_rollback_wrong_status",
                    proposal_id=proposal_id,
                    status=proposal.status,
                )
                return False

            # Capture current metrics
            metrics_after = self._get_current_metrics()

            # Mark as rolled back
            self._proposals.update_status(
                proposal_id,
                "rolled_back",
                metrics_after=metrics_after,
            )

            log.info(
                "evolution_proposal_rolled_back",
                proposal_id=proposal_id,
                optimization_type=proposal.optimization_type,
                target=proposal.target,
                metrics_before=proposal.metrics_before,
                metrics_after=metrics_after,
            )
            return True

        except Exception as exc:
            log.error("evolution_rollback_failed", proposal_id=proposal_id, error=str(exc))
            return False

    def reject_proposal(self, proposal_id: str) -> bool:
        """Manually reject a proposal."""
        try:
            proposal = self._proposals.get_proposal(proposal_id)
            if proposal is None:
                log.warning("evolution_reject_not_found", proposal_id=proposal_id)
                return False

            if proposal.status not in ("proposed", "applied"):
                log.warning(
                    "evolution_reject_wrong_status",
                    proposal_id=proposal_id,
                    status=proposal.status,
                )
                return False

            # If it was applied, capture metrics_after before rejecting
            if proposal.status == "applied":
                metrics_after = self._get_current_metrics()
                self._proposals.update_status(
                    proposal_id,
                    "rejected",
                    metrics_after=metrics_after,
                )
            else:
                self._proposals.update_status(proposal_id, "rejected")

            log.info("evolution_proposal_rejected", proposal_id=proposal_id)
            return True

        except Exception as exc:
            log.error("evolution_reject_failed", proposal_id=proposal_id, error=str(exc))
            return False

    # ------------------------------------------------------------------
    # Public API: status and details
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Get current GEPA evolution status.

        Returns:
        {
            "enabled": bool,
            "auto_apply": bool,
            "last_cycle": timestamp or None,
            "cycles_completed": int,
            "active_proposals": int,
            "pending_proposals": int,
            "total_applied": int,
            "total_rolled_back": int,
            "total_rejected": int,
            "recent_success_rate": float,  # from last 50 traces
            "improvement_trend": float,    # delta success rate vs 7 days ago
            "top_issues": list[dict],      # from last cycle's findings
        }
        """
        try:
            active = self._proposals.get_applied()
            pending = self._proposals.get_pending()
            rolled_back = self._proposals.get_by_status("rolled_back")
            rejected = self._proposals.get_by_status("rejected")

            # History counts include current state
            history = self._proposals.get_history(limit=1000)
            total_applied = sum(
                1
                for p in history
                if p.status in ("applied", "rolled_back", "rejected") and p.applied_at > 0
            )

            # Current success rate
            current_metrics = self._get_current_metrics()
            recent_success_rate = current_metrics.get("success_rate", 0.0)

            # Improvement trend: compare current vs 7 days ago
            seven_days_ago = time.time() - (7 * _24H_SECONDS)
            historical_metrics = self._get_historical_metrics(seven_days_ago)
            historical_success = historical_metrics.get("success_rate", 0.0)
            improvement_trend = recent_success_rate - historical_success

            # Top issues from last cycle
            top_issues: list[dict[str, Any]] = []
            if self._cycle_history:
                last_cycle = self._cycle_history[-1]
                # Re-run causal analysis to get the top issues for status
                # (lightweight — use cached findings if available)
                try:
                    since = last_cycle.timestamp - _24H_SECONDS
                    traces = self._trace_store.get_traces_since(since)
                    if traces:
                        findings = self._attributor.analyze_traces(traces)
                        targets = self._attributor.get_improvement_targets(findings)
                        top_issues = targets[:5]
                except Exception:
                    pass  # Non-critical for status

            return {
                "enabled": True,
                "auto_apply": self.auto_apply,
                "last_cycle": self._last_cycle_time if self._last_cycle_time > 0 else None,
                "cycles_completed": len(self._cycle_history),
                "active_proposals": len(active),
                "pending_proposals": len(pending),
                "total_applied": total_applied,
                "total_rolled_back": len(rolled_back),
                "total_rejected": len(rejected),
                "recent_success_rate": round(recent_success_rate, 4),
                "improvement_trend": round(improvement_trend, 4),
                "top_issues": top_issues,
            }
        except Exception as exc:
            log.error("evolution_get_status_failed", error=str(exc))
            return {
                "enabled": True,
                "auto_apply": self.auto_apply,
                "last_cycle": self._last_cycle_time if self._last_cycle_time > 0 else None,
                "cycles_completed": len(self._cycle_history),
                "active_proposals": 0,
                "pending_proposals": 0,
                "total_applied": 0,
                "total_rolled_back": 0,
                "total_rejected": 0,
                "recent_success_rate": 0.0,
                "improvement_trend": 0.0,
                "top_issues": [],
            }

    def get_proposal_detail(self, proposal_id: str) -> dict[str, Any] | None:
        """Get full proposal details including metrics."""
        try:
            proposal = self._proposals.get_proposal(proposal_id)
            if proposal is None:
                return None

            return {
                "proposal_id": proposal.proposal_id,
                "finding_id": proposal.finding_id,
                "optimization_type": proposal.optimization_type,
                "target": proposal.target,
                "description": proposal.description,
                "patch_before": proposal.patch_before,
                "patch_after": proposal.patch_after,
                "estimated_impact": proposal.estimated_impact,
                "confidence": proposal.confidence,
                "failure_category": proposal.failure_category,
                "tool_name": proposal.tool_name,
                "evidence_trace_ids": proposal.evidence_trace_ids,
                "status": proposal.status,
                "applied_at": proposal.applied_at if proposal.applied_at > 0 else None,
                "metrics_before": proposal.metrics_before,
                "metrics_after": proposal.metrics_after,
                "created_at": proposal.created_at,
            }
        except Exception as exc:
            log.error(
                "evolution_get_proposal_detail_failed", proposal_id=proposal_id, error=str(exc)
            )
            return None

    # ------------------------------------------------------------------
    # Private: metrics calculation
    # ------------------------------------------------------------------

    def _get_current_metrics(self) -> dict[str, float]:
        """Calculate current success metrics from recent traces.

        Returns {"success_rate": float, "avg_duration_ms": float, "error_rate": float}
        from the last 50 traces.
        """
        try:
            traces = self._trace_store.get_recent_traces(limit=50)
            return self._compute_metrics_from_traces(traces)
        except Exception as exc:
            log.error("evolution_get_current_metrics_failed", error=str(exc))
            return {"success_rate": 0.0, "avg_duration_ms": 0.0, "error_rate": 0.0}

    def _get_historical_metrics(self, since: float) -> dict[str, float]:
        """Calculate metrics from traces since a given timestamp."""
        try:
            traces = self._trace_store.get_traces_since(since)
            return self._compute_metrics_from_traces(traces)
        except Exception as exc:
            log.error("evolution_get_historical_metrics_failed", error=str(exc))
            return {"success_rate": 0.0, "avg_duration_ms": 0.0, "error_rate": 0.0}

    @staticmethod
    def _compute_metrics_from_traces(traces: list[Any]) -> dict[str, float]:
        """Compute success_rate, avg_duration_ms, and error_rate from a list of traces."""
        if not traces:
            return {"success_rate": 0.0, "avg_duration_ms": 0.0, "error_rate": 0.0}

        total = len(traces)
        success_count = 0
        error_count = 0
        total_duration = 0.0

        for trace in traces:
            score = getattr(trace, "success_score", 0.0) or 0.0
            if score >= 0.5:
                success_count += 1

            duration = getattr(trace, "total_duration_ms", 0) or 0
            total_duration += duration

            # Count traces with any failed steps
            failed_steps = getattr(trace, "failed_steps", [])
            if failed_steps:
                error_count += 1

        success_rate = success_count / total
        avg_duration = total_duration / total
        error_rate = error_count / total

        return {
            "success_rate": round(success_rate, 4),
            "avg_duration_ms": round(avg_duration, 1),
            "error_rate": round(error_rate, 4),
        }

    # ------------------------------------------------------------------
    # Private: cycle history management
    # ------------------------------------------------------------------

    def _append_cycle_result(self, result: EvolutionCycleResult) -> None:
        """Append a cycle result, keeping at most _MAX_CYCLE_HISTORY entries."""
        self._cycle_history.append(result)
        if len(self._cycle_history) > _MAX_CYCLE_HISTORY:
            self._cycle_history = self._cycle_history[-_MAX_CYCLE_HISTORY:]
