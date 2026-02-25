"""Replay Engine: Replays recorded runs against new or modified policies.

Given a previously recorded ``RunRecord``, the replay engine re-evaluates
every planned action through the Gatekeeper with (optionally) different
policy rules.  This enables counterfactual analysis -- "what *would* have
happened if these policies had been in effect?"

Architecture reference: Phase 2 Intelligence -- Run Recording + Replay.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jarvis.models import (
    DecisionDivergence,
    GateDecision,
    PlannedAction,
    ReplayResult,
    RunRecord,
    SessionContext,
)
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.core.gatekeeper import Gatekeeper

log = get_logger(__name__)


class ReplayEngine:
    """Replays historical runs through the Gatekeeper to detect divergences.

    Usage::

        engine = ReplayEngine(gatekeeper)
        result = engine.replay_run(run_record)
        # or with modified policies:
        result = engine.replay_run(run_record, new_policies={"rules": [...]})
        # counterfactual across many policy variants:
        results = engine.counterfactual_analysis(run_record, variants)
    """

    def __init__(self, gatekeeper: Gatekeeper) -> None:
        self._gatekeeper = gatekeeper

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def replay_run(
        self,
        run: RunRecord,
        new_policies: dict[str, Any] | None = None,
        policy_variant_name: str = "",
    ) -> ReplayResult:
        """Replay all plans in *run* through the gatekeeper.

        If *new_policies* is provided, the gatekeeper's policies are
        temporarily replaced before evaluation and restored afterwards.

        Args:
            run: The recorded run to replay.
            new_policies: Optional dict of policy data to apply during replay.
                          When ``None`` the current gatekeeper policies are used.
            policy_variant_name: A human-readable label for this policy variant
                                 (useful in counterfactual reports).

        Returns:
            A ``ReplayResult`` summarising divergences between the original
            and replayed decisions.
        """
        # Build a minimal SessionContext for the replay
        context = SessionContext(
            session_id=run.session_id,
            user_id="replay",
            channel="forensics",
        )

        # Optionally swap gatekeeper policies
        original_policies: list[Any] | None = None
        if new_policies is not None:
            original_policies = self._swap_policies(new_policies)

        try:
            all_divergences: list[DecisionDivergence] = []

            for plan_idx, plan in enumerate(run.plans):
                # Replay each plan's steps through the gatekeeper
                replayed_decisions = self._gatekeeper.evaluate_plan(
                    plan.steps, context
                )

                # Fetch original decisions for this plan (if available)
                original_decisions: list[GateDecision] = []
                if plan_idx < len(run.gate_decisions):
                    original_decisions = run.gate_decisions[plan_idx]

                divergences = self.compare_decisions(
                    original_decisions,
                    replayed_decisions,
                    step_offset=sum(
                        len(p.steps) for p in run.plans[:plan_idx]
                    ),
                )
                all_divergences.extend(divergences)

            # Determine whether the run would still have succeeded.
            # A simple heuristic: if any previously-allowed action is now
            # blocked (or vice-versa), the outcome *might* differ.
            would_have_succeeded: bool | None = None
            if all_divergences:
                # If any allowed action is now blocked, likely would fail
                any_new_block = any(
                    d.replayed_status == "BLOCK" and d.original_status != "BLOCK"
                    for d in all_divergences
                )
                any_new_allow = any(
                    d.original_status == "BLOCK" and d.replayed_status != "BLOCK"
                    for d in all_divergences
                )
                if any_new_block:
                    would_have_succeeded = False
                elif any_new_allow and not run.success:
                    would_have_succeeded = True
                else:
                    would_have_succeeded = run.success
            else:
                would_have_succeeded = run.success

            result = ReplayResult(
                run_id=run.id,
                divergences=all_divergences,
                original_success=run.success,
                would_have_succeeded=would_have_succeeded,
                policy_variant_name=policy_variant_name,
            )

            log.info(
                "replay_completed",
                run_id=run.id,
                divergence_count=len(all_divergences),
                policy_variant=policy_variant_name or "(current)",
            )
            return result

        finally:
            # Always restore original policies
            if original_policies is not None:
                self._restore_policies(original_policies)

    def compare_decisions(
        self,
        original: list[GateDecision],
        replayed: list[GateDecision],
        step_offset: int = 0,
    ) -> list[DecisionDivergence]:
        """Compare two lists of GateDecisions and return divergences.

        Args:
            original: The decisions from the original run.
            replayed: The decisions from the replayed evaluation.
            step_offset: Index offset added to divergence ``step_index``
                         (useful when comparing across multiple plans).

        Returns:
            A list of ``DecisionDivergence`` for every step where the
            original and replayed status differ.
        """
        divergences: list[DecisionDivergence] = []
        max_len = max(len(original), len(replayed))

        for i in range(max_len):
            orig = original[i] if i < len(original) else None
            repl = replayed[i] if i < len(replayed) else None

            orig_status = orig.status.value if orig else "MISSING"
            repl_status = repl.status.value if repl else "MISSING"

            if orig_status != repl_status:
                # Determine the tool name from whichever decision is available
                tool_name = ""
                if orig and orig.original_action:
                    tool_name = orig.original_action.tool
                elif repl and repl.original_action:
                    tool_name = repl.original_action.tool

                divergences.append(
                    DecisionDivergence(
                        step_index=step_offset + i,
                        tool_name=tool_name,
                        original_status=orig_status,
                        replayed_status=repl_status,
                        original_reason=orig.reason if orig else "",
                        replayed_reason=repl.reason if repl else "",
                    )
                )

        return divergences

    def counterfactual_analysis(
        self,
        run: RunRecord,
        policy_variants: dict[str, dict[str, Any]],
    ) -> list[ReplayResult]:
        """Test multiple policy variants against the same recorded run.

        Args:
            run: The recorded run to test against.
            policy_variants: A mapping of variant name to policy dict.
                             Each variant is replayed independently.

        Returns:
            A list of ``ReplayResult`` instances, one per variant, in the
            same order as the input dict.
        """
        results: list[ReplayResult] = []
        for variant_name, policies in policy_variants.items():
            log.info(
                "counterfactual_variant_start",
                run_id=run.id,
                variant=variant_name,
            )
            result = self.replay_run(
                run,
                new_policies=policies,
                policy_variant_name=variant_name,
            )
            results.append(result)

        log.info(
            "counterfactual_analysis_completed",
            run_id=run.id,
            variant_count=len(results),
            total_divergences=sum(len(r.divergences) for r in results),
        )
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _swap_policies(self, new_policies: dict[str, Any]) -> list[Any]:
        """Replace the gatekeeper's policy list and return the originals.

        The ``new_policies`` dict is expected to contain a ``"rules"`` key
        with a list of raw policy dicts (same format as the YAML files).
        Each dict is parsed through the gatekeeper's ``_parse_rule`` method.
        """
        original = self._gatekeeper.get_policies()

        raw_rules = new_policies.get("rules", [])
        parsed: list[Any] = []
        for rule_data in raw_rules:
            try:
                parsed.append(self._gatekeeper._parse_rule(rule_data))
            except Exception as exc:
                log.warning(
                    "replay_policy_parse_error",
                    rule=rule_data.get("name", "?"),
                    error=str(exc),
                )

        self._gatekeeper.set_policies(parsed)
        log.debug("replay_policies_swapped", count=len(parsed))
        return original

    def _restore_policies(self, original_policies: list[Any]) -> None:
        """Restore the gatekeeper's original policy list."""
        self._gatekeeper.set_policies(original_policies)
        log.debug("replay_policies_restored", count=len(original_policies))
