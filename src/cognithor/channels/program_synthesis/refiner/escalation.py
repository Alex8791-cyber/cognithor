# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Spec §6.6 — Refiner Escalation logic (Sprint-1 plan task 9 final slice).

The escalation orchestrator wires the four Refiner stages from
spec §6.6 in the prescribed order:

* **Stage 1 — Local-Edit ALWAYS** (`refiner.local_edit.LocalEditMutator`)
* **Stage 2 — Mode-Dispatch when ``score ≥ 0.3``** (Drei-Zonen-Mode):
    * ``α ≥ 0.45`` → Full-LLM (Two-Stage with retry)
    * ``0.35 ≤ α < 0.45`` → Hybrid (parallel symbolic + single-stage LLM)
    * ``α < 0.35`` → Symbolic-only
* **Stage 3 — CEGIS when ``score ≥ 0.5``** (`refiner.cegis.CEGISLoop`)

Between every stage the verifier re-evaluates: if the candidate
already crosses the success threshold (default ``0.95``) the loop
returns immediately, so we never waste budget on a winning
program.

The *strategies* (local-edit runner, the three repair modes, and
CEGIS) are dependency-injected. Sprint-1 ships the orchestrator
and exhaustive unit tests against stub strategies; later sprints
wire the real backends. Keeping injection at the API surface
means the loop is testable without an LLM, a verifier, or a
running search engine.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from cognithor.channels.program_synthesis.phase2.config import (
    DEFAULT_PHASE2_CONFIG,
    Phase2Config,
)

if TYPE_CHECKING:
    from cognithor.channels.program_synthesis.refiner.mode_controller import (
        RefinerMode,
        RefinerModeController,
    )
    from cognithor.channels.program_synthesis.search.candidate import (
        ProgramNode,
    )


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


RefinementStage = Literal["local", "repair_full_llm", "repair_hybrid", "repair_symbolic", "cegis"]


@dataclass(frozen=True)
class EscalationResult:
    """Outcome of one :meth:`RefinerEscalator.refine` call.

    ``program`` is the best program found across all stages — the
    starting program if no stage produced an improvement, or the
    last successful candidate otherwise.

    ``final_score`` is the verifier's score for ``program``.

    ``terminated_by`` records *why* the loop stopped:
        * ``"success"`` — a stage produced a candidate with
          ``score >= success_threshold``;
        * ``"exhausted"`` — every applicable stage ran without
          producing a new winner.

    ``refinement_path`` is the ordered tuple of stage tags whose
    candidates were *accepted* (improvement over the prior best
    score). Stages that ran but didn't help are absent. The shape
    matches the spec §6.6 ``refinement_path=("local", "repair_…",
    "cegis")``.

    ``mode_selected`` is the Stage-2 zone the mode controller
    picked (or ``None`` if Stage 2 was skipped because
    ``initial_score < mode_min_score``).
    """

    program: ProgramNode
    final_score: float
    terminated_by: Literal["success", "exhausted"]
    refinement_path: tuple[RefinementStage, ...] = field(default_factory=tuple)
    mode_selected: RefinerMode | None = None


# ---------------------------------------------------------------------------
# Strategy callables
# ---------------------------------------------------------------------------


# Each strategy returns either a fresh candidate (Program) or
# ``None`` if it had nothing to offer. The orchestrator scores any
# returned candidate through the injected ``verifier`` and only
# accepts it when it improves on the running best score.
LocalEditRunner = Callable[["ProgramNode"], Awaitable["ProgramNode | None"]]
RepairRunner = Callable[["ProgramNode"], Awaitable["ProgramNode | None"]]
CegisRunner = Callable[["ProgramNode"], Awaitable["ProgramNode | None"]]
Verifier = Callable[["ProgramNode"], Awaitable[float]]
BudgetExhaustedCheck = Callable[[], bool]


@dataclass
class _RunningBest:
    program: ProgramNode
    score: float


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class RefinerEscalator:
    """Spec §6.6 driver — runs Local-Edit → Mode-Dispatch → CEGIS in order.

    All stages are injected at construction so the loop is testable
    without a real verifier / LLM / search engine. The mode
    controller is the only "real" dependency — its hysteresis state
    persists across calls, which is the whole point.
    """

    def __init__(
        self,
        *,
        mode_controller: RefinerModeController,
        local_edit: LocalEditRunner,
        full_llm: RepairRunner,
        hybrid: RepairRunner,
        symbolic: RepairRunner,
        cegis: CegisRunner | None,
        verifier: Verifier,
        refiner_budget_exhausted: BudgetExhaustedCheck = lambda: False,
        cegis_budget_exhausted: BudgetExhaustedCheck = lambda: False,
        config: Phase2Config = DEFAULT_PHASE2_CONFIG,
    ) -> None:
        self._mode_controller = mode_controller
        self._local_edit = local_edit
        self._full_llm = full_llm
        self._hybrid = hybrid
        self._symbolic = symbolic
        self._cegis = cegis
        self._verifier = verifier
        self._refiner_budget_exhausted = refiner_budget_exhausted
        self._cegis_budget_exhausted = cegis_budget_exhausted
        self._config = config

    async def refine(
        self,
        program: ProgramNode,
        initial_score: float,
        current_alpha: float,
        *,
        success_threshold: float = 0.95,
        mode_min_score: float = 0.3,
        cegis_min_score: float = 0.5,
    ) -> EscalationResult:
        """Run Local-Edit → Mode-Dispatch → CEGIS in order; return the best.

        The loop is short-circuited the moment any stage produces a
        candidate at or above ``success_threshold``. Stages that
        *run* but don't improve on the running best score are not
        recorded in ``refinement_path``.
        """
        best = _RunningBest(program=program, score=initial_score)
        path: list[RefinementStage] = []
        mode_selected: RefinerMode | None = None

        # ── STAGE 1: Local-Edit ALWAYS ───────────────────────────
        candidate = await self._local_edit(best.program)
        if candidate is not None:
            score = await self._verifier(candidate)
            if score >= success_threshold:
                path.append("local")
                return EscalationResult(
                    program=candidate,
                    final_score=score,
                    terminated_by="success",
                    refinement_path=tuple(path),
                    mode_selected=mode_selected,
                )
            if score > best.score:
                best = _RunningBest(program=candidate, score=score)
                path.append("local")

        # ── STAGE 2: Mode-Dispatch when score >= 0.3 ─────────────
        if best.score >= mode_min_score and not self._refiner_budget_exhausted():
            mode_selected = self._mode_controller.select_mode(current_alpha)
            runner = self._runner_for_mode(mode_selected)
            stage_tag = self._stage_tag_for_mode(mode_selected)
            candidate = await runner(best.program)
            if candidate is not None:
                score = await self._verifier(candidate)
                if score >= success_threshold:
                    path.append(stage_tag)
                    return EscalationResult(
                        program=candidate,
                        final_score=score,
                        terminated_by="success",
                        refinement_path=tuple(path),
                        mode_selected=mode_selected,
                    )
                if score > best.score:
                    best = _RunningBest(program=candidate, score=score)
                    path.append(stage_tag)

        # ── STAGE 3: CEGIS when score >= 0.5 ─────────────────────
        if (
            self._cegis is not None
            and best.score >= cegis_min_score
            and not self._cegis_budget_exhausted()
        ):
            candidate = await self._cegis(best.program)
            if candidate is not None:
                score = await self._verifier(candidate)
                if score >= success_threshold:
                    path.append("cegis")
                    return EscalationResult(
                        program=candidate,
                        final_score=score,
                        terminated_by="success",
                        refinement_path=tuple(path),
                        mode_selected=mode_selected,
                    )
                if score > best.score:
                    best = _RunningBest(program=candidate, score=score)
                    path.append("cegis")

        return EscalationResult(
            program=best.program,
            final_score=best.score,
            terminated_by="exhausted",
            refinement_path=tuple(path),
            mode_selected=mode_selected,
        )

    # -- Internals ---------------------------------------------------

    def _runner_for_mode(self, mode: RefinerMode) -> RepairRunner:
        if mode == "full_llm":
            return self._full_llm
        if mode == "hybrid":
            return self._hybrid
        return self._symbolic

    def _stage_tag_for_mode(self, mode: RefinerMode) -> RefinementStage:
        if mode == "full_llm":
            return "repair_full_llm"
        if mode == "hybrid":
            return "repair_hybrid"
        return "repair_symbolic"


__all__ = [
    "BudgetExhaustedCheck",
    "CegisRunner",
    "EscalationResult",
    "LocalEditRunner",
    "RefinementStage",
    "RefinerEscalator",
    "RepairRunner",
    "Verifier",
]
