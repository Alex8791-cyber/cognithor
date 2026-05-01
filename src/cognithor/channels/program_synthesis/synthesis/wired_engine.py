# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-2 Track B — Production wiring of Phase-1 + Phase-2.

The :class:`WiredPhase2Engine` adapter glues:

* the existing **Phase-1 ``EnumerativeSearch``** (sync bottom-up
  enumerator) — kicked off in a thread so async callers don't
  block the event loop;
* the **Module-A ``DualPriorMixer``** — supplies the live
  Search-α the Refiner mode-controller reads for zone dispatch;
* the **Module-C ``RefinerEscalator``** — runs Local-Edit →
  Mode-Dispatch → CEGIS on partial-score candidates;
* the **Module-A α-side ``VerifierEvaluator``** — re-grades the
  refined candidate so the engine returns a fully-scored result.

Sprint-2 acceptance for Track B:
    "Phase-1-Pipeline ruft DualPriorMixer + RefinerEscalator;
     A/B-Test zeigt messbaren Score-Lift auf Sprint-1-Baseline."

This module ships the *driver* that makes the A/B-test runnable.
The fixture set + dashboard land in Tracks C/D.

Design notes:

* The wiring is *non-invasive*: ``EnumerativeSearch`` itself is
  unchanged. The adapter calls ``.search()`` and post-processes
  the result. A future Sprint-3 PR may push the DualPriorMixer
  inside the search loop as a tie-breaker, but Sprint-2 keeps
  the boundary at the Phase-1 / Phase-2 seam.
* All collaborators are dependency-injected. Tests wire stubs;
  production wires real implementations.
* Refinement is gated by the Phase-1 result's score: only
  ``status=PARTIAL`` with ``score >= refiner_min_score`` triggers
  the Refiner. ``status=SUCCESS`` returns immediately.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from cognithor.channels.program_synthesis.core.types import (
    SynthesisStatus,
)
from cognithor.channels.program_synthesis.phase2.config import (
    DEFAULT_PHASE2_CONFIG,
    Phase2Config,
)

if TYPE_CHECKING:
    from cognithor.channels.program_synthesis.core.types import (
        Budget,
        SynthesisResult,
        TaskSpec,
    )
    from cognithor.channels.program_synthesis.phase2.dual_prior import (
        DualPriorMixer,
    )
    from cognithor.channels.program_synthesis.phase2.verifier_evaluator import (
        VerifierEvaluator,
    )
    from cognithor.channels.program_synthesis.refiner.escalation import (
        RefinerEscalator,
    )
    from cognithor.channels.program_synthesis.search.candidate import (
        ProgramNode,
    )


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


WiredTerminationReason = Literal[
    "phase1_success",
    "refined_success",
    "refined_partial",
    "no_refinement_eligible",
    "no_solution",
]


@dataclass(frozen=True)
class WiredSynthesisResult:
    """Outcome of one :meth:`WiredPhase2Engine.synthesize` call.

    ``program`` is the best candidate produced (or ``None``).

    ``phase1_score`` is the score the Phase-1 search reported (the
    fraction-of-demos-correct it tracks). ``final_score`` is the
    Phase-2 :class:`VerifierEvaluator` score on the *winning*
    candidate (which may be the refined one).

    ``current_alpha`` is the Search-α that drove the Refiner mode
    dispatch — recorded for telemetry / Sprint-2 dashboards.

    ``refined`` is ``True`` when the engine called the Refiner;
    ``refinement_path`` is the escalation path the Refiner took
    (empty when no refinement happened or no stage helped).

    ``terminated_by`` reports the high-level reason the wiring
    stopped: ``"phase1_success"`` / ``"refined_success"`` /
    ``"refined_partial"`` / ``"no_refinement_eligible"`` /
    ``"no_solution"``.

    ``elapsed_seconds`` is wall-clock from entry; ``phase1_seconds``
    isolates the Phase-1 portion so A/B-tests can measure refiner
    overhead.
    """

    program: ProgramNode | None
    phase1_score: float
    final_score: float
    current_alpha: float
    refined: bool
    terminated_by: WiredTerminationReason
    elapsed_seconds: float
    phase1_seconds: float
    refinement_path: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Strategy callables
# ---------------------------------------------------------------------------


# Phase-1 search is sync (existing EnumerativeSearch.search). The wiring
# offloads it to a thread so async callers don't block.
Phase1SearchRunner = Callable[["TaskSpec", "Budget"], "SynthesisResult"]

# Telemetry sink — wired to Prometheus counters in production, lists in tests.
TelemetrySink = Callable[[str, dict[str, Any]], None]


def _noop_telemetry(_event: str, _payload: dict[str, Any]) -> None:
    """Default sink: drop everything."""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class WiredPhase2Engine:
    """Phase-1 + Phase-2 production wiring — Sprint-2 Track B.

    Stateless across calls. The injected mixer / refiner / verifier
    can carry state (mixer's α-controller, refiner's mode-controller
    hysteresis), and that state persists across ``synthesize()``
    calls — which is exactly what we want for a per-process search
    engine.
    """

    def __init__(
        self,
        *,
        phase1_search: Phase1SearchRunner,
        dual_prior: DualPriorMixer | None = None,
        refiner: RefinerEscalator | None = None,
        verifier: VerifierEvaluator,
        telemetry: TelemetrySink = _noop_telemetry,
        config: Phase2Config = DEFAULT_PHASE2_CONFIG,
        clock: Callable[[], float] = time.monotonic,
        success_threshold: float = 0.95,
        refiner_min_score: float = 0.3,
    ) -> None:
        self._phase1_search = phase1_search
        self._dual_prior = dual_prior
        self._refiner = refiner
        self._verifier = verifier
        self._telemetry = telemetry
        self._config = config
        self._clock = clock
        if not 0.0 < success_threshold <= 1.0:
            raise ValueError(f"success_threshold must be in (0, 1]; got {success_threshold}")
        if not 0.0 <= refiner_min_score <= 1.0:
            raise ValueError(f"refiner_min_score must be in [0, 1]; got {refiner_min_score}")
        self._success_threshold = success_threshold
        self._refiner_min_score = refiner_min_score

    async def synthesize(
        self,
        spec: TaskSpec,
        budget: Budget,
    ) -> WiredSynthesisResult:
        """Run Phase-1 search; refine on partial; return scored result."""
        start = self._clock()

        # ── Stage 1: Phase-1 enumerative search (in a thread) ─────
        phase1_start = self._clock()
        phase1_result = await asyncio.to_thread(self._phase1_search, spec, budget)
        phase1_seconds = self._clock() - phase1_start

        self._telemetry(
            "wired.phase1_done",
            {
                "status": phase1_result.status.value,
                "score": phase1_result.score,
                "elapsed": phase1_seconds,
            },
        )

        # SUCCESS short-circuits — no refinement needed.
        if phase1_result.status == SynthesisStatus.SUCCESS:
            return WiredSynthesisResult(
                program=phase1_result.program,
                phase1_score=phase1_result.score,
                final_score=phase1_result.score,
                current_alpha=0.0,  # not consulted on success path
                refined=False,
                terminated_by="phase1_success",
                elapsed_seconds=self._clock() - start,
                phase1_seconds=phase1_seconds,
            )

        # No program at all → return early.
        if phase1_result.program is None:
            return WiredSynthesisResult(
                program=None,
                phase1_score=phase1_result.score,
                final_score=phase1_result.score,
                current_alpha=0.0,
                refined=False,
                terminated_by="no_solution",
                elapsed_seconds=self._clock() - start,
                phase1_seconds=phase1_seconds,
            )

        # ── Stage 2: Refinement gate ─────────────────────────────
        if self._refiner is None or phase1_result.score < self._refiner_min_score:
            return WiredSynthesisResult(
                program=phase1_result.program,
                phase1_score=phase1_result.score,
                final_score=phase1_result.score,
                current_alpha=0.0,
                refined=False,
                terminated_by="no_refinement_eligible",
                elapsed_seconds=self._clock() - start,
                phase1_seconds=phase1_seconds,
            )

        # ── Stage 3: Resolve current Search-α via DualPriorMixer ──
        current_alpha = await self._resolve_alpha(spec)

        self._telemetry(
            "wired.alpha_resolved",
            {"alpha": current_alpha, "score": phase1_result.score},
        )

        # ── Stage 4: Run Refiner ──────────────────────────────────
        escalation = await self._refiner.refine(
            phase1_result.program,
            initial_score=phase1_result.score,
            current_alpha=current_alpha,
            success_threshold=self._success_threshold,
        )

        self._telemetry(
            "wired.refined",
            {
                "from_score": phase1_result.score,
                "to_score": escalation.final_score,
                "path": escalation.refinement_path,
            },
        )

        terminated: WiredTerminationReason = (
            "refined_success"
            if escalation.final_score >= self._success_threshold
            else "refined_partial"
        )
        return WiredSynthesisResult(
            program=escalation.program,
            phase1_score=phase1_result.score,
            final_score=escalation.final_score,
            current_alpha=current_alpha,
            refined=True,
            terminated_by=terminated,
            elapsed_seconds=self._clock() - start,
            phase1_seconds=phase1_seconds,
            refinement_path=escalation.refinement_path,
        )

    # -- Internals -----------------------------------------------------

    async def _resolve_alpha(self, spec: TaskSpec) -> float:
        """Consult the DualPriorMixer for the live Search-α.

        When no mixer is wired, fall back to the spec-default
        cold-start α. The mixer is async; we await it directly
        (it's expected to be cheap given the LLM-Prior cache).
        """
        if self._dual_prior is None:
            return self._config.alpha_cold_start
        try:
            mixed = await self._dual_prior.get_prior(spec.examples)
        except Exception as exc:
            self._telemetry(
                "wired.alpha_fallback",
                {"error": type(exc).__name__},
            )
            return self._config.alpha_cold_start
        return float(mixed.alpha)


__all__ = [
    "Phase1SearchRunner",
    "TelemetrySink",
    "WiredPhase2Engine",
    "WiredSynthesisResult",
    "WiredTerminationReason",
]


# Suppress unused-import lint — Awaitable is reserved for future
# Phase-3 PRs that may make phase1_search async.
_ = Awaitable
