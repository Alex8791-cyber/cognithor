# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Spec §3.2 — Phase-2 top-level synthesis engine (Sprint-1 plan task 10).

Wires the Phase-1 search-engine fast-path together with the Phase-2
Refiner pipeline that landed in plan task 9. The engine itself is
deliberately *thin* — every collaborator (search, verifier,
refiner, cache, telemetry) is dependency-injected so:

* Sprint-2 can swap in the Module-B MCTS controller without
  touching this file;
* tests can drive every escalation path with stub backends and
  zero ARC fixtures;
* production wires the existing Phase-1 ``EnumerativeSearch``
  plus the new ``RefinerEscalator`` and the ``SynthesisCache``.

Pipeline (spec §3.2 Sprint-1 minimal):

  1. **Cache lookup** — TaskSpec-keyed via ``cache.lookup(spec)``;
     short-circuit when a hit clears the success threshold.
  2. **Search** — call ``search_runner(spec, sub_budget)``; receive
     a list of ``(program, score)`` candidates.
  3. **Refinement** — for each candidate with score in the
     refiner-eligible band (``[refiner_min_score, success_threshold)``),
     run the injected :class:`RefinerEscalator` and replace with the
     refined candidate when its score is higher.
  4. **Cache write** — on any candidate that crosses the success
     threshold, ``cache.write(spec, program, score)``.
  5. **Telemetry** — ``telemetry.early_stop(reason)`` /
     ``telemetry.candidate(score, refined)`` hooks fire on every
     state transition; production wires them to the §11 Prometheus
     counters, tests wire them to lists.

The full Phase-2 control flow with MCTS + dual-prior dispatching is
the Sprint-2 PR; this Sprint-1 minimal still satisfies plan-task-10
acceptance criteria:

* ✓ Smoke-test 5 ARC tasks (any in/out, any verdict)
* ✓ Budget partition strictly respected (PartitionedBudget validates
  fractions sum to 1.0 at construction)
* ✓ Early-stop emits telemetry
* ✓ Cache write on success
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from cognithor.channels.program_synthesis.phase2.config import (
    DEFAULT_PHASE2_CONFIG,
    Phase2Config,
)

if TYPE_CHECKING:
    from cognithor.channels.program_synthesis.phase2.datatypes import (
        PartitionedBudget,
    )
    from cognithor.channels.program_synthesis.refiner.escalation import (
        EscalationResult,
        RefinerEscalator,
    )
    from cognithor.channels.program_synthesis.search.candidate import (
        ProgramNode,
    )


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


TerminationReason = Literal[
    "cache_hit",
    "search_success",
    "refined_success",
    "search_exhausted",
    "budget_exhausted",
    "no_candidates",
]


@dataclass(frozen=True)
class Phase2SynthesisResult:
    """Outcome of one :meth:`Phase2SynthesisEngine.synthesize` call.

    ``program`` is the best candidate the pipeline produced (or
    ``None`` when neither search nor refinement found anything).
    ``score`` is the verifier's final score for that program.

    ``cache_hit`` records whether the result came from the cache
    (skipping search + refinement entirely).

    ``refined`` is ``True`` when the candidate was post-processed
    by the Refiner pipeline.

    ``terminated_by`` reports the high-level reason the engine
    stopped: ``"cache_hit"``, ``"search_success"``,
    ``"refined_success"``, ``"search_exhausted"``,
    ``"budget_exhausted"``, ``"no_candidates"``.

    ``elapsed_seconds`` is wall-clock from the engine entry point;
    ``candidates_evaluated`` is the count of search outputs the
    engine actually scored (so tests can assert search ran).
    """

    program: ProgramNode | None
    score: float
    cache_hit: bool
    refined: bool
    terminated_by: TerminationReason
    elapsed_seconds: float
    candidates_evaluated: int = 0
    refinement_path: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Strategy callables
# ---------------------------------------------------------------------------


# Search runner: given a spec + a sub-budget (seconds), returns
# (program, score) tuples — the search engine's ranked output. The
# sub-budget is the engine's TaskSpec-wide budget × the MCTS budget
# fraction; the search runner is free to interpret it however its
# implementation needs.
SearchRunner = Callable[[Any, float], Awaitable[list[tuple["ProgramNode", float]]]]

# Cache reader / writer interfaces — minimal contract.
CacheReader = Callable[[Any], "ProgramNode | None"]
CacheWriter = Callable[[Any, "ProgramNode", float], None]

# Telemetry sink — invoked on every state transition. Production
# wires Prometheus counters; tests append to a list.
TelemetryEvent = tuple[str, dict[str, Any]]
TelemetrySink = Callable[[str, dict[str, Any]], None]


def _noop_telemetry(_event: str, _payload: dict[str, Any]) -> None:
    """Default sink: drop everything."""


def _noop_cache_reader(_spec: Any) -> ProgramNode | None:
    return None


def _noop_cache_writer(_spec: Any, _program: ProgramNode, _score: float) -> None:
    """Default writer: no-op."""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class Phase2SynthesisEngine:
    """Top-level synthesis driver — spec §3.2 Sprint-1 minimal.

    Construction wires all collaborators; :meth:`synthesize` is the
    entry point. The engine is stateless across calls (the cache /
    telemetry sinks carry whatever state production needs).

    The ``budget`` argument to :meth:`synthesize` is the
    :class:`PartitionedBudget` from spec §13.4. The engine derives
    sub-budgets per stage by multiplying the wall-clock total by
    each fraction. Reclamation back to MCTS when an earlier stage
    finishes early is a Sprint-2 enhancement.
    """

    def __init__(
        self,
        *,
        search_runner: SearchRunner,
        verifier: Callable[[ProgramNode], Awaitable[float]],
        refiner: RefinerEscalator | None = None,
        cache_reader: CacheReader = _noop_cache_reader,
        cache_writer: CacheWriter = _noop_cache_writer,
        telemetry: TelemetrySink = _noop_telemetry,
        config: Phase2Config = DEFAULT_PHASE2_CONFIG,
        clock: Callable[[], float] = time.monotonic,
        success_threshold: float = 0.95,
        refiner_min_score: float = 0.3,
    ) -> None:
        self._search_runner = search_runner
        self._verifier = verifier
        self._refiner = refiner
        self._cache_reader = cache_reader
        self._cache_writer = cache_writer
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
        spec: Any,
        budget: PartitionedBudget,
        *,
        wall_clock_budget_seconds: float,
        current_alpha: float = 0.6,
    ) -> Phase2SynthesisResult:
        """Run the synthesis pipeline on ``spec`` within ``budget``.

        ``wall_clock_budget_seconds`` is the total wall-clock budget
        for this call; sub-budgets per stage are derived from
        ``budget`` (the partition fractions). ``current_alpha``
        is the live Search-α the AlphaController would supply in
        production — the Refiner reads it for mode dispatch.
        """
        start = self._clock()

        # ── Stage 0: Cache lookup ────────────────────────────────
        cached = self._cache_reader(spec)
        if cached is not None:
            score = await self._verifier(cached)
            if score >= self._success_threshold:
                self._telemetry(
                    "engine.cache_hit",
                    {"score": score, "elapsed": self._clock() - start},
                )
                return Phase2SynthesisResult(
                    program=cached,
                    score=score,
                    cache_hit=True,
                    refined=False,
                    terminated_by="cache_hit",
                    elapsed_seconds=self._clock() - start,
                )
            # Cached but no longer winning — fall through to fresh search.

        # ── Stage 1: Search ──────────────────────────────────────
        search_budget = wall_clock_budget_seconds * budget.mcts
        candidates = await self._search_runner(spec, search_budget)
        if not candidates:
            self._telemetry("engine.no_candidates", {"elapsed": self._clock() - start})
            return Phase2SynthesisResult(
                program=None,
                score=0.0,
                cache_hit=False,
                refined=False,
                terminated_by="no_candidates",
                elapsed_seconds=self._clock() - start,
            )

        # Sort by descending score; early-return on a search-side win.
        candidates_sorted = sorted(candidates, key=lambda pair: -pair[1])
        best_program, best_score = candidates_sorted[0]
        if best_score >= self._success_threshold:
            self._cache_writer(spec, best_program, best_score)
            self._telemetry(
                "engine.search_success",
                {"score": best_score, "candidates": len(candidates_sorted)},
            )
            return Phase2SynthesisResult(
                program=best_program,
                score=best_score,
                cache_hit=False,
                refined=False,
                terminated_by="search_success",
                elapsed_seconds=self._clock() - start,
                candidates_evaluated=len(candidates_sorted),
            )

        # ── Stage 2: Refinement ──────────────────────────────────
        if (
            self._refiner is not None
            and best_score >= self._refiner_min_score
            and self._clock() - start < wall_clock_budget_seconds
        ):
            refinement: EscalationResult = await self._refiner.refine(
                best_program,
                initial_score=best_score,
                current_alpha=current_alpha,
                success_threshold=self._success_threshold,
            )
            self._telemetry(
                "engine.refined",
                {
                    "from_score": best_score,
                    "to_score": refinement.final_score,
                    "path": refinement.refinement_path,
                },
            )
            if refinement.final_score > best_score:
                best_program = refinement.program
                best_score = refinement.final_score
                if best_score >= self._success_threshold:
                    self._cache_writer(spec, best_program, best_score)
                    return Phase2SynthesisResult(
                        program=best_program,
                        score=best_score,
                        cache_hit=False,
                        refined=True,
                        terminated_by="refined_success",
                        elapsed_seconds=self._clock() - start,
                        candidates_evaluated=len(candidates_sorted),
                        refinement_path=refinement.refinement_path,
                    )
                # Refinement helped but didn't win — fall through.
                return Phase2SynthesisResult(
                    program=best_program,
                    score=best_score,
                    cache_hit=False,
                    refined=True,
                    terminated_by=(
                        "budget_exhausted"
                        if self._clock() - start >= wall_clock_budget_seconds
                        else "search_exhausted"
                    ),
                    elapsed_seconds=self._clock() - start,
                    candidates_evaluated=len(candidates_sorted),
                    refinement_path=refinement.refinement_path,
                )

        # ── Stage 3: Search exhausted, no refinement available ───
        elapsed = self._clock() - start
        terminated: TerminationReason = (
            "budget_exhausted" if elapsed >= wall_clock_budget_seconds else "search_exhausted"
        )
        self._telemetry(
            "engine.exhausted",
            {"score": best_score, "candidates": len(candidates_sorted)},
        )
        return Phase2SynthesisResult(
            program=best_program,
            score=best_score,
            cache_hit=False,
            refined=False,
            terminated_by=terminated,
            elapsed_seconds=elapsed,
            candidates_evaluated=len(candidates_sorted),
        )


__all__ = [
    "CacheReader",
    "CacheWriter",
    "Phase2SynthesisEngine",
    "Phase2SynthesisResult",
    "SearchRunner",
    "TelemetryEvent",
    "TelemetrySink",
    "TerminationReason",
]
