# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Bottom-up enumerative search engine (spec §8).

Builds a typed program bank one depth at a time. At each depth the
engine combines existing bank entries via every primitive whose
signature matches, drops observationally-equivalent duplicates via the
pruner, and returns as soon as it finds a candidate that produces the
expected output on *every* demo example.

Phase 1 is intentionally minimal:

* No LLM prior, no MCTS, no library learning — those land in Phase 2/3.
* No SGN cost-multipliers — the Phase-1 search uses the static DSL costs
  from the registry.
* No early exit on "partial-correct" candidates — only fully-correct
  programs win. Partials are returned at the end as ``status=PARTIAL``
  with the best-scoring candidate found.

The engine is sandbox-agnostic: any object satisfying the
:class:`~cognithor.channels.program_synthesis.search.executor.Executor`
protocol works. Tests pass an ``InProcessExecutor``; production wires
in the real subprocess sandbox in Week 4-5.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cognithor.channels.program_synthesis.core.types import (
    Budget,
    StageResult,
    SynthesisResult,
    SynthesisStatus,
    TaskSpec,
)
from cognithor.channels.program_synthesis.dsl.registry import (
    REGISTRY,
    PrimitiveRegistry,
)
from cognithor.channels.program_synthesis.search.candidate import (
    Const,
    InputRef,
    Program,
    ProgramNode,
)
from cognithor.channels.program_synthesis.search.equivalence import (
    ObservationalEquivalencePruner,
)
from cognithor.channels.program_synthesis.search.executor import (
    Executor,
    InProcessExecutor,
)


@dataclass
class _SearchStats:
    """Mutable counters tracked across one search run."""

    candidates_examined: int = 0
    start_time: float = field(default_factory=time.monotonic)

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.start_time


def _zero_arity_leaves(registry: PrimitiveRegistry) -> dict[str, list[ProgramNode]]:
    """Build the depth-0 bank from zero-arity primitives.

    Zero-arity primitives in the catalog (e.g. ``const_color_5``) are
    rendered as ``Program("const_color_5", (), "Color")`` so they share
    the Program execution path and the equivalence pruner sees them as
    regular tree nodes.
    """
    out: dict[str, list[ProgramNode]] = {}
    for spec in registry.primitives_with_arity(0):
        leaf = Program(
            primitive=spec.name,
            children=(),
            output_type=spec.signature.output,
        )
        out.setdefault(spec.signature.output, []).append(leaf)
    return out


def _outputs_match(actual: Any, expected: Any) -> bool:
    """Tolerant equality used by the demo verifier."""
    if isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray):
        return actual.shape == expected.shape and np.array_equal(actual, expected)
    return actual == expected


def _all_demos_correct(
    program: ProgramNode,
    spec: TaskSpec,
    executor: Executor,
) -> bool:
    """True iff *program* produces the expected output on every demo example."""
    for inp, expected in spec.examples:
        result = executor.execute(program, inp)
        if not result.ok:
            return False
        if not _outputs_match(result.value, expected):
            return False
    return True


def _matching_arg_combos(
    bank: dict[str, list[ProgramNode]],
    sig_inputs: tuple[str, ...],
    target_max_depth: int,
) -> list[tuple[ProgramNode, ...]]:
    """Cartesian product of bank entries matching *sig_inputs*.

    Only returns combinations whose **maximum** child depth equals
    *target_max_depth* — this is how the enumerator generates exactly
    depth-d candidates without re-emitting depth-(d-1) ones.
    """
    if not sig_inputs:
        return [()]

    per_slot: list[list[ProgramNode]] = []
    for tag in sig_inputs:
        candidates = bank.get(tag, [])
        if not candidates:
            return []
        per_slot.append(candidates)

    combos: list[tuple[ProgramNode, ...]] = []
    for combo in itertools.product(*per_slot):
        if max(c.depth() for c in combo) == target_max_depth:
            combos.append(combo)
    return combos


def _seed_color_bank(
    bank: dict[str, list[ProgramNode]],
) -> None:
    """Add 10 ``Const`` color leaves so primitives with Color args resolve.

    Color constants are also registered as zero-arity primitives in the
    catalog (``const_color_0`` … ``const_color_9``), but at the bank
    level we want both representations: ``Const`` for the cheap leaf
    used by recolor / swap_colors arguments, and the Program-form
    constants for direct enumeration.
    """
    for c in range(10):
        bank.setdefault("Color", []).append(Const(value=c, output_type="Color"))


def _budget_exhausted(budget: Budget, stats: _SearchStats) -> bool:
    return (
        stats.candidates_examined >= budget.max_candidates
        or stats.elapsed_seconds >= budget.wall_clock_seconds
    )


@dataclass(frozen=True)
class _CandidateScore:
    """Tracks the best partial candidate found during search."""

    program: ProgramNode
    score: float  # fraction of demos correct (0..1)


def _score_partial(
    program: ProgramNode,
    spec: TaskSpec,
    executor: Executor,
) -> float:
    if not spec.examples:
        return 0.0
    correct = 0
    for inp, expected in spec.examples:
        r = executor.execute(program, inp)
        if r.ok and _outputs_match(r.value, expected):
            correct += 1
    return correct / len(spec.examples)


class EnumerativeSearch:
    """Synchronous bottom-up search engine.

    Parameters are all dependency-injectable so tests can build their
    own minimal registries / executors / pruners. Production wires
    everything from the package-level singletons.
    """

    def __init__(
        self,
        registry: PrimitiveRegistry | None = None,
        executor: Executor | None = None,
        pruner: ObservationalEquivalencePruner | None = None,
    ) -> None:
        self._registry = registry if registry is not None else REGISTRY
        self._executor = (
            executor if executor is not None else InProcessExecutor(registry=self._registry)
        )
        self._pruner = (
            pruner
            if pruner is not None
            else ObservationalEquivalencePruner(executor=self._executor)
        )

    def search(self, spec: TaskSpec, budget: Budget) -> SynthesisResult:
        """Run bottom-up enumeration. See :func:`SynthesisResult` for the
        shape of the return value.

        The verifier trace currently records a single
        ``StageResult(stage="demo", ...)`` summarising whether the
        winning candidate matched all demos. Phase-2 work (CEGIS) and
        Week-4 work (full Verifier-pipeline) replace this with the
        five-stage trace from spec §10.
        """
        self._pruner.reset()
        stats = _SearchStats()

        demo_inputs = tuple(inp for inp, _ in spec.examples)

        # ------------------------------------------------------------
        # Depth 0: input ref + zero-arity primitives + Color constants
        # ------------------------------------------------------------
        bank: dict[str, list[ProgramNode]] = _zero_arity_leaves(self._registry)
        bank.setdefault("Grid", []).append(InputRef())
        _seed_color_bank(bank)

        # Trivial-task fast path: input == output.
        if _all_demos_correct(InputRef(), spec, self._executor):
            return _success(
                program=InputRef(),
                stats=stats,
                cache_hit=False,
            )

        # Register depth-0 leaves with the pruner so duplicates from
        # higher depths get collapsed against them.
        for type_tag, leaves in bank.items():
            for leaf in leaves:
                self._pruner.admit(leaf, type_tag, demo_inputs)

        # ------------------------------------------------------------
        # Depth 1..max_depth
        # ------------------------------------------------------------
        all_primitives = self._registry.all_primitives()

        best_partial: _CandidateScore | None = None

        for depth in range(1, budget.max_depth + 1):
            new_at_depth: dict[str, list[ProgramNode]] = {}

            for prim in all_primitives:
                if prim.signature.arity == 0:
                    # Already in the depth-0 bank.
                    continue

                combos = _matching_arg_combos(
                    bank, prim.signature.inputs, target_max_depth=depth - 1
                )
                for args in combos:
                    if _budget_exhausted(budget, stats):
                        return self._build_partial_or_no_solution(best_partial, stats, budget)

                    candidate = Program(
                        primitive=prim.name,
                        children=args,
                        output_type=prim.signature.output,
                    )
                    stats.candidates_examined += 1

                    type_tag = prim.signature.output
                    if not self._pruner.admit(candidate, type_tag, demo_inputs):
                        continue

                    # Type-correct, reliable, and structurally novel.
                    # Only Grid-output candidates can satisfy the spec
                    # (demos compare grids), so we early-exit only for
                    # Grid programs.
                    if type_tag == "Grid" and _all_demos_correct(candidate, spec, self._executor):
                        return _success(program=candidate, stats=stats, cache_hit=False)

                    if type_tag == "Grid":
                        score = _score_partial(candidate, spec, self._executor)
                        if score > 0 and (best_partial is None or score > best_partial.score):
                            best_partial = _CandidateScore(program=candidate, score=score)

                    new_at_depth.setdefault(type_tag, []).append(candidate)

            # Merge new depth-d candidates into the bank for the next
            # iteration.
            for type_tag, items in new_at_depth.items():
                bank.setdefault(type_tag, []).extend(items)

        return self._build_partial_or_no_solution(best_partial, stats, budget)

    # -- internals ---------------------------------------------------

    def _build_partial_or_no_solution(
        self,
        best_partial: _CandidateScore | None,
        stats: _SearchStats,
        budget: Budget,
    ) -> SynthesisResult:
        """Construct the SynthesisResult when no exact match was found."""
        if _budget_exhausted_clock(budget, stats):
            status = SynthesisStatus.TIMEOUT
        elif stats.candidates_examined >= budget.max_candidates:
            status = SynthesisStatus.BUDGET_EXCEEDED
        elif best_partial is not None:
            status = SynthesisStatus.PARTIAL
        else:
            status = SynthesisStatus.NO_SOLUTION

        if best_partial is not None:
            return SynthesisResult(
                status=status,
                program=best_partial.program,
                score=best_partial.score,
                confidence=0.0,
                cost_seconds=stats.elapsed_seconds,
                cost_candidates=stats.candidates_examined,
                verifier_trace=(
                    StageResult(
                        stage="demo",
                        passed=False,
                        detail=f"{best_partial.score:.0%} of demos matched",
                        duration_ms=stats.elapsed_seconds * 1000.0,
                    ),
                ),
            )
        return SynthesisResult(
            status=status,
            program=None,
            score=0.0,
            confidence=0.0,
            cost_seconds=stats.elapsed_seconds,
            cost_candidates=stats.candidates_examined,
            verifier_trace=(
                StageResult(
                    stage="demo",
                    passed=False,
                    detail="no candidate matched any demo",
                    duration_ms=stats.elapsed_seconds * 1000.0,
                ),
            ),
        )


def _budget_exhausted_clock(budget: Budget, stats: _SearchStats) -> bool:
    """Did the wall-clock cap fire (vs the candidate cap)?"""
    return stats.elapsed_seconds >= budget.wall_clock_seconds


def _success(
    program: ProgramNode,
    stats: _SearchStats,
    cache_hit: bool,
) -> SynthesisResult:
    """Build a SUCCESS SynthesisResult."""
    return SynthesisResult(
        status=SynthesisStatus.SUCCESS,
        program=program,
        score=1.0,
        confidence=1.0,
        cost_seconds=stats.elapsed_seconds,
        cost_candidates=stats.candidates_examined,
        verifier_trace=(
            StageResult(
                stage="demo",
                passed=True,
                detail="all demos matched",
                duration_ms=stats.elapsed_seconds * 1000.0,
            ),
        ),
        cache_hit=cache_hit,
    )


# Re-export to keep imports tidy. Tests construct the engine via
# ``EnumerativeSearch()``; the helpers above stay private.
__all__ = ["EnumerativeSearch"]
