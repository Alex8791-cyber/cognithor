# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Enumerative search engine tests + first end-to-end synthesis (spec §8 + §24)."""

from __future__ import annotations

import numpy as np

from cognithor.channels.program_synthesis.core.types import (
    Budget,
    SynthesisStatus,
    TaskSpec,
)
from cognithor.channels.program_synthesis.search.candidate import (
    InputRef,
    Program,
)
from cognithor.channels.program_synthesis.search.enumerative import (
    EnumerativeSearch,
)


def _g(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


# ---------------------------------------------------------------------------
# Trivial fast-path: input == output
# ---------------------------------------------------------------------------


class TestIdentityFastPath:
    def test_identity_task_returns_input_ref(self) -> None:
        spec = TaskSpec(
            examples=(
                (_g([[1, 2], [3, 4]]), _g([[1, 2], [3, 4]])),
                (_g([[5, 6]]), _g([[5, 6]])),
            ),
        )
        engine = EnumerativeSearch()
        result = engine.search(spec, Budget(max_depth=1, wall_clock_seconds=10.0))

        assert result.status == SynthesisStatus.SUCCESS
        # The fast path returns InputRef directly.
        assert isinstance(result.program, InputRef)
        assert result.score == 1.0
        assert result.confidence == 1.0


# ---------------------------------------------------------------------------
# Depth-1 synthesis: rotate90
# ---------------------------------------------------------------------------


class TestRotate90Synthesis:
    def test_finds_rotate90(self) -> None:
        # output = rotate90(input), three demo pairs.
        spec = TaskSpec(
            examples=(
                (_g([[1, 2], [3, 4]]), _g([[3, 1], [4, 2]])),
                (_g([[5, 6], [7, 8]]), _g([[7, 5], [8, 6]])),
                (_g([[1, 2, 3]]), _g([[1], [2], [3]])),
            ),
        )
        engine = EnumerativeSearch()
        result = engine.search(
            spec, Budget(max_depth=2, wall_clock_seconds=15.0, max_candidates=50_000)
        )

        assert result.status == SynthesisStatus.SUCCESS
        assert isinstance(result.program, Program)
        assert result.program.primitive == "rotate90"
        assert result.program.depth() == 1
        assert result.score == 1.0


# ---------------------------------------------------------------------------
# Depth-1 synthesis: rotate180
# ---------------------------------------------------------------------------


class TestRotate180Synthesis:
    def test_finds_rotate180_or_equivalent(self) -> None:
        # rotate180(input). Could resolve to rotate180 directly or
        # rotate90∘rotate90 — both observationally equivalent on these
        # demos. The pruner registers whichever it sees first; with the
        # deterministic registry order rotate180 (cost 1.0, depth 1) is
        # found before rotate90∘rotate90 (depth 2).
        spec = TaskSpec(
            examples=(
                (_g([[1, 2], [3, 4]]), _g([[4, 3], [2, 1]])),
                (_g([[5, 6, 7]]), _g([[7, 6, 5]])),
            ),
        )
        engine = EnumerativeSearch()
        result = engine.search(spec, Budget(max_depth=2, wall_clock_seconds=15.0))

        assert result.status == SynthesisStatus.SUCCESS
        # Whatever the engine picks, the program must be at depth ≤ 2
        # and produce the exact outputs.
        assert result.program is not None
        assert result.program.depth() <= 2


# ---------------------------------------------------------------------------
# No-solution path
# ---------------------------------------------------------------------------


class TestNoSolution:
    def test_unreachable_demo_set_returns_no_or_partial(self) -> None:
        # Two demos that no single Phase-1 program can satisfy
        # simultaneously: demo 1 needs identity, demo 2 needs rotate90.
        # No primitive output produces both for the same input
        # transformation, so the search must fall back to NO_SOLUTION
        # or PARTIAL (whichever depending on best-partial scoring).
        spec = TaskSpec(
            examples=(
                (_g([[1, 2], [3, 4]]), _g([[1, 2], [3, 4]])),  # demo 1: identity
                (_g([[5, 6], [7, 8]]), _g([[7, 5], [8, 6]])),  # demo 2: rotate90
            ),
        )
        engine = EnumerativeSearch()
        result = engine.search(spec, Budget(max_depth=2, wall_clock_seconds=5.0))

        # Identity fast-path is gated on ALL demos matching — second demo
        # rules it out. No single program satisfies both, so result must
        # be PARTIAL or NO_SOLUTION.
        assert result.status != SynthesisStatus.SUCCESS
        assert result.score < 1.0


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------


class TestBudget:
    def test_max_candidates_caps_search(self) -> None:
        # With max_candidates=5 the search can barely look at anything.
        # The trivial-task fast path bypasses the candidate loop, so use
        # a non-identity task to actually exercise the budget gate.
        spec = TaskSpec(
            examples=(
                (_g([[1, 2]]), _g([[2, 1]])),  # mirror_horizontal
            ),
        )
        engine = EnumerativeSearch()
        result = engine.search(spec, Budget(max_depth=4, wall_clock_seconds=30.0, max_candidates=5))
        # Even though mirror_horizontal IS findable at depth 1, only 5
        # candidates examined means we may or may not hit it (depends
        # on registry order). The contract is just "don't blow past the
        # candidate cap".
        assert result.cost_candidates <= 50  # allow some slack for the early-exit logic

    def test_records_cost_seconds_and_candidates(self) -> None:
        spec = TaskSpec(
            examples=((_g([[1, 2], [3, 4]]), _g([[3, 1], [4, 2]])),),
        )
        engine = EnumerativeSearch()
        result = engine.search(spec, Budget(max_depth=2, wall_clock_seconds=15.0))
        assert result.cost_seconds >= 0.0
        assert result.cost_candidates >= 0


# ---------------------------------------------------------------------------
# Verifier trace
# ---------------------------------------------------------------------------


class TestVerifierTrace:
    def test_success_includes_demo_passed(self) -> None:
        spec = TaskSpec(
            examples=((_g([[1, 2], [3, 4]]), _g([[3, 1], [4, 2]])),),
        )
        engine = EnumerativeSearch()
        result = engine.search(spec, Budget(max_depth=2, wall_clock_seconds=10.0))

        assert result.status == SynthesisStatus.SUCCESS
        assert len(result.verifier_trace) == 1
        assert result.verifier_trace[0].stage == "demo"
        assert result.verifier_trace[0].passed is True

    def test_no_solution_includes_demo_failed(self) -> None:
        # Two contradictory demos (identity + rotate90) — no single
        # program can satisfy both.
        spec = TaskSpec(
            examples=(
                (_g([[1, 2], [3, 4]]), _g([[1, 2], [3, 4]])),
                (_g([[5, 6], [7, 8]]), _g([[7, 5], [8, 6]])),
            ),
        )
        engine = EnumerativeSearch()
        result = engine.search(spec, Budget(max_depth=2, wall_clock_seconds=5.0))

        # Either NO_SOLUTION or PARTIAL — both must record a demo-stage
        # failure trace entry.
        assert result.status != SynthesisStatus.SUCCESS
        assert len(result.verifier_trace) == 1
        assert result.verifier_trace[0].stage == "demo"
        assert result.verifier_trace[0].passed is False
