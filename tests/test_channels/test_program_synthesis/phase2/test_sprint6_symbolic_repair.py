# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-6 Track A — Symbolic-Repair-Advisor live experiment tests."""

from __future__ import annotations

import numpy as np

from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.search.candidate import (
    InputRef,
    Program,
)
from cognithor.channels.program_synthesis.search.executor import InProcessExecutor
from cognithor.channels.program_synthesis.synthesis.sprint6_symbolic_repair_runner import (
    _candidates_from_hint,
    _parse_args,
    _score_program_on_demos,
    _try_symbolic_repair,
)


def _g(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


# ---------------------------------------------------------------------------
# _candidates_from_hint — replacement vs wrap, arity-aware
# ---------------------------------------------------------------------------


class TestCandidatesFromHint:
    def test_unary_replacement(self) -> None:
        cands = _candidates_from_hint("rotate90", [], [])
        assert len(cands) == 1
        assert cands[0].primitive == "rotate90"
        assert isinstance(cands[0].children[0], InputRef)

    def test_unary_wrap(self) -> None:
        base = Program("rotate180", (InputRef(),), "Grid")
        cands = _candidates_from_hint("rotate90", [], [], base=base)
        assert len(cands) == 1
        assert cands[0].primitive == "rotate90"
        # Child is the wrapped Phase-1 program.
        inner = cands[0].children[0]
        assert isinstance(inner, Program)
        assert inner.primitive == "rotate180"

    def test_recolor_arg_synthesis(self) -> None:
        cands = _candidates_from_hint(
            "recolor",
            palette_actual=[1, 2, 3],
            palette_expected=[5, 6],
        )
        # 3 srcs × 2 dsts (no equal pairs) = 6 candidates.
        assert len(cands) == 6
        for c in cands:
            assert c.primitive == "recolor"
            assert len(c.children) == 3

    def test_unknown_primitive_returns_empty(self) -> None:
        assert _candidates_from_hint("not_a_real_primitive", [], []) == []


# ---------------------------------------------------------------------------
# _score_program_on_demos
# ---------------------------------------------------------------------------


class TestScoreProgramOnDemos:
    def test_perfect_program(self) -> None:
        program = Program("identity", (InputRef(),), "Grid")
        demos = [(_g([[1, 2]]), _g([[1, 2]])), (_g([[3, 4]]), _g([[3, 4]]))]
        score = _score_program_on_demos(program, demos, InProcessExecutor())
        assert score == 1.0

    def test_partial_program(self) -> None:
        # rotate90 of [[1,2]] = [[1],[2]] — different shape, no match.
        program = Program("rotate90", (InputRef(),), "Grid")
        demos = [
            (_g([[1, 2], [3, 4]]), _g([[3, 1], [4, 2]])),  # rotate90 cw matches
            (_g([[1, 2]]), _g([[1, 2]])),  # rotate90 changes shape, no match
        ]
        score = _score_program_on_demos(program, demos, InProcessExecutor())
        assert score == 0.5

    def test_empty_demos(self) -> None:
        program = Program("identity", (InputRef(),), "Grid")
        assert _score_program_on_demos(program, [], InProcessExecutor()) == 0.0


# ---------------------------------------------------------------------------
# _try_symbolic_repair — the actual winning strategy on task 0202
# ---------------------------------------------------------------------------


class TestTrySymbolicRepair:
    def test_no_demos_returns_unchanged(self) -> None:
        program = Program("rotate90", (InputRef(),), "Grid")
        result, score, refined = _try_symbolic_repair(program, [], InProcessExecutor())
        assert result == program
        assert score == 0.0
        assert refined is False

    def test_already_winning_no_refinement(self) -> None:
        program = Program("identity", (InputRef(),), "Grid")
        demos = [(_g([[1, 2]]), _g([[1, 2]]))]
        result, score, refined = _try_symbolic_repair(program, demos, InProcessExecutor())
        assert refined is False
        assert score == 1.0

    def test_two_step_recolor_cascade_solves_color_drop_task(self) -> None:
        # Synthetic "remove colors 2 and 3" task — same shape as
        # the 0202_largest_object_only sub-problem the Sprint-6
        # cascade was designed for.
        demos = [
            (
                _g([[1, 1, 0, 2], [1, 1, 0, 0], [0, 0, 0, 3]]),
                _g([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]]),
            ),
            (_g([[5, 0, 0], [5, 5, 0], [5, 0, 0]]), _g([[5, 0, 0], [5, 5, 0], [5, 0, 0]])),
        ]
        # Phase-1 found mirror_vertical (wrong base, but produces
        # 0.5 partial score). The cascade should beat that with
        # a recolor-based program.
        phase1 = Program("mirror_vertical", (InputRef(),), "Grid")
        result, score, refined = _try_symbolic_repair(phase1, demos, InProcessExecutor())
        assert refined is True
        assert score >= 0.95  # crossed the success threshold


# ---------------------------------------------------------------------------
# CLI argparse contract
# ---------------------------------------------------------------------------


class TestArgparse:
    def test_default_subset_is_hard(self) -> None:
        args = _parse_args([])
        assert args.subset == "hard"

    def test_custom_subset(self) -> None:
        args = _parse_args(["--subset", "train"])
        assert args.subset == "train"

    def test_default_corpus_root(self) -> None:
        from pathlib import Path

        args = _parse_args([])
        assert args.corpus_root == Path("cognithor_bench/arc_agi3")
