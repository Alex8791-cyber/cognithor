# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Trace-Replay tests (Sprint-1 plan task 9 slice, spec §6.4)."""

from __future__ import annotations

from typing import Any

import numpy as np

from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.refiner import (
    TraceStep,
    find_divergence,
    find_first_failure,
    replay_trace,
)
from cognithor.channels.program_synthesis.search.candidate import (
    Const,
    InputRef,
    Program,
    ProgramNode,
)
from cognithor.channels.program_synthesis.search.executor import (
    ExecutionResult,
    InProcessExecutor,
)


def _g(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


# ---------------------------------------------------------------------------
# Replay — basic post-order walk
# ---------------------------------------------------------------------------


class TestReplayTrace:
    def test_input_ref_alone_yields_one_step(self) -> None:
        grid = _g([[1, 2], [3, 4]])
        trace = replay_trace(InputRef(), grid)
        assert len(trace) == 1
        step = trace[0]
        assert step.path == ()
        assert step.ok is True
        assert np.array_equal(step.value, grid)
        assert step.error is None

    def test_const_alone_yields_one_step(self) -> None:
        node = Const(value=7, output_type="Color")
        trace = replay_trace(node, _g([[0]]))
        assert len(trace) == 1
        assert trace[0].value == 7

    def test_post_order_for_unary_program(self) -> None:
        # rotate90(input)
        program = Program(
            primitive="rotate90",
            children=(InputRef(),),
            output_type="Grid",
        )
        grid = _g([[1, 2], [3, 4]])
        trace = replay_trace(program, grid)
        # 2 steps: child first, then root.
        assert len(trace) == 2
        # Child = InputRef at path (0,)
        assert trace[0].path == (0,)
        assert isinstance(trace[0].node, InputRef)
        # Root = Program at path ()
        assert trace[1].path == ()
        assert isinstance(trace[1].node, Program)
        # Final root output == np.rot90(grid, k=-1).
        assert np.array_equal(trace[1].value, np.rot90(grid, k=-1))

    def test_post_order_for_three_arg_program(self) -> None:
        # recolor(input, 1, 5) — 4 steps total (3 children + root).
        program = Program(
            primitive="recolor",
            children=(
                InputRef(),
                Const(value=1, output_type="Color"),
                Const(value=5, output_type="Color"),
            ),
            output_type="Grid",
        )
        grid = _g([[1, 2], [1, 3]])
        trace = replay_trace(program, grid)
        assert [s.path for s in trace] == [(0,), (1,), (2,), ()]
        # Root output: every 1 → 5.
        expected = _g([[5, 2], [5, 3]])
        assert trace[-1].ok
        assert np.array_equal(trace[-1].value, expected)


class TestReplayTraceNested:
    def test_nested_program_paths(self) -> None:
        # rotate180(rotate90(input)) = rotate270(input).
        inner = Program(
            primitive="rotate90",
            children=(InputRef(),),
            output_type="Grid",
        )
        outer = Program(
            primitive="rotate180",
            children=(inner,),
            output_type="Grid",
        )
        grid = _g([[1, 2], [3, 4]])
        trace = replay_trace(outer, grid)
        # Steps in post-order: InputRef(0,0), rotate90(0), rotate180().
        assert [s.path for s in trace] == [(0, 0), (0,), ()]
        # Innermost subtree is just the input grid.
        assert np.array_equal(trace[0].value, grid)
        # Middle subtree = np.rot90(grid, k=-1).
        assert np.array_equal(trace[1].value, np.rot90(grid, k=-1))
        # Root = rotate180 of that = rot90(rot90(grid)).
        # rotate180(rotate90(grid)) = np.rot90(grid, k=-1+2) = np.rot90(grid, k=1)
        assert np.array_equal(trace[2].value, np.rot90(grid, k=1))


# ---------------------------------------------------------------------------
# find_divergence — locates the deepest matching subtree
# ---------------------------------------------------------------------------


class TestFindDivergence:
    def test_root_matches_when_program_is_correct(self) -> None:
        program = Program(
            primitive="rotate90",
            children=(InputRef(),),
            output_type="Grid",
        )
        grid = _g([[1, 2], [3, 4]])
        expected = np.rot90(grid, k=-1)
        trace = replay_trace(program, grid)
        d = find_divergence(trace, expected)
        assert d is not None
        # The root step matches.
        assert d.path == ()

    def test_inner_subtree_matches_when_program_overshoots(self) -> None:
        # rotate180(rotate90(input)) — but the *expected* output is
        # what rotate90 alone produces. The deepest matching subtree
        # is the inner rotate90.
        inner = Program(
            primitive="rotate90",
            children=(InputRef(),),
            output_type="Grid",
        )
        outer = Program(
            primitive="rotate180",
            children=(inner,),
            output_type="Grid",
        )
        grid = _g([[1, 2], [3, 4]])
        expected = np.rot90(grid, k=-1)
        trace = replay_trace(outer, grid)
        d = find_divergence(trace, expected)
        assert d is not None
        assert d.path == (0,)  # the inner rotate90

    def test_input_ref_matches_when_program_is_overcomplicated(self) -> None:
        # rotate90 applied but expected == input → InputRef matches.
        program = Program(
            primitive="rotate90",
            children=(InputRef(),),
            output_type="Grid",
        )
        grid = _g([[1, 2], [3, 4]])
        trace = replay_trace(program, grid)
        d = find_divergence(trace, grid)
        assert d is not None
        # The InputRef leaf at path (0,) is the deepest match.
        assert d.path == (0,)
        assert isinstance(d.node, InputRef)

    def test_no_match_returns_none(self) -> None:
        program = Program(
            primitive="rotate90",
            children=(InputRef(),),
            output_type="Grid",
        )
        grid = _g([[1, 2], [3, 4]])
        unrelated = _g([[9, 9], [9, 9]])
        trace = replay_trace(program, grid)
        assert find_divergence(trace, unrelated) is None


# ---------------------------------------------------------------------------
# find_first_failure — innermost broken subtree
# ---------------------------------------------------------------------------


class _AlwaysFailExecutor:
    """Executor stub: every Program node raises; leaves succeed."""

    _real = InProcessExecutor()

    def execute(self, program: ProgramNode, input_grid: Any) -> ExecutionResult:
        if isinstance(program, Program):
            return ExecutionResult(ok=False, error="StubError")
        return self._real.execute(program, input_grid)


class TestFindFirstFailure:
    def test_returns_none_for_clean_run(self) -> None:
        program = Program(
            primitive="rotate90",
            children=(InputRef(),),
            output_type="Grid",
        )
        trace = replay_trace(program, _g([[1, 2], [3, 4]]))
        assert find_first_failure(trace) is None

    def test_returns_innermost_failure_first(self) -> None:
        # Outer → inner program. Both Program nodes fail under the stub.
        # The inner one runs first in post-order, so it should be
        # returned.
        inner = Program(
            primitive="rotate90",
            children=(InputRef(),),
            output_type="Grid",
        )
        outer = Program(
            primitive="rotate180",
            children=(inner,),
            output_type="Grid",
        )
        executor = _AlwaysFailExecutor()
        trace = replay_trace(outer, _g([[1, 2]]), executor=executor)
        first_fail = find_first_failure(trace)
        assert first_fail is not None
        assert first_fail.path == (0,)  # inner, not outer
        assert first_fail.error == "StubError"


# ---------------------------------------------------------------------------
# Type contract — TraceStep is frozen / hashable
# ---------------------------------------------------------------------------


class TestTraceStepDataclass:
    def test_step_is_hashable_when_value_hashable(self) -> None:
        step = TraceStep(
            path=(),
            node=InputRef(),
            ok=True,
            value=7,  # hashable scalar
            error=None,
        )
        # Frozen dataclass → hashable. (For numpy values the test
        # would have to exclude `value` from the hash; we don't do
        # that here — the contract says the dataclass *is* frozen,
        # not that every Value is hashable.)
        assert hash(step) == hash(step)

    def test_step_path_is_tuple(self) -> None:
        step = TraceStep(
            path=(0, 1, 2),
            node=InputRef(),
            ok=True,
            value=None,
            error=None,
        )
        assert step.path == (0, 1, 2)
