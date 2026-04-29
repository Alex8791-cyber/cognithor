# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""NumPy-solver fast-path tests (spec §15.3)."""

from __future__ import annotations

import numpy as np

from cognithor.channels.program_synthesis.core.types import (
    SynthesisStatus,
    TaskSpec,
)
from cognithor.channels.program_synthesis.integration.numpy_solver_bridge import (
    NumpySolverBridge,
)


def _g(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


def _rotate90_spec() -> TaskSpec:
    return TaskSpec(
        examples=(
            (_g([[1, 2], [3, 4]]), _g([[3, 1], [4, 2]])),
            (_g([[5, 6], [7, 8]]), _g([[7, 5], [8, 6]])),
        ),
    )


# ---------------------------------------------------------------------------
# Availability gate
# ---------------------------------------------------------------------------


class TestAvailability:
    def test_unavailable_when_no_solver_provided(self) -> None:
        bridge = NumpySolverBridge()
        assert not bridge.is_available()

    def test_available_when_solver_provided(self) -> None:
        bridge = NumpySolverBridge(solver_fn=lambda inp, demos: inp)
        assert bridge.is_available()


# ---------------------------------------------------------------------------
# try_solve
# ---------------------------------------------------------------------------


class TestTrySolve:
    def test_returns_none_when_unavailable(self) -> None:
        bridge = NumpySolverBridge()
        assert bridge.try_solve(_rotate90_spec()) is None

    def test_returns_none_for_empty_examples(self) -> None:
        bridge = NumpySolverBridge(solver_fn=lambda inp, demos: inp)
        spec = TaskSpec(examples=())
        assert bridge.try_solve(spec) is None

    def test_solves_when_solver_returns_correct_outputs(self) -> None:
        # Fake solver that "knows" rotate90.
        def rot90(inp: np.ndarray, demos):
            return np.rot90(inp, k=-1).copy().astype(np.int8)

        bridge = NumpySolverBridge(solver_fn=rot90)
        result = bridge.try_solve(_rotate90_spec())
        assert result is not None
        assert result.status == SynthesisStatus.SUCCESS
        assert result.score == 1.0
        assert result.cost_seconds == 0.0  # bridge stamps 0 — channel re-stamps later
        # Annotation must record the source so the cache layer can
        # distinguish fast-path vs enumerator wins.
        annotations = dict(result.annotations)
        assert annotations.get("source") == "numpy_fast_path"

    def test_returns_none_on_partial_match(self) -> None:
        # Solver that only solves the first demo correctly.
        call_count = {"n": 0}

        def flaky(inp: np.ndarray, demos):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return np.rot90(inp, k=-1).copy().astype(np.int8)
            # Wrong on subsequent demos.
            return np.zeros_like(inp)

        bridge = NumpySolverBridge(solver_fn=flaky)
        assert bridge.try_solve(_rotate90_spec()) is None

    def test_returns_none_when_solver_raises(self) -> None:
        def angry(inp: np.ndarray, demos):
            raise RuntimeError("solver crashed")

        bridge = NumpySolverBridge(solver_fn=angry)
        assert bridge.try_solve(_rotate90_spec()) is None

    def test_returns_none_when_solver_returns_none(self) -> None:
        bridge = NumpySolverBridge(solver_fn=lambda inp, demos: None)
        assert bridge.try_solve(_rotate90_spec()) is None
