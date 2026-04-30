# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""NumPy-Solver fast-path bridge (spec §15.3).

The existing ARC-AGI-3 NumPy grid solver in :mod:`cognithor.arc` is
several orders of magnitude faster than the enumerative search for
patterns it can recognise (rotations, mirrors, simple recolour). The
PSE channel checks the NumPy solver first; only on a miss does it fall
through to the slower symbolic search.

**Phase-1 guarantee:** Phase 1 does not regress on tasks the NumPy
solver already handles. If the bridge fails or the solver isn't
importable, the channel runs as if the bridge wasn't there.

The actual solver lives outside this package; this module wraps it in
the :class:`Executor` shape so the search engine + cache layer don't
need to special-case it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from cognithor.channels.program_synthesis.core.types import (
    StageResult,
    SynthesisResult,
    SynthesisStatus,
    TaskSpec,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _grid_match(actual: Any, expected: Any) -> bool:
    if isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray):
        return bool(actual.shape == expected.shape and np.array_equal(actual, expected))
    return bool(actual == expected)


class NumpySolverBridge:
    """Try the NumPy fast-path; return ``None`` on miss.

    The bridge accepts a ``solver_fn`` callable so consumers can wire
    in any compatible solver (the production wiring imports
    :func:`cognithor.arc.fast_grid_solver.solve_arc_task` once it
    stabilises). Tests inject lightweight callables.

    The callable's contract::

        solver_fn(input_grid: np.ndarray, examples: tuple[Example, ...])
            -> np.ndarray | None

    Returning ``None`` means "I can't solve this" → bridge yields a
    :class:`SynthesisResult` with status ``NO_SOLUTION`` and an empty
    program, signalling the channel to fall through.
    """

    def __init__(
        self,
        solver_fn: Callable[..., np.ndarray[Any, Any] | None] | None = None,
    ) -> None:
        self._solver_fn = solver_fn

    def is_available(self) -> bool:
        return self._solver_fn is not None

    def try_solve(self, spec: TaskSpec) -> SynthesisResult | None:
        """Attempt the fast-path; return None to signal "use enumerative search".

        If the solver returns a candidate, verify it on every demo input
        before declaring SUCCESS — the NumPy solver can produce
        plausible-looking but wrong outputs on borderline tasks.
        """
        if self._solver_fn is None:
            return None
        if not spec.examples:
            return None
        try:
            # Test the solver on every demo example. The solver is
            # deterministic, so a single mismatch means "doesn't apply".
            for inp, expected in spec.examples:
                actual = self._solver_fn(inp, spec.examples)
                if actual is None or not _grid_match(actual, expected):
                    return None
        except Exception:
            return None

        # All demos matched. Build a SUCCESS result. The program slot is
        # the solver's name as a marker — the actual replay path is the
        # solver itself, not a DSL tree.
        return SynthesisResult(
            status=SynthesisStatus.SUCCESS,
            program=None,
            score=1.0,
            confidence=1.0,
            cost_seconds=0.0,
            cost_candidates=0,
            verifier_trace=(
                StageResult(
                    stage="demo",
                    passed=True,
                    detail="numpy fast-path solved",
                    duration_ms=0.0,
                ),
            ),
            annotations=(("source", "numpy_fast_path"),),
        )


__all__ = ["NumpySolverBridge"]
