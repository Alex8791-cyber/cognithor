# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Program replay — K10 hard gate (spec §3 + §22).

Re-executes a program on its original input grid and verifies the
output is byte-identical to the originally produced output. The replay
must complete in P95 ≤ 100 ms.

Used by:
* The CLI ``cognithor pse replay <result.json>`` command (Week 7).
* The eval-suite K10 audit that runs over every solved task and asserts
  100 % reproducibility.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from cognithor.channels.program_synthesis.dsl.types_grid import Object, ObjectSet
from cognithor.channels.program_synthesis.search.executor import (
    InProcessExecutor,
)

if TYPE_CHECKING:
    from cognithor.channels.program_synthesis.search.candidate import ProgramNode
    from cognithor.channels.program_synthesis.search.executor import Executor


@dataclass(frozen=True)
class ReplayResult:
    """Outcome of one replay invocation.

    ``identical`` is the K10 verdict: True iff the new value equals the
    expected value byte-for-byte. ``duration_ms`` is the wall-clock
    measurement that feeds into the P95 ≤ 100 ms gate.
    """

    identical: bool
    duration_ms: float
    detail: str
    actual_summary: str
    expected_summary: str


def _value_equal(a: Any, b: Any) -> bool:
    """Strict equality for ARC value types.

    NumPy arrays compared elementwise; ObjectSet compared by ordered
    tuple; Object compared structurally. Anything else falls back to
    ``==``.
    """
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return bool(a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b))
    if isinstance(a, ObjectSet) and isinstance(b, ObjectSet):
        return bool(a.objects == b.objects)
    if isinstance(a, Object) and isinstance(b, Object):
        return bool(a == b)
    return bool(a == b)


def _summary(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return f"ndarray {value.shape} dtype={value.dtype}"
    if isinstance(value, ObjectSet):
        return f"ObjectSet len={len(value)}"
    if isinstance(value, Object):
        return f"Object color={value.color} size={value.size}"
    return f"{type(value).__name__} {value!r}"


def replay_program(
    program: ProgramNode,
    input_grid: Any,
    expected_value: Any,
    *,
    executor: Executor | None = None,
) -> ReplayResult:
    """Re-execute *program* and compare to *expected_value*.

    The K10 contract is:

    * ``identical=True`` for every solved program in the eval suite.
    * ``duration_ms`` ≤ 100 ms at P95 across the eval suite.

    This function reports per-call data; the eval-suite aggregator
    enforces the percentile gate.
    """
    ex = executor if executor is not None else InProcessExecutor()
    t0 = time.monotonic()
    result = ex.execute(program, input_grid)
    elapsed_ms = (time.monotonic() - t0) * 1000.0

    if not result.ok:
        return ReplayResult(
            identical=False,
            duration_ms=elapsed_ms,
            detail=f"replay failed: {result.error}",
            actual_summary=f"<error: {result.error}>",
            expected_summary=_summary(expected_value),
        )
    same = _value_equal(result.value, expected_value)
    return ReplayResult(
        identical=same,
        duration_ms=elapsed_ms,
        detail="byte-identical" if same else "value mismatch",
        actual_summary=_summary(result.value),
        expected_summary=_summary(expected_value),
    )
