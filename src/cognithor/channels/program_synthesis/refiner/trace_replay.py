# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Spec §6.4 — Trace-Replay (Sprint-1 plan task 9 slice).

Trace-Replay walks a candidate :class:`Program` tree in post-order
on a given input grid, capturing every intermediate value the
executor produces along the way. The Critic uses this trace to
localise *where* the candidate diverges from the expected output —
which subtree the Refiner should target first.

The module is executor-agnostic: the caller hands in any object
satisfying :class:`Executor` from
:mod:`cognithor.channels.program_synthesis.search.executor`. The
default helper :func:`replay_trace` defaults to
:class:`InProcessExecutor`.

Two public surfaces:

* :func:`replay_trace` — given ``(program, input_grid)``, return a
  ``tuple[TraceStep, ...]`` covering every node in post-order. The
  *last* step is the root (the program's final output).
* :func:`find_divergence` — given a trace and an ``expected``
  grid, return the deepest subtree whose output equals the
  expected value. Falls back to ``None`` when no subtree matches —
  the caller treats that as "the entire program is wrong".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from cognithor.channels.program_synthesis.search.candidate import Program
from cognithor.channels.program_synthesis.search.executor import (
    InProcessExecutor,
)

if TYPE_CHECKING:
    from cognithor.channels.program_synthesis.search.candidate import (
        ProgramNode,
    )
    from cognithor.channels.program_synthesis.search.executor import Executor


@dataclass(frozen=True)
class TraceStep:
    """One entry in the trace: the result of executing a single node.

    ``path`` is the root-relative index path. The root program has
    ``path=()``; its first child is ``(0,)``, its first child's
    second child is ``(0, 1)``, etc. The Refiner treats the path
    as a stable address for "go re-write the subtree at this
    location".

    ``ok`` is ``True`` iff the node produced a value (no exception
    propagated). ``value`` carries that value (often a numpy array,
    sometimes a primitive return type — int, tuple, etc.). When
    ``ok=False``, ``value`` is ``None`` and ``error`` carries the
    exception type name as a tag (so two failures of the same kind
    share a fingerprint).
    """

    path: tuple[int, ...]
    node: ProgramNode
    ok: bool
    value: Any
    error: str | None


def replay_trace(
    program: ProgramNode,
    input_grid: Any,
    *,
    executor: Executor | None = None,
) -> tuple[TraceStep, ...]:
    """Walk ``program`` post-order; execute each subtree on ``input_grid``.

    Returns the steps in **post-order** — children before their
    parent. The last step is therefore always the root. Failed
    subtrees still appear; their parent simply records ``ok=False``
    when its own evaluation raised because of them.

    The ``executor`` defaults to :class:`InProcessExecutor` —
    sandbox-free, fast, suitable for in-loop refinement. The caller
    can swap in the subprocess executor for production paths.
    """
    exe = executor if executor is not None else InProcessExecutor()
    steps: list[TraceStep] = []
    _walk(program, input_grid, (), exe, steps)
    return tuple(steps)


def _walk(
    node: ProgramNode,
    input_grid: Any,
    path: tuple[int, ...],
    executor: Executor,
    out: list[TraceStep],
) -> None:
    """Recurse post-order; append a TraceStep for ``node``."""
    if isinstance(node, Program):
        for idx, child in enumerate(node.children):
            _walk(child, input_grid, (*path, idx), executor, out)
    result = executor.execute(node, input_grid)
    out.append(
        TraceStep(
            path=path,
            node=node,
            ok=result.ok,
            value=result.value if result.ok else None,
            error=result.error,
        )
    )


def find_divergence(
    trace: tuple[TraceStep, ...],
    expected: Any,
) -> TraceStep | None:
    """Return the *deepest* subtree whose output equals ``expected``.

    Used by the Critic to decide: if some subtree already produces
    the expected output, the program is over-shooting (extra
    primitives applied on top); the Refiner should snip *above*
    that subtree.

    "Equality" is :func:`numpy.array_equal` for arrays, plain ``==``
    otherwise. Failed steps (``ok=False``) are skipped. Returns
    ``None`` when no step matches — the caller treats this as
    "the divergence is at the root, the whole program is wrong".
    """
    candidates = [step for step in trace if step.ok and _values_equal(step.value, expected)]
    if not candidates:
        return None
    # Paths are unique within a tree; deepest = longest path.
    return max(candidates, key=lambda s: len(s.path))


def find_first_failure(trace: tuple[TraceStep, ...]) -> TraceStep | None:
    """Return the first failing step in execution order, or ``None``.

    A program that aborts mid-way leaves all its ancestors as
    failures too — but the *innermost* failing leaf is the one the
    Refiner cares about, because that's where the bug originated.
    """
    failures = [step for step in trace if not step.ok]
    if not failures:
        return None
    # Earliest failure in post-order = innermost (children before parents).
    return failures[0]


def _values_equal(actual: Any, expected: Any) -> bool:
    """Equality that handles numpy arrays without surprises."""
    if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
        try:
            return bool(np.array_equal(actual, expected))
        except (TypeError, ValueError):
            return False
    try:
        return bool(actual == expected)
    except Exception:
        return False


__all__ = [
    "TraceStep",
    "find_divergence",
    "find_first_failure",
    "replay_trace",
]
