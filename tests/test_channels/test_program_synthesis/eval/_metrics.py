# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Spec §18.3 metrics aggregator for the ARC-AGI-3 eval suite.

Pure, side-effect-free helpers for computing the four documented
metrics from a list of per-task run results:

* ``Solved@30s`` — count of tasks where ``status == SUCCESS`` and
  ``cost_seconds <= 30.0``.
* ``Solved@5s`` — same with ``cost_seconds <= 5.0``.
* ``Median-Time-Solved`` — median of ``cost_seconds`` over the
  *solved* subset only.
* ``FP-Rate`` — fraction of programs that passed every demo pair but
  failed the held-out check (``demos_passed and not held_out_passed``).

Kept separate from the harness so the metrics can be unit-tested
without booting the channel or reading the manifest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True)
class TaskRunResult:
    """One task × one solver run.

    The harness produces one of these per (task, solver) pair —
    ``solver`` is either ``"pse"`` or ``"baseline"``.
    """

    task_id: str
    solver: str
    success: bool
    cost_seconds: float
    demos_passed: bool = True
    held_out_passed: bool = True


@dataclass(frozen=True)
class SubsetMetrics:
    """Aggregated metrics for one solver on one subset (train|held_out)."""

    solver: str
    subset: str
    n: int
    solved_at_30s: int
    solved_at_5s: int
    median_time_solved: float | None
    fp_rate: float | None  # None when subset has no held-out check


def _solved_at(results: Iterable[TaskRunResult], budget_seconds: float) -> int:
    return sum(1 for r in results if r.success and r.cost_seconds <= budget_seconds)


def _median_time_solved(results: Iterable[TaskRunResult]) -> float | None:
    times = sorted(r.cost_seconds for r in results if r.success)
    if not times:
        return None
    n = len(times)
    if n % 2 == 1:
        return times[n // 2]
    # Even count: average the two middle values.
    return (times[n // 2 - 1] + times[n // 2]) / 2.0


def _fp_rate(results: Iterable[TaskRunResult]) -> float | None:
    """Programs that pass demos but fail the held-out check.

    Spec §18.3: counted *over solved tasks only* — a task that the
    solver could not produce a program for cannot have an FP. Returns
    ``None`` if no task supplied a held-out signal at all (i.e. the
    subset is not the held-out subset).
    """
    materialised = list(results)
    relevant = [r for r in materialised if r.success]
    if not relevant:
        return None
    fps = sum(1 for r in relevant if r.demos_passed and not r.held_out_passed)
    return fps / len(relevant)


def aggregate(
    results: Iterable[TaskRunResult],
    *,
    solver: str,
    subset: str,
) -> SubsetMetrics:
    """Reduce a flat result list to one :class:`SubsetMetrics`."""
    materialised = [r for r in results if r.solver == solver]
    n = len(materialised)
    return SubsetMetrics(
        solver=solver,
        subset=subset,
        n=n,
        solved_at_30s=_solved_at(materialised, 30.0),
        solved_at_5s=_solved_at(materialised, 5.0),
        median_time_solved=_median_time_solved(materialised),
        fp_rate=_fp_rate(materialised),
    )


def k1_threshold_met(pse: SubsetMetrics, baseline: SubsetMetrics) -> bool:
    """Spec §18.4 / K1 success threshold.

    PSE must beat the baseline by **+5** on Solved@30s without
    regressing Solved@5s. Both inputs must come from the same subset.
    """
    if pse.subset != baseline.subset:
        raise ValueError(
            f"k1_threshold_met: subset mismatch ({pse.subset!r} vs {baseline.subset!r})"
        )
    return (
        pse.solved_at_30s >= baseline.solved_at_30s + 5
        and pse.solved_at_5s >= baseline.solved_at_5s
    )


def format_summary(pse: SubsetMetrics, baseline: SubsetMetrics) -> str:
    """Render a one-block-per-subset Markdown summary.

    The harness writes this verbatim into ``runs/<ts>/summary.md`` so
    the file can be diffed-in to ``benchmarks.md`` by hand.
    """
    if pse.subset != baseline.subset:
        raise ValueError("format_summary: subset mismatch")

    def _fmt_time(t: float | None) -> str:
        if t is None or math.isnan(t):
            return "—"
        return f"{t:.2f}s"

    def _fmt_rate(r: float | None) -> str:
        if r is None:
            return "—"
        return f"{r:.1%}"

    return (
        f"### Subset: {pse.subset} (n={pse.n})\n"
        f"\n"
        f"| Metric | Baseline ({baseline.solver}) | PSE | Δ |\n"
        f"|---|---|---|---|\n"
        f"| Solved@30s | {baseline.solved_at_30s} | {pse.solved_at_30s} "
        f"| {pse.solved_at_30s - baseline.solved_at_30s:+d} |\n"
        f"| Solved@5s | {baseline.solved_at_5s} | {pse.solved_at_5s} "
        f"| {pse.solved_at_5s - baseline.solved_at_5s:+d} |\n"
        f"| Median-Time-Solved | {_fmt_time(baseline.median_time_solved)} "
        f"| {_fmt_time(pse.median_time_solved)} | — |\n"
        f"| FP-Rate | {_fmt_rate(baseline.fp_rate)} | "
        f"{_fmt_rate(pse.fp_rate)} | — |\n"
        f"| K1 threshold met? | — | "
        f"{'✅' if k1_threshold_met(pse, baseline) else '❌'} | — |\n"
    )
