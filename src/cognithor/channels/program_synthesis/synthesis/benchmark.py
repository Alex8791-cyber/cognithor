# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Spec §12.3 / §12.4 — Phase-2 benchmark runner (Sprint-1 plan task 12).

A thin orchestrator that runs a list of tasks through the
:class:`Phase2SynthesisEngine` and aggregates per-task results
into a :class:`BenchmarkSummary`. Sprint-1 ships:

* :class:`BenchmarkTask` — one task spec + the budget to run it
  under, plus a stable id used by the report.
* :class:`BenchmarkTaskResult` — per-task outcome (status, score,
  elapsed seconds, refinement path).
* :class:`BenchmarkSummary` — aggregate over a run (success rate,
  P50/P95 wall-clock, refinement uplift).
* :func:`run_benchmark` — async driver.

The full Sprint-2 benchmark (20-Task Leak-Free-Held-Out subset +
Streamlit dashboard + nightly CI cron) lands in a follow-up PR;
the *driver* shipped here is the production entry point — once
the held-out fixtures are curated, they're handed in via the
``tasks`` argument and the driver runs unchanged.

The runner is engine-agnostic: any object with an
``async synthesize(spec, budget, *, wall_clock_budget_seconds,
current_alpha) -> Phase2SynthesisResult`` method is compatible
(structural typing). Tests inject a stub.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable

    from cognithor.channels.program_synthesis.phase2.datatypes import (
        PartitionedBudget,
    )
    from cognithor.channels.program_synthesis.synthesis.engine import (
        Phase2SynthesisResult,
    )


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkTask:
    """One task in the benchmark suite.

    ``task_id`` is a stable string the report keys results by
    (matches the leak-free-set filename stem in production —
    ``"0001_rotate90"`` etc.).

    ``spec`` is the :class:`TaskSpec` (or any object the engine
    accepts) the engine receives. ``budget`` is the partitioned
    budget; ``wall_clock_budget_seconds`` is the per-task
    wall-clock limit.

    ``current_alpha`` is the Search-α the engine should start
    with — production reads this from the live AlphaController;
    the benchmark may pin it (default 0.6 — middle of the band)
    so runs are reproducible.
    """

    task_id: str
    spec: Any
    budget: PartitionedBudget
    wall_clock_budget_seconds: float
    current_alpha: float = 0.6


@dataclass(frozen=True)
class BenchmarkTaskResult:
    """Per-task outcome — one row in the benchmark report."""

    task_id: str
    score: float
    elapsed_seconds: float
    terminated_by: str
    cache_hit: bool
    refined: bool
    refinement_path: tuple[str, ...] = ()


@dataclass(frozen=True)
class BenchmarkSummary:
    """Aggregate over a benchmark run.

    ``n_tasks`` is the count of tasks that ran (excludes ones the
    engine raised on — those are recorded separately in
    ``errors``).

    ``success_rate`` is the fraction of tasks whose ``score >=
    success_threshold`` (the threshold is parameterised so the
    same tasks can be re-graded under a stricter bar without
    re-running).

    ``cache_hit_rate`` and ``refined_rate`` track Stage-0 / Stage-2
    activation. ``refinement_uplift_rate`` is the fraction of
    refined tasks whose refinement actually crossed the success
    threshold (so the spec §6 refiner uplift can be measured).

    ``p50_seconds`` / ``p95_seconds`` are wall-clock percentiles.
    """

    n_tasks: int
    success_rate: float
    cache_hit_rate: float
    refined_rate: float
    refinement_uplift_rate: float
    p50_seconds: float
    p95_seconds: float
    per_task_results: tuple[BenchmarkTaskResult, ...] = field(default_factory=tuple)
    errors: tuple[tuple[str, str], ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Engine protocol
# ---------------------------------------------------------------------------


class _SynthesizeCapable(Protocol):
    """Structural type for any engine the benchmark accepts."""

    async def synthesize(
        self,
        spec: Any,
        budget: PartitionedBudget,
        *,
        wall_clock_budget_seconds: float,
        current_alpha: float = ...,
    ) -> Phase2SynthesisResult: ...


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def run_benchmark(
    engine: _SynthesizeCapable,
    tasks: Iterable[BenchmarkTask],
    *,
    success_threshold: float = 0.95,
) -> BenchmarkSummary:
    """Run every task in ``tasks`` through ``engine``; aggregate the results.

    The driver does *not* parallelise — Sprint-1 runs strictly
    sequentially so the benchmark is reproducible. Sprint-2 may add
    parallel-task execution gated by a CLI flag.

    A task that raises is recorded in the summary's ``errors`` list;
    other tasks continue. The summary's ``n_tasks`` excludes errored
    tasks so the success rate is over completed runs only.
    """
    materialised = list(tasks)
    per_task: list[BenchmarkTaskResult] = []
    errors: list[tuple[str, str]] = []

    for task in materialised:
        try:
            result = await engine.synthesize(
                task.spec,
                task.budget,
                wall_clock_budget_seconds=task.wall_clock_budget_seconds,
                current_alpha=task.current_alpha,
            )
        except Exception as exc:
            errors.append((task.task_id, type(exc).__name__))
            continue
        per_task.append(
            BenchmarkTaskResult(
                task_id=task.task_id,
                score=result.score,
                elapsed_seconds=result.elapsed_seconds,
                terminated_by=result.terminated_by,
                cache_hit=result.cache_hit,
                refined=result.refined,
                refinement_path=result.refinement_path,
            )
        )

    return _summarise(per_task, errors, success_threshold=success_threshold)


def _summarise(
    rows: list[BenchmarkTaskResult],
    errors: list[tuple[str, str]],
    *,
    success_threshold: float,
) -> BenchmarkSummary:
    n = len(rows)
    if n == 0:
        return BenchmarkSummary(
            n_tasks=0,
            success_rate=0.0,
            cache_hit_rate=0.0,
            refined_rate=0.0,
            refinement_uplift_rate=0.0,
            p50_seconds=0.0,
            p95_seconds=0.0,
            per_task_results=tuple(rows),
            errors=tuple(errors),
        )
    successes = sum(1 for r in rows if r.score >= success_threshold)
    cache_hits = sum(1 for r in rows if r.cache_hit)
    refined = sum(1 for r in rows if r.refined)
    refined_uplift = sum(1 for r in rows if r.refined and r.score >= success_threshold)
    return BenchmarkSummary(
        n_tasks=n,
        success_rate=successes / n,
        cache_hit_rate=cache_hits / n,
        refined_rate=refined / n,
        refinement_uplift_rate=(refined_uplift / refined) if refined else 0.0,
        p50_seconds=_percentile([r.elapsed_seconds for r in rows], 0.5),
        p95_seconds=_percentile([r.elapsed_seconds for r in rows], 0.95),
        per_task_results=tuple(rows),
        errors=tuple(errors),
    )


def _percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile — small-N safe."""
    if not values:
        return 0.0
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"percentile p must be in [0, 1]; got {p}")
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = p * (len(sorted_values) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return sorted_values[lo]
    weight = rank - lo
    return sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight


__all__ = [
    "BenchmarkSummary",
    "BenchmarkTask",
    "BenchmarkTaskResult",
    "run_benchmark",
]
