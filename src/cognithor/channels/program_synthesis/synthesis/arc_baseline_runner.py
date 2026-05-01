# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-4 — ARC-AGI-3 Phase-1 reality-check runner.

Standalone CLI that loads an ARC-AGI-3 corpus (typically the
``cognithor_bench/arc_agi3`` fixture set committed in v0.78), runs
the Phase-1 :class:`EnumerativeSearch` against every task, and
persists the resulting :class:`BenchmarkSummary` as a baseline
JSON. The Sprint-4 "what does Cognithor actually score on real
ARC?" question gets a number — call this a reality-check, not a
production benchmark.

Why a separate runner from ``benchmark_runner.py``?

* ``benchmark_runner.py`` is wired to the synthetic 20-task
  leak-free fixture set; its CLI semantics (``--phase2`` toggle,
  regression gate against a leak-free baseline) make sense in
  that context.
* The ARC corpus has different characteristics (variable demo
  count, larger grids, real-world transformations) that warrant
  their own runner so the Sprint-2/3 baseline isn't conflated
  with Sprint-4's reality-check measurements.

Example::

    python -m cognithor.channels.program_synthesis.synthesis.arc_baseline_runner \\
        --corpus-root cognithor_bench/arc_agi3 \\
        --subset train \\
        --output .ci/arc_agi3_phase1_baseline.json \\
        --markdown arc_phase1_report.md \\
        --wall-clock-budget-seconds 5.0

Sprint-4 acceptance:
    - Score on cognithor_bench/arc_agi3 train (8 tasks) persisted
    - Score on cognithor_bench/arc_agi3 held_out (4 tasks) persisted
    - Both runs produce valid JSON + Markdown reports
    - Exit code 0 (the runner doesn't gate on regression — it's a
      measurement tool, not a CI gate)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from cognithor.channels.program_synthesis.core.types import Budget
from cognithor.channels.program_synthesis.search.enumerative import EnumerativeSearch
from cognithor.channels.program_synthesis.synthesis.arc_corpus import (
    corpus_benchmark_tasks,
    corpus_hash,
    load_corpus,
)
from cognithor.channels.program_synthesis.synthesis.benchmark import (
    BenchmarkSummary,
    BenchmarkTaskResult,
)
from cognithor.channels.program_synthesis.synthesis.benchmark_report import (
    dump_summary,
    render_markdown,
)

# ---------------------------------------------------------------------------
# Phase-1 driver
# ---------------------------------------------------------------------------


async def _run_phase1_on_corpus(
    corpus_root: Path,
    subset: str | None,
    wall_clock_budget_seconds: float,
    success_threshold: float,
) -> BenchmarkSummary:
    """Run Phase-1 EnumerativeSearch across every task in the subset."""
    enumerative = EnumerativeSearch()
    tasks = list(
        corpus_benchmark_tasks(
            corpus_root,
            subset=subset,
            wall_clock_budget_seconds=wall_clock_budget_seconds,
        )
    )
    rows: list[BenchmarkTaskResult] = []
    errors: list[tuple[str, str]] = []
    for task in tasks:
        start = time.monotonic()
        try:
            budget = Budget(
                max_depth=4,
                wall_clock_seconds=wall_clock_budget_seconds,
                max_candidates=10_000,
            )
            result = await asyncio.to_thread(enumerative.search, task.spec, budget)
        except Exception as exc:
            errors.append((task.task_id, type(exc).__name__))
            continue
        elapsed = time.monotonic() - start
        score = float(result.score)
        terminated_by = (
            "search_success"
            if score >= success_threshold
            else ("no_candidates" if result.program is None else "search_exhausted")
        )
        rows.append(
            BenchmarkTaskResult(
                task_id=task.task_id,
                score=score,
                elapsed_seconds=elapsed,
                terminated_by=terminated_by,
                cache_hit=False,
                refined=False,
            )
        )

    return _aggregate(rows, errors, success_threshold=success_threshold)


def _aggregate(
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
    elapsed = sorted(r.elapsed_seconds for r in rows)
    return BenchmarkSummary(
        n_tasks=n,
        success_rate=successes / n,
        cache_hit_rate=0.0,
        refined_rate=0.0,
        refinement_uplift_rate=0.0,
        p50_seconds=_percentile(elapsed, 0.5),
        p95_seconds=_percentile(elapsed, 0.95),
        per_task_results=tuple(rows),
        errors=tuple(errors),
    )


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = p * (len(sorted_values) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return sorted_values[lo]
    weight = rank - lo
    return sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def _run_async(args: argparse.Namespace) -> int:
    summary = await _run_phase1_on_corpus(
        corpus_root=args.corpus_root,
        subset=args.subset,
        wall_clock_budget_seconds=args.wall_clock_budget_seconds,
        success_threshold=args.success_threshold,
    )

    # Compute the corpus hash so the report includes a fixture
    # fingerprint — analogous to leak_free_set_hash.
    tasks = load_corpus(args.corpus_root, subset=args.subset)
    bundle_hash = corpus_hash(tasks)

    payload = dump_summary(summary, bundle_hash=bundle_hash)
    Path(args.output).write_text(payload, encoding="utf-8")

    if args.markdown:
        title = f"ARC-AGI-3 Phase-1 Reality Check ({args.subset or 'all'})"
        Path(args.markdown).write_text(
            render_markdown(summary, title=title, bundle_hash=bundle_hash),
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "subset": args.subset or "all",
                "n_tasks": summary.n_tasks,
                "success_rate": summary.success_rate,
                "p50": summary.p50_seconds,
                "p95": summary.p95_seconds,
                "errors": len(summary.errors),
                "bundle_hash": bundle_hash,
            },
            sort_keys=True,
        )
    )
    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cognithor.channels.program_synthesis.synthesis.arc_baseline_runner",
        description=(
            "Sprint-4 reality-check: run Phase-1 EnumerativeSearch on the "
            "ARC-AGI-3 corpus and persist the baseline."
        ),
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=Path("cognithor_bench/arc_agi3"),
        help="ARC-AGI-3 corpus root (default: cognithor_bench/arc_agi3)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help=(
            "Corpus subset name from the manifest (e.g. 'train', 'held_out'). "
            "Default: load every .json under tasks/."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("arc_phase1_baseline.json"),
        help="JSON baseline destination (default: arc_phase1_baseline.json)",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Optional Markdown report destination",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.95,
        help="Score considered a success (default: 0.95)",
    )
    parser.add_argument(
        "--wall-clock-budget-seconds",
        type=float,
        default=5.0,
        help="Per-task wall-clock budget (default: 5.0s)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    return asyncio.run(_run_async(args))


if __name__ == "__main__":
    sys.exit(main())


# Reserved imports — kept as side-effect anchors for future
# Phase-2 wiring (Sprint-5+): ARCTask, asyncio types.
_ = (Any,)
