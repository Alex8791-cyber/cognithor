# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-7 Track A1 — Cascade-Generalisation across unary chains.

Sprint-6 (#269) demonstrated that 2-step recolor cascades close
task 0202 — but only because the diff signal was rich (palette
diff with multiple introduced colors). Task 0208 needed a
3-step ``rotate90 → mirror_horizontal → recolor`` chain, which
the per-demo advisor couldn't see.

Sprint-7 Track A1 generalises the cascade strategy:

* Enumerate every 1-, 2-, and 3-step composition of unary
  Grid→Grid primitives (~10 such primitives in the DSL).
* For each composition, enumerate recolor-tail variants where
  applicable (i.e. ``<chain>(<recolor(input, src, dst)>)`` and
  ``<recolor(<chain>(input), src, dst)>``).
* Score every candidate against ALL demos.
* Return the highest-scoring candidate.

The 1-step replacement and 2-step recolor cascade from Sprint-6
are subsumed by this exhaustive enumeration.

Search-budget: bounded by the cardinality of unary chains plus
the recolor argument-product. Roughly:

  unary chains:       sum_{d=1..3} 10^d  =  10 + 100 + 1000 = 1110
  recolor variants:   ~6 (src, dst) pairs per chain
  total candidates:   ~6 600 per task

Each verification is a few microseconds (numpy on small grids),
so a budget of < 1 second per task is plenty even on Windows.

Sprint-7 Track A1 success criterion (directive):
    The cascade-generalisation closes ≥ 1 additional task on the
    hard subset (i.e. score reaches 37.5 % from Sprint-6's 25 %).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from cognithor.channels.program_synthesis.core.types import Budget
from cognithor.channels.program_synthesis.dsl.registry import REGISTRY
from cognithor.channels.program_synthesis.search.candidate import (
    Const,
    InputRef,
    Program,
)
from cognithor.channels.program_synthesis.search.enumerative import EnumerativeSearch
from cognithor.channels.program_synthesis.search.executor import InProcessExecutor
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

if TYPE_CHECKING:
    from collections.abc import Iterable

    from cognithor.channels.program_synthesis.search.candidate import ProgramNode


# ---------------------------------------------------------------------------
# Unary chain enumeration
# ---------------------------------------------------------------------------


def _unary_grid_to_grid_primitives() -> list[str]:
    """Names of every arity-1 Grid→Grid primitive in the live DSL."""
    out: list[str] = []
    for name in REGISTRY.names():
        spec = REGISTRY.get(name)
        if spec.signature.arity != 1:
            continue
        if spec.signature.inputs != ("Grid",):
            continue
        if spec.signature.output != "Grid":
            continue
        out.append(name)
    return out


def enumerate_unary_chains(
    primitives: list[str],
    *,
    base: ProgramNode | None = None,
    max_depth: int = 3,
) -> Iterable[Program]:
    """Yield every ``primitive_d ∘ ... ∘ primitive_1 ∘ base`` chain up to depth ``max_depth``.

    Yields chains in increasing depth order so the caller can
    short-circuit when a winning candidate is found.
    """
    base_node: ProgramNode = base if base is not None else InputRef()
    for depth in range(1, max_depth + 1):
        for combo in product(primitives, repeat=depth):
            chain: ProgramNode = base_node
            for prim in combo:
                chain = Program(primitive=prim, children=(chain,), output_type="Grid")
            assert isinstance(chain, Program)
            yield chain


def enumerate_recolor_variants(
    chain: Program,
    palette_sources: list[int],
    palette_targets: list[int],
    *,
    max_recolor_depth: int = 2,
) -> Iterable[Program]:
    """Wrap ``chain`` with up to ``max_recolor_depth`` cascading recolors.

    Yields (in order):
    1. The chain itself (no recolor).
    2. ``recolor(chain, src, dst)`` for every (src, dst) pair —
       Sprint-6's working strategy.
    3. ``recolor(recolor(chain, src1, dst), src2, dst)`` for every
       2-element subset of sources at the same destination — Sprint-6's
       cascade strategy generalised over an arbitrary chain.
    """
    yield chain

    if "recolor" not in REGISTRY.names():
        return

    # 1-step recolor wrap.
    for src in palette_sources:
        for dst in palette_targets:
            if src == dst:
                continue
            yield Program(
                primitive="recolor",
                children=(
                    chain,
                    Const(value=src, output_type="Color"),
                    Const(value=dst, output_type="Color"),
                ),
                output_type="Grid",
            )

    if max_recolor_depth < 2:
        return

    # 2-step cascade — Sprint-6's recolor cascade generalised.
    from itertools import combinations

    sources_set = sorted(set(palette_sources))
    for dst in palette_targets:
        for combo in combinations(sources_set, 2):
            if dst in combo:
                continue
            inner = Program(
                primitive="recolor",
                children=(
                    chain,
                    Const(value=combo[0], output_type="Color"),
                    Const(value=dst, output_type="Color"),
                ),
                output_type="Grid",
            )
            yield Program(
                primitive="recolor",
                children=(
                    inner,
                    Const(value=combo[1], output_type="Color"),
                    Const(value=dst, output_type="Color"),
                ),
                output_type="Grid",
            )


# ---------------------------------------------------------------------------
# Score helper (lifted from Sprint-6)
# ---------------------------------------------------------------------------


def _score_program_on_demos(
    program: ProgramNode,
    demos: list[tuple[Any, Any]],
    executor: InProcessExecutor,
) -> float:
    if not demos:
        return 0.0
    correct = 0
    for inp, expected in demos:
        result = executor.execute(program, inp)
        if not result.ok:
            continue
        if isinstance(result.value, np.ndarray) or isinstance(expected, np.ndarray):
            try:
                if np.array_equal(result.value, expected):
                    correct += 1
            except (TypeError, ValueError):
                continue
        elif result.value == expected:
            correct += 1
    return correct / len(demos)


# ---------------------------------------------------------------------------
# Cascade strategy
# ---------------------------------------------------------------------------


def _palette_from_demos(demos: list[tuple[Any, Any]]) -> tuple[list[int], list[int]]:
    """Collect (input_palette, output_palette) across all demos."""
    in_palette: set[int] = set()
    out_palette: set[int] = set()
    for inp, out in demos:
        if isinstance(inp, np.ndarray):
            in_palette.update(int(v) for v in np.unique(inp))
        if isinstance(out, np.ndarray):
            out_palette.update(int(v) for v in np.unique(out))
    return sorted(in_palette), sorted(out_palette)


def cascade_repair(
    phase1_program: ProgramNode,
    demos: list[tuple[Any, Any]],
    executor: InProcessExecutor,
    *,
    max_depth: int = 3,
) -> tuple[ProgramNode, float, bool]:
    """Generalised cascade repair — enumerate unary chains + recolor variants.

    Returns ``(best_program, best_score, refined)`` where ``refined``
    is ``True`` when the cascade strategy found a candidate strictly
    better than the Phase-1 program.

    The strategy short-circuits once a candidate hits score 1.0
    (every demo correct).
    """
    p1_score = _score_program_on_demos(phase1_program, demos, executor)
    if p1_score >= 1.0 or not demos:
        return phase1_program, p1_score, False

    in_palette, out_palette = _palette_from_demos(demos)
    primitives = _unary_grid_to_grid_primitives()

    best_program: ProgramNode = phase1_program
    best_score = p1_score
    refined = False

    # Sprint-6's two starting points: InputRef and the Phase-1 program.
    for base_program in (InputRef(), phase1_program):
        for chain in enumerate_unary_chains(primitives, base=base_program, max_depth=max_depth):
            for candidate in enumerate_recolor_variants(
                chain, in_palette + out_palette, out_palette
            ):
                cs = _score_program_on_demos(candidate, demos, executor)
                if cs > best_score:
                    best_program = candidate
                    best_score = cs
                    refined = True
                    if best_score >= 1.0:
                        return best_program, best_score, refined
    return best_program, best_score, refined


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def _run_async(args: argparse.Namespace) -> int:
    enumerative = EnumerativeSearch()
    in_proc = InProcessExecutor()
    tasks = list(
        corpus_benchmark_tasks(
            args.corpus_root,
            subset=args.subset,
            wall_clock_budget_seconds=args.wall_clock_budget_seconds,
        )
    )

    rows: list[BenchmarkTaskResult] = []
    errors: list[tuple[str, str]] = []

    for task in tasks:
        start = time.monotonic()
        try:
            budget = Budget(
                max_depth=4,
                wall_clock_seconds=args.wall_clock_budget_seconds,
                max_candidates=10_000,
            )
            phase1 = await asyncio.to_thread(enumerative.search, task.spec, budget)
        except Exception as exc:
            errors.append((task.task_id, type(exc).__name__))
            continue

        if phase1.program is None:
            # Phase-1 found nothing. Try cascade from InputRef alone —
            # for tasks where the right answer is a chain Phase-1 didn't
            # explore. Use InputRef as the synthetic phase1_program; the
            # cascade_repair function treats this as a 0.0-baseline.
            best_program, best_score, refined = cascade_repair(
                InputRef(),
                list(task.spec.examples),
                in_proc,
                max_depth=args.max_chain_depth,
            )
            elapsed = time.monotonic() - start
            terminated_by = (
                "refined_success"
                if refined and best_score >= args.success_threshold
                else (
                    "search_success"
                    if best_score >= args.success_threshold
                    else (
                        "refined_partial"
                        if refined
                        else ("no_solution" if best_score == 0.0 else "search_exhausted")
                    )
                )
            )
            rows.append(
                BenchmarkTaskResult(
                    task_id=task.task_id,
                    score=best_score,
                    elapsed_seconds=elapsed,
                    terminated_by=terminated_by,
                    cache_hit=False,
                    refined=refined,
                    refinement_path=("cascade",) if refined else (),
                )
            )
            continue

        best_program, best_score, refined = cascade_repair(
            phase1.program,
            list(task.spec.examples),
            in_proc,
            max_depth=args.max_chain_depth,
        )
        elapsed = time.monotonic() - start
        terminated_by = (
            "refined_success"
            if refined and best_score >= args.success_threshold
            else (
                "search_success"
                if best_score >= args.success_threshold
                else ("refined_partial" if refined else "search_exhausted")
            )
        )
        rows.append(
            BenchmarkTaskResult(
                task_id=task.task_id,
                score=best_score,
                elapsed_seconds=elapsed,
                terminated_by=terminated_by,
                cache_hit=False,
                refined=refined,
                refinement_path=("cascade",) if refined else (),
            )
        )

    summary = _aggregate(rows, errors, success_threshold=args.success_threshold)

    bundle_hash = corpus_hash(load_corpus(args.corpus_root, subset=args.subset))
    payload = dump_summary(summary, bundle_hash=bundle_hash)
    Path(args.output).write_text(payload, encoding="utf-8")

    if args.markdown:
        title = f"Sprint-7 Track A1 — Cascade Generalisation ({args.subset or 'all'})"
        Path(args.markdown).write_text(
            render_markdown(summary, title=title, bundle_hash=bundle_hash),
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "engine": "phase1_plus_cascade",
                "subset": args.subset or "all",
                "n_tasks": summary.n_tasks,
                "success_rate": summary.success_rate,
                "refined_rate": summary.refined_rate,
                "refinement_uplift_rate": summary.refinement_uplift_rate,
                "p50": summary.p50_seconds,
                "p95": summary.p95_seconds,
            },
            sort_keys=True,
        )
    )
    return 0


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
    refined = sum(1 for r in rows if r.refined)
    refined_uplift = sum(1 for r in rows if r.refined and r.score >= success_threshold)
    elapsed = sorted(r.elapsed_seconds for r in rows)
    return BenchmarkSummary(
        n_tasks=n,
        success_rate=successes / n,
        cache_hit_rate=0.0,
        refined_rate=refined / n,
        refinement_uplift_rate=(refined_uplift / refined) if refined else 0.0,
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


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cognithor.channels.program_synthesis.synthesis.sprint7_cascade_runner",
        description=(
            "Sprint-7 Track A1: Phase-1 + generalised cascade-enumeration "
            "over unary chains + recolor variants."
        ),
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=Path("cognithor_bench/arc_agi3"),
    )
    parser.add_argument("--subset", type=str, default="hard")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sprint7_track_a1_report.json"),
    )
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--success-threshold", type=float, default=0.95)
    parser.add_argument("--max-chain-depth", type=int, default=3)
    parser.add_argument(
        "--wall-clock-budget-seconds",
        type=float,
        default=5.0,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    return asyncio.run(_run_async(args))


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "cascade_repair",
    "enumerate_recolor_variants",
    "enumerate_unary_chains",
    "main",
]
