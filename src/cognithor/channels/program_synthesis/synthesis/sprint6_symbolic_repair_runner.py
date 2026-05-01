# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-6 Track A — Symbolic-Repair-Advisor live experiment.

Sprint-5 Track 1 (#268) showed Phase-2 wiring activates on
borderline-partials but Local-Edit alone doesn't close them. The
two refined-but-unsolved tasks (0202 + 0208) need *structural*
mutations, exactly what the Symbolic-Repair-Advisor (#252) was
built for.

This runner is a **standalone experiment** that bypasses the
``WiredPhase2Engine`` (which builds its refiner once with
spec-agnostic closures) and instead, per task:

1. Runs Phase-1 ``EnumerativeSearch`` to get a candidate program.
2. Executes the candidate on the first demo to get actual.
3. Builds a :class:`DiffReport` against the demo's expected output.
4. Calls :func:`advise_repairs` to get repair suggestions.
5. For each suggestion with a ``primitive_hint``, constructs a
   candidate program ``<hint>(input)`` and verifies it against
   *every* demo.
6. Returns the highest-scoring candidate (or the original Phase-1
   result if no symbolic suggestion improves on it).

Why standalone instead of refactoring WiredPhase2Engine?
* The proper integration requires either a refiner-factory pattern
  (build per-task) or a ContextVar threaded through stage runners —
  both are non-trivial refactors.
* For Sprint-6 the *measurement* is what matters: does the
  Symbolic-Repair-Advisor actually close the 0202/0208 gap?
  Get the answer in one PR; then Sprint-7 does the proper wiring
  if the answer is yes.

Sprint-6 Track A success criterion:
    Symbolic-Repair-Advisor closes ≥ 1 of the 2 borderline-partials
    on the cognithor_bench/arc_agi3 hard subset (i.e. either 0202
    or 0208 reaches score ≥ 0.95).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from cognithor.channels.program_synthesis.core.types import Budget
from cognithor.channels.program_synthesis.dsl.registry import REGISTRY
from cognithor.channels.program_synthesis.refiner.diff_analyzer import analyze_diff
from cognithor.channels.program_synthesis.refiner.symbolic_repair import advise_repairs
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
    from cognithor.channels.program_synthesis.search.candidate import ProgramNode


# ---------------------------------------------------------------------------
# Symbolic-Repair experiment per task
# ---------------------------------------------------------------------------


def _candidates_from_hint(
    hint: str,
    palette_actual: list[int],
    palette_expected: list[int],
    *,
    base: ProgramNode | None = None,
) -> list[Program]:
    """Lift a primitive_hint to one or more candidate programs.

    For unary Grid→Grid primitives, returns ``[<hint>(input)]``.
    For ``recolor`` (3-arg: Grid + Color + Color), generates a
    Cartesian product over plausible (src, dst) pairs derived from
    the actual / expected palettes — the advisor's hint is a clue,
    the actual color args have to be inferred from the diff.
    For other higher-arity primitives we currently skip; Sprint-7
    can extend with primitive-specific arg synthesis.
    """
    spec = REGISTRY.get(hint) if hint in REGISTRY.names() else None
    if spec is None:
        return []
    arity = spec.signature.arity
    grid_arg: ProgramNode = base if base is not None else InputRef()
    if arity == 1:
        return [Program(primitive=hint, children=(grid_arg,), output_type="Grid")]
    if hint == "recolor" and arity == 3:
        # Try every (src ∈ actual_palette, dst ∈ expected_palette) pair.
        out: list[Program] = []
        seen: set[tuple[int, int]] = set()
        for src in palette_actual:
            for dst in palette_expected:
                if src == dst or (src, dst) in seen:
                    continue
                seen.add((src, dst))
                out.append(
                    Program(
                        primitive="recolor",
                        children=(
                            grid_arg,
                            Const(value=src, output_type="Color"),
                            Const(value=dst, output_type="Color"),
                        ),
                        output_type="Grid",
                    )
                )
        return out
    if hint == "swap_colors" and arity == 3:
        out_swap: list[Program] = []
        seen_swap: set[frozenset[int]] = set()
        full = sorted(set(palette_actual) | set(palette_expected))
        for c1 in full:
            for c2 in full:
                if c1 >= c2:
                    continue
                key = frozenset({c1, c2})
                if key in seen_swap:
                    continue
                seen_swap.add(key)
                out_swap.append(
                    Program(
                        primitive="swap_colors",
                        children=(
                            grid_arg,
                            Const(value=c1, output_type="Color"),
                            Const(value=c2, output_type="Color"),
                        ),
                        output_type="Grid",
                    )
                )
        return out_swap
    # Higher-arity primitives without ad-hoc arg synthesis: skip.
    return []


def _score_program_on_demos(
    program: ProgramNode,
    demos: list[tuple[Any, Any]],
    executor: InProcessExecutor,
) -> float:
    """Return fraction of demos on which the program produces the expected output."""
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


def _try_symbolic_repair(
    phase1_program: ProgramNode,
    demos: list[tuple[Any, Any]],
    executor: InProcessExecutor,
) -> tuple[ProgramNode, float, bool]:
    """Run the Symbolic-Repair-Advisor; return (best_program, best_score, refined).

    ``refined`` is ``True`` when the advisor proposed a candidate
    that beat the Phase-1 program's score. Falls back to the Phase-1
    program itself when nothing helps.
    """
    p1_score = _score_program_on_demos(phase1_program, demos, executor)

    if not demos:
        return phase1_program, p1_score, False

    # Pick a *failing* demo — the one where the program produces
    # something different from expected. Using a failing demo
    # guarantees the diff is non-empty so the advisor has something
    # to react to.
    failing_idx = None
    for i, (inp, expected) in enumerate(demos):
        result = executor.execute(phase1_program, inp)
        if not result.ok:
            failing_idx = i
            break
        if isinstance(result.value, np.ndarray) and isinstance(expected, np.ndarray):
            if not np.array_equal(result.value, expected):
                failing_idx = i
                break
        elif result.value != expected:
            failing_idx = i
            break
    if failing_idx is None:
        # All demos already pass — nothing to refine.
        return phase1_program, p1_score, False

    inp, expected = demos[failing_idx]
    actual_result = executor.execute(phase1_program, inp)
    if (
        not actual_result.ok
        or not isinstance(actual_result.value, np.ndarray)
        or not isinstance(expected, np.ndarray)
    ):
        return phase1_program, p1_score, False
    actual = actual_result.value

    diff = analyze_diff(actual, expected)
    suggestions = advise_repairs(actual, expected, diff)

    palette_actual = sorted({int(v) for v in np.unique(actual)})
    palette_expected = sorted({int(v) for v in np.unique(expected)})

    best_program: ProgramNode = phase1_program
    best_score = p1_score
    refined = False

    # Strategy A: replacement — `<hint>(input)` (or `<hint>(input, args)`).
    # Strategy B: wrap — `<hint>(<phase1_program>, args)`. The advisor's
    # hint is the *missing* operation; wrapping it on top of the
    # existing program addresses cases where Phase-1 produced a
    # near-correct base that just needs one more step.
    for sug in suggestions:
        if sug.primitive_hint is None:
            continue
        # A: replacement.
        for candidate in _candidates_from_hint(
            sug.primitive_hint, palette_actual, palette_expected
        ):
            cs = _score_program_on_demos(candidate, demos, executor)
            if cs > best_score:
                best_program = candidate
                best_score = cs
                refined = True
        # B: wrap on top of phase1_program.
        for candidate in _candidates_from_hint(
            sug.primitive_hint,
            palette_actual,
            palette_expected,
            base=phase1_program,
        ):
            cs = _score_program_on_demos(candidate, demos, executor)
            if cs > best_score:
                best_program = candidate
                best_score = cs
                refined = True

    # Strategy C: 2-step recolor composition. When R1 ColorRepair
    # introduces multiple missing colors, build cascading recolors
    # on either the Phase-1 program OR plain InputRef as base —
    # because Phase-1 may have produced the wrong base structure
    # (e.g. mirror_vertical when the right answer needs no flip).
    # Also try also looking at colors introduced from the *expected*
    # side (cells in input that should disappear).
    expected_set = set(palette_expected)
    actual_set = set(palette_actual)
    # Also consider input palette: a cell in input that's not in
    # expected is a candidate src→0 recoloring.
    input_palette: set[int] = set()
    for inp, _ in demos:
        if isinstance(inp, np.ndarray):
            input_palette.update(int(v) for v in np.unique(inp))
    candidate_srcs = (actual_set - expected_set) | (input_palette - expected_set)
    if "recolor" in REGISTRY.names() and len(candidate_srcs) >= 1:
        # Try every subset of candidate_srcs (up to size 3) cascaded
        # from each base, against each dst. Score *each intermediate*
        # wrapped — partial cascades may match better than full ones
        # when some srcs are spurious (only present on one demo).
        from itertools import combinations

        sorted_srcs = sorted(candidate_srcs)
        max_cascade = min(3, len(sorted_srcs))
        for base_template in (phase1_program, InputRef()):
            for dst in palette_expected:
                for k in range(1, max_cascade + 1):
                    for combo in combinations(sorted_srcs, k):
                        if dst in combo:
                            continue
                        base_p: ProgramNode = base_template
                        wrapped: Program | None = None
                        for src in combo:
                            wrapped = Program(
                                primitive="recolor",
                                children=(
                                    base_p,
                                    Const(value=src, output_type="Color"),
                                    Const(value=dst, output_type="Color"),
                                ),
                                output_type="Grid",
                            )
                            base_p = wrapped
                        if wrapped is None:
                            continue
                        cs = _score_program_on_demos(wrapped, demos, executor)
                        if cs > best_score:
                            best_program = wrapped
                            best_score = cs
                            refined = True

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
            elapsed = time.monotonic() - start
            rows.append(
                BenchmarkTaskResult(
                    task_id=task.task_id,
                    score=0.0,
                    elapsed_seconds=elapsed,
                    terminated_by="no_solution",
                    cache_hit=False,
                    refined=False,
                )
            )
            continue

        # Symbolic-repair attempt.
        best_program, best_score, refined = _try_symbolic_repair(
            phase1.program,
            list(task.spec.examples),
            in_proc,
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
                refinement_path=("symbolic_repair",) if refined else (),
            )
        )

    summary = _aggregate(rows, errors, success_threshold=args.success_threshold)

    bundle_hash = corpus_hash(load_corpus(args.corpus_root, subset=args.subset))
    payload = dump_summary(summary, bundle_hash=bundle_hash)
    Path(args.output).write_text(payload, encoding="utf-8")

    if args.markdown:
        title = f"Sprint-6 Track A — Symbolic-Repair-Advisor live ({args.subset or 'all'})"
        Path(args.markdown).write_text(
            render_markdown(summary, title=title, bundle_hash=bundle_hash),
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "engine": "phase1_plus_symbolic_repair",
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
        prog="cognithor.channels.program_synthesis.synthesis.sprint6_symbolic_repair_runner",
        description=(
            "Sprint-6 Track A experiment: run Phase-1 + Symbolic-Repair-Advisor"
            " on an ARC-AGI-3 corpus subset; measure refinement uplift."
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
        default=Path("sprint6_track_a_report.json"),
    )
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--success-threshold", type=float, default=0.95)
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


__all__ = ["main"]
