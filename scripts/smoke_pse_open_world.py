#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-22 — open-world readiness smoke for the PSE channel.

Runs a small spread of synthesis tasks of varying difficulty against
the central :class:`ProgramSynthesisChannel` to gauge how the engine
performs outside the curated ARC-AGI-3 benchmark fixtures.

Each task is shaped as a list of (input_grid, output_grid) examples.
We hit them via the MCP-handler ``handle_pse_synthesize`` so we
exercise the same code path a real Cognithor caller would use.

Reports per-task: status, cost_seconds, cost_candidates, cache_hit,
program shape (truncated). Final summary line groups by status.
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from typing import Any


def _grid(*rows: list[int]) -> list[list[int]]:
    return [list(row) for row in rows]


# Open-world task spread — deliberately heterogeneous so we surface
# both engine wins and engine limits.
TASKS: list[tuple[str, list[dict[str, Any]]]] = [
    (
        "identity-1x1",
        [
            {"input": _grid([0]), "output": _grid([0])},
            {"input": _grid([1]), "output": _grid([1])},
            {"input": _grid([5]), "output": _grid([5])},
        ],
    ),
    (
        "identity-3x3",
        [
            {
                "input": _grid([0, 1, 2], [3, 4, 5], [6, 7, 8]),
                "output": _grid([0, 1, 2], [3, 4, 5], [6, 7, 8]),
            },
            {
                "input": _grid([1, 1, 1], [2, 2, 2], [3, 3, 3]),
                "output": _grid([1, 1, 1], [2, 2, 2], [3, 3, 3]),
            },
        ],
    ),
    (
        "constant-output-zero",
        [
            {"input": _grid([5, 5], [5, 5]), "output": _grid([0, 0], [0, 0])},
            {"input": _grid([3, 3], [3, 3]), "output": _grid([0, 0], [0, 0])},
            {"input": _grid([7, 1], [2, 9]), "output": _grid([0, 0], [0, 0])},
        ],
    ),
    (
        "color-swap-0-and-1",
        [
            {
                "input": _grid([0, 1, 0], [1, 0, 1]),
                "output": _grid([1, 0, 1], [0, 1, 0]),
            },
            {
                "input": _grid([1, 1, 1], [0, 0, 0]),
                "output": _grid([0, 0, 0], [1, 1, 1]),
            },
            {
                "input": _grid([0, 0], [1, 1]),
                "output": _grid([1, 1], [0, 0]),
            },
        ],
    ),
    (
        "rotate-90",
        [
            {"input": _grid([1, 2], [3, 4]), "output": _grid([3, 1], [4, 2])},
            {
                "input": _grid([1, 2, 3], [4, 5, 6], [7, 8, 9]),
                "output": _grid([7, 4, 1], [8, 5, 2], [9, 6, 3]),
            },
        ],
    ),
    (
        "fill-diagonal",
        [
            {
                "input": _grid([0, 0, 0], [0, 0, 0], [0, 0, 0]),
                "output": _grid([5, 0, 0], [0, 5, 0], [0, 0, 5]),
            },
            {
                "input": _grid([0, 0], [0, 0]),
                "output": _grid([5, 0], [0, 5]),
            },
        ],
    ),
    (
        "grow-by-row",
        [
            {"input": _grid([1]), "output": _grid([1], [1])},
            {"input": _grid([2]), "output": _grid([2], [2])},
        ],
    ),
    (
        "size-variant",  # different example sizes — stress test
        [
            {"input": _grid([0]), "output": _grid([1])},
            {"input": _grid([0, 0], [0, 0]), "output": _grid([1, 1], [1, 1])},
            {
                "input": _grid([0, 0, 0], [0, 0, 0], [0, 0, 0]),
                "output": _grid([1, 1, 1], [1, 1, 1], [1, 1, 1]),
            },
        ],
    ),
]


async def run() -> int:
    try:
        from cognithor.channels.program_synthesis import DSL_VERSION, PSE_VERSION
        from cognithor.mcp.pse_tools import handle_pse_synthesize
    except ImportError as exc:
        print(f"FATAL: PSE module not importable: {exc}")
        return 2

    print("=== PSE open-world smoke ===")
    print(f"engine: PSE_VERSION={PSE_VERSION}, DSL_VERSION={DSL_VERSION}")
    print(f"tasks:  {len(TASKS)}")
    print()

    # Modest budget — these are mostly small grids, the engine should
    # handle them in seconds when it can handle them at all.
    budget = {"max_depth": 4, "max_candidates": 50_000, "wall_clock_seconds": 10.0}

    statuses: list[str] = []
    rows: list[tuple[str, str, float, int, str]] = []
    t0 = time.monotonic()
    for name, examples in TASKS:
        try:
            result = await handle_pse_synthesize(examples=examples, budget=budget)
        except Exception as exc:
            rows.append((name, f"crash: {type(exc).__name__}", 0.0, 0, str(exc)[:60]))
            statuses.append("crash")
            continue
        if "error" in result:
            rows.append((name, "input_error", 0.0, 0, result["error"][:60]))
            statuses.append("input_error")
            continue
        status = result.get("status", "?")
        statuses.append(status)
        program = result.get("program") or "<no program>"
        rows.append(
            (
                name,
                status,
                float(result.get("cost_seconds", 0.0)),
                int(result.get("cost_candidates", 0)),
                program[:60].replace("\n", " ") if isinstance(program, str) else "?",
            )
        )

    total_wall = time.monotonic() - t0

    print(f"{'task':<22} {'status':<14} {'sec':>7} {'cand':>9}  program-snippet")
    print("-" * 110)
    for name, status, sec, cand, prog in rows:
        print(f"{name:<22} {status:<14} {sec:>7.2f} {cand:>9}  {prog}")
    print()
    by_status = Counter(statuses)
    summary = ", ".join(f"{k}={v}" for k, v in sorted(by_status.items()))
    success_count = by_status.get("success", 0)
    print(f"summary: {summary}")
    print(f"success rate: {success_count}/{len(TASKS)} = {100 * success_count / len(TASKS):.1f}%")
    print(f"total wall-clock: {total_wall:.2f}s")
    return 0 if success_count > 0 else 1


def main() -> None:
    import sys

    sys.exit(asyncio.run(run()))


if __name__ == "__main__":
    main()
