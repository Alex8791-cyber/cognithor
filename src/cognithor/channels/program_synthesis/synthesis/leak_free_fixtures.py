# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-2 Track C — 20-Task Leak-Free benchmark fixture set.

Twenty synthetic ARC-AGI-3-shaped tasks for the Sprint-2 / Sprint-3
benchmark drivers. Each task exposes:

* a ``task_id`` — stable string keyed by category + ordinal;
* ``examples`` — paired ``(input, output)`` numpy grids;
* a known canonical solution primitive (in the task's ``solution_hint``
  metadata) so unit tests can assert "Phase-1 solves at least X of these
  by enumeration".

The tasks cover every transformation category the Symbolic-Prior
catalog (PR #257) recognises: rotation, mirror, recolor, scale,
crop, and mixed compositions. The split is intentionally biased
toward what Phase-1 can solve cleanly (so Sprint-2 A/B-tests can
isolate Refiner *uplift* on the partials).

Leak-Free guarantee:

The :func:`leak_free_set_hash` returns a SHA-256 over the canonical
serialisation of every task in :data:`LEAK_FREE_TASKS`. The
synthesis cache (spec §14.1) keys on
:meth:`TaskSpec.stable_hash`, which incorporates examples + held-out
+ test-input + constraints + domain. Because these fixtures are
*synthetic* and never appear in the live cache (the cache is empty
on a fresh installation, and these tasks are not part of any
training corpus), their stable hashes are leak-free by construction.

Tests assert the bundle hash is stable across runs — any drift
fails CI.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from cognithor.channels.program_synthesis.core.types import (
    Budget,
    Example,
    Grid,
    TaskSpec,
)
from cognithor.channels.program_synthesis.phase2.datatypes import (
    PartitionedBudget,
)
from cognithor.channels.program_synthesis.synthesis.benchmark import (
    BenchmarkTask,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


# ---------------------------------------------------------------------------
# Task metadata wrapper
# ---------------------------------------------------------------------------


TransformationCategory = Literal["rotation", "mirror", "recolor", "scale", "crop", "mixed"]


@dataclass(frozen=True)
class LeakFreeTask:
    """One synthetic ARC-AGI-3-shaped task with documented solution.

    ``solution_hint`` is the canonical primitive name(s) a human
    would expect to solve this task — used by sanity tests, NOT
    fed to the search engine.
    """

    task_id: str
    category: TransformationCategory
    examples: tuple[tuple[Grid, Grid], ...]
    solution_hint: str
    description: str

    def to_task_spec(self) -> TaskSpec:
        return TaskSpec(examples=tuple(_typed_example(p) for p in self.examples))

    def to_benchmark_task(
        self,
        *,
        budget: PartitionedBudget,
        wall_clock_budget_seconds: float,
        current_alpha: float = 0.6,
    ) -> BenchmarkTask:
        return BenchmarkTask(
            task_id=self.task_id,
            spec=self.to_task_spec(),
            budget=budget,
            wall_clock_budget_seconds=wall_clock_budget_seconds,
            current_alpha=current_alpha,
        )


def _typed_example(pair: tuple[Grid, Grid]) -> Example:
    inp, out = pair
    return (np.asarray(inp, dtype=np.int8), np.asarray(out, dtype=np.int8))


# ---------------------------------------------------------------------------
# Helpers for building synthetic grids
# ---------------------------------------------------------------------------


def _g(rows: list[list[int]]) -> np.ndarray[Any, Any]:
    return np.array(rows, dtype=np.int8)


def _rotate90_cw(grid: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.rot90(grid, k=-1)


def _rotate180(grid: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.rot90(grid, k=2)


def _rotate270_cw(grid: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.rot90(grid, k=1)


def _scale_up(grid: np.ndarray[Any, Any], factor: int) -> np.ndarray[Any, Any]:
    return np.repeat(np.repeat(grid, factor, axis=0), factor, axis=1)


def _scale_down_2x(grid: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    # Take every other row/column.
    return grid[::2, ::2]


def _recolor(grid: np.ndarray[Any, Any], src: int, dst: int) -> np.ndarray[Any, Any]:
    out = grid.copy()
    out[grid == src] = dst
    return out


# ---------------------------------------------------------------------------
# The 20 tasks
# ---------------------------------------------------------------------------


def _build_tasks() -> tuple[LeakFreeTask, ...]:
    out: list[LeakFreeTask] = []

    # ── Rotation (4) ─────────────────────────────────────────────
    g1 = _g([[1, 2], [3, 4]])
    out.append(
        LeakFreeTask(
            task_id="0001_rotate90_2x2",
            category="rotation",
            examples=(
                (g1, _rotate90_cw(g1)),
                (_g([[5, 6], [7, 8]]), _rotate90_cw(_g([[5, 6], [7, 8]]))),
            ),
            solution_hint="rotate90",
            description="2x2 grid, 90° clockwise rotation",
        )
    )

    g2 = _g([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    out.append(
        LeakFreeTask(
            task_id="0002_rotate180_3x3",
            category="rotation",
            examples=(
                (g2, _rotate180(g2)),
                (
                    _g([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
                    _rotate180(_g([[0, 1, 2], [3, 4, 5], [6, 7, 8]])),
                ),
            ),
            solution_hint="rotate180",
            description="3x3 grid, 180° rotation",
        )
    )

    g3 = _g([[1, 2], [3, 4]])
    out.append(
        LeakFreeTask(
            task_id="0003_rotate270_2x2",
            category="rotation",
            examples=((g3, _rotate270_cw(g3)),),
            solution_hint="rotate270",
            description="2x2 grid, 270° clockwise rotation",
        )
    )

    g4 = _g([[1, 2, 3, 4], [5, 6, 7, 8]])
    out.append(
        LeakFreeTask(
            task_id="0004_rotate90_2x4",
            category="rotation",
            examples=((g4, _rotate90_cw(g4)),),
            solution_hint="rotate90",
            description="2x4 rectangular grid, 90° clockwise (becomes 4x2)",
        )
    )

    # ── Mirror (3) ───────────────────────────────────────────────
    g5 = _g([[1, 2, 3], [4, 5, 6]])
    out.append(
        LeakFreeTask(
            task_id="0005_mirror_horizontal",
            category="mirror",
            examples=((g5, np.fliplr(g5)),),
            solution_hint="mirror_horizontal",
            description="Left-right mirror of a 2x3 grid",
        )
    )

    g6 = _g([[1, 2], [3, 4], [5, 6]])
    out.append(
        LeakFreeTask(
            task_id="0006_mirror_vertical",
            category="mirror",
            examples=((g6, np.flipud(g6)),),
            solution_hint="mirror_vertical",
            description="Top-bottom mirror of a 3x2 grid",
        )
    )

    g7 = _g([[1, 2], [3, 4]])
    out.append(
        LeakFreeTask(
            task_id="0007_transpose_2x2",
            category="mirror",
            examples=((g7, g7.T),),
            solution_hint="transpose",
            description="Transpose of a 2x2 grid (mirror_diagonal)",
        )
    )

    # ── Recolor (3) ──────────────────────────────────────────────
    g8 = _g([[1, 2], [1, 3]])
    out.append(
        LeakFreeTask(
            task_id="0008_recolor_1_to_5",
            category="recolor",
            examples=(
                (g8, _recolor(g8, 1, 5)),
                (_g([[1, 1], [2, 1]]), _recolor(_g([[1, 1], [2, 1]]), 1, 5)),
            ),
            solution_hint="recolor",
            description="Recolor every 1 to 5",
        )
    )

    g9 = _g([[0, 1], [2, 0]])
    out.append(
        LeakFreeTask(
            task_id="0009_recolor_0_to_7",
            category="recolor",
            examples=((g9, _recolor(g9, 0, 7)),),
            solution_hint="recolor",
            description="Recolor background (0) to 7",
        )
    )

    g10 = _g([[1, 2, 3], [3, 2, 1]])
    out.append(
        LeakFreeTask(
            task_id="0010_recolor_3_to_9",
            category="recolor",
            examples=((g10, _recolor(g10, 3, 9)),),
            solution_hint="recolor",
            description="Recolor 3 to 9 in a 2x3 palette",
        )
    )

    # ── Scale (3) ────────────────────────────────────────────────
    g11 = _g([[1, 2]])
    out.append(
        LeakFreeTask(
            task_id="0011_scale_up_2x_1x2",
            category="scale",
            examples=((g11, _scale_up(g11, 2)), (_g([[3, 4]]), _scale_up(_g([[3, 4]]), 2))),
            solution_hint="scale_up_2x",
            description="2× upscale of a 1x2 grid",
        )
    )

    g12 = _g([[1]])
    out.append(
        LeakFreeTask(
            task_id="0012_scale_up_3x_1x1",
            category="scale",
            examples=((g12, _scale_up(g12, 3)), (_g([[5]]), _scale_up(_g([[5]]), 3))),
            solution_hint="scale_up_3x",
            description="3× upscale of a 1x1 grid",
        )
    )

    g13 = _g([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])
    out.append(
        LeakFreeTask(
            task_id="0013_scale_down_2x_4x4",
            category="scale",
            examples=((g13, _scale_down_2x(g13)),),
            solution_hint="scale_down_2x",
            description="2× downscale of a block-pixel 4x4 grid",
        )
    )

    # ── Crop (3) ─────────────────────────────────────────────────
    # crop_bbox crops to the non-background bounding box.
    g14 = _g([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    out.append(
        LeakFreeTask(
            task_id="0014_crop_bbox_single_pixel",
            category="crop",
            examples=((g14, _g([[1]])),),
            solution_hint="crop_bbox",
            description="Crop to the bounding box of a single non-zero pixel",
        )
    )

    g15 = _g([[0, 0, 0], [0, 1, 2], [0, 3, 4]])
    out.append(
        LeakFreeTask(
            task_id="0015_crop_bbox_2x2_subregion",
            category="crop",
            examples=((g15, _g([[1, 2], [3, 4]])),),
            solution_hint="crop_bbox",
            description="Crop to a 2x2 non-zero subregion",
        )
    )

    g16 = _g([[5, 0], [0, 0]])
    out.append(
        LeakFreeTask(
            task_id="0016_crop_bbox_corner",
            category="crop",
            examples=((g16, _g([[5]])),),
            solution_hint="crop_bbox",
            description="Crop to a single non-zero corner cell",
        )
    )

    # ── Mixed (4) ────────────────────────────────────────────────
    # rotate90 then recolor.
    g17 = _g([[1, 2], [3, 4]])
    expected_17 = _recolor(_rotate90_cw(g17), 1, 5)
    out.append(
        LeakFreeTask(
            task_id="0017_rotate90_then_recolor",
            category="mixed",
            examples=((g17, expected_17),),
            solution_hint="recolor(rotate90(input), 1, 5)",
            description="Compose: rotate 90° clockwise then recolor 1→5",
        )
    )

    # mirror_horizontal then rotate180 (= mirror_vertical equivalent).
    g18 = _g([[1, 2, 3], [4, 5, 6]])
    expected_18 = _rotate180(np.fliplr(g18))
    out.append(
        LeakFreeTask(
            task_id="0018_mirror_h_then_rotate180",
            category="mixed",
            examples=((g18, expected_18),),
            solution_hint="rotate180(mirror_horizontal(input))",
            description="Compose: horizontal mirror + 180° rotation",
        )
    )

    # scale_up_2x then recolor (background -> something).
    g19 = _g([[1, 0]])
    expected_19 = _recolor(_scale_up(g19, 2), 0, 7)
    out.append(
        LeakFreeTask(
            task_id="0019_scale_up_then_recolor",
            category="mixed",
            examples=((g19, expected_19),),
            solution_hint="recolor(scale_up_2x(input), 0, 7)",
            description="Compose: 2× upscale + recolor 0→7",
        )
    )

    # transpose then mirror_horizontal.
    g20 = _g([[1, 2], [3, 4]])
    expected_20 = np.fliplr(g20.T)
    out.append(
        LeakFreeTask(
            task_id="0020_transpose_then_mirror",
            category="mixed",
            examples=((g20, expected_20),),
            solution_hint="mirror_horizontal(transpose(input))",
            description="Compose: transpose + horizontal mirror",
        )
    )

    return tuple(out)


LEAK_FREE_TASKS: tuple[LeakFreeTask, ...] = _build_tasks()
"""The frozen 20-task fixture set."""


# ---------------------------------------------------------------------------
# Leak-free hash + bundle helpers
# ---------------------------------------------------------------------------


def leak_free_set_hash(tasks: Sequence[LeakFreeTask] = LEAK_FREE_TASKS) -> str:
    """SHA-256 over the canonical (task_id, examples) serialisation.

    Stable across runs and Python versions. CI test asserts the
    expected digest so any drift fails immediately.
    """
    canonical: list[dict[str, Any]] = []
    for task in tasks:
        canonical.append(
            {
                "id": task.task_id,
                "category": task.category,
                "examples": [
                    [
                        np.asarray(inp, dtype=np.int8).tolist(),
                        np.asarray(out, dtype=np.int8).tolist(),
                    ]
                    for inp, out in task.examples
                ],
            }
        )
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def benchmark_tasks(
    *,
    budget: PartitionedBudget | None = None,
    wall_clock_budget_seconds: float = 5.0,
    current_alpha: float = 0.6,
) -> tuple[BenchmarkTask, ...]:
    """Convert every leak-free task into a :class:`BenchmarkTask`.

    Convenience for the benchmark driver — production flow is:

        from cognithor.channels.program_synthesis.synthesis import (
            benchmark_tasks, run_benchmark,
        )
        summary = await run_benchmark(engine, benchmark_tasks())
    """
    if budget is None:
        budget = PartitionedBudget.from_spec_default()
    return tuple(
        task.to_benchmark_task(
            budget=budget,
            wall_clock_budget_seconds=wall_clock_budget_seconds,
            current_alpha=current_alpha,
        )
        for task in LEAK_FREE_TASKS
    )


# Reserved for future modules; ``Budget`` from core types is not used in
# this fixture but is exported for downstream benchmark configuration.
_ = Budget


__all__ = [
    "LEAK_FREE_TASKS",
    "LeakFreeTask",
    "TransformationCategory",
    "benchmark_tasks",
    "leak_free_set_hash",
]
