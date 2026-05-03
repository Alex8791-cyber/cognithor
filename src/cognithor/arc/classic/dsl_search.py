"""Combinatorial DSL search engine for ARC-AGI-3.

Builds candidate transforms from DSL primitives, tests them against
example pairs at increasing depths (1, 2, 3), and returns solutions
ranked by complexity (Occam's Razor).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Literal

from cognithor.arc.classic import dsl
from cognithor.arc.classic.task_parser import ArcTask, Grid, Solution

__all__ = ["build_candidates", "search"]

# Type alias for a grid transform function
TransformFn = Callable[[Grid], Grid]

# ARC color range
_COLORS = range(10)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _grids_equal(a: Grid, b: Grid) -> bool:
    """Check if two grids are identical (same dimensions + same values)."""
    if len(a) != len(b):
        return False
    for row_a, row_b in zip(a, b, strict=False):
        if len(row_a) != len(row_b):
            return False
        if row_a != row_b:
            return False
    return True


def _collect_colors(examples: list[tuple[Grid, Grid]]) -> set[int]:
    """Collect all colors that appear in the example grids."""
    colors: set[int] = set()
    for inp, out in examples:
        for row in inp:
            colors.update(row)
        for row in out:
            colors.update(row)
    return colors


# ---------------------------------------------------------------------------
# Candidate builder
# ---------------------------------------------------------------------------


def build_candidates(
    examples: list[tuple[Grid, Grid]] | None = None,
) -> list[tuple[str, TransformFn]]:
    """Build the full list of depth-1 candidate transforms.

    Each entry is (human-readable name, fn: Grid -> Grid).
    When *examples* is provided, color-param ops are restricted to colors
    that actually appear in the examples (keeps the search space tight).
    """
    candidates: list[tuple[str, TransformFn]] = []

    # --- No-param ops (~8) ---
    candidates.append(("rotate_90", dsl.rotate_90))
    candidates.append(("rotate_180", dsl.rotate_180))
    candidates.append(("rotate_270", dsl.rotate_270))
    candidates.append(("flip_h", dsl.flip_h))
    candidates.append(("flip_v", dsl.flip_v))
    candidates.append(("transpose", dsl.transpose))
    candidates.append(("invert_colors", dsl.invert_colors))
    candidates.append(("crop_to_content", dsl.crop_to_content))

    # Determine color palette
    colors = sorted(_collect_colors(examples)) if examples else list(_COLORS)

    def _recolor(a: int, b: int) -> TransformFn:
        return lambda g: dsl.recolor(g, a, b)

    def _swap(a: int, b: int) -> TransformFn:
        return lambda g: dsl.swap_colors(g, a, b)

    def _replace_bg(c: int) -> TransformFn:
        return lambda g: dsl.replace_background(g, c)

    def _by_color(c: int) -> TransformFn:
        return lambda g: dsl.get_by_color(g, c)

    def _scale(f: int) -> TransformFn:
        return lambda g: dsl.scale_up(g, f)

    def _tile(nx: int, ny: int) -> TransformFn:
        return lambda g: dsl.tile(g, nx, ny)

    def _pad(n: int) -> TransformFn:
        return lambda g: dsl.pad(g, n)

    def _gravity(d: Literal["down", "up", "left", "right"]) -> TransformFn:
        return lambda g: dsl.gravity(g, d)

    # --- Color-param ops ---
    for a in colors:
        for b in colors:
            if a != b:
                candidates.append((f"recolor({a},{b})", _recolor(a, b)))
    for a in colors:
        for b in colors:
            if a < b:
                candidates.append((f"swap_colors({a},{b})", _swap(a, b)))
    for c in colors:
        candidates.append((f"replace_background({c})", _replace_bg(c)))
    for c in colors:
        candidates.append((f"get_by_color({c})", _by_color(c)))

    # --- Shape-param ops ---
    for factor in (2, 3):
        candidates.append((f"scale_up({factor})", _scale(factor)))
    for nx, ny in ((2, 1), (1, 2), (2, 2)):
        candidates.append((f"tile({nx},{ny})", _tile(nx, ny)))
    for n in (1, 2):
        candidates.append((f"pad({n})", _pad(n)))
    for direction in ("down", "up", "left", "right"):
        candidates.append((f"gravity({direction})", _gravity(direction)))

    return candidates


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _matches_all_examples(
    fn: TransformFn,
    examples: list[tuple[Grid, Grid]],
) -> bool:
    """Return True if fn(input) == output for ALL example pairs."""
    for inp, out in examples:
        try:
            result = fn(inp)
        except Exception:
            return False
        if not _grids_equal(result, out):
            return False
    return True


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def search(
    task: ArcTask,
    *,
    timeout: float = 10.0,
    max_depth: int = 3,
) -> list[Solution]:
    """Run combinatorial DSL search on *task*.

    Returns solutions sorted by complexity (depth, then name length).
    Stops early when a shallower depth finds solutions.
    """
    deadline = time.monotonic() + timeout
    examples = task.examples
    candidates = build_candidates(examples)
    solutions: list[Solution] = []

    def _timed_out() -> bool:
        return time.monotonic() >= deadline

    # --- Depth 1 ---
    for name, fn in candidates:
        if _timed_out():
            break
        if _matches_all_examples(fn, examples):
            solutions.append(
                Solution(
                    output=fn(task.test_input),
                    method="dsl_search",
                    description=name,
                    complexity=1,
                    transform_fn=fn,
                )
            )

    if solutions or max_depth < 2:
        solutions.sort(key=lambda s: (s.complexity, len(s.description)))
        return solutions

    # --- Depth 2 ---
    for name_a, fn_a in candidates:
        if _timed_out():
            break
        for name_b, fn_b in candidates:
            if _timed_out():
                break
            composed_name = f"{name_b}({name_a}(x))"

            def _composed(
                g: Grid,
                _fa: TransformFn = fn_a,
                _fb: TransformFn = fn_b,
            ) -> Grid:
                return _fb(_fa(g))

            if _matches_all_examples(_composed, examples):
                solutions.append(
                    Solution(
                        output=_composed(task.test_input),
                        method="dsl_search",
                        description=composed_name,
                        complexity=2,
                        transform_fn=_composed,
                    )
                )

    if solutions or max_depth < 3:
        solutions.sort(key=lambda s: (s.complexity, len(s.description)))
        return solutions

    # --- Depth 3 ---
    for name_a, fn_a in candidates:
        if _timed_out():
            break
        for name_b, fn_b in candidates:
            if _timed_out():
                break
            for name_c, fn_c in candidates:
                if _timed_out():
                    break
                composed_name = f"{name_c}({name_b}({name_a}(x)))"

                def _composed3(
                    g: Grid,
                    _fa: TransformFn = fn_a,
                    _fb: TransformFn = fn_b,
                    _fc: TransformFn = fn_c,
                ) -> Grid:
                    return _fc(_fb(_fa(g)))

                if _matches_all_examples(_composed3, examples):
                    solutions.append(
                        Solution(
                            output=_composed3(task.test_input),
                            method="dsl_search",
                            description=composed_name,
                            complexity=3,
                            transform_fn=_composed3,
                        )
                    )

    solutions.sort(key=lambda s: (s.complexity, len(s.description)))
    return solutions
