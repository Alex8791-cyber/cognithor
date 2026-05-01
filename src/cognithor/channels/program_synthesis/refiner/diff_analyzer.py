# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Spec §6.3 — Diff-Analyzer (Sprint-1 plan task 9 slice).

Compares a candidate program's actual output against the expected
demo output and reports the diff in three orthogonal dimensions:

* **Pixel-Diff** — set of ``(row, col)`` positions where the two
  grids disagree. Empty iff the grids are identical.
* **Struktur-Diff** — shape mismatch + grid-size disagreement.
* **Farb-Diff** — which colors are *introduced* (in actual but not
  expected), *missing* (in expected but not actual), and *swapped*
  (a hint when one color in expected got systematically replaced by
  another in actual).

The Diff-Report feeds the Symbolic-Repair heuristics: a Farb-Diff
suggests inserting a ``recolor`` primitive; a Struktur-Diff suggests
a ``scale`` or ``crop``; a Pixel-Diff localised to one quadrant
suggests targeted mutation.

The module is stateless + pure-math. No Phase-2 config dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StructureDiff:
    """Shape-level disagreement.

    ``shape_mismatch`` is True iff the two grids' ``.shape`` differ.
    ``actual_shape`` and ``expected_shape`` echo the inputs for
    downstream callers (the Symbolic-Repair heuristics use the row
    and column deltas to choose between scale-up, scale-down, crop,
    pad, etc.).
    """

    shape_mismatch: bool
    actual_shape: tuple[int, ...]
    expected_shape: tuple[int, ...]

    @property
    def row_delta(self) -> int:
        if not self.actual_shape or not self.expected_shape:
            return 0
        return self.actual_shape[0] - self.expected_shape[0]

    @property
    def col_delta(self) -> int:
        if len(self.actual_shape) < 2 or len(self.expected_shape) < 2:
            return 0
        return self.actual_shape[1] - self.expected_shape[1]


@dataclass(frozen=True)
class ColorDiff:
    """Per-color disagreement.

    * ``introduced`` — colors present in the actual output but not in
      the expected (the candidate over-coloured something).
    * ``missing`` — colors present in the expected but not in the
      actual (the candidate failed to produce a color).
    * ``shared`` — colors present in both.
    """

    introduced: frozenset[int]
    missing: frozenset[int]
    shared: frozenset[int]


@dataclass(frozen=True)
class PixelDiff:
    """Cell-level disagreement.

    ``positions`` is a tuple of ``(row, col)`` pairs where the two
    grids differ. Empty iff the grids match. ``count`` is a cached
    ``len(positions)`` so consumers don't recompute.

    For shape-mismatched inputs the analyzer returns an empty
    positions tuple — the Pixel-Diff is undefined when the dimensions
    don't line up; the caller should consult :class:`StructureDiff`
    instead.
    """

    positions: tuple[tuple[int, int], ...]
    count: int = 0


@dataclass(frozen=True)
class DiffReport:
    """Composite diff: structure + pixels + colors.

    ``identical`` is the convenience flag — True iff the two grids
    are pixel-identical (no structural mismatch, zero pixel diffs,
    no color introductions or missings).
    """

    structure: StructureDiff
    pixels: PixelDiff
    colors: ColorDiff
    identical: bool = False


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


def analyze_diff(actual: np.ndarray[Any, Any], expected: np.ndarray[Any, Any]) -> DiffReport:
    """Build a :class:`DiffReport` for the (actual, expected) pair.

    The function is total — even pathological inputs (different
    shapes, empty grids) produce a well-formed report; consumers
    branch on :attr:`StructureDiff.shape_mismatch` before reading
    the pixel/color sections.
    """
    if not isinstance(actual, np.ndarray):
        raise TypeError(f"analyze_diff: actual must be numpy.ndarray, got {type(actual).__name__}")
    if not isinstance(expected, np.ndarray):
        raise TypeError(
            f"analyze_diff: expected must be numpy.ndarray, got {type(expected).__name__}"
        )

    structure = StructureDiff(
        shape_mismatch=actual.shape != expected.shape,
        actual_shape=tuple(actual.shape),
        expected_shape=tuple(expected.shape),
    )

    if actual.shape == expected.shape and actual.size > 0:
        diff_mask = actual != expected
        positions = tuple((int(r), int(c)) for r, c in zip(*np.where(diff_mask), strict=True))
    else:
        positions = ()

    pixels = PixelDiff(positions=positions, count=len(positions))

    actual_colors = frozenset(int(c) for c in np.unique(actual)) if actual.size else frozenset()
    expected_colors = (
        frozenset(int(c) for c in np.unique(expected)) if expected.size else frozenset()
    )
    colors = ColorDiff(
        introduced=actual_colors - expected_colors,
        missing=expected_colors - actual_colors,
        shared=actual_colors & expected_colors,
    )

    identical = (
        not structure.shape_mismatch
        and pixels.count == 0
        and not colors.introduced
        and not colors.missing
    )

    return DiffReport(
        structure=structure,
        pixels=pixels,
        colors=colors,
        identical=identical,
    )


__all__ = [
    "ColorDiff",
    "DiffReport",
    "PixelDiff",
    "StructureDiff",
    "analyze_diff",
]


# Suppress unused-field warnings for spec-reserved fields.
_ = field
