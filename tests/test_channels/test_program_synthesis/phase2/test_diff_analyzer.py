# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Diff-Analyzer tests (Sprint-1 plan task 9 slice, spec §6.3)."""

from __future__ import annotations

import numpy as np
import pytest

from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.refiner import (
    ColorDiff,
    DiffReport,
    PixelDiff,
    StructureDiff,
    analyze_diff,
)


def _g(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


# ---------------------------------------------------------------------------
# Identical inputs
# ---------------------------------------------------------------------------


class TestIdenticalGrids:
    def test_identical_grids_report_no_diff(self) -> None:
        a = _g([[1, 2], [3, 4]])
        b = _g([[1, 2], [3, 4]])
        report = analyze_diff(a, b)
        assert isinstance(report, DiffReport)
        assert report.identical is True
        assert report.structure.shape_mismatch is False
        assert report.pixels.count == 0
        assert report.pixels.positions == ()
        assert report.colors.introduced == frozenset()
        assert report.colors.missing == frozenset()
        assert report.colors.shared == frozenset({1, 2, 3, 4})


# ---------------------------------------------------------------------------
# Structure diff
# ---------------------------------------------------------------------------


class TestStructureDiff:
    def test_shape_mismatch_flag_set(self) -> None:
        a = _g([[1, 2]])
        b = _g([[1, 2], [3, 4]])
        report = analyze_diff(a, b)
        assert report.structure.shape_mismatch is True
        assert report.structure.actual_shape == (1, 2)
        assert report.structure.expected_shape == (2, 2)
        assert report.identical is False

    def test_row_and_col_deltas(self) -> None:
        a = _g([[1, 2, 3]])  # 1×3
        b = _g([[1], [2]])  # 2×1
        diff = analyze_diff(a, b).structure
        assert diff.row_delta == -1  # actual smaller by 1 row
        assert diff.col_delta == 2  # actual larger by 2 cols

    def test_pixel_section_empty_on_shape_mismatch(self) -> None:
        a = _g([[1, 2]])
        b = _g([[1, 2], [3, 4]])
        report = analyze_diff(a, b)
        # Pixel diff is undefined when shapes differ — analyzer
        # returns empty positions, leaves the structural flag set.
        assert report.pixels.count == 0
        assert report.pixels.positions == ()


# ---------------------------------------------------------------------------
# Pixel diff
# ---------------------------------------------------------------------------


class TestPixelDiff:
    def test_one_cell_differs(self) -> None:
        a = _g([[1, 2], [3, 4]])
        b = _g([[1, 2], [3, 9]])  # bottom-right differs
        report = analyze_diff(a, b)
        assert report.pixels.count == 1
        assert report.pixels.positions == ((1, 1),)

    def test_all_cells_differ(self) -> None:
        a = _g([[0, 0], [0, 0]])
        b = _g([[1, 1], [1, 1]])
        report = analyze_diff(a, b)
        assert report.pixels.count == 4
        assert set(report.pixels.positions) == {(0, 0), (0, 1), (1, 0), (1, 1)}

    def test_empty_grids_no_pixel_diff(self) -> None:
        a = np.zeros((0, 0), dtype=np.int8)
        b = np.zeros((0, 0), dtype=np.int8)
        report = analyze_diff(a, b)
        assert report.pixels.count == 0


# ---------------------------------------------------------------------------
# Color diff
# ---------------------------------------------------------------------------


class TestColorDiff:
    def test_introduced_and_missing(self) -> None:
        # actual has {1, 2, 9}; expected has {1, 2, 5}.
        # introduced = {9}, missing = {5}, shared = {1, 2}.
        a = _g([[1, 2], [9, 9]])
        b = _g([[1, 2], [5, 5]])
        report = analyze_diff(a, b)
        assert report.colors.introduced == frozenset({9})
        assert report.colors.missing == frozenset({5})
        assert report.colors.shared == frozenset({1, 2})

    def test_identical_colors_no_introduce_or_missing(self) -> None:
        # Different cell positions but same color set.
        a = _g([[1, 2], [3, 4]])
        b = _g([[4, 3], [2, 1]])
        report = analyze_diff(a, b)
        assert report.colors.introduced == frozenset()
        assert report.colors.missing == frozenset()
        assert report.colors.shared == frozenset({1, 2, 3, 4})
        # Pixels still differ.
        assert report.pixels.count > 0

    def test_color_diff_robust_to_shape_mismatch(self) -> None:
        # Even with shape mismatch the color sections still report.
        a = _g([[1, 2, 3]])
        b = _g([[5], [5]])
        report = analyze_diff(a, b)
        assert report.colors.introduced == frozenset({1, 2, 3})
        assert report.colors.missing == frozenset({5})


# ---------------------------------------------------------------------------
# Type guards
# ---------------------------------------------------------------------------


class TestTypeGuards:
    def test_non_ndarray_actual_raises(self) -> None:
        with pytest.raises(TypeError, match="actual must be"):
            analyze_diff([[1]], _g([[1]]))  # type: ignore[arg-type]

    def test_non_ndarray_expected_raises(self) -> None:
        with pytest.raises(TypeError, match="expected must be"):
            analyze_diff(_g([[1]]), [[1]])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Public dataclass shape sanity
# ---------------------------------------------------------------------------


class TestPublicShapes:
    def test_all_dataclasses_frozen_and_hashable(self) -> None:
        s = StructureDiff(shape_mismatch=False, actual_shape=(2, 2), expected_shape=(2, 2))
        c = ColorDiff(introduced=frozenset(), missing=frozenset(), shared=frozenset())
        p = PixelDiff(positions=(), count=0)
        # frozen → hashable.
        assert hash(s) == hash(s)
        assert hash(c) == hash(c)
        assert hash(p) == hash(p)
