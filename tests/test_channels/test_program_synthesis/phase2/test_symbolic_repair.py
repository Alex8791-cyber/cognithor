# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Symbolic-Repair Advisor tests (Sprint-1 plan task 9 slice, §6.5.2)."""

from __future__ import annotations

import numpy as np

from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.refiner import (
    RepairSuggestion,
    advise_repairs,
    analyze_diff,
)
from cognithor.channels.program_synthesis.refiner.symbolic_repair import (
    r1_color_repair,
    r2_scale_repair,
    r3_rotation_repair,
    r4_mirror_repair,
    r5_local_repair,
)


def _g(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


# ---------------------------------------------------------------------------
# R1 — color repair
# ---------------------------------------------------------------------------


class TestR1ColorRepair:
    def test_fires_on_introduced_and_missing(self) -> None:
        a = _g([[1, 2], [9, 9]])  # actual has 9 instead of 5
        e = _g([[1, 2], [5, 5]])
        diff = analyze_diff(a, e)
        s = r1_color_repair(diff)
        assert s is not None
        assert s.kind == "color_repair"
        assert s.primitive_hint == "recolor"
        assert s.confidence == 0.8

    def test_lower_confidence_on_shape_mismatch(self) -> None:
        a = _g([[9]])
        e = _g([[5, 5]])  # shape differs AND color differs
        diff = analyze_diff(a, e)
        s = r1_color_repair(diff)
        assert s is not None
        assert s.confidence == 0.4

    def test_does_not_fire_when_colors_identical(self) -> None:
        a = _g([[1, 2], [3, 4]])
        e = _g([[4, 3], [2, 1]])  # same colors, different positions
        diff = analyze_diff(a, e)
        assert r1_color_repair(diff) is None


# ---------------------------------------------------------------------------
# R2 — scale repair
# ---------------------------------------------------------------------------


class TestR2ScaleRepair:
    def test_fires_on_2x_upscale(self) -> None:
        a = _g([[1, 2]])
        e = _g([[1, 2, 1, 2], [1, 2, 1, 2]])
        diff = analyze_diff(a, e)
        s = r2_scale_repair(diff)
        assert s is not None
        assert s.primitive_hint == "scale_up_2x"

    def test_fires_on_3x_upscale(self) -> None:
        a = _g([[1]])
        e = _g([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        diff = analyze_diff(a, e)
        s = r2_scale_repair(diff)
        assert s is not None
        assert s.primitive_hint == "scale_up_3x"

    def test_fires_on_2x_downscale(self) -> None:
        a = _g([[1, 1], [1, 1]])
        e = _g([[1]])
        diff = analyze_diff(a, e)
        s = r2_scale_repair(diff)
        assert s is not None
        assert s.primitive_hint == "scale_down_2x"

    def test_does_not_fire_on_non_integer_ratio(self) -> None:
        a = _g([[1, 2]])  # 1×2
        e = _g([[1, 2, 3]])  # 1×3 (not 2× or 3×)
        diff = analyze_diff(a, e)
        assert r2_scale_repair(diff) is None


# ---------------------------------------------------------------------------
# R3 — rotation repair
# ---------------------------------------------------------------------------


class TestR3RotationRepair:
    def test_fires_on_90_rotation(self) -> None:
        a = _g([[1, 2], [3, 4]])
        e = np.rot90(a, k=1)
        diff = analyze_diff(a, e)
        s = r3_rotation_repair(a, e, diff)
        assert s is not None
        assert s.primitive_hint == "rotate90"

    def test_fires_on_180_rotation(self) -> None:
        a = _g([[1, 2], [3, 4]])
        e = np.rot90(a, k=2)
        diff = analyze_diff(a, e)
        s = r3_rotation_repair(a, e, diff)
        assert s is not None
        assert s.primitive_hint == "rotate180"

    def test_does_not_fire_on_unrelated_diff(self) -> None:
        a = _g([[1, 2], [3, 4]])
        e = _g([[9, 9], [9, 9]])
        diff = analyze_diff(a, e)
        assert r3_rotation_repair(a, e, diff) is None


# ---------------------------------------------------------------------------
# R4 — mirror repair
# ---------------------------------------------------------------------------


class TestR4MirrorRepair:
    def test_fires_on_horizontal_flip(self) -> None:
        a = _g([[1, 2, 3], [4, 5, 6]])
        e = np.fliplr(a)
        diff = analyze_diff(a, e)
        s = r4_mirror_repair(a, e, diff)
        assert s is not None
        assert s.primitive_hint == "mirror_horizontal"

    def test_fires_on_vertical_flip(self) -> None:
        a = _g([[1, 2], [3, 4]])
        e = np.flipud(a)
        diff = analyze_diff(a, e)
        s = r4_mirror_repair(a, e, diff)
        assert s is not None
        assert s.primitive_hint == "mirror_vertical"

    def test_does_not_fire_on_shape_mismatch(self) -> None:
        a = _g([[1, 2]])
        e = _g([[1], [2]])
        diff = analyze_diff(a, e)
        assert r4_mirror_repair(a, e, diff) is None


# ---------------------------------------------------------------------------
# R5 — local repair
# ---------------------------------------------------------------------------


class TestR5LocalRepair:
    def test_fires_on_small_pixel_diff(self) -> None:
        a = _g([[1, 2], [3, 4]])
        e = _g([[1, 2], [3, 9]])  # one pixel differs
        diff = analyze_diff(a, e)
        s = r5_local_repair(diff)
        assert s is not None
        assert s.primitive_hint is None  # local-edit, not a primitive
        assert "1 pixel" in s.detail

    def test_does_not_fire_on_large_pixel_diff(self) -> None:
        a = np.zeros((10, 10), dtype=np.int8)
        e = np.ones((10, 10), dtype=np.int8)  # 100 pixels differ
        diff = analyze_diff(a, e)
        assert r5_local_repair(diff) is None

    def test_does_not_fire_on_shape_mismatch(self) -> None:
        a = _g([[1]])
        e = _g([[1, 2]])
        diff = analyze_diff(a, e)
        assert r5_local_repair(diff) is None


# ---------------------------------------------------------------------------
# Advisor — sorts by confidence
# ---------------------------------------------------------------------------


class TestAdviseRepairs:
    def test_pure_rotation_yields_only_rotation_repair(self) -> None:
        a = _g([[1, 2], [3, 4]])
        e = np.rot90(a, k=1)
        diff = analyze_diff(a, e)
        suggestions = advise_repairs(a, e, diff)
        kinds = [s.kind for s in suggestions]
        assert "rotation_repair" in kinds
        # Top suggestion should be rotation (highest confidence 0.95).
        assert suggestions[0].kind == "rotation_repair"

    def test_color_only_diff_yields_color_repair(self) -> None:
        a = _g([[1, 2], [9, 9]])
        e = _g([[1, 2], [5, 5]])
        diff = analyze_diff(a, e)
        suggestions = advise_repairs(a, e, diff)
        assert any(s.kind == "color_repair" for s in suggestions)

    def test_identical_grids_yield_no_suggestions(self) -> None:
        a = _g([[1, 2]])
        e = _g([[1, 2]])
        diff = analyze_diff(a, e)
        suggestions = advise_repairs(a, e, diff)
        assert suggestions == []

    def test_sorted_by_confidence_descending(self) -> None:
        # Build a diff that fires multiple rules.
        a = _g([[1, 2], [3, 4]])
        # rotation 90 → also color set differs vs ... no actually rotation
        # preserves colors. Let me build a small-pixel diff (R5) +
        # color repair (R1).
        e = _g([[1, 2], [3, 9]])  # one pixel differs, color {9} introduced
        diff = analyze_diff(a, e)
        from itertools import pairwise

        suggestions = advise_repairs(a, e, diff)
        # All confidences must be non-increasing.
        for prev, nxt in pairwise(suggestions):
            assert prev.confidence >= nxt.confidence

    def test_suggestion_dataclass_is_hashable(self) -> None:
        s = RepairSuggestion(
            kind="color_repair",
            primitive_hint="recolor",
            confidence=0.5,
            detail="x",
        )
        # Frozen dataclass → hashable.
        assert hash(s) == hash(s)
