# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""align_to (H3) + AlignMode tests (spec §7.5)."""

from __future__ import annotations

import pytest

from cognithor.channels.program_synthesis.core.exceptions import TypeMismatchError
from cognithor.channels.program_synthesis.dsl.primitives import (
    AlignMode,
    align_to,
)
from cognithor.channels.program_synthesis.dsl.registry import REGISTRY
from cognithor.channels.program_synthesis.dsl.types_grid import Object

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _square(top_left: tuple[int, int], side: int = 2, color: int = 1) -> Object:
    r0, c0 = top_left
    cells = tuple((r0 + dr, c0 + dc) for dr in range(side) for dc in range(side))
    return Object(color=color, cells=cells)


# ---------------------------------------------------------------------------
# AlignMode enum
# ---------------------------------------------------------------------------


class TestAlignModeEnum:
    def test_nine_modes_present(self) -> None:
        assert len(list(AlignMode)) == 9

    def test_exact_value_strings(self) -> None:
        assert AlignMode.CENTER.value == "center"
        assert AlignMode.LEFT.value == "left"
        assert AlignMode.RIGHT.value == "right"
        assert AlignMode.TOP.value == "top"
        assert AlignMode.BOTTOM.value == "bottom"
        assert AlignMode.TOP_LEFT.value == "top_left"
        assert AlignMode.TOP_RIGHT.value == "top_right"
        assert AlignMode.BOTTOM_LEFT.value == "bottom_left"
        assert AlignMode.BOTTOM_RIGHT.value == "bottom_right"

    def test_string_subclass_lets_value_pass_to_string_apis(self) -> None:
        assert isinstance(AlignMode.CENTER, str)
        assert AlignMode.LEFT == "left"


# ---------------------------------------------------------------------------
# align_to — single-axis modes
# ---------------------------------------------------------------------------


class TestAlignTo:
    # Reference object B is a 4×4 square at (10..13, 10..13).
    B = _square((10, 10), side=4, color=2)

    def test_center_aligns_both_axes(self) -> None:
        a = _square((0, 0), side=2, color=1)
        # B center ((10+13)//2, (10+13)//2) == (11, 11).
        # A center ((0+1)//2, (0+1)//2) == (0, 0).
        # Delta = (11, 11).
        result = align_to(a, self.B, AlignMode.CENTER)
        assert result.color == a.color
        # New cells: original (0..1)×(0..1) + (11, 11).
        assert result.cells == ((11, 11), (11, 12), (12, 11), (12, 12))

    def test_left_aligns_left_edge_centers_y(self) -> None:
        a = _square((0, 0), side=2, color=1)
        # Left edge of B is c0=10. dx = 10 - 0 = 10.
        # dy = 11 - 0 = 11.
        result = align_to(a, self.B, AlignMode.LEFT)
        # Cells start at (11, 10).
        assert (11, 10) in result.cells

    def test_right_aligns_right_edge(self) -> None:
        a = _square((0, 0), side=2, color=1)
        # Right edge of B is c1-1 == 13. A's right edge is c1-1 == 1.
        # dx = 13 - 1 = 12.
        # dy = centred = 11.
        result = align_to(a, self.B, AlignMode.RIGHT)
        # Cells end at column 13.
        max_c = max(c for _, c in result.cells)
        assert max_c == 13

    def test_top_aligns_top_edge_centers_x(self) -> None:
        a = _square((0, 0), side=2, color=1)
        # Top of B is r0=10. dy = 10 - 0 = 10.
        # dx centred = 11 - 0 = 11.
        result = align_to(a, self.B, AlignMode.TOP)
        # Top row of A's bbox is now 10.
        min_r = min(r for r, _ in result.cells)
        assert min_r == 10

    def test_bottom_aligns_bottom_edge(self) -> None:
        a = _square((0, 0), side=2, color=1)
        result = align_to(a, self.B, AlignMode.BOTTOM)
        max_r = max(r for r, _ in result.cells)
        assert max_r == 13

    def test_top_left_corner(self) -> None:
        a = _square((0, 0), side=2, color=1)
        result = align_to(a, self.B, AlignMode.TOP_LEFT)
        # A's top-left should now match B's top-left == (10, 10).
        min_r = min(r for r, _ in result.cells)
        min_c = min(c for _, c in result.cells)
        assert (min_r, min_c) == (10, 10)

    def test_top_right_corner(self) -> None:
        a = _square((0, 0), side=2, color=1)
        result = align_to(a, self.B, AlignMode.TOP_RIGHT)
        # A's top-right should match B's top-right == (10, 13).
        min_r = min(r for r, _ in result.cells)
        max_c = max(c for _, c in result.cells)
        assert (min_r, max_c) == (10, 13)

    def test_bottom_left_corner(self) -> None:
        a = _square((0, 0), side=2, color=1)
        result = align_to(a, self.B, AlignMode.BOTTOM_LEFT)
        max_r = max(r for r, _ in result.cells)
        min_c = min(c for _, c in result.cells)
        assert (max_r, min_c) == (13, 10)

    def test_bottom_right_corner(self) -> None:
        a = _square((0, 0), side=2, color=1)
        result = align_to(a, self.B, AlignMode.BOTTOM_RIGHT)
        max_r = max(r for r, _ in result.cells)
        max_c = max(c for _, c in result.cells)
        assert (max_r, max_c) == (13, 13)


# ---------------------------------------------------------------------------
# Edge cases + type validation
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_already_aligned_returns_input(self) -> None:
        a = _square((10, 10), side=2, color=1)
        b = _square((10, 10), side=2, color=2)
        # CENTER between identical bboxes → no movement.
        result = align_to(a, b, AlignMode.CENTER)
        # Returns the same Object instance when no shift is needed.
        assert result is a

    def test_empty_a_returns_a(self) -> None:
        a = Object(color=1, cells=())
        b = _square((5, 5), side=2, color=2)
        assert align_to(a, b, AlignMode.CENTER) is a

    def test_empty_b_returns_a(self) -> None:
        a = _square((0, 0), side=2, color=1)
        b = Object(color=2, cells=())
        assert align_to(a, b, AlignMode.CENTER) is a

    def test_string_mode_accepted_and_normalised(self) -> None:
        a = _square((0, 0), side=2, color=1)
        b = _square((10, 10), side=4, color=2)
        # AlignMode is StrEnum; passing the raw string should resolve.
        as_string = align_to(a, b, "center")  # type: ignore[arg-type]
        as_enum = align_to(a, b, AlignMode.CENTER)
        assert as_string.cells == as_enum.cells

    def test_unknown_string_mode_rejected(self) -> None:
        a = _square((0, 0), side=2, color=1)
        b = _square((10, 10), side=4, color=2)
        with pytest.raises(TypeMismatchError, match="unknown AlignMode"):
            align_to(a, b, "diagonal")  # type: ignore[arg-type]

    def test_non_object_rejected(self) -> None:
        b = _square((10, 10), side=4, color=2)
        with pytest.raises(TypeMismatchError):
            align_to("nope", b, AlignMode.CENTER)  # type: ignore[arg-type]

    def test_non_align_mode_rejected(self) -> None:
        a = _square((0, 0), side=2, color=1)
        b = _square((10, 10), side=4, color=2)
        with pytest.raises(TypeMismatchError):
            align_to(a, b, 42)  # type: ignore[arg-type]

    def test_color_preserved(self) -> None:
        a = _square((0, 0), side=2, color=7)
        b = _square((10, 10), side=4, color=2)
        assert align_to(a, b, AlignMode.CENTER).color == 7


# ---------------------------------------------------------------------------
# Algebraic identities
# ---------------------------------------------------------------------------


class TestAlgebraicIdentities:
    def test_aligning_to_self_after_shift_inverts(self) -> None:
        # Shift A, then align it back to its origin → original cells.
        b = _square((0, 0), side=2, color=2)
        a_shifted = Object(color=1, cells=((100, 100), (100, 101), (101, 100), (101, 101)))
        # CENTER align of a_shifted to b returns it to (0,0)..(1,1).
        result = align_to(a_shifted, b, AlignMode.CENTER)
        assert result.cells == ((0, 0), (0, 1), (1, 0), (1, 1))

    def test_top_left_of_self_equal_to_self(self) -> None:
        a = _square((5, 5), side=2, color=1)
        result = align_to(a, a, AlignMode.TOP_LEFT)
        assert result.cells == a.cells


# ---------------------------------------------------------------------------
# Registry side-effect.
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_align_to_registered(self) -> None:
        assert "align_to" in REGISTRY.names()

    def test_signature(self) -> None:
        spec = REGISTRY.get("align_to")
        assert spec.signature.inputs == ("Object", "Object", "AlignMode")
        assert spec.signature.output == "Object"
