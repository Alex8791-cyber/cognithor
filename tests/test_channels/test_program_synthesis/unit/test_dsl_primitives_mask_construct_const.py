# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Mask / logic, construction, and color-constant primitive tests (spec §16.2)."""

from __future__ import annotations

import numpy as np
import pytest

from cognithor.channels.program_synthesis.core.exceptions import TypeMismatchError
from cognithor.channels.program_synthesis.dsl.primitives import (
    frame,
    mask_and,
    mask_apply,
    mask_eq,
    mask_ne,
    mask_not,
    mask_or,
    mask_xor,
    overlay,
    stack_horizontal,
    stack_vertical,
)
from cognithor.channels.program_synthesis.dsl.registry import REGISTRY


def _g(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


def _m(rows: list[list[bool]]) -> np.ndarray:
    return np.array(rows, dtype=np.bool_)


# ---------------------------------------------------------------------------
# mask_eq / mask_ne
# ---------------------------------------------------------------------------


class TestMaskEq:
    def test_returns_bool_mask(self) -> None:
        m = mask_eq(_g([[1, 2], [1, 3]]), color=1)
        assert m.dtype == np.bool_
        assert np.array_equal(m, _m([[True, False], [True, False]]))

    def test_no_match_all_false(self) -> None:
        m = mask_eq(_g([[1, 2]]), color=9)
        assert not m.any()

    def test_all_match(self) -> None:
        m = mask_eq(_g([[5, 5]]), color=5)
        assert m.all()

    def test_rejects_non_grid(self) -> None:
        with pytest.raises(TypeMismatchError):
            mask_eq([[1, 2]], color=1)  # type: ignore[arg-type]

    def test_complement_of_mask_ne(self) -> None:
        g = _g([[1, 2, 1]])
        eq = mask_eq(g, 1)
        ne = mask_ne(g, 1)
        assert np.array_equal(eq, np.logical_not(ne))


class TestMaskNe:
    def test_returns_bool_mask(self) -> None:
        m = mask_ne(_g([[1, 2], [1, 3]]), color=1)
        assert np.array_equal(m, _m([[False, True], [False, True]]))

    def test_all_distinct_all_true(self) -> None:
        m = mask_ne(_g([[1, 2]]), color=9)
        assert m.all()

    def test_dtype_preserved(self) -> None:
        assert mask_ne(_g([[5]]), color=5).dtype == np.bool_

    def test_rejects_out_of_range_color(self) -> None:
        with pytest.raises(TypeMismatchError):
            mask_ne(_g([[1]]), color=10)

    def test_returns_copy(self) -> None:
        g = _g([[1]])
        m = mask_ne(g, color=2)
        m[0, 0] = False
        # Source untouched.
        assert g[0, 0] == 1


# ---------------------------------------------------------------------------
# mask_apply
# ---------------------------------------------------------------------------


class TestMaskApply:
    def test_paints_masked_cells(self) -> None:
        out = mask_apply(_g([[1, 2]]), _m([[True, False]]), color=9)
        assert np.array_equal(out, _g([[9, 2]]))

    def test_empty_mask_returns_copy(self) -> None:
        g = _g([[1, 2]])
        out = mask_apply(g, _m([[False, False]]), color=9)
        assert np.array_equal(out, g)
        out[0, 0] = 7
        assert g[0, 0] == 1

    def test_full_mask_overwrites_entire_grid(self) -> None:
        out = mask_apply(_g([[1, 2]]), _m([[True, True]]), color=9)
        assert np.array_equal(out, _g([[9, 9]]))

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(TypeMismatchError, match="shape"):
            mask_apply(_g([[1, 2]]), _m([[True, False, True]]), color=0)

    def test_rejects_non_bool_mask(self) -> None:
        with pytest.raises(TypeMismatchError):
            mask_apply(_g([[1, 2]]), _g([[1, 0]]), color=0)


# ---------------------------------------------------------------------------
# mask_and / or / xor / not
# ---------------------------------------------------------------------------


class TestMaskAnd:
    def test_known(self) -> None:
        out = mask_and(_m([[True, False]]), _m([[True, True]]))
        assert np.array_equal(out, _m([[True, False]]))

    def test_with_self_is_identity(self) -> None:
        a = _m([[True, False, True]])
        assert np.array_equal(mask_and(a, a), a)

    def test_with_all_false_is_all_false(self) -> None:
        a = _m([[True, False]])
        b = _m([[False, False]])
        assert not mask_and(a, b).any()

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(TypeMismatchError, match="shape"):
            mask_and(_m([[True, False]]), _m([[True, False, True]]))

    def test_dtype_bool(self) -> None:
        assert mask_and(_m([[True]]), _m([[True]])).dtype == np.bool_


class TestMaskOr:
    def test_known(self) -> None:
        out = mask_or(_m([[True, False]]), _m([[False, True]]))
        assert np.array_equal(out, _m([[True, True]]))

    def test_with_self_is_identity(self) -> None:
        a = _m([[True, False, True]])
        assert np.array_equal(mask_or(a, a), a)

    def test_or_all_true_is_all_true(self) -> None:
        a = _m([[True, False]])
        b = _m([[True, True]])
        assert mask_or(a, b).all()

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(TypeMismatchError):
            mask_or(_m([[True]]), _m([[True, True]]))

    def test_distributes_over_and(self) -> None:
        # De Morgan: NOT (a AND b) == (NOT a) OR (NOT b)
        a = _m([[True, False, True]])
        b = _m([[True, True, False]])
        lhs = mask_not(mask_and(a, b))
        rhs = mask_or(mask_not(a), mask_not(b))
        assert np.array_equal(lhs, rhs)


class TestMaskXor:
    def test_known(self) -> None:
        out = mask_xor(_m([[True, False]]), _m([[True, True]]))
        assert np.array_equal(out, _m([[False, True]]))

    def test_with_self_is_all_false(self) -> None:
        a = _m([[True, False, True]])
        assert not mask_xor(a, a).any()

    def test_xor_with_all_false_is_identity(self) -> None:
        a = _m([[True, False, True]])
        b = _m([[False, False, False]])
        assert np.array_equal(mask_xor(a, b), a)

    def test_commutative(self) -> None:
        a = _m([[True, False]])
        b = _m([[False, True]])
        assert np.array_equal(mask_xor(a, b), mask_xor(b, a))

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(TypeMismatchError):
            mask_xor(_m([[True]]), _m([[True, True]]))


class TestMaskNot:
    def test_known(self) -> None:
        assert np.array_equal(mask_not(_m([[True, False]])), _m([[False, True]]))

    def test_involution(self) -> None:
        a = _m([[True, False, True]])
        assert np.array_equal(mask_not(mask_not(a)), a)

    def test_dtype_bool(self) -> None:
        assert mask_not(_m([[True]])).dtype == np.bool_

    def test_rejects_non_mask(self) -> None:
        with pytest.raises(TypeMismatchError):
            mask_not(_g([[1, 0]]))

    def test_returns_copy(self) -> None:
        a = _m([[True]])
        out = mask_not(a)
        out[0, 0] = True
        assert a[0, 0]


# ---------------------------------------------------------------------------
# stack_horizontal / vertical
# ---------------------------------------------------------------------------


class TestStackHorizontal:
    def test_known(self) -> None:
        out = stack_horizontal(_g([[1]]), _g([[2]]))
        assert np.array_equal(out, _g([[1, 2]]))

    def test_grows_columns(self) -> None:
        out = stack_horizontal(_g([[1, 2], [3, 4]]), _g([[5], [6]]))
        assert out.shape == (2, 3)

    def test_row_mismatch_raises(self) -> None:
        with pytest.raises(TypeMismatchError, match="row mismatch"):
            stack_horizontal(_g([[1]]), _g([[2], [3]]))

    def test_dtype_preserved(self) -> None:
        assert stack_horizontal(_g([[1]]), _g([[2]])).dtype == np.int8

    def test_rejects_non_grid(self) -> None:
        with pytest.raises(TypeMismatchError):
            stack_horizontal(_g([[1]]), [[2]])  # type: ignore[arg-type]


class TestStackVertical:
    def test_known(self) -> None:
        out = stack_vertical(_g([[1, 2]]), _g([[3, 4]]))
        assert np.array_equal(out, _g([[1, 2], [3, 4]]))

    def test_grows_rows(self) -> None:
        out = stack_vertical(_g([[1, 2]]), _g([[3, 4], [5, 6]]))
        assert out.shape == (3, 2)

    def test_col_mismatch_raises(self) -> None:
        with pytest.raises(TypeMismatchError, match="col mismatch"):
            stack_vertical(_g([[1, 2]]), _g([[3]]))

    def test_compose_h_v_equals_block(self) -> None:
        a = _g([[1]])
        b = _g([[2]])
        c = _g([[3]])
        d = _g([[4]])
        block = stack_vertical(stack_horizontal(a, b), stack_horizontal(c, d))
        assert np.array_equal(block, _g([[1, 2], [3, 4]]))

    def test_rejects_non_grid(self) -> None:
        with pytest.raises(TypeMismatchError):
            stack_vertical([[1]], _g([[2]]))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# overlay
# ---------------------------------------------------------------------------


class TestOverlay:
    def test_replaces_non_transparent_cells(self) -> None:
        out = overlay(_g([[1, 1]]), _g([[0, 2]]), transparent_color=0)
        assert np.array_equal(out, _g([[1, 2]]))

    def test_full_transparent_returns_base(self) -> None:
        base = _g([[1, 2]])
        out = overlay(base, _g([[0, 0]]), transparent_color=0)
        assert np.array_equal(out, base)

    def test_no_transparent_overwrites_all(self) -> None:
        out = overlay(_g([[1, 1]]), _g([[2, 3]]), transparent_color=0)
        assert np.array_equal(out, _g([[2, 3]]))

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(TypeMismatchError, match="shape mismatch"):
            overlay(_g([[1]]), _g([[1, 1]]), transparent_color=0)

    def test_returns_copy(self) -> None:
        base = _g([[1]])
        out = overlay(base, _g([[2]]), transparent_color=0)
        out[0, 0] = 9
        assert base[0, 0] == 1


# ---------------------------------------------------------------------------
# frame
# ---------------------------------------------------------------------------


class TestFrame:
    def test_2x2_fully_framed(self) -> None:
        # Every cell of a 2×2 is on the border.
        out = frame(_g([[1, 2], [3, 4]]), color=0)
        assert np.array_equal(out, _g([[0, 0], [0, 0]]))

    def test_3x3_keeps_centre(self) -> None:
        g = _g([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        out = frame(g, color=0)
        # Border = 0, centre = 5
        assert out[1, 1] == 5
        assert out[0, 0] == 0
        assert out[-1, -1] == 0

    def test_1x1_becomes_color(self) -> None:
        out = frame(_g([[7]]), color=3)
        assert out[0, 0] == 3

    def test_returns_copy(self) -> None:
        g = _g([[1, 2], [3, 4]])
        out = frame(g, color=0)
        out[0, 0] = 9
        assert g[0, 0] == 1

    def test_rejects_out_of_range_color(self) -> None:
        with pytest.raises(TypeMismatchError):
            frame(_g([[1]]), color=-1)


# ---------------------------------------------------------------------------
# Color constants
# ---------------------------------------------------------------------------


class TestColorConstants:
    def test_all_10_constants_registered(self) -> None:
        for c in range(10):
            spec = REGISTRY.get(f"const_color_{c}")
            assert spec.signature.arity == 0
            assert spec.signature.output == "Color"

    def test_const_returns_correct_int(self) -> None:
        for c in range(10):
            spec = REGISTRY.get(f"const_color_{c}")
            assert spec.fn() == c

    def test_const_cost_is_cheap(self) -> None:
        for c in range(10):
            spec = REGISTRY.get(f"const_color_{c}")
            assert spec.cost == 0.5

    def test_const_zero_separate_from_const_nine(self) -> None:
        zero = REGISTRY.get("const_color_0").fn()
        nine = REGISTRY.get("const_color_9").fn()
        assert zero == 0
        assert nine == 9

    def test_const_returns_python_int(self) -> None:
        assert isinstance(REGISTRY.get("const_color_5").fn(), int)


# ---------------------------------------------------------------------------
# Registry side-effect.
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_all_21_primitives_registered(self) -> None:
        names = REGISTRY.names()
        for expected in (
            "mask_eq",
            "mask_ne",
            "mask_apply",
            "mask_and",
            "mask_or",
            "mask_xor",
            "mask_not",
            "stack_horizontal",
            "stack_vertical",
            "overlay",
            "frame",
            *(f"const_color_{c}" for c in range(10)),
        ):
            assert expected in names, f"{expected} not registered"

    def test_base_primitives_present(self) -> None:
        # The Phase-1 base catalog has 56 primitives. Higher-order
        # primitives (Phase 1.5) extend the registry further; this
        # test only locks down the floor.
        assert len(REGISTRY) >= 56
