# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Size/scale + spatial primitive tests (spec §16.2).

Five tests per primitive, same template as
test_dsl_primitives_geom_color.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from cognithor.channels.program_synthesis.core.exceptions import TypeMismatchError
from cognithor.channels.program_synthesis.dsl.primitives import (
    crop_bbox,
    gravity_down,
    gravity_left,
    gravity_right,
    gravity_up,
    pad_with,
    scale_down_2x,
    scale_up_2x,
    scale_up_3x,
    shift,
    tile_2x,
    wrap_shift,
)
from cognithor.channels.program_synthesis.dsl.registry import REGISTRY


def _g(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


# ---------------------------------------------------------------------------
# scale_up_2x
# ---------------------------------------------------------------------------


class TestScaleUp2x:
    def test_known(self) -> None:
        out = scale_up_2x(_g([[1, 2]]))
        assert np.array_equal(out, _g([[1, 1, 2, 2], [1, 1, 2, 2]]))

    def test_doubles_dimensions(self) -> None:
        g = _g([[1, 2], [3, 4]])
        out = scale_up_2x(g)
        assert out.shape == (4, 4)

    def test_dtype_preserved(self) -> None:
        out = scale_up_2x(_g([[1]]))
        assert out.dtype == np.int8

    def test_returns_copy(self) -> None:
        g = _g([[1]])
        out = scale_up_2x(g)
        out[0, 0] = 9
        assert g[0, 0] == 1

    def test_rejects_non_grid(self) -> None:
        with pytest.raises(TypeMismatchError):
            scale_up_2x([[1]])  # type: ignore[arg-type]


class TestScaleUp3x:
    def test_known(self) -> None:
        out = scale_up_3x(_g([[1]]))
        assert np.array_equal(out, _g([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))

    def test_triples_dimensions(self) -> None:
        out = scale_up_3x(_g([[1, 2]]))
        assert out.shape == (3, 6)

    def test_compose_with_scale_down(self) -> None:
        g = _g([[1, 2], [3, 4]])
        # 3x then down by 2 yields a 3x3 sub-grid (odd → truncated)
        scaled = scale_up_3x(g)
        assert scaled.shape == (6, 6)
        assert np.array_equal(scale_down_2x(scaled), _g([[1, 1, 2], [1, 1, 2], [3, 3, 4]]))

    def test_dtype_preserved(self) -> None:
        assert scale_up_3x(_g([[5]])).dtype == np.int8

    def test_rejects_1d(self) -> None:
        with pytest.raises(TypeMismatchError):
            scale_up_3x(np.array([1, 2], dtype=np.int8))


class TestScaleDown2x:
    def test_inverse_of_scale_up_2x(self) -> None:
        g = _g([[1, 2], [3, 4]])
        assert np.array_equal(scale_down_2x(scale_up_2x(g)), g)

    def test_known(self) -> None:
        out = scale_down_2x(_g([[1, 1, 2, 2], [1, 1, 2, 2]]))
        assert np.array_equal(out, _g([[1, 2]]))

    def test_truncates_odd_dimensions(self) -> None:
        g = _g([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        out = scale_down_2x(g)
        assert out.shape == (2, 2)
        assert np.array_equal(out, _g([[1, 3], [7, 9]]))

    def test_rejects_too_small(self) -> None:
        with pytest.raises(TypeMismatchError, match="too small"):
            scale_down_2x(_g([[1]]))

    def test_rejects_non_grid(self) -> None:
        with pytest.raises(TypeMismatchError):
            scale_down_2x("nope")  # type: ignore[arg-type]


class TestTile2x:
    def test_known(self) -> None:
        out = tile_2x(_g([[1, 2]]))
        assert np.array_equal(out, _g([[1, 2, 1, 2], [1, 2, 1, 2]]))

    def test_doubles_each_dimension(self) -> None:
        g = _g([[1, 2], [3, 4]])
        assert tile_2x(g).shape == (4, 4)

    def test_returns_copy(self) -> None:
        g = _g([[5]])
        out = tile_2x(g)
        out[0, 0] = 9
        assert g[0, 0] == 5

    def test_dtype_preserved(self) -> None:
        assert tile_2x(_g([[1]])).dtype == np.int8

    def test_rejects_non_grid(self) -> None:
        with pytest.raises(TypeMismatchError):
            tile_2x(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# crop_bbox
# ---------------------------------------------------------------------------


class TestCropBbox:
    def test_isolates_single_object(self) -> None:
        out = crop_bbox(_g([[0, 0, 0], [0, 5, 0], [0, 0, 0]]))
        assert np.array_equal(out, _g([[5]]))

    def test_uniform_grid_returns_1x1(self) -> None:
        out = crop_bbox(_g([[3, 3], [3, 3]]))
        assert out.shape == (1, 1)
        assert out[0, 0] == 3

    def test_preserves_shape_when_bbox_full(self) -> None:
        # Background color = 0 (4 cells) tied with 5 (4 cells); argmax picks 0.
        # Non-bg cells span the full grid → bbox is the original.
        g = _g([[5, 0], [0, 5]])
        out = crop_bbox(g)
        assert out.shape == g.shape

    def test_rejects_non_grid(self) -> None:
        with pytest.raises(TypeMismatchError):
            crop_bbox([[1]])  # type: ignore[arg-type]

    def test_dtype_preserved(self) -> None:
        out = crop_bbox(_g([[0, 0, 0], [0, 5, 0]]))
        assert out.dtype == np.int8


# ---------------------------------------------------------------------------
# pad_with
# ---------------------------------------------------------------------------


class TestPadWith:
    def test_known(self) -> None:
        out = pad_with(_g([[1]]), color=0, width=1)
        assert np.array_equal(out, _g([[0, 0, 0], [0, 1, 0], [0, 0, 0]]))

    def test_zero_width_returns_copy(self) -> None:
        g = _g([[1, 2]])
        out = pad_with(g, color=9, width=0)
        assert np.array_equal(out, g)
        out[0, 0] = 9
        assert g[0, 0] == 1

    def test_grows_dimensions_by_2w(self) -> None:
        out = pad_with(_g([[1]]), color=0, width=3)
        assert out.shape == (7, 7)

    def test_rejects_negative_width(self) -> None:
        with pytest.raises(TypeMismatchError, match="< 0"):
            pad_with(_g([[1]]), color=0, width=-1)

    def test_rejects_out_of_range_color(self) -> None:
        with pytest.raises(TypeMismatchError):
            pad_with(_g([[1]]), color=10, width=1)


# ---------------------------------------------------------------------------
# gravity
# ---------------------------------------------------------------------------


class TestGravityDown:
    def test_pulls_columns_to_bottom(self) -> None:
        # Background = 0 (most common, 4 cells)
        out = gravity_down(_g([[1, 0], [0, 2], [0, 0]]))
        assert np.array_equal(out, _g([[0, 0], [0, 0], [1, 2]]))

    def test_idempotent(self) -> None:
        g = _g([[1, 0], [0, 2], [0, 0]])
        once = gravity_down(g)
        twice = gravity_down(once)
        assert np.array_equal(once, twice)

    def test_uniform_grid_unchanged(self) -> None:
        g = _g([[3, 3], [3, 3]])
        assert np.array_equal(gravity_down(g), g)

    def test_dtype_preserved(self) -> None:
        out = gravity_down(_g([[1, 0]]))
        assert out.dtype == np.int8

    def test_rejects_non_grid(self) -> None:
        with pytest.raises(TypeMismatchError):
            gravity_down([[1, 0]])  # type: ignore[arg-type]


class TestGravityUp:
    def test_pulls_columns_to_top(self) -> None:
        out = gravity_up(_g([[0, 0], [1, 0], [0, 2]]))
        assert np.array_equal(out, _g([[1, 2], [0, 0], [0, 0]]))

    def test_inverse_of_gravity_down_on_pre_packed(self) -> None:
        # If non-bg pixels are already at bottom, up reverses to top.
        packed = _g([[0, 0], [0, 0], [1, 2]])
        assert np.array_equal(gravity_up(packed), _g([[1, 2], [0, 0], [0, 0]]))

    def test_idempotent(self) -> None:
        g = _g([[0, 0], [1, 0], [0, 2]])
        once = gravity_up(g)
        twice = gravity_up(once)
        assert np.array_equal(once, twice)

    def test_returns_copy(self) -> None:
        g = _g([[0, 0], [1, 0]])
        out = gravity_up(g)
        out[0, 0] = 9
        # Source untouched (we don't share buffers).
        assert g[0, 0] == 0

    def test_rejects_wrong_dtype(self) -> None:
        with pytest.raises(TypeMismatchError):
            gravity_up(np.array([[0, 1]], dtype=np.int32))


class TestGravityLeft:
    def test_pulls_rows_to_left(self) -> None:
        out = gravity_left(_g([[0, 1, 0, 2]]))
        assert np.array_equal(out, _g([[1, 2, 0, 0]]))

    def test_uniform_unchanged(self) -> None:
        g = _g([[5, 5, 5]])
        assert np.array_equal(gravity_left(g), g)

    def test_idempotent(self) -> None:
        g = _g([[0, 1, 0, 2]])
        assert np.array_equal(gravity_left(gravity_left(g)), gravity_left(g))

    def test_dtype_preserved(self) -> None:
        out = gravity_left(_g([[0, 1, 0]]))
        assert out.dtype == np.int8

    def test_rejects_string(self) -> None:
        with pytest.raises(TypeMismatchError):
            gravity_left("nope")  # type: ignore[arg-type]


class TestGravityRight:
    def test_pulls_rows_to_right(self) -> None:
        out = gravity_right(_g([[1, 0, 2, 0]]))
        assert np.array_equal(out, _g([[0, 0, 1, 2]]))

    def test_idempotent(self) -> None:
        g = _g([[1, 0, 2, 0]])
        assert np.array_equal(gravity_right(gravity_right(g)), gravity_right(g))

    def test_compose_with_mirror_h_equivalent_to_gravity_left(self) -> None:
        from cognithor.channels.program_synthesis.dsl.primitives import (
            mirror_horizontal,
        )

        g = _g([[1, 0, 2, 0]])
        # mirror_h(gravity_right(g)) == gravity_left(mirror_h(g))
        lhs = mirror_horizontal(gravity_right(g))
        rhs = gravity_left(mirror_horizontal(g))
        assert np.array_equal(lhs, rhs)

    def test_returns_copy(self) -> None:
        g = _g([[1, 0]])
        out = gravity_right(g)
        out[0, 0] = 9
        assert g[0, 0] == 1

    def test_rejects_wrong_dtype(self) -> None:
        with pytest.raises(TypeMismatchError):
            gravity_right(np.array([[1, 0]], dtype=np.float32))


# ---------------------------------------------------------------------------
# shift
# ---------------------------------------------------------------------------


class TestShift:
    def test_down_one_row(self) -> None:
        # bg = 0 (most common after shift since input has 0s and unique values)
        g = _g([[1, 2], [3, 4]])
        # bg is whichever color tied at lowest index — for [1,2,3,4] all single
        # → argmax picks 0 (count 0 from bincount).
        # Wait: most_common_color uses argmax over bincount(minlength=10);
        # for unique [1,2,3,4] all have count=1, argmax picks lowest index → 0
        # (count 0 → argmax returns first 0-valued? Actually argmax returns
        # FIRST occurrence of MAX value; max count is 1 across indices 1-4.
        # argmax returns 1.)
        out = shift(g, dy=1, dx=0)
        # Top row should be background (1), bottom row gets the original top row.
        assert out.shape == g.shape
        assert int(out[1, 0]) == 1 and int(out[1, 1]) == 2

    def test_zero_shift_is_copy(self) -> None:
        g = _g([[1, 2], [3, 4]])
        out = shift(g, dy=0, dx=0)
        assert np.array_equal(out, g)
        out[0, 0] = 9
        assert g[0, 0] == 1

    def test_huge_shift_collapses_to_background(self) -> None:
        g = _g([[1, 2], [3, 4]])
        out = shift(g, dy=100, dx=100)
        # All cells should be the background color.
        assert len(np.unique(out)) == 1

    def test_rejects_non_int_dy(self) -> None:
        with pytest.raises(TypeMismatchError):
            shift(_g([[1, 2]]), dy=1.5, dx=0)  # type: ignore[arg-type]

    def test_dtype_preserved(self) -> None:
        out = shift(_g([[1, 2]]), dy=0, dx=1)
        assert out.dtype == np.int8


class TestWrapShift:
    def test_known_down_one(self) -> None:
        out = wrap_shift(_g([[1, 2], [3, 4]]), dy=1, dx=0)
        assert np.array_equal(out, _g([[3, 4], [1, 2]]))

    def test_full_period_returns_input(self) -> None:
        g = _g([[1, 2], [3, 4]])
        # dy = grid height → full wrap → identical
        assert np.array_equal(wrap_shift(g, dy=2, dx=0), g)
        assert np.array_equal(wrap_shift(g, dy=0, dx=2), g)

    def test_inverse(self) -> None:
        g = _g([[1, 2, 3], [4, 5, 6]])
        forward = wrap_shift(g, dy=1, dx=2)
        back = wrap_shift(forward, dy=-1, dx=-2)
        assert np.array_equal(back, g)

    def test_rejects_non_int(self) -> None:
        with pytest.raises(TypeMismatchError):
            wrap_shift(_g([[1, 2]]), dy="1", dx=0)  # type: ignore[arg-type]

    def test_dtype_preserved(self) -> None:
        out = wrap_shift(_g([[1, 2]]), dy=0, dx=1)
        assert out.dtype == np.int8


# ---------------------------------------------------------------------------
# Registry side-effect: every primitive in this group is registered.
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_all_12_primitives_registered(self) -> None:
        names = REGISTRY.names()
        for expected in (
            "scale_up_2x",
            "scale_up_3x",
            "scale_down_2x",
            "tile_2x",
            "crop_bbox",
            "pad_with",
            "gravity_down",
            "gravity_up",
            "gravity_left",
            "gravity_right",
            "shift",
            "wrap_shift",
        ):
            assert expected in names, f"{expected} not registered"

    def test_signature_consistency(self) -> None:
        # pad_with: (Grid, Color, Int) -> Grid
        spec = REGISTRY.get("pad_with")
        assert spec.signature.inputs == ("Grid", "Color", "Int")
        # shift: (Grid, Int, Int) -> Grid
        spec = REGISTRY.get("shift")
        assert spec.signature.inputs == ("Grid", "Int", "Int")
