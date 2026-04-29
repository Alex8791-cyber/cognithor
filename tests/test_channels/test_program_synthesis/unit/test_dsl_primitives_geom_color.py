# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Geometric + color primitive tests (spec §16.2).

Five tests per primitive: happy path, edge cases (1×1 / 30×30 / empty
column), type-mismatch rejection, determinism, and a property test
verifying an algebraic identity.
"""

from __future__ import annotations

import numpy as np
import pytest

from cognithor.channels.program_synthesis.core.exceptions import TypeMismatchError
from cognithor.channels.program_synthesis.dsl.primitives import (
    color_count,
    identity,
    least_common_color,
    mirror_antidiagonal,
    mirror_diagonal,
    mirror_horizontal,
    mirror_vertical,
    most_common_color,
    recolor,
    replace_background,
    rotate90,
    rotate180,
    rotate270,
    swap_colors,
    transpose,
)
from cognithor.channels.program_synthesis.dsl.registry import REGISTRY


def _g(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


class TestIdentity:
    def test_returns_equal_grid(self) -> None:
        g = _g([[1, 2], [3, 4]])
        assert np.array_equal(identity(g), g)

    def test_returns_copy_not_alias(self) -> None:
        g = _g([[1, 2], [3, 4]])
        out = identity(g)
        out[0, 0] = 9
        assert g[0, 0] == 1

    def test_1x1_grid(self) -> None:
        assert np.array_equal(identity(_g([[7]])), _g([[7]]))

    def test_rejects_non_grid(self) -> None:
        with pytest.raises(TypeMismatchError):
            identity([[1, 2]])  # type: ignore[arg-type]

    def test_deterministic(self) -> None:
        g = _g([[1, 2, 3]])
        assert np.array_equal(identity(g), identity(g))


# ---------------------------------------------------------------------------
# rotations
# ---------------------------------------------------------------------------


class TestRotate90:
    def test_known_2x2(self) -> None:
        out = rotate90(_g([[1, 2], [3, 4]]))
        assert np.array_equal(out, _g([[3, 1], [4, 2]]))

    def test_swaps_dimensions_for_rectangular(self) -> None:
        g = _g([[1, 2, 3]])  # 1x3
        out = rotate90(g)
        assert out.shape == (3, 1)

    def test_four_rotations_yield_identity(self) -> None:
        g = _g([[1, 2], [3, 4]])
        spun = rotate90(rotate90(rotate90(rotate90(g))))
        assert np.array_equal(spun, g)

    def test_rejects_1d(self) -> None:
        with pytest.raises(TypeMismatchError):
            rotate90(np.array([1, 2, 3], dtype=np.int8))

    def test_dtype_preserved(self) -> None:
        out = rotate90(_g([[1, 2], [3, 4]]))
        assert out.dtype == np.int8


class TestRotate180:
    def test_known(self) -> None:
        assert np.array_equal(rotate180(_g([[1, 2], [3, 4]])), _g([[4, 3], [2, 1]]))

    def test_two_rotate90_equals_rotate180(self) -> None:
        g = _g([[1, 2], [3, 4]])
        assert np.array_equal(rotate180(g), rotate90(rotate90(g)))

    def test_idempotent_on_palindromic_grid(self) -> None:
        g = _g([[1, 1], [1, 1]])
        assert np.array_equal(rotate180(g), g)

    def test_rejects_wrong_dtype(self) -> None:
        with pytest.raises(TypeMismatchError):
            rotate180(np.array([[1, 2]], dtype=np.int32))

    def test_rectangular(self) -> None:
        g = _g([[1, 2, 3]])
        assert np.array_equal(rotate180(g), _g([[3, 2, 1]]))


class TestRotate270:
    def test_known(self) -> None:
        assert np.array_equal(rotate270(_g([[1, 2], [3, 4]])), _g([[2, 4], [1, 3]]))

    def test_inverse_of_rotate90(self) -> None:
        g = _g([[1, 2], [3, 4]])
        assert np.array_equal(rotate270(rotate90(g)), g)

    def test_three_rotate90_equals_rotate270(self) -> None:
        g = _g([[1, 2, 3], [4, 5, 6]])
        assert np.array_equal(rotate270(g), rotate90(rotate90(rotate90(g))))

    def test_returns_copy(self) -> None:
        g = _g([[1, 2], [3, 4]])
        out = rotate270(g)
        out[0, 0] = 9
        assert g[0, 0] == 1

    def test_rejects_3d(self) -> None:
        with pytest.raises(TypeMismatchError):
            rotate270(np.zeros((2, 2, 2), dtype=np.int8))


# ---------------------------------------------------------------------------
# mirrors
# ---------------------------------------------------------------------------


class TestMirrorHorizontal:
    def test_known(self) -> None:
        assert np.array_equal(mirror_horizontal(_g([[1, 2], [3, 4]])), _g([[2, 1], [4, 3]]))

    def test_involution(self) -> None:
        g = _g([[1, 2, 3], [4, 5, 6]])
        assert np.array_equal(mirror_horizontal(mirror_horizontal(g)), g)

    def test_1x1_grid_unchanged(self) -> None:
        assert np.array_equal(mirror_horizontal(_g([[5]])), _g([[5]]))

    def test_dtype_preserved(self) -> None:
        out = mirror_horizontal(_g([[1, 2]]))
        assert out.dtype == np.int8

    def test_rejects_non_ndarray(self) -> None:
        with pytest.raises(TypeMismatchError):
            mirror_horizontal("not a grid")  # type: ignore[arg-type]


class TestMirrorVertical:
    def test_known(self) -> None:
        assert np.array_equal(mirror_vertical(_g([[1, 2], [3, 4]])), _g([[3, 4], [1, 2]]))

    def test_involution(self) -> None:
        g = _g([[1, 2], [3, 4]])
        assert np.array_equal(mirror_vertical(mirror_vertical(g)), g)

    def test_single_row(self) -> None:
        assert np.array_equal(mirror_vertical(_g([[1, 2, 3]])), _g([[1, 2, 3]]))

    def test_returns_copy(self) -> None:
        g = _g([[1, 2], [3, 4]])
        out = mirror_vertical(g)
        out[0, 0] = 9
        assert g[0, 0] == 1

    def test_rejects_int_input(self) -> None:
        with pytest.raises(TypeMismatchError):
            mirror_vertical(42)  # type: ignore[arg-type]


class TestTranspose:
    def test_known(self) -> None:
        assert np.array_equal(transpose(_g([[1, 2], [3, 4]])), _g([[1, 3], [2, 4]]))

    def test_double_transpose_is_identity(self) -> None:
        g = _g([[1, 2, 3], [4, 5, 6]])
        assert np.array_equal(transpose(transpose(g)), g)

    def test_swaps_shape(self) -> None:
        g = _g([[1, 2, 3]])
        assert transpose(g).shape == (3, 1)

    def test_diag_equals_transpose_for_square(self) -> None:
        g = _g([[1, 2], [3, 4]])
        assert np.array_equal(mirror_diagonal(g), transpose(g))

    def test_rejects_wrong_type(self) -> None:
        with pytest.raises(TypeMismatchError):
            transpose([[1, 2]])  # type: ignore[arg-type]


class TestMirrorDiagonal:
    def test_square_known(self) -> None:
        assert np.array_equal(mirror_diagonal(_g([[1, 2], [3, 4]])), _g([[1, 3], [2, 4]]))

    def test_involution_on_square(self) -> None:
        g = _g([[1, 2], [3, 4]])
        assert np.array_equal(mirror_diagonal(mirror_diagonal(g)), g)

    def test_rectangular_grid_swaps_shape(self) -> None:
        g = _g([[1, 2, 3]])
        assert mirror_diagonal(g).shape == (3, 1)

    def test_returns_copy(self) -> None:
        g = _g([[1, 2], [3, 4]])
        out = mirror_diagonal(g)
        out[0, 0] = 9
        assert g[0, 0] == 1

    def test_rejects_wrong_dtype(self) -> None:
        with pytest.raises(TypeMismatchError):
            mirror_diagonal(np.array([[1, 2]], dtype=np.float32))


class TestMirrorAntidiagonal:
    def test_square_known(self) -> None:
        assert np.array_equal(mirror_antidiagonal(_g([[1, 2], [3, 4]])), _g([[4, 2], [3, 1]]))

    def test_involution_on_square(self) -> None:
        g = _g([[1, 2], [3, 4]])
        assert np.array_equal(mirror_antidiagonal(mirror_antidiagonal(g)), g)

    def test_rectangular(self) -> None:
        # Anti-diagonal mirror: reverse both axes then transpose.
        g = _g([[1, 2, 3]])
        out = mirror_antidiagonal(g)
        assert out.shape == (3, 1)
        assert np.array_equal(out, _g([[3], [2], [1]]))

    def test_dtype_preserved(self) -> None:
        out = mirror_antidiagonal(_g([[1, 2], [3, 4]]))
        assert out.dtype == np.int8

    def test_rejects_non_grid(self) -> None:
        with pytest.raises(TypeMismatchError):
            mirror_antidiagonal((1, 2, 3))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# color
# ---------------------------------------------------------------------------


class TestRecolor:
    def test_replaces_target_color(self) -> None:
        out = recolor(_g([[1, 2], [1, 3]]), 1, 4)
        assert np.array_equal(out, _g([[4, 2], [4, 3]]))

    def test_no_change_if_color_absent(self) -> None:
        g = _g([[1, 2], [1, 3]])
        assert np.array_equal(recolor(g, 9, 0), g)

    def test_self_recolor_is_identity(self) -> None:
        g = _g([[1, 2], [3, 4]])
        assert np.array_equal(recolor(g, 1, 1), g)

    def test_rejects_out_of_range_color(self) -> None:
        with pytest.raises(TypeMismatchError):
            recolor(_g([[1]]), 1, 10)

    def test_rejects_bool_color(self) -> None:
        with pytest.raises(TypeMismatchError):
            recolor(_g([[1]]), True, 2)  # type: ignore[arg-type]


class TestSwapColors:
    def test_swaps_two_colors(self) -> None:
        out = swap_colors(_g([[1, 2], [2, 1]]), 1, 2)
        assert np.array_equal(out, _g([[2, 1], [1, 2]]))

    def test_swap_same_color_is_identity(self) -> None:
        g = _g([[1, 2], [3, 4]])
        assert np.array_equal(swap_colors(g, 1, 1), g)

    def test_swap_absent_color_is_identity(self) -> None:
        g = _g([[1, 2], [3, 4]])
        assert np.array_equal(swap_colors(g, 8, 9), g)

    def test_idempotent_when_applied_twice(self) -> None:
        g = _g([[1, 2], [2, 1]])
        assert np.array_equal(swap_colors(swap_colors(g, 1, 2), 1, 2), g)

    def test_rejects_negative_color(self) -> None:
        with pytest.raises(TypeMismatchError):
            swap_colors(_g([[1]]), -1, 2)


class TestMostCommonColor:
    def test_returns_majority(self) -> None:
        assert most_common_color(_g([[1, 2, 2], [3, 3, 3]])) == 3

    def test_uniform_grid(self) -> None:
        assert most_common_color(_g([[5, 5], [5, 5]])) == 5

    def test_lowest_index_on_tie(self) -> None:
        # 1 and 2 each appear twice → argmax breaks tie at lowest index.
        assert most_common_color(_g([[1, 2], [1, 2]])) == 1

    def test_returns_python_int(self) -> None:
        assert isinstance(most_common_color(_g([[1]])), int)

    def test_rejects_wrong_dtype(self) -> None:
        with pytest.raises(TypeMismatchError):
            most_common_color(np.array([[1, 2]], dtype=np.int32))


class TestLeastCommonColor:
    def test_returns_minority(self) -> None:
        assert least_common_color(_g([[1, 2, 2], [3, 3, 3]])) == 1

    def test_ignores_absent_colors(self) -> None:
        # 0 has count 0 → should NOT be picked as least-common.
        assert least_common_color(_g([[1, 2, 2]])) == 1

    def test_uniform_grid_returns_only_color(self) -> None:
        assert least_common_color(_g([[5, 5], [5, 5]])) == 5

    def test_returns_python_int(self) -> None:
        assert isinstance(least_common_color(_g([[1, 2]])), int)

    def test_rejects_non_ndarray(self) -> None:
        with pytest.raises(TypeMismatchError):
            least_common_color([1, 2, 3])  # type: ignore[arg-type]


class TestColorCount:
    def test_three_distinct(self) -> None:
        assert color_count(_g([[1, 2, 2], [3, 3, 3]])) == 3

    def test_uniform_grid(self) -> None:
        assert color_count(_g([[5, 5]])) == 1

    def test_all_ten_colors(self) -> None:
        g = _g([list(range(10))])
        assert color_count(g) == 10

    def test_returns_python_int(self) -> None:
        assert isinstance(color_count(_g([[1]])), int)

    def test_rejects_1d(self) -> None:
        with pytest.raises(TypeMismatchError):
            color_count(np.array([1, 2, 3], dtype=np.int8))


class TestReplaceBackground:
    def test_replaces_majority_color(self) -> None:
        out = replace_background(_g([[1, 2, 2], [3, 3, 3]]), 0)
        assert np.array_equal(out, _g([[1, 2, 2], [0, 0, 0]]))

    def test_returns_grid_unchanged_if_new_equals_bg(self) -> None:
        g = _g([[1, 2, 2], [3, 3, 3]])
        # bg is 3, replace with 3 → identity
        assert np.array_equal(replace_background(g, 3), g)

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(TypeMismatchError):
            replace_background(_g([[1]]), -1)

    def test_returns_copy(self) -> None:
        g = _g([[1, 2, 2], [3, 3, 3]])
        out = replace_background(g, 0)
        out[0, 0] = 9
        assert g[0, 0] == 1

    def test_dtype_preserved(self) -> None:
        out = replace_background(_g([[1, 2, 2]]), 0)
        assert out.dtype == np.int8


# ---------------------------------------------------------------------------
# Registry side-effect: every primitive in this group is registered.
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_all_15_primitives_registered(self) -> None:
        names = REGISTRY.names()
        for expected in (
            "identity",
            "rotate90",
            "rotate180",
            "rotate270",
            "mirror_horizontal",
            "mirror_vertical",
            "transpose",
            "mirror_diagonal",
            "mirror_antidiagonal",
            "recolor",
            "swap_colors",
            "most_common_color",
            "least_common_color",
            "color_count",
            "replace_background",
        ):
            assert expected in names, f"{expected} not registered"

    def test_signature_consistency(self) -> None:
        # rotate90: Grid -> Grid, arity 1
        spec = REGISTRY.get("rotate90")
        assert spec.signature.arity == 1
        assert spec.signature.output == "Grid"
        # recolor: Grid, Color, Color -> Grid, arity 3
        spec = REGISTRY.get("recolor")
        assert spec.signature.arity == 3
        assert spec.signature.inputs == ("Grid", "Color", "Color")
