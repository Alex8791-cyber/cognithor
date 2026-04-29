# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Object-detection primitive tests (spec §16.2)."""

from __future__ import annotations

import numpy as np
import pytest

from cognithor.channels.program_synthesis.core.exceptions import TypeMismatchError
from cognithor.channels.program_synthesis.dsl.primitives import (
    bounding_box,
    connected_components_4,
    connected_components_8,
    largest_object,
    object_count,
    objects_of_color,
    render_objects,
    smallest_object,
)
from cognithor.channels.program_synthesis.dsl.registry import REGISTRY
from cognithor.channels.program_synthesis.dsl.types_grid import Object, ObjectSet


def _g(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


# ---------------------------------------------------------------------------
# connected_components_4
# ---------------------------------------------------------------------------


class TestConnectedComponents4:
    def test_finds_two_separate_components(self) -> None:
        # bg = 0 (4 zeros, 4 ones — tie broken by lowest index → 0).
        # Two color-1 components, each 2 cells, separated by zeros.
        result = connected_components_4(_g([[1, 0, 0, 1], [1, 0, 0, 1]]))
        assert len(result) == 2
        for o in result:
            assert o.color == 1
            assert o.size == 2

    def test_empty_when_uniform(self) -> None:
        # Uniform grid → bg matches everywhere → no objects.
        result = connected_components_4(_g([[3, 3], [3, 3]]))
        assert result.is_empty()

    def test_diagonal_pixels_not_connected(self) -> None:
        # Two pixels diagonal of each other → 4-connectivity = 2 separate objects.
        result = connected_components_4(_g([[1, 0], [0, 1]]))
        assert len(result) == 2

    def test_dtype_independent_of_input_shape(self) -> None:
        result = connected_components_4(_g([[5]]))
        assert isinstance(result, ObjectSet)

    def test_rejects_non_grid(self) -> None:
        with pytest.raises(TypeMismatchError):
            connected_components_4([[1, 0]])  # type: ignore[arg-type]


class TestConnectedComponents8:
    def test_diagonal_pixels_form_one_object(self) -> None:
        # 8-connectivity: diagonal neighbours connect → 1 object.
        result = connected_components_8(_g([[1, 0], [0, 1]]))
        assert len(result) == 1
        assert result[0].size == 2

    def test_matches_4_when_no_diagonals(self) -> None:
        # bg = 0 (4 zeros vs 4 ones, lowest-index tie wins).
        # Same shape used in TestConnectedComponents4.
        g = _g([[1, 0, 0, 1], [1, 0, 0, 1]])
        cc4 = connected_components_4(g)
        cc8 = connected_components_8(g)
        assert len(cc4) == len(cc8)

    def test_returns_object_set(self) -> None:
        assert isinstance(connected_components_8(_g([[1]])), ObjectSet)

    def test_empty_when_uniform(self) -> None:
        assert connected_components_8(_g([[5, 5]])).is_empty()

    def test_rejects_wrong_dtype(self) -> None:
        with pytest.raises(TypeMismatchError):
            connected_components_8(np.array([[1, 0]], dtype=np.int32))


# ---------------------------------------------------------------------------
# objects_of_color
# ---------------------------------------------------------------------------


class TestObjectsOfColor:
    def test_filters_by_color(self) -> None:
        # color=1 has one 2-cell object; color=2 has one 1-cell object.
        result = objects_of_color(_g([[1, 0, 2], [1, 0, 0]]), color=1)
        assert len(result) == 1
        assert result[0].color == 1
        assert result[0].size == 2

    def test_color_absent_returns_empty(self) -> None:
        assert objects_of_color(_g([[1, 1]]), color=9).is_empty()

    def test_treats_color_as_foreground_regardless_of_bg(self) -> None:
        # Even if color matches the background, return the components.
        result = objects_of_color(_g([[0, 0], [0, 0]]), color=0)
        assert len(result) == 1  # one big 4-cell component
        assert result[0].size == 4

    def test_rejects_out_of_range_color(self) -> None:
        with pytest.raises(TypeMismatchError):
            objects_of_color(_g([[1]]), color=10)

    def test_rejects_non_grid(self) -> None:
        with pytest.raises(TypeMismatchError):
            objects_of_color([[1]], color=1)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# largest_object / smallest_object
# ---------------------------------------------------------------------------


class TestLargestObject:
    def test_picks_largest_by_size(self) -> None:
        small = Object(color=1, cells=((0, 0),))
        big = Object(color=2, cells=((0, 0), (0, 1), (1, 0)))
        assert largest_object(ObjectSet((small, big))) is big

    def test_ties_break_by_discovery_order(self) -> None:
        a = Object(color=1, cells=((0, 0), (0, 1)))
        b = Object(color=2, cells=((1, 0), (1, 1)))
        # Both size=2; first wins.
        assert largest_object(ObjectSet((a, b))) is a

    def test_empty_raises(self) -> None:
        with pytest.raises(TypeMismatchError, match="empty"):
            largest_object(ObjectSet())

    def test_single_object_returns_it(self) -> None:
        only = Object(color=1, cells=((0, 0),))
        assert largest_object(ObjectSet((only,))) is only

    def test_rejects_non_object_set(self) -> None:
        with pytest.raises(TypeMismatchError):
            largest_object([1, 2])  # type: ignore[arg-type]


class TestSmallestObject:
    def test_picks_smallest_by_size(self) -> None:
        small = Object(color=1, cells=((0, 0),))
        big = Object(color=2, cells=((0, 0), (0, 1), (1, 0)))
        assert smallest_object(ObjectSet((big, small))) is small

    def test_ties_break_by_discovery_order(self) -> None:
        a = Object(color=1, cells=((0, 0),))
        b = Object(color=2, cells=((1, 0),))
        assert smallest_object(ObjectSet((a, b))) is a

    def test_empty_raises(self) -> None:
        with pytest.raises(TypeMismatchError, match="empty"):
            smallest_object(ObjectSet())

    def test_single_object_returns_it(self) -> None:
        only = Object(color=4, cells=((2, 2),))
        assert smallest_object(ObjectSet((only,))) is only

    def test_rejects_non_object_set(self) -> None:
        with pytest.raises(TypeMismatchError):
            smallest_object("nope")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# bounding_box
# ---------------------------------------------------------------------------


class TestBoundingBox:
    def test_renders_object_in_tight_grid(self) -> None:
        o = Object(color=5, cells=((0, 0), (0, 1), (1, 1)))
        out = bounding_box(o)
        assert np.array_equal(out, _g([[5, 5], [0, 5]]))

    def test_single_cell(self) -> None:
        o = Object(color=7, cells=((3, 4),))
        out = bounding_box(o)
        assert out.shape == (1, 1)
        assert out[0, 0] == 7

    def test_empty_object_returns_1x1_zero(self) -> None:
        out = bounding_box(Object(color=2, cells=()))
        assert out.shape == (1, 1)
        assert out[0, 0] == 0

    def test_dtype_is_int8(self) -> None:
        o = Object(color=3, cells=((0, 0),))
        assert bounding_box(o).dtype == np.int8

    def test_rejects_non_object(self) -> None:
        with pytest.raises(TypeMismatchError):
            bounding_box("nope")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# object_count
# ---------------------------------------------------------------------------


class TestObjectCount:
    def test_zero_for_empty(self) -> None:
        assert object_count(ObjectSet()) == 0

    def test_three(self) -> None:
        s = ObjectSet(
            (
                Object(color=1, cells=((0, 0),)),
                Object(color=2, cells=((0, 1),)),
                Object(color=3, cells=((1, 0),)),
            )
        )
        assert object_count(s) == 3

    def test_returns_python_int(self) -> None:
        assert isinstance(object_count(ObjectSet()), int)

    def test_matches_components_size(self) -> None:
        # bg = 0 (lowest-index tie wins on equal counts).
        g = _g([[1, 0, 0, 1], [1, 0, 0, 1]])
        assert object_count(connected_components_4(g)) == 2

    def test_rejects_non_object_set(self) -> None:
        with pytest.raises(TypeMismatchError):
            object_count([1, 2, 3])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# render_objects
# ---------------------------------------------------------------------------


class TestRenderObjects:
    def test_paints_object_onto_base(self) -> None:
        objs = ObjectSet((Object(color=2, cells=((0, 0),)),))
        out = render_objects(objs, _g([[0, 0], [0, 0]]))
        assert np.array_equal(out, _g([[2, 0], [0, 0]]))

    def test_empty_set_returns_base_copy(self) -> None:
        base = _g([[1, 2], [3, 4]])
        out = render_objects(ObjectSet(), base)
        assert np.array_equal(out, base)
        out[0, 0] = 9
        assert base[0, 0] == 1

    def test_later_objects_overwrite_earlier(self) -> None:
        objs = ObjectSet(
            (
                Object(color=1, cells=((0, 0),)),
                Object(color=5, cells=((0, 0),)),
            )
        )
        out = render_objects(objs, _g([[0]]))
        assert out[0, 0] == 5

    def test_out_of_bounds_cells_are_clipped(self) -> None:
        objs = ObjectSet((Object(color=9, cells=((5, 5), (0, 0))),))
        out = render_objects(objs, _g([[0, 0], [0, 0]]))
        # Only (0,0) lands inside the 2×2 base; (5,5) is dropped.
        assert out[0, 0] == 9
        assert int(out.sum()) == 9  # only one cell painted

    def test_rejects_non_grid_base(self) -> None:
        with pytest.raises(TypeMismatchError):
            render_objects(ObjectSet(), [[0]])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Registry side-effect.
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_all_8_primitives_registered(self) -> None:
        names = REGISTRY.names()
        for expected in (
            "connected_components_4",
            "connected_components_8",
            "objects_of_color",
            "largest_object",
            "smallest_object",
            "bounding_box",
            "object_count",
            "render_objects",
        ):
            assert expected in names, f"{expected} not registered"

    def test_signatures(self) -> None:
        spec = REGISTRY.get("largest_object")
        assert spec.signature.inputs == ("ObjectSet",)
        assert spec.signature.output == "Object"

        spec = REGISTRY.get("render_objects")
        assert spec.signature.inputs == ("ObjectSet", "Grid")
        assert spec.signature.output == "Grid"
