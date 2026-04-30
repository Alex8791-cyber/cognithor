# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""sort_objects (H4) + SortKey tests (spec §7.5 v1.2)."""

from __future__ import annotations

import pytest

from cognithor.channels.program_synthesis.core.exceptions import TypeMismatchError
from cognithor.channels.program_synthesis.dsl.primitives import (
    SortKey,
    sort_objects,
)
from cognithor.channels.program_synthesis.dsl.registry import REGISTRY
from cognithor.channels.program_synthesis.dsl.types_grid import Object, ObjectSet


def _obj(*, color: int, cells: tuple[tuple[int, int], ...]) -> Object:
    return Object(color=color, cells=cells)


def _three_by_size() -> ObjectSet:
    """Three objects with sizes 3, 1, 2 (deliberately out of order)."""
    return ObjectSet(
        objects=(
            _obj(color=1, cells=((0, 0), (0, 1), (0, 2))),  # size 3
            _obj(color=2, cells=((1, 0),)),  # size 1
            _obj(color=3, cells=((2, 0), (2, 1))),  # size 2
        )
    )


# ---------------------------------------------------------------------------
# SortKey enum
# ---------------------------------------------------------------------------


class TestSortKeyEnum:
    def test_seven_keys_present(self) -> None:
        assert len(list(SortKey)) == 7

    def test_exact_value_strings(self) -> None:
        assert SortKey.SIZE_ASC.value == "size_asc"
        assert SortKey.SIZE_DESC.value == "size_desc"
        assert SortKey.COLOR_ASC.value == "color_asc"
        assert SortKey.COLOR_DESC.value == "color_desc"
        assert SortKey.POSITION_ROW.value == "position_row"
        assert SortKey.POSITION_COL.value == "position_col"
        assert SortKey.DISTANCE_FROM_CENTER.value == "distance_from_center"

    def test_str_subclass(self) -> None:
        assert isinstance(SortKey.SIZE_ASC, str)
        assert SortKey.SIZE_ASC == "size_asc"


# ---------------------------------------------------------------------------
# Sort-by-size
# ---------------------------------------------------------------------------


class TestSortBySize:
    def test_size_asc(self) -> None:
        s = _three_by_size()  # sizes 3, 1, 2
        result = sort_objects(s, SortKey.SIZE_ASC)
        assert tuple(o.size for o in result.objects) == (1, 2, 3)

    def test_size_desc(self) -> None:
        s = _three_by_size()
        result = sort_objects(s, SortKey.SIZE_DESC)
        assert tuple(o.size for o in result.objects) == (3, 2, 1)

    def test_size_ties_break_by_discovery_order(self) -> None:
        a = _obj(color=1, cells=((0, 0),))
        b = _obj(color=2, cells=((1, 0),))
        c = _obj(color=3, cells=((2, 0),))
        s = ObjectSet(objects=(a, b, c))
        # All size 1 → SIZE_ASC must preserve discovery order.
        result = sort_objects(s, SortKey.SIZE_ASC)
        assert tuple(o.color for o in result.objects) == (1, 2, 3)


# ---------------------------------------------------------------------------
# Sort-by-color
# ---------------------------------------------------------------------------


class TestSortByColor:
    def test_color_asc(self) -> None:
        s = ObjectSet(
            objects=(
                _obj(color=5, cells=((0, 0),)),
                _obj(color=2, cells=((0, 1),)),
                _obj(color=8, cells=((0, 2),)),
            )
        )
        result = sort_objects(s, SortKey.COLOR_ASC)
        assert tuple(o.color for o in result.objects) == (2, 5, 8)

    def test_color_desc(self) -> None:
        s = ObjectSet(
            objects=(
                _obj(color=5, cells=((0, 0),)),
                _obj(color=2, cells=((0, 1),)),
                _obj(color=8, cells=((0, 2),)),
            )
        )
        result = sort_objects(s, SortKey.COLOR_DESC)
        assert tuple(o.color for o in result.objects) == (8, 5, 2)


# ---------------------------------------------------------------------------
# Sort-by-position
# ---------------------------------------------------------------------------


class TestSortByPosition:
    def test_position_row_top_to_bottom_then_left_to_right(self) -> None:
        # Three single-cell objects at distinct positions.
        a = _obj(color=1, cells=((5, 0),))
        b = _obj(color=2, cells=((0, 5),))
        c = _obj(color=3, cells=((0, 0),))
        s = ObjectSet(objects=(a, b, c))
        result = sort_objects(s, SortKey.POSITION_ROW)
        # POSITION_ROW: by (row, col) of top-left.
        # c=(0,0), b=(0,5), a=(5,0) → c, b, a.
        assert tuple(o.color for o in result.objects) == (3, 2, 1)

    def test_position_col_left_to_right_then_top_to_bottom(self) -> None:
        a = _obj(color=1, cells=((5, 0),))
        b = _obj(color=2, cells=((0, 5),))
        c = _obj(color=3, cells=((0, 0),))
        s = ObjectSet(objects=(a, b, c))
        result = sort_objects(s, SortKey.POSITION_COL)
        # POSITION_COL: by (col, row).
        # c=(0,0)→key=(0,0); a=(5,0)→key=(0,5); b=(0,5)→key=(5,0)
        # → c, a, b.
        assert tuple(o.color for o in result.objects) == (3, 1, 2)


# ---------------------------------------------------------------------------
# Sort-by-distance
# ---------------------------------------------------------------------------


class TestSortByDistance:
    def test_distance_from_center(self) -> None:
        # Three objects at increasing distance from origin.
        near = _obj(color=1, cells=((0, 0),))
        mid = _obj(color=2, cells=((3, 4),))  # distance² = 9 + 16 = 25
        far = _obj(color=3, cells=((10, 0),))  # distance² = 100
        # Insert in descending-distance order.
        s = ObjectSet(objects=(far, mid, near))
        result = sort_objects(s, SortKey.DISTANCE_FROM_CENTER)
        assert tuple(o.color for o in result.objects) == (1, 2, 3)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_set_unchanged(self) -> None:
        empty = ObjectSet()
        result = sort_objects(empty, SortKey.SIZE_ASC)
        assert result is empty

    def test_single_element_unchanged(self) -> None:
        single = ObjectSet(objects=(_obj(color=5, cells=((0, 0),)),))
        result = sort_objects(single, SortKey.SIZE_ASC)
        assert result is single

    def test_already_sorted_returns_same_order(self) -> None:
        a = _obj(color=1, cells=((0, 0),))
        b = _obj(color=2, cells=((0, 0), (0, 1)))
        c = _obj(color=3, cells=((0, 0), (0, 1), (0, 2)))
        s = ObjectSet(objects=(a, b, c))
        result = sort_objects(s, SortKey.SIZE_ASC)
        assert tuple(o.color for o in result.objects) == (1, 2, 3)

    def test_string_key_accepted(self) -> None:
        s = _three_by_size()
        as_string = sort_objects(s, "size_asc")  # type: ignore[arg-type]
        as_enum = sort_objects(s, SortKey.SIZE_ASC)
        assert tuple(o.size for o in as_string.objects) == tuple(o.size for o in as_enum.objects)

    def test_unknown_string_key_rejected(self) -> None:
        s = _three_by_size()
        with pytest.raises(TypeMismatchError, match="unknown SortKey"):
            sort_objects(s, "alphabetic")  # type: ignore[arg-type]

    def test_non_object_set_rejected(self) -> None:
        with pytest.raises(TypeMismatchError):
            sort_objects([1, 2, 3], SortKey.SIZE_ASC)  # type: ignore[arg-type]

    def test_non_sort_key_rejected(self) -> None:
        with pytest.raises(TypeMismatchError):
            sort_objects(_three_by_size(), 42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Algebraic identities
# ---------------------------------------------------------------------------


class TestAlgebraicIdentities:
    def test_asc_then_desc_is_reverse(self) -> None:
        s = _three_by_size()
        asc = sort_objects(s, SortKey.SIZE_ASC)
        desc = sort_objects(s, SortKey.SIZE_DESC)
        # Reversed sizes must match. (Discovery-order tie-break makes
        # this strict for distinct sizes only.)
        assert tuple(o.size for o in asc.objects)[::-1] == tuple(o.size for o in desc.objects)

    def test_sort_is_idempotent(self) -> None:
        s = _three_by_size()
        once = sort_objects(s, SortKey.SIZE_ASC)
        twice = sort_objects(once, SortKey.SIZE_ASC)
        assert tuple(o.color for o in once.objects) == tuple(o.color for o in twice.objects)

    def test_input_set_not_mutated(self) -> None:
        s = _three_by_size()
        original = tuple(o.color for o in s.objects)
        sort_objects(s, SortKey.SIZE_ASC)
        assert tuple(o.color for o in s.objects) == original


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_sort_objects_registered(self) -> None:
        assert "sort_objects" in REGISTRY.names()

    def test_signature(self) -> None:
        spec = REGISTRY.get("sort_objects")
        assert spec.signature.inputs == ("ObjectSet", "SortKey")
        assert spec.signature.output == "ObjectSet"
