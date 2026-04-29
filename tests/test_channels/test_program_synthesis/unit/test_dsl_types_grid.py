# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Object / ObjectSet tests (spec §6 + §7.5 prerequisite types)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from cognithor.channels.program_synthesis.dsl.types_grid import (
    Object,
    ObjectSet,
)


class TestObject:
    def test_size_matches_cells(self) -> None:
        o = Object(color=2, cells=((0, 0), (0, 1), (1, 1)))
        assert o.size == 3

    def test_bbox(self) -> None:
        o = Object(color=2, cells=((0, 0), (0, 1), (1, 1)))
        # half-open: r1 = max_r + 1, c1 = max_c + 1
        assert o.bbox == (0, 2, 0, 2)

    def test_empty_bbox(self) -> None:
        o = Object(color=0, cells=())
        assert o.bbox == (0, 0, 0, 0)

    def test_cells_canonicalised_on_construction(self) -> None:
        # Cells given in arbitrary order should be sorted at construction.
        o = Object(color=1, cells=((1, 1), (0, 0), (0, 1)))
        assert o.cells == ((0, 0), (0, 1), (1, 1))

    def test_is_rectangle_true_for_rect(self) -> None:
        # Full 2×2 block.
        o = Object(color=3, cells=((0, 0), (0, 1), (1, 0), (1, 1)))
        assert o.is_rectangle()

    def test_is_rectangle_false_for_l_shape(self) -> None:
        o = Object(color=3, cells=((0, 0), (1, 0), (1, 1)))
        assert not o.is_rectangle()

    def test_is_square_requires_rectangle_and_equal_sides(self) -> None:
        sq = Object(color=4, cells=((0, 0), (0, 1), (1, 0), (1, 1)))
        assert sq.is_square()
        rect = Object(color=4, cells=((0, 0), (0, 1)))
        assert rect.is_rectangle() and not rect.is_square()

    def test_color_validation(self) -> None:
        with pytest.raises(ValueError, match="out of ARC range"):
            Object(color=10, cells=())

    def test_frozen(self) -> None:
        o = Object(color=1, cells=((0, 0),))
        with pytest.raises(FrozenInstanceError):
            o.color = 2  # type: ignore[misc]

    def test_equality_is_structural(self) -> None:
        a = Object(color=1, cells=((0, 1), (0, 0)))
        b = Object(color=1, cells=((0, 0), (0, 1)))
        # Both canonicalised the same way.
        assert a == b
        assert hash(a) == hash(b)


class TestObjectSet:
    def test_empty(self) -> None:
        s = ObjectSet()
        assert s.is_empty()
        assert len(s) == 0

    def test_iter_and_index(self) -> None:
        a = Object(color=1, cells=((0, 0),))
        b = Object(color=2, cells=((1, 1),))
        s = ObjectSet(objects=(a, b))
        assert list(s) == [a, b]
        assert s[0] is a
        assert s[1] is b

    def test_frozen(self) -> None:
        s = ObjectSet(objects=())
        with pytest.raises(FrozenInstanceError):
            s.objects = ()  # type: ignore[misc]
