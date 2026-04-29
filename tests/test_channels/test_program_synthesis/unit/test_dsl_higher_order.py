# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Higher-order primitive tests — H1 map_objects + H2 filter_objects (spec §7.5)."""

from __future__ import annotations

import numpy as np
import pytest

from cognithor.channels.program_synthesis.core.exceptions import TypeMismatchError
from cognithor.channels.program_synthesis.dsl.lambdas import (
    identity_lambda,
    recolor_lambda,
    shift_lambda,
)
from cognithor.channels.program_synthesis.dsl.predicates import (
    color_eq,
    is_largest_in,
    pred_not,
    size_gt,
)
from cognithor.channels.program_synthesis.dsl.primitives import (
    filter_objects,
    map_objects,
)
from cognithor.channels.program_synthesis.dsl.registry import REGISTRY
from cognithor.channels.program_synthesis.dsl.types_grid import Object, ObjectSet


def _g(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


def _three_objects() -> ObjectSet:
    return ObjectSet(
        objects=(
            Object(color=1, cells=((0, 0),)),
            Object(color=2, cells=((1, 0), (1, 1))),
            Object(color=3, cells=((2, 0), (2, 1), (2, 2))),
        )
    )


# ---------------------------------------------------------------------------
# map_objects (H1)
# ---------------------------------------------------------------------------


class TestMapObjects:
    def test_recolor_every_object(self) -> None:
        s = _three_objects()
        result = map_objects(s, recolor_lambda(5))
        assert len(result) == 3
        for o in result:
            assert o.color == 5

    def test_identity_is_no_op(self) -> None:
        s = _three_objects()
        result = map_objects(s, identity_lambda())
        assert len(result) == len(s)
        for a, b in zip(result.objects, s.objects, strict=True):
            assert a == b

    def test_shift_every_object(self) -> None:
        s = _three_objects()
        result = map_objects(s, shift_lambda(10, 0))
        # Every cell should be shifted down by 10 rows.
        for original, shifted in zip(s.objects, result.objects, strict=True):
            for (or_, oc), (sr, sc) in zip(original.cells, shifted.cells, strict=True):
                assert sr == or_ + 10
                assert sc == oc

    def test_empty_input_returns_empty(self) -> None:
        result = map_objects(ObjectSet(), recolor_lambda(5))
        assert result.is_empty()

    def test_preserves_order(self) -> None:
        s = _three_objects()
        result = map_objects(s, recolor_lambda(7))
        # Order must match input.
        assert len(result) == 3

    def test_rejects_non_object_set(self) -> None:
        with pytest.raises(TypeMismatchError):
            map_objects([1, 2], recolor_lambda(5))  # type: ignore[arg-type]

    def test_rejects_non_lambda(self) -> None:
        with pytest.raises(TypeMismatchError):
            map_objects(_three_objects(), "not a lambda")  # type: ignore[arg-type]

    def test_input_set_not_mutated(self) -> None:
        s = _three_objects()
        original_colors = [o.color for o in s.objects]
        map_objects(s, recolor_lambda(9))
        # Source set untouched.
        assert [o.color for o in s.objects] == original_colors


# ---------------------------------------------------------------------------
# filter_objects (H2)
# ---------------------------------------------------------------------------


class TestFilterObjects:
    def test_filter_by_color(self) -> None:
        s = _three_objects()
        result = filter_objects(s, color_eq(2))
        assert len(result) == 1
        assert result[0].color == 2

    def test_filter_by_size_gt(self) -> None:
        s = _three_objects()
        result = filter_objects(s, size_gt(1))
        # Two objects have size > 1 (size 2 and size 3).
        assert len(result) == 2

    def test_no_match_returns_empty(self) -> None:
        s = _three_objects()
        result = filter_objects(s, color_eq(9))
        assert result.is_empty()

    def test_all_match_returns_full_set(self) -> None:
        s = _three_objects()
        result = filter_objects(s, size_gt(0))
        assert len(result) == 3

    def test_empty_input_returns_empty(self) -> None:
        result = filter_objects(ObjectSet(), color_eq(1))
        assert result.is_empty()

    def test_preserves_order(self) -> None:
        s = _three_objects()
        result = filter_objects(s, size_gt(0))
        # Must keep discovery order.
        assert tuple(o.color for o in result.objects) == (1, 2, 3)

    def test_negation_combinator(self) -> None:
        s = _three_objects()
        # NOT color_eq(2) → keep colors 1 and 3.
        result = filter_objects(s, pred_not(color_eq(2)))
        colors = sorted(o.color for o in result.objects)
        assert colors == [1, 3]

    def test_largest_in_uses_input_set_context(self) -> None:
        s = _three_objects()
        # is_largest_in(s) — only the size=3 object survives.
        result = filter_objects(s, is_largest_in(s))
        assert len(result) == 1
        assert result[0].size == 3

    def test_rejects_non_object_set(self) -> None:
        with pytest.raises(TypeMismatchError):
            filter_objects("nope", color_eq(1))  # type: ignore[arg-type]

    def test_rejects_non_predicate(self) -> None:
        with pytest.raises(TypeMismatchError):
            filter_objects(_three_objects(), "not a predicate")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Compose: filter then map (the canonical Phase-1.5 pattern)
# ---------------------------------------------------------------------------


class TestCompose:
    def test_filter_then_map_recolor(self) -> None:
        # Spec §24.4 example: blue objects → red.
        s = _three_objects()  # colors 1, 2, 3
        # Keep only color=2, then recolor to 5.
        filtered = filter_objects(s, color_eq(2))
        recolored = map_objects(filtered, recolor_lambda(5))
        assert len(recolored) == 1
        assert recolored[0].color == 5

    def test_compose_stays_pure(self) -> None:
        s = _three_objects()
        original = tuple((o.color, o.cells) for o in s.objects)
        filter_objects(s, color_eq(2))
        map_objects(s, recolor_lambda(5))
        # Source set untouched after both calls.
        assert tuple((o.color, o.cells) for o in s.objects) == original


# ---------------------------------------------------------------------------
# Registry side-effect
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_higher_order_primitives_registered(self) -> None:
        names = REGISTRY.names()
        assert "map_objects" in names
        assert "filter_objects" in names

    def test_signatures(self) -> None:
        spec = REGISTRY.get("map_objects")
        assert spec.signature.inputs == ("ObjectSet", "Lambda")
        assert spec.signature.output == "ObjectSet"
        spec = REGISTRY.get("filter_objects")
        assert spec.signature.inputs == ("ObjectSet", "Predicate")
        assert spec.signature.output == "ObjectSet"

    def test_total_catalog_grows_to_58(self) -> None:
        # 56 base primitives + 2 higher-order so far (H1 + H2). H3-H5
        # bring it to 61 (or more if helper sub-primitives land).
        assert len(REGISTRY) == 58
