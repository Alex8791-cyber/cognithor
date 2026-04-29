# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Predicate type + evaluator tests (spec §6.4 + §7.5)."""

from __future__ import annotations

import pytest

from cognithor.channels.program_synthesis.dsl.predicates import (
    PREDICATE_CONSTRUCTORS,
    Predicate,
    PredicateContext,
    color_eq,
    color_in,
    evaluate_predicate,
    is_largest_in,
    is_rectangle,
    is_smallest_in,
    is_square,
    pred_and,
    pred_not,
    pred_or,
    size_eq,
    size_gt,
    size_lt,
    touches_border,
)
from cognithor.channels.program_synthesis.dsl.types_grid import (
    Object,
    ObjectSet,
)


def _square_3() -> Object:
    return Object(
        color=2,
        cells=((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)),
    )


def _l_shape_4() -> Object:
    # L-shape: 4 cells, NOT rectangular.
    return Object(color=3, cells=((0, 0), (1, 0), (2, 0), (2, 1)))


def _rect_2x3() -> Object:
    return Object(
        color=4,
        cells=((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
    )


# ---------------------------------------------------------------------------
# Predicate dataclass
# ---------------------------------------------------------------------------


class TestPredicate:
    def test_known_constructor_accepted(self) -> None:
        p = Predicate(constructor="color_eq", args=(5,))
        assert p.constructor == "color_eq"
        assert p.args == (5,)
        assert p.output_type == "Bool"

    def test_unknown_constructor_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown predicate constructor"):
            Predicate(constructor="not_a_thing")

    def test_to_source_zero_arity(self) -> None:
        p = Predicate(constructor="is_rectangle")
        assert p.to_source() == "is_rectangle()"

    def test_to_source_with_int_arg(self) -> None:
        p = Predicate(constructor="size_gt", args=(5,))
        assert p.to_source() == "size_gt(5)"

    def test_to_source_with_tuple_arg(self) -> None:
        p = Predicate(constructor="color_in", args=((1, 2, 3),))
        assert "1, 2, 3" in p.to_source()

    def test_to_source_with_nested_predicate(self) -> None:
        inner = Predicate(constructor="color_eq", args=(2,))
        outer = Predicate(constructor="not", args=(inner,))
        assert outer.to_source() == "not(color_eq(2))"

    def test_predicate_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        p = Predicate(constructor="color_eq", args=(5,))
        with pytest.raises(FrozenInstanceError):
            p.constructor = "size_eq"  # type: ignore[misc]

    def test_predicate_constructors_thirteen(self) -> None:
        # Spec §7.5 table — 13 named constructors total
        # (10 leaf + 3 combinators).
        assert len(PREDICATE_CONSTRUCTORS) == 13
        for name in (
            "color_eq",
            "color_in",
            "size_eq",
            "size_gt",
            "size_lt",
            "is_rectangle",
            "is_square",
            "is_largest_in",
            "is_smallest_in",
            "touches_border",
            "not",
            "and",
            "or",
        ):
            assert name in PREDICATE_CONSTRUCTORS


# ---------------------------------------------------------------------------
# evaluate_predicate — leaf constructors
# ---------------------------------------------------------------------------


class TestColorEq:
    def test_match(self) -> None:
        assert evaluate_predicate(color_eq(2), _square_3()) is True

    def test_mismatch(self) -> None:
        assert evaluate_predicate(color_eq(7), _square_3()) is False


class TestColorIn:
    def test_color_in_set(self) -> None:
        assert evaluate_predicate(color_in((1, 2, 3)), _square_3()) is True

    def test_color_not_in_set(self) -> None:
        assert evaluate_predicate(color_in((5, 6, 7)), _square_3()) is False

    def test_empty_set_always_false(self) -> None:
        assert evaluate_predicate(color_in(()), _square_3()) is False

    def test_non_tuple_arg_raises(self) -> None:
        bad = Predicate(constructor="color_in", args=([1, 2, 3],))  # list not tuple
        with pytest.raises(TypeError, match="tuple"):
            evaluate_predicate(bad, _square_3())


class TestSizeComparisons:
    def test_size_eq(self) -> None:
        assert evaluate_predicate(size_eq(9), _square_3()) is True
        assert evaluate_predicate(size_eq(8), _square_3()) is False

    def test_size_gt(self) -> None:
        assert evaluate_predicate(size_gt(8), _square_3()) is True
        assert evaluate_predicate(size_gt(9), _square_3()) is False

    def test_size_lt(self) -> None:
        assert evaluate_predicate(size_lt(10), _square_3()) is True
        assert evaluate_predicate(size_lt(9), _square_3()) is False


class TestShapePredicates:
    def test_is_rectangle_true_for_rectangle(self) -> None:
        assert evaluate_predicate(is_rectangle(), _rect_2x3()) is True

    def test_is_rectangle_false_for_l_shape(self) -> None:
        assert evaluate_predicate(is_rectangle(), _l_shape_4()) is False

    def test_is_square_true_for_3x3(self) -> None:
        assert evaluate_predicate(is_square(), _square_3()) is True

    def test_is_square_false_for_2x3(self) -> None:
        assert evaluate_predicate(is_square(), _rect_2x3()) is False


class TestExtremaPredicates:
    def test_is_largest_in(self) -> None:
        small = Object(color=1, cells=((0, 0),))
        big = _square_3()
        s = ObjectSet(objects=(small, big))
        assert evaluate_predicate(is_largest_in(s), big) is True
        assert evaluate_predicate(is_largest_in(s), small) is False

    def test_is_smallest_in(self) -> None:
        small = Object(color=1, cells=((0, 0),))
        big = _square_3()
        s = ObjectSet(objects=(small, big))
        assert evaluate_predicate(is_smallest_in(s), small) is True
        assert evaluate_predicate(is_smallest_in(s), big) is False

    def test_empty_set_returns_false(self) -> None:
        empty = ObjectSet()
        assert evaluate_predicate(is_largest_in(empty), _square_3()) is False
        assert evaluate_predicate(is_smallest_in(empty), _square_3()) is False

    def test_ties_break_by_discovery_order(self) -> None:
        # Two objects size=2 — only the FIRST is is_largest_in/True.
        a = Object(color=1, cells=((0, 0), (0, 1)))
        b = Object(color=2, cells=((1, 0), (1, 1)))
        s = ObjectSet(objects=(a, b))
        assert evaluate_predicate(is_largest_in(s), a) is True
        assert evaluate_predicate(is_largest_in(s), b) is False

    def test_non_objectset_arg_raises(self) -> None:
        bad = Predicate(constructor="is_largest_in", args=([1, 2],))
        with pytest.raises(TypeError, match="ObjectSet"):
            evaluate_predicate(bad, _square_3())


class TestTouchesBorder:
    def test_corner_object_touches(self) -> None:
        # Object at (0, 0) on a 5×5 grid touches the border.
        obj = Object(color=1, cells=((0, 0),))
        ctx = PredicateContext(grid_shape=(5, 5))
        assert evaluate_predicate(touches_border(), obj, ctx) is True

    def test_inner_object_does_not_touch(self) -> None:
        obj = Object(color=1, cells=((2, 2), (2, 3)))
        ctx = PredicateContext(grid_shape=(5, 5))
        assert evaluate_predicate(touches_border(), obj, ctx) is False

    def test_missing_grid_shape_raises(self) -> None:
        obj = Object(color=1, cells=((0, 0),))
        with pytest.raises(ValueError, match="grid_shape"):
            evaluate_predicate(touches_border(), obj)

    def test_empty_object_does_not_touch(self) -> None:
        obj = Object(color=1, cells=())
        ctx = PredicateContext(grid_shape=(5, 5))
        assert evaluate_predicate(touches_border(), obj, ctx) is False


# ---------------------------------------------------------------------------
# evaluate_predicate — combinators
# ---------------------------------------------------------------------------


class TestNot:
    def test_inverts_truthy(self) -> None:
        p = pred_not(color_eq(2))
        assert evaluate_predicate(p, _square_3()) is False

    def test_inverts_falsy(self) -> None:
        p = pred_not(color_eq(7))
        assert evaluate_predicate(p, _square_3()) is True

    def test_double_not_is_identity(self) -> None:
        inner = color_eq(2)
        double = pred_not(pred_not(inner))
        assert evaluate_predicate(double, _square_3()) == evaluate_predicate(inner, _square_3())

    def test_non_predicate_arg_raises(self) -> None:
        bad = Predicate(constructor="not", args=("not a predicate",))
        with pytest.raises(TypeError, match="Predicate"):
            evaluate_predicate(bad, _square_3())


class TestAnd:
    def test_both_true(self) -> None:
        p = pred_and(color_eq(2), is_square())
        assert evaluate_predicate(p, _square_3()) is True

    def test_one_false(self) -> None:
        p = pred_and(color_eq(2), is_rectangle())  # both true for 3x3 — pick a false
        # Use a clearly-false combo.
        p = pred_and(color_eq(2), color_eq(7))
        assert evaluate_predicate(p, _square_3()) is False

    def test_short_circuits_on_first_false(self) -> None:
        # If the first arg is false, the second should not even be
        # evaluated. Use a combinator that would crash on its own as
        # the second arg.
        bad = Predicate(constructor="not", args=("not a predicate",))
        p = pred_and(color_eq(99), bad)  # color_eq(99) is False on color=2
        # Should NOT raise — short-circuit kicks in.
        assert evaluate_predicate(p, _square_3()) is False


class TestOr:
    def test_both_false(self) -> None:
        p = pred_or(color_eq(7), color_eq(8))
        assert evaluate_predicate(p, _square_3()) is False

    def test_one_true(self) -> None:
        p = pred_or(color_eq(2), color_eq(7))
        assert evaluate_predicate(p, _square_3()) is True

    def test_short_circuits_on_first_true(self) -> None:
        bad = Predicate(constructor="not", args=("not a predicate",))
        p = pred_or(color_eq(2), bad)  # first is True; second never evaluated
        assert evaluate_predicate(p, _square_3()) is True


# ---------------------------------------------------------------------------
# De Morgan algebra (combinator interactions)
# ---------------------------------------------------------------------------


class TestDeMorgan:
    def test_not_and_equals_or_not(self) -> None:
        a = color_eq(2)
        b = is_square()
        lhs = pred_not(pred_and(a, b))
        rhs = pred_or(pred_not(a), pred_not(b))
        assert evaluate_predicate(lhs, _square_3()) == evaluate_predicate(rhs, _square_3())

    def test_not_or_equals_and_not(self) -> None:
        a = color_eq(2)
        b = is_square()
        lhs = pred_not(pred_or(a, b))
        rhs = pred_and(pred_not(a), pred_not(b))
        assert evaluate_predicate(lhs, _square_3()) == evaluate_predicate(rhs, _square_3())
