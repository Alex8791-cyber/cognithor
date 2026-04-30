# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""branch (H5) tests (spec §7.5 v1.2 — conditional-Lambda combinator)."""

from __future__ import annotations

import pytest

from cognithor.channels.program_synthesis.core.exceptions import TypeMismatchError
from cognithor.channels.program_synthesis.dsl.lambdas import (
    branch_lambda,
    evaluate_lambda,
    identity_lambda,
    recolor_lambda,
)
from cognithor.channels.program_synthesis.dsl.predicates import (
    color_eq,
    pred_not,
    size_gt,
    size_lt,
)
from cognithor.channels.program_synthesis.dsl.primitives import (
    branch,
    map_objects,
)
from cognithor.channels.program_synthesis.dsl.registry import REGISTRY
from cognithor.channels.program_synthesis.dsl.types_grid import Object, ObjectSet


def _obj(*, color: int, cells: tuple[tuple[int, int], ...]) -> Object:
    return Object(color=color, cells=cells)


# ---------------------------------------------------------------------------
# branch — basic happy paths
# ---------------------------------------------------------------------------


class TestBranchBasics:
    def test_predicate_true_runs_then_branch(self) -> None:
        # size_gt(2) on a 3-cell object → true → then_fn (recolor 9).
        big = _obj(color=1, cells=((0, 0), (0, 1), (0, 2)))
        fn = branch(size_gt(2), recolor_lambda(9), identity_lambda())
        result = evaluate_lambda(fn, big)
        assert result.color == 9

    def test_predicate_false_runs_else_branch(self) -> None:
        # size_gt(2) on a 1-cell object → false → else_fn (identity).
        small = _obj(color=1, cells=((0, 0),))
        fn = branch(size_gt(2), recolor_lambda(9), identity_lambda())
        result = evaluate_lambda(fn, small)
        assert result.color == 1

    def test_returns_lambda_not_object(self) -> None:
        # branch is a primitive that produces a Lambda, not an Object.
        from cognithor.channels.program_synthesis.dsl.lambdas import Lambda

        fn = branch(color_eq(1), identity_lambda(), recolor_lambda(2))
        assert isinstance(fn, Lambda)
        assert fn.constructor == "branch_lambda"

    def test_branch_inside_map_objects(self) -> None:
        # Spec §7.5 example: large objects → red, small → unchanged.
        s = ObjectSet(
            objects=(
                _obj(color=1, cells=((0, 0),)),  # size 1
                _obj(color=1, cells=((1, 0), (1, 1))),  # size 2
                _obj(color=1, cells=((2, 0), (2, 1), (2, 2))),  # size 3
            )
        )
        fn = branch(size_gt(1), recolor_lambda(2), identity_lambda())
        result = map_objects(s, fn)
        # Sizes 2 and 3 → recoloured to 2; size 1 stays 1.
        colors = tuple(o.color for o in result.objects)
        assert colors == (1, 2, 2)


# ---------------------------------------------------------------------------
# Type validation
# ---------------------------------------------------------------------------


class TestBranchValidation:
    def test_non_predicate_arg_rejected(self) -> None:
        with pytest.raises(TypeMismatchError):
            branch("not a predicate", identity_lambda(), identity_lambda())  # type: ignore[arg-type]

    def test_non_lambda_then_branch_rejected(self) -> None:
        with pytest.raises(TypeMismatchError):
            branch(color_eq(1), "not a lambda", identity_lambda())  # type: ignore[arg-type]

    def test_non_lambda_else_branch_rejected(self) -> None:
        with pytest.raises(TypeMismatchError):
            branch(color_eq(1), identity_lambda(), 42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Sub-tiefe ≤ 1 — nested branches forbidden
# ---------------------------------------------------------------------------


class TestSubDepth:
    def test_nested_branch_in_then_rejected(self) -> None:
        inner = branch(color_eq(1), identity_lambda(), recolor_lambda(2))
        with pytest.raises(TypeMismatchError, match="nested"):
            branch(size_gt(0), inner, identity_lambda())

    def test_nested_branch_in_else_rejected(self) -> None:
        inner = branch(color_eq(1), identity_lambda(), recolor_lambda(2))
        with pytest.raises(TypeMismatchError, match="nested"):
            branch(size_gt(0), identity_lambda(), inner)


# ---------------------------------------------------------------------------
# Algebraic identities (spec §7.5 calls these out explicitly)
# ---------------------------------------------------------------------------


class TestBranchAlgebra:
    def test_branch_with_always_true_predicate_equals_then_fn(self) -> None:
        # Always-true predicate (e.g. size_gt(-1)) → behaves like then_fn.
        always_true = size_gt(-1)
        fn = branch(always_true, recolor_lambda(7), identity_lambda())
        obj = _obj(color=1, cells=((0, 0),))
        result = evaluate_lambda(fn, obj)
        assert result.color == 7

    def test_branch_with_always_false_predicate_equals_else_fn(self) -> None:
        always_false = size_lt(0)  # no object can have negative size
        fn = branch(always_false, recolor_lambda(7), identity_lambda())
        obj = _obj(color=1, cells=((0, 0),))
        result = evaluate_lambda(fn, obj)
        # else_fn is identity → unchanged.
        assert result.color == 1

    def test_branch_with_negated_predicate_swaps_branches(self) -> None:
        # branch(not(p), f, g) ≡ branch(p, g, f).
        obj = _obj(color=1, cells=((0, 0),))
        positive = branch(color_eq(1), recolor_lambda(2), recolor_lambda(3))
        negated = branch(pred_not(color_eq(1)), recolor_lambda(3), recolor_lambda(2))
        a = evaluate_lambda(positive, obj)
        b = evaluate_lambda(negated, obj)
        assert a.color == b.color


# ---------------------------------------------------------------------------
# Direct branch_lambda builder also enforces sub-tiefe at eval time
# ---------------------------------------------------------------------------


class TestBranchLambdaBuilder:
    def test_builder_constructs_lambda(self) -> None:
        from cognithor.channels.program_synthesis.dsl.lambdas import Lambda

        fn = branch_lambda(color_eq(1), identity_lambda(), identity_lambda())
        assert isinstance(fn, Lambda)
        assert fn.constructor == "branch_lambda"

    def test_evaluator_rejects_nested_at_eval(self) -> None:
        # The builder doesn't pre-check (the @primitive does, but the
        # raw builder allows it for the search engine to construct
        # invalid candidates and have them rejected at evaluation).
        inner = branch_lambda(color_eq(1), identity_lambda(), identity_lambda())
        nested = branch_lambda(color_eq(1), inner, identity_lambda())
        obj = _obj(color=1, cells=((0, 0),))
        with pytest.raises(ValueError, match="nested branches"):
            evaluate_lambda(nested, obj)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_branch_primitive_registered(self) -> None:
        assert "branch" in REGISTRY.names()

    def test_branch_signature(self) -> None:
        spec = REGISTRY.get("branch")
        assert spec.signature.inputs == ("Predicate", "Lambda", "Lambda")
        assert spec.signature.output == "Lambda"

    def test_branch_cost_matches_spec(self) -> None:
        spec = REGISTRY.get("branch")
        # spec §7.5 mandates cost 3.5 — H5 is the most expensive
        # higher-order primitive.
        assert spec.cost == 3.5

    def test_lambda_constructors_now_four(self) -> None:
        from cognithor.channels.program_synthesis.dsl.lambdas import (
            LAMBDA_CONSTRUCTORS,
        )

        assert len(LAMBDA_CONSTRUCTORS) == 4
        assert "branch_lambda" in LAMBDA_CONSTRUCTORS
