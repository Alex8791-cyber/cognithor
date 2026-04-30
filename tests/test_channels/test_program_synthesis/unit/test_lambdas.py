# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Lambda type + evaluator tests (spec §6.4 + §7.5)."""

from __future__ import annotations

import pytest

from cognithor.channels.program_synthesis.dsl.lambdas import (
    LAMBDA_CONSTRUCTORS,
    Lambda,
    evaluate_lambda,
    identity_lambda,
    recolor_lambda,
    shift_lambda,
)
from cognithor.channels.program_synthesis.dsl.types_grid import Object


def _obj(color: int = 1) -> Object:
    return Object(color=color, cells=((0, 0), (0, 1), (1, 0)))


# ---------------------------------------------------------------------------
# Lambda dataclass
# ---------------------------------------------------------------------------


class TestLambda:
    def test_known_constructor_accepted(self) -> None:
        fn = Lambda(constructor="recolor_lambda", args=(5,))
        assert fn.constructor == "recolor_lambda"
        assert fn.args == (5,)
        assert fn.variable_type == "Object"
        assert fn.output_type == "Object"

    def test_unknown_constructor_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown lambda constructor"):
            Lambda(constructor="not_a_thing")

    def test_to_source_zero_arity(self) -> None:
        assert Lambda(constructor="identity_lambda").to_source() == "identity_lambda()"

    def test_to_source_with_int_arg(self) -> None:
        assert recolor_lambda(5).to_source() == "recolor_lambda(5)"

    def test_to_source_with_two_args(self) -> None:
        assert shift_lambda(1, 2).to_source() == "shift_lambda(1, 2)"

    def test_lambda_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        fn = recolor_lambda(5)
        with pytest.raises(FrozenInstanceError):
            fn.constructor = "identity_lambda"  # type: ignore[misc]

    def test_base_constructors_registered(self) -> None:
        # Floor + presence check — Phase 1.5 grows the set as H1+H5 land.
        assert len(LAMBDA_CONSTRUCTORS) >= 3
        for name in ("identity_lambda", "recolor_lambda", "shift_lambda"):
            assert name in LAMBDA_CONSTRUCTORS


# ---------------------------------------------------------------------------
# evaluate_lambda
# ---------------------------------------------------------------------------


class TestIdentityLambda:
    def test_returns_same_object(self) -> None:
        obj = _obj(color=4)
        result = evaluate_lambda(identity_lambda(), obj)
        assert result is obj or result == obj


class TestRecolorLambda:
    def test_recolors_to_new_value(self) -> None:
        obj = _obj(color=1)
        result = evaluate_lambda(recolor_lambda(7), obj)
        assert result.color == 7
        # Cells are preserved.
        assert result.cells == obj.cells

    def test_recolor_to_same_color_is_no_op(self) -> None:
        obj = _obj(color=3)
        result = evaluate_lambda(recolor_lambda(3), obj)
        assert result.color == 3

    def test_out_of_range_color_raises(self) -> None:
        with pytest.raises(TypeError, match="out of ARC range"):
            evaluate_lambda(recolor_lambda(99), _obj())

    def test_negative_color_raises(self) -> None:
        with pytest.raises(TypeError, match="out of ARC range"):
            evaluate_lambda(recolor_lambda(-1), _obj())

    def test_bool_arg_rejected(self) -> None:
        bad = Lambda(constructor="recolor_lambda", args=(True,))
        with pytest.raises(TypeError, match="int"):
            evaluate_lambda(bad, _obj())


class TestShiftLambda:
    def test_shifts_cells_by_delta(self) -> None:
        obj = Object(color=2, cells=((0, 0), (1, 1)))
        result = evaluate_lambda(shift_lambda(2, 3), obj)
        # cells are sorted on construction by Object — so the output is
        # ((2, 3), (3, 4)).
        assert result.cells == ((2, 3), (3, 4))
        assert result.color == 2

    def test_zero_shift_preserves_cells(self) -> None:
        obj = _obj(color=5)
        result = evaluate_lambda(shift_lambda(0, 0), obj)
        assert result.cells == obj.cells

    def test_negative_shift_works(self) -> None:
        obj = Object(color=1, cells=((5, 5),))
        result = evaluate_lambda(shift_lambda(-3, -3), obj)
        assert result.cells == ((2, 2),)

    def test_non_int_dy_raises(self) -> None:
        bad = Lambda(constructor="shift_lambda", args=(1.5, 0))
        with pytest.raises(TypeError, match="int"):
            evaluate_lambda(bad, _obj())

    def test_bool_dx_rejected(self) -> None:
        bad = Lambda(constructor="shift_lambda", args=(0, True))
        with pytest.raises(TypeError, match="int"):
            evaluate_lambda(bad, _obj())


class TestEvaluatorContract:
    def test_input_object_not_mutated(self) -> None:
        obj = _obj(color=1)
        original_color = obj.color
        original_cells = obj.cells
        evaluate_lambda(recolor_lambda(9), obj)
        assert obj.color == original_color
        assert obj.cells == original_cells

    def test_pure_function_deterministic(self) -> None:
        obj = _obj(color=1)
        a = evaluate_lambda(recolor_lambda(7), obj)
        b = evaluate_lambda(recolor_lambda(7), obj)
        assert a == b
