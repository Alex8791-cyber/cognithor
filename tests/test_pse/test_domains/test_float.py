"""Tests for the Float-precision domain (Sprint-26.3)."""

from __future__ import annotations

import math

import pytest

from cognithor.channels.program_synthesis.domains.float_dsl import (
    FLOAT_PRIMITIVE_NAMES,
    FloatCatalog,
    FloatDomain,
    FloatPrimitive,
    FloatVerifierError,
    build_float_catalog,
    register_float_domain,
)
from cognithor.channels.program_synthesis.domains.registry import DomainRegistry


class TestFloatCatalog:
    def test_builds(self) -> None:
        cat = build_float_catalog()
        assert isinstance(cat, FloatCatalog)
        assert len(cat) == len(FLOAT_PRIMITIVE_NAMES)

    def test_at_least_15_primitives(self) -> None:
        assert len(FLOAT_PRIMITIVE_NAMES) >= 15

    def test_invalid_primitive_name(self) -> None:
        with pytest.raises(ValueError, match="Invalid Float"):
            FloatPrimitive(name="bad-!", fn=lambda: 0.0, cost=0.1)

    def test_negative_cost_rejected(self) -> None:
        with pytest.raises(ValueError, match=">= 0"):
            FloatPrimitive(name="p", fn=lambda: 0.0, cost=-0.1)


class TestSums:
    """Kahan/Neumaier reduce accumulator drift over many small additions."""

    def test_naive_drifts_over_many_small(self) -> None:
        cat = build_float_catalog()
        # 1e6 additions of 0.1 — naive accumulates rounding error.
        values = [0.1] * 100_000
        out = cat.get("naive_sum").fn(values)
        # Naive: typical drift > 1e-9 on 100k summands.
        assert abs(out - 10_000.0) > 1e-9

    def test_kahan_handles_drift_over_many_small(self) -> None:
        cat = build_float_catalog()
        values = [0.1] * 100_000
        out = cat.get("kahan_sum").fn(values)
        assert abs(out - 10_000.0) < 1e-9

    def test_neumaier_handles_drift_over_many_small(self) -> None:
        cat = build_float_catalog()
        values = [0.1] * 100_000
        out = cat.get("neumaier_sum").fn(values)
        assert abs(out - 10_000.0) < 1e-9

    def test_neumaier_handles_partial_sum_smaller_than_addend(self) -> None:
        cat = build_float_catalog()
        # Classic case where Neumaier wins over Kahan: partial sum
        # smaller than the next addend at one step.
        values = [1.0, 1e100, 1.0, -1e100]
        out = cat.get("neumaier_sum").fn(values)
        assert out == pytest.approx(2.0, abs=1e-6)


class TestSafeOps:
    def test_safe_div_zero(self) -> None:
        cat = build_float_catalog()
        assert cat.get("safe_div").fn(1.0, 0.0) == 0.0
        assert cat.get("safe_div").fn(1.0, 0.0, default=-1.0) == -1.0

    def test_safe_div_normal(self) -> None:
        cat = build_float_catalog()
        assert cat.get("safe_div").fn(10.0, 4.0) == 2.5

    def test_safe_log_negative(self) -> None:
        cat = build_float_catalog()
        out = cat.get("safe_log").fn(-1.0)
        assert math.isinf(out) and out < 0  # -inf default

    def test_safe_sqrt_negative(self) -> None:
        cat = build_float_catalog()
        assert cat.get("safe_sqrt").fn(-4.0, default=0.0) == 0.0


class TestComparison:
    def test_nearly_equal_within_eps(self) -> None:
        cat = build_float_catalog()
        assert cat.get("nearly_equal").fn(0.1 + 0.2, 0.3) is True

    def test_nearly_equal_handles_inf(self) -> None:
        cat = build_float_catalog()
        assert cat.get("nearly_equal").fn(float("inf"), float("inf")) is True
        assert cat.get("nearly_equal").fn(float("inf"), 1.0) is False

    def test_nearly_equal_nan_is_false(self) -> None:
        cat = build_float_catalog()
        assert cat.get("nearly_equal").fn(float("nan"), float("nan")) is False

    def test_relative_error_zero_expected(self) -> None:
        cat = build_float_catalog()
        assert cat.get("relative_error").fn(2.0, 0.0) == 2.0

    def test_relative_error_normal(self) -> None:
        cat = build_float_catalog()
        out = cat.get("relative_error").fn(110.0, 100.0)
        assert out == pytest.approx(0.1)


class TestFinitenessHelpers:
    def test_clamp_finite_nan(self) -> None:
        cat = build_float_catalog()
        assert cat.get("clamp_finite").fn(float("nan")) == 0.0

    def test_clamp_finite_pos_inf(self) -> None:
        cat = build_float_catalog()
        out = cat.get("clamp_finite").fn(float("inf"), hi=100.0)
        assert out == 100.0

    def test_replace_nan_default(self) -> None:
        cat = build_float_catalog()
        assert cat.get("replace_nan").fn(float("nan"), default=0.0) == 0.0

    def test_replace_inf(self) -> None:
        cat = build_float_catalog()
        assert cat.get("replace_inf").fn(float("inf"), default=42.0) == 42.0

    def test_is_subnormal_false_for_normal(self) -> None:
        cat = build_float_catalog()
        assert cat.get("is_subnormal").fn(1.0) is False

    def test_is_subnormal_true_for_denormal(self) -> None:
        cat = build_float_catalog()
        assert cat.get("is_subnormal").fn(5e-310) is True

    def test_is_nan(self) -> None:
        cat = build_float_catalog()
        assert cat.get("is_nan").fn(float("nan")) is True
        assert cat.get("is_nan").fn(1.0) is False

    def test_is_inf(self) -> None:
        cat = build_float_catalog()
        assert cat.get("is_inf").fn(float("inf")) is True
        assert cat.get("is_inf").fn(1.0) is False

    def test_next_after_increases(self) -> None:
        cat = build_float_catalog()
        out = cat.get("next_after").fn(1.0, 2.0)
        assert out > 1.0

    def test_round_half_to_even(self) -> None:
        cat = build_float_catalog()
        # Banker's rounding: 0.5 → 0, 1.5 → 2, 2.5 → 2
        assert cat.get("round_half_to_even").fn(0.5) == 0
        assert cat.get("round_half_to_even").fn(1.5) == 2
        assert cat.get("round_half_to_even").fn(2.5) == 2


class TestFloatDomain:
    def test_metadata(self) -> None:
        d = FloatDomain()
        m = d.metadata
        assert m.name == "float"
        assert m.benchmark_target == 0.70

    def test_register(self) -> None:
        reg = DomainRegistry()
        register_float_domain(reg)
        assert isinstance(reg.get("float"), FloatDomain)

    def test_verify_passing(self) -> None:
        d = FloatDomain()
        ok = d.verify(
            [{"primitive": "kahan_sum", "args": {}}],
            [{"input": [1.0, 2.0, 3.0], "output": 6.0}],
        )
        assert ok

    def test_verify_uses_eps_match(self) -> None:
        d = FloatDomain()
        ok = d.verify(
            [{"primitive": "naive_sum", "args": {}}],
            [{"input": [0.1, 0.2], "output": 0.3}],  # 0.30000000000000004 vs 0.3
        )
        assert ok  # eps comparison default

    def test_verify_mismatch_raises(self) -> None:
        d = FloatDomain()
        with pytest.raises(FloatVerifierError, match="!= expected"):
            d.verify(
                [{"primitive": "naive_sum", "args": {}}],
                [{"input": [1.0, 2.0], "output": 99.0}],
            )

    def test_verify_dict_program(self) -> None:
        d = FloatDomain()
        ok = d.verify(
            {"program": [{"primitive": "is_nan", "args": {}}]},
            [{"input": float("nan"), "output": True}],
        )
        assert ok

    def test_verify_step_value_error(self) -> None:
        d = FloatDomain()
        # Pass invalid kwargs to trigger TypeError → raised as VerifierError
        with pytest.raises(FloatVerifierError, match="TypeError|ValueError"):
            d.verify(
                [{"primitive": "kahan_sum", "args": {"unknown": 1}}],
                [{"input": [], "output": 0.0}],
            )

    def test_verify_program_must_be_list_or_dict(self) -> None:
        d = FloatDomain()
        with pytest.raises(FloatVerifierError, match="must be"):
            d.verify("nope", [])

    def test_verify_step_not_a_mapping(self) -> None:
        d = FloatDomain()
        with pytest.raises(FloatVerifierError, match="must be a mapping"):
            d.verify(["not-a-step"], [])

    def test_verify_unknown_primitive(self) -> None:
        d = FloatDomain()
        with pytest.raises(FloatVerifierError, match="Unknown Float"):
            d.verify(
                [{"primitive": "no_such_thing", "args": {}}],
                [{"input": 1.0, "output": 1.0}],
            )

    def test_bool_is_strict_match_not_eps(self) -> None:
        d = FloatDomain()
        ok = d.verify(
            [{"primitive": "is_nan", "args": {}}],
            [{"input": float("nan"), "output": True}],
        )
        assert ok
        with pytest.raises(FloatVerifierError):
            d.verify(
                [{"primitive": "is_nan", "args": {}}],
                [{"input": 1.0, "output": True}],
            )
