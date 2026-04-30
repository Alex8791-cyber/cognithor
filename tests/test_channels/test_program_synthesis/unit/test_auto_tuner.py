# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Cost-Auto-Tuner tests (spec §7.6)."""

from __future__ import annotations

import pytest

from cognithor.channels.program_synthesis.dsl.auto_tuner import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_ROUNDS,
    MIN_COST,
    BenchmarkSample,
    TuneResult,
    auto_tune,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_default_learning_rate_matches_spec(self) -> None:
        # Spec §7.6: ε = 0.05.
        assert DEFAULT_LEARNING_RATE == 0.05

    def test_default_rounds_matches_spec(self) -> None:
        # Spec §7.6: R = 5.
        assert DEFAULT_ROUNDS == 5

    def test_min_cost_floor_protects_ranking(self) -> None:
        # Cost floor stops adversarial benchmark data from collapsing
        # the ranking to "everything is free".
        assert MIN_COST > 0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_inputs_same_outputs(self) -> None:
        costs = {"rotate90": 1.0, "recolor": 1.5}
        samples = (
            BenchmarkSample(solving_program_primitives=("rotate90",)),
            BenchmarkSample(failed_candidate_primitives=("recolor",)),
        )
        a = auto_tune(costs, samples)
        b = auto_tune(costs, samples)
        assert a.final_costs == b.final_costs
        assert len(a.rounds) == len(b.rounds)


# ---------------------------------------------------------------------------
# Update equation
# ---------------------------------------------------------------------------


class TestUpdateEquation:
    def test_successful_primitive_cost_drops(self) -> None:
        # rotate90 helps solve all 3 tasks → its cost should fall.
        samples = tuple(BenchmarkSample(solving_program_primitives=("rotate90",)) for _ in range(3))
        result = auto_tune({"rotate90": 1.0}, samples, rounds=1)
        assert result.final_costs["rotate90"] < 1.0

    def test_failing_primitive_cost_rises(self) -> None:
        # recolor only appears in failed candidates → cost should rise.
        samples = (
            BenchmarkSample(failed_candidate_primitives=("recolor",)),
            BenchmarkSample(failed_candidate_primitives=("recolor",)),
        )
        result = auto_tune({"recolor": 1.5}, samples, rounds=1)
        assert result.final_costs["recolor"] > 1.5

    def test_unrelated_primitive_unchanged(self) -> None:
        # transpose appears in neither side → cost unchanged.
        samples = (BenchmarkSample(solving_program_primitives=("rotate90",)),)
        result = auto_tune({"rotate90": 1.0, "transpose": 1.0}, samples, rounds=1)
        assert result.final_costs["transpose"] == 1.0

    def test_min_cost_floor_kicks_in(self) -> None:
        # Pathological: a primitive solves every task. Without the
        # floor, repeated rounds drive the cost to ~0.
        samples = tuple(
            BenchmarkSample(solving_program_primitives=("rotate90",)) for _ in range(50)
        )
        result = auto_tune({"rotate90": 1.0}, samples, learning_rate=0.5, rounds=20)
        assert result.final_costs["rotate90"] >= MIN_COST


# ---------------------------------------------------------------------------
# Convergence + early termination
# ---------------------------------------------------------------------------


class TestConvergence:
    def test_no_samples_no_changes(self) -> None:
        costs = {"rotate90": 1.0, "recolor": 1.5}
        result = auto_tune(costs, ())
        assert result.final_costs == costs
        # First round produces no change → early terminate.
        assert result.converged_early

    def test_runs_full_rounds_when_data_keeps_changing(self) -> None:
        samples = (
            BenchmarkSample(
                solving_program_primitives=("rotate90", "recolor"),
                failed_candidate_primitives=("transpose",),
            ),
        )
        costs = {"rotate90": 1.0, "recolor": 1.5, "transpose": 1.0}
        result = auto_tune(costs, samples, rounds=3)
        # 3 rounds requested, no convergence until floor → 3 rounds.
        assert len(result.rounds) == 3

    def test_early_termination_on_uniform_round(self) -> None:
        # Once the costs floor at MIN_COST and one primitive's failure
        # weight stabilises, successive rounds produce identical
        # output → early termination.
        samples = (BenchmarkSample(solving_program_primitives=("rotate90",)),)
        result = auto_tune(
            {"rotate90": 0.1},  # already at floor
            samples,
            rounds=10,
        )
        # rotate90 is the only primitive; success rate 1.0 with cost
        # at floor → no change → early terminate after first round.
        assert result.converged_early
        assert len(result.rounds) == 1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_learning_rate_must_be_in_range(self) -> None:
        with pytest.raises(ValueError, match="learning_rate"):
            auto_tune({"x": 1.0}, (), learning_rate=0.0)
        with pytest.raises(ValueError, match="learning_rate"):
            auto_tune({"x": 1.0}, (), learning_rate=1.0)
        with pytest.raises(ValueError, match="learning_rate"):
            auto_tune({"x": 1.0}, (), learning_rate=-0.1)

    def test_rounds_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="rounds"):
            auto_tune({"x": 1.0}, (), rounds=0)
        with pytest.raises(ValueError, match="rounds"):
            auto_tune({"x": 1.0}, (), rounds=-3)


# ---------------------------------------------------------------------------
# TuneResult shape + log
# ---------------------------------------------------------------------------


class TestTuneResult:
    def test_initial_costs_preserved(self) -> None:
        costs = {"rotate90": 1.0, "recolor": 1.5}
        samples = (BenchmarkSample(solving_program_primitives=("rotate90",)),)
        result = auto_tune(costs, samples)
        assert result.initial_costs == costs

    def test_round_log_records_before_after(self) -> None:
        costs = {"rotate90": 1.0}
        samples = (BenchmarkSample(solving_program_primitives=("rotate90",)),)
        result = auto_tune(costs, samples, rounds=1)
        assert len(result.rounds) >= 1
        first = result.rounds[0]
        assert first.round_number == 1
        assert first.costs_before == costs
        assert first.costs_after == result.final_costs

    def test_tune_result_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        result = auto_tune({"x": 1.0}, ())
        with pytest.raises(FrozenInstanceError):
            result.final_costs = {}  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Algebraic / contract identities
# ---------------------------------------------------------------------------


class TestContract:
    def test_empty_catalog_is_no_op(self) -> None:
        result = auto_tune({}, ())
        assert result.final_costs == {}

    def test_costs_never_negative(self) -> None:
        # Even with all primitives failing, cost stays positive.
        samples = (BenchmarkSample(failed_candidate_primitives=("rotate90",)) for _ in range(100))
        result = auto_tune({"rotate90": 0.5}, tuple(samples), rounds=10)
        for c in result.final_costs.values():
            assert c > 0

    def test_costs_keyed_identically_to_input(self) -> None:
        costs = {"rotate90": 1.0, "recolor": 1.5}
        samples = (BenchmarkSample(solving_program_primitives=("rotate90",)),)
        result = auto_tune(costs, samples)
        assert set(result.final_costs.keys()) == set(costs.keys())

    def test_pure_function_no_input_mutation(self) -> None:
        costs = {"rotate90": 1.0}
        original = dict(costs)
        samples = (BenchmarkSample(solving_program_primitives=("rotate90",)),)
        auto_tune(costs, samples)
        assert costs == original

    def test_returns_TuneResult(self) -> None:
        result = auto_tune({"x": 1.0}, ())
        assert isinstance(result, TuneResult)
