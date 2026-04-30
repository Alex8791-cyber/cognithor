# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Triviality-rule tests (Sprint-1 plan task 8 slice, spec §7.3.1)."""

from __future__ import annotations

import numpy as np

from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.phase2 import triviality_score
from cognithor.channels.program_synthesis.phase2.triviality import (
    r1_output_equals_input,
    r2_output_is_constant,
    r3_single_pixel_diff,
    r4_near_identity,
    r5_output_unchanged_across_demos,
)


def _g(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


# ---------------------------------------------------------------------------
# R1 — output equals input
# ---------------------------------------------------------------------------


class TestR1OutputEqualsInput:
    def test_fires_when_every_demo_returns_input(self) -> None:
        inputs = [_g([[1, 2], [3, 4]]), _g([[5]])]
        actuals = [_g([[1, 2], [3, 4]]), _g([[5]])]  # same as inputs
        assert r1_output_equals_input(actuals, inputs) == 1.0

    def test_does_not_fire_when_one_demo_differs(self) -> None:
        inputs = [_g([[1, 2]]), _g([[3]])]
        actuals = [_g([[1, 2]]), _g([[9]])]  # second differs
        assert r1_output_equals_input(actuals, inputs) == 0.0

    def test_empty_lists_do_not_fire(self) -> None:
        assert r1_output_equals_input([], []) == 0.0


# ---------------------------------------------------------------------------
# R2 — output is constant
# ---------------------------------------------------------------------------


class TestR2OutputIsConstant:
    def test_fires_for_single_value_grids(self) -> None:
        actuals = [_g([[5, 5], [5, 5]]), _g([[7, 7]])]
        assert r2_output_is_constant(actuals) == 1.0

    def test_does_not_fire_for_two_value_grid(self) -> None:
        actuals = [_g([[1, 2]])]
        assert r2_output_is_constant(actuals) == 0.0

    def test_empty_grid_does_not_fire(self) -> None:
        actuals = [np.zeros((0, 0), dtype=np.int8)]
        assert r2_output_is_constant(actuals) == 0.0


# ---------------------------------------------------------------------------
# R3 — single pixel diff
# ---------------------------------------------------------------------------


class TestR3SinglePixelDiff:
    def test_fires_when_only_one_pixel_changes(self) -> None:
        inputs = [_g([[1, 2], [3, 4]])]
        actuals = [_g([[9, 2], [3, 4]])]  # one cell changed
        assert r3_single_pixel_diff(actuals, inputs) == 1.0

    def test_does_not_fire_when_two_pixels_change(self) -> None:
        inputs = [_g([[1, 2], [3, 4]])]
        actuals = [_g([[9, 9], [3, 4]])]  # two cells changed
        assert r3_single_pixel_diff(actuals, inputs) == 0.0

    def test_threshold_is_overridable(self) -> None:
        inputs = [_g([[1, 2], [3, 4]])]
        actuals = [_g([[9, 9], [3, 4]])]  # two changes
        assert r3_single_pixel_diff(actuals, inputs, max_diff_pixels=2) == 1.0

    def test_shape_mismatch_does_not_fire(self) -> None:
        inputs = [_g([[1, 2]])]
        actuals = [_g([[1, 2, 3]])]
        assert r3_single_pixel_diff(actuals, inputs) == 0.0


# ---------------------------------------------------------------------------
# R4 — near identity (≥ 95 % of cells match by default)
# ---------------------------------------------------------------------------


class TestR4NearIdentity:
    def test_fires_at_full_match(self) -> None:
        inputs = [_g([[1] * 100])]
        actuals = [_g([[1] * 100])]
        assert r4_near_identity(actuals, inputs) == 1.0

    def test_fires_at_96_percent_match(self) -> None:
        # 96 / 100 cells match — above 95 % default threshold.
        inputs = [_g([[1] * 100])]
        actuals = [_g([[2, 2, 2, 2] + [1] * 96])]
        assert r4_near_identity(actuals, inputs) == 1.0

    def test_does_not_fire_below_threshold(self) -> None:
        # 50 % match.
        inputs = [_g([[1, 1, 1, 1]])]
        actuals = [_g([[1, 1, 9, 9]])]
        assert r4_near_identity(actuals, inputs) == 0.0

    def test_threshold_is_overridable(self) -> None:
        inputs = [_g([[1, 1, 1, 1]])]
        actuals = [_g([[1, 1, 9, 9]])]
        assert r4_near_identity(actuals, inputs, threshold=0.5) == 1.0


# ---------------------------------------------------------------------------
# R5 — output unchanged across demos
# ---------------------------------------------------------------------------


class TestR5OutputUnchangedAcrossDemos:
    def test_fires_when_all_demos_have_identical_output(self) -> None:
        actuals = [_g([[1, 2]]), _g([[1, 2]]), _g([[1, 2]])]
        assert r5_output_unchanged_across_demos(actuals) == 1.0

    def test_does_not_fire_when_outputs_differ(self) -> None:
        actuals = [_g([[1, 2]]), _g([[3, 4]])]
        assert r5_output_unchanged_across_demos(actuals) == 0.0

    def test_does_not_fire_with_single_demo(self) -> None:
        # With only 1 demo there's nothing to compare across.
        actuals = [_g([[1, 2]])]
        assert r5_output_unchanged_across_demos(actuals) == 0.0


# ---------------------------------------------------------------------------
# Aggregate triviality_score
# ---------------------------------------------------------------------------


class TestTrivialityScoreAggregate:
    def test_legitimate_program_scores_high(self) -> None:
        # Legitimate transformation: rotate90.
        inputs = [_g([[1, 2], [3, 4]])]
        actuals = [_g([[3, 1], [4, 2]])]
        expecteds = [_g([[3, 1], [4, 2]])]
        score = triviality_score(actuals, expecteds, inputs)
        # No rule fires → score = 1.0 (fully non-trivial).
        assert score == 1.0

    def test_identity_program_scores_zero(self) -> None:
        # Trivial: actual == input on every demo. R1 fires + R4 fires.
        inputs = [_g([[1, 2], [3, 4]])]
        actuals = [_g([[1, 2], [3, 4]])]  # identity
        expecteds = [_g([[3, 1], [4, 2]])]  # what the task wanted
        score = triviality_score(actuals, expecteds, inputs)
        assert score == 0.0

    def test_constant_output_scores_zero(self) -> None:
        # Trivial: the candidate emits a single-color grid.
        inputs = [_g([[1, 2], [3, 4]])]
        actuals = [_g([[7, 7], [7, 7]])]
        expecteds = [_g([[3, 1], [4, 2]])]
        score = triviality_score(actuals, expecteds, inputs)
        assert score == 0.0

    def test_one_demo_fixed_output_scores_zero_with_two_demos(self) -> None:
        # The candidate emits a fixed output regardless of input —
        # R5 fires because both demos got identical outputs.
        inputs = [_g([[1, 2]]), _g([[5, 6]])]
        actuals = [_g([[9, 9]]), _g([[9, 9]])]  # identical outputs
        expecteds = [_g([[2, 1]]), _g([[6, 5]])]
        score = triviality_score(actuals, expecteds, inputs)
        assert score == 0.0

    def test_empty_actuals_score_one(self) -> None:
        # Pathological edge case — no observations to judge.
        assert triviality_score([], [], []) == 1.0

    def test_plan_acceptance_50_trivial_examples(self) -> None:
        # Plan acceptance criterion: 50 trivial programs all score
        # ≤ 0.3. Build a synthetic 50-program corpus mixing R1/R2/R5
        # cases and check the aggregate stays well under the bar.
        trivial_count = 0
        for i in range(50):
            inputs = [_g([[i, i + 1]])]
            if i % 3 == 0:
                actuals = list(inputs)  # R1: identity
            elif i % 3 == 1:
                actuals = [_g([[7, 7]])]  # R2: constant
            else:
                actuals = [_g([[i, i + 1]])]  # R1 again (identity-shaped)
            expecteds = [_g([[i + 1, i]])]
            if triviality_score(actuals, expecteds, inputs) <= 0.3:
                trivial_count += 1
        assert trivial_count == 50, f"only {trivial_count}/50 trivial programs hit the 0.3 cutoff"
