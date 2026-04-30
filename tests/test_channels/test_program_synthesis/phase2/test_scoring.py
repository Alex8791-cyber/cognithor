# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Verifier score-aggregation tests (Sprint-1 plan task 8, spec §7.2)."""

from __future__ import annotations

import pytest

from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.phase2 import (
    DEFAULT_PHASE2_CONFIG,
    Phase2Config,
    VerifierScoreInputs,
    VerifierScoreWeights,
    aggregate_verifier_score,
    load_heuristics,
)

# ---------------------------------------------------------------------------
# VerifierScoreWeights — invariants
# ---------------------------------------------------------------------------


class TestVerifierScoreWeights:
    def test_default_matches_spec_anchors(self) -> None:
        w = VerifierScoreWeights()
        assert w.demo_pass_rate == 0.55
        assert w.partial_pixel_match == 0.13
        assert w.invariants_satisfied == 0.08
        # Spec §17 D-criterion binding: 12 % triviality + 12 % suspicion.
        assert w.triviality_score == 0.12
        assert w.suspicion_score == 0.12

    def test_default_sums_to_one(self) -> None:
        w = VerifierScoreWeights()
        total = (
            w.demo_pass_rate
            + w.partial_pixel_match
            + w.invariants_satisfied
            + w.triviality_score
            + w.suspicion_score
        )
        assert abs(total - 1.0) < 1e-9

    def test_construction_rejects_non_unit_sum(self) -> None:
        with pytest.raises(ValueError, match="must sum to 1.0"):
            VerifierScoreWeights(
                demo_pass_rate=0.5,
                partial_pixel_match=0.1,
                invariants_satisfied=0.1,
                triviality_score=0.1,
                suspicion_score=0.1,
            )

    def test_construction_rejects_out_of_range_weight(self) -> None:
        with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
            VerifierScoreWeights(
                demo_pass_rate=1.5,
                partial_pixel_match=0.0,
                invariants_satisfied=0.0,
                triviality_score=0.0,
                suspicion_score=0.0,
            )


# ---------------------------------------------------------------------------
# aggregate_verifier_score — math
# ---------------------------------------------------------------------------


class TestAggregateVerifierScore:
    def test_perfect_inputs_yield_one(self) -> None:
        inputs = VerifierScoreInputs(
            demo_pass_rate=1.0,
            partial_pixel_match=1.0,
            invariants_satisfied=1.0,
            triviality_score=1.0,
            suspicion_score=1.0,
        )
        score = aggregate_verifier_score(inputs)
        assert abs(score - 1.0) < 1e-9

    def test_zero_inputs_yield_zero(self) -> None:
        inputs = VerifierScoreInputs(
            demo_pass_rate=0.0,
            partial_pixel_match=0.0,
            invariants_satisfied=0.0,
            triviality_score=0.0,
            suspicion_score=0.0,
        )
        score = aggregate_verifier_score(inputs)
        assert score == 0.0

    def test_demo_pass_only_weighted_at_55_percent(self) -> None:
        inputs = VerifierScoreInputs(
            demo_pass_rate=1.0,
            partial_pixel_match=0.0,
            invariants_satisfied=0.0,
            triviality_score=0.0,
            suspicion_score=0.0,
        )
        score = aggregate_verifier_score(inputs)
        assert abs(score - 0.55) < 1e-9

    def test_triviality_plus_suspicion_combined_24_percent(self) -> None:
        # Spec §17 binding: 12 % + 12 %.
        inputs = VerifierScoreInputs(
            demo_pass_rate=0.0,
            partial_pixel_match=0.0,
            invariants_satisfied=0.0,
            triviality_score=1.0,
            suspicion_score=1.0,
        )
        score = aggregate_verifier_score(inputs)
        assert abs(score - 0.24) < 1e-9

    def test_input_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
            VerifierScoreInputs(
                demo_pass_rate=1.5,
                partial_pixel_match=0.0,
                invariants_satisfied=0.0,
                triviality_score=0.0,
                suspicion_score=0.0,
            )

    def test_custom_weights_propagate(self) -> None:
        # Push everything into demo_pass_rate.
        config = Phase2Config(
            verifier_score_weights=VerifierScoreWeights(
                demo_pass_rate=1.0,
                partial_pixel_match=0.0,
                invariants_satisfied=0.0,
                triviality_score=0.0,
                suspicion_score=0.0,
            )
        )
        inputs = VerifierScoreInputs(
            demo_pass_rate=0.7,
            partial_pixel_match=1.0,
            invariants_satisfied=1.0,
            triviality_score=1.0,
            suspicion_score=1.0,
        )
        # Only demo_pass_rate carries weight → score = 0.7.
        score = aggregate_verifier_score(inputs, config=config)
        assert abs(score - 0.7) < 1e-9

    def test_default_config_used_when_none_passed(self) -> None:
        inputs = VerifierScoreInputs(
            demo_pass_rate=0.5,
            partial_pixel_match=0.5,
            invariants_satisfied=0.5,
            triviality_score=0.5,
            suspicion_score=0.5,
        )
        # Half-everything with default weights → 0.5.
        assert aggregate_verifier_score(inputs) == aggregate_verifier_score(
            inputs, config=DEFAULT_PHASE2_CONFIG
        )
        assert abs(aggregate_verifier_score(inputs) - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# YAML loader → Phase2Config.verifier_score_weights round-trip
# ---------------------------------------------------------------------------


class TestYamlRoundTrip:
    def test_loaded_weights_match_yaml(self) -> None:
        cfg = load_heuristics().phase2_config
        w = cfg.verifier_score_weights
        # Match the spec-anchored YAML values.
        assert w.demo_pass_rate == 0.55
        assert w.partial_pixel_match == 0.13
        assert w.invariants_satisfied == 0.08
        assert w.triviality_score == 0.12
        assert w.suspicion_score == 0.12

    def test_loaded_weights_sum_to_one(self) -> None:
        cfg = load_heuristics().phase2_config
        w = cfg.verifier_score_weights
        total = (
            w.demo_pass_rate
            + w.partial_pixel_match
            + w.invariants_satisfied
            + w.triviality_score
            + w.suspicion_score
        )
        assert abs(total - 1.0) < 1e-9
