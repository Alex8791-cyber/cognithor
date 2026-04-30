# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""α-mixer + sample-size-dampening tests (spec v1.4 §4.4 + §4.4.4)."""

from __future__ import annotations

import pytest

from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.phase2 import (
    DEFAULT_PHASE2_CONFIG,
    Phase2Config,
    alpha_bounds,
    apply_sample_size_dampening,
    mix_alpha,
)


class TestAlphaBounds:
    def test_default_bounds_match_spec_quarter_to_eighty_five(self) -> None:
        # Spec §4.4.4 verbatim: α ∈ [0.25, 0.85] under the default
        # entropy [0.5, 0.85] × performance [0.5, 1.0] product.
        lo, hi = alpha_bounds()
        assert lo == 0.25
        assert hi == 0.85

    def test_bounds_track_config_override(self) -> None:
        config = Phase2Config(
            alpha_entropy_lower=0.4,
            alpha_entropy_upper=0.7,
            alpha_performance_lower=0.5,
            alpha_performance_upper=0.9,
        )
        lo, hi = alpha_bounds(config=config)
        assert lo == 0.4 * 0.5
        assert hi == 0.7 * 0.9


class TestMixAlphaInRange:
    def test_default_floor(self) -> None:
        # Both factors at their minimums → spec floor.
        assert mix_alpha(0.5, 0.5) == 0.25

    def test_default_ceiling(self) -> None:
        # Both at their maximums → spec ceiling.
        assert mix_alpha(0.85, 1.0) == 0.85

    def test_mid_value(self) -> None:
        # 0.7 · 0.8 = 0.56
        assert abs(mix_alpha(0.7, 0.8) - 0.56) < 1e-9


class TestMixAlphaClamping:
    """Inputs outside the spec bands clamp to the band edges."""

    def test_below_entropy_floor_clamps_up(self) -> None:
        # α_entropy = 0.2 < 0.5 lower bound → clamps to 0.5.
        # Then 0.5 · 0.7 = 0.35.
        assert mix_alpha(0.2, 0.7) == 0.5 * 0.7

    def test_above_entropy_ceiling_clamps_down(self) -> None:
        # α_entropy = 0.95 > 0.85 → clamps to 0.85.
        assert mix_alpha(0.95, 0.7) == 0.85 * 0.7

    def test_above_performance_ceiling_clamps_down(self) -> None:
        # α_performance = 1.2 > 1.0 → clamps to 1.0.
        assert mix_alpha(0.7, 1.2) == 0.7 * 1.0

    def test_negative_factor_clamps_to_lower(self) -> None:
        assert mix_alpha(-0.5, 0.7) == 0.5 * 0.7


class TestSampleSizeDampening:
    """Spec §4.4 — n / (n + n0) dampening."""

    def test_zero_samples_returns_zero(self) -> None:
        assert apply_sample_size_dampening(0.9, 0) == 0.0

    def test_n_equals_n0_dampens_to_half(self) -> None:
        # Default n0 = 4. At n = 4: factor = 4 / 8 = 0.5.
        assert apply_sample_size_dampening(0.9, 4) == 0.45

    def test_n_well_above_n0_approaches_unity(self) -> None:
        # n=400 → 400/404 ≈ 0.99 → result ≈ 0.891.
        result = apply_sample_size_dampening(0.9, 400)
        assert 0.89 < result < 0.9

    def test_large_n0_dampens_more(self) -> None:
        # Bigger n0 means stronger dampening at low n.
        config = Phase2Config(sample_size_dampening_n0=100)
        # At n=4 with n0=100: factor = 4/104 ≈ 0.038 → result ≈ 0.034.
        result = apply_sample_size_dampening(0.9, 4, config=config)
        assert 0.03 < result < 0.04

    def test_base_confidence_outside_unit_raises(self) -> None:
        with pytest.raises(ValueError, match="base_confidence must be"):
            apply_sample_size_dampening(1.5, 4)
        with pytest.raises(ValueError, match="base_confidence must be"):
            apply_sample_size_dampening(-0.1, 4)

    def test_negative_n_samples_raises(self) -> None:
        with pytest.raises(ValueError, match="n_samples must be"):
            apply_sample_size_dampening(0.5, -1)


class TestConfigInvariants:
    """Spec §4.4.4 — α-bands themselves must validate at construction."""

    def test_bad_alpha_entropy_band_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha_entropy"):
            Phase2Config(alpha_entropy_lower=0.7, alpha_entropy_upper=0.5)

    def test_alpha_band_outside_unit_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha_performance"):
            Phase2Config(alpha_performance_lower=-0.1)
        with pytest.raises(ValueError, match="alpha_performance"):
            Phase2Config(alpha_performance_upper=1.5)

    def test_dampening_n0_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="dampening_n0"):
            Phase2Config(sample_size_dampening_n0=0)


class TestDefaultConfigUsedWhenNonePassed:
    def test_mix_alpha_defaults_match_explicit_default(self) -> None:
        assert mix_alpha(0.7, 0.8) == mix_alpha(0.7, 0.8, config=DEFAULT_PHASE2_CONFIG)

    def test_dampening_defaults_match_explicit_default(self) -> None:
        assert apply_sample_size_dampening(0.9, 4) == apply_sample_size_dampening(
            0.9, 4, config=DEFAULT_PHASE2_CONFIG
        )
