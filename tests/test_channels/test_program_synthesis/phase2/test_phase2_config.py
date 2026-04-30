# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Phase-2 config sanity tests (spec v1.4 Sprint-1 rule).

Spec rule: every heuristic constant must be config-driven, not
hardcoded. These tests pin:

* the default values match the spec defaults exactly (so a Sprint-1
  data-driven A/B can be detected as a real change at PR review);
* invariants (zone ordering, multiplier ordering) are enforced at
  construction time, not at first use;
* overriding the config is a frictionless, no-monkeypatch operation.
"""

from __future__ import annotations

import pytest

# Load integration first to avoid the existing PSE import-cycle.
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.phase2.config import (
    DEFAULT_PHASE2_CONFIG,
    Phase2Config,
)


class TestSpecDefaults:
    """The default values are what the spec freezes."""

    def test_high_impact_multiplier_default_three(self) -> None:
        assert DEFAULT_PHASE2_CONFIG.high_impact_multiplier == 3.0

    def test_structural_abstraction_multiplier_default_one_point_five(self) -> None:
        assert DEFAULT_PHASE2_CONFIG.structural_abstraction_multiplier == 1.5

    def test_regular_multiplier_default_one(self) -> None:
        assert DEFAULT_PHASE2_CONFIG.regular_primitive_multiplier == 1.0

    def test_repair_alpha_zone_thresholds_default(self) -> None:
        assert DEFAULT_PHASE2_CONFIG.repair_alpha_zone1_lower == 0.45
        assert DEFAULT_PHASE2_CONFIG.repair_alpha_zone3_upper == 0.35

    def test_hysteresis_window_default_three(self) -> None:
        assert DEFAULT_PHASE2_CONFIG.refiner_hysteresis_window == 3

    def test_reserves_disabled_by_default(self) -> None:
        # Spec §22.4.2: only enable if interaction tests fail.
        assert DEFAULT_PHASE2_CONFIG.enable_argument_quality_factor is False
        assert DEFAULT_PHASE2_CONFIG.enable_few_demos_dampening is False

    def test_default_config_is_frozen(self) -> None:
        # Frozen dataclass — assignment must raise FrozenInstanceError.
        from dataclasses import FrozenInstanceError

        with pytest.raises(FrozenInstanceError):
            DEFAULT_PHASE2_CONFIG.high_impact_multiplier = 99.0  # type: ignore[misc]


class TestInvariantsEnforced:
    def test_zone_thresholds_must_be_strictly_ordered(self) -> None:
        with pytest.raises(ValueError, match="repair α thresholds"):
            Phase2Config(
                repair_alpha_zone1_lower=0.30,
                repair_alpha_zone3_upper=0.40,  # would create a negative graybereich
            )

    def test_zone_thresholds_must_be_in_open_unit(self) -> None:
        with pytest.raises(ValueError, match="repair α thresholds"):
            Phase2Config(repair_alpha_zone3_upper=0.0)
        with pytest.raises(ValueError, match="repair α thresholds"):
            Phase2Config(repair_alpha_zone1_lower=1.0)

    def test_hysteresis_window_must_be_at_least_one(self) -> None:
        with pytest.raises(ValueError, match="hysteresis_window"):
            Phase2Config(refiner_hysteresis_window=0)

    def test_multipliers_must_be_ordered_regular_le_struct_le_high(self) -> None:
        with pytest.raises(ValueError, match="multipliers must be ordered"):
            Phase2Config(
                regular_primitive_multiplier=2.0,
                structural_abstraction_multiplier=1.0,  # below regular
            )

    def test_collapsed_multipliers_allowed_for_ablation(self) -> None:
        # Spec rationale in the docstring: an A/B ablation that
        # collapses the classes (all == 1.0) must construct cleanly.
        cfg = Phase2Config(
            regular_primitive_multiplier=1.0,
            structural_abstraction_multiplier=1.0,
            high_impact_multiplier=1.0,
        )
        assert cfg.regular_primitive_multiplier == 1.0


class TestOverrideErgonomics:
    """Spec rule: override should not require monkeypatching."""

    def test_dataclass_replace_is_supported(self) -> None:
        from dataclasses import replace

        cfg = replace(DEFAULT_PHASE2_CONFIG, high_impact_multiplier=4.5)
        assert cfg.high_impact_multiplier == 4.5
        # Other defaults untouched.
        assert cfg.refiner_hysteresis_window == 3

    def test_constructor_kwargs_supported(self) -> None:
        cfg = Phase2Config(refiner_hysteresis_window=5)
        assert cfg.refiner_hysteresis_window == 5
