# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""F2 — Refiner mode-selection + hysteresis (spec v1.4 §6.5.2)."""

from __future__ import annotations

from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.phase2 import Phase2Config
from cognithor.channels.program_synthesis.refiner import RefinerModeController


class TestThreeZoneSelection:
    """Spec v1.4 §6.5.2 — three α zones."""

    def test_zone_full_llm_at_alpha_05(self) -> None:
        c = RefinerModeController()
        assert c.select_mode(0.5) == "full_llm"

    def test_zone_full_llm_at_lower_boundary(self) -> None:
        c = RefinerModeController()
        # exactly the lower bound of zone 1 → full_llm (≥)
        assert c.select_mode(0.45) == "full_llm"

    def test_zone_hybrid_at_alpha_04(self) -> None:
        c = RefinerModeController()
        assert c.select_mode(0.40) == "hybrid"

    def test_zone_hybrid_at_lower_boundary(self) -> None:
        c = RefinerModeController()
        # exactly the lower bound of zone 2 → hybrid (≥)
        assert c.select_mode(0.35) == "hybrid"

    def test_zone_symbolic_below_zone3_upper(self) -> None:
        c = RefinerModeController()
        assert c.select_mode(0.34) == "symbolic"

    def test_zone_symbolic_at_alpha_zero(self) -> None:
        c = RefinerModeController()
        assert c.select_mode(0.0) == "symbolic"


class TestHysteresisHoldsAtBoundary:
    """Spec v1.4 §6.5.2 — once-chosen mode is sticky for ``window`` calls."""

    def test_alpha_oscillation_at_boundary_holds_first_mode(self) -> None:
        # Spec test: α-Sequenz [0.46, 0.44, 0.46, 0.44] with window=3
        # → bleibt im ersten Mode (full_llm) für mind. 3 Calls.
        c = RefinerModeController()
        assert c.select_mode(0.46) == "full_llm"
        assert c.select_mode(0.44) == "full_llm"  # held by hysteresis
        assert c.select_mode(0.46) == "full_llm"  # in zone again
        # 3 calls accumulated; next can flip if a different mode is proposed.
        assert c.select_mode(0.44) == "hybrid"

    def test_hysteresis_counter_increments_on_held_change(self) -> None:
        c = RefinerModeController()
        c.select_mode(0.5)  # full_llm, fresh
        c.select_mode(0.4)  # would be hybrid; held → full_llm
        assert c.hysteresis_holds_total == 1
        c.select_mode(0.4)  # would be hybrid; still held
        assert c.hysteresis_holds_total == 2

    def test_hysteresis_does_not_block_same_mode(self) -> None:
        # Returning the same mode should never bump the hold counter.
        c = RefinerModeController()
        c.select_mode(0.5)
        c.select_mode(0.6)
        c.select_mode(0.7)
        assert c.hysteresis_holds_total == 0
        assert c.calls_in_current_mode == 3

    def test_window_can_be_overridden_via_config(self) -> None:
        config = Phase2Config(refiner_hysteresis_window=1)
        c = RefinerModeController(config=config)
        assert c.select_mode(0.5) == "full_llm"
        # window=1 means *one* call sticks, then a different proposal flips.
        assert c.select_mode(0.3) == "symbolic"


class TestResetClearsState:
    def test_reset_returns_controller_to_fresh(self) -> None:
        c = RefinerModeController()
        c.select_mode(0.5)
        c.select_mode(0.5)
        c.reset()
        assert c.current_mode is None
        assert c.calls_in_current_mode == 0
        assert c.hysteresis_holds_total == 0


class TestConfigOverride:
    """Spec Sprint-1 rule: thresholds are config, not constants."""

    def test_custom_thresholds_change_zone_boundaries(self) -> None:
        # Tighten the graybereich to [0.40, 0.50].
        config = Phase2Config(
            repair_alpha_zone1_lower=0.50,
            repair_alpha_zone3_upper=0.40,
        )
        c = RefinerModeController(config=config)
        # α=0.45 used to be full_llm; under the tighter bounds it's hybrid.
        assert c.select_mode(0.45) == "hybrid"
        # α=0.50 hits the new lower bound of zone 1.
        c.reset()
        assert c.select_mode(0.50) == "full_llm"
