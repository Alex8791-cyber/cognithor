# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""α-Controller + Prior-Performance-Tracker tests (plan task 6)."""

from __future__ import annotations

import pytest

from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.phase2 import (
    DEFAULT_PHASE2_CONFIG,
    AlphaController,
    Phase2Config,
    PriorObservation,
    PriorPerformanceTracker,
    load_heuristics,
)

# ---------------------------------------------------------------------------
# PriorObservation invariants
# ---------------------------------------------------------------------------


class TestPriorObservation:
    def test_construction_round_trip(self) -> None:
        o = PriorObservation(llm_success=0.7, symbolic_success=0.4)
        assert o.llm_success == 0.7
        assert o.symbolic_success == 0.4

    def test_out_of_range_llm_rejected(self) -> None:
        with pytest.raises(ValueError, match="llm_success"):
            PriorObservation(llm_success=1.5, symbolic_success=0.5)

    def test_negative_symbolic_rejected(self) -> None:
        with pytest.raises(ValueError, match="symbolic_success"):
            PriorObservation(llm_success=0.5, symbolic_success=-0.1)


# ---------------------------------------------------------------------------
# PriorPerformanceTracker — sliding window
# ---------------------------------------------------------------------------


class TestPriorPerformanceTracker:
    def test_starts_cold(self) -> None:
        t = PriorPerformanceTracker()
        assert t.is_warm() is False
        assert t.n_observations == 0
        assert t.average_llm_success() == 0.0

    def test_records_and_averages(self) -> None:
        t = PriorPerformanceTracker()
        t.record(PriorObservation(llm_success=0.8, symbolic_success=0.4))
        t.record(PriorObservation(llm_success=0.6, symbolic_success=0.6))
        assert t.is_warm()
        assert t.n_observations == 2
        assert abs(t.average_llm_success() - 0.7) < 1e-9
        assert abs(t.average_symbolic_success() - 0.5) < 1e-9

    def test_window_evicts_oldest(self) -> None:
        # Window size 3 — recording 4 entries drops the first.
        config = Phase2Config(alpha_performance_window=3)
        t = PriorPerformanceTracker(config=config)
        t.record(PriorObservation(llm_success=0.0, symbolic_success=1.0))
        t.record(PriorObservation(llm_success=0.5, symbolic_success=0.5))
        t.record(PriorObservation(llm_success=0.5, symbolic_success=0.5))
        t.record(PriorObservation(llm_success=1.0, symbolic_success=0.0))
        # After eviction, average_llm = (0.5+0.5+1.0)/3 ≈ 0.667.
        assert abs(t.average_llm_success() - 2 / 3) < 1e-9

    def test_reset_clears(self) -> None:
        t = PriorPerformanceTracker()
        t.record(PriorObservation(llm_success=1.0, symbolic_success=1.0))
        t.reset()
        assert t.is_warm() is False
        assert t.n_observations == 0


# ---------------------------------------------------------------------------
# AlphaController — cold-start + hysteresis
# ---------------------------------------------------------------------------


class TestAlphaController:
    def test_cold_start_returns_configured_default(self) -> None:
        c = AlphaController()
        # Default cold_start_alpha = 0.85 — that's exactly the
        # performance_upper bound, so it returns 0.85.
        assert c.alpha_performance() == 0.85

    def test_cold_start_clamped_into_band(self) -> None:
        # If cold-start were above the band ceiling, it'd clamp.
        # Set the band explicitly tighter than the cold-start.
        config = Phase2Config(
            alpha_performance_lower=0.6,
            alpha_performance_upper=0.8,
            alpha_cold_start=0.8,
        )
        c = AlphaController(config=config)
        assert c.alpha_performance() == 0.8

    def test_one_low_observation_does_not_trip_hysteresis(self) -> None:
        c = AlphaController()
        c.observe(PriorObservation(llm_success=0.1, symbolic_success=0.5))
        # Only 1 consecutive low observation; default hysteresis is 5.
        # The controller must NOT have lowered yet.
        assert c.alpha_performance() == DEFAULT_PHASE2_CONFIG.alpha_performance_upper

    def test_consecutive_lows_drop_to_band_floor(self) -> None:
        # Plan acceptance criterion: künstlich-verschlechterter LLM
        # → α sinkt unter 0.4 nach Window-Iterationen. With default
        # hysteresis=5 + alpha_performance_lower=0.5, the controller
        # drops to exactly 0.5 on the 5th consecutive low.
        c = AlphaController()
        for _ in range(5):
            c.observe(PriorObservation(llm_success=0.1, symbolic_success=0.5))
        assert c.alpha_performance() == 0.5

    def test_a_good_observation_resets_consecutive_counter(self) -> None:
        c = AlphaController()
        # 4 lows…
        for _ in range(4):
            c.observe(PriorObservation(llm_success=0.1, symbolic_success=0.5))
        # …then one good one resets the streak.
        c.observe(PriorObservation(llm_success=0.9, symbolic_success=0.5))
        assert c.consecutive_low_count == 0
        # And the next observation cycle still has 4 to go before drop.
        for _ in range(4):
            c.observe(PriorObservation(llm_success=0.1, symbolic_success=0.5))
        # Still hasn't tripped — only 4 consecutive lows after reset.
        assert c.alpha_performance() == DEFAULT_PHASE2_CONFIG.alpha_performance_upper

    def test_reset_clears_state(self) -> None:
        c = AlphaController()
        for _ in range(5):
            c.observe(PriorObservation(llm_success=0.1, symbolic_success=0.5))
        c.reset()
        assert c.consecutive_low_count == 0
        assert c.tracker.is_warm() is False
        # Cold-start again.
        assert c.alpha_performance() == 0.85

    def test_custom_low_threshold_propagates(self) -> None:
        # 0.5 success counts as "good" by default (above 0.4 threshold).
        # With a tighter threshold of 0.7, it now counts as "low".
        c = AlphaController(low_llm_threshold=0.7)
        for _ in range(5):
            c.observe(PriorObservation(llm_success=0.5, symbolic_success=0.5))
        assert c.alpha_performance() == DEFAULT_PHASE2_CONFIG.alpha_performance_lower


# ---------------------------------------------------------------------------
# YAML round-trip
# ---------------------------------------------------------------------------


class TestYamlRoundTrip:
    def test_loaded_alpha_fields_match_yaml(self) -> None:
        cfg = load_heuristics().phase2_config
        assert cfg.alpha_hysteresis_iterations == 5
        assert cfg.alpha_performance_window == 10
        assert cfg.alpha_cold_start == 0.85
