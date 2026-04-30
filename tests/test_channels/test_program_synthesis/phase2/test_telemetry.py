# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Phase-2 telemetry counter tests (spec v1.4 §11)."""

from __future__ import annotations

from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.observability.metrics import Registry
from cognithor.channels.program_synthesis.phase2 import phase2_counters


class TestPhase2Counters:
    def test_returns_four_named_counters(self) -> None:
        c = phase2_counters(registry=Registry())
        assert set(c) == {
            "refiner_mode_total",
            "refiner_hybrid_winner_total",
            "refiner_mode_hysteresis_held_total",
            "structural_abstraction_token_total",
        }

    def test_counter_names_match_spec_prometheus_strings(self) -> None:
        registry = Registry()
        phase2_counters(registry=registry)
        # ``Registry.snapshot().counters`` is name → label-dict, so the
        # spec-mandated Prometheus names must be top-level keys.
        emitted = set(registry.snapshot().counters)
        assert "cognithor_synthesis_refiner_mode_total" in emitted
        assert "cognithor_synthesis_refiner_hybrid_winner_total" in emitted
        assert "cognithor_synthesis_refiner_mode_hysteresis_held_total" in emitted
        assert "cognithor_synthesis_structural_abstraction_token_total" in emitted

    def test_counters_are_independent_per_registry(self) -> None:
        # Two registries → two distinct sets of counters.
        c1 = phase2_counters(registry=Registry())
        c2 = phase2_counters(registry=Registry())
        c1["refiner_mode_total"].inc(1.0, mode="full_llm")
        # c2 unchanged.
        assert c2["refiner_mode_total"] is not c1["refiner_mode_total"]

    def test_inc_increments_and_partitions_by_label(self) -> None:
        registry = Registry()
        c = phase2_counters(registry=registry)
        c["refiner_mode_total"].inc(1.0, mode="full_llm")
        c["refiner_mode_total"].inc(1.0, mode="hybrid")
        c["refiner_mode_total"].inc(1.0, mode="hybrid")
        target = registry.snapshot().counters["cognithor_synthesis_refiner_mode_total"]
        assert target[(("mode", "full_llm"),)] == 1.0
        assert target[(("mode", "hybrid"),)] == 2.0

    def test_default_registry_used_when_none_passed(self) -> None:
        # Spec: should bind on the default registry without an
        # explicit argument so the channel can call it once at boot.
        c = phase2_counters()
        assert "refiner_mode_total" in c
        assert "structural_abstraction_token_total" in c
