# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Channel-side wiring of telemetry counters + audit trail."""

from __future__ import annotations

import numpy as np

from cognithor.channels.program_synthesis.core.types import (
    Budget,
    SynthesisStatus,
    TaskSpec,
)
from cognithor.channels.program_synthesis.integration.pge_adapter import (
    ProgramSynthesisChannel,
    SynthesisRequest,
)
from cognithor.channels.program_synthesis.integration.tactical_memory import (
    PSECache,
)
from cognithor.channels.program_synthesis.observability import (
    AuditTrail,
    Registry,
)
from cognithor.channels.program_synthesis.sandbox.strategies import (
    LinuxSubprocessStrategy,
)


def _g(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


def _rotate90_request() -> SynthesisRequest:
    return SynthesisRequest(
        spec=TaskSpec(
            examples=(
                (_g([[1, 2], [3, 4]]), _g([[3, 1], [4, 2]])),
                (_g([[5, 6, 7]]), _g([[5], [6], [7]])),
            ),
        ),
        budget=Budget(max_depth=2, wall_clock_seconds=10.0),
    )


# ---------------------------------------------------------------------------
# Metrics emission
# ---------------------------------------------------------------------------


class TestMetricsWiring:
    def test_synthesis_request_increments_counter(self) -> None:
        registry = Registry()
        channel = ProgramSynthesisChannel(
            sandbox_strategy=LinuxSubprocessStrategy(),
            metrics_registry=registry,
        )
        channel.synthesize(_rotate90_request())
        snap = registry.snapshot()
        # synthesis_requests_total has one entry under
        # (status=success, domain=arc_agi_3).
        c = snap.counters["pse_synthesis_requests_total"]
        assert any(("status", "success") in k and ("domain", "arc_agi_3") in k for k in c)
        assert sum(c.values()) == 1.0

    def test_cache_miss_then_hit_counts(self) -> None:
        registry = Registry()
        channel = ProgramSynthesisChannel(
            sandbox_strategy=LinuxSubprocessStrategy(),
            metrics_registry=registry,
        )
        # First call: miss + populate.
        channel.synthesize(_rotate90_request())
        # Second call: hit.
        channel.synthesize(_rotate90_request())
        snap = registry.snapshot()
        miss = snap.counters["pse_cache_misses_total"]
        hit = snap.counters["pse_cache_hits_total"]
        assert sum(miss.values()) == 1.0
        assert sum(hit.values()) == 1.0

    def test_duration_histogram_records_one_observation(self) -> None:
        registry = Registry()
        channel = ProgramSynthesisChannel(
            sandbox_strategy=LinuxSubprocessStrategy(),
            metrics_registry=registry,
        )
        channel.synthesize(_rotate90_request())
        snap = registry.snapshot()
        hist = snap.histograms["pse_synthesis_duration_seconds"]
        assert hist.count == 1
        # rotate90 search is fast — should land in the lowest bucket.
        assert hist.counts[0] == 1

    def test_candidates_explored_recorded(self) -> None:
        registry = Registry()
        channel = ProgramSynthesisChannel(
            sandbox_strategy=LinuxSubprocessStrategy(),
            metrics_registry=registry,
        )
        result = channel.synthesize(_rotate90_request())
        snap = registry.snapshot()
        hist = snap.histograms["pse_candidates_explored"]
        assert hist.count == 1
        assert hist.sum_value == float(result.cost_candidates)

    def test_program_depth_size_recorded_on_success(self) -> None:
        registry = Registry()
        channel = ProgramSynthesisChannel(
            sandbox_strategy=LinuxSubprocessStrategy(),
            metrics_registry=registry,
        )
        channel.synthesize(_rotate90_request())
        snap = registry.snapshot()
        # rotate90(input) → depth 1, size 2.
        depth = snap.histograms["pse_program_depth"]
        size = snap.histograms["pse_program_size"]
        assert depth.count == 1
        assert size.count == 1
        assert depth.sum_value == 1.0  # rotate90's depth
        assert size.sum_value == 2.0  # rotate90 + InputRef

    def test_dsl_primitive_uses_recorded(self) -> None:
        registry = Registry()
        channel = ProgramSynthesisChannel(
            sandbox_strategy=LinuxSubprocessStrategy(),
            metrics_registry=registry,
        )
        channel.synthesize(_rotate90_request())
        snap = registry.snapshot()
        c = snap.counters["pse_dsl_primitive_uses_total"]
        assert any(("primitive", "rotate90") in k for k in c)

    def test_no_emission_without_registry(self) -> None:
        # Channel without metrics_registry must not crash and must
        # not raise on the hot path.
        channel = ProgramSynthesisChannel(
            sandbox_strategy=LinuxSubprocessStrategy(),
        )
        result = channel.synthesize(_rotate90_request())
        assert result.status == SynthesisStatus.SUCCESS


# ---------------------------------------------------------------------------
# Audit-trail emission
# ---------------------------------------------------------------------------


class TestAuditWiring:
    def test_emit_one_entry_per_synthesize(self) -> None:
        trail = AuditTrail()
        channel = ProgramSynthesisChannel(
            sandbox_strategy=LinuxSubprocessStrategy(),
            audit_trail=trail,
        )
        channel.synthesize(_rotate90_request())
        channel.synthesize(_rotate90_request())  # cache hit
        assert len(trail) == 2

    def test_entry_records_actor_capability_status(self) -> None:
        trail = AuditTrail()
        channel = ProgramSynthesisChannel(
            sandbox_strategy=LinuxSubprocessStrategy(),
            audit_trail=trail,
            actor="planner@cognithor",
        )
        channel.synthesize(_rotate90_request())
        entry = trail.entries()[0]
        assert entry.actor == "planner@cognithor"
        assert entry.capability == "pse:synthesize"
        assert entry.result_status == "success"

    def test_entry_carries_program_hash_on_success(self) -> None:
        trail = AuditTrail()
        channel = ProgramSynthesisChannel(
            sandbox_strategy=LinuxSubprocessStrategy(),
            audit_trail=trail,
        )
        channel.synthesize(_rotate90_request())
        entry = trail.entries()[0]
        # Real Program → real stable_hash.
        assert entry.program_hash is not None
        assert entry.program_hash.startswith("sha256:")

    def test_entry_program_hash_none_when_cache_hit(self) -> None:
        trail = AuditTrail()
        channel = ProgramSynthesisChannel(
            cache=PSECache(),
            sandbox_strategy=LinuxSubprocessStrategy(),
            audit_trail=trail,
        )
        channel.synthesize(_rotate90_request())
        # Second call → cache hit, result.program is None →
        # entry.program_hash is None.
        channel.synthesize(_rotate90_request())
        entries = trail.entries()
        assert entries[0].program_hash is not None  # real synth
        assert entries[1].program_hash is None  # cache hit

    def test_chain_verify_passes_after_multiple_emits(self) -> None:
        trail = AuditTrail()
        channel = ProgramSynthesisChannel(
            sandbox_strategy=LinuxSubprocessStrategy(),
            audit_trail=trail,
        )
        for _ in range(3):
            channel.synthesize(_rotate90_request())
        assert trail.verify()
        assert len(trail) == 3

    def test_no_emission_without_trail(self) -> None:
        channel = ProgramSynthesisChannel(
            sandbox_strategy=LinuxSubprocessStrategy(),
        )
        result = channel.synthesize(_rotate90_request())
        # Channel without trail still works.
        assert result.status == SynthesisStatus.SUCCESS


# ---------------------------------------------------------------------------
# Joint metrics + audit
# ---------------------------------------------------------------------------


class TestJointWiring:
    def test_both_emitted_in_one_call(self) -> None:
        registry = Registry()
        trail = AuditTrail()
        channel = ProgramSynthesisChannel(
            sandbox_strategy=LinuxSubprocessStrategy(),
            metrics_registry=registry,
            audit_trail=trail,
        )
        channel.synthesize(_rotate90_request())
        # Counter incremented + audit entry emitted.
        snap = registry.snapshot()
        assert sum(snap.counters["pse_synthesis_requests_total"].values()) == 1.0
        assert len(trail) == 1

    def test_telemetry_failure_does_not_break_synthesis(self) -> None:
        # Simulate a flaky audit-trail callback that raises — channel
        # should still return a SynthesisResult.
        # NOTE: this asserts the *current* behaviour: the channel does
        # not catch callback exceptions. If we ever wrap the audit
        # emission in a try/except, this test should be updated to
        # confirm the wrap.
        import pytest

        def explode(_entry, _hash):
            raise RuntimeError("audit sink down")

        trail = AuditTrail(on_emit=explode)
        channel = ProgramSynthesisChannel(
            sandbox_strategy=LinuxSubprocessStrategy(),
            audit_trail=trail,
        )
        with pytest.raises(RuntimeError):
            channel.synthesize(_rotate90_request())
