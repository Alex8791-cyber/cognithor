# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Telemetry + audit-trail tests (spec §13.2 + §17)."""

from __future__ import annotations

import json

import pytest

from cognithor.channels.program_synthesis.observability import (
    CANDIDATES_EXPLORED_BUCKETS,
    DURATION_SECONDS_BUCKETS,
    PROGRAM_DEPTH_BUCKETS,
    PROGRAM_SIZE_BUCKETS,
    AuditEntry,
    AuditTrail,
    Counter,
    Histogram,
    Registry,
    audit_entry_for,
    standard_counters,
    standard_histograms,
)

# ---------------------------------------------------------------------------
# Counter
# ---------------------------------------------------------------------------


class TestCounter:
    def test_starts_at_zero(self) -> None:
        c = Counter("test")
        assert c.value() == 0.0

    def test_inc_default_one(self) -> None:
        c = Counter("test")
        c.inc()
        c.inc()
        assert c.value() == 2.0

    def test_inc_custom_amount(self) -> None:
        c = Counter("test")
        c.inc(0.5)
        c.inc(0.5)
        assert c.value() == 1.0

    def test_negative_inc_rejected(self) -> None:
        c = Counter("test")
        with pytest.raises(ValueError, match="non-negative"):
            c.inc(-1.0)

    def test_labels_isolate_counts(self) -> None:
        c = Counter("test")
        c.inc(status="success")
        c.inc(status="success")
        c.inc(status="error")
        assert c.value(status="success") == 2.0
        assert c.value(status="error") == 1.0

    def test_snapshot_contains_all_label_combos(self) -> None:
        c = Counter("test")
        c.inc(status="success", domain="arc_agi_3")
        c.inc(status="error", domain="arc_agi_3")
        snap = c.snapshot()
        assert len(snap) == 2

    def test_reset_clears(self) -> None:
        c = Counter("test")
        c.inc()
        c.reset()
        assert c.value() == 0.0


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------


class TestHistogram:
    def test_observe_increments_correct_buckets(self) -> None:
        h = Histogram("test", buckets=(1.0, 5.0, 10.0))
        h.observe(0.5)  # ≤ 1.0 → buckets [1.0, 5.0, 10.0] all incremented
        h.observe(2.0)  # ≤ 5.0, not 1.0
        h.observe(20.0)  # > 10.0 → no bucket incremented (caught in +Inf)
        snap = h.snapshot()
        assert snap.counts == (1, 2, 2)
        assert snap.sum_value == 22.5
        assert snap.count == 3

    def test_buckets_must_be_ascending(self) -> None:
        with pytest.raises(ValueError, match="ascending"):
            Histogram("test", buckets=(5.0, 1.0))

    def test_empty_buckets_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Histogram("test", buckets=())

    def test_snapshot_immutable(self) -> None:
        from dataclasses import FrozenInstanceError

        h = Histogram("test", buckets=(1.0,))
        h.observe(0.5)
        snap = h.snapshot()
        with pytest.raises(FrozenInstanceError):
            snap.count = 99  # type: ignore[misc]

    def test_reset_clears_state(self) -> None:
        h = Histogram("test", buckets=(1.0,))
        h.observe(0.5)
        h.reset()
        snap = h.snapshot()
        assert snap.count == 0
        assert snap.sum_value == 0.0
        assert snap.counts == (0,)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_counter_idempotent(self) -> None:
        r = Registry()
        a = r.counter("foo")
        b = r.counter("foo")
        # Same instance — multiple registrations don't shadow.
        assert a is b

    def test_histogram_idempotent(self) -> None:
        r = Registry()
        a = r.histogram("foo", buckets=(1.0,))
        b = r.histogram("foo", buckets=(1.0,))
        assert a is b

    def test_histogram_bucket_mismatch_rejected(self) -> None:
        r = Registry()
        r.histogram("foo", buckets=(1.0,))
        with pytest.raises(ValueError, match="different buckets"):
            r.histogram("foo", buckets=(2.0,))

    def test_names_returns_sorted(self) -> None:
        r = Registry()
        r.counter("zebra")
        r.histogram("alpha", buckets=(1.0,))
        r.counter("middle")
        assert r.names() == ("alpha", "middle", "zebra")

    def test_snapshot_includes_both_kinds(self) -> None:
        r = Registry()
        r.counter("c").inc()
        r.histogram("h", buckets=(1.0,)).observe(0.5)
        snap = r.snapshot()
        assert "c" in snap.counters
        assert "h" in snap.histograms

    def test_reset_clears_all(self) -> None:
        r = Registry()
        c = r.counter("c")
        h = r.histogram("h", buckets=(1.0,))
        c.inc()
        h.observe(0.5)
        r.reset()
        assert c.value() == 0.0
        assert h.snapshot().count == 0


class TestStandardMetrics:
    def test_standard_counters_match_spec(self) -> None:
        r = Registry()
        cs = standard_counters(r)
        assert "synthesis_requests_total" in cs
        assert "sandbox_violations_total" in cs
        assert "cache_hits_total" in cs
        assert "cache_misses_total" in cs
        assert "dsl_primitive_uses_total" in cs

    def test_standard_histograms_match_spec(self) -> None:
        r = Registry()
        hs = standard_histograms(r)
        assert "synthesis_duration_seconds" in hs
        assert "candidates_explored" in hs
        assert "program_depth" in hs
        assert "program_size" in hs

    def test_bucket_constants_match_spec(self) -> None:
        # Spec §17.2 bucket definitions.
        assert DURATION_SECONDS_BUCKETS == (0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0)
        assert CANDIDATES_EXPLORED_BUCKETS == (100, 1_000, 10_000, 100_000, 1_000_000)
        assert PROGRAM_DEPTH_BUCKETS == (1, 2, 3, 4, 5, 6)
        assert PROGRAM_SIZE_BUCKETS == (1, 5, 10, 20, 50)


# ---------------------------------------------------------------------------
# Audit Trail
# ---------------------------------------------------------------------------


class TestAuditEntry:
    def test_to_json_canonical_keys_sorted(self) -> None:
        entry = audit_entry_for(
            actor="planner@cognithor",
            capability="pse:synthesize",
            spec_hash="sha256:abc",
            budget={"max_depth": 4},
            result_status="success",
            program_hash="sha256:def",
            duration_ms=1234,
            candidates_explored=5678,
        )
        body = entry.to_json()
        # Keys appear in sorted order.
        decoded = json.loads(body)
        assert list(decoded.keys()) == sorted(decoded.keys())

    def test_to_json_no_nan(self) -> None:
        # Spec mandates allow_nan=False — any non-finite slips fail loudly.
        entry = AuditEntry(
            ts="2026-04-30T00:00:00.000Z",
            actor="x",
            capability="pse:synthesize",
            spec_hash="sha256:0",
            budget={"v": float("nan")},
            result_status="success",
            program_hash=None,
            duration_ms=0,
            candidates_explored=0,
        )
        with pytest.raises(ValueError):
            entry.to_json()

    def test_extra_round_trip(self) -> None:
        entry = audit_entry_for(
            actor="x",
            capability="pse:synthesize",
            spec_hash="sha256:0",
            budget={},
            result_status="success",
            program_hash=None,
            duration_ms=0,
            candidates_explored=0,
            extra={"source": "fast_path", "version": "1.2.0"},
        )
        body = json.loads(entry.to_json())
        assert body["extra"]["source"] == "fast_path"
        assert body["extra"]["version"] == "1.2.0"

    def test_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        entry = audit_entry_for(
            actor="x",
            capability="pse:synthesize",
            spec_hash="sha256:0",
            budget={},
            result_status="success",
            program_hash=None,
            duration_ms=0,
            candidates_explored=0,
        )
        with pytest.raises(FrozenInstanceError):
            entry.actor = "tampered"  # type: ignore[misc]


class TestAuditTrail:
    def _entry(self, *, status: str = "success") -> AuditEntry:
        return audit_entry_for(
            actor="planner@cognithor",
            capability="pse:synthesize",
            spec_hash="sha256:abc",
            budget={"max_depth": 4},
            result_status=status,
            program_hash="sha256:def",
            duration_ms=1234,
            candidates_explored=5678,
        )

    def test_initial_genesis_hash(self) -> None:
        trail = AuditTrail()
        assert trail.latest_hash() == AuditTrail.GENESIS_HASH
        assert len(trail) == 0

    def test_emit_advances_hash(self) -> None:
        trail = AuditTrail()
        h1 = trail.emit(self._entry())
        assert h1 != AuditTrail.GENESIS_HASH
        h2 = trail.emit(self._entry(status="partial"))
        assert h2 != h1

    def test_chain_verify_passes_clean(self) -> None:
        trail = AuditTrail()
        for status in ("success", "partial", "no_solution"):
            trail.emit(self._entry(status=status))
        assert trail.verify()

    def test_chain_verify_fails_on_tamper(self) -> None:
        trail = AuditTrail()
        trail.emit(self._entry())
        trail.emit(self._entry(status="error"))
        # Tamper: replace an entry with a different one in-place via
        # the internal list. (Real attackers would re-write the
        # JSON-line file; here we simulate that.)
        trail._entries[0] = self._entry(status="success_TAMPERED")  # type: ignore[index]
        assert not trail.verify()

    def test_on_emit_callback_invoked(self) -> None:
        seen: list[tuple[AuditEntry, str]] = []
        trail = AuditTrail(on_emit=lambda e, h: seen.append((e, h)))
        trail.emit(self._entry())
        trail.emit(self._entry(status="error"))
        assert len(seen) == 2
        # Hash returned to caller matches what the callback received.
        assert seen[1][1] == trail.latest_hash()

    def test_reset_returns_to_genesis(self) -> None:
        trail = AuditTrail()
        trail.emit(self._entry())
        trail.reset()
        assert trail.latest_hash() == AuditTrail.GENESIS_HASH
        assert len(trail) == 0

    def test_entries_returns_tuple_view(self) -> None:
        trail = AuditTrail()
        trail.emit(self._entry())
        entries = trail.entries()
        assert isinstance(entries, tuple)
        assert len(entries) == 1
