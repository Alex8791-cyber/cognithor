"""Tests for the SQLite-backed CloudEscalationStore (TRUST-8 disk persistence)."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from cognithor.security.cloud_escalation import (
    EscalationEvent,
    EscalationReason,
    EscalationSummary,
)
from cognithor.security.cloud_escalation_store import (
    CloudEscalationStore,
    CloudEscalationStoreError,
)


def _utc(
    year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0
) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)


def _event(
    *,
    reason: EscalationReason = EscalationReason.OWNER_OVERRIDE,
    from_backend: str = "ollama",
    to_backend: str = "anthropic",
    prompt_tokens: int = 100,
    response_tokens: int = 50,
    cost_usd_micro: int = 0,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    owner_consented: bool = True,
    run_id: str = "",
    request_id: str = "",
    notes: str = "",
) -> EscalationEvent:
    started = started_at or _utc(2026, 5, 9, 12, 0, 0)
    return EscalationEvent(
        reason=reason,
        from_backend=from_backend,
        to_backend=to_backend,
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
        cost_usd_micro=cost_usd_micro,
        started_at=started,
        completed_at=completed_at or started + timedelta(milliseconds=500),
        owner_consented=owner_consented,
        run_id=run_id,
        request_id=request_id,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Lifecycle + schema
# ---------------------------------------------------------------------------


class TestStoreLifecycle:
    def test_creates_schema_in_memory(self) -> None:
        with CloudEscalationStore(":memory:") as store:
            assert store.schema_version == 1
            assert len(store) == 0

    def test_context_manager_persists_to_disk(self, tmp_path) -> None:
        db = tmp_path / "x.sqlite"
        with CloudEscalationStore(db) as store:
            store.append(_event(run_id="run-1"))

        with CloudEscalationStore(db) as store:
            assert len(store) == 1
            assert store.events()[0].run_id == "run-1"

    def test_close_idempotent(self) -> None:
        store = CloudEscalationStore(":memory:")
        store.close()
        store.close()  # must not raise

    def test_reopen_existing_does_not_reset(self, tmp_path) -> None:
        db = tmp_path / "x.sqlite"
        with CloudEscalationStore(db) as store:
            store.append(_event())
        # Re-open
        with CloudEscalationStore(db) as store:
            assert len(store) == 1


# ---------------------------------------------------------------------------
# Schema-version guard
# ---------------------------------------------------------------------------


class TestSchemaVersionGuard:
    def test_rejects_newer_schema(self, tmp_path) -> None:
        db = tmp_path / "future.sqlite"
        CloudEscalationStore(db).close()
        conn = sqlite3.connect(db)
        try:
            conn.execute(
                "INSERT INTO _schema_meta (version, applied_at) VALUES (?, ?)",
                (999, datetime.now(UTC).isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

        with pytest.raises(CloudEscalationStoreError, match="version 999"):
            CloudEscalationStore(db)

    def test_accepts_equal_version(self, tmp_path) -> None:
        db = tmp_path / "x.sqlite"
        CloudEscalationStore(db).close()
        # Re-open at v1 must be a no-op
        CloudEscalationStore(db).close()


# ---------------------------------------------------------------------------
# Append + read-back
# ---------------------------------------------------------------------------


class TestAppendAndRead:
    def test_returns_row_id(self) -> None:
        with CloudEscalationStore(":memory:") as store:
            assert store.append(_event(run_id="r1")) == 1
            assert store.append(_event(run_id="r2")) == 2

    def test_full_field_round_trip(self) -> None:
        original = _event(
            reason=EscalationReason.CONTEXT_TOO_LARGE,
            from_backend="ollama",
            to_backend="anthropic",
            prompt_tokens=420,
            response_tokens=80,
            cost_usd_micro=15_000,  # $0.015
            started_at=_utc(2026, 5, 9, 14, 30, 15),
            completed_at=_utc(2026, 5, 9, 14, 30, 16),
            owner_consented=True,
            run_id="run-abc",
            request_id="req-1",
            notes="planner step #3",
        )
        with CloudEscalationStore(":memory:") as store:
            store.append(original)
            ev = store.events()[0]
            assert ev.reason == original.reason
            assert ev.from_backend == original.from_backend
            assert ev.to_backend == original.to_backend
            assert ev.prompt_tokens == original.prompt_tokens
            assert ev.response_tokens == original.response_tokens
            assert ev.cost_usd_micro == original.cost_usd_micro
            assert ev.started_at == original.started_at
            assert ev.completed_at == original.completed_at
            assert ev.owner_consented is True
            assert ev.run_id == original.run_id
            assert ev.request_id == original.request_id
            assert ev.notes == original.notes

    def test_completed_at_none_round_trips(self) -> None:
        ev = EscalationEvent(
            reason=EscalationReason.UNKNOWN,
            from_backend="ollama",
            to_backend="anthropic",
            prompt_tokens=0,
            response_tokens=0,
            completed_at=None,
        )
        with CloudEscalationStore(":memory:") as store:
            store.append(ev)
            recovered = store.events()[0]
            assert recovered.completed_at is None

    def test_owner_consented_false_round_trips(self) -> None:
        """Boolean must survive the SQLite-INTEGER round-trip in both
        directions — a False mistakenly stored as truthy would silently
        flip the privacy-relevant ``owner_consented`` flag."""
        ev = _event(owner_consented=False)
        with CloudEscalationStore(":memory:") as store:
            store.append(ev)
            assert store.events()[0].owner_consented is False

    def test_clear_drops_everything(self) -> None:
        with CloudEscalationStore(":memory:") as store:
            for _ in range(5):
                store.append(_event())
            assert len(store) == 5
            store.clear()
            assert len(store) == 0

    def test_events_returned_in_insertion_order(self) -> None:
        with CloudEscalationStore(":memory:") as store:
            store.append(_event(reason=EscalationReason.OWNER_OVERRIDE))
            store.append(_event(reason=EscalationReason.LOCAL_BACKEND_DOWN))
            store.append(_event(reason=EscalationReason.CONTEXT_TOO_LARGE))
            evs = store.events()
            assert [e.reason for e in evs] == [
                EscalationReason.OWNER_OVERRIDE,
                EscalationReason.LOCAL_BACKEND_DOWN,
                EscalationReason.CONTEXT_TOO_LARGE,
            ]


# ---------------------------------------------------------------------------
# Read filters — parity with EscalationLedger
# ---------------------------------------------------------------------------


class TestReadFilters:
    def test_by_reason(self) -> None:
        with CloudEscalationStore(":memory:") as store:
            store.append(_event(reason=EscalationReason.OWNER_OVERRIDE))
            store.append(_event(reason=EscalationReason.OWNER_OVERRIDE))
            store.append(_event(reason=EscalationReason.UNKNOWN))
            assert len(store.by_reason(EscalationReason.OWNER_OVERRIDE)) == 2
            assert len(store.by_reason(EscalationReason.UNKNOWN)) == 1
            # Empty-bucket query
            assert store.by_reason(EscalationReason.RATE_LIMITED_LOCAL) == ()

    def test_by_destination(self) -> None:
        with CloudEscalationStore(":memory:") as store:
            store.append(_event(to_backend="anthropic"))
            store.append(_event(to_backend="openai"))
            store.append(_event(to_backend="anthropic"))
            assert len(store.by_destination("anthropic")) == 2
            assert len(store.by_destination("openai")) == 1
            assert store.by_destination("missing-provider") == ()

    def test_by_run_empty_string_returns_empty(self) -> None:
        """The in-memory ledger short-circuits on empty run_id so
        sentinel values used by un-tracked runs don't accidentally
        leak the entire ledger. The store must mirror this exactly."""
        with CloudEscalationStore(":memory:") as store:
            store.append(_event(run_id="run-1"))
            store.append(_event(run_id="run-2"))
            store.append(_event(run_id=""))
            # Even though one event has run_id="" stored, we don't return
            # the whole ledger when queried with empty string.
            assert store.by_run("") == ()
            assert len(store.by_run("run-1")) == 1

    def test_by_run_missing_returns_empty(self) -> None:
        with CloudEscalationStore(":memory:") as store:
            store.append(_event(run_id="run-1"))
            assert store.by_run("nope") == ()

    def test_in_window_inclusive(self) -> None:
        with CloudEscalationStore(":memory:") as store:
            early = _utc(2026, 5, 9, 10)
            mid = _utc(2026, 5, 9, 11)
            late = _utc(2026, 5, 9, 12)
            for ts in (early, mid, late):
                store.append(_event(started_at=ts))
            assert len(store.in_window(start=early, end=late)) == 3
            assert len(store.in_window(start=mid, end=mid)) == 1

    def test_in_window_swapped_window_raises(self) -> None:
        with CloudEscalationStore(":memory:") as store:
            with pytest.raises(ValueError, match="start must be <= end"):
                store.in_window(
                    start=_utc(2026, 5, 9, 12),
                    end=_utc(2026, 5, 9, 10),
                )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestSummarise:
    def test_empty_summary_is_zero(self) -> None:
        with CloudEscalationStore(":memory:") as store:
            summary = store.summarise()
            assert summary.event_count == 0
            assert summary.total_cost_usd_micro == 0
            assert summary.total_prompt_tokens == 0
            assert summary.total_response_tokens == 0

    def test_buckets_match_in_memory_contract(self) -> None:
        with CloudEscalationStore(":memory:") as store:
            store.append(_event(reason=EscalationReason.OWNER_OVERRIDE, to_backend="anthropic"))
            store.append(_event(reason=EscalationReason.OWNER_OVERRIDE, to_backend="anthropic"))
            store.append(_event(reason=EscalationReason.UNKNOWN, to_backend="openai"))
            s = store.summarise()
            assert s.event_count == 3
            assert s.by_reason[EscalationReason.OWNER_OVERRIDE] == 2
            assert s.by_reason[EscalationReason.UNKNOWN] == 1
            assert s.by_destination == {"anthropic": 2, "openai": 1}

    def test_token_and_cost_totals(self) -> None:
        with CloudEscalationStore(":memory:") as store:
            store.append(
                _event(
                    prompt_tokens=100,
                    response_tokens=50,
                    cost_usd_micro=10_000,
                )
            )
            store.append(
                _event(
                    prompt_tokens=200,
                    response_tokens=80,
                    cost_usd_micro=25_000,
                )
            )
            s = store.summarise()
            assert s.total_prompt_tokens == 300
            assert s.total_response_tokens == 130
            assert s.total_cost_usd_micro == 35_000  # $0.035 cumulative
            assert s.total_cost_usd == pytest.approx(0.035)

    def test_summarise_with_explicit_subset(self) -> None:
        with CloudEscalationStore(":memory:") as store:
            store.append(_event(run_id="run-1"))
            store.append(_event(run_id="run-2"))
            scoped = store.by_run("run-1")
            s = store.summarise(events=scoped)
            assert s.event_count == 1

    def test_summary_is_immutable(self) -> None:
        import dataclasses

        with CloudEscalationStore(":memory:") as store:
            s = store.summarise()
        assert isinstance(s, EscalationSummary)
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.event_count = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Snapshot — JSON-safe
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_empty_snapshot(self) -> None:
        with CloudEscalationStore(":memory:") as store:
            assert store.snapshot() == []

    def test_round_trip_via_json(self) -> None:
        with CloudEscalationStore(":memory:") as store:
            store.append(
                _event(
                    reason=EscalationReason.CONTEXT_TOO_LARGE,
                    to_backend="anthropic",
                    prompt_tokens=200,
                    response_tokens=120,
                    cost_usd_micro=12_000,
                    run_id="run-x",
                )
            )
            rows = store.snapshot()
            serialised = json.dumps(rows)
            parsed = json.loads(serialised)
            assert isinstance(parsed, list)
            row = parsed[0]
            assert row["reason"] == "context_too_large"
            assert row["to_backend"] == "anthropic"
            assert row["cost_usd_micro"] == 12_000
            assert row["cost_usd"] == pytest.approx(0.012)
            assert row["run_id"] == "run-x"

    def test_snapshot_keys_match_in_memory_ledger(self) -> None:
        """Trace-UI consumers must NOT branch on whether the source is
        in-memory or on-disk. Pin the key set."""
        from cognithor.security.cloud_escalation import EscalationLedger

        memory_ledger = EscalationLedger()
        memory_ledger.record(_event(run_id="r"))
        memory_keys = set(memory_ledger.snapshot()[0].keys())

        with CloudEscalationStore(":memory:") as store:
            store.append(_event(run_id="r"))
            disk_keys = set(store.snapshot()[0].keys())

        assert memory_keys == disk_keys, (
            f"snapshot keys diverge — memory_only={memory_keys - disk_keys}, "
            f"disk_only={disk_keys - memory_keys}"
        )


# ---------------------------------------------------------------------------
# File persistence + concurrency
# ---------------------------------------------------------------------------


class TestFilePersistence:
    def test_writes_visible_after_reopen(self, tmp_path) -> None:
        db = tmp_path / "x.sqlite"
        with CloudEscalationStore(db) as store:
            store.append(_event(run_id="run-1", to_backend="anthropic"))
            store.append(_event(run_id="run-2", to_backend="openai"))
        with CloudEscalationStore(db) as store:
            assert len(store) == 2
            backends = {e.to_backend for e in store.events()}
            assert backends == {"anthropic", "openai"}

    def test_concurrent_writers_share_db(self, tmp_path) -> None:
        db = tmp_path / "shared.sqlite"
        with (
            CloudEscalationStore(db) as a,
            CloudEscalationStore(db) as b,
        ):
            a.append(_event(run_id="run-a"))
            b.append(_event(run_id="run-b"))
        with CloudEscalationStore(db) as reader:
            assert len(reader) == 2
            assert {e.run_id for e in reader.events()} == {"run-a", "run-b"}


# ---------------------------------------------------------------------------
# Indices — query-shape sanity
# ---------------------------------------------------------------------------


class TestIndices:
    def test_all_four_hot_indices_present(self) -> None:
        """Pin the four hot read-path indices: run_id, to_backend,
        reason, started_at. Future schema edits MUST NOT silently
        drop one — Trace-UI panels would table-scan on every render."""
        with CloudEscalationStore(":memory:") as store:
            cursor = store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='escalation_events'"
            )
            names = {row[0] for row in cursor.fetchall()}
            assert "idx_escalation_events_run_id" in names
            assert "idx_escalation_events_to_backend" in names
            assert "idx_escalation_events_reason" in names
            assert "idx_escalation_events_started_at" in names
