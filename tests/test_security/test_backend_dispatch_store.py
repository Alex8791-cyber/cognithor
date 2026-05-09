"""Tests for the SQLite-backed BackendDispatchStore (TRUST-8 disk persistence)."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from cognithor.security.backend_dispatch import (
    BackendDispatchEvent,
    DispatchOutcome,
    DispatchSummary,
)
from cognithor.security.backend_dispatch_store import (
    BackendDispatchStore,
    BackendDispatchStoreError,
)


def _utc(
    year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0
) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)


def _event(
    *,
    backend_type: str = "ollama",
    model: str = "qwen3:30b",
    outcome: DispatchOutcome = DispatchOutcome.SUCCESS,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    prompt_tokens: int = -1,
    response_tokens: int = -1,
    error_kind: str = "",
    error_msg: str = "",
    is_fallback: bool = False,
    run_id: str = "",
    request_id: str = "",
    notes: str = "",
) -> BackendDispatchEvent:
    started = started_at or _utc(2026, 5, 9, 12, 0, 0)
    return BackendDispatchEvent(
        backend_type=backend_type,
        model=model,
        outcome=outcome,
        started_at=started,
        completed_at=completed_at or started + timedelta(milliseconds=420),
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
        error_kind=error_kind,
        error_msg=error_msg,
        is_fallback=is_fallback,
        run_id=run_id,
        request_id=request_id,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Lifecycle + schema
# ---------------------------------------------------------------------------


class TestStoreLifecycle:
    def test_creates_schema_in_memory(self) -> None:
        store = BackendDispatchStore(":memory:")
        try:
            assert store.schema_version == 1
            assert len(store) == 0
        finally:
            store.close()

    def test_context_manager_closes(self, tmp_path) -> None:
        db = tmp_path / "x.sqlite"
        with BackendDispatchStore(db) as store:
            store.append(_event())
            assert len(store) == 1

        # After exit, the connection is closed — re-opening must work
        # cleanly and pick up the persisted row.
        with BackendDispatchStore(db) as store:
            assert len(store) == 1

    def test_close_is_idempotent(self) -> None:
        store = BackendDispatchStore(":memory:")
        store.close()
        # Second close must NOT raise (e.g. when called from a finally
        # block after an early return).
        store.close()

    def test_reopen_existing_file_no_op(self, tmp_path) -> None:
        """Opening an already-initialised DB does not reset / duplicate
        the schema. The IF NOT EXISTS guards in the schema script
        protect against schema reset on every connect."""
        db = tmp_path / "x.sqlite"
        BackendDispatchStore(db).close()
        # Inject an event via raw SQL to prove the next open preserves state
        conn = sqlite3.connect(db)
        try:
            conn.execute(
                "INSERT INTO dispatch_events (backend_type, outcome, started_at) VALUES (?, ?, ?)",
                ("ollama", "success", _utc(2026, 5, 9, 0, 0).isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

        with BackendDispatchStore(db) as store:
            assert len(store) == 1
            ev = store.events()[0]
            assert ev.backend_type == "ollama"


class TestSchemaVersionGuard:
    def test_rejects_newer_schema_loudly(self, tmp_path) -> None:
        """A future cognithor build that wrote schema v999 must NOT be
        opened by today's code — the writer's column layout is unknown
        and rolling forward could corrupt data."""
        db = tmp_path / "future.sqlite"
        # Create the DB normally then forcibly bump the version.
        BackendDispatchStore(db).close()
        conn = sqlite3.connect(db)
        try:
            conn.execute(
                "INSERT INTO _schema_meta (version, applied_at) VALUES (?, ?)",
                (999, datetime.now(UTC).isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

        with pytest.raises(BackendDispatchStoreError, match="version 999"):
            BackendDispatchStore(db)

    def test_accepts_equal_schema_version(self, tmp_path) -> None:
        db = tmp_path / "x.sqlite"
        BackendDispatchStore(db).close()
        # Re-open: should not raise (equal version is the happy path).
        BackendDispatchStore(db).close()


# ---------------------------------------------------------------------------
# Append + read-back
# ---------------------------------------------------------------------------


class TestAppendAndRead:
    def test_append_returns_row_id(self) -> None:
        with BackendDispatchStore(":memory:") as store:
            row_id = store.append(_event(run_id="run-1"))
            assert row_id == 1
            row_id_2 = store.append(_event(run_id="run-2"))
            assert row_id_2 == 2

    def test_round_trip_preserves_all_fields(self) -> None:
        """Every public field on the event must round-trip through SQLite
        unchanged."""
        original = _event(
            backend_type="anthropic",
            model="claude-opus-4-7",
            outcome=DispatchOutcome.BACKEND_ERROR,
            started_at=_utc(2026, 5, 9, 14, 30, 15),
            completed_at=_utc(2026, 5, 9, 14, 30, 16),
            prompt_tokens=420,
            response_tokens=80,
            error_kind="ProviderRateLimited",
            error_msg="429 Too Many Requests",
            is_fallback=True,
            run_id="run-abc",
            request_id="req-1",
            notes="planner step #3",
        )
        with BackendDispatchStore(":memory:") as store:
            store.append(original)
            ev = store.events()[0]
            assert ev.backend_type == original.backend_type
            assert ev.model == original.model
            assert ev.outcome == original.outcome
            assert ev.started_at == original.started_at
            assert ev.completed_at == original.completed_at
            assert ev.prompt_tokens == original.prompt_tokens
            assert ev.response_tokens == original.response_tokens
            assert ev.error_kind == original.error_kind
            assert ev.error_msg == original.error_msg
            assert ev.is_fallback is True
            assert ev.run_id == original.run_id
            assert ev.request_id == original.request_id
            assert ev.notes == original.notes

    def test_completed_at_none_round_trips(self) -> None:
        """A still-in-flight event has ``completed_at=None``. The store
        must preserve that as NULL → None on read-back, not silently
        coerce to ``started_at`` or ``''``."""
        ev = BackendDispatchEvent(
            backend_type="ollama",
            model="m",
            outcome=DispatchOutcome.SUCCESS,
            completed_at=None,
        )
        with BackendDispatchStore(":memory:") as store:
            store.append(ev)
            recovered = store.events()[0]
            assert recovered.completed_at is None

    def test_clear_drops_everything(self) -> None:
        with BackendDispatchStore(":memory:") as store:
            for _ in range(5):
                store.append(_event())
            assert len(store) == 5
            store.clear()
            assert len(store) == 0

    def test_events_returned_in_insertion_order(self) -> None:
        with BackendDispatchStore(":memory:") as store:
            store.append(_event(backend_type="ollama"))
            store.append(_event(backend_type="anthropic"))
            store.append(_event(backend_type="openai"))
            evs = store.events()
            assert [e.backend_type for e in evs] == ["ollama", "anthropic", "openai"]


# ---------------------------------------------------------------------------
# Read filters — parity with BackendDispatchLedger
# ---------------------------------------------------------------------------


class TestReadFilters:
    def test_by_run(self) -> None:
        with BackendDispatchStore(":memory:") as store:
            store.append(_event(run_id="run-1"))
            store.append(_event(run_id="run-2"))
            store.append(_event(run_id="run-1"))
            scoped = store.by_run("run-1")
            assert len(scoped) == 2
            assert all(e.run_id == "run-1" for e in scoped)
            assert store.by_run("run-3") == ()

    def test_by_backend(self) -> None:
        with BackendDispatchStore(":memory:") as store:
            store.append(_event(backend_type="ollama"))
            store.append(_event(backend_type="anthropic"))
            store.append(_event(backend_type="ollama"))
            assert len(store.by_backend("ollama")) == 2
            assert len(store.by_backend("anthropic")) == 1
            assert store.by_backend("missing") == ()

    def test_by_outcome(self) -> None:
        with BackendDispatchStore(":memory:") as store:
            store.append(_event(outcome=DispatchOutcome.SUCCESS))
            store.append(_event(outcome=DispatchOutcome.BACKEND_ERROR))
            store.append(_event(outcome=DispatchOutcome.CIRCUIT_OPEN))
            assert len(store.by_outcome(DispatchOutcome.SUCCESS)) == 1
            assert len(store.by_outcome(DispatchOutcome.BACKEND_ERROR)) == 1
            assert len(store.by_outcome(DispatchOutcome.CIRCUIT_OPEN)) == 1

    def test_in_window_inclusive(self) -> None:
        with BackendDispatchStore(":memory:") as store:
            early = _utc(2026, 5, 9, 10)
            mid = _utc(2026, 5, 9, 11)
            late = _utc(2026, 5, 9, 12)
            for ts in (early, mid, late):
                store.append(_event(started_at=ts))
            assert len(store.in_window(start=early, end=late)) == 3
            assert len(store.in_window(start=mid, end=mid)) == 1


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestSummarise:
    def test_empty_ledger_summary_vacuous_success(self) -> None:
        with BackendDispatchStore(":memory:") as store:
            summary = store.summarise()
            assert summary.event_count == 0
            assert summary.success_rate == 1.0
            assert summary.total_prompt_tokens == 0
            assert summary.total_response_tokens == 0

    def test_buckets_match_in_memory_ledger_contract(self) -> None:
        with BackendDispatchStore(":memory:") as store:
            store.append(_event(backend_type="ollama", outcome=DispatchOutcome.SUCCESS))
            store.append(_event(backend_type="ollama", outcome=DispatchOutcome.BACKEND_ERROR))
            store.append(_event(backend_type="anthropic", outcome=DispatchOutcome.SUCCESS))
            s = store.summarise()
            assert s.event_count == 3
            assert s.success_count == 2
            assert s.by_backend == {"ollama": 2, "anthropic": 1}
            assert s.by_outcome[DispatchOutcome.SUCCESS] == 2
            assert s.by_outcome[DispatchOutcome.BACKEND_ERROR] == 1

    def test_token_totals_propagate_unknown(self) -> None:
        """Same -1 propagation rule as the in-memory ledger."""
        with BackendDispatchStore(":memory:") as store:
            store.append(_event(prompt_tokens=100, response_tokens=50))
            store.append(_event(prompt_tokens=-1, response_tokens=20))
            s = store.summarise()
            assert s.total_prompt_tokens == -1
            assert s.total_response_tokens == 70

    def test_summarise_with_explicit_subset(self) -> None:
        """Caller can pre-scope via ``by_run`` and pass into ``summarise``."""
        with BackendDispatchStore(":memory:") as store:
            store.append(_event(run_id="run-1", backend_type="ollama"))
            store.append(_event(run_id="run-2", backend_type="anthropic"))
            scoped = store.by_run("run-1")
            s = store.summarise(events=scoped)
            assert s.event_count == 1
            assert s.by_backend == {"ollama": 1}

    def test_summary_is_immutable(self) -> None:
        with BackendDispatchStore(":memory:") as store:
            s = store.summarise()
        # Can't reassign — the dataclass is frozen
        import dataclasses

        with pytest.raises(dataclasses.FrozenInstanceError):
            s.event_count = 99  # type: ignore[misc]
        # Right type
        assert isinstance(s, DispatchSummary)


# ---------------------------------------------------------------------------
# Snapshot — JSON-safe round-trip
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_empty_snapshot(self) -> None:
        with BackendDispatchStore(":memory:") as store:
            assert store.snapshot() == []

    def test_snapshot_round_trip_via_json(self) -> None:
        with BackendDispatchStore(":memory:") as store:
            store.append(
                _event(
                    backend_type="anthropic",
                    model="claude-opus-4-7",
                    prompt_tokens=200,
                    response_tokens=120,
                    is_fallback=True,
                    run_id="run-x",
                )
            )
            rows = store.snapshot()
            serialised = json.dumps(rows)
            parsed = json.loads(serialised)
            assert isinstance(parsed, list)
            assert parsed[0]["backend_type"] == "anthropic"
            assert parsed[0]["run_id"] == "run-x"
            assert parsed[0]["is_fallback"] is True

    def test_snapshot_matches_in_memory_ledger_keys(self) -> None:
        """The snapshot column names MUST match the in-memory ledger's
        snapshot — Trace-UI consumers should not have to branch on
        whether the source is in-memory or on-disk."""
        from cognithor.security.backend_dispatch import BackendDispatchLedger

        memory_ledger = BackendDispatchLedger()
        memory_ledger.record(_event(run_id="run-x"))
        memory_keys = set(memory_ledger.snapshot()[0].keys())

        with BackendDispatchStore(":memory:") as store:
            store.append(_event(run_id="run-x"))
            disk_keys = set(store.snapshot()[0].keys())

        assert memory_keys == disk_keys, (
            f"snapshot keys diverge between in-memory and on-disk ledgers — "
            f"memory_only={memory_keys - disk_keys}, "
            f"disk_only={disk_keys - memory_keys}"
        )


# ---------------------------------------------------------------------------
# Persistence across processes — file-based DB
# ---------------------------------------------------------------------------


class TestFilePersistence:
    def test_writes_visible_after_reopen(self, tmp_path) -> None:
        db = tmp_path / "x.sqlite"
        with BackendDispatchStore(db) as store:
            store.append(_event(run_id="run-1"))
            store.append(_event(run_id="run-2"))
        # Fresh process — re-open and see both events
        with BackendDispatchStore(db) as store:
            assert len(store) == 2
            assert {e.run_id for e in store.events()} == {"run-1", "run-2"}

    def test_concurrent_open_does_not_corrupt(self, tmp_path) -> None:
        """Two stores against the same file must not corrupt each
        other — both writers commit cleanly under WAL."""
        db = tmp_path / "shared.sqlite"
        with BackendDispatchStore(db) as a, BackendDispatchStore(db) as b:
            a.append(_event(run_id="run-a"))
            b.append(_event(run_id="run-b"))

        with BackendDispatchStore(db) as reader:
            assert len(reader) == 2
            run_ids = {e.run_id for e in reader.events()}
            assert run_ids == {"run-a", "run-b"}


# ---------------------------------------------------------------------------
# Indices — query-shape sanity
# ---------------------------------------------------------------------------


class TestIndices:
    def test_run_id_index_exists(self) -> None:
        """Without an index on run_id, by_run() table-scans on every
        Trace-UI panel render. The schema script creates one — pin
        it so a future schema edit can't accidentally drop it."""
        with BackendDispatchStore(":memory:") as store:
            cursor = store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='dispatch_events'"
            )
            names = {row[0] for row in cursor.fetchall()}
            assert "idx_dispatch_events_run_id" in names
            assert "idx_dispatch_events_backend_type" in names
            assert "idx_dispatch_events_started_at" in names
