"""Tests for the SQLite-backed MigrationLedgerStore (TRUST-10 disk persistence)."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

import pytest

from cognithor.security.migration_ledger import (
    MigrationChainError,
    MigrationDomain,
    MigrationLedger,
    MigrationStatus,
    MigrationStep,
)
from cognithor.security.migration_ledger_store import (
    MigrationLedgerStore,
    MigrationLedgerStoreError,
)


def _hash(seed: str) -> str:
    import hashlib

    return hashlib.sha256(seed.encode()).hexdigest()


def _step(
    *,
    domain: MigrationDomain = MigrationDomain.MEMORY_SEMANTIC,
    source_version: str = "v0",
    target_version: str = "v1",
    status: MigrationStatus = MigrationStatus.APPLIED,
    applied_at: datetime | None = None,
    applied_by: str = "system",
    item_count: int = -1,
    checksum_before: str = "",
    checksum_after: str = "",
    rollback_of: str = "",
    migration_id: str = "",
    notes: str = "",
) -> MigrationStep:
    return MigrationStep(
        domain=domain,
        source_version=source_version,
        target_version=target_version,
        status=status,
        applied_at=applied_at or datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC),
        applied_by=applied_by,
        item_count=item_count,
        checksum_before=checksum_before,
        checksum_after=checksum_after,
        rollback_of=rollback_of,
        migration_id=migration_id,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Lifecycle + schema
# ---------------------------------------------------------------------------


class TestStoreLifecycle:
    def test_creates_schema_in_memory(self) -> None:
        with MigrationLedgerStore(":memory:") as store:
            assert store.schema_version == 1
            assert len(store) == 0

    def test_context_manager_persists_to_disk(self, tmp_path) -> None:
        db = tmp_path / "x.sqlite"
        with MigrationLedgerStore(db) as store:
            store.record(_step(migration_id="m1"))

        with MigrationLedgerStore(db) as store:
            assert len(store) == 1
            assert store.head_version(MigrationDomain.MEMORY_SEMANTIC) == "v1"

    def test_close_idempotent(self) -> None:
        store = MigrationLedgerStore(":memory:")
        store.close()
        store.close()


# ---------------------------------------------------------------------------
# Schema-version guard
# ---------------------------------------------------------------------------


class TestSchemaVersionGuard:
    def test_rejects_newer_schema(self, tmp_path) -> None:
        db = tmp_path / "future.sqlite"
        MigrationLedgerStore(db).close()
        conn = sqlite3.connect(db)
        try:
            conn.execute(
                "INSERT INTO _schema_meta (version, applied_at) VALUES (?, ?)",
                (999, datetime.now(UTC).isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

        with pytest.raises(MigrationLedgerStoreError, match="version 999"):
            MigrationLedgerStore(db)

    def test_accepts_equal_version(self, tmp_path) -> None:
        db = tmp_path / "x.sqlite"
        MigrationLedgerStore(db).close()
        MigrationLedgerStore(db).close()


# ---------------------------------------------------------------------------
# Record + chain integrity
# ---------------------------------------------------------------------------


class TestChainIntegrity:
    def test_first_step_with_no_head_succeeds(self) -> None:
        with MigrationLedgerStore(":memory:") as store:
            store.record(_step())
            assert store.head_version(MigrationDomain.MEMORY_SEMANTIC) == "v1"

    def test_continued_chain_succeeds(self) -> None:
        with MigrationLedgerStore(":memory:") as store:
            store.record(_step(source_version="v0", target_version="v1"))
            store.record(_step(source_version="v1", target_version="v2"))
            store.record(_step(source_version="v2", target_version="v3"))
            assert store.head_version(MigrationDomain.MEMORY_SEMANTIC) == "v3"

    def test_chain_mismatch_rejected(self) -> None:
        with MigrationLedgerStore(":memory:") as store:
            store.record(_step(source_version="v0", target_version="v1"))
            with pytest.raises(MigrationChainError, match="chain mismatch"):
                store.record(_step(source_version="v5", target_version="v6"))

    def test_chain_mismatch_does_not_partially_persist(self) -> None:
        """If validation rejects a step, the row MUST NOT land on
        disk — otherwise re-playing the history would corrupt the
        chain on next open."""
        with MigrationLedgerStore(":memory:") as store:
            store.record(_step(source_version="v0", target_version="v1"))
            with pytest.raises(MigrationChainError):
                store.record(_step(source_version="v5", target_version="v6"))
            assert len(store) == 1
            assert store.head_version(MigrationDomain.MEMORY_SEMANTIC) == "v1"

    def test_pending_and_failed_skip_head_check(self) -> None:
        """PENDING and FAILED don't move the head and don't require
        ``source_version == head``. Mirrors the in-memory contract."""
        with MigrationLedgerStore(":memory:") as store:
            store.record(_step(source_version="v0", target_version="v1"))
            # PENDING must be allowed without head match
            store.record(
                _step(
                    source_version="v99",
                    target_version="v100",
                    status=MigrationStatus.PENDING,
                )
            )
            # FAILED also allowed
            store.record(
                _step(
                    source_version="v99",
                    target_version="v99",
                    status=MigrationStatus.FAILED,
                )
            )
            # Head still on v1
            assert store.head_version(MigrationDomain.MEMORY_SEMANTIC) == "v1"

    def test_rollback_of_unknown_id_rejected(self) -> None:
        with MigrationLedgerStore(":memory:") as store:
            store.record(_step(source_version="v0", target_version="v1"))
            with pytest.raises(MigrationChainError, match="unknown migration_id"):
                store.record(
                    _step(
                        source_version="v1",
                        target_version="v0",
                        status=MigrationStatus.ROLLED_BACK,
                        rollback_of="nope",
                    )
                )

    def test_rollback_succeeds_then_head_moves_back(self) -> None:
        """Happy-path rollback: head moves back to the rollback's
        target_version."""
        with MigrationLedgerStore(":memory:") as store:
            store.record(
                _step(
                    source_version="v0",
                    target_version="v1",
                    migration_id="m1",
                )
            )
            store.record(
                _step(
                    source_version="v1",
                    target_version="v0",
                    status=MigrationStatus.ROLLED_BACK,
                    rollback_of="m1",
                )
            )
            assert store.head_version(MigrationDomain.MEMORY_SEMANTIC) == "v0"

    def test_duplicate_migration_id_rejected(self) -> None:
        with MigrationLedgerStore(":memory:") as store:
            store.record(
                _step(
                    source_version="v0",
                    target_version="v1",
                    migration_id="dup",
                )
            )
            with pytest.raises(MigrationChainError, match="duplicate migration_id"):
                store.record(
                    _step(
                        source_version="v1",
                        target_version="v2",
                        migration_id="dup",
                    )
                )


# ---------------------------------------------------------------------------
# Cross-process chain integrity
# ---------------------------------------------------------------------------


class TestCrossProcessChain:
    def test_chain_mismatch_after_reopen_rejected(self, tmp_path) -> None:
        """Re-open + new write must still respect the chain. Pins
        the disk-layer's most load-bearing invariant: history persists."""
        db = tmp_path / "x.sqlite"
        with MigrationLedgerStore(db) as store:
            store.record(_step(source_version="v0", target_version="v1"))

        # New process, same db
        with MigrationLedgerStore(db) as store:
            assert store.head_version(MigrationDomain.MEMORY_SEMANTIC) == "v1"
            with pytest.raises(MigrationChainError, match="chain mismatch"):
                store.record(_step(source_version="v9", target_version="v10"))
            # Still exactly one step on disk
            assert len(store) == 1

    def test_continued_chain_across_processes(self, tmp_path) -> None:
        db = tmp_path / "x.sqlite"
        with MigrationLedgerStore(db) as a:
            a.record(_step(source_version="v0", target_version="v1"))
        with MigrationLedgerStore(db) as b:
            b.record(_step(source_version="v1", target_version="v2"))
        with MigrationLedgerStore(db) as reader:
            assert reader.head_version(MigrationDomain.MEMORY_SEMANTIC) == "v2"
            assert len(reader) == 2


# ---------------------------------------------------------------------------
# Read API parity
# ---------------------------------------------------------------------------


class TestReadParity:
    def test_steps_in_insertion_order(self) -> None:
        with MigrationLedgerStore(":memory:") as store:
            store.record(_step(source_version="v0", target_version="v1"))
            store.record(_step(source_version="v1", target_version="v2"))
            store.record(_step(source_version="v2", target_version="v3"))
            tgts = [s.target_version for s in store.steps()]
            assert tgts == ["v1", "v2", "v3"]

    def test_for_domain_filters_correctly(self) -> None:
        with MigrationLedgerStore(":memory:") as store:
            store.record(
                _step(
                    domain=MigrationDomain.MEMORY_SEMANTIC,
                    source_version="v0",
                    target_version="v1",
                )
            )
            store.record(
                _step(
                    domain=MigrationDomain.AUDIT_LOG,
                    source_version="a0",
                    target_version="a1",
                )
            )
            store.record(
                _step(
                    domain=MigrationDomain.MEMORY_SEMANTIC,
                    source_version="v1",
                    target_version="v2",
                )
            )
            sem = store.for_domain(MigrationDomain.MEMORY_SEMANTIC)
            assert len(sem) == 2
            audit = store.for_domain(MigrationDomain.AUDIT_LOG)
            assert len(audit) == 1

    def test_head_version_independent_per_domain(self) -> None:
        with MigrationLedgerStore(":memory:") as store:
            store.record(
                _step(
                    domain=MigrationDomain.MEMORY_SEMANTIC,
                    source_version="v0",
                    target_version="v1",
                )
            )
            store.record(
                _step(
                    domain=MigrationDomain.AUDIT_LOG,
                    source_version="a0",
                    target_version="a1",
                )
            )
            assert store.head_version(MigrationDomain.MEMORY_SEMANTIC) == "v1"
            assert store.head_version(MigrationDomain.AUDIT_LOG) == "a1"
            assert store.head_version(MigrationDomain.MEMORY_VAULT) is None

    def test_get_by_migration_id(self) -> None:
        with MigrationLedgerStore(":memory:") as store:
            store.record(_step(source_version="v0", target_version="v1", migration_id="m1"))
            assert store.get("m1") is not None
            assert store.get("nope") is None
            assert store.get("") is None

    def test_applied_only_excludes_pending_and_failed(self) -> None:
        with MigrationLedgerStore(":memory:") as store:
            store.record(
                _step(
                    source_version="v0",
                    target_version="v1",
                    status=MigrationStatus.APPLIED,
                )
            )
            store.record(
                _step(
                    source_version="v99",
                    target_version="v100",
                    status=MigrationStatus.PENDING,
                )
            )
            store.record(
                _step(
                    source_version="v99",
                    target_version="v99",
                    status=MigrationStatus.FAILED,
                )
            )
            applied = store.applied_only(MigrationDomain.MEMORY_SEMANTIC)
            assert len(applied) == 1


# ---------------------------------------------------------------------------
# Snapshot — dict shape mirrored from in-memory ledger
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_empty_snapshot_shape(self) -> None:
        with MigrationLedgerStore(":memory:") as store:
            snap = store.snapshot()
            assert snap == {"head_version": {}, "steps": []}

    def test_round_trip_via_json(self) -> None:
        with MigrationLedgerStore(":memory:") as store:
            store.record(
                _step(
                    source_version="v0",
                    target_version="v1",
                    migration_id="m1",
                    item_count=1234,
                    checksum_before=_hash("before"),
                    checksum_after=_hash("after"),
                    notes="memory schema bump",
                )
            )
            snap = store.snapshot()
            serialised = json.dumps(snap)
            parsed = json.loads(serialised)
            assert isinstance(parsed, dict)
            assert parsed["head_version"]["memory_semantic"] == "v1"
            assert len(parsed["steps"]) == 1
            step = parsed["steps"][0]
            assert step["migration_id"] == "m1"
            assert step["item_count"] == 1234
            assert step["checksum_before"] == _hash("before")

    def test_snapshot_matches_in_memory_ledger_for_same_history(self) -> None:
        """Trace-UI consumers must NOT branch on whether the source
        is in-memory or on-disk. Pin the snapshot shape exactly."""
        history = [
            _step(source_version="v0", target_version="v1", migration_id="m1"),
            _step(source_version="v1", target_version="v2", migration_id="m2"),
        ]
        memory_ledger = MigrationLedger()
        for s in history:
            memory_ledger.record(s)

        with MigrationLedgerStore(":memory:") as store:
            for s in history:
                store.record(s)
            assert store.snapshot() == memory_ledger.snapshot()


# ---------------------------------------------------------------------------
# Indices
# ---------------------------------------------------------------------------


class TestIndices:
    def test_all_four_hot_indices_present(self) -> None:
        with MigrationLedgerStore(":memory:") as store:
            cursor = store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='migration_steps'"
            )
            names = {row[0] for row in cursor.fetchall()}
            assert "idx_migration_steps_domain" in names
            assert "idx_migration_steps_status" in names
            assert "idx_migration_steps_migration_id" in names
            assert "idx_migration_steps_applied_at" in names


# ---------------------------------------------------------------------------
# Field round-trip — pin the rare/optional axes
# ---------------------------------------------------------------------------


class TestFieldRoundTrip:
    def test_full_round_trip(self) -> None:
        original = _step(
            source_version="v0",
            target_version="v1",
            applied_at=datetime(2026, 5, 9, 14, 30, 15, tzinfo=UTC),
            applied_by="alex@cognithor.ai",
            item_count=42,
            checksum_before=_hash("before"),
            checksum_after=_hash("after"),
            migration_id="m1",
            notes="manual backfill",
        )
        with MigrationLedgerStore(":memory:") as store:
            store.record(original)
            recovered = store.steps()[0]
            assert recovered == original
