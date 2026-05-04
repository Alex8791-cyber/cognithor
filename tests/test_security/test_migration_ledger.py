"""Tests for the TRUST-10 migration-ledger foundation."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest

from cognithor.security.migration_ledger import (
    MIGRATION_LEDGER,
    MigrationChainError,
    MigrationDomain,
    MigrationLedger,
    MigrationStatus,
    MigrationStep,
)

_HEX = "a" * 64
_HEX2 = "b" * 64


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def _step(
    *,
    domain: MigrationDomain = MigrationDomain.AUDIT_LOG,
    source_version: str = "v1",
    target_version: str = "v2",
    status: MigrationStatus = MigrationStatus.APPLIED,
    applied_at: datetime | None = None,
    applied_by: str = "system",
    item_count: int = 0,
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
        applied_at=applied_at if applied_at is not None else _utc(2026, 5, 4, 12, 0),
        applied_by=applied_by,
        item_count=item_count,
        checksum_before=checksum_before,
        checksum_after=checksum_after,
        rollback_of=rollback_of,
        migration_id=migration_id,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# MigrationStep validation
# ---------------------------------------------------------------------------


class TestMigrationStepValidation:
    def test_minimal_step(self) -> None:
        step = _step()
        assert step.domain == MigrationDomain.AUDIT_LOG
        assert step.source_version == "v1"
        assert step.target_version == "v2"
        assert step.status == MigrationStatus.APPLIED
        assert step.is_data_bearing is True
        assert step.applied_at.tzinfo == UTC

    def test_frozen(self) -> None:
        step = _step()
        with pytest.raises(dataclasses.FrozenInstanceError):
            step.notes = "tamper"  # type: ignore[misc]

    def test_empty_source_version_rejected(self) -> None:
        with pytest.raises(ValueError, match="source_version"):
            _step(source_version="")

    def test_empty_target_version_rejected(self) -> None:
        with pytest.raises(ValueError, match="target_version"):
            _step(target_version="")

    def test_no_op_only_allowed_for_failed(self) -> None:
        # A step that doesn't change the version is only legal as a
        # FAILED record (we tried, it didn't take).
        with pytest.raises(ValueError, match="differ"):
            _step(source_version="v1", target_version="v1")
        # Same shape, but FAILED — allowed.
        step = _step(
            source_version="v1",
            target_version="v1",
            status=MigrationStatus.FAILED,
        )
        assert step.status == MigrationStatus.FAILED

    def test_item_count_validation(self) -> None:
        # -1 is allowed (schema-only)
        step = _step(item_count=-1)
        assert step.is_data_bearing is False
        # any non-negative int allowed
        step = _step(item_count=42)
        assert step.is_data_bearing is True
        # below -1 rejected
        with pytest.raises(ValueError, match="item_count"):
            _step(item_count=-2)

    def test_checksum_shape_validation(self) -> None:
        # Empty is fine.
        _step(checksum_before="")
        # 64 lowercase hex is fine.
        _step(checksum_before=_HEX, checksum_after=_HEX2)
        # Wrong length rejected.
        with pytest.raises(ValueError, match="checksum_before"):
            _step(checksum_before="a" * 32)
        # Uppercase rejected.
        with pytest.raises(ValueError, match="checksum_after"):
            _step(checksum_after="A" * 64)

    def test_rollback_of_requires_rolled_back_status(self) -> None:
        with pytest.raises(ValueError, match="rollback_of"):
            _step(rollback_of="audit_log:v1:v2", status=MigrationStatus.APPLIED)
        # ROLLED_BACK + rollback_of is fine.
        _step(
            source_version="v2",
            target_version="v1",
            rollback_of="audit_log:v1:v2",
            status=MigrationStatus.ROLLED_BACK,
        )


# ---------------------------------------------------------------------------
# MigrationLedger basic ops
# ---------------------------------------------------------------------------


class TestMigrationLedgerBasic:
    def test_empty_ledger(self) -> None:
        ledger = MigrationLedger()
        assert len(ledger) == 0
        assert ledger.steps() == ()
        assert ledger.head_version(MigrationDomain.AUDIT_LOG) is None

    def test_record_applied_advances_head(self) -> None:
        ledger = MigrationLedger()
        step = _step(
            source_version="v1",
            target_version="v2",
            status=MigrationStatus.APPLIED,
        )
        ledger.record(step)
        assert len(ledger) == 1
        assert ledger.head_version(MigrationDomain.AUDIT_LOG) == "v2"

    def test_record_pending_does_not_advance_head(self) -> None:
        ledger = MigrationLedger()
        ledger.record(_step(status=MigrationStatus.PENDING))
        assert ledger.head_version(MigrationDomain.AUDIT_LOG) is None

    def test_record_failed_does_not_advance_head(self) -> None:
        ledger = MigrationLedger()
        # FAILED step keeps head pinned at source.
        ledger.record(
            _step(
                source_version="v1",
                target_version="v1",
                status=MigrationStatus.FAILED,
            )
        )
        assert ledger.head_version(MigrationDomain.AUDIT_LOG) is None

    def test_clear(self) -> None:
        ledger = MigrationLedger()
        ledger.record(_step(migration_id="audit_log:v1:v2"))
        ledger.clear()
        assert len(ledger) == 0
        assert ledger.head_version(MigrationDomain.AUDIT_LOG) is None
        assert ledger.get("audit_log:v1:v2") is None


class TestMigrationLedgerChainEnforcement:
    def test_applied_chain_must_match_head(self) -> None:
        ledger = MigrationLedger()
        ledger.record(_step(source_version="v1", target_version="v2"))
        # Try to apply from v1 again — chain head is at v2.
        with pytest.raises(MigrationChainError, match="chain mismatch"):
            ledger.record(_step(source_version="v1", target_version="v3"))

    def test_consecutive_applied_steps_chain(self) -> None:
        ledger = MigrationLedger()
        ledger.record(_step(source_version="v1", target_version="v2"))
        ledger.record(_step(source_version="v2", target_version="v3"))
        ledger.record(_step(source_version="v3", target_version="v4"))
        assert ledger.head_version(MigrationDomain.AUDIT_LOG) == "v4"

    def test_domains_have_independent_chains(self) -> None:
        ledger = MigrationLedger()
        ledger.record(
            _step(domain=MigrationDomain.AUDIT_LOG, source_version="v1", target_version="v2")
        )
        ledger.record(
            _step(
                domain=MigrationDomain.MEMORY_VAULT,
                source_version="2026.01",
                target_version="2026.04",
            )
        )
        assert ledger.head_version(MigrationDomain.AUDIT_LOG) == "v2"
        assert ledger.head_version(MigrationDomain.MEMORY_VAULT) == "2026.04"

    def test_pending_step_does_not_block_applied(self) -> None:
        ledger = MigrationLedger()
        ledger.record(_step(status=MigrationStatus.PENDING))
        # Head stays None — APPLIED from v1 is still legal.
        ledger.record(_step(status=MigrationStatus.APPLIED))
        assert ledger.head_version(MigrationDomain.AUDIT_LOG) == "v2"

    def test_duplicate_migration_id_rejected(self) -> None:
        ledger = MigrationLedger()
        ledger.record(_step(migration_id="audit_log:v1:v2"))
        with pytest.raises(MigrationChainError, match="duplicate migration_id"):
            ledger.record(
                _step(
                    source_version="v2",
                    target_version="v3",
                    migration_id="audit_log:v1:v2",
                )
            )


class TestMigrationLedgerRollback:
    def test_rollback_of_unknown_step_rejected(self) -> None:
        ledger = MigrationLedger()
        with pytest.raises(MigrationChainError, match="unknown migration_id"):
            ledger.record(
                _step(
                    source_version="v2",
                    target_version="v1",
                    status=MigrationStatus.ROLLED_BACK,
                    rollback_of="audit_log:v1:v2",
                )
            )

    def test_rollback_must_match_domain(self) -> None:
        ledger = MigrationLedger()
        ledger.record(
            _step(
                domain=MigrationDomain.AUDIT_LOG,
                migration_id="audit_log:v1:v2",
            )
        )
        with pytest.raises(MigrationChainError, match="domain"):
            ledger.record(
                _step(
                    domain=MigrationDomain.MEMORY_VAULT,
                    source_version="v2",
                    target_version="v1",
                    status=MigrationStatus.ROLLED_BACK,
                    rollback_of="audit_log:v1:v2",
                )
            )

    def test_rollback_of_pending_rejected(self) -> None:
        ledger = MigrationLedger()
        ledger.record(_step(status=MigrationStatus.PENDING, migration_id="audit_log:v1:v2"))
        with pytest.raises(MigrationChainError, match="only APPLIED"):
            ledger.record(
                _step(
                    source_version="v2",
                    target_version="v1",
                    status=MigrationStatus.ROLLED_BACK,
                    rollback_of="audit_log:v1:v2",
                )
            )

    def test_rollback_advances_head_back(self) -> None:
        ledger = MigrationLedger()
        ledger.record(
            _step(
                source_version="v1",
                target_version="v2",
                migration_id="audit_log:v1:v2",
            )
        )
        ledger.record(
            _step(
                source_version="v2",
                target_version="v1",
                status=MigrationStatus.ROLLED_BACK,
                rollback_of="audit_log:v1:v2",
            )
        )
        # ROLLED_BACK moves the head back to v1.
        assert ledger.head_version(MigrationDomain.AUDIT_LOG) == "v1"


class TestMigrationLedgerQueries:
    def test_for_domain(self) -> None:
        ledger = MigrationLedger()
        a = _step(domain=MigrationDomain.AUDIT_LOG, source_version="v1", target_version="v2")
        b = _step(
            domain=MigrationDomain.MEMORY_VAULT,
            source_version="2026.01",
            target_version="2026.04",
        )
        c = _step(domain=MigrationDomain.AUDIT_LOG, source_version="v2", target_version="v3")
        ledger.record(a)
        ledger.record(b)
        ledger.record(c)
        assert ledger.for_domain(MigrationDomain.AUDIT_LOG) == (a, c)
        assert ledger.for_domain(MigrationDomain.MEMORY_VAULT) == (b,)
        assert ledger.for_domain(MigrationDomain.MEMORY_EPISODIC) == ()

    def test_applied_only_filters_pending_and_failed(self) -> None:
        ledger = MigrationLedger()
        applied = _step(source_version="v1", target_version="v2")
        ledger.record(applied)
        # FAILED retry from v2.
        failed = _step(
            source_version="v2",
            target_version="v2",
            status=MigrationStatus.FAILED,
        )
        ledger.record(failed)
        # PENDING declaration of next step.
        pending = _step(
            source_version="v2",
            target_version="v3",
            status=MigrationStatus.PENDING,
        )
        ledger.record(pending)
        # APPLIED finally.
        applied2 = _step(source_version="v2", target_version="v3")
        ledger.record(applied2)
        result = ledger.applied_only(MigrationDomain.AUDIT_LOG)
        assert result == (applied, applied2)

    def test_get_by_migration_id(self) -> None:
        ledger = MigrationLedger()
        step = _step(migration_id="audit_log:v1:v2")
        ledger.record(step)
        assert ledger.get("audit_log:v1:v2") is step
        assert ledger.get("missing") is None


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


class TestMigrationLedgerSnapshot:
    def test_snapshot_empty(self) -> None:
        snap = MigrationLedger().snapshot()
        assert snap == {"head_version": {}, "steps": []}

    def test_snapshot_round_trip(self) -> None:
        ledger = MigrationLedger()
        applied = _step(
            source_version="v1",
            target_version="v2",
            applied_by="alex@cognithor.ai",
            item_count=420,
            checksum_before=_HEX,
            checksum_after=_HEX2,
            migration_id="audit_log:v1:v2",
            notes="add hash chain",
        )
        ledger.record(applied)
        snap = ledger.snapshot()
        assert snap["head_version"] == {"audit_log": "v2"}
        steps = snap["steps"]
        assert isinstance(steps, list)
        assert len(steps) == 1
        entry = steps[0]
        assert entry["domain"] == "audit_log"
        assert entry["source_version"] == "v1"
        assert entry["target_version"] == "v2"
        assert entry["status"] == "applied"
        assert entry["applied_by"] == "alex@cognithor.ai"
        assert entry["item_count"] == 420
        assert entry["checksum_before"] == _HEX
        assert entry["checksum_after"] == _HEX2
        assert entry["rollback_of"] == ""
        assert entry["migration_id"] == "audit_log:v1:v2"
        assert entry["notes"] == "add hash chain"

    def test_snapshot_head_version_sorted_by_domain(self) -> None:
        ledger = MigrationLedger()
        ledger.record(
            _step(domain=MigrationDomain.PACK_MANIFEST, source_version="v1", target_version="v2")
        )
        ledger.record(
            _step(domain=MigrationDomain.AUDIT_LOG, source_version="v1", target_version="v2")
        )
        ledger.record(
            _step(
                domain=MigrationDomain.MEMORY_VAULT,
                source_version="2026.01",
                target_version="2026.04",
            )
        )
        snap = ledger.snapshot()
        head = snap["head_version"]
        assert isinstance(head, dict)
        # Insertion order in JSON-emitted dict tracks our explicit sort by domain.value
        assert list(head.keys()) == ["audit_log", "memory_vault", "pack_manifest"]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


class TestProcessLocalLedger:
    def test_default_is_a_migration_ledger(self) -> None:
        assert isinstance(MIGRATION_LEDGER, MigrationLedger)
