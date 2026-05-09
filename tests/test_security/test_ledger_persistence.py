"""Tests for boot-time TRUST disk-persistence wiring."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

from cognithor.models import RiskLevel
from cognithor.security.backend_dispatch import (
    BACKEND_DISPATCH_LEDGER,
    BackendDispatchEvent,
    DispatchOutcome,
)
from cognithor.security.cloud_escalation import (
    ESCALATION_LEDGER,
    EscalationEvent,
    EscalationReason,
)
from cognithor.security.cost_ledger import (
    COST_LEDGER,
    CostEntry,
    CostKind,
)
from cognithor.security.fingerprint import (
    FINGERPRINT_LEDGER,
    BinaryKind,
    ToolFingerprint,
)
from cognithor.security.ledger_persistence import (
    CanonicalStores,
    default_audit_dir,
    open_canonical_stores_and_bind,
    reset_for_tests,
)
from cognithor.security.migration_ledger import (
    MIGRATION_LEDGER,
    MigrationDomain,
    MigrationStatus,
    MigrationStep,
)
from cognithor.security.permission_scope import (
    SCOPE_REGISTRY,
    PermissionScope,
    ScopeAxis,
)


@pytest.fixture(autouse=True)
def _reset_state(tmp_path):
    """Each test gets a fresh audit dir and a clean global bind state."""
    reset_for_tests()
    BACKEND_DISPATCH_LEDGER.clear()
    ESCALATION_LEDGER.clear()
    FINGERPRINT_LEDGER.clear()
    COST_LEDGER.clear()
    MIGRATION_LEDGER.clear()
    SCOPE_REGISTRY.clear()
    yield
    reset_for_tests()
    BACKEND_DISPATCH_LEDGER.clear()
    ESCALATION_LEDGER.clear()
    FINGERPRINT_LEDGER.clear()
    COST_LEDGER.clear()
    MIGRATION_LEDGER.clear()
    SCOPE_REGISTRY.clear()


def _hash(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Bundle / lifecycle
# ---------------------------------------------------------------------------


class TestBundleLifecycle:
    def test_open_creates_audit_files(self, tmp_path) -> None:
        bundle = open_canonical_stores_and_bind(audit_dir=tmp_path)
        assert isinstance(bundle, CanonicalStores)
        assert bundle.errors == {}
        for name in (
            "backend_dispatch.sqlite",
            "cloud_escalation.sqlite",
            "fingerprints.sqlite",
            "cost_ledger.sqlite",
            "migrations.sqlite",
            "scope_registry.sqlite",
        ):
            assert (tmp_path / name).exists(), f"{name} not created"

    def test_double_call_is_idempotent(self, tmp_path) -> None:
        first = open_canonical_stores_and_bind(audit_dir=tmp_path)
        second = open_canonical_stores_and_bind(audit_dir=tmp_path)
        assert first is second

    def test_default_audit_dir_creates_under_home(self, tmp_path, monkeypatch) -> None:
        """Production callers go through ``default_audit_dir()`` —
        verify it builds the canonical layout under ``~/.cognithor/audit/``."""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        # On Windows ``Path.home()`` reads USERPROFILE; on POSIX, HOME.
        target = default_audit_dir()
        assert target.exists()
        assert target.name == "audit"
        assert target.parent.name == ".cognithor"


# ---------------------------------------------------------------------------
# Write-through — append-style ledgers
# ---------------------------------------------------------------------------


class TestWriteThroughAppendLedgers:
    def test_dispatch_record_persists_to_disk(self, tmp_path) -> None:
        bundle = open_canonical_stores_and_bind(audit_dir=tmp_path)
        ev = BackendDispatchEvent(
            backend_type="ollama",
            model="qwen3:30b",
            outcome=DispatchOutcome.SUCCESS,
            prompt_tokens=100,
            response_tokens=50,
            started_at=datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC),
            completed_at=datetime(2026, 5, 9, 12, 0, 1, tzinfo=UTC),
            run_id="run-1",
        )
        BACKEND_DISPATCH_LEDGER.record(ev)
        assert bundle.dispatch is not None
        assert len(bundle.dispatch) == 1
        recovered = bundle.dispatch.events()[0]
        assert recovered.run_id == "run-1"
        assert recovered.outcome == DispatchOutcome.SUCCESS

    def test_escalation_record_persists_to_disk(self, tmp_path) -> None:
        bundle = open_canonical_stores_and_bind(audit_dir=tmp_path)
        ev = EscalationEvent(
            reason=EscalationReason.OWNER_OVERRIDE,
            from_backend="ollama",
            to_backend="anthropic",
            prompt_tokens=200,
            response_tokens=80,
            started_at=datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC),
            completed_at=datetime(2026, 5, 9, 12, 0, 5, tzinfo=UTC),
            run_id="run-1",
        )
        ESCALATION_LEDGER.record(ev)
        assert bundle.escalation is not None
        assert len(bundle.escalation) == 1
        assert bundle.escalation.events()[0].run_id == "run-1"

    def test_cost_record_persists_to_disk(self, tmp_path) -> None:
        bundle = open_canonical_stores_and_bind(audit_dir=tmp_path)
        e = CostEntry(
            kind=CostKind.LLM_INFERENCE,
            tool="qwen3:30b",
            cost_usd_micro=1500,
            backend="ollama",
            run_id="run-1",
        )
        COST_LEDGER.record(e)
        assert bundle.cost is not None
        assert len(bundle.cost) == 1


# ---------------------------------------------------------------------------
# Write-through — registry-style ledgers
# ---------------------------------------------------------------------------


class TestWriteThroughRegistryLedgers:
    def test_fingerprint_register_persists(self, tmp_path) -> None:
        bundle = open_canonical_stores_and_bind(audit_dir=tmp_path)
        fp = ToolFingerprint(
            name="ollama",
            kind=BinaryKind.BINARY,
            content_hash=_hash("ollama-bytes"),
            version="0.5.7",
        )
        FINGERPRINT_LEDGER.register(fp)
        assert bundle.fingerprint is not None
        assert len(bundle.fingerprint) == 1
        assert bundle.fingerprint.get(fp.content_hash) == fp

    def test_scope_register_persists(self, tmp_path) -> None:
        bundle = open_canonical_stores_and_bind(audit_dir=tmp_path)
        scope = PermissionScope(
            axis=ScopeAxis.CHANNEL,
            identity="telegram",
            tool_allowlist=frozenset({"web_fetch"}),
            max_risk=RiskLevel.YELLOW,
        )
        SCOPE_REGISTRY.register(scope)
        assert bundle.scope is not None
        assert len(bundle.scope) == 1
        recovered = bundle.scope.get(ScopeAxis.CHANNEL, "telegram")
        assert recovered == scope

    def test_scope_remove_persists(self, tmp_path) -> None:
        bundle = open_canonical_stores_and_bind(audit_dir=tmp_path)
        scope = PermissionScope(axis=ScopeAxis.CHANNEL, identity="telegram")
        SCOPE_REGISTRY.register(scope)
        assert bundle.scope is not None
        assert len(bundle.scope) == 1
        SCOPE_REGISTRY.remove(ScopeAxis.CHANNEL, "telegram")
        assert len(bundle.scope) == 0

    def test_migration_record_persists(self, tmp_path) -> None:
        bundle = open_canonical_stores_and_bind(audit_dir=tmp_path)
        step = MigrationStep(
            domain=MigrationDomain.MEMORY_VAULT,
            source_version="v0",
            target_version="v1",
            status=MigrationStatus.APPLIED,
            applied_by="system",
            migration_id="memory_vault:v0:v1",
        )
        MIGRATION_LEDGER.record(step)
        assert bundle.migration is not None
        # Disk has at least the new step (plus any TRUST-10 self-audit
        # emitter steps replayed at bind time).
        recovered = bundle.migration.get("memory_vault:v0:v1")
        assert recovered is not None
        assert recovered.target_version == "v1"


# ---------------------------------------------------------------------------
# Cross-restart seeding (registry-style ledgers)
# ---------------------------------------------------------------------------


class TestCrossRestartSeeding:
    def test_fingerprint_seeded_from_disk_on_second_open(self, tmp_path) -> None:
        # First boot: register a fingerprint
        open_canonical_stores_and_bind(audit_dir=tmp_path)
        fp = ToolFingerprint(
            name="ollama",
            kind=BinaryKind.BINARY,
            content_hash=_hash("ollama-v1"),
            version="0.5.7",
        )
        FINGERPRINT_LEDGER.register(fp)
        assert fp.content_hash in FINGERPRINT_LEDGER

        # Simulate restart: reset bind state + clear the in-memory ledger
        reset_for_tests()
        FINGERPRINT_LEDGER.clear()
        assert fp.content_hash not in FINGERPRINT_LEDGER

        # Second boot: in-memory ledger should be re-seeded from disk
        open_canonical_stores_and_bind(audit_dir=tmp_path)
        assert fp.content_hash in FINGERPRINT_LEDGER

    def test_scope_seeded_from_disk_on_second_open(self, tmp_path) -> None:
        open_canonical_stores_and_bind(audit_dir=tmp_path)
        scope = PermissionScope(
            axis=ScopeAxis.CHANNEL,
            identity="telegram",
            max_risk=RiskLevel.YELLOW,
        )
        SCOPE_REGISTRY.register(scope)

        reset_for_tests()
        SCOPE_REGISTRY.clear()
        assert SCOPE_REGISTRY.get(ScopeAxis.CHANNEL, "telegram") is None

        open_canonical_stores_and_bind(audit_dir=tmp_path)
        seeded = SCOPE_REGISTRY.get(ScopeAxis.CHANNEL, "telegram")
        assert seeded is not None
        assert seeded.max_risk == RiskLevel.YELLOW

    def test_migration_replay_persists_pre_bind_steps(self, tmp_path) -> None:
        """A migration step appended to MIGRATION_LEDGER BEFORE
        ``open_canonical_stores_and_bind`` runs must still reach
        disk via the replay step. This is the TRUST-10 self-audit
        emitter use case: many security modules call
        ``_record_*_migration()`` at import time."""
        # Pre-bind: emit a step into the in-memory ledger
        step = MigrationStep(
            domain=MigrationDomain.MEMORY_VAULT,
            source_version="v0",
            target_version="v1",
            status=MigrationStatus.APPLIED,
            applied_by="system",
            migration_id="pre-bind-replay-target",
        )
        MIGRATION_LEDGER.record(step)
        assert len(MIGRATION_LEDGER) >= 1

        # Bind with a fresh (empty) disk store — the replay step
        # should pull the in-memory step onto disk.
        bundle = open_canonical_stores_and_bind(audit_dir=tmp_path)
        assert bundle.migration is not None
        recovered = bundle.migration.get("pre-bind-replay-target")
        assert recovered is not None


# ---------------------------------------------------------------------------
# Best-effort error suppression
# ---------------------------------------------------------------------------


class TestErrorSuppression:
    def test_disk_write_failure_does_not_break_in_memory_ledger(self, tmp_path) -> None:
        """If the disk store raises, the in-memory ledger MUST still
        accept the write. This is the boot-survival contract."""
        bundle = open_canonical_stores_and_bind(audit_dir=tmp_path)
        assert bundle.dispatch is not None

        # Force the disk store into a bad state — close the underlying
        # connection mid-flight. Subsequent writes raise sqlite3.ProgrammingError.
        bundle.dispatch.close()

        ev = BackendDispatchEvent(
            backend_type="ollama",
            model="qwen3:30b",
            outcome=DispatchOutcome.SUCCESS,
            prompt_tokens=10,
            response_tokens=5,
            started_at=datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC),
            completed_at=datetime(2026, 5, 9, 12, 0, 1, tzinfo=UTC),
        )
        BACKEND_DISPATCH_LEDGER.record(ev)  # must not raise
        assert len(BACKEND_DISPATCH_LEDGER) == 1
