"""Tests for the TRUST-1 trust-bundle composer."""

from __future__ import annotations

from cognithor.memory.provenance import (
    ExpiryPolicy,
    ProvenanceLedger,
    ProvenanceTag,
    SourceType,
)
from cognithor.models import RiskLevel
from cognithor.security.cloud_escalation import (
    EscalationEvent,
    EscalationLedger,
    EscalationReason,
)
from cognithor.security.cost_ledger import (
    CostEntry,
    CostKind,
    CostLedger,
)
from cognithor.security.fingerprint import (
    BinaryKind,
    FingerprintLedger,
    ToolFingerprint,
)
from cognithor.security.migration_ledger import (
    MigrationDomain,
    MigrationLedger,
    MigrationStatus,
    MigrationStep,
)
from cognithor.security.permission_scope import (
    PermissionScope,
    ScopeAxis,
    ScopeRegistry,
)
from cognithor.security.trust_bundle import (
    TRUST_BUNDLE_SCHEMA_VERSION,
    TrustLedgers,
    build_trust_bundle,
)

_HEX_A = "a" * 64
_HEX_B = "b" * 64


def _isolated_ledgers() -> TrustLedgers:
    """Build a TrustLedgers with all-fresh ledger instances."""
    return TrustLedgers(
        permission_scope=ScopeRegistry(),
        cost=CostLedger(),
        fingerprint=FingerprintLedger(),
        escalation=EscalationLedger(),
        provenance=ProvenanceLedger(),
        migration=MigrationLedger(),
    )


# ---------------------------------------------------------------------------
# Empty bundle
# ---------------------------------------------------------------------------


class TestEmptyBundle:
    def test_empty_inputs_produce_well_shaped_bundle(self) -> None:
        bundle = build_trust_bundle("run-42", ledgers=_isolated_ledgers())
        assert bundle["schema_version"] == TRUST_BUNDLE_SCHEMA_VERSION
        assert bundle["run_id"] == "run-42"
        assert bundle["permission_scopes"] == []
        assert bundle["cost"] == {
            "summary": {
                "entry_count": 0,
                "total_cost_usd_micro": 0,
                "total_cost_usd": 0.0,
                "by_kind": {},
                "by_tool": {},
                "by_backend": {},
            },
            "entries": [],
        }
        assert bundle["fingerprints"] == {"all": [], "divergent_names": []}
        assert bundle["escalations"]["summary"]["event_count"] == 0
        assert bundle["escalations"]["entries"] == []
        assert bundle["provenance"] == {}
        assert bundle["migrations"] == {"head_version": {}, "steps": []}

    def test_empty_run_id_keeps_run_scoped_sections_empty(self) -> None:
        # Even if ledgers contain entries, an empty run_id must NOT
        # leak every cost/escalation entry into the bundle.
        ledgers = _isolated_ledgers()
        ledgers.cost.record(
            CostEntry(
                kind=CostKind.LLM_INFERENCE,
                tool="qwen3:30b",
                cost_usd_micro=10_000,
                run_id="run-42",
            )
        )
        ledgers.escalation.record(
            EscalationEvent(
                reason=EscalationReason.OWNER_OVERRIDE,
                from_backend="ollama:qwen3:30b",
                to_backend="anthropic:claude-opus-4-7",
                prompt_tokens=100,
                response_tokens=50,
                run_id="run-42",
            )
        )
        bundle = build_trust_bundle("", ledgers=ledgers)
        assert bundle["cost"]["entries"] == []
        assert bundle["cost"]["summary"]["entry_count"] == 0
        assert bundle["escalations"]["entries"] == []
        assert bundle["escalations"]["summary"]["event_count"] == 0


# ---------------------------------------------------------------------------
# Run filtering
# ---------------------------------------------------------------------------


class TestRunScopedFiltering:
    def test_cost_filter_by_run(self) -> None:
        ledgers = _isolated_ledgers()
        ledgers.cost.record(
            CostEntry(
                kind=CostKind.LLM_INFERENCE,
                tool="qwen3:30b",
                cost_usd_micro=10_000,
                run_id="run-42",
            )
        )
        ledgers.cost.record(
            CostEntry(
                kind=CostKind.LLM_INFERENCE,
                tool="anthropic:claude-opus-4-7",
                cost_usd_micro=99_999,
                run_id="run-99",
            )
        )
        bundle = build_trust_bundle("run-42", ledgers=ledgers)
        cost = bundle["cost"]
        assert isinstance(cost, dict)
        summary = cost["summary"]
        assert isinstance(summary, dict)
        assert summary["entry_count"] == 1
        assert summary["total_cost_usd_micro"] == 10_000
        entries = cost["entries"]
        assert isinstance(entries, list)
        assert len(entries) == 1
        assert entries[0]["tool"] == "qwen3:30b"
        assert entries[0]["run_id"] == "run-42"

    def test_escalation_filter_by_run(self) -> None:
        ledgers = _isolated_ledgers()
        ledgers.escalation.record(
            EscalationEvent(
                reason=EscalationReason.OWNER_OVERRIDE,
                from_backend="ollama:qwen3:30b",
                to_backend="anthropic:claude-opus-4-7",
                prompt_tokens=1000,
                response_tokens=200,
                cost_usd_micro=15_000,
                run_id="run-42",
            )
        )
        ledgers.escalation.record(
            EscalationEvent(
                reason=EscalationReason.CONTEXT_TOO_LARGE,
                from_backend="ollama:qwen3:30b",
                to_backend="openai:gpt-5",
                prompt_tokens=20_000,
                response_tokens=2_000,
                cost_usd_micro=200_000,
                run_id="run-99",
            )
        )
        bundle = build_trust_bundle("run-42", ledgers=ledgers)
        escalations = bundle["escalations"]
        assert isinstance(escalations, dict)
        summary = escalations["summary"]
        assert isinstance(summary, dict)
        assert summary["event_count"] == 1
        assert summary["total_cost_usd_micro"] == 15_000
        assert summary["by_reason"] == {"owner_override": 1}
        entries = escalations["entries"]
        assert isinstance(entries, list)
        assert len(entries) == 1
        assert entries[0]["run_id"] == "run-42"


# ---------------------------------------------------------------------------
# Full snapshot sections
# ---------------------------------------------------------------------------


class TestFullSnapshotSections:
    def test_permission_scope_snapshot_inline(self) -> None:
        ledgers = _isolated_ledgers()
        ledgers.permission_scope.register(
            PermissionScope(
                axis=ScopeAxis.CHANNEL,
                identity="telegram",
                tool_allowlist=frozenset({"web_search", "memory_search"}),
                tool_denylist=frozenset(),
                max_risk=RiskLevel.YELLOW,
            )
        )
        ledgers.permission_scope.register(
            PermissionScope(
                axis=ScopeAxis.USER,
                identity="alex",
                tool_allowlist=frozenset(),
                tool_denylist=frozenset({"shell"}),
                max_risk=RiskLevel.RED,
            )
        )
        bundle = build_trust_bundle("run-42", ledgers=ledgers)
        scopes = bundle["permission_scopes"]
        assert isinstance(scopes, list)
        assert len(scopes) == 2
        # list_scopes() sorts by (axis, identity) — channel before user.
        assert scopes[0]["axis"] == "channel"
        assert scopes[0]["identity"] == "telegram"
        assert scopes[0]["tool_allowlist"] == ["memory_search", "web_search"]
        assert scopes[0]["tool_denylist"] == []
        assert scopes[0]["max_risk"] == "yellow"
        assert scopes[1]["axis"] == "user"
        assert scopes[1]["tool_denylist"] == ["shell"]

    def test_fingerprint_snapshot_includes_divergent_names(self) -> None:
        ledgers = _isolated_ledgers()
        ledgers.fingerprint.register(
            ToolFingerprint(
                name="web_fetch",
                kind=BinaryKind.TOOL,
                content_hash=_HEX_A,
            )
        )
        ledgers.fingerprint.register(
            ToolFingerprint(
                name="web_fetch",
                kind=BinaryKind.TOOL,
                content_hash=_HEX_B,
            )
        )
        bundle = build_trust_bundle("run-42", ledgers=ledgers)
        fingerprints = bundle["fingerprints"]
        assert isinstance(fingerprints, dict)
        all_fps = fingerprints["all"]
        assert isinstance(all_fps, list)
        assert len(all_fps) == 2
        assert fingerprints["divergent_names"] == ["web_fetch"]

    def test_provenance_snapshot_full(self) -> None:
        ledgers = _isolated_ledgers()
        ledgers.provenance.tag(
            "fact-1",
            ProvenanceTag(
                source_type=SourceType.TOOL_OUTPUT,
                source_id="audit-7",
                expiry_policy=ExpiryPolicy.PERMANENT,
            ),
        )
        bundle = build_trust_bundle("run-42", ledgers=ledgers)
        provenance = bundle["provenance"]
        assert isinstance(provenance, dict)
        assert "fact-1" in provenance
        chain = provenance["fact-1"]
        assert isinstance(chain, list)
        assert len(chain) == 1
        assert chain[0]["source_type"] == "tool_output"

    def test_migration_snapshot_full(self) -> None:
        ledgers = _isolated_ledgers()
        ledgers.migration.record(
            MigrationStep(
                domain=MigrationDomain.AUDIT_LOG,
                source_version="v1",
                target_version="v2",
                status=MigrationStatus.APPLIED,
                migration_id="audit_log:v1:v2",
            )
        )
        bundle = build_trust_bundle("run-42", ledgers=ledgers)
        migrations = bundle["migrations"]
        assert isinstance(migrations, dict)
        assert migrations["head_version"] == {"audit_log": "v2"}
        steps = migrations["steps"]
        assert isinstance(steps, list)
        assert len(steps) == 1


# ---------------------------------------------------------------------------
# Cross-ledger consistency
# ---------------------------------------------------------------------------


class TestCrossLedgerConsistency:
    def test_run_id_stitches_cost_and_escalation(self) -> None:
        # The same run_id must surface aligned data in both the cost
        # and escalation sections — that's the value-add of the bundle.
        ledgers = _isolated_ledgers()
        ledgers.cost.record(
            CostEntry(
                kind=CostKind.LLM_INFERENCE,
                tool="anthropic:claude-opus-4-7",
                cost_usd_micro=15_000,
                backend="anthropic",
                run_id="run-42",
            )
        )
        ledgers.escalation.record(
            EscalationEvent(
                reason=EscalationReason.OWNER_OVERRIDE,
                from_backend="ollama:qwen3:30b",
                to_backend="anthropic:claude-opus-4-7",
                prompt_tokens=100,
                response_tokens=50,
                cost_usd_micro=15_000,
                run_id="run-42",
            )
        )
        bundle = build_trust_bundle("run-42", ledgers=ledgers)
        cost = bundle["cost"]
        escalations = bundle["escalations"]
        assert isinstance(cost, dict)
        assert isinstance(escalations, dict)
        cost_summary = cost["summary"]
        esc_summary = escalations["summary"]
        assert isinstance(cost_summary, dict)
        assert isinstance(esc_summary, dict)
        # Same total — the operator can verify the two ledgers agree.
        assert cost_summary["total_cost_usd_micro"] == esc_summary["total_cost_usd_micro"]

    def test_canonical_ledgers_used_when_omitted(self) -> None:
        # build_trust_bundle(...) without ledgers= must reach for the
        # process-local singletons. We don't assert specific contents
        # (production state may have been wired) — just that the keys
        # are present and the schema is honoured.
        bundle = build_trust_bundle("run-non-existent")
        assert bundle["schema_version"] == TRUST_BUNDLE_SCHEMA_VERSION
        assert bundle["run_id"] == "run-non-existent"
        for key in (
            "permission_scopes",
            "cost",
            "fingerprints",
            "escalations",
            "provenance",
            "migrations",
        ):
            assert key in bundle
