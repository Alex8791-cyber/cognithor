"""Tests for the TRUST-9 ProvenanceTag foundation."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

import pytest

from cognithor.memory.provenance import (
    PROVENANCE_LEDGER,
    ExpiryPolicy,
    ProvenanceLedger,
    ProvenanceTag,
    SourceType,
    make_ttl_tag,
)


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


# ---------------------------------------------------------------------------
# ProvenanceTag construction + validation
# ---------------------------------------------------------------------------


class TestProvenanceTagBasics:
    def test_minimal_tag(self) -> None:
        tag = ProvenanceTag(
            source_type=SourceType.CHAT_UTTERANCE,
            source_id="msg-42",
        )
        assert tag.source_type == SourceType.CHAT_UTTERANCE
        assert tag.source_id == "msg-42"
        assert tag.expiry_policy == ExpiryPolicy.PERMANENT
        assert tag.confidence == 1.0
        assert tag.attribution_chain == ()
        assert tag.notes == ""
        # ingested_at defaults to now(UTC)
        assert tag.ingested_at.tzinfo == UTC

    def test_frozen_via_dataclass(self) -> None:
        tag = ProvenanceTag(source_type=SourceType.TOOL_OUTPUT, source_id="audit-1")
        with pytest.raises(dataclasses.FrozenInstanceError):
            tag.source_id = "audit-2"  # type: ignore[misc]

    def test_effective_valid_from_falls_back_to_ingested_at(self) -> None:
        ingested = _utc(2026, 5, 4, 12, 0)
        tag = ProvenanceTag(
            source_type=SourceType.TOOL_OUTPUT,
            source_id="audit-1",
            ingested_at=ingested,
        )
        assert tag.effective_valid_from == ingested

    def test_effective_valid_from_uses_explicit_value(self) -> None:
        ingested = _utc(2026, 5, 4, 12, 0)
        valid_from = _utc(2026, 5, 1, 0, 0)
        tag = ProvenanceTag(
            source_type=SourceType.CONFIG_IMPORT,
            source_id="config.yaml",
            ingested_at=ingested,
            valid_from=valid_from,
        )
        assert tag.effective_valid_from == valid_from

    def test_with_chain_appends_parents(self) -> None:
        base = ProvenanceTag(
            source_type=SourceType.AGENT_INFERENCE,
            source_id="run-1",
            attribution_chain=("a",),
        )
        derived = base.with_chain("b", "c")
        assert derived.attribution_chain == ("a", "b", "c")
        # original is unchanged
        assert base.attribution_chain == ("a",)


class TestProvenanceTagValidation:
    def test_empty_source_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            ProvenanceTag(source_type=SourceType.UNKNOWN, source_id="")

    def test_confidence_below_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"\[0.0, 1.0\]"):
            ProvenanceTag(
                source_type=SourceType.AGENT_INFERENCE,
                source_id="run-1",
                confidence=-0.1,
            )

    def test_confidence_above_one_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"\[0.0, 1.0\]"):
            ProvenanceTag(
                source_type=SourceType.AGENT_INFERENCE,
                source_id="run-1",
                confidence=1.5,
            )

    def test_ttl_without_valid_until_rejected(self) -> None:
        with pytest.raises(ValueError, match="valid_until"):
            ProvenanceTag(
                source_type=SourceType.TOOL_OUTPUT,
                source_id="audit-1",
                expiry_policy=ExpiryPolicy.TTL,
            )

    def test_valid_from_after_valid_until_rejected(self) -> None:
        with pytest.raises(ValueError, match="valid_from"):
            ProvenanceTag(
                source_type=SourceType.CONFIG_IMPORT,
                source_id="config.yaml",
                valid_from=_utc(2026, 5, 5),
                valid_until=_utc(2026, 5, 4),
            )

    def test_valid_from_equal_to_valid_until_allowed(self) -> None:
        # Edge case — single-instant validity is legal.
        instant = _utc(2026, 5, 4, 12, 0)
        tag = ProvenanceTag(
            source_type=SourceType.SCHEDULED_INGEST,
            source_id="cron-1",
            valid_from=instant,
            valid_until=instant,
            expiry_policy=ExpiryPolicy.TTL,
        )
        assert tag.valid_from == tag.valid_until


# ---------------------------------------------------------------------------
# ProvenanceLedger basic ops
# ---------------------------------------------------------------------------


class TestProvenanceLedgerBasic:
    def test_empty_ledger(self) -> None:
        ledger = ProvenanceLedger()
        assert len(ledger) == 0
        assert ledger.current("missing") is None
        assert ledger.chain("missing") == ()
        assert "missing" not in ledger

    def test_tag_appends_to_chain(self) -> None:
        ledger = ProvenanceLedger()
        tag = ProvenanceTag(source_type=SourceType.TOOL_OUTPUT, source_id="audit-1")
        ledger.tag("item-1", tag)
        assert len(ledger) == 1
        assert "item-1" in ledger
        assert ledger.current("item-1") is tag
        assert ledger.chain("item-1") == (tag,)

    def test_tag_empty_item_id_rejected(self) -> None:
        ledger = ProvenanceLedger()
        tag = ProvenanceTag(source_type=SourceType.UNKNOWN, source_id="x")
        with pytest.raises(ValueError, match="non-empty"):
            ledger.tag("", tag)

    def test_tag_appends_history(self) -> None:
        ledger = ProvenanceLedger()
        t1 = ProvenanceTag(source_type=SourceType.CHAT_UTTERANCE, source_id="msg-1")
        t2 = ProvenanceTag(source_type=SourceType.AGENT_INFERENCE, source_id="run-1")
        ledger.tag("item-1", t1)
        ledger.tag("item-1", t2)
        assert ledger.chain("item-1") == (t1, t2)
        assert ledger.current("item-1") is t2

    def test_remove_existing(self) -> None:
        ledger = ProvenanceLedger()
        ledger.tag(
            "item-1",
            ProvenanceTag(source_type=SourceType.UNKNOWN, source_id="x"),
        )
        assert ledger.remove("item-1") is True
        assert "item-1" not in ledger

    def test_remove_missing_returns_false(self) -> None:
        ledger = ProvenanceLedger()
        assert ledger.remove("nope") is False

    def test_clear(self) -> None:
        ledger = ProvenanceLedger()
        ledger.tag(
            "a",
            ProvenanceTag(source_type=SourceType.UNKNOWN, source_id="x"),
        )
        ledger.tag(
            "b",
            ProvenanceTag(source_type=SourceType.UNKNOWN, source_id="y"),
        )
        ledger.clear()
        assert len(ledger) == 0

    def test_items_sorted_by_id(self) -> None:
        ledger = ProvenanceLedger()
        for item_id in ("zeta", "alpha", "mu"):
            ledger.tag(
                item_id,
                ProvenanceTag(source_type=SourceType.UNKNOWN, source_id=item_id),
            )
        assert [k for k, _ in ledger.items()] == ["alpha", "mu", "zeta"]


# ---------------------------------------------------------------------------
# Expiry semantics
# ---------------------------------------------------------------------------


class TestProvenanceLedgerExpiry:
    def test_permanent_never_expires(self) -> None:
        ledger = ProvenanceLedger()
        ledger.tag(
            "item-1",
            ProvenanceTag(
                source_type=SourceType.CONFIG_IMPORT,
                source_id="config.yaml",
                expiry_policy=ExpiryPolicy.PERMANENT,
            ),
        )
        far_future = _utc(2099, 1, 1)
        assert ledger.expired(now=far_future) == []

    def test_ttl_expires_after_valid_until(self) -> None:
        ledger = ProvenanceLedger()
        ledger.tag(
            "item-1",
            ProvenanceTag(
                source_type=SourceType.TOOL_OUTPUT,
                source_id="audit-1",
                valid_from=_utc(2026, 5, 4, 0, 0),
                valid_until=_utc(2026, 5, 4, 12, 0),
                expiry_policy=ExpiryPolicy.TTL,
            ),
        )
        assert ledger.expired(now=_utc(2026, 5, 4, 11, 0)) == []
        assert ledger.expired(now=_utc(2026, 5, 4, 13, 0)) == ["item-1"]

    def test_ttl_at_boundary_not_expired(self) -> None:
        ledger = ProvenanceLedger()
        boundary = _utc(2026, 5, 4, 12, 0)
        ledger.tag(
            "item-1",
            ProvenanceTag(
                source_type=SourceType.TOOL_OUTPUT,
                source_id="audit-1",
                valid_from=_utc(2026, 5, 4, 0, 0),
                valid_until=boundary,
                expiry_policy=ExpiryPolicy.TTL,
            ),
        )
        # cutoff > valid_until is the expiry condition; at-boundary stays fresh.
        assert ledger.expired(now=boundary) == []

    def test_manual_never_auto_expires(self) -> None:
        ledger = ProvenanceLedger()
        ledger.tag(
            "item-1",
            ProvenanceTag(
                source_type=SourceType.USER_DIRECTIVE,
                source_id="owner-says",
                valid_until=_utc(2020, 1, 1),
                expiry_policy=ExpiryPolicy.MANUAL,
            ),
        )
        assert ledger.expired(now=_utc(2099, 1, 1)) == []

    def test_replace_on_new_not_reported_by_expired(self) -> None:
        ledger = ProvenanceLedger()
        old = ProvenanceTag(
            source_type=SourceType.AGENT_INFERENCE,
            source_id="run-1",
            expiry_policy=ExpiryPolicy.REPLACE_ON_NEW,
        )
        new = ProvenanceTag(
            source_type=SourceType.AGENT_INFERENCE,
            source_id="run-2",
            expiry_policy=ExpiryPolicy.REPLACE_ON_NEW,
        )
        ledger.tag("item-1", old)
        ledger.tag("item-1", new)
        # Head is fresh — superseded() reports the history.
        assert ledger.expired(now=_utc(2099, 1, 1)) == []
        assert ledger.superseded("item-1") == (old,)

    def test_expired_default_now(self) -> None:
        ledger = ProvenanceLedger()
        ledger.tag(
            "item-1",
            ProvenanceTag(
                source_type=SourceType.TOOL_OUTPUT,
                source_id="audit-1",
                valid_until=_utc(2020, 1, 1),
                expiry_policy=ExpiryPolicy.TTL,
            ),
        )
        # Default now() is real-time and definitely past 2020-01-01.
        assert ledger.expired() == ["item-1"]

    def test_expired_returns_sorted_ids(self) -> None:
        ledger = ProvenanceLedger()
        for item_id in ("zeta", "alpha", "mu"):
            ledger.tag(
                item_id,
                ProvenanceTag(
                    source_type=SourceType.TOOL_OUTPUT,
                    source_id=item_id,
                    valid_until=_utc(2020, 1, 1),
                    expiry_policy=ExpiryPolicy.TTL,
                ),
            )
        assert ledger.expired(now=_utc(2099, 1, 1)) == ["alpha", "mu", "zeta"]

    def test_superseded_single_tag_returns_empty(self) -> None:
        ledger = ProvenanceLedger()
        ledger.tag(
            "item-1",
            ProvenanceTag(source_type=SourceType.UNKNOWN, source_id="x"),
        )
        assert ledger.superseded("item-1") == ()

    def test_superseded_unknown_item(self) -> None:
        ledger = ProvenanceLedger()
        assert ledger.superseded("missing") == ()


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


class TestProvenanceLedgerFilter:
    def test_filter_by_source_type(self) -> None:
        ledger = ProvenanceLedger()
        ledger.tag(
            "a",
            ProvenanceTag(source_type=SourceType.TOOL_OUTPUT, source_id="audit-1"),
        )
        ledger.tag(
            "b",
            ProvenanceTag(source_type=SourceType.CHAT_UTTERANCE, source_id="msg-1"),
        )
        ledger.tag(
            "c",
            ProvenanceTag(source_type=SourceType.TOOL_OUTPUT, source_id="audit-2"),
        )
        result = ledger.filter_by_source_type(SourceType.TOOL_OUTPUT)
        assert [item_id for item_id, _ in result] == ["a", "c"]
        assert all(tag.source_type == SourceType.TOOL_OUTPUT for _, tag in result)

    def test_filter_by_source_id(self) -> None:
        ledger = ProvenanceLedger()
        # Two memory items derived from the same audit-log run.
        ledger.tag(
            "fact-1",
            ProvenanceTag(source_type=SourceType.TOOL_OUTPUT, source_id="run-42"),
        )
        ledger.tag(
            "fact-2",
            ProvenanceTag(source_type=SourceType.TOOL_OUTPUT, source_id="run-42"),
        )
        ledger.tag(
            "fact-3",
            ProvenanceTag(source_type=SourceType.TOOL_OUTPUT, source_id="run-99"),
        )
        assert ledger.filter_by_source_id("run-42") == ["fact-1", "fact-2"]

    def test_filter_uses_current_tag_only(self) -> None:
        ledger = ProvenanceLedger()
        ledger.tag(
            "item-1",
            ProvenanceTag(source_type=SourceType.CHAT_UTTERANCE, source_id="msg-1"),
        )
        ledger.tag(
            "item-1",
            ProvenanceTag(source_type=SourceType.TOOL_OUTPUT, source_id="audit-1"),
        )
        # Earlier CHAT_UTTERANCE tag is in the chain but not the head.
        assert ledger.filter_by_source_type(SourceType.CHAT_UTTERANCE) == []
        assert [item_id for item_id, _ in ledger.filter_by_source_type(SourceType.TOOL_OUTPUT)] == [
            "item-1"
        ]


# ---------------------------------------------------------------------------
# Snapshot serialisation
# ---------------------------------------------------------------------------


class TestProvenanceLedgerSnapshot:
    def test_snapshot_empty(self) -> None:
        assert ProvenanceLedger().snapshot() == {}

    def test_snapshot_round_trip_shape(self) -> None:
        ledger = ProvenanceLedger()
        ingested = _utc(2026, 5, 4, 12, 0)
        valid_from = _utc(2026, 5, 4, 0, 0)
        valid_until = _utc(2026, 5, 5, 0, 0)
        tag = ProvenanceTag(
            source_type=SourceType.TOOL_OUTPUT,
            source_id="audit-1",
            source_url="https://example.test/audit/1",
            ingested_at=ingested,
            valid_from=valid_from,
            valid_until=valid_until,
            expiry_policy=ExpiryPolicy.TTL,
            confidence=0.85,
            attribution_chain=("parent-1", "parent-2"),
            notes="probe",
        )
        ledger.tag("item-1", tag)
        snap = ledger.snapshot()
        assert list(snap.keys()) == ["item-1"]
        entry = snap["item-1"][0]
        assert entry["source_type"] == "tool_output"
        assert entry["source_id"] == "audit-1"
        assert entry["source_url"] == "https://example.test/audit/1"
        assert entry["ingested_at"] == ingested.isoformat()
        assert entry["valid_from"] == valid_from.isoformat()
        assert entry["valid_until"] == valid_until.isoformat()
        assert entry["expiry_policy"] == "ttl"
        assert entry["confidence"] == 0.85
        assert entry["attribution_chain"] == ["parent-1", "parent-2"]
        assert entry["notes"] == "probe"

    def test_snapshot_handles_none_dates(self) -> None:
        ledger = ProvenanceLedger()
        ledger.tag(
            "item-1",
            ProvenanceTag(
                source_type=SourceType.CONFIG_IMPORT,
                source_id="config.yaml",
            ),
        )
        entry = ledger.snapshot()["item-1"][0]
        assert entry["valid_from"] is None
        assert entry["valid_until"] is None

    def test_snapshot_emits_full_chain_oldest_first(self) -> None:
        ledger = ProvenanceLedger()
        ledger.tag(
            "item-1",
            ProvenanceTag(source_type=SourceType.CHAT_UTTERANCE, source_id="msg-1"),
        )
        ledger.tag(
            "item-1",
            ProvenanceTag(source_type=SourceType.AGENT_INFERENCE, source_id="run-1"),
        )
        chain = ledger.snapshot()["item-1"]
        assert len(chain) == 2
        assert chain[0]["source_id"] == "msg-1"
        assert chain[1]["source_id"] == "run-1"

    def test_snapshot_keys_are_sorted(self) -> None:
        ledger = ProvenanceLedger()
        for item_id in ("z", "a", "m"):
            ledger.tag(
                item_id,
                ProvenanceTag(source_type=SourceType.UNKNOWN, source_id=item_id),
            )
        assert list(ledger.snapshot().keys()) == ["a", "m", "z"]


# ---------------------------------------------------------------------------
# make_ttl_tag helper
# ---------------------------------------------------------------------------


class TestMakeTtlTag:
    def test_builds_ttl_policy(self) -> None:
        tag = make_ttl_tag(
            source_type=SourceType.AGENT_INFERENCE,
            source_id="run-42",
            ttl=timedelta(hours=24),
            confidence=0.7,
            notes="extracted user timezone",
        )
        assert tag.expiry_policy == ExpiryPolicy.TTL
        assert tag.valid_until is not None
        assert tag.valid_from is not None
        assert tag.valid_until - tag.valid_from == timedelta(hours=24)
        assert tag.confidence == 0.7
        assert tag.notes == "extracted user timezone"

    def test_zero_ttl_allowed(self) -> None:
        # A zero TTL is degenerate but not invalid — used as a fence
        # by callers that immediately re-tag.
        tag = make_ttl_tag(
            source_type=SourceType.UNKNOWN,
            source_id="probe",
            ttl=timedelta(seconds=0),
        )
        assert tag.valid_from == tag.valid_until

    def test_negative_ttl_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            make_ttl_tag(
                source_type=SourceType.UNKNOWN,
                source_id="probe",
                ttl=timedelta(seconds=-1),
            )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


class TestProcessLocalLedger:
    def test_default_ledger_is_a_provenance_ledger(self) -> None:
        assert isinstance(PROVENANCE_LEDGER, ProvenanceLedger)


class TestProvenanceLedgerSelfAuditMigration:
    """TRUST-10 self-audit: importing the provenance module records
    a v0→v1 migration step into the canonical MIGRATION_LEDGER.
    """

    def test_migration_step_landed(self) -> None:
        from cognithor.security.migration_ledger import (
            MIGRATION_LEDGER,
            MigrationDomain,
            MigrationStatus,
        )

        # The module-level _record_provenance_ledger_migration() ran
        # at import time. Look the step up by its stable id.
        step = MIGRATION_LEDGER.get("provenance_ledger:v0-no-ledger:v1-append-only-ledger")
        assert step is not None
        assert step.status == MigrationStatus.APPLIED
        assert step.domain == MigrationDomain.PROVENANCE_LEDGER
        assert step.applied_by == "system"
        assert (
            MIGRATION_LEDGER.head_version(MigrationDomain.PROVENANCE_LEDGER)
            == "v1-append-only-ledger"
        )

    def test_repeated_calls_idempotent(self) -> None:
        # Re-running the recorder must NOT raise — duplicate
        # migration_id is suppressed.
        from cognithor.memory.provenance import (
            _record_provenance_ledger_migration,
        )

        _record_provenance_ledger_migration()
        _record_provenance_ledger_migration()
        _record_provenance_ledger_migration()
