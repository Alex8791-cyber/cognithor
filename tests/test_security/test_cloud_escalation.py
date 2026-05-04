"""Tests for the TRUST-8 cloud-escalation ledger foundation."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

import pytest

from cognithor.security.cloud_escalation import (
    ESCALATION_LEDGER,
    EscalationEvent,
    EscalationLedger,
    EscalationReason,
    EscalationSummary,
)


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def _event(
    *,
    reason: EscalationReason = EscalationReason.CONTEXT_TOO_LARGE,
    from_backend: str = "ollama:qwen3:30b",
    to_backend: str = "anthropic:claude-opus-4-7",
    prompt_tokens: int = 1000,
    response_tokens: int = 200,
    cost_usd_micro: int = 12_000,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    owner_consented: bool = False,
    run_id: str = "",
    request_id: str = "",
    notes: str = "",
) -> EscalationEvent:
    """Test helper — minimal event with sane defaults."""
    return EscalationEvent(
        reason=reason,
        from_backend=from_backend,
        to_backend=to_backend,
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
        cost_usd_micro=cost_usd_micro,
        started_at=started_at if started_at is not None else _utc(2026, 5, 4, 12, 0),
        completed_at=completed_at,
        owner_consented=owner_consented,
        run_id=run_id,
        request_id=request_id,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# EscalationEvent construction + validation
# ---------------------------------------------------------------------------


class TestEscalationEventBasics:
    def test_minimal_event(self) -> None:
        ev = EscalationEvent(
            reason=EscalationReason.OWNER_OVERRIDE,
            from_backend="ollama:qwen3:30b",
            to_backend="anthropic:claude-opus-4-7",
            prompt_tokens=512,
            response_tokens=128,
        )
        assert ev.cost_usd_micro == 0
        assert ev.cost_usd == 0.0
        assert ev.total_tokens == 640
        assert ev.completed_at is None
        assert ev.owner_consented is False
        assert ev.started_at.tzinfo == UTC

    def test_cost_usd_property(self) -> None:
        ev = _event(cost_usd_micro=2_500_000)  # $2.50
        assert ev.cost_usd == 2.5

    def test_frozen_via_dataclass(self) -> None:
        ev = _event()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ev.notes = "tamper"  # type: ignore[misc]


class TestEscalationEventValidation:
    def test_empty_from_backend_rejected(self) -> None:
        with pytest.raises(ValueError, match="from_backend"):
            EscalationEvent(
                reason=EscalationReason.UNKNOWN,
                from_backend="",
                to_backend="anthropic:claude-opus-4-7",
                prompt_tokens=0,
                response_tokens=0,
            )

    def test_empty_to_backend_rejected(self) -> None:
        with pytest.raises(ValueError, match="to_backend"):
            EscalationEvent(
                reason=EscalationReason.UNKNOWN,
                from_backend="ollama:x",
                to_backend="",
                prompt_tokens=0,
                response_tokens=0,
            )

    def test_negative_prompt_tokens_rejected(self) -> None:
        with pytest.raises(ValueError, match="prompt_tokens"):
            _event(prompt_tokens=-1)

    def test_negative_response_tokens_rejected(self) -> None:
        with pytest.raises(ValueError, match="response_tokens"):
            _event(response_tokens=-1)

    def test_negative_cost_rejected(self) -> None:
        with pytest.raises(ValueError, match="cost_usd_micro"):
            _event(cost_usd_micro=-1)

    def test_completed_before_started_rejected(self) -> None:
        with pytest.raises(ValueError, match="completed_at"):
            _event(
                started_at=_utc(2026, 5, 4, 13, 0),
                completed_at=_utc(2026, 5, 4, 12, 0),
            )

    def test_completed_equal_started_allowed(self) -> None:
        # Edge case — instantaneous escalation (cached response).
        instant = _utc(2026, 5, 4, 12, 0)
        ev = _event(started_at=instant, completed_at=instant)
        assert ev.completed_at == ev.started_at


# ---------------------------------------------------------------------------
# EscalationLedger basic ops
# ---------------------------------------------------------------------------


class TestEscalationLedgerBasic:
    def test_empty_ledger(self) -> None:
        ledger = EscalationLedger()
        assert len(ledger) == 0
        assert ledger.events() == ()
        assert ledger.by_reason(EscalationReason.UNKNOWN) == ()
        assert ledger.by_destination("anthropic:claude-opus-4-7") == ()
        assert ledger.by_run("run-1") == ()

    def test_record_appends(self) -> None:
        ledger = EscalationLedger()
        ev = _event()
        ledger.record(ev)
        assert len(ledger) == 1
        assert ledger.events() == (ev,)

    def test_record_preserves_insertion_order(self) -> None:
        ledger = EscalationLedger()
        e1 = _event(notes="first", started_at=_utc(2026, 5, 4, 10, 0))
        e2 = _event(notes="second", started_at=_utc(2026, 5, 4, 11, 0))
        e3 = _event(notes="third", started_at=_utc(2026, 5, 4, 12, 0))
        ledger.record(e1)
        ledger.record(e2)
        ledger.record(e3)
        assert ledger.events() == (e1, e2, e3)

    def test_clear(self) -> None:
        ledger = EscalationLedger()
        ledger.record(_event())
        ledger.record(_event())
        ledger.clear()
        assert len(ledger) == 0


class TestEscalationLedgerFilter:
    def test_by_reason(self) -> None:
        ledger = EscalationLedger()
        owner = _event(reason=EscalationReason.OWNER_OVERRIDE, notes="owner")
        oversized = _event(reason=EscalationReason.CONTEXT_TOO_LARGE, notes="big")
        owner2 = _event(reason=EscalationReason.OWNER_OVERRIDE, notes="owner2")
        ledger.record(owner)
        ledger.record(oversized)
        ledger.record(owner2)
        assert ledger.by_reason(EscalationReason.OWNER_OVERRIDE) == (owner, owner2)
        assert ledger.by_reason(EscalationReason.CONTEXT_TOO_LARGE) == (oversized,)
        assert ledger.by_reason(EscalationReason.UNKNOWN) == ()

    def test_by_destination(self) -> None:
        ledger = EscalationLedger()
        e1 = _event(to_backend="anthropic:claude-opus-4-7")
        e2 = _event(to_backend="openai:gpt-5")
        e3 = _event(to_backend="anthropic:claude-opus-4-7")
        ledger.record(e1)
        ledger.record(e2)
        ledger.record(e3)
        assert ledger.by_destination("anthropic:claude-opus-4-7") == (e1, e3)
        assert ledger.by_destination("openai:gpt-5") == (e2,)
        assert ledger.by_destination("missing") == ()

    def test_by_run_skips_empty_run_id(self) -> None:
        ledger = EscalationLedger()
        ledger.record(_event(run_id=""))
        # An empty run_id query must NOT match the empty-run_id events,
        # otherwise the cross-reference becomes meaningless.
        assert ledger.by_run("") == ()

    def test_by_run(self) -> None:
        ledger = EscalationLedger()
        e1 = _event(run_id="run-42")
        e2 = _event(run_id="run-99")
        e3 = _event(run_id="run-42")
        ledger.record(e1)
        ledger.record(e2)
        ledger.record(e3)
        assert ledger.by_run("run-42") == (e1, e3)
        assert ledger.by_run("run-99") == (e2,)
        assert ledger.by_run("run-missing") == ()

    def test_in_window(self) -> None:
        ledger = EscalationLedger()
        e1 = _event(started_at=_utc(2026, 5, 4, 10, 0))
        e2 = _event(started_at=_utc(2026, 5, 4, 11, 0))
        e3 = _event(started_at=_utc(2026, 5, 4, 12, 0))
        ledger.record(e1)
        ledger.record(e2)
        ledger.record(e3)
        result = ledger.in_window(start=_utc(2026, 5, 4, 10, 30), end=_utc(2026, 5, 4, 11, 30))
        assert result == (e2,)

    def test_in_window_inclusive_at_boundaries(self) -> None:
        ledger = EscalationLedger()
        ev = _event(started_at=_utc(2026, 5, 4, 12, 0))
        ledger.record(ev)
        # Both ends inclusive.
        assert ledger.in_window(start=_utc(2026, 5, 4, 12, 0), end=_utc(2026, 5, 4, 12, 0)) == (ev,)

    def test_in_window_invalid_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="start"):
            EscalationLedger().in_window(start=_utc(2026, 5, 4, 12, 0), end=_utc(2026, 5, 4, 11, 0))


# ---------------------------------------------------------------------------
# Summarise
# ---------------------------------------------------------------------------


class TestEscalationSummary:
    def test_empty_summary(self) -> None:
        summary = EscalationLedger().summarise()
        assert isinstance(summary, EscalationSummary)
        assert summary.event_count == 0
        assert summary.total_tokens == 0
        assert summary.total_cost_usd == 0.0
        assert summary.by_reason == {}
        assert summary.by_destination == {}

    def test_full_ledger_summary(self) -> None:
        ledger = EscalationLedger()
        ledger.record(
            _event(
                reason=EscalationReason.OWNER_OVERRIDE,
                to_backend="anthropic:claude-opus-4-7",
                prompt_tokens=1000,
                response_tokens=200,
                cost_usd_micro=15_000,
            )
        )
        ledger.record(
            _event(
                reason=EscalationReason.CONTEXT_TOO_LARGE,
                to_backend="openai:gpt-5",
                prompt_tokens=20_000,
                response_tokens=2_000,
                cost_usd_micro=200_000,
            )
        )
        ledger.record(
            _event(
                reason=EscalationReason.OWNER_OVERRIDE,
                to_backend="anthropic:claude-opus-4-7",
                prompt_tokens=500,
                response_tokens=100,
                cost_usd_micro=8_000,
            )
        )
        summary = ledger.summarise()
        assert summary.event_count == 3
        assert summary.total_prompt_tokens == 21_500
        assert summary.total_response_tokens == 2_300
        assert summary.total_tokens == 23_800
        assert summary.total_cost_usd_micro == 223_000
        assert summary.total_cost_usd == 0.223
        assert summary.by_reason == {
            EscalationReason.OWNER_OVERRIDE: 2,
            EscalationReason.CONTEXT_TOO_LARGE: 1,
        }
        assert summary.by_destination == {
            "anthropic:claude-opus-4-7": 2,
            "openai:gpt-5": 1,
        }

    def test_summarise_subset(self) -> None:
        # summarise() accepts the result of any by_* query — used for
        # per-run cost reports.
        ledger = EscalationLedger()
        ledger.record(_event(run_id="run-42", cost_usd_micro=10_000))
        ledger.record(_event(run_id="run-99", cost_usd_micro=99_999))
        ledger.record(_event(run_id="run-42", cost_usd_micro=20_000))
        run42 = ledger.by_run("run-42")
        summary = ledger.summarise(run42)
        assert summary.event_count == 2
        assert summary.total_cost_usd_micro == 30_000


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


class TestEscalationLedgerSnapshot:
    def test_snapshot_empty(self) -> None:
        assert EscalationLedger().snapshot() == []

    def test_snapshot_round_trip_shape(self) -> None:
        ledger = EscalationLedger()
        started = _utc(2026, 5, 4, 12, 0)
        completed = started + timedelta(seconds=5)
        ev = _event(
            reason=EscalationReason.OWNER_OVERRIDE,
            from_backend="ollama:qwen3:30b",
            to_backend="anthropic:claude-opus-4-7",
            prompt_tokens=1234,
            response_tokens=200,
            cost_usd_micro=15_000,
            started_at=started,
            completed_at=completed,
            owner_consented=True,
            run_id="run-42",
            request_id="req-7",
            notes="user typed /cloud",
        )
        ledger.record(ev)
        snap = ledger.snapshot()
        assert len(snap) == 1
        entry = snap[0]
        assert entry["reason"] == "owner_override"
        assert entry["from_backend"] == "ollama:qwen3:30b"
        assert entry["to_backend"] == "anthropic:claude-opus-4-7"
        assert entry["prompt_tokens"] == 1234
        assert entry["response_tokens"] == 200
        assert entry["cost_usd_micro"] == 15_000
        assert entry["cost_usd"] == 0.015
        assert entry["started_at"] == started.isoformat()
        assert entry["completed_at"] == completed.isoformat()
        assert entry["owner_consented"] is True
        assert entry["run_id"] == "run-42"
        assert entry["request_id"] == "req-7"
        assert entry["notes"] == "user typed /cloud"

    def test_snapshot_handles_none_completed_at(self) -> None:
        ledger = EscalationLedger()
        ledger.record(_event(completed_at=None))
        assert ledger.snapshot()[0]["completed_at"] is None

    def test_snapshot_preserves_insertion_order(self) -> None:
        ledger = EscalationLedger()
        ledger.record(_event(notes="first"))
        ledger.record(_event(notes="second"))
        ledger.record(_event(notes="third"))
        notes = [entry["notes"] for entry in ledger.snapshot()]
        assert notes == ["first", "second", "third"]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


class TestProcessLocalLedger:
    def test_default_ledger_is_an_escalation_ledger(self) -> None:
        assert isinstance(ESCALATION_LEDGER, EscalationLedger)
