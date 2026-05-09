"""Tests for the TRUST-8 backend-dispatch tracking foundation."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

import pytest

from cognithor.security.backend_dispatch import (
    BACKEND_DISPATCH_LEDGER,
    BackendDispatchEvent,
    BackendDispatchLedger,
    DispatchOutcome,
    DispatchSummary,
    record_backend_dispatch,
)


def _utc(
    year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0
) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)


# ---------------------------------------------------------------------------
# BackendDispatchEvent — construction + validation
# ---------------------------------------------------------------------------


class TestBackendDispatchEventBasics:
    def test_minimal_event(self) -> None:
        ev = BackendDispatchEvent(
            backend_type="ollama",
            model="qwen3:30b",
            outcome=DispatchOutcome.SUCCESS,
        )
        assert ev.backend_type == "ollama"
        assert ev.model == "qwen3:30b"
        assert ev.outcome == DispatchOutcome.SUCCESS
        # Defaults
        assert ev.prompt_tokens == -1  # unknown sentinel
        assert ev.response_tokens == -1
        assert ev.error_kind == ""
        assert ev.is_fallback is False
        assert ev.run_id == ""
        # Auto-set timestamps
        assert ev.started_at.tzinfo == UTC
        assert ev.completed_at is None  # explicit None when not set

    def test_frozen(self) -> None:
        ev = BackendDispatchEvent(backend_type="ollama", model="x", outcome=DispatchOutcome.SUCCESS)
        with pytest.raises(dataclasses.FrozenInstanceError):
            ev.backend_type = "other"  # type: ignore[misc]

    def test_succeeded_property(self) -> None:
        ok = BackendDispatchEvent(backend_type="x", model="m", outcome=DispatchOutcome.SUCCESS)
        bad = BackendDispatchEvent(
            backend_type="x", model="m", outcome=DispatchOutcome.BACKEND_ERROR
        )
        assert ok.succeeded is True
        assert bad.succeeded is False

    def test_latency_s_returns_none_when_in_flight(self) -> None:
        ev = BackendDispatchEvent(
            backend_type="x",
            model="m",
            outcome=DispatchOutcome.SUCCESS,
            completed_at=None,
        )
        assert ev.latency_s is None

    def test_latency_s_seconds(self) -> None:
        start = _utc(2026, 5, 9, 12, 0, 0)
        ev = BackendDispatchEvent(
            backend_type="x",
            model="m",
            outcome=DispatchOutcome.SUCCESS,
            started_at=start,
            completed_at=start + timedelta(milliseconds=750),
        )
        assert ev.latency_s == pytest.approx(0.75)

    def test_total_tokens_returns_minus_one_when_either_unknown(self) -> None:
        ev = BackendDispatchEvent(
            backend_type="x",
            model="m",
            outcome=DispatchOutcome.SUCCESS,
            prompt_tokens=10,
            response_tokens=-1,
        )
        assert ev.total_tokens == -1

    def test_total_tokens_sums_when_both_known(self) -> None:
        ev = BackendDispatchEvent(
            backend_type="x",
            model="m",
            outcome=DispatchOutcome.SUCCESS,
            prompt_tokens=120,
            response_tokens=80,
        )
        assert ev.total_tokens == 200


class TestBackendDispatchEventValidation:
    def test_empty_backend_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="backend_type"):
            BackendDispatchEvent(backend_type="", model="m", outcome=DispatchOutcome.SUCCESS)

    def test_negative_prompt_tokens_below_minus_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="prompt_tokens"):
            BackendDispatchEvent(
                backend_type="x",
                model="m",
                outcome=DispatchOutcome.SUCCESS,
                prompt_tokens=-5,
            )

    def test_negative_response_tokens_below_minus_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="response_tokens"):
            BackendDispatchEvent(
                backend_type="x",
                model="m",
                outcome=DispatchOutcome.SUCCESS,
                response_tokens=-99,
            )

    def test_completed_before_started_rejected(self) -> None:
        start = _utc(2026, 5, 9, 12, 0, 0)
        with pytest.raises(ValueError, match="completed_at"):
            BackendDispatchEvent(
                backend_type="x",
                model="m",
                outcome=DispatchOutcome.SUCCESS,
                started_at=start,
                completed_at=start - timedelta(seconds=1),
            )

    def test_long_error_msg_truncated_to_200(self) -> None:
        ev = BackendDispatchEvent(
            backend_type="x",
            model="m",
            outcome=DispatchOutcome.BACKEND_ERROR,
            error_kind="ValueError",
            error_msg="X" * 1000,
        )
        assert len(ev.error_msg) == 200


# ---------------------------------------------------------------------------
# BackendDispatchLedger
# ---------------------------------------------------------------------------


class TestBackendDispatchLedger:
    def test_empty(self) -> None:
        ledger = BackendDispatchLedger()
        assert len(ledger) == 0
        assert ledger.events() == ()

    def test_record_appends(self) -> None:
        ledger = BackendDispatchLedger()
        ev = BackendDispatchEvent(backend_type="ollama", model="x", outcome=DispatchOutcome.SUCCESS)
        ledger.record(ev)
        assert len(ledger) == 1
        assert ledger.events()[0] is ev

    def test_by_run_filters_by_exact_match(self) -> None:
        ledger = BackendDispatchLedger()
        ev_a = BackendDispatchEvent(
            backend_type="x", model="m", outcome=DispatchOutcome.SUCCESS, run_id="run-1"
        )
        ev_b = BackendDispatchEvent(
            backend_type="x", model="m", outcome=DispatchOutcome.SUCCESS, run_id="run-2"
        )
        ledger.record(ev_a)
        ledger.record(ev_b)
        assert ledger.by_run("run-1") == (ev_a,)
        assert ledger.by_run("run-2") == (ev_b,)
        assert ledger.by_run("missing") == ()

    def test_by_backend(self) -> None:
        ledger = BackendDispatchLedger()
        ollama = BackendDispatchEvent(
            backend_type="ollama", model="m", outcome=DispatchOutcome.SUCCESS
        )
        anthropic = BackendDispatchEvent(
            backend_type="anthropic", model="m", outcome=DispatchOutcome.SUCCESS
        )
        ledger.record(ollama)
        ledger.record(anthropic)
        assert ledger.by_backend("ollama") == (ollama,)
        assert ledger.by_backend("anthropic") == (anthropic,)

    def test_by_outcome(self) -> None:
        ledger = BackendDispatchLedger()
        ok = BackendDispatchEvent(backend_type="x", model="m", outcome=DispatchOutcome.SUCCESS)
        bad = BackendDispatchEvent(
            backend_type="x", model="m", outcome=DispatchOutcome.BACKEND_ERROR
        )
        ledger.record(ok)
        ledger.record(bad)
        assert ledger.by_outcome(DispatchOutcome.SUCCESS) == (ok,)
        assert ledger.by_outcome(DispatchOutcome.BACKEND_ERROR) == (bad,)

    def test_in_window_inclusive(self) -> None:
        ledger = BackendDispatchLedger()
        early = _utc(2026, 5, 9, 10, 0)
        mid = _utc(2026, 5, 9, 11, 0)
        late = _utc(2026, 5, 9, 12, 0)
        for ts in (early, mid, late):
            ledger.record(
                BackendDispatchEvent(
                    backend_type="x",
                    model="m",
                    outcome=DispatchOutcome.SUCCESS,
                    started_at=ts,
                )
            )
        # Inclusive on both ends
        scoped = ledger.in_window(start=early, end=late)
        assert len(scoped) == 3
        # Strict middle
        scoped_mid = ledger.in_window(start=mid, end=mid)
        assert len(scoped_mid) == 1

    def test_clear_drops_all(self) -> None:
        ledger = BackendDispatchLedger()
        ledger.record(
            BackendDispatchEvent(backend_type="x", model="m", outcome=DispatchOutcome.SUCCESS)
        )
        ledger.clear()
        assert len(ledger) == 0


# ---------------------------------------------------------------------------
# Summarise
# ---------------------------------------------------------------------------


class TestSummarise:
    def test_empty_ledger_summary_has_vacuous_success(self) -> None:
        ledger = BackendDispatchLedger()
        summary = ledger.summarise()
        assert summary.event_count == 0
        # Empty buckets must NOT flag red — vacuous success
        assert summary.success_rate == 1.0
        assert summary.total_prompt_tokens == 0
        assert summary.total_response_tokens == 0

    def test_success_rate_with_mix(self) -> None:
        ledger = BackendDispatchLedger()
        ledger.record(
            BackendDispatchEvent(backend_type="x", model="m", outcome=DispatchOutcome.SUCCESS)
        )
        ledger.record(
            BackendDispatchEvent(backend_type="x", model="m", outcome=DispatchOutcome.SUCCESS)
        )
        ledger.record(
            BackendDispatchEvent(backend_type="x", model="m", outcome=DispatchOutcome.BACKEND_ERROR)
        )
        s = ledger.summarise()
        assert s.event_count == 3
        assert s.success_count == 2
        assert s.success_rate == pytest.approx(2 / 3)

    def test_by_backend_buckets(self) -> None:
        ledger = BackendDispatchLedger()
        ledger.record(
            BackendDispatchEvent(backend_type="ollama", model="m", outcome=DispatchOutcome.SUCCESS)
        )
        ledger.record(
            BackendDispatchEvent(backend_type="ollama", model="m", outcome=DispatchOutcome.SUCCESS)
        )
        ledger.record(
            BackendDispatchEvent(
                backend_type="anthropic", model="m", outcome=DispatchOutcome.SUCCESS
            )
        )
        s = ledger.summarise()
        assert s.by_backend == {"ollama": 2, "anthropic": 1}

    def test_by_outcome_buckets(self) -> None:
        ledger = BackendDispatchLedger()
        ledger.record(
            BackendDispatchEvent(backend_type="x", model="m", outcome=DispatchOutcome.SUCCESS)
        )
        ledger.record(
            BackendDispatchEvent(backend_type="x", model="m", outcome=DispatchOutcome.CIRCUIT_OPEN)
        )
        ledger.record(
            BackendDispatchEvent(backend_type="x", model="m", outcome=DispatchOutcome.CIRCUIT_OPEN)
        )
        s = ledger.summarise()
        assert s.by_outcome[DispatchOutcome.SUCCESS] == 1
        assert s.by_outcome[DispatchOutcome.CIRCUIT_OPEN] == 2

    def test_fallback_count(self) -> None:
        ledger = BackendDispatchLedger()
        ledger.record(
            BackendDispatchEvent(
                backend_type="ollama", model="m", outcome=DispatchOutcome.SUCCESS, is_fallback=True
            )
        )
        ledger.record(
            BackendDispatchEvent(
                backend_type="ollama", model="m", outcome=DispatchOutcome.SUCCESS, is_fallback=False
            )
        )
        s = ledger.summarise()
        assert s.fallback_count == 1

    def test_token_totals_propagate_unknown(self) -> None:
        """If ANY event reports -1 for prompt_tokens, the total must be
        -1 (not silently under-count)."""
        ledger = BackendDispatchLedger()
        ledger.record(
            BackendDispatchEvent(
                backend_type="x",
                model="m",
                outcome=DispatchOutcome.SUCCESS,
                prompt_tokens=100,
                response_tokens=50,
            )
        )
        ledger.record(
            BackendDispatchEvent(
                backend_type="x",
                model="m",
                outcome=DispatchOutcome.SUCCESS,
                prompt_tokens=-1,
                response_tokens=20,
            )
        )
        s = ledger.summarise()
        assert s.total_prompt_tokens == -1, "unknown contributor must propagate"
        assert s.total_response_tokens == 70

    def test_token_totals_when_all_known(self) -> None:
        ledger = BackendDispatchLedger()
        ledger.record(
            BackendDispatchEvent(
                backend_type="x",
                model="m",
                outcome=DispatchOutcome.SUCCESS,
                prompt_tokens=100,
                response_tokens=50,
            )
        )
        ledger.record(
            BackendDispatchEvent(
                backend_type="x",
                model="m",
                outcome=DispatchOutcome.SUCCESS,
                prompt_tokens=200,
                response_tokens=80,
            )
        )
        s = ledger.summarise()
        assert s.total_prompt_tokens == 300
        assert s.total_response_tokens == 130

    def test_summary_is_immutable_dataclass(self) -> None:
        s = DispatchSummary(
            event_count=0,
            success_count=0,
            by_backend={},
            by_outcome={},
            fallback_count=0,
            total_prompt_tokens=0,
            total_response_tokens=0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.event_count = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_empty_snapshot(self) -> None:
        ledger = BackendDispatchLedger()
        assert ledger.snapshot() == []

    def test_snapshot_round_trip_shape(self) -> None:
        ledger = BackendDispatchLedger()
        ts = _utc(2026, 5, 9, 12, 0, 0)
        ledger.record(
            BackendDispatchEvent(
                backend_type="ollama",
                model="qwen3:30b",
                outcome=DispatchOutcome.SUCCESS,
                started_at=ts,
                completed_at=ts + timedelta(milliseconds=420),
                prompt_tokens=120,
                response_tokens=60,
                is_fallback=True,
                run_id="run-abc",
                request_id="req-1",
                notes="planner step",
            )
        )
        rows = ledger.snapshot()
        assert len(rows) == 1
        row = rows[0]
        assert row["backend_type"] == "ollama"
        assert row["model"] == "qwen3:30b"
        assert row["outcome"] == "success"
        assert row["latency_s"] == pytest.approx(0.42)
        assert row["prompt_tokens"] == 120
        assert row["response_tokens"] == 60
        assert row["is_fallback"] is True
        assert row["run_id"] == "run-abc"
        assert row["request_id"] == "req-1"
        # Timestamps as ISO-format strings
        assert isinstance(row["started_at"], str)
        assert isinstance(row["completed_at"], str)

    def test_snapshot_is_json_safe(self) -> None:
        """The snapshot must round-trip through json without losing data."""
        import json

        ledger = BackendDispatchLedger()
        # Pin both timestamps explicitly to avoid the race where the
        # default_factory for started_at runs AFTER the explicit
        # completed_at kwarg is read (CPU-scheduling-dependent under
        # microsecond resolution on Windows runners).
        start = _utc(2026, 5, 9, 12, 0, 0)
        ledger.record(
            BackendDispatchEvent(
                backend_type="x",
                model="m",
                outcome=DispatchOutcome.SUCCESS,
                started_at=start,
                completed_at=start + timedelta(milliseconds=50),
            )
        )
        serialised = json.dumps(ledger.snapshot())
        parsed = json.loads(serialised)
        assert isinstance(parsed, list)
        assert len(parsed) == 1


# ---------------------------------------------------------------------------
# record_backend_dispatch convenience
# ---------------------------------------------------------------------------


class TestRecordBackendDispatch:
    def test_records_into_supplied_ledger(self) -> None:
        ledger = BackendDispatchLedger()
        ev = record_backend_dispatch(
            backend_type="ollama",
            model="qwen3:30b",
            outcome=DispatchOutcome.SUCCESS,
            started_at=datetime.now(UTC),
            ledger=ledger,
        )
        assert len(ledger) == 1
        assert ledger.events()[0] is ev
        # completed_at auto-filled when omitted
        assert ev.completed_at is not None

    def test_returns_event_for_caller_chaining(self) -> None:
        ledger = BackendDispatchLedger()
        result = record_backend_dispatch(
            backend_type="ollama",
            model="m",
            outcome=DispatchOutcome.SUCCESS,
            started_at=datetime.now(UTC),
            ledger=ledger,
        )
        assert isinstance(result, BackendDispatchEvent)
        assert result.outcome == DispatchOutcome.SUCCESS

    def test_explicit_completed_at_preserved(self) -> None:
        ledger = BackendDispatchLedger()
        start = _utc(2026, 5, 9, 10)
        end = _utc(2026, 5, 9, 11)
        ev = record_backend_dispatch(
            backend_type="x",
            model="m",
            outcome=DispatchOutcome.SUCCESS,
            started_at=start,
            completed_at=end,
            ledger=ledger,
        )
        assert ev.completed_at == end

    def test_default_ledger_is_canonical(self) -> None:
        """``ledger=None`` writes into :data:`BACKEND_DISPATCH_LEDGER`."""
        # Snapshot the canonical ledger's state, append, then trim back.
        BACKEND_DISPATCH_LEDGER.clear()
        record_backend_dispatch(
            backend_type="x",
            model="m",
            outcome=DispatchOutcome.SUCCESS,
            started_at=datetime.now(UTC),
        )
        try:
            assert len(BACKEND_DISPATCH_LEDGER) == 1
        finally:
            BACKEND_DISPATCH_LEDGER.clear()


# ---------------------------------------------------------------------------
# Privacy contract — assert no prompt/response content can land
# ---------------------------------------------------------------------------


class TestPrivacyContract:
    """The dispatch surface must NEVER carry prompt or response content.
    These tests pin the contract so a future field addition can't quietly
    leak content into the audit trail."""

    def test_dataclass_has_no_content_field(self) -> None:
        field_names = {f.name for f in dataclasses.fields(BackendDispatchEvent)}
        # Defensive: any of these names appearing in the dataclass
        # would be a privacy regression. Add new forbidden names here
        # if reviewers spot risky additions.
        forbidden = {
            "prompt",
            "response",
            "messages",
            "completion",
            "content",
            "system_prompt",
            "user_prompt",
            "assistant_text",
        }
        assert field_names.isdisjoint(forbidden), (
            f"BackendDispatchEvent leaks content fields: {field_names & forbidden}"
        )

    def test_notes_field_documented_as_metadata_only(self) -> None:
        """The ``notes`` field exists for breadcrumbs but the docstring
        explicitly forbids content. This is enforced by review, not by
        type-system; document it so a future reviewer keeps the rule."""
        doc = BackendDispatchEvent.__doc__ or ""
        # Combined with field-level docstring style — we look at the
        # class docstring for the contract statement.
        assert "metadata" in doc.lower() or "no prompt" in doc.lower(), (
            "BackendDispatchEvent docstring must state the metadata-only contract"
        )
