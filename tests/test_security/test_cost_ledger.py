"""Tests for the TRUST-6 cost-ledger foundation."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest

from cognithor.security.cost_ledger import (
    COST_LEDGER,
    BudgetReport,
    BudgetStatus,
    CostEntry,
    CostKind,
    CostLedger,
    CostSummary,
)


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def _entry(
    *,
    kind: CostKind = CostKind.LLM_INFERENCE,
    tool: str = "qwen3:30b",
    cost_usd_micro: int = 10_000,
    backend: str = "ollama",
    run_id: str = "",
    channel: str = "",
    domain: str = "",
    prompt_tokens: int = 100,
    response_tokens: int = 50,
    unit_count: int = -1,
    occurred_at: datetime | None = None,
    notes: str = "",
) -> CostEntry:
    return CostEntry(
        kind=kind,
        tool=tool,
        cost_usd_micro=cost_usd_micro,
        backend=backend,
        run_id=run_id,
        channel=channel,
        domain=domain,
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
        unit_count=unit_count,
        occurred_at=occurred_at if occurred_at is not None else _utc(2026, 5, 4, 12, 0),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# CostEntry validation
# ---------------------------------------------------------------------------


class TestCostEntryValidation:
    def test_minimal_entry(self) -> None:
        e = _entry()
        assert e.kind == CostKind.LLM_INFERENCE
        assert e.tool == "qwen3:30b"
        assert e.cost_usd_micro == 10_000
        assert e.cost_usd == 0.01
        assert e.occurred_at.tzinfo == UTC

    def test_frozen(self) -> None:
        e = _entry()
        with pytest.raises(dataclasses.FrozenInstanceError):
            e.notes = "tamper"  # type: ignore[misc]

    def test_empty_tool_rejected(self) -> None:
        with pytest.raises(ValueError, match="tool"):
            _entry(tool="")

    def test_negative_cost_rejected(self) -> None:
        with pytest.raises(ValueError, match="cost_usd_micro"):
            _entry(cost_usd_micro=-1)

    def test_token_counts_below_minus_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="prompt_tokens"):
            _entry(prompt_tokens=-2)
        with pytest.raises(ValueError, match="response_tokens"):
            _entry(response_tokens=-2)
        with pytest.raises(ValueError, match="unit_count"):
            _entry(unit_count=-2)


# ---------------------------------------------------------------------------
# Ledger basic ops
# ---------------------------------------------------------------------------


class TestCostLedgerBasic:
    def test_empty(self) -> None:
        ledger = CostLedger()
        assert len(ledger) == 0
        assert ledger.entries() == ()
        assert ledger.by_run("run-1") == ()
        assert ledger.by_tool("nope") == ()
        assert ledger.by_kind(CostKind.OTHER) == ()

    def test_record_appends(self) -> None:
        ledger = CostLedger()
        e = _entry()
        ledger.record(e)
        assert ledger.entries() == (e,)

    def test_record_preserves_insertion_order(self) -> None:
        ledger = CostLedger()
        a = _entry(notes="first", occurred_at=_utc(2026, 5, 4, 10, 0))
        b = _entry(notes="second", occurred_at=_utc(2026, 5, 4, 11, 0))
        c = _entry(notes="third", occurred_at=_utc(2026, 5, 4, 12, 0))
        ledger.record(a)
        ledger.record(b)
        ledger.record(c)
        assert ledger.entries() == (a, b, c)

    def test_clear(self) -> None:
        ledger = CostLedger()
        ledger.record(_entry())
        ledger.record(_entry())
        ledger.clear()
        assert len(ledger) == 0


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


class TestCostLedgerFilters:
    def test_by_run(self) -> None:
        ledger = CostLedger()
        a = _entry(run_id="run-42")
        b = _entry(run_id="run-99")
        c = _entry(run_id="run-42")
        ledger.record(a)
        ledger.record(b)
        ledger.record(c)
        assert ledger.by_run("run-42") == (a, c)
        assert ledger.by_run("run-99") == (b,)
        assert ledger.by_run("missing") == ()

    def test_by_run_empty_string_returns_empty(self) -> None:
        ledger = CostLedger()
        ledger.record(_entry(run_id=""))
        assert ledger.by_run("") == ()

    def test_by_tool(self) -> None:
        ledger = CostLedger()
        a = _entry(tool="qwen3:30b")
        b = _entry(tool="anthropic:claude-opus-4-7")
        ledger.record(a)
        ledger.record(b)
        assert ledger.by_tool("qwen3:30b") == (a,)
        assert ledger.by_tool("anthropic:claude-opus-4-7") == (b,)

    def test_by_kind(self) -> None:
        ledger = CostLedger()
        llm = _entry(kind=CostKind.LLM_INFERENCE)
        emb = _entry(kind=CostKind.EMBEDDING, tool="openai:embed-3")
        ledger.record(llm)
        ledger.record(emb)
        assert ledger.by_kind(CostKind.LLM_INFERENCE) == (llm,)
        assert ledger.by_kind(CostKind.EMBEDDING) == (emb,)

    def test_in_window(self) -> None:
        ledger = CostLedger()
        a = _entry(occurred_at=_utc(2026, 5, 4, 10, 0))
        b = _entry(occurred_at=_utc(2026, 5, 4, 11, 0))
        c = _entry(occurred_at=_utc(2026, 5, 4, 12, 0))
        ledger.record(a)
        ledger.record(b)
        ledger.record(c)
        assert ledger.in_window(start=_utc(2026, 5, 4, 10, 30), end=_utc(2026, 5, 4, 11, 30)) == (
            b,
        )

    def test_in_window_inverted_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="start"):
            CostLedger().in_window(start=_utc(2026, 5, 4, 12, 0), end=_utc(2026, 5, 4, 11, 0))


# ---------------------------------------------------------------------------
# Summarise
# ---------------------------------------------------------------------------


class TestCostSummary:
    def test_empty_summary(self) -> None:
        summary = CostLedger().summarise()
        assert isinstance(summary, CostSummary)
        assert summary.entry_count == 0
        assert summary.total_cost_usd_micro == 0
        assert summary.total_cost_usd == 0.0
        assert summary.by_kind == {}
        assert summary.by_tool == {}

    def test_full_aggregation_with_unknown_axis_fallback(self) -> None:
        # One entry has no channel, one has no domain — both must
        # bucket under "_unknown" and not vanish.
        ledger = CostLedger()
        ledger.record(
            _entry(
                tool="qwen3:30b",
                backend="ollama",
                channel="telegram",
                domain="sql",
                cost_usd_micro=10_000,
                run_id="run-42",
            )
        )
        ledger.record(
            _entry(
                tool="anthropic:claude-opus-4-7",
                backend="anthropic",
                channel="cli",
                domain="",
                cost_usd_micro=20_000,
                run_id="run-99",
            )
        )
        ledger.record(
            _entry(
                kind=CostKind.EMBEDDING,
                tool="openai:embed-3",
                backend="openai",
                channel="",
                domain="memory",
                cost_usd_micro=500,
                run_id="",
            )
        )
        summary = ledger.summarise()
        assert summary.entry_count == 3
        assert summary.total_cost_usd_micro == 30_500
        assert summary.by_kind == {
            CostKind.LLM_INFERENCE: 30_000,
            CostKind.EMBEDDING: 500,
        }
        assert summary.by_tool == {
            "qwen3:30b": 10_000,
            "anthropic:claude-opus-4-7": 20_000,
            "openai:embed-3": 500,
        }
        assert summary.by_channel == {
            "telegram": 10_000,
            "cli": 20_000,
            "_unknown": 500,
        }
        assert summary.by_domain == {
            "sql": 10_000,
            "_unknown": 20_000,
            "memory": 500,
        }
        assert summary.by_run == {
            "run-42": 10_000,
            "run-99": 20_000,
            "_unknown": 500,
        }

    def test_summarise_subset(self) -> None:
        ledger = CostLedger()
        ledger.record(_entry(run_id="run-42", cost_usd_micro=10_000))
        ledger.record(_entry(run_id="run-99", cost_usd_micro=99_999))
        ledger.record(_entry(run_id="run-42", cost_usd_micro=20_000))
        run42 = ledger.by_run("run-42")
        summary = ledger.summarise(run42)
        assert summary.entry_count == 2
        assert summary.total_cost_usd_micro == 30_000

    def test_top_n_returns_sorted_descending(self) -> None:
        ledger = CostLedger()
        ledger.record(_entry(tool="a", cost_usd_micro=1_000))
        ledger.record(_entry(tool="b", cost_usd_micro=5_000))
        ledger.record(_entry(tool="c", cost_usd_micro=3_000))
        ledger.record(_entry(tool="b", cost_usd_micro=2_000))
        summary = ledger.summarise()
        top = summary.top_n("tool", n=2)
        assert top == [("b", 7_000), ("c", 3_000)]

    def test_top_n_n_zero_returns_empty(self) -> None:
        ledger = CostLedger()
        ledger.record(_entry(cost_usd_micro=1_000))
        assert ledger.summarise().top_n("tool", n=0) == []

    def test_top_n_invalid_axis_rejected(self) -> None:
        summary = CostLedger().summarise()
        with pytest.raises(ValueError, match="axis"):
            summary.top_n("not_an_axis")


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


class TestBudgetStatus:
    def test_under_budget(self) -> None:
        ledger = CostLedger()
        ledger.record(_entry(cost_usd_micro=1_000))
        report = ledger.budget_status(limit_usd_micro=10_000)
        assert isinstance(report, BudgetReport)
        assert report.status == BudgetStatus.UNDER
        assert report.spent_usd_micro == 1_000
        assert report.remaining_usd_micro == 9_000
        assert report.utilisation == pytest.approx(0.1)

    def test_approaching_at_default_threshold(self) -> None:
        ledger = CostLedger()
        # At 80 % the report flips to APPROACHING.
        ledger.record(_entry(cost_usd_micro=8_000))
        report = ledger.budget_status(limit_usd_micro=10_000)
        assert report.status == BudgetStatus.APPROACHING
        assert report.utilisation == pytest.approx(0.8)

    def test_just_below_threshold_still_under(self) -> None:
        ledger = CostLedger()
        ledger.record(_entry(cost_usd_micro=7_999))
        report = ledger.budget_status(limit_usd_micro=10_000)
        assert report.status == BudgetStatus.UNDER

    def test_exceeded(self) -> None:
        ledger = CostLedger()
        ledger.record(_entry(cost_usd_micro=12_000))
        report = ledger.budget_status(limit_usd_micro=10_000)
        assert report.status == BudgetStatus.EXCEEDED
        assert report.remaining_usd_micro == -2_000

    def test_zero_limit_with_zero_spent(self) -> None:
        report = CostLedger().budget_status(limit_usd_micro=0)
        assert report.status == BudgetStatus.UNDER

    def test_zero_limit_with_any_spend_exceeded(self) -> None:
        ledger = CostLedger()
        ledger.record(_entry(cost_usd_micro=1))
        report = ledger.budget_status(limit_usd_micro=0)
        assert report.status == BudgetStatus.EXCEEDED

    def test_negative_limit_rejected(self) -> None:
        with pytest.raises(ValueError, match="limit_usd_micro"):
            CostLedger().budget_status(limit_usd_micro=-1)

    def test_threshold_outside_unit_interval_rejected(self) -> None:
        with pytest.raises(ValueError, match="approaching_threshold"):
            CostLedger().budget_status(limit_usd_micro=1_000, approaching_threshold=0.0)
        with pytest.raises(ValueError, match="approaching_threshold"):
            CostLedger().budget_status(limit_usd_micro=1_000, approaching_threshold=1.0)

    def test_custom_threshold(self) -> None:
        ledger = CostLedger()
        ledger.record(_entry(cost_usd_micro=5_000))
        # 50 % threshold flips at exactly half.
        report = ledger.budget_status(limit_usd_micro=10_000, approaching_threshold=0.5)
        assert report.status == BudgetStatus.APPROACHING

    def test_budget_status_with_scope(self) -> None:
        ledger = CostLedger()
        ledger.record(_entry(run_id="run-42", cost_usd_micro=5_000))
        ledger.record(_entry(run_id="run-99", cost_usd_micro=20_000))
        run42 = ledger.by_run("run-42")
        report = ledger.budget_status(limit_usd_micro=10_000, scope=run42)
        assert report.status == BudgetStatus.UNDER
        assert report.spent_usd_micro == 5_000


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


class TestCostLedgerSnapshot:
    def test_snapshot_empty(self) -> None:
        assert CostLedger().snapshot() == []

    def test_snapshot_round_trip(self) -> None:
        ledger = CostLedger()
        when = _utc(2026, 5, 4, 12, 0)
        e = _entry(
            kind=CostKind.LLM_INFERENCE,
            tool="qwen3:30b",
            cost_usd_micro=15_000,
            backend="ollama",
            run_id="run-42",
            channel="telegram",
            domain="sql",
            prompt_tokens=1234,
            response_tokens=200,
            unit_count=-1,
            occurred_at=when,
            notes="planner step",
        )
        ledger.record(e)
        entry = ledger.snapshot()[0]
        assert entry["kind"] == "llm_inference"
        assert entry["tool"] == "qwen3:30b"
        assert entry["cost_usd_micro"] == 15_000
        assert entry["cost_usd"] == 0.015
        assert entry["backend"] == "ollama"
        assert entry["run_id"] == "run-42"
        assert entry["channel"] == "telegram"
        assert entry["domain"] == "sql"
        assert entry["prompt_tokens"] == 1234
        assert entry["response_tokens"] == 200
        assert entry["unit_count"] == -1
        assert entry["occurred_at"] == when.isoformat()
        assert entry["notes"] == "planner step"

    def test_snapshot_preserves_insertion_order(self) -> None:
        ledger = CostLedger()
        ledger.record(_entry(notes="a"))
        ledger.record(_entry(notes="b"))
        ledger.record(_entry(notes="c"))
        notes = [entry["notes"] for entry in ledger.snapshot()]
        assert notes == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


class TestProcessLocalLedger:
    def test_default_is_a_cost_ledger(self) -> None:
        assert isinstance(COST_LEDGER, CostLedger)
