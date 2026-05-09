"""Tests for the SQLite-backed CostLedgerStore (TRUST-6 disk persistence)."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from cognithor.security.cost_ledger import (
    BudgetStatus,
    CostEntry,
    CostKind,
    CostLedger,
    CostSummary,
)
from cognithor.security.cost_ledger_store import (
    CostLedgerStore,
    CostLedgerStoreError,
)


def _utc(
    year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0
) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)


def _entry(
    *,
    kind: CostKind = CostKind.LLM_INFERENCE,
    tool: str = "qwen3:30b",
    cost_usd_micro: int = 100,
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
        occurred_at=occurred_at or _utc(2026, 5, 9, 12, 0, 0),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Lifecycle + schema
# ---------------------------------------------------------------------------


class TestStoreLifecycle:
    def test_creates_schema_in_memory(self) -> None:
        with CostLedgerStore(":memory:") as store:
            assert store.schema_version == 1
            assert len(store) == 0

    def test_context_manager_persists_to_disk(self, tmp_path) -> None:
        db = tmp_path / "x.sqlite"
        with CostLedgerStore(db) as store:
            store.record(_entry(run_id="run-1"))

        with CostLedgerStore(db) as store:
            assert len(store) == 1
            assert store.entries()[0].run_id == "run-1"

    def test_close_idempotent(self) -> None:
        store = CostLedgerStore(":memory:")
        store.close()
        store.close()  # must not raise

    def test_reopen_existing_does_not_reset(self, tmp_path) -> None:
        db = tmp_path / "x.sqlite"
        with CostLedgerStore(db) as store:
            store.record(_entry())
        with CostLedgerStore(db) as store:
            assert len(store) == 1


# ---------------------------------------------------------------------------
# Schema-version guard
# ---------------------------------------------------------------------------


class TestSchemaVersionGuard:
    def test_rejects_newer_schema(self, tmp_path) -> None:
        db = tmp_path / "future.sqlite"
        CostLedgerStore(db).close()
        conn = sqlite3.connect(db)
        try:
            conn.execute(
                "INSERT INTO _schema_meta (version, applied_at) VALUES (?, ?)",
                (999, datetime.now(UTC).isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

        with pytest.raises(CostLedgerStoreError, match="version 999"):
            CostLedgerStore(db)

    def test_accepts_equal_version(self, tmp_path) -> None:
        db = tmp_path / "x.sqlite"
        CostLedgerStore(db).close()
        CostLedgerStore(db).close()


# ---------------------------------------------------------------------------
# Append + read-back
# ---------------------------------------------------------------------------


class TestAppendAndRead:
    def test_returns_row_id(self) -> None:
        with CostLedgerStore(":memory:") as store:
            assert store.record(_entry(run_id="r1")) == 1
            assert store.record(_entry(run_id="r2")) == 2

    def test_full_field_round_trip(self) -> None:
        original = _entry(
            kind=CostKind.VISION_TOKENS,
            tool="qwen3-vl-32b",
            cost_usd_micro=42_500,
            backend="vllm",
            run_id="run-vl-7",
            channel="webui",
            domain="vision",
            prompt_tokens=2048,
            response_tokens=180,
            unit_count=3,
            occurred_at=_utc(2026, 5, 9, 14, 30, 15),
            notes="frame-batch /pse/run/7",
        )
        with CostLedgerStore(":memory:") as store:
            store.record(original)
            recovered = store.entries()[0]
            assert recovered == original

    def test_unknown_token_counts_round_trip_as_minus_one(self) -> None:
        """The in-memory contract uses ``-1`` to mean "unknown"; if
        the disk store accidentally coerced it (e.g. via DEFAULT
        clauses overriding NULLs), summaries on storage-class costs
        would silently flip from ``unknown`` to ``0`` and skew
        budget alerts."""
        ev = _entry(
            kind=CostKind.STORAGE,
            tool="s3-bucket-snapshots",
            cost_usd_micro=2_000,
            prompt_tokens=-1,
            response_tokens=-1,
            unit_count=-1,
        )
        with CostLedgerStore(":memory:") as store:
            store.record(ev)
            r = store.entries()[0]
            assert r.prompt_tokens == -1
            assert r.response_tokens == -1
            assert r.unit_count == -1

    def test_clear_drops_everything(self) -> None:
        with CostLedgerStore(":memory:") as store:
            for _ in range(5):
                store.record(_entry())
            assert len(store) == 5
            store.clear()
            assert len(store) == 0

    def test_entries_returned_in_insertion_order(self) -> None:
        with CostLedgerStore(":memory:") as store:
            for tool in ("a", "b", "c"):
                store.record(_entry(tool=tool))
            tools = [e.tool for e in store.entries()]
            assert tools == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Read filters
# ---------------------------------------------------------------------------


class TestReadFilters:
    def test_by_run_filters_correctly(self) -> None:
        with CostLedgerStore(":memory:") as store:
            store.record(_entry(run_id="run-1"))
            store.record(_entry(run_id="run-1"))
            store.record(_entry(run_id="run-2"))
            assert len(store.by_run("run-1")) == 2
            assert len(store.by_run("run-2")) == 1

    def test_by_run_empty_string_short_circuits(self) -> None:
        """Mirrors the in-memory ledger's contract: empty run_id
        returns ``()`` so sentinel queries don't leak the whole
        ledger."""
        with CostLedgerStore(":memory:") as store:
            store.record(_entry(run_id=""))
            assert store.by_run("") == ()

    def test_by_run_missing_returns_empty(self) -> None:
        with CostLedgerStore(":memory:") as store:
            store.record(_entry(run_id="run-1"))
            assert store.by_run("nope") == ()

    def test_by_tool_filters_correctly(self) -> None:
        with CostLedgerStore(":memory:") as store:
            store.record(_entry(tool="qwen3:30b"))
            store.record(_entry(tool="qwen3:30b"))
            store.record(_entry(tool="openai:gpt"))
            assert len(store.by_tool("qwen3:30b")) == 2
            assert len(store.by_tool("openai:gpt")) == 1

    def test_by_tool_empty_string_short_circuits(self) -> None:
        with CostLedgerStore(":memory:") as store:
            # cannot store a CostEntry with tool="" — __post_init__
            # rejects it. So we just probe the empty-input contract.
            assert store.by_tool("") == ()

    def test_by_kind_filters_correctly(self) -> None:
        with CostLedgerStore(":memory:") as store:
            store.record(_entry(kind=CostKind.LLM_INFERENCE))
            store.record(_entry(kind=CostKind.EMBEDDING))
            store.record(_entry(kind=CostKind.LLM_INFERENCE))
            assert len(store.by_kind(CostKind.LLM_INFERENCE)) == 2
            assert len(store.by_kind(CostKind.EMBEDDING)) == 1
            assert store.by_kind(CostKind.STORAGE) == ()

    def test_in_window_inclusive(self) -> None:
        with CostLedgerStore(":memory:") as store:
            early = _utc(2026, 5, 9, 10)
            mid = _utc(2026, 5, 9, 11)
            late = _utc(2026, 5, 9, 12)
            for ts in (early, mid, late):
                store.record(_entry(occurred_at=ts))
            assert len(store.in_window(start=early, end=late)) == 3
            assert len(store.in_window(start=mid, end=mid)) == 1

    def test_in_window_swapped_window_raises(self) -> None:
        with CostLedgerStore(":memory:") as store:
            with pytest.raises(ValueError, match="start must be <= end"):
                store.in_window(
                    start=_utc(2026, 5, 9, 12),
                    end=_utc(2026, 5, 9, 10),
                )


# ---------------------------------------------------------------------------
# Aggregation — must agree with in-memory ledger
# ---------------------------------------------------------------------------


class TestSummarise:
    def test_empty_summary_is_zero(self) -> None:
        with CostLedgerStore(":memory:") as store:
            summary = store.summarise()
            assert summary.entry_count == 0
            assert summary.total_cost_usd_micro == 0

    def test_six_axis_histograms(self) -> None:
        events = [
            _entry(
                kind=CostKind.LLM_INFERENCE,
                tool="qwen3",
                backend="ollama",
                run_id="r1",
                channel="cli",
                domain="planner",
                cost_usd_micro=1_000,
            ),
            _entry(
                kind=CostKind.LLM_INFERENCE,
                tool="qwen3",
                backend="ollama",
                run_id="r1",
                channel="cli",
                domain="planner",
                cost_usd_micro=2_000,
            ),
            _entry(
                kind=CostKind.EMBEDDING,
                tool="oai-embed",
                backend="openai",
                run_id="r2",
                channel="webui",
                domain="memory",
                cost_usd_micro=500,
            ),
        ]
        with CostLedgerStore(":memory:") as store:
            for e in events:
                store.record(e)
            s = store.summarise()
            assert isinstance(s, CostSummary)
            assert s.entry_count == 3
            assert s.total_cost_usd_micro == 3_500
            assert s.by_kind[CostKind.LLM_INFERENCE] == 3_000
            assert s.by_kind[CostKind.EMBEDDING] == 500
            assert s.by_tool == {"qwen3": 3_000, "oai-embed": 500}
            assert s.by_backend == {"ollama": 3_000, "openai": 500}
            assert s.by_channel == {"cli": 3_000, "webui": 500}
            assert s.by_domain == {"planner": 3_000, "memory": 500}
            assert s.by_run == {"r1": 3_000, "r2": 500}

    def test_summary_matches_in_memory_for_same_entries(self) -> None:
        """Double-bookkeeping check: the disk store's summary MUST
        agree key-for-key with the in-memory ledger's summary over
        the same entries — otherwise budget alerts diverge based on
        which ledger the dashboard happens to query."""
        events = [
            _entry(
                kind=CostKind.LLM_INFERENCE,
                tool="qwen3",
                backend="",  # falls into "_unknown"
                cost_usd_micro=1_500,
            ),
            _entry(
                kind=CostKind.TOOL_API,
                tool="search",
                cost_usd_micro=750,
            ),
        ]
        memory_ledger = CostLedger()
        with CostLedgerStore(":memory:") as store:
            for e in events:
                memory_ledger.record(e)
                store.record(e)
            mem = memory_ledger.summarise()
            disk = store.summarise()
            assert mem.total_cost_usd_micro == disk.total_cost_usd_micro
            assert mem.by_kind == disk.by_kind
            assert mem.by_tool == disk.by_tool
            assert mem.by_backend == disk.by_backend
            assert mem.by_channel == disk.by_channel
            assert mem.by_domain == disk.by_domain
            assert mem.by_run == disk.by_run

    def test_summarise_with_explicit_subset(self) -> None:
        with CostLedgerStore(":memory:") as store:
            store.record(_entry(run_id="r1", cost_usd_micro=100))
            store.record(_entry(run_id="r2", cost_usd_micro=900))
            scoped = store.by_run("r1")
            s = store.summarise(entries=scoped)
            assert s.entry_count == 1
            assert s.total_cost_usd_micro == 100


# ---------------------------------------------------------------------------
# Budget alerting
# ---------------------------------------------------------------------------


class TestBudgetStatus:
    def test_under_when_well_below_limit(self) -> None:
        with CostLedgerStore(":memory:") as store:
            store.record(_entry(cost_usd_micro=1_000))
            r = store.budget_status(limit_usd_micro=10_000)
            assert r.status == BudgetStatus.UNDER
            assert r.spent_usd_micro == 1_000
            assert r.remaining_usd_micro == 9_000

    def test_approaching_at_eighty_percent(self) -> None:
        with CostLedgerStore(":memory:") as store:
            store.record(_entry(cost_usd_micro=8_000))
            r = store.budget_status(limit_usd_micro=10_000)
            assert r.status == BudgetStatus.APPROACHING

    def test_exceeded_above_limit(self) -> None:
        with CostLedgerStore(":memory:") as store:
            store.record(_entry(cost_usd_micro=15_000))
            r = store.budget_status(limit_usd_micro=10_000)
            assert r.status == BudgetStatus.EXCEEDED
            assert r.remaining_usd_micro == -5_000

    def test_zero_limit_with_spend_is_exceeded(self) -> None:
        with CostLedgerStore(":memory:") as store:
            store.record(_entry(cost_usd_micro=1))
            r = store.budget_status(limit_usd_micro=0)
            assert r.status == BudgetStatus.EXCEEDED


# ---------------------------------------------------------------------------
# Snapshot — JSON-safe + key parity
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_empty_snapshot(self) -> None:
        with CostLedgerStore(":memory:") as store:
            assert store.snapshot() == []

    def test_round_trip_via_json(self) -> None:
        with CostLedgerStore(":memory:") as store:
            store.record(
                _entry(
                    kind=CostKind.NETWORK,
                    tool="vercel-egress",
                    cost_usd_micro=12_500,
                    domain="hosting",
                    notes="weekly egress sample",
                )
            )
            rows = store.snapshot()
            serialised = json.dumps(rows)
            parsed = json.loads(serialised)
            row = parsed[0]
            assert row["kind"] == "network"
            assert row["tool"] == "vercel-egress"
            assert row["cost_usd_micro"] == 12_500
            assert row["cost_usd"] == pytest.approx(0.0125)
            assert row["domain"] == "hosting"
            assert row["notes"] == "weekly egress sample"

    def test_snapshot_keys_match_in_memory_ledger(self) -> None:
        memory_ledger = CostLedger()
        memory_ledger.record(_entry(run_id="r"))
        memory_keys = set(memory_ledger.snapshot()[0].keys())

        with CostLedgerStore(":memory:") as store:
            store.record(_entry(run_id="r"))
            disk_keys = set(store.snapshot()[0].keys())

        assert memory_keys == disk_keys, (
            f"snapshot keys diverge — memory_only={memory_keys - disk_keys}, "
            f"disk_only={disk_keys - memory_keys}"
        )


# ---------------------------------------------------------------------------
# File persistence
# ---------------------------------------------------------------------------


class TestFilePersistence:
    def test_writes_visible_after_reopen(self, tmp_path) -> None:
        db = tmp_path / "x.sqlite"
        with CostLedgerStore(db) as store:
            store.record(_entry(run_id="r1", cost_usd_micro=100))
            store.record(_entry(run_id="r2", cost_usd_micro=200))
        with CostLedgerStore(db) as store:
            assert len(store) == 2
            assert store.summarise().total_cost_usd_micro == 300

    def test_concurrent_writers_share_db(self, tmp_path) -> None:
        db = tmp_path / "shared.sqlite"
        with (
            CostLedgerStore(db) as a,
            CostLedgerStore(db) as b,
        ):
            a.record(_entry(run_id="run-a", cost_usd_micro=10))
            b.record(_entry(run_id="run-b", cost_usd_micro=20))
        with CostLedgerStore(db) as reader:
            assert len(reader) == 2
            assert reader.summarise().total_cost_usd_micro == 30


# ---------------------------------------------------------------------------
# Indices — query-shape sanity
# ---------------------------------------------------------------------------


class TestIndices:
    def test_all_four_hot_indices_present(self) -> None:
        """Pin the four hot read-path indices: tool, run_id, kind,
        occurred_at."""
        with CostLedgerStore(":memory:") as store:
            cursor = store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='cost_entries'"
            )
            names = {row[0] for row in cursor.fetchall()}
            assert "idx_cost_entries_tool" in names
            assert "idx_cost_entries_run_id" in names
            assert "idx_cost_entries_kind" in names
            assert "idx_cost_entries_occurred_at" in names


# ---------------------------------------------------------------------------
# Window edge case — explicit window over a few days
# ---------------------------------------------------------------------------


class TestWindowEdgeCases:
    def test_window_excludes_outside(self) -> None:
        with CostLedgerStore(":memory:") as store:
            base = _utc(2026, 5, 9, 0, 0, 0)
            for delta_h in (0, 6, 12, 18, 24, 30, 36):
                store.record(_entry(occurred_at=base + timedelta(hours=delta_h)))
            window = store.in_window(
                start=base + timedelta(hours=6),
                end=base + timedelta(hours=24),
            )
            # 6h, 12h, 18h, 24h → 4 entries
            assert len(window) == 4
