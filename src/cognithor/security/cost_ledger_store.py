"""SQLite-backed disk persistence for the TRUST-6 cost ledger.

Companion to :mod:`cognithor.security.cost_ledger`. The in-memory
ledger answers "how much have I spent today" within a single process;
this module turns the same query into one an operator can ask the
day after a process restart — the reviewer-feedback budget question
without grepping process state.

Design parity with :mod:`cognithor.security.backend_dispatch_store`
(#515), :mod:`cognithor.security.cloud_escalation_store` (#516), and
:mod:`cognithor.security.fingerprint_store` (#517):

* Plain SQLite — privacy contract preserved (no prompts, no responses,
  no model weights). All columns are budget metadata only.
* Append-only — ``record(entry)`` is the single mutation.
* Schema-versioned via ``_schema_meta``; loud refusal on
  newer-than-build databases.
* WAL journal mode + ``check_same_thread=False`` so concurrent
  gateway processes can share the file.
* Read-API parity with :class:`CostLedger`: ``__len__``, ``entries``,
  ``by_run``, ``by_tool``, ``by_kind``, ``in_window``, ``summarise``,
  ``budget_status``, ``snapshot``.
* Indices on ``tool``, ``run_id``, ``kind``, ``occurred_at`` —
  the four hot read paths. Pinned by a ``TestIndices`` contract.

Cost-axis specifics:

* ``cost_usd_micro`` is stored as ``INTEGER NOT NULL`` and summed
  via SQL where convenient — but ``summarise`` builds a six-axis
  histogram which is cheaper to compute in Python over the result
  set than to issue six grouped queries. Budget alerting reuses the
  same shared aggregator.
* ``-1`` token / unit counts mean "unknown" (per the in-memory
  contract); they round-trip as-is through the integer column.
"""

from __future__ import annotations

import contextlib
import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from cognithor.security.cost_ledger import (
    BudgetReport,
    CostEntry,
    CostKind,
    CostLedger,
    CostSummary,
)
from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

log = get_logger(__name__)


_SCHEMA_VERSION = 1


_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS _schema_meta (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cost_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    tool TEXT NOT NULL,
    cost_usd_micro INTEGER NOT NULL,
    backend TEXT NOT NULL DEFAULT '',
    run_id TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL DEFAULT '',
    domain TEXT NOT NULL DEFAULT '',
    prompt_tokens INTEGER NOT NULL DEFAULT -1,
    response_tokens INTEGER NOT NULL DEFAULT -1,
    unit_count INTEGER NOT NULL DEFAULT -1,
    occurred_at TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT ''
);

-- Indices match the four hot read paths on the in-memory ledger:
-- by_tool, by_run, by_kind, in_window. Pinning them prevents a
-- future migration from silently dropping one — Trace-UI budget
-- panels would table-scan on every render.
CREATE INDEX IF NOT EXISTS idx_cost_entries_tool
    ON cost_entries(tool);
CREATE INDEX IF NOT EXISTS idx_cost_entries_run_id
    ON cost_entries(run_id);
CREATE INDEX IF NOT EXISTS idx_cost_entries_kind
    ON cost_entries(kind);
CREATE INDEX IF NOT EXISTS idx_cost_entries_occurred_at
    ON cost_entries(occurred_at);

INSERT OR IGNORE INTO _schema_meta (version, applied_at)
    VALUES ({_SCHEMA_VERSION}, '{datetime.now(UTC).isoformat()}');
"""


class CostLedgerStoreError(Exception):
    """Raised when the SQLite layer is in an unrecoverable state.

    Mirrors the sibling stores. Examples include a schema version
    this build doesn't know how to read, or a corrupt file whose
    connection won't open cleanly.
    """


class CostLedgerStore:
    """SQLite-backed append-only persistence for ``CostEntry``.

    Tests use ``:memory:`` databases for speed; production wires the
    canonical instance to ``~/.cognithor/audit/cost_ledger.sqlite``.

    Lifecycle mirrors the sibling stores — context-manager or
    explicit ``close()``. Re-opening an existing file is a no-op
    (idempotent schema script).
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._verify_schema_version()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __enter__(self) -> CostLedgerStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the SQLite connection. Safe to call multiple times."""
        with contextlib.suppress(sqlite3.ProgrammingError):
            self._conn.close()

    # ------------------------------------------------------------------
    # Schema-version handshake
    # ------------------------------------------------------------------

    def _verify_schema_version(self) -> None:
        cursor = self._conn.execute("SELECT MAX(version) FROM _schema_meta")
        row = cursor.fetchone()
        actual_version = row[0] if row else None
        if actual_version is None:
            return  # fresh DB
        if actual_version > _SCHEMA_VERSION:
            msg = (
                f"CostLedgerStore at {self._db_path!r} reports schema "
                f"version {actual_version} but this build only knows "
                f"version {_SCHEMA_VERSION}. Refusing to open to avoid "
                "data corruption — upgrade cognithor or point at a "
                "different file."
            )
            raise CostLedgerStoreError(msg)

    @property
    def schema_version(self) -> int:
        cursor = self._conn.execute("SELECT MAX(version) FROM _schema_meta")
        row = cursor.fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def record(self, entry: CostEntry) -> int:
        """Append ``entry`` and return its assigned row id."""
        cursor = self._conn.execute(
            """
            INSERT INTO cost_entries (
                kind, tool, cost_usd_micro,
                backend, run_id, channel, domain,
                prompt_tokens, response_tokens, unit_count,
                occurred_at, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.kind.value,
                entry.tool,
                entry.cost_usd_micro,
                entry.backend,
                entry.run_id,
                entry.channel,
                entry.domain,
                entry.prompt_tokens,
                entry.response_tokens,
                entry.unit_count,
                entry.occurred_at.isoformat(),
                entry.notes,
            ),
        )
        self._conn.commit()
        row_id = cursor.lastrowid
        if row_id is None:  # pragma: no cover — defensive
            msg = "sqlite returned no lastrowid after cost_entries INSERT"
            raise CostLedgerStoreError(msg)
        return row_id

    def clear(self) -> None:
        """Drop all entries. Test helper; production never calls this."""
        self._conn.execute("DELETE FROM cost_entries")
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read API (mirror of CostLedger)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        cursor = self._conn.execute("SELECT COUNT(*) FROM cost_entries")
        row = cursor.fetchone()
        return int(row[0]) if row else 0

    def entries(self) -> tuple[CostEntry, ...]:
        """Return every entry in insertion order."""
        cursor = self._conn.execute("SELECT * FROM cost_entries ORDER BY id ASC")
        return tuple(_row_to_entry(row, cursor) for row in cursor.fetchall())

    def by_run(self, run_id: str) -> tuple[CostEntry, ...]:
        """Entries tied to ``run_id`` — the TRUST-1 cross-reference query.

        Mirrors the in-memory ledger's empty-string short-circuit so
        sentinel values used by un-tracked runs don't accidentally
        return the entire ledger."""
        if not run_id:
            return ()
        cursor = self._conn.execute(
            "SELECT * FROM cost_entries WHERE run_id = ? ORDER BY id ASC",
            (run_id,),
        )
        return tuple(_row_to_entry(row, cursor) for row in cursor.fetchall())

    def by_tool(self, tool: str) -> tuple[CostEntry, ...]:
        """Entries for a single tool — same empty-string contract as
        the in-memory ledger."""
        if not tool:
            return ()
        cursor = self._conn.execute(
            "SELECT * FROM cost_entries WHERE tool = ? ORDER BY id ASC",
            (tool,),
        )
        return tuple(_row_to_entry(row, cursor) for row in cursor.fetchall())

    def by_kind(self, kind: CostKind) -> tuple[CostEntry, ...]:
        cursor = self._conn.execute(
            "SELECT * FROM cost_entries WHERE kind = ? ORDER BY id ASC",
            (kind.value,),
        )
        return tuple(_row_to_entry(row, cursor) for row in cursor.fetchall())

    def in_window(self, *, start: datetime, end: datetime) -> tuple[CostEntry, ...]:
        """Entries with ``start <= occurred_at <= end``.

        Validates the window orientation up-front to match the
        in-memory ledger's contract — a swapped window is a caller
        bug worth surfacing immediately.
        """
        if start > end:
            msg = "in_window: start must be <= end"
            raise ValueError(msg)
        cursor = self._conn.execute(
            "SELECT * FROM cost_entries "
            "WHERE occurred_at >= ? AND occurred_at <= ? "
            "ORDER BY id ASC",
            (start.isoformat(), end.isoformat()),
        )
        return tuple(_row_to_entry(row, cursor) for row in cursor.fetchall())

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def summarise(
        self,
        entries: tuple[CostEntry, ...] | None = None,
    ) -> CostSummary:
        """Compute a :class:`CostSummary` over ``entries`` (default:
        all). Delegates to the in-memory :class:`CostLedger` so the
        six-axis histogram lives in exactly one place — any future
        change to the bucketing rule lands in both ledgers
        simultaneously.
        """
        scope = entries if entries is not None else self.entries()
        # Re-use the in-memory ledger's aggregator. We don't keep
        # the helper ledger around — it's a stateless calculator
        # here, not a parallel store.
        helper = CostLedger()
        for e in scope:
            helper.record(e)
        return helper.summarise()

    def budget_status(
        self,
        *,
        limit_usd_micro: int,
        scope: tuple[CostEntry, ...] | None = None,
        approaching_threshold: float = 0.80,
    ) -> BudgetReport:
        """Compare cumulative cost against a budget. Delegates to the
        in-memory ledger so the threshold rules stay in one place."""
        helper = CostLedger()
        for e in scope if scope is not None else self.entries():
            helper.record(e)
        return helper.budget_status(
            limit_usd_micro=limit_usd_micro,
            approaching_threshold=approaching_threshold,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def snapshot(self) -> list[dict[str, object]]:
        """JSON-serialisable insertion-order snapshot, identical key
        set with the in-memory ledger's ``snapshot()`` so downstream
        consumers (Trace-UI, REST, run-receipt) don't branch on
        source."""
        rows: list[dict[str, object]] = []
        for e in self.entries():
            rows.append(
                {
                    "kind": e.kind.value,
                    "tool": e.tool,
                    "cost_usd_micro": e.cost_usd_micro,
                    "cost_usd": round(e.cost_usd, 6),
                    "backend": e.backend,
                    "run_id": e.run_id,
                    "channel": e.channel,
                    "domain": e.domain,
                    "prompt_tokens": e.prompt_tokens,
                    "response_tokens": e.response_tokens,
                    "unit_count": e.unit_count,
                    "occurred_at": e.occurred_at.isoformat(),
                    "notes": e.notes,
                }
            )
        return rows


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _row_to_entry(row: tuple[object, ...], cursor: sqlite3.Cursor) -> CostEntry:
    """Parse one ``cost_entries`` row back into a frozen
    :class:`CostEntry`. Tolerates future column additions (always at
    the end of the table) by indexing via column name."""
    column_names = [d[0] for d in cursor.description]
    data = dict(zip(column_names, row, strict=False))
    return CostEntry(
        kind=CostKind(str(data["kind"])),
        tool=str(data["tool"]),
        cost_usd_micro=int(data["cost_usd_micro"]),
        backend=str(data["backend"]),
        run_id=str(data["run_id"]),
        channel=str(data["channel"]),
        domain=str(data["domain"]),
        prompt_tokens=int(data["prompt_tokens"]),
        response_tokens=int(data["response_tokens"]),
        unit_count=int(data["unit_count"]),
        occurred_at=datetime.fromisoformat(str(data["occurred_at"])),
        notes=str(data["notes"]),
    )


__all__ = [
    "CostLedgerStore",
    "CostLedgerStoreError",
]
