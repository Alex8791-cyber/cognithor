"""SQLite-backed disk persistence for the TRUST-8 cloud-escalation ledger.

Companion to :mod:`cognithor.security.cloud_escalation`. The in-memory
ledger answers "did this request leave the box" in O(1) within a
single process; this module turns the same query into one an operator
can ask the day after a process restart.

Design parity with :mod:`cognithor.security.backend_dispatch_store`
(landed in PR #515):

* Plain SQLite — metadata-only privacy contract (backend ids, token
  counts, USD cost in micro-units, reason taxonomy). No prompts, no
  responses.
* Append-only — ``append(event)`` is the single mutation.
* Schema-versioned — ``_schema_meta`` table tracks the current
  version; loud refusal to open DBs whose version is > current build.
* WAL journal mode + ``check_same_thread=False`` so concurrent
  gateway processes can share the file safely.
* Read-API parity with :class:`EscalationLedger` (in-memory):
  ``__len__``, ``events``, ``by_reason``, ``by_destination``,
  ``by_run``, ``in_window``, ``summarise``, ``snapshot``.
* Indices on ``run_id``, ``to_backend``, ``reason``, ``started_at``
  for the four hot Trace-UI read paths.

The escalation events have one extra field vs dispatch events:
``cost_usd_micro`` (integer micro-USD, summable without float drift).
The schema reserves an integer column for it; the in-memory summary
matches the on-disk one.
"""

from __future__ import annotations

import contextlib
import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from cognithor.security.cloud_escalation import (
    EscalationEvent,
    EscalationReason,
    EscalationSummary,
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

CREATE TABLE IF NOT EXISTS escalation_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reason TEXT NOT NULL,
    from_backend TEXT NOT NULL,
    to_backend TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    response_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd_micro INTEGER NOT NULL DEFAULT 0,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    owner_consented INTEGER NOT NULL DEFAULT 0,
    run_id TEXT NOT NULL DEFAULT '',
    request_id TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT ''
);

-- Indices match the four hot read paths on the in-memory ledger:
-- by_reason, by_destination, by_run, in_window. Pinning them in the
-- schema prevents a future migration from silently dropping one.
CREATE INDEX IF NOT EXISTS idx_escalation_events_run_id
    ON escalation_events(run_id);
CREATE INDEX IF NOT EXISTS idx_escalation_events_to_backend
    ON escalation_events(to_backend);
CREATE INDEX IF NOT EXISTS idx_escalation_events_reason
    ON escalation_events(reason);
CREATE INDEX IF NOT EXISTS idx_escalation_events_started_at
    ON escalation_events(started_at);

INSERT OR IGNORE INTO _schema_meta (version, applied_at)
    VALUES ({_SCHEMA_VERSION}, '{datetime.now(UTC).isoformat()}');
"""


class CloudEscalationStoreError(Exception):
    """Raised when the SQLite layer is in an unrecoverable state.

    Mirrors :class:`BackendDispatchStoreError` — examples include a
    schema version this build doesn't know how to read, or a corrupt
    file whose connection won't open cleanly.
    """


class CloudEscalationStore:
    """SQLite-backed append-only persistence for ``EscalationEvent``.

    Tests use ``:memory:`` databases for speed; production wires the
    canonical instance to ``~/.cognithor/audit/cloud_escalation.sqlite``.

    Lifecycle mirrors :class:`BackendDispatchStore` — context-manager
    or explicit ``close()``. Re-opening an existing file is a no-op
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

    def __enter__(self) -> CloudEscalationStore:
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
                f"CloudEscalationStore at {self._db_path!r} reports schema "
                f"version {actual_version} but this build only knows "
                f"version {_SCHEMA_VERSION}. Refusing to open to avoid "
                "data corruption — upgrade cognithor or point at a "
                "different file."
            )
            raise CloudEscalationStoreError(msg)

    @property
    def schema_version(self) -> int:
        cursor = self._conn.execute("SELECT MAX(version) FROM _schema_meta")
        row = cursor.fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def append(self, event: EscalationEvent) -> int:
        """Insert ``event`` and return its assigned row id."""
        cursor = self._conn.execute(
            """
            INSERT INTO escalation_events (
                reason, from_backend, to_backend,
                prompt_tokens, response_tokens, cost_usd_micro,
                started_at, completed_at, owner_consented,
                run_id, request_id, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.reason.value,
                event.from_backend,
                event.to_backend,
                event.prompt_tokens,
                event.response_tokens,
                event.cost_usd_micro,
                event.started_at.isoformat(),
                event.completed_at.isoformat() if event.completed_at else None,
                1 if event.owner_consented else 0,
                event.run_id,
                event.request_id,
                event.notes,
            ),
        )
        self._conn.commit()
        row_id = cursor.lastrowid
        if row_id is None:  # pragma: no cover — defensive
            msg = "sqlite returned no lastrowid after escalation_events INSERT"
            raise CloudEscalationStoreError(msg)
        return row_id

    def clear(self) -> None:
        """Drop all events. Test helper; production never calls this."""
        self._conn.execute("DELETE FROM escalation_events")
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read API (mirror of EscalationLedger)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        cursor = self._conn.execute("SELECT COUNT(*) FROM escalation_events")
        row = cursor.fetchone()
        return int(row[0]) if row else 0

    def events(self) -> tuple[EscalationEvent, ...]:
        """Return every event in insertion order."""
        cursor = self._conn.execute("SELECT * FROM escalation_events ORDER BY id ASC")
        return tuple(_row_to_event(row, cursor) for row in cursor.fetchall())

    def by_reason(self, reason: EscalationReason) -> tuple[EscalationEvent, ...]:
        cursor = self._conn.execute(
            "SELECT * FROM escalation_events WHERE reason = ? ORDER BY id ASC",
            (reason.value,),
        )
        return tuple(_row_to_event(row, cursor) for row in cursor.fetchall())

    def by_destination(self, to_backend: str) -> tuple[EscalationEvent, ...]:
        cursor = self._conn.execute(
            "SELECT * FROM escalation_events WHERE to_backend = ? ORDER BY id ASC",
            (to_backend,),
        )
        return tuple(_row_to_event(row, cursor) for row in cursor.fetchall())

    def by_run(self, run_id: str) -> tuple[EscalationEvent, ...]:
        """Events tied to ``run_id`` — the TRUST-1 cross-reference query.

        Mirrors the in-memory ledger's empty-string short-circuit so
        sentinel values used by un-tracked runs don't accidentally
        return the entire ledger."""
        if not run_id:
            return ()
        cursor = self._conn.execute(
            "SELECT * FROM escalation_events WHERE run_id = ? ORDER BY id ASC",
            (run_id,),
        )
        return tuple(_row_to_event(row, cursor) for row in cursor.fetchall())

    def in_window(self, *, start: datetime, end: datetime) -> tuple[EscalationEvent, ...]:
        """Events with ``start <= started_at <= end``.

        Validates the window orientation up-front to match the
        in-memory ledger's contract — a swapped window is a caller
        bug worth surfacing immediately.
        """
        if start > end:
            msg = "in_window: start must be <= end"
            raise ValueError(msg)
        cursor = self._conn.execute(
            "SELECT * FROM escalation_events "
            "WHERE started_at >= ? AND started_at <= ? "
            "ORDER BY id ASC",
            (start.isoformat(), end.isoformat()),
        )
        return tuple(_row_to_event(row, cursor) for row in cursor.fetchall())

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def summarise(
        self,
        events: tuple[EscalationEvent, ...] | None = None,
    ) -> EscalationSummary:
        """Compute an :class:`EscalationSummary` over ``events``
        (default: all). Same contract as the in-memory ledger so the
        rendered Trace-UI tile is identical regardless of source.
        """
        scope = events if events is not None else self.events()
        by_reason: dict[EscalationReason, int] = {}
        by_destination: dict[str, int] = {}
        total_prompt = 0
        total_response = 0
        total_cost = 0
        for ev in scope:
            by_reason[ev.reason] = by_reason.get(ev.reason, 0) + 1
            by_destination[ev.to_backend] = by_destination.get(ev.to_backend, 0) + 1
            total_prompt += ev.prompt_tokens
            total_response += ev.response_tokens
            total_cost += ev.cost_usd_micro
        return EscalationSummary(
            event_count=len(scope),
            total_prompt_tokens=total_prompt,
            total_response_tokens=total_response,
            total_cost_usd_micro=total_cost,
            by_reason=by_reason,
            by_destination=by_destination,
        )

    def snapshot(self) -> list[dict[str, object]]:
        """JSON-serialisable list of all events, identical key set
        with the in-memory ledger's ``snapshot()`` so downstream
        consumers (Trace-UI, REST, run-receipt) don't branch on
        source."""
        rows: list[dict[str, object]] = []
        for ev in self.events():
            rows.append(
                {
                    "reason": ev.reason.value,
                    "from_backend": ev.from_backend,
                    "to_backend": ev.to_backend,
                    "prompt_tokens": ev.prompt_tokens,
                    "response_tokens": ev.response_tokens,
                    "cost_usd_micro": ev.cost_usd_micro,
                    "cost_usd": round(ev.cost_usd, 6),
                    "started_at": ev.started_at.isoformat(),
                    "completed_at": (
                        ev.completed_at.isoformat() if ev.completed_at is not None else None
                    ),
                    "owner_consented": ev.owner_consented,
                    "run_id": ev.run_id,
                    "request_id": ev.request_id,
                    "notes": ev.notes,
                }
            )
        return rows


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _row_to_event(row: tuple[object, ...], cursor: sqlite3.Cursor) -> EscalationEvent:
    """Parse one ``escalation_events`` row back into a frozen
    :class:`EscalationEvent`. Tolerates future column additions
    (always at the end of the table) by indexing via column name."""
    column_names = [d[0] for d in cursor.description]
    data = dict(zip(column_names, row, strict=False))
    return EscalationEvent(
        reason=EscalationReason(str(data["reason"])),
        from_backend=str(data["from_backend"]),
        to_backend=str(data["to_backend"]),
        prompt_tokens=int(data["prompt_tokens"]),
        response_tokens=int(data["response_tokens"]),
        cost_usd_micro=int(data["cost_usd_micro"]),
        started_at=datetime.fromisoformat(str(data["started_at"])),
        completed_at=(
            datetime.fromisoformat(str(data["completed_at"])) if data.get("completed_at") else None
        ),
        owner_consented=bool(data["owner_consented"]),
        run_id=str(data["run_id"]),
        request_id=str(data["request_id"]),
        notes=str(data["notes"]),
    )


__all__ = [
    "CloudEscalationStore",
    "CloudEscalationStoreError",
]
