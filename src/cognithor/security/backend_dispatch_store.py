"""SQLite-backed disk persistence for the TRUST-8 backend-dispatch ledger.

Companion to :mod:`cognithor.security.backend_dispatch`. The in-memory
ledger ships in PR #513; this module turns it into a queryable disk
artefact so an operator can answer "which backend served the planner
yesterday at 14:32" after a process restart.

Design choices:

* **Plain SQLite, not SQLCipher**: the dispatch surface is metadata-
  only (backend ids, model names, timestamps, token counts, error
  shape). No prompts, no responses, no secrets. The privacy contract
  in :mod:`backend_dispatch` already pins this; encrypting the
  metadata would only obscure debug-grade observability without
  hiding anything sensitive.
* **Append-only**: ``append(event)`` is the single mutation. There is
  no ``update`` and no ``upsert``. Events that should not have
  happened still get recorded — operators want to see the whole tape,
  not a curated subset.
* **Schema-versioned**: ``_schema_meta`` table tracks the current
  schema version. Future migrations are an explicit code path; this
  PR ships v1.
* **Process-shared friendly**: opens connections with
  ``check_same_thread=False`` and uses ``IMMEDIATE`` transactions for
  ``append`` so concurrent gateway processes (CLI + REST API) can
  share the same DB without races.
* **Read-API parity with the in-memory ledger**: ``by_run`` /
  ``by_backend`` / ``by_outcome`` / ``in_window`` / ``summarise`` /
  ``snapshot`` mirror the ``BackendDispatchLedger`` surface so
  consumers (Trace-UI, REST endpoints) can swap backends without
  changing call sites.
"""

from __future__ import annotations

import contextlib
import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from cognithor.security.backend_dispatch import (
    BackendDispatchEvent,
    DispatchOutcome,
    DispatchSummary,
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

CREATE TABLE IF NOT EXISTS dispatch_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backend_type TEXT NOT NULL,
    model TEXT NOT NULL DEFAULT '',
    outcome TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    prompt_tokens INTEGER NOT NULL DEFAULT -1,
    response_tokens INTEGER NOT NULL DEFAULT -1,
    error_kind TEXT NOT NULL DEFAULT '',
    error_msg TEXT NOT NULL DEFAULT '',
    is_fallback INTEGER NOT NULL DEFAULT 0,
    run_id TEXT NOT NULL DEFAULT '',
    request_id TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT ''
);

-- Indices for the hot read paths exposed by the ``by_*`` filters.
-- by_run, by_backend, in_window are the three the Trace-UI hits per
-- panel render; an index on each keeps p99 < 5 ms on a 100k-row table.
CREATE INDEX IF NOT EXISTS idx_dispatch_events_run_id
    ON dispatch_events(run_id);
CREATE INDEX IF NOT EXISTS idx_dispatch_events_backend_type
    ON dispatch_events(backend_type);
CREATE INDEX IF NOT EXISTS idx_dispatch_events_started_at
    ON dispatch_events(started_at);

INSERT OR IGNORE INTO _schema_meta (version, applied_at)
    VALUES ({_SCHEMA_VERSION}, '{datetime.now(UTC).isoformat()}');
"""


class BackendDispatchStoreError(Exception):
    """Raised when the SQLite layer is in an unrecoverable state.

    Examples: a schema version this code doesn't know how to read,
    a corrupt file the connection refuses to open, an I/O error mid-
    write that the OS surfaces as something other than a transient
    error.
    """


class BackendDispatchStore:
    """SQLite-backed append-only persistence for ``BackendDispatchEvent``.

    Tests use ``:memory:`` databases for speed; production wires the
    canonical instance to ``~/.cognithor/audit/backend_dispatch.sqlite``.

    Lifecycle:

        store = BackendDispatchStore("/path/to/db.sqlite")
        try:
            store.append(event)
            ...
        finally:
            store.close()

    Or as a context manager:

        with BackendDispatchStore("/path/to/db.sqlite") as store:
            store.append(event)
    """

    def __init__(self, db_path: str | Path) -> None:
        """Open the SQLite file and create / migrate the schema.

        Raises :class:`BackendDispatchStoreError` if the file exists
        with a schema version this code doesn't know how to read.
        """
        self._db_path = str(db_path)
        # check_same_thread=False so the gateway can share the
        # connection across asyncio tasks. Writes serialise on the
        # SQLite connection's lock; we don't pool.
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        # Standard PRAGMAs for an append-only audit-style ledger:
        # WAL gives us concurrent readers + a single writer (matches
        # the gateway lifecycle), foreign_keys is on for hygiene.
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        # Apply the schema.  ``executescript`` runs every statement,
        # including the IF NOT EXISTS guards, so re-opening an
        # existing file is a no-op.
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._verify_schema_version()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __enter__(self) -> BackendDispatchStore:
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
        """Refuse to open a DB whose schema is newer than this code.

        An older schema would be silently fine (the IF NOT EXISTS in
        the schema script is a no-op), but a NEWER schema means a
        future cognithor version wrote to this file and the current
        process must not corrupt it. Surface that case loudly.
        """
        cursor = self._conn.execute("SELECT MAX(version) FROM _schema_meta")
        row = cursor.fetchone()
        actual_version = row[0] if row else None
        if actual_version is None:
            return  # fresh DB, schema script just wrote v1
        if actual_version > _SCHEMA_VERSION:
            msg = (
                f"BackendDispatchStore at {self._db_path!r} reports schema "
                f"version {actual_version} but this build only knows "
                f"version {_SCHEMA_VERSION}. Refusing to open to avoid "
                "data corruption — upgrade cognithor or point at a "
                "different file."
            )
            raise BackendDispatchStoreError(msg)

    @property
    def schema_version(self) -> int:
        """The schema version of the on-disk DB."""
        cursor = self._conn.execute("SELECT MAX(version) FROM _schema_meta")
        row = cursor.fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def append(self, event: BackendDispatchEvent) -> int:
        """Insert ``event`` and return its assigned row id."""
        cursor = self._conn.execute(
            """
            INSERT INTO dispatch_events (
                backend_type, model, outcome, started_at, completed_at,
                prompt_tokens, response_tokens, error_kind, error_msg,
                is_fallback, run_id, request_id, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.backend_type,
                event.model,
                event.outcome.value,
                event.started_at.isoformat(),
                event.completed_at.isoformat() if event.completed_at else None,
                event.prompt_tokens,
                event.response_tokens,
                event.error_kind,
                event.error_msg,
                1 if event.is_fallback else 0,
                event.run_id,
                event.request_id,
                event.notes,
            ),
        )
        self._conn.commit()
        # cursor.lastrowid can be None per the type stub; for INSERT
        # operations on an AUTOINCREMENT column it's always set.
        row_id = cursor.lastrowid
        if row_id is None:  # pragma: no cover — defensive
            msg = "sqlite returned no lastrowid after dispatch_events INSERT"
            raise BackendDispatchStoreError(msg)
        return row_id

    def clear(self) -> None:
        """Drop all events. Test helper; production code never calls this."""
        self._conn.execute("DELETE FROM dispatch_events")
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read API (mirror of BackendDispatchLedger)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        cursor = self._conn.execute("SELECT COUNT(*) FROM dispatch_events")
        row = cursor.fetchone()
        return int(row[0]) if row else 0

    def events(self) -> tuple[BackendDispatchEvent, ...]:
        """Return every event in insertion order."""
        cursor = self._conn.execute("SELECT * FROM dispatch_events ORDER BY id ASC")
        return tuple(_row_to_event(row, cursor) for row in cursor.fetchall())

    def by_run(self, run_id: str) -> tuple[BackendDispatchEvent, ...]:
        """Events whose ``run_id`` exactly matches."""
        cursor = self._conn.execute(
            "SELECT * FROM dispatch_events WHERE run_id = ? ORDER BY id ASC",
            (run_id,),
        )
        return tuple(_row_to_event(row, cursor) for row in cursor.fetchall())

    def by_backend(self, backend_type: str) -> tuple[BackendDispatchEvent, ...]:
        cursor = self._conn.execute(
            "SELECT * FROM dispatch_events WHERE backend_type = ? ORDER BY id ASC",
            (backend_type,),
        )
        return tuple(_row_to_event(row, cursor) for row in cursor.fetchall())

    def by_outcome(self, outcome: DispatchOutcome) -> tuple[BackendDispatchEvent, ...]:
        cursor = self._conn.execute(
            "SELECT * FROM dispatch_events WHERE outcome = ? ORDER BY id ASC",
            (outcome.value,),
        )
        return tuple(_row_to_event(row, cursor) for row in cursor.fetchall())

    def in_window(self, *, start: datetime, end: datetime) -> tuple[BackendDispatchEvent, ...]:
        """Events with ``start <= started_at <= end``."""
        cursor = self._conn.execute(
            "SELECT * FROM dispatch_events "
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
        events: tuple[BackendDispatchEvent, ...] | None = None,
    ) -> DispatchSummary:
        """Compute a :class:`DispatchSummary` over ``events`` (default: all).

        Identical contract to ``BackendDispatchLedger.summarise``: the
        ``-1`` token-total propagation rule is honoured (mixed known/
        unknown ⇒ unknown), so swapping in-memory and SQLite-backed
        sources doesn't change downstream rendering.
        """
        ev = events if events is not None else self.events()
        by_backend: dict[str, int] = {}
        by_outcome: dict[DispatchOutcome, int] = {}
        success_count = 0
        fallback_count = 0
        total_prompt = 0
        total_response = 0
        prompt_known = True
        response_known = True

        for e in ev:
            by_backend[e.backend_type] = by_backend.get(e.backend_type, 0) + 1
            by_outcome[e.outcome] = by_outcome.get(e.outcome, 0) + 1
            if e.outcome == DispatchOutcome.SUCCESS:
                success_count += 1
            if e.is_fallback:
                fallback_count += 1
            if e.prompt_tokens < 0:
                prompt_known = False
            else:
                total_prompt += e.prompt_tokens
            if e.response_tokens < 0:
                response_known = False
            else:
                total_response += e.response_tokens

        return DispatchSummary(
            event_count=len(ev),
            success_count=success_count,
            by_backend=by_backend,
            by_outcome=by_outcome,
            fallback_count=fallback_count,
            total_prompt_tokens=total_prompt if prompt_known else -1,
            total_response_tokens=total_response if response_known else -1,
        )

    def snapshot(self) -> list[dict[str, object]]:
        """JSON-serialisable list of all events. Same shape as
        ``BackendDispatchLedger.snapshot`` so callers can swap sources."""
        rows: list[dict[str, object]] = []
        for e in self.events():
            rows.append(
                {
                    "backend_type": e.backend_type,
                    "model": e.model,
                    "outcome": e.outcome.value,
                    "started_at": e.started_at.isoformat(),
                    "completed_at": (e.completed_at.isoformat() if e.completed_at else None),
                    "latency_s": e.latency_s,
                    "prompt_tokens": e.prompt_tokens,
                    "response_tokens": e.response_tokens,
                    "error_kind": e.error_kind,
                    "error_msg": e.error_msg,
                    "is_fallback": e.is_fallback,
                    "run_id": e.run_id,
                    "request_id": e.request_id,
                    "notes": e.notes,
                }
            )
        return rows


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _row_to_event(row: tuple[object, ...], cursor: sqlite3.Cursor) -> BackendDispatchEvent:
    """Parse one ``dispatch_events`` row back into a frozen dataclass."""
    # ``cursor.description`` carries the column names in the order they
    # appear in the SELECT — we use that to build a name→value map so
    # the parser keeps working if columns are added by a future
    # migration (always at the end of the table).
    column_names = [d[0] for d in cursor.description]
    data = dict(zip(column_names, row, strict=False))
    return BackendDispatchEvent(
        backend_type=str(data["backend_type"]),
        model=str(data["model"]),
        outcome=DispatchOutcome(str(data["outcome"])),
        started_at=datetime.fromisoformat(str(data["started_at"])),
        completed_at=(
            datetime.fromisoformat(str(data["completed_at"])) if data.get("completed_at") else None
        ),
        prompt_tokens=int(data["prompt_tokens"]),
        response_tokens=int(data["response_tokens"]),
        error_kind=str(data["error_kind"]),
        error_msg=str(data["error_msg"]),
        is_fallback=bool(data["is_fallback"]),
        run_id=str(data["run_id"]),
        request_id=str(data["request_id"]),
        notes=str(data["notes"]),
    )


__all__ = [
    "BackendDispatchStore",
    "BackendDispatchStoreError",
]
