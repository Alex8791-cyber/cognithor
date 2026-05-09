"""SQLite-backed disk persistence for the TRUST-7 fingerprint ledger.

Companion to :mod:`cognithor.security.fingerprint`. The in-memory
ledger answers "what bytes did I see during this process" in O(1);
this module turns the same query into one an operator can ask the
day after a process restart — answering "did this exact build of
``ollama`` run yesterday's audit window?".

Design parity with :mod:`cognithor.security.backend_dispatch_store`
(#515) and :mod:`cognithor.security.cloud_escalation_store` (#516):

* Plain SQLite — the artefact metadata is small and auditor-friendly.
  No prompts, no responses, no model weights — just hashes and names.
* Append-only on the hash dimension: registering a hash that already
  exists is a no-op (mirrors the in-memory ledger's idempotency).
* Schema-versioned via ``_schema_meta``; loud refusal on newer-than-
  build databases.
* WAL journal mode + ``check_same_thread=False`` so concurrent
  gateway processes can share the file.
* Read-API parity with :class:`FingerprintLedger`: ``__len__``,
  ``__contains__``, ``get``, ``history``, ``names``, ``filter_by_kind``,
  ``divergent_names``, ``snapshot``.
* Indices on ``content_hash`` (UNIQUE), ``name``, ``kind``,
  ``captured_at`` — the four hot read paths. Pinned by a
  ``TestIndices`` contract.

Schema notes:

* ``content_hash`` is ``TEXT UNIQUE NOT NULL`` — the SQLite-level
  enforcement of the in-memory ledger's "first-write wins" rule.
* The ``register`` method swallows
  :class:`sqlite3.IntegrityError` on the unique index and returns
  ``False`` to mirror the in-memory contract exactly.
"""

from __future__ import annotations

import contextlib
import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from cognithor.security.fingerprint import BinaryKind, ToolFingerprint
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

CREATE TABLE IF NOT EXISTS fingerprint_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    version TEXT NOT NULL DEFAULT '',
    captured_at TEXT NOT NULL,
    source_path TEXT NOT NULL DEFAULT '',
    upstream_url TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT ''
);

-- Indices match the four hot read paths on the in-memory ledger:
-- get (by hash, already covered by UNIQUE), history (by name),
-- filter_by_kind (by kind), and snapshot ordering (by captured_at).
-- Pinning them in the schema prevents a future migration from
-- silently dropping one — Trace-UI panels would table-scan on every
-- render.
CREATE INDEX IF NOT EXISTS idx_fingerprint_events_name
    ON fingerprint_events(name);
CREATE INDEX IF NOT EXISTS idx_fingerprint_events_kind
    ON fingerprint_events(kind);
CREATE INDEX IF NOT EXISTS idx_fingerprint_events_captured_at
    ON fingerprint_events(captured_at);

INSERT OR IGNORE INTO _schema_meta (version, applied_at)
    VALUES ({_SCHEMA_VERSION}, '{datetime.now(UTC).isoformat()}');
"""


class FingerprintStoreError(Exception):
    """Raised when the SQLite layer is in an unrecoverable state.

    Mirrors :class:`BackendDispatchStoreError` and
    :class:`CloudEscalationStoreError`. Examples include a schema
    version this build doesn't know how to read, or a corrupt file
    whose connection won't open cleanly.
    """


class FingerprintStore:
    """SQLite-backed append-only persistence for ``ToolFingerprint``.

    Tests use ``:memory:`` databases for speed; production wires the
    canonical instance to ``~/.cognithor/audit/fingerprints.sqlite``.

    Lifecycle mirrors :class:`BackendDispatchStore` and
    :class:`CloudEscalationStore` — context-manager or explicit
    ``close()``. Re-opening an existing file is a no-op (idempotent
    schema script).
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

    def __enter__(self) -> FingerprintStore:
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
                f"FingerprintStore at {self._db_path!r} reports schema "
                f"version {actual_version} but this build only knows "
                f"version {_SCHEMA_VERSION}. Refusing to open to avoid "
                "data corruption — upgrade cognithor or point at a "
                "different file."
            )
            raise FingerprintStoreError(msg)

    @property
    def schema_version(self) -> int:
        cursor = self._conn.execute("SELECT MAX(version) FROM _schema_meta")
        row = cursor.fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def register(self, fingerprint: ToolFingerprint) -> bool:
        """Register ``fingerprint``. Returns True if it was new.

        Mirrors :meth:`FingerprintLedger.register` exactly:

        * Re-registering an identical hash is a no-op and returns
          False (the SQLite UNIQUE constraint enforces this; the
          IntegrityError is caught and translated to ``False``).
        * Registering a *new* hash for an existing name appends and
          returns True — the name-history grows by one entry.
        """
        try:
            self._conn.execute(
                """
                INSERT INTO fingerprint_events (
                    name, kind, content_hash, version, captured_at,
                    source_path, upstream_url, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fingerprint.name,
                    fingerprint.kind.value,
                    fingerprint.content_hash,
                    fingerprint.version,
                    fingerprint.captured_at.isoformat(),
                    fingerprint.source_path,
                    fingerprint.upstream_url,
                    fingerprint.notes,
                ),
            )
        except sqlite3.IntegrityError:
            # Hash already present — idempotent register, mirrors
            # the in-memory ledger's "return False on duplicate".
            return False
        self._conn.commit()
        return True

    def remove(self, content_hash: str) -> bool:
        """Drop a fingerprint by hash. Returns True iff it existed.

        Mirrors :meth:`FingerprintLedger.remove`. Used by the test
        suite and post-mortem replay; production code shouldn't
        normally forget fingerprints.
        """
        cursor = self._conn.execute(
            "DELETE FROM fingerprint_events WHERE content_hash = ?",
            (content_hash,),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def clear(self) -> None:
        """Drop all fingerprints. Test helper; production never calls."""
        self._conn.execute("DELETE FROM fingerprint_events")
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read API (mirror of FingerprintLedger)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        cursor = self._conn.execute("SELECT COUNT(*) FROM fingerprint_events")
        row = cursor.fetchone()
        return int(row[0]) if row else 0

    def __contains__(self, content_hash: object) -> bool:
        if not isinstance(content_hash, str):
            return False
        cursor = self._conn.execute(
            "SELECT 1 FROM fingerprint_events WHERE content_hash = ? LIMIT 1",
            (content_hash,),
        )
        return cursor.fetchone() is not None

    def get(self, content_hash: str) -> ToolFingerprint | None:
        """Return the fingerprint with ``content_hash`` or ``None``."""
        cursor = self._conn.execute(
            "SELECT * FROM fingerprint_events WHERE content_hash = ? LIMIT 1",
            (content_hash,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return _row_to_fingerprint(row, cursor)

    def history(self, name: str) -> tuple[ToolFingerprint, ...]:
        """Return every fingerprint under ``name`` (oldest captured_at first).

        Mirrors :meth:`FingerprintLedger.history`. Tie-break on row
        ``id`` so two captures with the same timestamp keep insertion
        order.
        """
        cursor = self._conn.execute(
            "SELECT * FROM fingerprint_events WHERE name = ? ORDER BY captured_at ASC, id ASC",
            (name,),
        )
        return tuple(_row_to_fingerprint(row, cursor) for row in cursor.fetchall())

    def names(self) -> list[str]:
        """Return the set of registered names, sorted."""
        cursor = self._conn.execute(
            "SELECT DISTINCT name FROM fingerprint_events ORDER BY name ASC"
        )
        return [str(row[0]) for row in cursor.fetchall()]

    def filter_by_kind(self, kind: BinaryKind) -> list[ToolFingerprint]:
        """Return all fingerprints of ``kind``, sorted by name then hash."""
        cursor = self._conn.execute(
            "SELECT * FROM fingerprint_events WHERE kind = ? ORDER BY name ASC, content_hash ASC",
            (kind.value,),
        )
        return [_row_to_fingerprint(row, cursor) for row in cursor.fetchall()]

    def divergent_names(self) -> list[str]:
        """Names with more than one distinct hash registered.

        Mirrors :meth:`FingerprintLedger.divergent_names`. The
        smoking-gun query when post-mortem reconstruction shows
        behaviour drift: "show me every tool that changed during this
        audit window".
        """
        cursor = self._conn.execute(
            "SELECT name FROM fingerprint_events "
            "GROUP BY name "
            "HAVING COUNT(DISTINCT content_hash) > 1 "
            "ORDER BY name ASC"
        )
        return [str(row[0]) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def snapshot(self) -> list[dict[str, object]]:
        """JSON-serialisable snapshot, identical key set with the
        in-memory ledger's ``snapshot()`` so downstream consumers
        (Trace-UI, REST, run-receipt) don't branch on source.

        Stable ordering: by ``name`` then ``captured_at`` then
        ``content_hash`` — same as :meth:`FingerprintLedger.snapshot`.
        """
        cursor = self._conn.execute(
            "SELECT * FROM fingerprint_events ORDER BY name ASC, captured_at ASC, content_hash ASC"
        )
        rows: list[dict[str, object]] = []
        for row in cursor.fetchall():
            fp = _row_to_fingerprint(row, cursor)
            rows.append(
                {
                    "name": fp.name,
                    "kind": fp.kind.value,
                    "content_hash": fp.content_hash,
                    "short_hash": fp.short_hash,
                    "version": fp.version,
                    "captured_at": fp.captured_at.isoformat(),
                    "source_path": fp.source_path,
                    "upstream_url": fp.upstream_url,
                    "notes": fp.notes,
                }
            )
        return rows


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _row_to_fingerprint(row: tuple[object, ...], cursor: sqlite3.Cursor) -> ToolFingerprint:
    """Parse one ``fingerprint_events`` row back into a frozen
    :class:`ToolFingerprint`. Tolerates future column additions
    (always at the end of the table) by indexing via column name."""
    column_names = [d[0] for d in cursor.description]
    data = dict(zip(column_names, row, strict=False))
    return ToolFingerprint(
        name=str(data["name"]),
        kind=BinaryKind(str(data["kind"])),
        content_hash=str(data["content_hash"]),
        version=str(data["version"]),
        captured_at=datetime.fromisoformat(str(data["captured_at"])),
        source_path=str(data["source_path"]),
        upstream_url=str(data["upstream_url"]),
        notes=str(data["notes"]),
    )


__all__ = [
    "FingerprintStore",
    "FingerprintStoreError",
]
