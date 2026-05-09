"""SQLite-backed disk persistence for the TRUST-10 migration ledger.

Companion to :mod:`cognithor.security.migration_ledger`. The
in-memory ledger answers "what schema is this domain on right now"
within a single process; this module turns the same query into one
an operator can ask after a process restart — closing the
reviewer-feedback gap "the migration trail is implicit".

Design parity with :mod:`cognithor.security.backend_dispatch_store`
(#515), :mod:`cognithor.security.cloud_escalation_store` (#516),
:mod:`cognithor.security.fingerprint_store` (#517), and
:mod:`cognithor.security.cost_ledger_store` (#518):

* Plain SQLite, append-only, WAL journal mode +
  ``check_same_thread=False``.
* Schema-versioned via ``_schema_meta``; loud refusal on
  newer-than-build databases.
* Read-API parity with :class:`MigrationLedger`: ``__len__``,
  ``steps``, ``for_domain``, ``head_version``, ``get``,
  ``applied_only``, ``snapshot``.
* Indices on ``domain``, ``status``, ``migration_id``,
  ``applied_at`` — the four hot read paths. Pinned by a
  ``TestIndices`` contract.

What's different from the sibling stores
----------------------------------------

The migration ledger has a non-trivial **per-domain chain
invariant**: ``record()`` rejects a step whose ``source_version``
doesn't match the domain head (for APPLIED / ROLLED_BACK statuses),
plus rollback-of and migration_id uniqueness.

To avoid divergence, ``MigrationLedgerStore.record`` re-uses the
in-memory ledger's validation by replaying the on-disk history
through a fresh :class:`MigrationLedger` and letting it call its
own ``record()``. If that raises :class:`MigrationChainError`, the
on-disk write is rejected. If it accepts, we INSERT and commit.

Cost: O(n) per record, where n is total steps for that ledger. For
realistic migration ledgers (tens to low hundreds of steps over the
project's lifetime) this is negligible. The alternative (re-implementing
chain validation in SQL) would split the integrity contract into two
copies that could drift apart silently — exactly the bug TRUST-10
exists to prevent.

Snapshot shape
--------------

The migration-ledger snapshot is a **dict** (not a list) containing
``head_version`` + ``steps``. We mirror that contract exactly so
Trace-UI consumers don't branch on source.
"""

from __future__ import annotations

import contextlib
import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from cognithor.security.migration_ledger import (
    MigrationDomain,
    MigrationLedger,
    MigrationStatus,
    MigrationStep,
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

CREATE TABLE IF NOT EXISTS migration_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    domain TEXT NOT NULL,
    source_version TEXT NOT NULL,
    target_version TEXT NOT NULL,
    status TEXT NOT NULL,
    applied_at TEXT NOT NULL,
    applied_by TEXT NOT NULL DEFAULT '',
    item_count INTEGER NOT NULL DEFAULT -1,
    checksum_before TEXT NOT NULL DEFAULT '',
    checksum_after TEXT NOT NULL DEFAULT '',
    rollback_of TEXT NOT NULL DEFAULT '',
    migration_id TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT ''
);

-- Indices match the four hot read paths on the in-memory ledger:
-- for_domain, applied_only (filters by status), get (by migration_id),
-- and snapshot ordering (by applied_at). Pinning them prevents a
-- future migration from silently dropping one — Trace-UI panels
-- would table-scan on every render.
CREATE INDEX IF NOT EXISTS idx_migration_steps_domain
    ON migration_steps(domain);
CREATE INDEX IF NOT EXISTS idx_migration_steps_status
    ON migration_steps(status);
CREATE INDEX IF NOT EXISTS idx_migration_steps_migration_id
    ON migration_steps(migration_id);
CREATE INDEX IF NOT EXISTS idx_migration_steps_applied_at
    ON migration_steps(applied_at);

INSERT OR IGNORE INTO _schema_meta (version, applied_at)
    VALUES ({_SCHEMA_VERSION}, '{datetime.now(UTC).isoformat()}');
"""


class MigrationLedgerStoreError(Exception):
    """Raised when the SQLite layer is in an unrecoverable state.

    Mirrors the sibling stores. Examples include a schema version
    this build doesn't know how to read, or a corrupt file whose
    connection won't open cleanly.
    """


class MigrationLedgerStore:
    """SQLite-backed append-only persistence for ``MigrationStep``.

    Tests use ``:memory:`` databases for speed; production wires the
    canonical instance to ``~/.cognithor/audit/migrations.sqlite``.

    Lifecycle mirrors the sibling stores. Re-opening an existing file
    is a no-op (idempotent schema script).
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

    def __enter__(self) -> MigrationLedgerStore:
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
                f"MigrationLedgerStore at {self._db_path!r} reports schema "
                f"version {actual_version} but this build only knows "
                f"version {_SCHEMA_VERSION}. Refusing to open to avoid "
                "data corruption — upgrade cognithor or point at a "
                "different file."
            )
            raise MigrationLedgerStoreError(msg)

    @property
    def schema_version(self) -> int:
        cursor = self._conn.execute("SELECT MAX(version) FROM _schema_meta")
        row = cursor.fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def record(self, step: MigrationStep) -> int:
        """Append ``step`` and return its assigned row id.

        Validation re-uses :class:`MigrationLedger`'s own ``record()``
        — we replay the on-disk history into a fresh in-memory
        ledger and let it call ``record(step)``. If the in-memory
        ledger raises :class:`MigrationChainError` (chain mismatch,
        unknown rollback target, duplicate migration_id, etc.), the
        SQLite write is rejected and the exception propagates.

        This deliberately keeps the chain-integrity contract in
        exactly one place — divergence between the in-memory and
        on-disk ledgers is the kind of bug TRUST-10 is supposed to
        catch, so we don't risk it by re-implementing the rules in
        SQL.
        """
        helper = self._rebuild_in_memory()
        helper.record(step)  # raises on chain violations

        cursor = self._conn.execute(
            """
            INSERT INTO migration_steps (
                domain, source_version, target_version, status,
                applied_at, applied_by, item_count,
                checksum_before, checksum_after,
                rollback_of, migration_id, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                step.domain.value,
                step.source_version,
                step.target_version,
                step.status.value,
                step.applied_at.isoformat(),
                step.applied_by,
                step.item_count,
                step.checksum_before,
                step.checksum_after,
                step.rollback_of,
                step.migration_id,
                step.notes,
            ),
        )
        self._conn.commit()
        row_id = cursor.lastrowid
        if row_id is None:  # pragma: no cover — defensive
            msg = "sqlite returned no lastrowid after migration_steps INSERT"
            raise MigrationLedgerStoreError(msg)
        return row_id

    def clear(self) -> None:
        """Drop all steps. Test helper; production never calls this."""
        self._conn.execute("DELETE FROM migration_steps")
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read API (mirror of MigrationLedger)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        cursor = self._conn.execute("SELECT COUNT(*) FROM migration_steps")
        row = cursor.fetchone()
        return int(row[0]) if row else 0

    def steps(self) -> tuple[MigrationStep, ...]:
        """All steps in insertion order."""
        cursor = self._conn.execute("SELECT * FROM migration_steps ORDER BY id ASC")
        return tuple(_row_to_step(row, cursor) for row in cursor.fetchall())

    def for_domain(self, domain: MigrationDomain) -> tuple[MigrationStep, ...]:
        """Steps for a single domain in insertion order."""
        cursor = self._conn.execute(
            "SELECT * FROM migration_steps WHERE domain = ? ORDER BY id ASC",
            (domain.value,),
        )
        return tuple(_row_to_step(row, cursor) for row in cursor.fetchall())

    def head_version(self, domain: MigrationDomain) -> str | None:
        """Current head version for ``domain``.

        Computed from the in-memory replay so the answer matches
        :meth:`MigrationLedger.head_version` exactly — including the
        edge case where a ROLLED_BACK step moves the head backward
        to a previous APPLIED target.
        """
        return self._rebuild_in_memory().head_version(domain)

    def get(self, migration_id: str) -> MigrationStep | None:
        """Return the step with ``migration_id`` or ``None``."""
        if not migration_id:
            return None
        cursor = self._conn.execute(
            "SELECT * FROM migration_steps WHERE migration_id = ? LIMIT 1",
            (migration_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return _row_to_step(row, cursor)

    def applied_only(self, domain: MigrationDomain) -> tuple[MigrationStep, ...]:
        """APPLIED + ROLLED_BACK steps for ``domain`` (head-moving subset)."""
        cursor = self._conn.execute(
            "SELECT * FROM migration_steps WHERE domain = ? AND status IN (?, ?) ORDER BY id ASC",
            (
                domain.value,
                MigrationStatus.APPLIED.value,
                MigrationStatus.ROLLED_BACK.value,
            ),
        )
        return tuple(_row_to_step(row, cursor) for row in cursor.fetchall())

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, object]:
        """JSON-serialisable representation, identical shape to the
        in-memory :meth:`MigrationLedger.snapshot`. Trace-UI
        consumers don't branch on source.

        Shape::

            {
              "head_version": {"<domain>": "<version>", ...},
              "steps": [<step-dict>, ...]
            }
        """
        helper = self._rebuild_in_memory()
        return helper.snapshot()

    # ------------------------------------------------------------------
    # Internal — rebuild an in-memory ledger from disk
    # ------------------------------------------------------------------

    def _rebuild_in_memory(self) -> MigrationLedger:
        """Re-play the on-disk history into a fresh in-memory ledger.

        Used by ``record`` (chain validation), ``head_version``, and
        ``snapshot``. Bypasses the in-memory ledger's chain checks
        for the replay itself — the on-disk history is already valid
        because every step was validated when it was first written.
        """
        ledger = MigrationLedger()
        for s in self.steps():
            # Replay raw — the chain validation already happened on
            # the way in. We want the resulting state, not another
            # round of validation against a half-built head map.
            ledger._steps.append(s)
            if s.migration_id:
                ledger._by_id[s.migration_id] = s
            if s.status in {MigrationStatus.APPLIED, MigrationStatus.ROLLED_BACK}:
                ledger._head_version[s.domain] = s.target_version
        return ledger


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _row_to_step(row: tuple[object, ...], cursor: sqlite3.Cursor) -> MigrationStep:
    """Parse one ``migration_steps`` row back into a frozen
    :class:`MigrationStep`. Tolerates future column additions
    (always at the end of the table) by indexing via column name."""
    column_names = [d[0] for d in cursor.description]
    data = dict(zip(column_names, row, strict=False))
    return MigrationStep(
        domain=MigrationDomain(str(data["domain"])),
        source_version=str(data["source_version"]),
        target_version=str(data["target_version"]),
        status=MigrationStatus(str(data["status"])),
        applied_at=datetime.fromisoformat(str(data["applied_at"])),
        applied_by=str(data["applied_by"]),
        item_count=int(data["item_count"]),
        checksum_before=str(data["checksum_before"]),
        checksum_after=str(data["checksum_after"]),
        rollback_of=str(data["rollback_of"]),
        migration_id=str(data["migration_id"]),
        notes=str(data["notes"]),
    )


__all__ = [
    "MigrationLedgerStore",
    "MigrationLedgerStoreError",
]
