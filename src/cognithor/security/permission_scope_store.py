"""SQLite-backed disk persistence for the TRUST-5 scope registry.

Companion to :mod:`cognithor.security.permission_scope`. Closes the
remaining gap of the disk-persistence sprint: the gatekeeper's
per-(axis, identity) scope rules survive a process restart and don't
vanish into a re-loaded YAML config blob whose lineage is opaque.

Design parity with :mod:`cognithor.security.backend_dispatch_store`
(#515), :mod:`cognithor.security.cloud_escalation_store` (#516),
:mod:`cognithor.security.fingerprint_store` (#517),
:mod:`cognithor.security.cost_ledger_store` (#518), and
:mod:`cognithor.security.migration_ledger_store` (#519):

* Plain SQLite, WAL journal mode + ``check_same_thread=False``.
* Schema-versioned via ``_schema_meta``; loud refusal on
  newer-than-build databases.
* Read-API parity with :class:`ScopeRegistry`: ``__len__``,
  ``get``, ``list_scopes``, ``register``, ``remove``, ``clear``,
  ``evaluate``, ``assert_allowed``.
* Indices on ``axis`` for the hot ``list_scopes`` filter — the
  primary read path is by ``(axis, identity)`` which the table's
  composite primary key already covers.

What's different from the sibling stores
----------------------------------------

The scope registry is a **registry**, not an append-only ledger:
``register(scope)`` upserts on ``(axis, identity)``, replacing any
prior scope for that key. The disk store mirrors this with
``INSERT … ON CONFLICT(axis, identity) DO UPDATE`` so the contract
"calling register twice on the same key keeps only the latest" is
preserved across reopens.

Single source of truth for evaluation
-------------------------------------

``evaluate`` and ``assert_allowed`` rebuild an in-memory
:class:`ScopeRegistry` from the on-disk rows and delegate to its
own evaluator, so the most-restrictive-wins precedence logic stays
in exactly one place — divergence between in-memory and disk
verdicts is the kind of bug TRUST-5 exists to prevent.

Privacy contract
----------------

All columns are policy metadata. No prompts, no responses, no user
content — only axis names, identity strings (e.g. ``"telegram"``,
``"alex@cognithor.ai"``), tool names, and risk-level enums. Same
auditor-friendly shape as the sibling stores.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from cognithor.models import RiskLevel
from cognithor.security.permission_scope import (
    PermissionScope,
    ScopeAxis,
    ScopeRegistry,
    ScopeVerdict,
    ScopeViolation,
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

CREATE TABLE IF NOT EXISTS scope_entries (
    axis TEXT NOT NULL,
    identity TEXT NOT NULL,
    tool_allowlist TEXT NOT NULL DEFAULT '[]',
    tool_denylist TEXT NOT NULL DEFAULT '[]',
    max_risk TEXT NOT NULL DEFAULT 'red',
    PRIMARY KEY (axis, identity)
);

-- Hot read path: list_scopes() filters / sorts by axis. The (axis,
-- identity) PK already covers point lookups in the primary read
-- direction. Pinning this index prevents a future migration from
-- silently dropping it.
CREATE INDEX IF NOT EXISTS idx_scope_entries_axis
    ON scope_entries(axis);

INSERT OR IGNORE INTO _schema_meta (version, applied_at)
    VALUES ({_SCHEMA_VERSION}, '{datetime.now(UTC).isoformat()}');
"""


class PermissionScopeStoreError(Exception):
    """Raised when the SQLite layer is in an unrecoverable state.

    Mirrors the sibling stores. Examples include a schema version
    this build doesn't know how to read, or a corrupt file whose
    connection won't open cleanly.
    """


class PermissionScopeStore:
    """SQLite-backed persistence for ``PermissionScope`` rules.

    Tests use ``:memory:`` databases for speed; production wires the
    canonical instance to ``~/.cognithor/audit/scope_registry.sqlite``.

    Lifecycle mirrors the sibling stores. Re-opening an existing
    file is a no-op (idempotent schema script).
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

    def __enter__(self) -> PermissionScopeStore:
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
                f"PermissionScopeStore at {self._db_path!r} reports schema "
                f"version {actual_version} but this build only knows "
                f"version {_SCHEMA_VERSION}. Refusing to open to avoid "
                "data corruption — upgrade cognithor or point at a "
                "different file."
            )
            raise PermissionScopeStoreError(msg)

    @property
    def schema_version(self) -> int:
        cursor = self._conn.execute("SELECT MAX(version) FROM _schema_meta")
        row = cursor.fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def register(self, scope: PermissionScope) -> None:
        """Upsert ``scope``. Existing key is overwritten silently —
        scopes are config-driven and re-config should not raise.

        Mirrors :meth:`ScopeRegistry.register` exactly.
        """
        self._conn.execute(
            """
            INSERT INTO scope_entries (
                axis, identity, tool_allowlist, tool_denylist, max_risk
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(axis, identity) DO UPDATE SET
                tool_allowlist = excluded.tool_allowlist,
                tool_denylist = excluded.tool_denylist,
                max_risk = excluded.max_risk
            """,
            (
                scope.axis.value,
                scope.identity,
                json.dumps(sorted(scope.tool_allowlist)),
                json.dumps(sorted(scope.tool_denylist)),
                scope.max_risk.value,
            ),
        )
        self._conn.commit()

    def remove(self, axis: ScopeAxis, identity: str) -> bool:
        """Remove a scope by key. Returns True if a scope was deleted."""
        cursor = self._conn.execute(
            "DELETE FROM scope_entries WHERE axis = ? AND identity = ?",
            (axis.value, identity),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def clear(self) -> None:
        """Drop all scopes."""
        self._conn.execute("DELETE FROM scope_entries")
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read API (mirror of ScopeRegistry)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        cursor = self._conn.execute("SELECT COUNT(*) FROM scope_entries")
        row = cursor.fetchone()
        return int(row[0]) if row else 0

    def get(self, axis: ScopeAxis, identity: str) -> PermissionScope | None:
        cursor = self._conn.execute(
            "SELECT * FROM scope_entries WHERE axis = ? AND identity = ? LIMIT 1",
            (axis.value, identity),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return _row_to_scope(row, cursor)

    def list_scopes(self) -> list[PermissionScope]:
        """Return all scopes sorted by (axis, identity) for stable display."""
        cursor = self._conn.execute("SELECT * FROM scope_entries ORDER BY axis ASC, identity ASC")
        return [_row_to_scope(row, cursor) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Evaluation — delegates to the in-memory registry
    # ------------------------------------------------------------------

    def evaluate(
        self,
        scope_keys: list[tuple[ScopeAxis, str]],
        tool_name: str,
        tool_risk: RiskLevel,
    ) -> ScopeVerdict:
        """Return the most-restrictive verdict for the given tool call.

        Delegates to :meth:`ScopeRegistry.evaluate` over a fresh
        in-memory registry built from the on-disk rows. The
        precedence rules (deny > max_risk > allowlist) live in
        exactly one place — the in-memory registry — and the disk
        layer doesn't risk silent drift by re-implementing them.
        """
        return self._rebuild_in_memory().evaluate(scope_keys, tool_name, tool_risk)

    def assert_allowed(
        self,
        scope_keys: list[tuple[ScopeAxis, str]],
        tool_name: str,
        tool_risk: RiskLevel,
    ) -> None:
        """Raise :class:`ScopeViolation` when the call is denied.

        Mirrors :meth:`ScopeRegistry.assert_allowed` exactly via the
        same delegation pattern.
        """
        self._rebuild_in_memory().assert_allowed(scope_keys, tool_name, tool_risk)

    # ------------------------------------------------------------------
    # Internal — rebuild an in-memory registry from disk
    # ------------------------------------------------------------------

    def _rebuild_in_memory(self) -> ScopeRegistry:
        """Re-build a fresh in-memory ScopeRegistry from the on-disk rows.

        Used by ``evaluate`` and ``assert_allowed``. Cost is O(n)
        per call; for realistic scope sets (tens of rules) this is
        negligible. Trade-off: keeps evaluation logic in exactly one
        place.
        """
        registry = ScopeRegistry()
        for scope in self.list_scopes():
            registry.register(scope)
        return registry


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _row_to_scope(row: tuple[object, ...], cursor: sqlite3.Cursor) -> PermissionScope:
    """Parse one ``scope_entries`` row back into a frozen
    :class:`PermissionScope`."""
    column_names = [d[0] for d in cursor.description]
    data = dict(zip(column_names, row, strict=False))
    raw_allow = json.loads(str(data["tool_allowlist"]))
    raw_deny = json.loads(str(data["tool_denylist"]))
    if not isinstance(raw_allow, list) or not isinstance(raw_deny, list):
        msg = "scope_entries row has malformed tool_allowlist/denylist"
        raise PermissionScopeStoreError(msg)
    return PermissionScope(
        axis=ScopeAxis(str(data["axis"])),
        identity=str(data["identity"]),
        tool_allowlist=frozenset(str(t) for t in raw_allow),
        tool_denylist=frozenset(str(t) for t in raw_deny),
        max_risk=RiskLevel(str(data["max_risk"])),
    )


__all__ = [
    "PermissionScopeStore",
    "PermissionScopeStoreError",
    "ScopeViolation",  # re-export for caller convenience
]
