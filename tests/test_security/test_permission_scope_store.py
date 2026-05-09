"""Tests for the SQLite-backed PermissionScopeStore (TRUST-5 disk persistence)."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pytest

from cognithor.models import RiskLevel
from cognithor.security.permission_scope import (
    PermissionScope,
    ScopeAxis,
    ScopeRegistry,
    ScopeViolation,
)
from cognithor.security.permission_scope_store import (
    PermissionScopeStore,
    PermissionScopeStoreError,
)


def _scope(
    *,
    axis: ScopeAxis = ScopeAxis.CHANNEL,
    identity: str = "telegram",
    allow: tuple[str, ...] = (),
    deny: tuple[str, ...] = (),
    max_risk: RiskLevel = RiskLevel.RED,
) -> PermissionScope:
    return PermissionScope(
        axis=axis,
        identity=identity,
        tool_allowlist=frozenset(allow),
        tool_denylist=frozenset(deny),
        max_risk=max_risk,
    )


# ---------------------------------------------------------------------------
# Lifecycle + schema
# ---------------------------------------------------------------------------


class TestStoreLifecycle:
    def test_creates_schema_in_memory(self) -> None:
        with PermissionScopeStore(":memory:") as store:
            assert store.schema_version == 1
            assert len(store) == 0

    def test_context_manager_persists_to_disk(self, tmp_path) -> None:
        db = tmp_path / "x.sqlite"
        with PermissionScopeStore(db) as store:
            store.register(_scope(identity="telegram", allow=("web_fetch",)))
        with PermissionScopeStore(db) as store:
            assert len(store) == 1
            scope = store.get(ScopeAxis.CHANNEL, "telegram")
            assert scope is not None
            assert scope.tool_allowlist == frozenset({"web_fetch"})

    def test_close_idempotent(self) -> None:
        store = PermissionScopeStore(":memory:")
        store.close()
        store.close()


# ---------------------------------------------------------------------------
# Schema-version guard
# ---------------------------------------------------------------------------


class TestSchemaVersionGuard:
    def test_rejects_newer_schema(self, tmp_path) -> None:
        db = tmp_path / "future.sqlite"
        PermissionScopeStore(db).close()
        conn = sqlite3.connect(db)
        try:
            conn.execute(
                "INSERT INTO _schema_meta (version, applied_at) VALUES (?, ?)",
                (999, datetime.now(UTC).isoformat()),
            )
            conn.commit()
        finally:
            conn.close()
        with pytest.raises(PermissionScopeStoreError, match="version 999"):
            PermissionScopeStore(db)

    def test_accepts_equal_version(self, tmp_path) -> None:
        db = tmp_path / "x.sqlite"
        PermissionScopeStore(db).close()
        PermissionScopeStore(db).close()


# ---------------------------------------------------------------------------
# Upsert (register replaces) + remove
# ---------------------------------------------------------------------------


class TestRegisterAndRemove:
    def test_register_inserts_row(self) -> None:
        with PermissionScopeStore(":memory:") as store:
            store.register(_scope(identity="telegram"))
            assert len(store) == 1

    def test_register_upserts_on_same_key(self) -> None:
        """``ScopeRegistry.register`` replaces silently. The disk
        store must mirror that — re-registering the same
        ``(axis, identity)`` overwrites, doesn't insert a duplicate."""
        with PermissionScopeStore(":memory:") as store:
            store.register(_scope(identity="telegram", allow=("web_fetch",)))
            store.register(
                _scope(
                    identity="telegram",
                    allow=("read_kanban",),
                    max_risk=RiskLevel.YELLOW,
                )
            )
            assert len(store) == 1
            scope = store.get(ScopeAxis.CHANNEL, "telegram")
            assert scope is not None
            assert scope.tool_allowlist == frozenset({"read_kanban"})
            assert scope.max_risk == RiskLevel.YELLOW

    def test_register_distinct_keys_appends(self) -> None:
        with PermissionScopeStore(":memory:") as store:
            store.register(_scope(axis=ScopeAxis.CHANNEL, identity="telegram"))
            store.register(_scope(axis=ScopeAxis.CHANNEL, identity="slack"))
            store.register(_scope(axis=ScopeAxis.USER, identity="alex@cognithor.ai"))
            assert len(store) == 3

    def test_remove_returns_true_when_present(self) -> None:
        with PermissionScopeStore(":memory:") as store:
            store.register(_scope(identity="telegram"))
            assert store.remove(ScopeAxis.CHANNEL, "telegram") is True
            assert len(store) == 0

    def test_remove_returns_false_when_absent(self) -> None:
        with PermissionScopeStore(":memory:") as store:
            assert store.remove(ScopeAxis.CHANNEL, "ghost") is False

    def test_clear_drops_everything(self) -> None:
        with PermissionScopeStore(":memory:") as store:
            store.register(_scope(identity="t1"))
            store.register(_scope(identity="t2"))
            store.register(_scope(identity="t3"))
            store.clear()
            assert len(store) == 0


# ---------------------------------------------------------------------------
# Round-trip — every field
# ---------------------------------------------------------------------------


class TestFieldRoundTrip:
    def test_full_round_trip(self) -> None:
        original = _scope(
            axis=ScopeAxis.WORKFLOW,
            identity="morning_brief",
            allow=("web_fetch", "read_calendar", "list_kanban"),
            deny=("execute_shell", "delete_memory"),
            max_risk=RiskLevel.YELLOW,
        )
        with PermissionScopeStore(":memory:") as store:
            store.register(original)
            recovered = store.get(ScopeAxis.WORKFLOW, "morning_brief")
            assert recovered == original

    def test_empty_allow_and_deny_lists_round_trip(self) -> None:
        """The default-empty path matters: a scope with empty
        allowlist must remain "all tools allowed (subject to denylist
        + max_risk)" semantics on read-back."""
        with PermissionScopeStore(":memory:") as store:
            store.register(_scope(identity="telegram"))
            scope = store.get(ScopeAxis.CHANNEL, "telegram")
            assert scope is not None
            assert scope.tool_allowlist == frozenset()
            assert scope.tool_denylist == frozenset()


# ---------------------------------------------------------------------------
# Read API parity
# ---------------------------------------------------------------------------


class TestReadParity:
    def test_get_present_and_absent(self) -> None:
        with PermissionScopeStore(":memory:") as store:
            store.register(_scope(identity="telegram"))
            assert store.get(ScopeAxis.CHANNEL, "telegram") is not None
            assert store.get(ScopeAxis.CHANNEL, "ghost") is None

    def test_list_scopes_sorted(self) -> None:
        with PermissionScopeStore(":memory:") as store:
            store.register(_scope(axis=ScopeAxis.USER, identity="zeta"))
            store.register(_scope(axis=ScopeAxis.CHANNEL, identity="slack"))
            store.register(_scope(axis=ScopeAxis.CHANNEL, identity="telegram"))
            store.register(_scope(axis=ScopeAxis.USER, identity="alpha"))
            keys = [(s.axis.value, s.identity) for s in store.list_scopes()]
            assert keys == [
                ("channel", "slack"),
                ("channel", "telegram"),
                ("user", "alpha"),
                ("user", "zeta"),
            ]


# ---------------------------------------------------------------------------
# Evaluate / assert_allowed — must match in-memory verdict
# ---------------------------------------------------------------------------


class TestEvaluation:
    def test_denylist_blocks_call(self) -> None:
        """Denylisted tool is rejected even at GREEN risk."""
        with PermissionScopeStore(":memory:") as store:
            store.register(_scope(identity="telegram", deny=("execute_shell",)))
            verdict = store.evaluate(
                [(ScopeAxis.CHANNEL, "telegram")],
                "execute_shell",
                RiskLevel.GREEN,
            )
            assert verdict.allowed is False
            assert "denylist" in verdict.reasons[0]

    def test_max_risk_ceiling(self) -> None:
        with PermissionScopeStore(":memory:") as store:
            store.register(_scope(identity="telegram", max_risk=RiskLevel.YELLOW))
            v_under = store.evaluate(
                [(ScopeAxis.CHANNEL, "telegram")],
                "web_fetch",
                RiskLevel.YELLOW,
            )
            v_over = store.evaluate(
                [(ScopeAxis.CHANNEL, "telegram")],
                "execute_shell",
                RiskLevel.RED,
            )
            assert v_under.allowed is True
            assert v_over.allowed is False
            assert "exceeds max_risk" in v_over.reasons[0]

    def test_allowlist_blocks_unlisted_tool(self) -> None:
        with PermissionScopeStore(":memory:") as store:
            store.register(_scope(identity="telegram", allow=("web_fetch",)))
            v_listed = store.evaluate(
                [(ScopeAxis.CHANNEL, "telegram")],
                "web_fetch",
                RiskLevel.GREEN,
            )
            v_unlisted = store.evaluate(
                [(ScopeAxis.CHANNEL, "telegram")],
                "read_kanban",
                RiskLevel.GREEN,
            )
            assert v_listed.allowed is True
            assert v_unlisted.allowed is False
            assert "not in allowlist" in v_unlisted.reasons[0]

    def test_no_matching_scope_allows(self) -> None:
        """No matching scope → allow (the gatekeeper applies the
        global default in that case)."""
        with PermissionScopeStore(":memory:") as store:
            store.register(_scope(identity="telegram"))
            verdict = store.evaluate(
                [(ScopeAxis.CHANNEL, "ghost")],
                "web_fetch",
                RiskLevel.GREEN,
            )
            assert verdict.allowed is True

    def test_evaluate_matches_in_memory_registry_for_same_rules(self) -> None:
        """Trace-UI consumers must NOT see different verdicts based
        on which registry the call hit. Pin verdict equality across
        identical rule sets."""
        rules = [
            _scope(identity="telegram", allow=("web_fetch", "read_kanban")),
            _scope(
                axis=ScopeAxis.USER,
                identity="alex@cognithor.ai",
                max_risk=RiskLevel.ORANGE,
            ),
        ]
        memory_registry = ScopeRegistry()
        for r in rules:
            memory_registry.register(r)

        with PermissionScopeStore(":memory:") as store:
            for r in rules:
                store.register(r)

            scope_keys = [
                (ScopeAxis.CHANNEL, "telegram"),
                (ScopeAxis.USER, "alex@cognithor.ai"),
            ]
            for tool, risk in (
                ("web_fetch", RiskLevel.GREEN),
                ("execute_shell", RiskLevel.RED),
                ("read_kanban", RiskLevel.YELLOW),
            ):
                m = memory_registry.evaluate(scope_keys, tool, risk)
                d = store.evaluate(scope_keys, tool, risk)
                assert m.allowed == d.allowed
                assert m.reasons == d.reasons

    def test_assert_allowed_raises_on_deny(self) -> None:
        with PermissionScopeStore(":memory:") as store:
            store.register(_scope(identity="telegram", deny=("execute_shell",)))
            with pytest.raises(ScopeViolation, match="execute_shell"):
                store.assert_allowed(
                    [(ScopeAxis.CHANNEL, "telegram")],
                    "execute_shell",
                    RiskLevel.GREEN,
                )

    def test_assert_allowed_returns_silently_on_allow(self) -> None:
        with PermissionScopeStore(":memory:") as store:
            store.register(_scope(identity="telegram"))
            store.assert_allowed(
                [(ScopeAxis.CHANNEL, "telegram")],
                "web_fetch",
                RiskLevel.GREEN,
            )  # must not raise


# ---------------------------------------------------------------------------
# File persistence + concurrency
# ---------------------------------------------------------------------------


class TestFilePersistence:
    def test_writes_visible_after_reopen(self, tmp_path) -> None:
        db = tmp_path / "x.sqlite"
        with PermissionScopeStore(db) as store:
            store.register(_scope(identity="telegram"))
            store.register(_scope(identity="slack"))
        with PermissionScopeStore(db) as store:
            assert len(store) == 2
            ids = {s.identity for s in store.list_scopes()}
            assert ids == {"telegram", "slack"}

    def test_upsert_persisted_across_processes(self, tmp_path) -> None:
        """Two processes hitting the same key must both end up with
        the latest value — the disk-layer's primary contract."""
        db = tmp_path / "x.sqlite"
        with PermissionScopeStore(db) as a:
            a.register(_scope(identity="telegram", allow=("web_fetch",)))
        with PermissionScopeStore(db) as b:
            b.register(
                _scope(
                    identity="telegram",
                    allow=("read_kanban",),
                    max_risk=RiskLevel.YELLOW,
                )
            )
        with PermissionScopeStore(db) as reader:
            assert len(reader) == 1
            scope = reader.get(ScopeAxis.CHANNEL, "telegram")
            assert scope is not None
            assert scope.tool_allowlist == frozenset({"read_kanban"})
            assert scope.max_risk == RiskLevel.YELLOW


# ---------------------------------------------------------------------------
# Indices
# ---------------------------------------------------------------------------


class TestIndices:
    def test_axis_index_present(self) -> None:
        """Pin the secondary index — list_scopes() filters / sorts on
        ``axis`` and would table-scan if the index disappears."""
        with PermissionScopeStore(":memory:") as store:
            cursor = store._conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='index' AND tbl_name='scope_entries' "
                "AND name = 'idx_scope_entries_axis'"
            )
            assert cursor.fetchone() is not None

    def test_composite_pk_present(self) -> None:
        """The (axis, identity) composite primary key is the
        idempotency contract — without it, ``register`` upsert
        becomes plain insert and duplicates accumulate."""
        with PermissionScopeStore(":memory:") as store:
            cursor = store._conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='scope_entries'"
            )
            row = cursor.fetchone()
            assert row is not None
            sql = str(row[0])
            assert "PRIMARY KEY (axis, identity)" in sql
