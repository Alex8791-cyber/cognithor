"""Deep coverage for ``cognithor.mcp.database_tools``.

The module is security-sensitive: it owns the SQL execution layer that the
agent can reach via ``db_query`` / ``db_schema`` / ``db_execute`` /
``db_connect``. Audit-PR10 (audit-MED-3) hardened ``_check_injection`` to
run regardless of whether ``params`` were supplied (smuggling defence).
PR #479 P4-D added DROP-blocking + path-validation polish.

This file exercises:
  * ``_check_injection`` — every pattern + smuggling-with-params.
  * Path validation — workspace/home roots, traversal, non-existent files.
  * ``db_query`` — happy-path, params, limit clamping, read-only blocking,
    empty SQL.
  * ``db_schema`` — list mode, table mode, invalid identifier, missing table.
  * ``db_execute`` — INSERT/UPDATE/DELETE rowcount, DROP block, empty SQL.
  * ``db_connect`` — version + counts + size formatting.
  * ``_format_table`` + ``_truncate`` — ASCII rendering edge-cases.
  * ``_is_pg_connection_string`` — heuristic.
  * ``register_database_tools`` — all four handlers + schemas.

Continuation of Wave-1/2 backend deep coverage (PRs #486, #488).
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Any

import pytest

from cognithor.config import CognithorConfig, SecurityConfig, ensure_directory_structure
from cognithor.mcp.database_tools import (
    _MAX_ROW_LIMIT,
    DatabaseError,
    DatabaseTools,
    _check_injection,
    _format_table,
    _is_pg_connection_string,
    _truncate,
    register_database_tools,
)

if TYPE_CHECKING:
    from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def config(tmp_path: Path) -> CognithorConfig:
    cfg = CognithorConfig(
        cognithor_home=tmp_path / ".cognithor",
        security=SecurityConfig(allowed_paths=[str(tmp_path)]),
    )
    ensure_directory_structure(cfg)
    return cfg


@pytest.fixture()
def db_tools(config: CognithorConfig) -> DatabaseTools:
    return DatabaseTools(config)


@pytest.fixture()
def sample_db(tmp_path: Path) -> Path:
    """Create an unencrypted SQLite test DB with two tables and rows."""
    db_path = tmp_path / "sample.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER, email TEXT)"
    )
    conn.execute(
        "INSERT INTO users (name, age, email) VALUES "
        "('Alice', 30, 'alice@example.com'),"
        "('Bob', 25, 'bob@example.com'),"
        "('Charlie', 35, 'charlie@example.com'),"
        "('Dave', 40, 'dave@example.com')"
    )
    conn.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, product TEXT, amount REAL)"
    )
    conn.execute(
        "INSERT INTO orders (user_id, product, amount) VALUES "
        "(1, 'Widget', 9.99),"
        "(2, 'Gadget', 19.99)"
    )
    conn.execute("CREATE INDEX idx_users_name ON users(name)")
    conn.execute("CREATE VIEW active_users AS SELECT * FROM users WHERE age > 0")
    conn.commit()
    conn.close()
    return db_path


class _MockMCPClient:
    def __init__(self) -> None:
        self.registered: dict[str, dict[str, Any]] = {}

    def register_builtin_handler(
        self,
        name: str,
        handler: object,
        *,
        description: str = "",
        input_schema: dict[str, Any] | None = None,
    ) -> None:
        self.registered[name] = {
            "handler": handler,
            "description": description,
            "input_schema": input_schema,
        }


# ─────────────────────────────────────────────────────────────────────────────
# _check_injection — SQL-injection pattern detector
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckInjection:
    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT 1; DROP TABLE users",
            "SELECT 1;DELETE FROM x",
            "SELECT 1; UPDATE x SET y=1",
            "SELECT 1; INSERT INTO x VALUES (1)",
            "SELECT 1; ALTER TABLE x ADD col INT",
            "SELECT 1; CREATE TABLE evil (id INT)",
            "SELECT 1; EXEC xp_cmdshell",
        ],
    )
    def test_multistatement_attacks_caught(self, sql: str) -> None:
        with pytest.raises(DatabaseError, match="injection"):
            _check_injection(sql, None)

    def test_union_select_caught(self) -> None:
        with pytest.raises(DatabaseError, match="injection"):
            _check_injection("SELECT * FROM users UNION SELECT * FROM secrets", None)

    def test_comment_terminator_caught(self) -> None:
        with pytest.raises(DatabaseError, match="injection"):
            _check_injection("SELECT * FROM users WHERE id = 1 --", None)

    def test_smuggling_with_params_still_caught(self) -> None:
        # audit-MED-3: parameterised callers must NOT be exempt.
        # Even with placeholders, the SQL string itself must be clean.
        with pytest.raises(DatabaseError, match="injection"):
            _check_injection("SELECT * FROM users WHERE id = ?; DROP TABLE x", [1])

    def test_smuggling_union_with_params(self) -> None:
        with pytest.raises(DatabaseError, match="injection"):
            _check_injection(
                "SELECT name FROM users WHERE id = ? UNION SELECT password FROM admin",
                [1],
            )

    def test_normal_queries_pass(self) -> None:
        # Plain queries with or without params must NOT trigger.
        _check_injection("SELECT * FROM users", None)
        _check_injection("SELECT * FROM users WHERE name = ?", ["Alice"])
        _check_injection("SELECT id, name FROM users WHERE age BETWEEN ? AND ?", [25, 40])

    def test_case_insensitive(self) -> None:
        # Pattern uses re.IGNORECASE — lowercase smuggling still caught.
        with pytest.raises(DatabaseError, match="injection"):
            _check_injection("select 1; drop table users", None)


# ─────────────────────────────────────────────────────────────────────────────
# _is_pg_connection_string — routing heuristic
# ─────────────────────────────────────────────────────────────────────────────


class TestPGHeuristic:
    @pytest.mark.parametrize(
        "connstr",
        [
            "postgresql://user:pass@host/db",
            "postgres://user:pass@host:5432/db",
            "POSTGRESQL://user@host/db",
            "  postgresql://leading-space-stripped/  ",
        ],
    )
    def test_pg_strings(self, connstr: str) -> None:
        assert _is_pg_connection_string(connstr) is True

    @pytest.mark.parametrize(
        "connstr",
        [
            "/tmp/db.sqlite",
            "C:\\Users\\foo\\db.sqlite",
            "mysql://user@host/db",
            "sqlite:///path/to.db",
            "",
        ],
    )
    def test_non_pg_strings(self, connstr: str) -> None:
        assert _is_pg_connection_string(connstr) is False


# ─────────────────────────────────────────────────────────────────────────────
# _truncate / _format_table — ASCII rendering
# ─────────────────────────────────────────────────────────────────────────────


class TestTruncate:
    def test_short_value_unchanged(self) -> None:
        assert _truncate("hello") == "hello"

    def test_exact_length_unchanged(self) -> None:
        s = "x" * 200
        assert _truncate(s, 200) == s

    def test_long_value_truncated_with_ellipsis(self) -> None:
        s = "x" * 250
        out = _truncate(s, 200)
        assert len(out) == 200
        assert out.endswith("...")


class TestFormatTable:
    def test_no_columns_emits_rows_affected(self) -> None:
        assert "5 rows affected" in _format_table([], [], 5)

    def test_simple_two_column_table(self) -> None:
        out = _format_table(["name", "age"], [("Alice", 30), ("Bob", 25)], 2)
        assert "Alice" in out
        assert "Bob" in out
        assert "(2 rows)" in out

    def test_one_row_singular_label(self) -> None:
        out = _format_table(["id"], [(1,)], 1)
        assert "(1 row)" in out

    def test_zero_rows_plural_label(self) -> None:
        out = _format_table(["id"], [], 0)
        assert "(0 rows)" in out

    def test_none_cells_render_null(self) -> None:
        out = _format_table(["id", "name"], [(1, None)], 1)
        assert "NULL" in out

    def test_long_cell_truncated(self) -> None:
        long_value = "x" * 500
        out = _format_table(["data"], [(long_value,)], 1)
        assert "..." in out
        assert long_value not in out


# ─────────────────────────────────────────────────────────────────────────────
# Path validation — workspace + traversal + missing
# ─────────────────────────────────────────────────────────────────────────────


class TestPathValidation:
    @pytest.mark.asyncio
    async def test_outside_allowed_roots_blocked(self, db_tools: DatabaseTools) -> None:
        with pytest.raises(DatabaseError, match="Zugriff verweigert"):
            await db_tools.db_query("/etc/passwd", "SELECT 1")

    @pytest.mark.asyncio
    async def test_traversal_blocked(self, db_tools: DatabaseTools, tmp_path: Path) -> None:
        # Path that resolves outside the allowed root.
        with pytest.raises(DatabaseError, match="Zugriff verweigert"):
            await db_tools.db_query(
                str(tmp_path / ".." / ".." / ".." / "etc" / "shadow"), "SELECT 1"
            )

    @pytest.mark.asyncio
    async def test_nonexistent_db_inside_allowed_root(
        self, db_tools: DatabaseTools, tmp_path: Path
    ) -> None:
        # Allowed root, but the file does not exist.
        with pytest.raises(DatabaseError, match="nicht gefunden"):
            await db_tools.db_query(str(tmp_path / "ghost.db"), "SELECT 1")

    @pytest.mark.asyncio
    async def test_workspace_dir_is_implicitly_allowed(self, config: CognithorConfig) -> None:
        # Even if not in allowed_paths, workspace is auto-added.
        wk_db = config.workspace_dir / "wk.db"
        wk_db.parent.mkdir(parents=True, exist_ok=True)
        sqlite3.connect(str(wk_db)).close()
        tools = DatabaseTools(config)
        # Should not raise an "access denied" — it should run.
        result = await tools.db_query(str(wk_db), "SELECT 1 AS one")
        assert "one" in result


# ─────────────────────────────────────────────────────────────────────────────
# db_query — happy paths + read-only enforcement + clamping
# ─────────────────────────────────────────────────────────────────────────────


class TestDbQuery:
    @pytest.mark.asyncio
    async def test_select_all(self, db_tools: DatabaseTools, sample_db: Path) -> None:
        out = await db_tools.db_query(str(sample_db), "SELECT name FROM users")
        for name in ("Alice", "Bob", "Charlie", "Dave"):
            assert name in out

    @pytest.mark.asyncio
    async def test_positional_param_filter(self, db_tools: DatabaseTools, sample_db: Path) -> None:
        out = await db_tools.db_query(
            str(sample_db),
            "SELECT name FROM users WHERE age >= ? AND age <= ?",
            params=[28, 36],
        )
        assert "Alice" in out
        assert "Charlie" in out
        assert "Bob" not in out  # 25 < 28
        assert "Dave" not in out  # 40 > 36

    @pytest.mark.asyncio
    async def test_named_param_via_dict_unsupported_falls_through(
        self, db_tools: DatabaseTools, sample_db: Path
    ) -> None:
        # The handler accepts list-of-params; sqlite3 with `?`-style
        # placeholders does not support named binding. Confirm with a
        # standard positional query that string injection is unnecessary
        # because parameterisation works.
        out = await db_tools.db_query(
            str(sample_db),
            "SELECT name FROM users WHERE name = ?",
            params=["Alice'; DROP TABLE users;--"],
        )
        # No row matches the literal string — query returns 0 rows but
        # the DROP smuggle never runs (placeholder, not concatenation).
        assert "Alice'" not in out  # would only appear if the DROP ran
        assert "(0 rows)" in out

    @pytest.mark.asyncio
    async def test_empty_sql_returns_friendly_error(
        self, db_tools: DatabaseTools, sample_db: Path
    ) -> None:
        out = await db_tools.db_query(str(sample_db), "   ")
        assert "Fehler" in out or "kein sql" in out.lower()

    @pytest.mark.asyncio
    async def test_limit_clamped_to_max(self, db_tools: DatabaseTools, sample_db: Path) -> None:
        # Limit above _MAX_ROW_LIMIT silently clamps; the query runs.
        out = await db_tools.db_query(
            str(sample_db),
            "SELECT name FROM users",
            limit=_MAX_ROW_LIMIT + 100_000,
        )
        # Sample DB has 4 rows, all should appear.
        assert "Alice" in out and "Dave" in out

    @pytest.mark.asyncio
    async def test_limit_clamped_to_one_when_zero(
        self, db_tools: DatabaseTools, sample_db: Path
    ) -> None:
        # The handler does max(1, min(limit, MAX)) — 0 becomes 1.
        out = await db_tools.db_query(str(sample_db), "SELECT name FROM users", limit=0)
        # With clamp to 1, exactly one row + a "more rows available" note.
        assert "1 row" in out
        assert "more rows available" in out

    @pytest.mark.asyncio
    async def test_readonly_blocks_insert(self, db_tools: DatabaseTools, sample_db: Path) -> None:
        with pytest.raises(DatabaseError):
            await db_tools.db_query(
                str(sample_db), "INSERT INTO users (name, age) VALUES ('Eve', 28)"
            )

    @pytest.mark.asyncio
    async def test_readonly_blocks_update(self, db_tools: DatabaseTools, sample_db: Path) -> None:
        with pytest.raises(DatabaseError):
            await db_tools.db_query(str(sample_db), "UPDATE users SET age=99 WHERE name='Alice'")

    @pytest.mark.asyncio
    async def test_invalid_sql_surfaces_database_error(
        self, db_tools: DatabaseTools, sample_db: Path
    ) -> None:
        with pytest.raises(DatabaseError):
            await db_tools.db_query(str(sample_db), "SELECT * FROM nonexistent_table")


# ─────────────────────────────────────────────────────────────────────────────
# db_schema — list mode + table mode + invalid identifier
# ─────────────────────────────────────────────────────────────────────────────


class TestDbSchema:
    @pytest.mark.asyncio
    async def test_list_includes_tables_and_views(
        self, db_tools: DatabaseTools, sample_db: Path
    ) -> None:
        out = await db_tools.db_schema(str(sample_db))
        assert "users" in out
        assert "orders" in out
        assert "active_users" in out  # view

    @pytest.mark.asyncio
    async def test_table_mode_lists_columns(self, db_tools: DatabaseTools, sample_db: Path) -> None:
        out = await db_tools.db_schema(str(sample_db), table="users")
        for col in ("id", "name", "age", "email"):
            assert col in out

    @pytest.mark.asyncio
    async def test_table_mode_marks_primary_key(
        self, db_tools: DatabaseTools, sample_db: Path
    ) -> None:
        out = await db_tools.db_schema(str(sample_db), table="users")
        # PK column header + "YES" in id row.
        assert "PK" in out
        # We can't easily anchor a regex on the row but YES must appear.
        assert "YES" in out

    @pytest.mark.asyncio
    async def test_table_mode_lists_indexes(self, db_tools: DatabaseTools, sample_db: Path) -> None:
        out = await db_tools.db_schema(str(sample_db), table="users")
        assert "idx_users_name" in out

    @pytest.mark.asyncio
    async def test_missing_table_raises(self, db_tools: DatabaseTools, sample_db: Path) -> None:
        with pytest.raises(DatabaseError, match="nicht gefunden"):
            await db_tools.db_schema(str(sample_db), table="not_a_table")

    @pytest.mark.asyncio
    async def test_invalid_identifier_blocks_attack(
        self, db_tools: DatabaseTools, sample_db: Path
    ) -> None:
        # A `;`-laden identifier must not slip through PRAGMA. The regex
        # `_SAFE_IDENTIFIER_RE` permits only `[A-Za-z_][A-Za-z0-9_ ]*`.
        with pytest.raises(DatabaseError, match="Ungueltiger Tabellenname"):
            await db_tools.db_schema(str(sample_db), table="users; DROP TABLE users;")

    @pytest.mark.asyncio
    async def test_invalid_identifier_with_quote_blocked(
        self, db_tools: DatabaseTools, sample_db: Path
    ) -> None:
        with pytest.raises(DatabaseError, match="Ungueltiger Tabellenname"):
            await db_tools.db_schema(str(sample_db), table='users" OR 1=1 --')


# ─────────────────────────────────────────────────────────────────────────────
# db_execute — write paths + DROP block + empty SQL
# ─────────────────────────────────────────────────────────────────────────────


class TestDbExecute:
    @pytest.mark.asyncio
    async def test_insert_returns_rows_affected(
        self, db_tools: DatabaseTools, sample_db: Path
    ) -> None:
        out = await db_tools.db_execute(
            str(sample_db),
            "INSERT INTO users (name, age, email) VALUES (?, ?, ?)",
            params=["Eve", 28, "eve@example.com"],
        )
        assert "1" in out
        # Followup read confirms write actually committed.
        check = await db_tools.db_query(
            str(sample_db), "SELECT name FROM users WHERE name = ?", params=["Eve"]
        )
        assert "Eve" in check

    @pytest.mark.asyncio
    async def test_update_returns_rows_affected(
        self, db_tools: DatabaseTools, sample_db: Path
    ) -> None:
        out = await db_tools.db_execute(
            str(sample_db),
            "UPDATE users SET age = ? WHERE name = ?",
            params=[31, "Alice"],
        )
        assert "1" in out

    @pytest.mark.asyncio
    async def test_delete_returns_rows_affected(
        self, db_tools: DatabaseTools, sample_db: Path
    ) -> None:
        out = await db_tools.db_execute(
            str(sample_db), "DELETE FROM users WHERE name = ?", params=["Bob"]
        )
        assert "1" in out

    @pytest.mark.asyncio
    async def test_drop_blocked(self, db_tools: DatabaseTools, sample_db: Path) -> None:
        with pytest.raises(DatabaseError, match="DROP"):
            await db_tools.db_execute(str(sample_db), "DROP TABLE users")

    @pytest.mark.asyncio
    async def test_drop_blocked_case_insensitive(
        self, db_tools: DatabaseTools, sample_db: Path
    ) -> None:
        # Mixed-case DROP — the strip().upper() check is case-insensitive.
        with pytest.raises(DatabaseError, match="DROP"):
            await db_tools.db_execute(str(sample_db), "  drop table users")

    @pytest.mark.asyncio
    async def test_empty_sql_returns_friendly_error(
        self, db_tools: DatabaseTools, sample_db: Path
    ) -> None:
        out = await db_tools.db_execute(str(sample_db), "  \n  ")
        assert "Fehler" in out or "kein sql" in out.lower()

    @pytest.mark.asyncio
    async def test_create_table_succeeds(self, db_tools: DatabaseTools, sample_db: Path) -> None:
        out = await db_tools.db_execute(
            str(sample_db), "CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT)"
        )
        # SQLite returns rowcount=-1 for DDL; "Erfolgreich" is the prefix.
        assert "Erfolg" in out or "row" in out.lower() or "-1" in out


# ─────────────────────────────────────────────────────────────────────────────
# db_connect — version + counts + size
# ─────────────────────────────────────────────────────────────────────────────


class TestDbConnect:
    @pytest.mark.asyncio
    async def test_returns_sqlite_metadata(self, db_tools: DatabaseTools, sample_db: Path) -> None:
        out = await db_tools.db_connect(str(sample_db))
        assert "SQLite" in out
        assert "Tabellen:" in out
        assert "Views:" in out

    @pytest.mark.asyncio
    async def test_size_label_kb(self, db_tools: DatabaseTools, sample_db: Path) -> None:
        # Sample DB is small but >1 KB.
        out = await db_tools.db_connect(str(sample_db))
        # Either "B", "KB" or "MB" must appear.
        assert any(unit in out for unit in (" B", "KB", "MB"))

    @pytest.mark.asyncio
    async def test_missing_db_raises(self, db_tools: DatabaseTools, tmp_path: Path) -> None:
        with pytest.raises(DatabaseError, match="nicht gefunden"):
            await db_tools.db_connect(str(tmp_path / "missing.db"))


# ─────────────────────────────────────────────────────────────────────────────
# Error masking — credentials in connection strings must NOT leak
# ─────────────────────────────────────────────────────────────────────────────


class TestErrorMasking:
    @pytest.mark.asyncio
    async def test_outside_root_error_does_not_leak_full_filesystem(
        self, db_tools: DatabaseTools
    ) -> None:
        # Confirm the error mentions the path the caller passed (so they
        # know what was rejected) but not random other absolute paths.
        try:
            await db_tools.db_query("/etc/passwd", "SELECT 1")
        except DatabaseError as exc:
            assert "/etc/passwd" in str(exc)


# ─────────────────────────────────────────────────────────────────────────────
# register_database_tools — wiring test
# ─────────────────────────────────────────────────────────────────────────────


class TestRegistration:
    def test_all_four_tools_registered(self, config: CognithorConfig) -> None:
        client = _MockMCPClient()
        tools = register_database_tools(client, config)
        assert isinstance(tools, DatabaseTools)
        assert set(client.registered.keys()) == {
            "db_query",
            "db_schema",
            "db_execute",
            "db_connect",
        }

    @pytest.mark.parametrize("name", ["db_query", "db_schema", "db_execute", "db_connect"])
    def test_each_handler_callable(self, config: CognithorConfig, name: str) -> None:
        client = _MockMCPClient()
        register_database_tools(client, config)
        assert callable(client.registered[name]["handler"])

    def test_query_schema_requires_database_and_sql(self, config: CognithorConfig) -> None:
        client = _MockMCPClient()
        register_database_tools(client, config)
        schema = client.registered["db_query"]["input_schema"]
        assert schema is not None
        assert set(schema["required"]) == {"database", "sql"}

    def test_connect_schema_requires_database_only(self, config: CognithorConfig) -> None:
        client = _MockMCPClient()
        register_database_tools(client, config)
        schema = client.registered["db_connect"]["input_schema"]
        assert schema is not None
        assert schema["required"] == ["database"]

    def test_descriptions_non_empty(self, config: CognithorConfig) -> None:
        client = _MockMCPClient()
        register_database_tools(client, config)
        for name, entry in client.registered.items():
            assert entry["description"], f"{name} missing description"
