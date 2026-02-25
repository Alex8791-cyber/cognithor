"""Tests fuer den SQLiteBackend.

Smoke-Tests: Verbindung, execute, fetchone, fetchall, commit, close,
placeholder, backend_type.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from jarvis.db.sqlite_backend import SQLiteBackend

if TYPE_CHECKING:
    from pathlib import Path


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def backend(tmp_path: Path) -> SQLiteBackend:
    """Erstellt einen SQLiteBackend mit temporaerer Datenbank."""
    db = SQLiteBackend(tmp_path / "test.db")
    return db


# ============================================================================
# Tests
# ============================================================================


class TestSQLiteBackendProperties:
    def test_placeholder_returns_question_mark(self, backend: SQLiteBackend) -> None:
        assert backend.placeholder == "?"

    def test_backend_type_returns_sqlite(self, backend: SQLiteBackend) -> None:
        assert backend.backend_type == "sqlite"

    def test_conn_property(self, backend: SQLiteBackend) -> None:
        assert backend.conn is not None


class TestSQLiteBackendOperations:
    @pytest.mark.asyncio
    async def test_execute_create_table(self, backend: SQLiteBackend) -> None:
        await backend.execute(
            "CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, name TEXT)"
        )
        # Table should exist -- inserting should not raise
        await backend.execute("INSERT INTO test (id, name) VALUES (?, ?)", (1, "alice"))

    @pytest.mark.asyncio
    async def test_fetchone(self, backend: SQLiteBackend) -> None:
        await backend.execute(
            "CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, name TEXT)"
        )
        await backend.execute("INSERT INTO test (id, name) VALUES (?, ?)", (1, "bob"))

        row = await backend.fetchone("SELECT * FROM test WHERE id = ?", (1,))
        assert row is not None
        assert row["id"] == 1
        assert row["name"] == "bob"

    @pytest.mark.asyncio
    async def test_fetchone_returns_none_for_missing(self, backend: SQLiteBackend) -> None:
        await backend.execute(
            "CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, name TEXT)"
        )
        row = await backend.fetchone("SELECT * FROM test WHERE id = ?", (999,))
        assert row is None

    @pytest.mark.asyncio
    async def test_fetchall(self, backend: SQLiteBackend) -> None:
        await backend.execute(
            "CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, name TEXT)"
        )
        await backend.execute("INSERT INTO test (id, name) VALUES (?, ?)", (1, "a"))
        await backend.execute("INSERT INTO test (id, name) VALUES (?, ?)", (2, "b"))
        await backend.execute("INSERT INTO test (id, name) VALUES (?, ?)", (3, "c"))

        rows = await backend.fetchall("SELECT * FROM test ORDER BY id")
        assert len(rows) == 3
        assert rows[0]["name"] == "a"
        assert rows[2]["name"] == "c"

    @pytest.mark.asyncio
    async def test_fetchall_empty(self, backend: SQLiteBackend) -> None:
        await backend.execute(
            "CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, name TEXT)"
        )
        rows = await backend.fetchall("SELECT * FROM test")
        assert rows == []

    @pytest.mark.asyncio
    async def test_commit(self, backend: SQLiteBackend) -> None:
        await backend.execute(
            "CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, name TEXT)"
        )
        await backend.execute("INSERT INTO test (id, name) VALUES (?, ?)", (1, "x"))
        await backend.commit()  # Should not raise

    @pytest.mark.asyncio
    async def test_close(self, backend: SQLiteBackend) -> None:
        await backend.close()
        # After close, _conn should be None
        assert backend._conn is None

    @pytest.mark.asyncio
    async def test_close_idempotent(self, backend: SQLiteBackend) -> None:
        await backend.close()
        await backend.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_executescript(self, backend: SQLiteBackend) -> None:
        script = """
        CREATE TABLE IF NOT EXISTS t1 (id INTEGER PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS t2 (id INTEGER PRIMARY KEY);
        """
        await backend.executescript(script)
        # Both tables should exist
        await backend.execute("INSERT INTO t1 (id) VALUES (?)", (1,))
        await backend.execute("INSERT INTO t2 (id) VALUES (?)", (1,))

    @pytest.mark.asyncio
    async def test_executemany(self, backend: SQLiteBackend) -> None:
        await backend.execute(
            "CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, name TEXT)"
        )
        await backend.executemany(
            "INSERT INTO test (id, name) VALUES (?, ?)",
            [(1, "a"), (2, "b"), (3, "c")],
        )
        rows = await backend.fetchall("SELECT * FROM test ORDER BY id")
        assert len(rows) == 3
