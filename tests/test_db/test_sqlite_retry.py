"""Tests for SQLite retry-on-locked logic."""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from jarvis.db.sqlite_backend import SQLiteBackend


def _make_flaky_conn(real_conn: sqlite3.Connection, fail_count: int):
    """Return a mock Connection that raises 'database is locked' *fail_count* times."""
    mock = MagicMock(wraps=real_conn)
    call_counter = {"n": 0}

    def _flaky_execute(*args, **kwargs):
        call_counter["n"] += 1
        if call_counter["n"] <= fail_count:
            raise sqlite3.OperationalError("database is locked")
        return real_conn.execute(*args, **kwargs)

    mock.execute = MagicMock(side_effect=_flaky_execute)
    mock.executemany = real_conn.executemany
    mock.executescript = real_conn.executescript
    mock.commit = MagicMock()
    mock.close = real_conn.close
    mock.row_factory = real_conn.row_factory
    return mock, call_counter


class TestRetryOnLocked:
    """Retry logic for 'database is locked' errors."""

    @pytest.fixture()
    def backend(self, tmp_path):
        return SQLiteBackend(tmp_path / "test.db")

    def test_succeeds_after_transient_lock(self, backend):
        """Retry succeeds when lock clears."""
        mock_conn, counter = _make_flaky_conn(backend._conn, fail_count=2)
        backend._conn = mock_conn
        backend._execute_sync("SELECT 1")
        assert counter["n"] == 3

    def test_exhausted_retries_raises(self, backend):
        """Raises after max retries."""

        def always_locked(*a, **kw):
            raise sqlite3.OperationalError("database is locked")

        mock_conn = MagicMock()
        mock_conn.execute = MagicMock(side_effect=always_locked)
        backend._conn = mock_conn
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            backend._execute_sync("SELECT 1")

    def test_no_retry_on_other_errors(self, backend):
        """Non-locked errors raise immediately."""
        call_count = 0

        def syntax_error(*a, **kw):
            nonlocal call_count
            call_count += 1
            raise sqlite3.OperationalError("near 'SELCT': syntax error")

        mock_conn = MagicMock()
        mock_conn.execute = MagicMock(side_effect=syntax_error)
        backend._conn = mock_conn
        with pytest.raises(sqlite3.OperationalError, match="syntax error"):
            backend._execute_sync("SELCT 1")
        assert call_count == 1

    def test_custom_retry_params(self, tmp_path):
        """Constructor accepts custom retry parameters."""
        backend = SQLiteBackend(tmp_path / "custom.db", max_retries=2, retry_base_delay=0.01)
        assert backend._max_retries == 2
        assert backend._retry_base_delay == 0.01

    def test_fetchone_retries(self, backend):
        """fetchone also retries on locked."""
        mock_conn, counter = _make_flaky_conn(backend._conn, fail_count=1)
        backend._conn = mock_conn
        result = backend._fetchone_sync("SELECT 1 AS val")
        assert counter["n"] == 2
        assert result is not None

    def test_fetchall_retries(self, backend):
        """fetchall also retries on locked."""
        mock_conn, counter = _make_flaky_conn(backend._conn, fail_count=1)
        backend._conn = mock_conn
        result = backend._fetchall_sync("SELECT 1 AS val")
        assert counter["n"] == 2
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_async_execute_retries(self, backend):
        """Async wrapper inherits retry behavior."""
        mock_conn, counter = _make_flaky_conn(backend._conn, fail_count=1)
        backend._conn = mock_conn
        await backend.execute("SELECT 1")
        assert counter["n"] == 2
