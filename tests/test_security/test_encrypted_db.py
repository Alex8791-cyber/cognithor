"""Tests for encrypted database wrapper."""

from __future__ import annotations

import pytest
from jarvis.security.encrypted_db import encrypted_connect, is_encryption_available


def test_fallback_to_sqlite3(tmp_path):
    """Without SQLCipher, should fall back to standard sqlite3."""
    db_path = str(tmp_path / "test.db")
    conn = encrypted_connect(db_path, key="test_key")
    conn.execute("CREATE TABLE test (id INTEGER)")
    conn.execute("INSERT INTO test VALUES (1)")
    conn.commit()
    row = conn.execute("SELECT id FROM test").fetchone()
    assert row[0] == 1
    conn.close()


def test_reopen_database(tmp_path):
    """Database should be reopenable."""
    db_path = str(tmp_path / "test.db")
    conn1 = encrypted_connect(db_path, key="test_key")
    conn1.execute("CREATE TABLE test (id INTEGER)")
    conn1.execute("INSERT INTO test VALUES (42)")
    conn1.commit()
    conn1.close()

    conn2 = encrypted_connect(db_path, key="test_key")
    row = conn2.execute("SELECT id FROM test").fetchone()
    assert row[0] == 42
    conn2.close()


def test_empty_key_uses_sqlite3(tmp_path):
    """Empty key should use standard sqlite3."""
    db_path = str(tmp_path / "test.db")
    conn = encrypted_connect(db_path, key="")
    conn.execute("CREATE TABLE test (id INTEGER)")
    conn.commit()
    conn.close()


def test_is_encryption_available():
    """Should return bool without crashing."""
    result = is_encryption_available()
    assert isinstance(result, bool)
