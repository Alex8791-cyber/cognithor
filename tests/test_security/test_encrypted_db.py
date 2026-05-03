"""Tests for encrypted database wrapper."""

from __future__ import annotations

import pytest

from cognithor.security import encrypted_db
from cognithor.security.encrypted_db import encrypted_connect, is_encryption_available


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


# ---------------------------------------------------------------------------
# SEC-HIGH-4: fail-closed when encryption requested but unavailable
# ---------------------------------------------------------------------------


@pytest.fixture
def reset_caches(monkeypatch):
    """Reset module-level caches so each test sees a clean state."""
    monkeypatch.setattr(encrypted_db, "_encryption_enabled_cache", None)
    monkeypatch.setattr(encrypted_db, "_allow_plaintext_fallback_cache", None)
    yield


class TestFailClosedSqlcipher:
    """SEC-HIGH-4 (autonomous security audit, 2026-05-04): when the
    operator enables encryption in config but SQLCipher / a key is
    unavailable, ``encrypted_connect`` MUST raise rather than silently
    storing plaintext on disk.
    """

    def test_raises_when_encryption_enabled_no_key_no_sqlcipher(
        self, tmp_path, monkeypatch, reset_caches
    ):
        """Default behaviour: refuse the plaintext fallback."""
        monkeypatch.setattr(encrypted_db, "_check_encryption_enabled", lambda: True)
        monkeypatch.setattr(encrypted_db, "_check_allow_plaintext_fallback", lambda: False)
        # Force "no key" + "no sqlcipher" — covers the worst case.
        monkeypatch.setattr(encrypted_db, "_get_db_key", lambda: "")
        monkeypatch.setattr(encrypted_db, "_sqlcipher_available", False)

        db_path = str(tmp_path / "test.db")
        with pytest.raises(RuntimeError, match="Refusing to open"):
            encrypted_connect(db_path)

    def test_allows_plaintext_when_explicit_opt_in(self, tmp_path, monkeypatch, reset_caches):
        """``allow_plaintext_fallback=True`` reactivates the legacy
        behaviour for operators who knowingly accept it.
        """
        monkeypatch.setattr(encrypted_db, "_check_encryption_enabled", lambda: True)
        monkeypatch.setattr(encrypted_db, "_check_allow_plaintext_fallback", lambda: True)
        monkeypatch.setattr(encrypted_db, "_get_db_key", lambda: "")
        monkeypatch.setattr(encrypted_db, "_sqlcipher_available", False)

        db_path = str(tmp_path / "test.db")
        conn = encrypted_connect(db_path)
        # Plaintext sqlite3 connection — write something and confirm.
        conn.execute("CREATE TABLE x (id INTEGER)")
        conn.execute("INSERT INTO x VALUES (1)")
        conn.commit()
        assert conn.execute("SELECT id FROM x").fetchone()[0] == 1
        conn.close()

    def test_no_raise_when_encryption_disabled_in_config(self, tmp_path, monkeypatch, reset_caches):
        """Existing path: encryption_enabled=false in config still
        permits plaintext sqlite3 (the early-out at the top of
        encrypted_connect short-circuits before our guard).
        """
        monkeypatch.setattr(encrypted_db, "_check_encryption_enabled", lambda: False)
        monkeypatch.setattr(encrypted_db, "_check_allow_plaintext_fallback", lambda: False)

        db_path = str(tmp_path / "test.db")
        conn = encrypted_connect(db_path)
        conn.execute("CREATE TABLE x (id INTEGER)")
        conn.commit()
        conn.close()

    def test_explicit_empty_key_still_allowed(self, tmp_path, monkeypatch, reset_caches):
        """``key=""`` is the documented escape-hatch for callers that
        deliberately want plain sqlite3. Must keep working.
        """
        monkeypatch.setattr(encrypted_db, "_check_encryption_enabled", lambda: True)
        monkeypatch.setattr(encrypted_db, "_check_allow_plaintext_fallback", lambda: False)

        db_path = str(tmp_path / "test.db")
        # Note: passing key="" bypasses the _check_encryption_enabled
        # path and the fail-closed guard never triggers — the function
        # treats this as "I know what I'm doing".
        conn = encrypted_connect(db_path, key="")
        conn.execute("CREATE TABLE x (id INTEGER)")
        conn.commit()
        conn.close()
