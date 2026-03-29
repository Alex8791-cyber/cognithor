"""Encrypted database connection wrapper.

Uses SQLCipher (pysqlcipher3) for encryption at rest if available.
Falls back to standard sqlite3 with a WARNING if not installed.

Key management (priority order):
1. JARVIS_DB_KEY environment variable (for CI/Docker)
2. OS Keyring (Windows Credential Locker / macOS Keychain / Linux SecretService)
3. Cognithor CredentialStore (Fernet-encrypted file, fallback)

The OS Keyring is the recommended approach: if someone clones your disk,
they cannot access the encrypted databases without your Windows login /
macOS Keychain password. The key never touches the filesystem in plaintext.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any

from jarvis.utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["encrypted_connect", "is_encryption_available", "compatible_row_factory"]


def _dict_row_factory(cursor: Any, row: tuple) -> sqlite3.Row | dict:
    """Row factory that works with both sqlite3 and sqlcipher3 cursors.

    sqlite3.Row requires a sqlite3.Cursor, which sqlcipher3 doesn't provide.
    This factory returns a dict-like object that supports both index and key access.
    """
    columns = [d[0] for d in cursor.description]
    return dict(zip(columns, row))


def compatible_row_factory() -> Any:
    """Return a row factory compatible with the active DB backend.

    Use this instead of sqlite3.Row when the DB might be encrypted:
        conn.row_factory = compatible_row_factory()
    """
    if _sqlcipher_available:
        return _dict_row_factory
    return sqlite3.Row


_sqlcipher_available = False
try:
    from pysqlcipher3 import dbapi2 as sqlcipher

    _sqlcipher_available = True
except ImportError:
    try:
        import sqlcipher3 as sqlcipher

        _sqlcipher_available = True
    except ImportError:
        sqlcipher = None

_KEYRING_SERVICE = "cognithor"
_KEYRING_KEY_NAME = "db_encryption_key"


def is_encryption_available() -> bool:
    """Check if SQLCipher is available."""
    return _sqlcipher_available


def _get_db_key() -> str:
    """Get the database encryption key.

    Priority:
    1. JARVIS_DB_KEY environment variable (CI/Docker/explicit)
    2. OS Keyring (Windows Credential Locker / macOS Keychain / Linux SecretService)
       — key is bound to the OS user session, NOT stored on disk
       — a disk clone without the OS login cannot retrieve the key
    3. Cognithor CredentialStore (Fernet file, legacy fallback)
    4. Empty string (triggers WARNING, no encryption)
    """
    # 1. Environment variable (highest priority, for CI/Docker)
    key = os.environ.get("JARVIS_DB_KEY", "")
    if key:
        return key

    # 2. OS Keyring — the recommended approach
    try:
        import keyring

        existing = keyring.get_password(_KEYRING_SERVICE, _KEYRING_KEY_NAME)
        if existing:
            return existing

        # Auto-generate and store in keyring
        import secrets

        new_key = secrets.token_hex(32)
        keyring.set_password(_KEYRING_SERVICE, _KEYRING_KEY_NAME, new_key)
        log.info("db_encryption_key_stored_in_os_keyring")
        return new_key
    except ImportError:
        log.debug("keyring_not_installed", hint="pip install keyring for OS-level key storage")
    except Exception as e:
        log.debug("keyring_unavailable", error=str(e)[:80])

    # 3. Cognithor CredentialStore (Fernet file — legacy fallback)
    try:
        from jarvis.security.credentials import CredentialStore

        store = CredentialStore()
        existing = store.retrieve("system", "db_encryption_key")
        if existing:
            log.debug("db_key_from_credential_store")
            return existing
        # Generate new key
        import secrets

        new_key = secrets.token_hex(32)
        store.store("system", "db_encryption_key", new_key)
        log.info(
            "db_encryption_key_generated_credential_store",
            hint="Install 'keyring' package for OS-level key protection",
        )
        return new_key
    except Exception:
        log.debug("credential_store_unavailable_for_db_key", exc_info=True)

    # 4. No key available
    return ""


def _migrate_to_encrypted(
    db_path: str, hex_key: str, check_same_thread: bool = False
) -> sqlite3.Connection | None:
    """Migrate an existing unencrypted DB to SQLCipher encrypted.

    Strategy: open plain DB, create encrypted copy, replace original.
    """
    import shutil
    import tempfile

    backup_path = db_path + ".unencrypted.bak"
    tmp_path = db_path + ".encrypting"

    try:
        # 1. Open the unencrypted DB with plain sqlite3
        plain_conn = sqlite3.connect(db_path, check_same_thread=check_same_thread)
        plain_conn.execute("SELECT count(*) FROM sqlite_master")  # Verify it's readable

        # 2. Use sqlcipher_export to create encrypted copy
        enc_conn = sqlcipher.connect(tmp_path, check_same_thread=check_same_thread)
        enc_conn.execute(f"PRAGMA key = \"x'{hex_key}'\"")  # noqa: S608
        enc_conn.execute("PRAGMA journal_mode=WAL")

        # 3. Copy all data: dump from plain, execute in encrypted
        for line in plain_conn.iterdump():
            try:
                enc_conn.execute(line)
            except Exception:
                pass  # Skip duplicate CREATE statements etc. (catches both sqlite3 and sqlcipher3)
        enc_conn.commit()

        # 4. Verify encrypted DB
        enc_conn.execute("SELECT count(*) FROM sqlite_master")

        plain_conn.close()
        enc_conn.close()

        # 5. Backup original, replace with encrypted
        shutil.copy2(db_path, backup_path)
        shutil.move(tmp_path, db_path)

        # 6. Open the now-encrypted DB
        conn = sqlcipher.connect(db_path, check_same_thread=check_same_thread)
        conn.execute(f"PRAGMA key = \"x'{hex_key}'\"")  # noqa: S608
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("SELECT count(*) FROM sqlite_master")
        return conn
    except Exception:
        # Cleanup on failure
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def encrypted_connect(
    db_path: str,
    key: str | None = None,
    check_same_thread: bool = False,
    timeout: float = 5.0,
) -> sqlite3.Connection:
    """Open a database connection, encrypted if SQLCipher is available.

    Args:
        db_path: Path to the SQLite database file
        key: Encryption key. If None, auto-detected from env/keyring/credential store.
        check_same_thread: sqlite3 check_same_thread parameter
        timeout: How many seconds to wait for the database lock (default 5.0)

    Returns:
        sqlite3.Connection (either encrypted or standard)
    """
    if key is None:
        key = _get_db_key()

    if _sqlcipher_available and key:
        hex_key = key.encode().hex()
        try:
            conn = sqlcipher.connect(db_path, check_same_thread=check_same_thread, timeout=timeout)
            conn.execute(f"PRAGMA key = \"x'{hex_key}'\"")  # noqa: S608
            conn.execute("PRAGMA journal_mode=WAL")
            # Test that the key works
            conn.execute("SELECT count(*) FROM sqlite_master")
            log.debug("encrypted_db_opened", path=db_path[-30:])
            return conn
        except Exception:
            # DB exists but is unencrypted — migrate it to encrypted
            if os.path.exists(db_path) and os.path.getsize(db_path) > 0:
                try:
                    conn = _migrate_to_encrypted(db_path, hex_key, check_same_thread)
                    if conn:
                        log.info("encrypted_db_migrated", path=db_path[-30:])
                        return conn
                except Exception as e2:
                    log.warning(
                        "encrypted_db_migration_failed", path=db_path[-30:], error=str(e2)[:50]
                    )
            # New empty DB — create encrypted from scratch
            elif not os.path.exists(db_path) or os.path.getsize(db_path) == 0:
                try:
                    conn = sqlcipher.connect(
                        db_path, check_same_thread=check_same_thread, timeout=timeout
                    )
                    conn.execute(f"PRAGMA key = \"x'{hex_key}'\"")  # noqa: S608
                    conn.execute("PRAGMA journal_mode=WAL")
                    log.debug("encrypted_db_created", path=db_path[-30:])
                    return conn
                except Exception:
                    pass

    if _sqlcipher_available and not key:
        log.warning(
            "sqlcipher_available_but_no_key",
            path=db_path[-30:],
            hint="Install 'keyring' package or set JARVIS_DB_KEY env var",
        )
    elif not _sqlcipher_available:
        log.warning(
            "sqlcipher_not_installed",
            hint="pip install pysqlcipher3 keyring",
        )

    # Fallback: standard sqlite3 (unencrypted)
    conn = sqlite3.connect(db_path, check_same_thread=check_same_thread, timeout=timeout)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn
