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

__all__ = ["encrypted_connect", "is_encryption_available"]

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
        log.info("db_encryption_key_generated_credential_store",
                 hint="Install 'keyring' package for OS-level key protection")
        return new_key
    except Exception:
        log.debug("credential_store_unavailable_for_db_key", exc_info=True)

    # 4. No key available
    return ""


def encrypted_connect(
    db_path: str,
    key: str | None = None,
    check_same_thread: bool = False,
) -> sqlite3.Connection:
    """Open a database connection, encrypted if SQLCipher is available.

    Args:
        db_path: Path to the SQLite database file
        key: Encryption key. If None, auto-detected from env/keyring/credential store.
        check_same_thread: sqlite3 check_same_thread parameter

    Returns:
        sqlite3.Connection (either encrypted or standard)
    """
    if key is None:
        key = _get_db_key()

    if _sqlcipher_available and key:
        try:
            conn = sqlcipher.connect(db_path, check_same_thread=check_same_thread)
            # Use hex key format to prevent SQL injection from user-supplied keys
            hex_key = key.encode().hex()
            conn.execute(f"PRAGMA key = \"x'{hex_key}'\"")  # noqa: S608
            conn.execute("PRAGMA journal_mode=WAL")
            # Test that the key works
            conn.execute("SELECT count(*) FROM sqlite_master")
            log.debug("encrypted_db_opened", path=db_path[-30:])
            return conn
        except Exception as e:
            log.warning("sqlcipher_open_failed_falling_back", path=db_path[-30:], error=str(e)[:50])

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
    conn = sqlite3.connect(db_path, check_same_thread=check_same_thread)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn
