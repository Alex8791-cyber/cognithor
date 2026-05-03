"""Encrypted database connection wrapper.

Uses SQLCipher (pysqlcipher3) for encryption at rest if available.
Falls back to standard sqlite3 with a WARNING if not installed.

Key management (priority order):
1. COGNITHOR_DB_KEY environment variable (for CI/Docker)
2. OS Keyring (Windows Credential Locker / macOS Keychain / Linux SecretService)
3. Cognithor CredentialStore (Fernet-encrypted file, fallback)

The OS Keyring is the recommended approach: if someone clones your disk,
they cannot access the encrypted databases without your Windows login /
macOS Keychain password. The key never touches the filesystem in plaintext.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
from typing import Any, cast

from cognithor.utils.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "DatabaseError",
    "IntegrityError",
    "OperationalError",
    "compatible_row_factory",
    "encrypted_connect",
    "is_encryption_available",
]


class _DictRow(dict):  # type: ignore[type-arg]
    """Dict that also supports integer index access like sqlite3.Row."""

    __slots__ = ("_values",)

    def __init__(self, columns: list[str], values: tuple[Any, ...]) -> None:
        super().__init__(zip(columns, values, strict=False))
        self._values = values

    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, int):
            return self._values[key]
        return super().__getitem__(key)


def _dict_row_factory(cursor: Any, row: tuple[Any, ...]) -> _DictRow:
    """Row factory that works with both sqlite3 and sqlcipher3 cursors.

    sqlite3.Row requires a sqlite3.Cursor, which sqlcipher3 doesn't provide.
    This factory returns a dict-like object that supports both index and key access.
    """
    columns = [d[0] for d in cursor.description]
    return _DictRow(columns, row)


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
    import sqlcipher3 as sqlcipher  # type: ignore[import-untyped]

    _sqlcipher_available = True
except ImportError:
    try:
        from pysqlcipher3 import dbapi2 as sqlcipher  # type: ignore[import-not-found]

        _sqlcipher_available = True
    except ImportError:
        sqlcipher = None

# Unified exception aliases — resolve to the active backend's types so that
# callers can catch ``encrypted_db.OperationalError`` without caring whether
# sqlcipher3 or sqlite3 is in use.
if _sqlcipher_available and sqlcipher is not None:
    OperationalError: type[Exception] = sqlcipher.OperationalError
    DatabaseError: type[Exception] = sqlcipher.DatabaseError
    IntegrityError: type[Exception] = sqlcipher.IntegrityError
else:
    OperationalError = sqlite3.OperationalError
    DatabaseError = sqlite3.DatabaseError
    IntegrityError = sqlite3.IntegrityError

_KEYRING_SERVICE = "cognithor"
_KEYRING_KEY_NAME = "db_encryption_key"

# Cache the encryption_enabled flag to avoid re-reading config.yaml on every connect
_encryption_enabled_cache: bool | None = None


def _check_encryption_enabled() -> bool:
    """Check whether database encryption is enabled in config.yaml.

    Reads the YAML file directly (no dependency on CognithorConfig) and caches
    the result for the lifetime of the process.  Falls back to False if the
    config cannot be read.
    """
    global _encryption_enabled_cache
    if _encryption_enabled_cache is not None:
        return _encryption_enabled_cache

    try:
        from pathlib import Path

        # Standard Jarvis home locations
        candidates = [
            Path(os.environ.get("COGNITHOR_HOME", "")) / "config.yaml",
            Path.home() / ".cognithor" / "config.yaml",
        ]
        for cfg_path in candidates:
            if cfg_path.is_file():
                import yaml  # type: ignore[import-untyped]

                with open(cfg_path) as f:
                    data = yaml.safe_load(f) or {}
                db_section = data.get("database", {})
                _encryption_enabled_cache = bool(db_section.get("encryption_enabled", False))
                return _encryption_enabled_cache
    except Exception:
        log.debug("encrypted_db_config_read_failed", exc_info=True)

    # Default: encryption disabled (safe fallback — avoids VirtualLock on first run)
    _encryption_enabled_cache = False
    return _encryption_enabled_cache


def is_encryption_available() -> bool:
    """Check if SQLCipher is available."""
    return _sqlcipher_available


def _get_db_key() -> str:
    """Get the database encryption key.

    Priority:
    1. COGNITHOR_DB_KEY environment variable (CI/Docker/explicit)
    2. OS Keyring (Windows Credential Locker / macOS Keychain / Linux SecretService)
       — key is bound to the OS user session, NOT stored on disk
       — a disk clone without the OS login cannot retrieve the key
    3. Cognithor CredentialStore (Fernet file, legacy fallback)
    4. Empty string (triggers WARNING, no encryption)
    """
    # 1. Environment variable (highest priority, for CI/Docker)
    key = os.environ.get("COGNITHOR_DB_KEY", "")
    if key:
        return key

    # 2. OS Keyring — the recommended approach
    try:
        import keyring

        existing = keyring.get_password(_KEYRING_SERVICE, _KEYRING_KEY_NAME)
        if existing:
            return existing

        # Legacy fallback: try "jarvis" service name (pre-rename)
        try:
            stored = keyring.get_password("jarvis", _KEYRING_KEY_NAME)
            if stored:
                # Migrate to new service name
                keyring.set_password(_KEYRING_SERVICE, _KEYRING_KEY_NAME, stored)
                log.info("keyring_key_migrated", from_service="jarvis", to_service=_KEYRING_SERVICE)
                return stored
        except Exception:
            pass

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
        from cognithor.security.credentials import CredentialStore

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

    db_path = str(db_path)  # Ensure string for path concatenation
    backup_path = db_path + ".unencrypted.bak"
    tmp_path = db_path + ".encrypting"

    try:
        # 1. Open the unencrypted DB with plain sqlite3
        plain_conn = sqlite3.connect(db_path, check_same_thread=check_same_thread)
        plain_conn.execute("SELECT count(*) FROM sqlite_master")  # Verify it's readable

        # 2. Use sqlcipher_export to create encrypted copy
        enc_conn = sqlcipher.connect(tmp_path, check_same_thread=check_same_thread)
        enc_conn.execute(f"PRAGMA key = \"x'{hex_key}'\"")
        enc_conn.execute("PRAGMA cipher_memory_security = OFF")
        enc_conn.execute("PRAGMA journal_mode=WAL")

        # 3. Copy all data: dump from plain, execute in encrypted
        for line in plain_conn.iterdump():
            with contextlib.suppress(Exception):
                enc_conn.execute(line)
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
        conn.execute(f"PRAGMA key = \"x'{hex_key}'\"")
        conn.execute("PRAGMA cipher_memory_security = OFF")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("SELECT count(*) FROM sqlite_master")
        return cast("sqlite3.Connection | None", conn)

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
            Pass empty string "" to force plain sqlite3 (no encryption).
        check_same_thread: sqlite3 check_same_thread parameter
        timeout: How many seconds to wait for the database lock (default 5.0)

    Returns:
        sqlite3.Connection (either encrypted or standard)
    """
    db_path = str(db_path)  # Ensure string — Path objects break concatenation + slicing
    # Respect the global encryption_enabled config flag.
    # If encryption is disabled, skip SQLCipher entirely — avoids
    # VirtualLock quota exhaustion on Windows from dozens of open
    # SQLCipher connections calling sqlcipher_mlock().
    if key is None:
        _encryption_enabled = _check_encryption_enabled()

        if not _encryption_enabled:
            # Encryption disabled in config — prefer plain sqlite3.
            # But if the DB was previously encrypted (by old code that
            # always used SQLCipher), we must still open it with SQLCipher
            # to avoid "file is not a database" errors.
            try:
                conn = sqlite3.connect(
                    db_path, check_same_thread=check_same_thread, timeout=timeout
                )
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("SELECT count(*) FROM sqlite_master")
                return conn
            except sqlite3.DatabaseError:
                # DB exists and is encrypted — open with SQLCipher + memory_security OFF
                if (
                    _sqlcipher_available
                    and os.path.exists(db_path)
                    and os.path.getsize(db_path) > 0
                ):
                    fallback_key = _get_db_key()
                    if fallback_key:
                        hex_key = fallback_key.encode().hex()
                        try:
                            conn = sqlcipher.connect(
                                db_path, check_same_thread=check_same_thread, timeout=timeout
                            )
                            conn.execute(f"PRAGMA key = \"x'{hex_key}'\"")
                            conn.execute("PRAGMA cipher_memory_security = OFF")
                            conn.execute("PRAGMA journal_mode=WAL")
                            conn.execute("SELECT count(*) FROM sqlite_master")
                            log.info(
                                "encrypted_db_legacy_open",
                                path=db_path[-40:],
                                hint="DB still encrypted despite encryption_enabled=false",
                            )
                            return conn
                        except Exception:
                            log.debug("encrypted_db_legacy_open_failed", exc_info=True)
                # Re-raise if nothing worked
                raise

        key = _get_db_key()

    if _sqlcipher_available and key:
        hex_key = key.encode().hex()
        try:
            conn = sqlcipher.connect(db_path, check_same_thread=check_same_thread, timeout=timeout)
            conn.execute(f"PRAGMA key = \"x'{hex_key}'\"")
            # Disable VirtualLock — prevents ERROR_WORKING_SET_QUOTA (1453)
            # on Windows when many DBs are open. Encryption keys may land in
            # pagefile, but Windows can encrypt it (EncryptPagingFile=1).
            conn.execute("PRAGMA cipher_memory_security = OFF")
            conn.execute("PRAGMA journal_mode=WAL")
            # Test that the key works
            conn.execute("SELECT count(*) FROM sqlite_master")
            log.debug("encrypted_db_opened", path=db_path[-30:])
            return conn
        except Exception:
            # DB exists but is unencrypted — migrate it to encrypted
            if os.path.exists(db_path) and os.path.getsize(db_path) > 0:
                try:
                    conn = _migrate_to_encrypted(db_path, hex_key, check_same_thread)  # type: ignore[assignment]
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
                    conn.execute(f"PRAGMA key = \"x'{hex_key}'\"")
                    conn.execute("PRAGMA cipher_memory_security = OFF")
                    conn.execute("PRAGMA journal_mode=WAL")
                    log.debug("encrypted_db_created", path=db_path[-30:])
                    return conn
                except Exception:
                    log.debug("encrypted_db_new_file_create_failed", exc_info=True)

    if _sqlcipher_available and not key:
        if not getattr(encrypted_connect, "_warned_no_key", False):
            encrypted_connect._warned_no_key = True  # type: ignore[attr-defined]
            log.warning(
                "sqlcipher_available_but_no_key",
                hint="Install 'keyring' package or set COGNITHOR_DB_KEY env var",
            )
    elif not _sqlcipher_available and not getattr(encrypted_connect, "_warned", False):
        encrypted_connect._warned = True  # type: ignore[attr-defined]
        log.warning(
            "sqlcipher_not_installed",
            hint="pip install pysqlcipher3 keyring",
        )

    # Fallback: standard sqlite3 (unencrypted)
    conn = sqlite3.connect(db_path, check_same_thread=check_same_thread, timeout=timeout)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn
