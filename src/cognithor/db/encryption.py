"""SQLite Encryption Helper — optional SQLCipher + OS Keyring.

Provides a centralized way to open SQLite connections with optional
encryption via SQLCipher, storing the encryption key in the OS keyring
(Windows Credential Locker, macOS Keychain, Linux SecretService).

If pysqlcipher3 or keyring are not installed, falls back gracefully to
plain sqlite3 with appropriate warnings.
"""

from __future__ import annotations

import logging
import secrets
import sqlite3
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)

_KEYRING_SERVICE = "cognithor"
_KEYRING_USERNAME = "db_encryption_key"


def open_sqlite(
    db_path: str | Path,
    encryption_key: str | None = None,
) -> sqlite3.Connection:
    """Open a SQLite connection, optionally encrypted with SQLCipher.

    Args:
        db_path: Path to the database file.
        encryption_key: Hex key for SQLCipher. If provided and pysqlcipher3
            is available, PRAGMA key is issued immediately after connect.
            If pysqlcipher3 is not installed, falls back to plain sqlite3
            with a warning.

    Returns:
        A sqlite3.Connection (or pysqlcipher3-compatible Connection).
    """
    db_path = str(db_path)

    if encryption_key:
        try:
            from pysqlcipher3 import dbapi2 as sqlcipher

            conn = sqlcipher.connect(db_path, check_same_thread=False)
            # PRAGMA key cannot use parameterized queries; escape single quotes
            safe_key = encryption_key.replace("'", "''")
            conn.execute(f"PRAGMA key='{safe_key}'")
            # Disable VirtualLock to prevent Windows quota exhaustion
            conn.execute("PRAGMA cipher_memory_security = OFF")
            log.info("SQLCipher-Verbindung hergestellt: %s", db_path)
            return cast("sqlite3.Connection", conn)

        except ImportError:
            log.warning(
                "SQLCipher angefordert aber pysqlcipher3 nicht installiert. "
                "Fallback auf unverschluesseltes sqlite3: %s",
                db_path,
            )

    conn = sqlite3.connect(db_path, check_same_thread=False)
    return cast("sqlite3.Connection", conn)


def get_encryption_key(config: Any = None) -> str | None:
    """Read the encryption key from the OS keyring.

    Args:
        config: Optional config object. If it has a ``database`` attribute
            with ``encryption_enabled=False``, returns None immediately.

    Returns:
        The stored hex key, or None if encryption is disabled or keyring
        is not available.
    """
    # Check config flag first
    if config is not None:
        db_cfg = getattr(config, "database", None)
        if db_cfg is not None and not getattr(db_cfg, "encryption_enabled", False):
            return None

    try:
        import keyring
    except ImportError:
        log.warning(
            "keyring-Bibliothek nicht installiert. SQLite-Verschluesselung nicht verfuegbar."
        )
        return None

    try:
        key = keyring.get_password(_KEYRING_SERVICE, _KEYRING_USERNAME)
        if key:
            log.debug("Verschluesselungsschluessel aus Keyring geladen.")
        return key
    except Exception:
        log.warning("Keyring-Zugriff fehlgeschlagen.", exc_info=True)
        return None


def init_encryption(passphrase: str | None = None) -> str:
    """Generate or import an encryption key and store it in the OS keyring.

    Args:
        passphrase: If provided, use this as the key. Otherwise generate
            a random 64-char hex string (32 bytes of entropy).

    Returns:
        The key that was stored.

    Raises:
        ImportError: If keyring library is not installed.
        RuntimeError: If storing the key fails.
    """
    try:
        import keyring
    except ImportError as exc:
        raise ImportError(
            "keyring-Bibliothek wird fuer Verschluesselung benoetigt: pip install keyring"
        ) from exc

    key = passphrase if passphrase else secrets.token_hex(32)

    try:
        keyring.set_password(_KEYRING_SERVICE, _KEYRING_USERNAME, key)
        log.info("Verschluesselungsschluessel im OS-Keyring gespeichert.")
    except Exception as exc:
        raise RuntimeError(f"Keyring-Speicherung fehlgeschlagen: {exc}") from exc

    return key


def remove_encryption_key() -> bool:
    """Remove the encryption key from the OS keyring.

    Returns:
        True if the key was removed, False if keyring is not available
        or the key did not exist.
    """
    try:
        import keyring
    except ImportError:
        log.warning("keyring nicht installiert — nichts zu entfernen.")
        return False

    try:
        keyring.delete_password(_KEYRING_SERVICE, _KEYRING_USERNAME)
        log.info("Verschluesselungsschluessel aus Keyring entfernt.")
        return True
    except keyring.errors.PasswordDeleteError:
        log.info("Kein Verschluesselungsschluessel im Keyring vorhanden.")
        return False
    except Exception:
        log.warning("Keyring-Loesch-Fehler.", exc_info=True)
        return False
