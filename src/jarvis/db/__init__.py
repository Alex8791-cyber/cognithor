"""Jarvis · Database Abstraction Layer.

Unterstuetzt SQLite (Default) und PostgreSQL (Optional).
Optional: SQLCipher-Verschluesselung mit OS-Keyring.
"""

from jarvis.db.backend import DatabaseBackend
from jarvis.db.encryption import (
    get_encryption_key,
    init_encryption,
    open_sqlite,
    remove_encryption_key,
)
from jarvis.db.factory import create_backend

# SQLite busy timeout in milliseconds.
# All modules that open SQLite connections should use this constant
# to ensure consistent lock-wait behavior across the application.
SQLITE_BUSY_TIMEOUT_MS = 5000

__all__ = [
    "DatabaseBackend",
    "SQLITE_BUSY_TIMEOUT_MS",
    "create_backend",
    "get_encryption_key",
    "init_encryption",
    "open_sqlite",
    "remove_encryption_key",
]
