"""SecureTokenStore: Encrypted runtime token management.

Lightweight in-memory store that encrypts tokens with an ephemeral
Fernet key. Protects secrets in RAM from memory dumps.

No disk I/O, no PBKDF2 -- purely for runtime protection.
Fallback: Base64 obfuscation when cryptography is not installed.

Bible reference: §11.2 (Credential-Management)
"""

from __future__ import annotations

import base64
import logging
import ssl
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Optionaler Import: cryptography fuer Fernet
_HAS_CRYPTO = False
try:
    from cryptography.fernet import Fernet

    _HAS_CRYPTO = True
except ImportError:
    pass  # Optional: cryptography not installed, encryption features unavailable


class SecureTokenStore:
    """Ephemeral in-memory token store with Fernet encryption.

    Generates a random Fernet key on startup and encrypts
    all stored tokens with it. The key exists only in RAM.

    Without cryptography package: Falls back to Base64 obfuscation with warning.
    """

    def __init__(self) -> None:
        self._fernet: Any = None
        if _HAS_CRYPTO:
            self._fernet = Fernet(Fernet.generate_key())
        else:
            logger.error(
                "SECURITY DEGRADATION: cryptography not installed -- "
                "Token store uses Base64 fallback (NOT encrypted!). "
                "Tokens are readable in RAM as plaintext. "
                "Installiere mit: pip install cryptography"
            )
        self._tokens: dict[str, bytes] = {}  # name → ciphertext/encoded
        self._lock = threading.Lock()

    def store(self, name: str, value: str) -> None:
        """Stores a token encrypted.

        Args:
            name: Unique name (e.g. 'telegram_bot_token').
            value: Plaintext token.
        """
        data = value.encode("utf-8")
        if self._fernet is not None:
            encrypted = self._fernet.encrypt(data)
        else:
            # PASS-4 SEC-MED: previously logged the token *name* on every
            # store() call, building a persistent inventory of which
            # secrets the process holds (cognithor.jsonl). The startup
            # warning in __init__ already covers the security degradation;
            # per-token name-leak is unnecessary noise + a soft inventory
            # leak. Drop the per-call log; the value is already not
            # logged.
            encrypted = base64.b85encode(data)
        with self._lock:
            self._tokens[name] = encrypted

    def retrieve(self, name: str) -> str:
        """Decrypts and returns a token.

        Args:
            name: Token name.

        Returns:
            Plaintext token.

        Raises:
            KeyError: If the token does not exist.
        """
        with self._lock:
            encrypted = self._tokens[name]  # KeyError wenn nicht vorhanden
        if self._fernet is not None:
            return self._fernet.decrypt(encrypted).decode("utf-8")  # type: ignore[no-any-return]
        return base64.b85decode(encrypted).decode("utf-8")

    def clear(self) -> None:
        """Clears all stored tokens."""
        with self._lock:
            self._tokens.clear()

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._tokens


# ============================================================================
# Singleton
# ============================================================================

_instance: SecureTokenStore | None = None
_instance_lock = threading.Lock()


def get_token_store() -> SecureTokenStore:
    """Returns the global SecureTokenStore instance (singleton)."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = SecureTokenStore()
    return _instance


# ============================================================================
# TLS-Helper
# ============================================================================


def create_ssl_context(certfile: str, keyfile: str) -> ssl.SSLContext | None:
    """Creates an SSLContext if certificate and key are present.

    Args:
        certfile: Path to the SSL certificate (PEM).
        keyfile: Path to the SSL private key (PEM).

    Returns:
        Configured SSLContext or None if files are missing.
    """
    if not certfile or not keyfile:
        return None

    cert_path = Path(certfile)
    key_path = Path(keyfile)

    if not cert_path.is_file() or not key_path.is_file():
        logger.warning(
            "TLS certificates not found: cert=%s, key=%s",
            certfile,
            keyfile,
        )
        return None

    try:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(str(cert_path), str(key_path))
        # Secure defaults
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        logger.info("TLS context created: %s", certfile)
        return ctx
    except Exception:
        logger.exception("Error creating TLS context")
        return None
