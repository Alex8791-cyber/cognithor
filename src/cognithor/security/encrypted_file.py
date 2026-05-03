"""Transparent file encryption for user data at rest.

Encrypts/decrypts files using Fernet (AES-256-CBC + HMAC-SHA256).
Key is retrieved from the same keyring chain as the DB encryption key.

Usage:
    from cognithor.security.encrypted_file import efile

    # Write encrypted
    efile.write("path/to/note.md", "# My Secret Research\n...")

    # Read transparently (detects encrypted vs plaintext)
    content = efile.read("path/to/note.md")

    # Migrate existing plaintext file to encrypted
    efile.migrate("path/to/old_note.md")

Files are stored with a magic header (COGNITHOR_ENC_V1) followed by
Fernet-encrypted content. The read() method auto-detects whether a
file is encrypted or plaintext, so migration is seamless.
"""

from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path

from cognithor.utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["EncryptedFileIO", "efile"]

_MAGIC_HEADER = b"COGNITHOR_ENC_V1\n"
_FERNET_AVAILABLE = False

try:
    from cryptography.fernet import Fernet

    _FERNET_AVAILABLE = True
except ImportError:
    Fernet = None  # type: ignore


class EncryptedFileIO:
    """Transparent file encryption using Fernet + OS Keyring key.

    - write() always encrypts (if key available)
    - read() auto-detects encrypted vs plaintext
    - migrate() converts plaintext to encrypted in-place
    """

    def __init__(self) -> None:
        self._fernet: Fernet | None = None  # type: ignore
        self._initialized = False

    def _ensure_init(self) -> None:
        """Lazy-init: get key from keyring on first use."""
        if self._initialized:
            return
        self._initialized = True

        if not _FERNET_AVAILABLE:
            log.debug("encrypted_file_no_cryptography", hint="pip install cryptography")
            return

        # If COGNITHOR_DB_KEY is explicitly set, always enable encryption.
        # Otherwise respect the global encryption_enabled config flag.
        if not os.environ.get("COGNITHOR_DB_KEY") and not self._check_encryption_enabled():
            return

        key = self._get_key()
        if key:
            # Derive a Fernet key (32 bytes, base64-encoded) from the DB key
            derived = hashlib.sha256(key.encode()).digest()
            fernet_key = base64.urlsafe_b64encode(derived)
            self._fernet = Fernet(fernet_key)

    def _check_encryption_enabled(self) -> bool:
        """Check whether file encryption is enabled in config.yaml.

        Mirrors the same logic as encrypted_db._check_encryption_enabled().
        """
        try:
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
                    return bool(db_section.get("encryption_enabled", False))
        except Exception:
            log.debug("encrypted_file_config_read_failed", exc_info=True)
        return False

    def _get_key(self) -> str:
        """Get encryption key from the same chain as encrypted_db."""
        # 1. Env var
        key = os.environ.get("COGNITHOR_DB_KEY", "")
        if key:
            return key

        # 2. OS Keyring
        try:
            import keyring

            existing = keyring.get_password("cognithor", "db_encryption_key")
            if existing:
                return existing
        except Exception:
            log.debug("encrypted_file_keyring_read_failed", exc_info=True)

        # 3. CredentialStore
        try:
            from cognithor.security.credentials import CredentialStore

            store = CredentialStore()
            existing = store.retrieve("system", "db_encryption_key")
            if existing:
                return existing
        except Exception:
            log.debug("encrypted_file_credential_store_read_failed", exc_info=True)

        return ""

    @property
    def is_available(self) -> bool:
        """True if encryption is available (key + cryptography installed)."""
        self._ensure_init()
        return self._fernet is not None

    def write(self, path: str | Path, content: str, encoding: str = "utf-8") -> None:
        """Write content to file, encrypted if key is available."""
        self._ensure_init()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self._fernet:
            encrypted = self._fernet.encrypt(content.encode(encoding))
            with open(path, "wb") as f:
                f.write(_MAGIC_HEADER)
                f.write(encrypted)
        else:
            with open(path, "w", encoding=encoding) as f:
                f.write(content)

    def read(self, path: str | Path, encoding: str = "utf-8") -> str:
        """Read file, auto-detecting encrypted vs plaintext."""
        self._ensure_init()
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Check for magic header
        with open(path, "rb") as f:
            header = f.read(len(_MAGIC_HEADER))
            if header == _MAGIC_HEADER:
                # Encrypted file
                encrypted_data = f.read()
                if self._fernet:
                    try:
                        return self._fernet.decrypt(encrypted_data).decode(encoding)
                    except Exception as e:
                        log.error(
                            "encrypted_file_decrypt_failed", path=str(path)[-40:], error=str(e)[:50]
                        )
                        raise
                else:
                    raise RuntimeError(
                        f"File {path} is encrypted but no decryption key available. "
                        f"Set COGNITHOR_DB_KEY or install keyring."
                    )

        # Plaintext file — read normally
        with open(path, encoding=encoding) as f:
            return f.read()

    def is_encrypted(self, path: str | Path) -> bool:
        """Check if a file has the encryption header."""
        path = Path(path)
        if not path.exists():
            return False
        with open(path, "rb") as f:
            return f.read(len(_MAGIC_HEADER)) == _MAGIC_HEADER

    def migrate(self, path: str | Path, encoding: str = "utf-8") -> bool:
        """Convert a plaintext file to encrypted in-place.

        Returns True if migrated, False if already encrypted or no key.
        """
        self._ensure_init()
        path = Path(path)

        if not path.exists():
            return False
        if self.is_encrypted(path):
            return False
        if not self._fernet:
            return False

        # Read plaintext
        with open(path, encoding=encoding) as f:
            content = f.read()

        # Write encrypted
        self.write(path, content, encoding)
        return True

    def migrate_directory(self, directory: str | Path, pattern: str = "*.md") -> int:
        """Migrate all matching files in a directory to encrypted.

        Returns count of files migrated.
        """
        self._ensure_init()
        if not self._fernet:
            return 0

        directory = Path(directory)
        if not directory.exists():
            return 0

        count = 0
        for path in directory.rglob(pattern):
            if path.is_file() and not self.is_encrypted(path):
                try:
                    if self.migrate(path):
                        count += 1
                except Exception:
                    log.debug("encrypted_file_migrate_failed", path=str(path)[-40:], exc_info=True)
        return count


# Singleton instance for convenient access
efile = EncryptedFileIO()
