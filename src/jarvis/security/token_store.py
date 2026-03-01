"""SecureTokenStore: Verschlüsselte Runtime-Token-Verwaltung.

Leichtgewichtiger In-Memory-Store, der Tokens mit einem ephemeren
Fernet-Key verschlüsselt. Schützt Secrets im RAM vor Memory-Dumps.

Kein Disk-I/O, kein PBKDF2 -- rein für Runtime-Schutz.
Fallback: Base64-Obfuskation wenn cryptography nicht installiert.

Bibel-Referenz: §11.2 (Credential-Management)
"""

from __future__ import annotations

import base64
import logging
import ssl
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# Optionaler Import: cryptography für Fernet
_HAS_CRYPTO = False
try:
    from cryptography.fernet import Fernet

    _HAS_CRYPTO = True
except ImportError:
    pass


class SecureTokenStore:
    """Ephemerer In-Memory-Token-Store mit Fernet-Verschlüsselung.

    Generiert beim Start einen zufälligen Fernet-Key und verschlüsselt
    alle gespeicherten Tokens damit. Der Key existiert nur im RAM.

    Ohne cryptography-Paket: Fallback auf Base64-Obfuskation mit Warning.
    """

    def __init__(self) -> None:
        if _HAS_CRYPTO:
            self._fernet = Fernet(Fernet.generate_key())
        else:
            self._fernet = None
            logger.warning(
                "cryptography nicht installiert -- Token-Store nutzt Base64-Fallback. "
                "Installiere mit: pip install cryptography"
            )
        self._tokens: dict[str, bytes] = {}  # name → ciphertext/encoded
        self._lock = threading.Lock()

    def store(self, name: str, value: str) -> None:
        """Speichert einen Token verschlüsselt.

        Args:
            name: Eindeutiger Name (z.B. 'telegram_bot_token').
            value: Klartext-Token.
        """
        data = value.encode("utf-8")
        if self._fernet is not None:
            encrypted = self._fernet.encrypt(data)
        else:
            encrypted = base64.b85encode(data)
        with self._lock:
            self._tokens[name] = encrypted

    def retrieve(self, name: str) -> str:
        """Entschlüsselt und gibt einen Token zurück.

        Args:
            name: Token-Name.

        Returns:
            Klartext-Token.

        Raises:
            KeyError: Wenn der Token nicht existiert.
        """
        with self._lock:
            encrypted = self._tokens[name]  # KeyError wenn nicht vorhanden
        if self._fernet is not None:
            return self._fernet.decrypt(encrypted).decode("utf-8")
        return base64.b85decode(encrypted).decode("utf-8")

    def clear(self) -> None:
        """Löscht alle gespeicherten Tokens."""
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
    """Gibt die globale SecureTokenStore-Instanz zurück (Singleton)."""
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
    """Erstellt einen SSLContext wenn Zertifikat und Key vorhanden sind.

    Args:
        certfile: Pfad zum SSL-Zertifikat (PEM).
        keyfile: Pfad zum SSL-Privat-Key (PEM).

    Returns:
        Konfigurierter SSLContext oder None wenn Dateien fehlen.
    """
    if not certfile or not keyfile:
        return None

    cert_path = Path(certfile)
    key_path = Path(keyfile)

    if not cert_path.is_file() or not key_path.is_file():
        logger.warning(
            "TLS-Zertifikate nicht gefunden: cert=%s, key=%s",
            certfile,
            keyfile,
        )
        return None

    try:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(str(cert_path), str(key_path))
        # Sichere Defaults
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        logger.info("TLS-Kontext erstellt: %s", certfile)
        return ctx
    except Exception:
        logger.exception("Fehler beim Erstellen des TLS-Kontexts")
        return None
