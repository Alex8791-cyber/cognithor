"""Credential-Manager: Sichere Verwaltung von Secrets.

Speichert Credentials verschlüsselt (Fernet/AES-256) auf Disk.
Der Planner (LLM) sieht NIEMALS Klartext-Credentials.
Der Gatekeeper injiziert Credentials in den Executor-Kontext.

Sicherheitsgarantien:
  - Credentials werden mit AES-256 (via Fernet) verschlüsselt
  - Master-Key wird aus Passphrase + Salt abgeleitet (PBKDF2)
  - Planner hat keinen Zugriff auf den Store
  - Audit-Log maskiert alle Credential-Werte
  - Datei-Permissions werden auf 0600 gesetzt

Bibel-Referenz: §11.2 (Credential-Management)
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jarvis.models import CredentialEntry
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# Optionaler Import: cryptography für Fernet
_HAS_CRYPTO = False
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    _HAS_CRYPTO = True
except ImportError:
    pass


def _derive_key(passphrase: str, salt: bytes) -> bytes:
    """Leitet einen Fernet-Key aus Passphrase + Salt ab (PBKDF2)."""
    if not _HAS_CRYPTO:
        raise RuntimeError("cryptography-Paket nicht installiert. pip install cryptography")
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=600_000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))
    return key


def _obfuscate_key(passphrase: str, salt: bytes) -> bytes:
    """Veraltet -- wird nicht mehr verwendet.

    Existiert nur noch für Abwärtskompatibilität beim Lesen alter Stores.
    """
    combined = passphrase.encode() + salt
    key = hashlib.sha256(combined).digest()
    return base64.urlsafe_b64encode(key)


class CredentialStore:
    """Verschlüsselter Credential-Store. [B§11.2]

    Speichert Key-Value-Paare verschlüsselt als JSON auf Disk.
    Unterstützt zwei Modi:
      1. Fernet (AES-256) -- wenn `cryptography` installiert ist
      2. Base64-Obfuskation -- Fallback für Entwicklung (UNSICHER)

    Der Planner hat keinen direkten Zugriff. Credentials werden
    vom Gatekeeper über inject_credentials() bereitgestellt.
    """

    def __init__(
        self,
        store_path: Path | None = None,
        passphrase: str | None = None,
    ) -> None:
        """Initialisiert den Credential-Store.

        Args:
            store_path: Pfad zur verschlüsselten Store-Datei.
            passphrase: Master-Passphrase für Verschlüsselung.
                       Wenn None: JARVIS_CREDENTIAL_KEY Env-Variable.
        """
        self._store_path = store_path or (Path.home() / ".jarvis" / "credentials.enc")
        self._store_path.parent.mkdir(parents=True, exist_ok=True)

        self._passphrase = passphrase or os.environ.get("JARVIS_CREDENTIAL_KEY", "")
        self._salt = self._load_or_create_salt()
        self._fernet = self._init_fernet()
        self._entries: dict[str, _StoredCredential] = {}
        self._loaded = False

    def _init_fernet(self) -> Any:
        """Initialisiert Fernet-Verschlüsselung."""
        if not self._passphrase:
            log.warning("credential_store_no_passphrase")
            return None
        if not _HAS_CRYPTO:
            raise RuntimeError(
                "cryptography-Paket erforderlich für Credential-Verschlüsselung. "
                "pip install cryptography"
            )
        key = _derive_key(self._passphrase, self._salt)
        return Fernet(key)

    def _load_or_create_salt(self) -> bytes:
        """Lädt oder erstellt den Salt für Key-Derivation."""
        salt_path = self._store_path.parent / ".credential_salt"
        if salt_path.exists():
            return salt_path.read_bytes()
        salt = os.urandom(16)
        salt_path.write_bytes(salt)
        self._set_file_permissions(salt_path)
        return salt

    def store(self, service: str, key: str, value: str, agent_id: str = "") -> CredentialEntry:
        """Speichert ein Credential.

        Args:
            service: Service-Name (z.B. 'telegram', 'searxng').
            key: Schlüssel (z.B. 'api_key', 'bot_token').
            value: Klartext-Wert.
            agent_id: Optionale Agent-Zuordnung. Leer = global verfügbar.

        Returns:
            CredentialEntry (ohne Klartext-Wert).
        """
        self._ensure_loaded()
        lookup = f"{service}:{key}" if not agent_id else f"{agent_id}/{service}:{key}"
        encrypted = self._encrypt(value)

        self._entries[lookup] = _StoredCredential(
            service=service,
            key=key,
            encrypted_value=encrypted,
            created_at=datetime.now(UTC),
            agent_id=agent_id,
        )
        self._save()

        log.info(
            "credential_stored",
            service=service,
            key=key,
            agent_id=agent_id or "global",
        )
        return CredentialEntry(
            service=service,
            key=key,
            encrypted=self._fernet is not None,
        )

    def retrieve(self, service: str, key: str, agent_id: str = "") -> str | None:
        """Ruft ein Credential ab (NUR für Executor/Gatekeeper).

        Prüft zuerst agent-spezifische Credentials, dann globale.

        Args:
            service: Service-Name.
            key: Schlüssel.
            agent_id: Agent-ID für scoped Zugriff.

        Returns:
            Klartext-Wert oder None wenn nicht gefunden.
        """
        self._ensure_loaded()

        # 1. Agent-spezifisches Credential
        if agent_id:
            agent_lookup = f"{agent_id}/{service}:{key}"
            stored = self._entries.get(agent_lookup)
            if stored:
                stored.last_accessed = datetime.now(UTC)
                return self._decrypt(stored.encrypted_value)

        # 2. Globales Credential (Fallback)
        global_lookup = f"{service}:{key}"
        stored = self._entries.get(global_lookup)
        if not stored:
            return None

        stored.last_accessed = datetime.now(UTC)
        return self._decrypt(stored.encrypted_value)

    def delete(self, service: str, key: str, agent_id: str = "") -> bool:
        """Löscht ein Credential.

        Args:
            service: Service-Name.
            key: Schlüssel.
            agent_id: Wenn gesetzt, agent-spezifisches Credential loeschen.

        Returns:
            True wenn gelöscht, False wenn nicht gefunden.
        """
        self._ensure_loaded()
        if agent_id:
            lookup = f"{agent_id}/{service}:{key}"
            if lookup in self._entries:
                del self._entries[lookup]
                self._save()
                log.info("credential_deleted", service=service, key=key, agent_id=agent_id)
                return True
            return False
        lookup = f"{service}:{key}"
        if lookup not in self._entries:
            return False

        del self._entries[lookup]
        self._save()
        log.info("credential_deleted", service=service, key=key)
        return True

    def list_entries(self, agent_id: str = "") -> list[CredentialEntry]:
        """Listet Credentials (ohne Werte).

        Args:
            agent_id: Wenn gesetzt, nur Credentials dieses Agenten + globale.
                     Leer = alle Credentials.

        Returns:
            Liste von CredentialEntry-Objekten.
        """
        self._ensure_loaded()
        results = []
        for e in self._entries.values():
            if agent_id and e.agent_id and e.agent_id != agent_id:
                continue  # Anderer Agent → überspringen
            results.append(
                CredentialEntry(
                    service=e.service,
                    key=e.key,
                    encrypted=self._fernet is not None,
                    created_at=e.created_at,
                    last_accessed=e.last_accessed,
                )
            )
        return results

    def has(self, service: str, key: str, agent_id: str = "") -> bool:
        """Prüft ob ein Credential existiert."""
        self._ensure_loaded()
        if agent_id:
            if f"{agent_id}/{service}:{key}" in self._entries:
                return True
        return f"{service}:{key}" in self._entries

    def inject_credentials(self, params: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
        """Injiziert Credentials in Tool-Parameter.

        Wird vom Gatekeeper aufgerufen, NICHT vom Planner.

        Args:
            params: Die Tool-Parameter.
            mapping: Mapping von Param-Name → 'service:key'.
                    z.B. {'api_key': 'searxng:api_key'}

        Returns:
            Kopie der Parameter mit injizierten Credentials.
        """
        result = dict(params)
        for param_name, credential_ref in mapping.items():
            parts = credential_ref.split(":", 1)
            if len(parts) != 2:
                continue
            service, key = parts
            value = self.retrieve(service, key)
            if value is not None:
                result[param_name] = value
        return result

    @property
    def count(self) -> int:
        """Anzahl gespeicherter Credentials."""
        self._ensure_loaded()
        return len(self._entries)

    @property
    def is_encrypted(self) -> bool:
        """True wenn echte Verschlüsselung aktiv ist."""
        return self._fernet is not None and _HAS_CRYPTO

    # ========================================================================
    # Private Methoden
    # ========================================================================

    def _encrypt(self, plaintext: str) -> str:
        """Verschlüsselt einen Klartext-Wert."""
        if not self._fernet:
            raise RuntimeError(
                "cryptography-Paket erforderlich für Credential-Verschlüsselung. "
                "pip install cryptography"
            )
        return self._fernet.encrypt(plaintext.encode()).decode()  # type: ignore[no-any-return]

    def _decrypt(self, ciphertext: str) -> str | None:
        """Entschlüsselt einen verschlüsselten Wert."""
        try:
            if not self._fernet:
                raise RuntimeError(
                    "cryptography-Paket erforderlich für Credential-Entschlüsselung. "
                    "pip install cryptography"
                )
            return self._fernet.decrypt(ciphertext.encode()).decode()  # type: ignore[no-any-return]
        except Exception as exc:
            # Ciphertext-Prefix fuer Debugging (erste 8 Zeichen, keine Secrets exponiert)
            preview = ciphertext[:8] + "..." if len(ciphertext) > 8 else ciphertext
            log.warning(
                "credential_decrypt_failed",
                error=str(exc),
                ciphertext_preview=preview,
            )
            return None

    def _ensure_loaded(self) -> None:
        """Lädt den Store von Disk wenn nötig."""
        if self._loaded:
            return
        if self._store_path.exists():
            try:
                raw = self._store_path.read_text(encoding="utf-8")
                data = json.loads(raw)
                for lookup, entry_data in data.items():
                    self._entries[lookup] = _StoredCredential(
                        service=entry_data["service"],
                        key=entry_data["key"],
                        encrypted_value=entry_data["encrypted_value"],
                        created_at=datetime.fromisoformat(
                            entry_data.get("created_at", "2025-01-01T00:00:00+00:00")
                        ),
                        last_accessed=(
                            datetime.fromisoformat(entry_data["last_accessed"])
                            if entry_data.get("last_accessed")
                            else None
                        ),
                        agent_id=entry_data.get("agent_id", ""),
                    )
            except (json.JSONDecodeError, KeyError, OSError) as exc:
                log.error("credential_store_load_failed", error=str(exc))
        self._loaded = True

    def _save(self) -> None:
        """Speichert den Store auf Disk."""
        data: dict[str, Any] = {}
        for lookup, entry in self._entries.items():
            data[lookup] = {
                "service": entry.service,
                "key": entry.key,
                "encrypted_value": entry.encrypted_value,
                "created_at": entry.created_at.isoformat(),
                "last_accessed": (entry.last_accessed.isoformat() if entry.last_accessed else None),
                "agent_id": entry.agent_id,
            }
        raw = json.dumps(data, ensure_ascii=False, indent=2)
        self._store_path.write_text(raw, encoding="utf-8")
        self._set_file_permissions(self._store_path)

    @staticmethod
    def _set_file_permissions(path: Path) -> None:
        """Setzt Datei-Permissions auf Owner-only (0600)."""
        with contextlib.suppress(OSError):
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)


class _StoredCredential:
    """Internes Speicher-Format für Credentials."""

    __slots__ = (
        "agent_id",
        "created_at",
        "encrypted_value",
        "key",
        "last_accessed",
        "service",
    )

    def __init__(
        self,
        service: str,
        key: str,
        encrypted_value: str,
        created_at: datetime | None = None,
        last_accessed: datetime | None = None,
        agent_id: str = "",
    ) -> None:
        self.service = service
        self.key = key
        self.encrypted_value = encrypted_value
        self.created_at = created_at or datetime.now(UTC)
        self.last_accessed = last_accessed
        self.agent_id = agent_id
