"""Device Pairing — QR-Code basiertes Token-Pairing mit Expiry.

Ermoeglicht sicheres Verbinden von Mobile-Apps mit dem Cognithor-Server:
  1. Server generiert Pairing-Code (JWT mit exp claim)
  2. Mobile scannt QR-Code → speichert Token lokal
  3. Server trackt gepaarte Geraete → Revoke moeglich

Tokens haben ein konfigurierbares TTL (default: 180 Tage).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cognithor.utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_TOKEN_TTL_DAYS = 180
MAX_PAIRED_DEVICES = 20


@dataclass
class PairedDevice:
    device_id: str
    name: str
    paired_at: float
    expires_at: float
    last_seen: float = 0.0
    revoked: bool = False


@dataclass
class PairingToken:
    token: str
    device_id: str
    expires_at: float


class DevicePairingManager:
    """Verwaltet gepaarte Geraete und ihre Tokens."""

    def __init__(
        self,
        master_secret: str,
        storage_path: Path | None = None,
        token_ttl_days: int = DEFAULT_TOKEN_TTL_DAYS,
    ) -> None:
        self._master_secret = master_secret
        self._token_ttl_days = token_ttl_days
        self._storage_path = storage_path or (Path.home() / ".cognithor" / "paired_devices.json")
        self._devices: dict[str, PairedDevice] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_pairing_token(self, device_name: str = "Unknown Device") -> PairingToken:
        """Erstellt einen neuen Pairing-Token fuer ein Geraet.

        Returns:
            PairingToken mit token, device_id und Ablaufzeit.
        """
        device_id = secrets.token_hex(8)
        expires_at = time.time() + (self._token_ttl_days * 86400)

        # Token = HMAC(master_secret, device_id + expires_at)
        payload = f"{device_id}:{expires_at:.0f}"
        token = hmac.new(
            self._master_secret.encode(),
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()

        device = PairedDevice(
            device_id=device_id,
            name=device_name,
            paired_at=time.time(),
            expires_at=expires_at,
        )
        self._devices[device_id] = device
        self._save()

        log.info("device_paired", device_id=device_id, name=device_name)
        return PairingToken(token=token, device_id=device_id, expires_at=expires_at)

    def verify_token(self, token: str, device_id: str) -> bool:
        """Prueft ob ein Token gueltig ist."""
        device = self._devices.get(device_id)
        if device is None:
            return False
        if device.revoked:
            return False
        if time.time() > device.expires_at:
            return False

        # Verify HMAC
        payload = f"{device_id}:{device.expires_at:.0f}"
        expected = hmac.new(
            self._master_secret.encode(),
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(token, expected):
            return False

        # Update last_seen
        device.last_seen = time.time()
        self._save()
        return True

    def revoke_device(self, device_id: str) -> bool:
        """Widerruft den Zugang eines Geraets."""
        device = self._devices.get(device_id)
        if device is None:
            return False
        device.revoked = True
        self._save()
        log.info("device_revoked", device_id=device_id, name=device.name)
        return True

    def list_devices(self) -> list[dict[str, Any]]:
        """Listet alle gepaarten Geraete."""
        now = time.time()
        result = []
        for d in self._devices.values():
            result.append(
                {
                    "device_id": d.device_id,
                    "name": d.name,
                    "paired_at": d.paired_at,
                    "expires_at": d.expires_at,
                    "last_seen": d.last_seen,
                    "revoked": d.revoked,
                    "expired": now > d.expires_at,
                    "active": not d.revoked and now <= d.expires_at,
                }
            )
        return result

    def cleanup_expired(self) -> int:
        """Entfernt abgelaufene und widerrufene Geraete."""
        now = time.time()
        to_remove = [
            did
            for did, d in self._devices.items()
            if d.revoked or now > d.expires_at + 86400 * 30  # 30 Tage Karenz
        ]
        for did in to_remove:
            del self._devices[did]
        if to_remove:
            self._save()
        return len(to_remove)

    def qr_payload(self, pairing_token: PairingToken, host: str, port: int) -> str:
        """Erstellt den QR-Code-Inhalt fuer die Mobile-App."""
        return json.dumps(
            {
                "type": "cognithor_pair",
                "host": host,
                "port": port,
                "device_id": pairing_token.device_id,
                "token": pairing_token.token,
                "expires_at": pairing_token.expires_at,
            },
            separators=(",", ":"),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {did: asdict(d) for did, d in self._devices.items()}
            self._storage_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            log.warning("device_pairing_save_failed", error=str(exc))

    def _load(self) -> None:
        if not self._storage_path.exists():
            return
        try:
            data = json.loads(self._storage_path.read_text(encoding="utf-8"))
            for did, d in data.items():
                self._devices[did] = PairedDevice(**d)
        except Exception as exc:
            log.warning("device_pairing_load_failed", error=str(exc))
