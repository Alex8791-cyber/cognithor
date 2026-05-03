"""Network Endpoint Manager — Multi-Interface API Binding.

Erkennt und verwaltet Netzwerk-Interfaces fuer den API-Server:
  - Auto-Detection: Tailscale, ZeroTier, WireGuard, LAN, Loopback
  - User-konfigurierbar: welche Interfaces aktiv sein sollen
  - Persistente Config in ~/.cognithor/network_endpoints.json

Unterstuetzte VPN/Overlay-Netzwerke:
  - Tailscale:  100.64.0.0/10  (CGNAT)
  - ZeroTier:   172.16.0.0/12  (oft 172.2x.x.x)
  - WireGuard:  typisch 10.x.x.x (konfigurierbar)
  - Cloudflare: erkennt via Interface-Name
"""

from __future__ import annotations

import ipaddress
import json
import socket
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from cognithor.utils.logging import get_logger

log = get_logger(__name__)


class InterfaceType(StrEnum):
    LOOPBACK = "loopback"
    LAN = "lan"
    TAILSCALE = "tailscale"
    ZEROTIER = "zerotier"
    VPN = "vpn"
    UNKNOWN = "unknown"


# Known overlay network ranges
_TAILSCALE_NET = ipaddress.ip_network("100.64.0.0/10")
_ZEROTIER_NETS = [
    ipaddress.ip_network("172.16.0.0/12"),  # ZeroTier default
    ipaddress.ip_network("10.147.0.0/16"),  # ZeroTier managed
    ipaddress.ip_network("10.244.0.0/16"),  # ZeroTier managed
]
_PRIVATE_NETS = [
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
]


@dataclass
class DetectedInterface:
    ip: str
    interface_type: str  # InterfaceType value
    trusted: bool
    label: str  # Human-readable (e.g. "Tailscale (100.100.1.5)")


@dataclass
class EndpointConfig:
    """Persistierte Endpoint-Konfiguration."""

    enabled_ips: list[str]  # Explizit aktivierte IPs
    auto_detect: bool = True  # Auto-Detection aktiviert
    always_bind_loopback: bool = True  # Loopback immer aktiv


def classify_ip(ip_str: str) -> InterfaceType:
    """Klassifiziert eine IP nach Netzwerk-Typ."""
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        if ip_str == "localhost":
            return InterfaceType.LOOPBACK
        return InterfaceType.UNKNOWN

    if addr.is_loopback:
        return InterfaceType.LOOPBACK
    if addr in _TAILSCALE_NET:
        return InterfaceType.TAILSCALE
    for net in _ZEROTIER_NETS:
        if addr in net:
            return InterfaceType.ZEROTIER
    for net in _PRIVATE_NETS:
        if addr in net:
            return InterfaceType.LAN
    return InterfaceType.UNKNOWN


def detect_interfaces() -> list[DetectedInterface]:
    """Erkennt alle lokalen Netzwerk-Interfaces."""
    seen: set[str] = set()
    result: list[DetectedInterface] = []

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            addr = info[4][0]
            if not isinstance(addr, str):
                continue
            if addr in seen or addr.startswith("fe80"):
                continue
            seen.add(addr)
            itype = classify_ip(addr)
            trusted = itype in (InterfaceType.LOOPBACK, InterfaceType.TAILSCALE)
            label = _make_label(addr, itype)
            result.append(
                DetectedInterface(
                    ip=addr,
                    interface_type=itype.value,
                    trusted=trusted,
                    label=label,
                )
            )
    except (OSError, socket.gaierror):
        pass

    # Ensure loopback is always present
    if "127.0.0.1" not in seen:
        result.insert(
            0,
            DetectedInterface(
                ip="127.0.0.1",
                interface_type=InterfaceType.LOOPBACK.value,
                trusted=True,
                label="Loopback (127.0.0.1)",
            ),
        )

    return result


def _make_label(ip: str, itype: InterfaceType) -> str:
    labels = {
        InterfaceType.LOOPBACK: "Loopback",
        InterfaceType.TAILSCALE: "Tailscale",
        InterfaceType.ZEROTIER: "ZeroTier",
        InterfaceType.VPN: "VPN",
        InterfaceType.LAN: "LAN",
        InterfaceType.UNKNOWN: "Unknown",
    }
    return f"{labels.get(itype, 'Unknown')} ({ip})"


class NetworkEndpointManager:
    """Verwaltet Netzwerk-Endpoints mit Persistenz."""

    def __init__(self, config_path: Path | None = None) -> None:
        self._config_path = config_path or (Path.home() / ".cognithor" / "network_endpoints.json")
        self._config = self._load()

    def get_detected_interfaces(self) -> list[dict[str, Any]]:
        """Alle erkannten Interfaces mit Status."""
        interfaces = detect_interfaces()
        enabled = set(self._config.enabled_ips)
        result = []
        for iface in interfaces:
            result.append(
                {
                    **asdict(iface),
                    "enabled": (
                        iface.ip in enabled
                        or (self._config.auto_detect and iface.trusted)
                        or (
                            self._config.always_bind_loopback
                            and iface.interface_type == InterfaceType.LOOPBACK.value
                        )
                    ),
                }
            )
        return result

    def get_bind_host(self) -> str:
        """Bestimmt den optimalen Bind-Host basierend auf Config.

        Wenn mehrere Interfaces aktiv: 0.0.0.0
        Wenn nur Loopback: 127.0.0.1
        """
        active = self._get_active_ips()
        non_loopback = [ip for ip in active if classify_ip(ip) != InterfaceType.LOOPBACK]
        if non_loopback:
            return "0.0.0.0"
        return "127.0.0.1"

    def get_active_ips(self) -> list[str]:
        """Gibt die aktiven IPs zurueck (fuer trusted-IP-Check)."""
        return self._get_active_ips()

    def set_enabled_ips(self, ips: list[str]) -> None:
        """Setzt die explizit aktivierten IPs."""
        self._config.enabled_ips = list(ips)
        self._save()
        log.info("network_endpoints_updated", enabled=ips)

    def set_auto_detect(self, enabled: bool) -> None:
        """Aktiviert/deaktiviert Auto-Detection."""
        self._config.auto_detect = enabled
        self._save()

    def _get_active_ips(self) -> list[str]:
        interfaces = detect_interfaces()
        active: list[str] = []
        # Explicitly enabled IPs are always active (even if not detected)
        for ip in self._config.enabled_ips:
            if ip not in active:
                active.append(ip)
        # Auto-detected trusted interfaces
        for iface in interfaces:
            if iface.ip in active:
                continue
            if (self._config.auto_detect and iface.trusted) or (
                self._config.always_bind_loopback
                and iface.interface_type == InterfaceType.LOOPBACK.value
            ):
                active.append(iface.ip)
        return active

    def _load(self) -> EndpointConfig:
        if not self._config_path.exists():
            return EndpointConfig(enabled_ips=[], auto_detect=True)
        try:
            data = json.loads(self._config_path.read_text(encoding="utf-8"))
            return EndpointConfig(
                enabled_ips=data.get("enabled_ips", []),
                auto_detect=data.get("auto_detect", True),
                always_bind_loopback=data.get("always_bind_loopback", True),
            )
        except Exception:
            return EndpointConfig(enabled_ips=[], auto_detect=True)

    def _save(self) -> None:
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            self._config_path.write_text(
                json.dumps(asdict(self._config), indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            log.warning("network_endpoints_save_failed", error=str(exc))
