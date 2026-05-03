"""Netzwerk-Utilities — Interface-Detection, Tailscale, trusted IPs.

Erkennt lokale Netzwerk-Interfaces und bestimmt welche IPs
als vertrauenswuerdig gelten (Loopback, Tailscale CGNAT).
"""

from __future__ import annotations

import ipaddress
import socket

from cognithor.utils.logging import get_logger

log = get_logger(__name__)

# Tailscale CGNAT range: 100.64.0.0/10
TAILSCALE_NETWORK = ipaddress.ip_network("100.64.0.0/10")


def get_local_ips() -> list[str]:
    """Gibt alle lokalen IP-Adressen zurueck."""
    ips: list[str] = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            addr = info[4][0]
            if not isinstance(addr, str):
                continue
            if addr not in ips and not addr.startswith("fe80"):
                ips.append(addr)
    except (OSError, socket.gaierror):
        pass
    return ips


def get_tailscale_ip() -> str | None:
    """Erkennt die Tailscale-IP (100.x.x.x) auf diesem Host.

    Returns:
        Die Tailscale-IP als String oder None.
    """
    for ip in get_local_ips():
        if is_tailscale_ip(ip):
            log.info("tailscale_ip_detected", ip=ip)
            return ip
    return None


def is_tailscale_ip(ip: str) -> bool:
    """Prueft ob eine IP im Tailscale CGNAT-Bereich liegt."""
    try:
        return ipaddress.ip_address(ip) in TAILSCALE_NETWORK
    except ValueError:
        return False


def is_loopback(ip: str) -> bool:
    """Prueft ob eine IP eine Loopback-Adresse ist."""
    try:
        return ipaddress.ip_address(ip).is_loopback
    except ValueError:
        return ip.startswith("127.") or ip == "::1" or ip == "localhost"


def is_trusted_ip(ip: str) -> bool:
    """Prueft ob eine IP vertrauenswuerdig ist.

    Vertrauenswuerdig: Loopback (127.x, ::1) oder Tailscale (100.64/10).
    """
    return is_loopback(ip) or is_tailscale_ip(ip)


def get_smart_bind_hosts(explicit_host: str | None = None) -> list[str]:
    """Bestimmt die optimalen Bind-Adressen.

    Wenn explizit gesetzt: nur diese.
    Sonst: 127.0.0.1 + Tailscale-IP (falls vorhanden).

    Returns:
        Liste von Bind-Adressen.
    """
    if explicit_host:
        return [explicit_host]

    hosts = ["127.0.0.1"]
    ts_ip = get_tailscale_ip()
    if ts_ip:
        hosts.append(ts_ip)
    return hosts


def get_primary_bind_host(explicit_host: str | None = None) -> str:
    """Bestimmt den primaeren Bind-Host fuer den API-Server.

    - Explizit gesetzt → verwenden
    - Tailscale vorhanden → 0.0.0.0 (aber bootstrap nur fuer trusted IPs)
    - Sonst → 127.0.0.1

    Returns:
        Der Bind-Host als String.
    """
    if explicit_host:
        return explicit_host

    ts_ip = get_tailscale_ip()
    if ts_ip:
        # Tailscale vorhanden: bind auf alle Interfaces, aber bootstrap
        # ist durch is_trusted_ip() geschuetzt.
        log.info(
            "smart_bind_tailscale",
            tailscale_ip=ts_ip,
            bind="0.0.0.0",
        )
        return "0.0.0.0"

    return "127.0.0.1"


def get_reachable_url(bind_host: str, port: int) -> str:
    """Bestimmt die beste URL unter der der Server erreichbar ist.

    Fuer QR-Codes und Pairing: nicht 0.0.0.0 anzeigen,
    sondern die konkrete IP.
    """
    if bind_host in ("0.0.0.0", "::"):
        # Bevorzuge Tailscale-IP, dann LAN-IP, dann localhost
        ts_ip = get_tailscale_ip()
        if ts_ip:
            return f"http://{ts_ip}:{port}"
        lan_ips = [ip for ip in get_local_ips() if not is_loopback(ip) and not is_tailscale_ip(ip)]
        if lan_ips:
            return f"http://{lan_ips[0]}:{port}"
    if bind_host in ("127.0.0.1", "localhost", "::1"):
        return f"http://localhost:{port}"
    return f"http://{bind_host}:{port}"
