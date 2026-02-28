"""Healthcheck-Endpoint für Monitoring und Deployment.

Bietet einen einfachen HTTP-Endpoint (GET /health) der den
Systemzustand in JSON zurückgibt. Kann von systemd, Docker,
oder Monitoring-Tools verwendet werden.

Bibel-Referenz: §15.5 (systemd + Healthcheck)
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# Start-Zeitpunkt
_start_time = time.monotonic()
_start_datetime = datetime.now(UTC)


def health_status(
    *,
    llm_available: bool = False,
    llm_backend: str = "ollama",
    channels_active: list[str] | None = None,
    memory_stats: dict[str, Any] | None = None,
    models_loaded: list[str] | None = None,
    errors: list[str] | None = None,
    # Rückwärtskompatibilität
    ollama_available: bool | None = None,
) -> dict[str, Any]:
    """Erstellt einen Health-Status-Report.

    Returns:
        Dict mit dem aktuellen Systemzustand:
        {
            "status": "healthy" | "degraded" | "unhealthy",
            "uptime_seconds": int,
            "started_at": "2026-02-22T10:00:00Z",
            "llm_backend": "openai",
            "llm_available": true/false,
            "ollama": true/false,  (backward compat)
            "channels": ["cli", "telegram"],
            "memory": {...},
            "models": ["gpt-5.2"],
            "errors": [],
        }
    """
    # Rückwärtskompatibilität: ollama_available als Alias für llm_available
    if ollama_available is not None and not llm_available:
        llm_available = ollama_available

    uptime = int(time.monotonic() - _start_time)
    error_list = list(errors) if errors else []

    # Status bestimmen
    if not llm_available:
        status = "degraded"
        error_list.append(f"LLM-Backend '{llm_backend}' nicht erreichbar")
    elif error_list:
        status = "degraded"
    else:
        status = "healthy"

    return {
        "status": status,
        "uptime_seconds": uptime,
        "started_at": _start_datetime.isoformat(),
        "timestamp": datetime.now(UTC).isoformat(),
        "llm_backend": llm_backend,
        "llm_available": llm_available,
        "ollama": llm_available if llm_backend == "ollama" else False,  # backward compat
        "channels": channels_active or [],
        "memory": memory_stats or {},
        "models": models_loaded or [],
        "errors": error_list,
    }
