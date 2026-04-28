"""Cognithor · Config-Routes-Paket.

Public API: `create_config_routes()` — registriert alle Config-Endpoints
auf einer FastAPI-App. Backwards-kompatibel zum frueheren
`cognithor.channels.config_routes` Modul (vor 2026-04-29 ein Single-File
mit ~6 600 LOC). Implementation liegt in `_factory.py`; die zugehoerigen
`_register_*_routes()`-Helper wandern schrittweise in Sub-Module.

Refactor-Plan: `docs/superpowers/plans/2026-04-29-config-routes-split.md`.
"""

from __future__ import annotations

from cognithor.channels.config_routes._factory import create_config_routes

__all__ = ["create_config_routes"]
