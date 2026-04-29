"""Cognithor · Config-Routes Factory — Orchestrator.

`create_config_routes()` haengt 24 `_register_*_routes()`-Helfer aus den
Sub-Modulen unter `cognithor.channels.config_routes/` an eine FastAPI-App.
Reine Aufruf-Reihenfolge — keine Endpoint-Logik liegt mehr hier; jede
Domaene hat ihr eigenes Sub-Modul (`system`, `config`, `session`, `skills`,
`monitoring`, `security`, `governance`, `evolution`, `infrastructure`,
`ui`, `workflows`, `learning`, `autonomous`, `social`).

Public API: `create_config_routes`, re-exportiert ueber `__init__.py`.

Architektur-Bibel: §12 (Konfiguration), §9.3 (Web UI). Refactor-Plan:
`docs/superpowers/plans/2026-04-29-config-routes-split.md`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cognithor.config_manager import ConfigManager


from cognithor.channels.config_routes.autonomous import (
    _register_autonomous_routes,
    _register_feedback_routes,
)
from cognithor.channels.config_routes.config import _register_config_routes
from cognithor.channels.config_routes.evolution import (
    _register_gepa_evolution_routes,
    _register_prompt_evolution_routes,
    _register_self_improvement_routes,
)
from cognithor.channels.config_routes.governance import _register_governance_routes
from cognithor.channels.config_routes.infrastructure import (
    _register_backend_routes,
    _register_infrastructure_routes,
    _register_portal_routes,
)
from cognithor.channels.config_routes.learning import (
    _register_ingest_routes,
    _register_learning_routes,
)
from cognithor.channels.config_routes.monitoring import (
    _register_monitoring_routes,
    _register_prometheus_routes,
)
from cognithor.channels.config_routes.security import _register_security_routes
from cognithor.channels.config_routes.session import (
    _register_memory_routes,
    _register_session_routes,
)
from cognithor.channels.config_routes.skills import (
    _register_hermes_routes,
    _register_skill_registry_routes,
    _register_skill_routes,
)
from cognithor.channels.config_routes.social import _register_social_routes
from cognithor.channels.config_routes.system import _register_system_routes
from cognithor.channels.config_routes.ui import _register_ui_routes
from cognithor.channels.config_routes.workflows import _register_workflow_graph_routes

# ======================================================================
# Public entry-point
# ======================================================================


def create_config_routes(
    app: Any,
    config_manager: ConfigManager,
    *,
    verify_token_dep: Any = None,
    gateway: Any = None,
) -> None:
    """Registriert Config-API-Endpoints auf einer FastAPI-App.

    Args:
        app: FastAPI-App-Instanz.
        config_manager: ConfigManager fuer Read/Write.
        verify_token_dep: Optional FastAPI Depends() fuer Auth.
        gateway: Optional Gateway-Instanz fuer Singleton-Zugriff.
    """
    deps = [verify_token_dep] if verify_token_dep else []

    # Shared MonitoringHub (singleton per app) -- created lazily and used
    # across monitoring, SSE, and audit routes.
    _hub_holder: dict[str, Any] = {"hub": None}

    def _get_hub() -> Any:
        if _hub_holder["hub"] is None:
            from cognithor.gateway.monitoring import MonitoringHub

            _hub_holder["hub"] = MonitoringHub()
        return _hub_holder["hub"]

    _register_system_routes(app, deps, config_manager, gateway)
    _register_config_routes(app, deps, config_manager, gateway)
    _register_session_routes(app, deps, gateway)
    _register_memory_routes(app, deps, gateway)
    _register_skill_routes(app, deps, gateway)
    _register_monitoring_routes(app, deps, _get_hub, config_manager)
    _register_prometheus_routes(app, _get_hub, gateway)
    _register_security_routes(app, deps, gateway)
    _register_governance_routes(app, deps, gateway)
    _register_prompt_evolution_routes(app, deps, gateway)
    _register_infrastructure_routes(app, deps, gateway)
    _register_portal_routes(app, deps, gateway)
    _register_ui_routes(app, deps, config_manager, gateway)
    _register_workflow_graph_routes(app, deps, gateway)
    _register_learning_routes(app, deps, gateway)
    _register_ingest_routes(app, deps, gateway)
    _register_hermes_routes(app, deps, gateway)
    _register_skill_registry_routes(app, deps, gateway)
    _register_self_improvement_routes(app, deps, gateway)
    _register_gepa_evolution_routes(app, deps, gateway)
    _register_backend_routes(app, deps, config_manager, gateway)
    _register_autonomous_routes(app, deps, gateway)
    _register_feedback_routes(app, deps, gateway)
    _register_social_routes(app, deps, gateway)
