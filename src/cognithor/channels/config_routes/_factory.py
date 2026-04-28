"""Cognithor · Config-Routes Factory — wird schrittweise in Sub-Module aufgeteilt.

Dieses Modul enthaelt aktuell die komplette `create_config_routes()`-Funktion
sowie alle 24 `_register_*_routes()`-Helper. Im Rahmen des Refactor-Plans
(siehe `docs/superpowers/plans/2026-04-29-config-routes-split.md`) wandern die
Helper schrittweise in eigene Sub-Module unter `cognithor.channels.config_routes/`.
Bis dahin bleiben sie hier — Public-API ist `create_config_routes()`, re-exportiert
ueber `cognithor.channels.config_routes.__init__`.

REST-Endpoints fuer die Konfigurationsverwaltung via WebUI:

  - GET/PATCH /api/v1/config          → Gesamte Konfiguration
  - GET/PATCH /api/v1/config/{section} → Einzelne Sektion
  - GET/POST/DELETE /api/v1/agents     → Agent-Verwaltung
  - GET/POST/DELETE /api/v1/credentials → Credential-Verwaltung
  - GET /api/v1/status                  → System-Status Dashboard

Architektur-Bibel: §12 (Konfiguration), §9.3 (Web UI)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import TYPE_CHECKING, Any

import yaml

try:
    from starlette.requests import Request
except ImportError:
    Request = Any  # type: ignore[assignment,misc]

try:
    from fastapi import HTTPException
except ImportError:
    try:
        from starlette.exceptions import HTTPException  # type: ignore[assignment]
    except ImportError:
        HTTPException = Exception  # type: ignore[assignment,misc]

from pathlib import Path

from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.config_manager import ConfigManager

log = get_logger(__name__)


from cognithor.channels.config_routes.config import _register_config_routes
from cognithor.channels.config_routes.governance import _register_governance_routes
from cognithor.channels.config_routes.monitoring import (
    _register_monitoring_routes,
    _register_prometheus_routes,
)
from cognithor.channels.config_routes.security import _register_security_routes
from cognithor.channels.config_routes.session import (
    _register_memory_routes,
    _register_session_routes,
)
from cognithor.channels.config_routes.system import _register_system_routes

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


# ======================================================================
# Skill management routes
# ======================================================================


def _register_skill_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """Marketplace, updater, commands, skill-CLI, connectors, workflows,
    models, i18n, setup-wizard."""

    # -- Marketplace ------------------------------------------------------

    @app.get("/api/v1/marketplace/feed", dependencies=deps)
    async def marketplace_feed() -> dict[str, Any]:
        """Kuratierter Feed fuer die Startseite."""
        try:
            from cognithor.skills.marketplace import SkillMarketplace

            return SkillMarketplace().curated_feed()
        except Exception as exc:
            log.error("marketplace_feed_failed", error=str(exc))
            return {"error": "Marketplace-Feed nicht verfuegbar"}

    @app.get("/api/v1/marketplace/search", dependencies=deps)
    async def marketplace_search(
        q: str = "",
        category: str = "",
        verified_only: bool = False,
        sort_by: str = "relevance",
        max_results: int = 20,
    ) -> dict[str, Any]:
        """Durchsucht den Skill-Marktplatz."""
        try:
            from cognithor.skills.marketplace import SkillMarketplace

            mp = SkillMarketplace()
            results = mp.search(
                query=q,
                category=category,
                verified_only=verified_only,
                sort_by=sort_by,
                max_results=max_results,
            )
            return {"results": [r.to_dict() for r in results], "count": len(results)}
        except Exception as exc:
            log.error("marketplace_search_failed", error=str(exc))
            return {"error": "Marketplace-Suche fehlgeschlagen"}

    @app.get("/api/v1/marketplace/categories", dependencies=deps)
    async def marketplace_categories() -> dict[str, Any]:
        """Alle Skill-Kategorien mit Counts."""
        try:
            from cognithor.skills.marketplace import SkillMarketplace

            return {"categories": [c.to_dict() for c in SkillMarketplace().categories()]}
        except Exception as exc:
            log.error("marketplace_categories_failed", error=str(exc))
            return {"error": "Kategorien nicht verfuegbar"}

    @app.get("/api/v1/marketplace/featured", dependencies=deps)
    async def marketplace_featured(n: int = 10) -> dict[str, Any]:
        """Featured-Skills."""
        try:
            from cognithor.skills.marketplace import SkillMarketplace

            return {"featured": [s.to_dict() for s in SkillMarketplace().featured(n)]}
        except Exception as exc:
            log.error("marketplace_featured_failed", error=str(exc))
            return {"error": "Featured-Skills nicht verfuegbar"}

    @app.get("/api/v1/marketplace/trending", dependencies=deps)
    async def marketplace_trending(window: str = "24h", n: int = 10) -> dict[str, Any]:
        """Trending-Skills."""
        try:
            from cognithor.skills.marketplace import SkillMarketplace

            return {"trending": [s.to_dict() for s in SkillMarketplace().trending(max_results=n)]}
        except Exception as exc:
            log.error("marketplace_trending_failed", error=str(exc))
            return {"error": "Trending-Skills nicht verfuegbar"}

    @app.get("/api/v1/marketplace/stats", dependencies=deps)
    async def marketplace_stats() -> dict[str, Any]:
        """Marktplatz-Statistiken."""
        try:
            from cognithor.skills.marketplace import SkillMarketplace

            return SkillMarketplace().stats()
        except Exception as exc:
            log.error("marketplace_stats_failed", error=str(exc))
            return {"error": "Marketplace-Statistiken nicht verfuegbar"}

    # -- Skill-Updater ----------------------------------------------------

    @app.get("/api/v1/updater/stats", dependencies=deps)
    async def updater_stats() -> dict[str, Any]:
        """Skill-Updater-Statistiken."""
        try:
            from cognithor.skills.updater import SkillUpdater

            return SkillUpdater().stats()
        except Exception as exc:
            log.error("updater_stats_failed", error=str(exc))
            return {"error": "Updater-Statistiken nicht verfuegbar"}

    @app.get("/api/v1/updater/pending", dependencies=deps)
    async def updater_pending() -> dict[str, Any]:
        """Ausstehende Updates."""
        try:
            from cognithor.skills.updater import SkillUpdater

            u = SkillUpdater()
            return {"updates": [c.to_dict() for c in u.pending_updates()]}
        except Exception as exc:
            log.error("updater_pending_failed", error=str(exc))
            return {"error": "Ausstehende Updates nicht verfuegbar"}

    @app.get("/api/v1/updater/recalls", dependencies=deps)
    async def updater_recalls() -> dict[str, Any]:
        """Aktive Security-Recalls."""
        try:
            from cognithor.skills.updater import SkillUpdater

            u = SkillUpdater()
            return {"recalls": [r.to_dict() for r in u.active_recalls()]}
        except Exception as exc:
            log.error("updater_recalls_failed", error=str(exc))
            return {"error": "Recalls nicht verfuegbar"}

    @app.get("/api/v1/updater/history", dependencies=deps)
    async def updater_history(n: int = 20) -> dict[str, Any]:
        """Update-Historie."""
        try:
            from cognithor.skills.updater import SkillUpdater

            return {"history": SkillUpdater().update_history(n)}
        except Exception as exc:
            log.error("updater_history_failed", error=str(exc))
            return {"error": "Update-Historie nicht verfuegbar"}

    # -- Commands ---------------------------------------------------------

    @app.get("/api/v1/commands/list", dependencies=deps)
    async def list_commands() -> dict[str, Any]:
        """Alle registrierten Slash-Commands."""
        try:
            from cognithor.channels.commands import CommandRegistry

            reg = CommandRegistry()
            return {
                "commands": [c.to_dict() for c in reg.list_commands()],
                "count": reg.command_count,
            }
        except Exception as exc:
            log.error("commands_list_failed", error=str(exc))
            return {"error": "Commands konnten nicht geladen werden"}

    @app.get("/api/v1/commands/slack", dependencies=deps)
    async def commands_slack() -> dict[str, Any]:
        """Slack Slash-Command-Definitionen."""
        try:
            from cognithor.channels.commands import CommandRegistry

            return {"definitions": CommandRegistry().slack_definitions()}
        except Exception as exc:
            log.error("commands_slack_failed", error=str(exc))
            return {"error": "Slack-Commands nicht verfuegbar"}

    @app.get("/api/v1/commands/discord", dependencies=deps)
    async def commands_discord() -> dict[str, Any]:
        """Discord Application-Command-Definitionen."""
        try:
            from cognithor.channels.commands import CommandRegistry

            return {"definitions": CommandRegistry().discord_definitions()}
        except Exception as exc:
            log.error("commands_discord_failed", error=str(exc))
            return {"error": "Discord-Commands nicht verfuegbar"}

    # -- Connectors -------------------------------------------------------

    @app.get("/api/v1/connectors/list", dependencies=deps)
    async def list_connectors() -> dict[str, Any]:
        """Alle registrierten Konnektoren."""
        reg = getattr(gateway, "_connector_registry", None)
        if reg is None:
            return {"connectors": [], "count": 0}
        return {"connectors": reg.list_connectors(), "count": reg.connector_count}

    @app.get("/api/v1/connectors/stats", dependencies=deps)
    async def connector_stats() -> dict[str, Any]:
        """Konnektor-Statistiken."""
        reg = getattr(gateway, "_connector_registry", None)
        if reg is None:
            return {
                "total_connectors": 0,
                "connectors": [],
                "scope_guard": {"policies": 0, "violations": 0},
            }
        return reg.stats()

    # -- Workflows (categories + legacy start — main endpoints in _register_workflow_graph_routes)

    @app.get("/api/v1/workflows/templates/categories", dependencies=deps)
    async def workflow_categories() -> dict[str, Any]:
        """Workflow-Kategorien."""
        lib = getattr(gateway, "_template_library", None)
        if lib is None:
            return {"categories": []}
        return {"categories": lib.categories()}

    @app.post("/api/v1/workflows/start", dependencies=deps)
    async def workflow_start(request: Request) -> dict[str, Any]:
        """Workflow-Instanz starten (legacy endpoint)."""
        try:
            engine = getattr(gateway, "_workflow_engine", None)
            lib = getattr(gateway, "_template_library", None)
            if engine is None or lib is None:
                return {"error": "Workflow-Engine nicht verfügbar"}
            body = await request.json()
            template_id = body.get("template_id", "")
            template = lib.get(template_id)
            if not template:
                return {"error": f"Template nicht gefunden: {template_id}"}
            inst = engine.start(template, created_by=body.get("created_by", ""))
            return inst.to_dict()
        except Exception as exc:
            log.error("workflow_start_failed", error=str(exc))
            return {"error": "Workflow konnte nicht gestartet werden"}

    # -- Models -----------------------------------------------------------

    @app.get("/api/v1/models/list", dependencies=deps)
    async def model_list() -> dict[str, Any]:
        """Alle registrierten ML-Modelle."""
        reg = getattr(gateway, "_model_registry", None)
        if reg is None:
            return {"models": [], "count": 0}
        return {"models": reg.list_all(), "count": reg.model_count}

    @app.get("/api/v1/models/available", dependencies=deps)
    async def available_models() -> dict[str, Any]:
        """Listet alle in Ollama/Backend verfuegbaren Modelle auf."""
        router = getattr(gateway, "_model_router", None)
        if router is None:
            return {"models": [], "source": "none"}
        # Refresh the model list
        with contextlib.suppress(Exception):
            await router.initialize()
        models = sorted(router._available_models) if router._available_models else []
        # Also return currently configured models for reference
        cfg = getattr(gateway, "_config", None)
        if cfg is None:
            return {
                "models": models,
                "configured": {},
                "warnings": [],
                "source": "backend" if router._backend else "ollama",
            }
        configured = {
            "planner": cfg.models.planner.name,
            "executor": cfg.models.executor.name,
            "coder": cfg.models.coder.name,
            "embedding": cfg.models.embedding.name,
        }
        warnings = []
        for role, name in configured.items():
            if models and name not in models:
                warnings.append(
                    f"Modell '{name}' ({role}) ist nicht verfügbar. "
                    f"Installieren: ollama pull {name}"
                )
        return {
            "models": models,
            "configured": configured,
            "warnings": warnings,
            "source": "backend" if router._backend else "ollama",
        }

    @app.get("/api/v1/models/stats", dependencies=deps)
    async def model_stats() -> dict[str, Any]:
        """Model-Registry Statistiken."""
        reg = getattr(gateway, "_model_registry", None)
        if reg is None:
            return {"total_models": 0, "providers": [], "capabilities": [], "languages": []}
        return reg.stats()

    # -- i18n -------------------------------------------------------------

    @app.get("/api/v1/i18n/locales", dependencies=deps)
    async def i18n_locales() -> dict[str, Any]:
        """Verfuegbare Sprachen."""
        from cognithor.i18n import get_available_locales, get_locale

        return {"locales": get_available_locales(), "default": get_locale()}

    @app.get("/api/v1/i18n/translate/{key}", dependencies=deps)
    async def i18n_translate(key: str, locale: str = "") -> dict[str, Any]:
        """Einzelnen Key uebersetzen."""
        from cognithor.i18n import get_locale, set_locale, t

        if locale and locale != get_locale():
            prev = get_locale()
            set_locale(locale)
            result = t(key)
            set_locale(prev)
            return {"key": key, "translation": result}
        return {"key": key, "translation": t(key)}

    @app.get("/api/v1/i18n/stats", dependencies=deps)
    async def i18n_stats() -> dict[str, Any]:
        """i18n-Statistiken."""
        from cognithor.i18n import get_available_locales, get_locale

        locales = get_available_locales()
        return {"default_locale": get_locale(), "locale_count": len(locales), "locales": locales}

    # -- Skill-CLI (Phase 35) ---------------------------------------------

    @app.get("/api/v1/skill-cli/stats", dependencies=deps)
    async def skill_cli_stats() -> dict[str, Any]:
        """Skill-CLI Statistiken."""
        cli = getattr(gateway, "_skill_cli", None)
        if cli is None:
            return {"scaffolder": {"templates": 0}}
        return cli.stats()

    @app.get("/api/v1/skill-cli/templates", dependencies=deps)
    async def skill_cli_templates() -> dict[str, Any]:
        """Verfuegbare Skill-Templates."""
        cli = getattr(gateway, "_skill_cli", None)
        if cli is None:
            return {"templates": []}
        return {"templates": cli.scaffolder.available_templates()}

    @app.get("/api/v1/skill-cli/rewards", dependencies=deps)
    async def skill_cli_rewards() -> dict[str, Any]:
        """Reward-System Statistiken."""
        cli = getattr(gateway, "_skill_cli", None)
        if cli is None:
            return {"contributors": 0}
        return cli.rewards.stats()

    # -- Setup-Wizard (Phase 36) ------------------------------------------

    @app.get("/api/v1/setup/state", dependencies=deps)
    async def setup_state() -> dict[str, Any]:
        """Setup-Wizard Status."""
        wiz = getattr(gateway, "_setup_wizard", None)
        if wiz is None:
            return {"step": "unavailable"}
        return wiz.state.to_dict()

    @app.get("/api/v1/setup/stats", dependencies=deps)
    async def setup_stats() -> dict[str, Any]:
        """Setup-Wizard Statistiken."""
        wiz = getattr(gateway, "_setup_wizard", None)
        if wiz is None:
            return {"state": {}}
        return wiz.stats()


# ======================================================================
# Prompt-Evolution routes
# ======================================================================


def _register_prompt_evolution_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """Stats, manual evolve trigger, and enable/disable toggle."""

    @app.get("/api/v1/prompt-evolution/stats", dependencies=deps)
    async def prompt_evolution_stats() -> dict[str, Any]:
        engine = getattr(gateway, "_prompt_evolution", None)
        enabled = engine is not None
        stats: dict[str, Any] = {"enabled": enabled}
        if engine:
            with contextlib.suppress(Exception):
                stats.update(engine.get_stats("system_prompt"))
        return stats

    @app.post("/api/v1/prompt-evolution/evolve", dependencies=deps)
    async def prompt_evolution_evolve() -> dict[str, Any]:
        engine = getattr(gateway, "_prompt_evolution", None)
        if engine is None:
            return {"error": "prompt_evolution is disabled"}
        # Check ImprovementGate
        gate = getattr(gateway, "_improvement_gate", None)
        if gate is not None:
            from cognithor.governance.improvement_gate import GateVerdict, ImprovementDomain

            verdict = gate.check(ImprovementDomain.PROMPT_TUNING)
            if verdict != GateVerdict.ALLOWED:
                return {"error": f"gate_blocked: {verdict.value}"}
        try:
            result = await engine.maybe_evolve("system_prompt")
            return {"evolved": result is not None, "version_id": result}
        except Exception as exc:
            log.error("prompt_evolution_evolve_failed", error=str(exc))
            return {"error": "Prompt-Evolution fehlgeschlagen"}

    @app.post("/api/v1/prompt-evolution/toggle", dependencies=deps)
    async def prompt_evolution_toggle(request: Request) -> dict[str, Any]:
        body = await request.json()
        enabled = body.get("enabled", False)

        if enabled:
            if getattr(gateway, "_prompt_evolution", None) is None:
                try:
                    from cognithor.learning.prompt_evolution import PromptEvolutionEngine

                    cfg = gateway._config
                    pe_db = str(cfg.db_path.with_name("memory_prompt_evolution.db"))
                    engine = PromptEvolutionEngine(
                        db_path=pe_db,
                        min_sessions_per_arm=cfg.prompt_evolution.min_sessions_per_arm,
                        significance_threshold=cfg.prompt_evolution.significance_threshold,
                        max_concurrent_tests=cfg.prompt_evolution.max_concurrent_tests,
                    )
                    engine.set_evolution_interval_hours(
                        cfg.prompt_evolution.evolution_interval_hours
                    )
                    gateway._prompt_evolution = engine
                    planner = getattr(gateway, "_planner", None)
                    if planner:
                        planner._prompt_evolution = engine
                except Exception as exc:
                    log.error("prompt_evolution_toggle_failed", error=str(exc))
                    return {
                        "error": "Prompt-Evolution konnte nicht aktiviert werden",
                        "enabled": False,
                    }
        else:
            # Disable: disconnect from planner but keep engine for stats
            planner = getattr(gateway, "_planner", None)
            if planner:
                planner._prompt_evolution = None
            gateway._prompt_evolution = None

        return {"enabled": getattr(gateway, "_prompt_evolution", None) is not None}


# ======================================================================
# Infrastructure routes (ecosystem, performance, portal)
# ======================================================================


def _register_infrastructure_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """Ecosystem control, performance manager."""

    # -- Ecosystem-Kontrolle (Phase 28) -----------------------------------

    @app.get("/api/v1/ecosystem/stats", dependencies=deps)
    async def ecosystem_stats() -> dict[str, Any]:
        """Ecosystem-Controller Statistiken."""
        ctrl = getattr(gateway, "_ecosystem_controller", None)
        if ctrl is None:
            return {"curator": {}, "fraud": {}}
        return ctrl.stats()

    @app.get("/api/v1/ecosystem/curator", dependencies=deps)
    async def ecosystem_curator() -> dict[str, Any]:
        """Kuration-Statistiken."""
        ctrl = getattr(gateway, "_ecosystem_controller", None)
        if ctrl is None:
            return {"total_reviews": 0}
        return ctrl.curator.stats()

    @app.get("/api/v1/ecosystem/fraud", dependencies=deps)
    async def ecosystem_fraud() -> dict[str, Any]:
        """Fraud-Detection Statistiken."""
        ctrl = getattr(gateway, "_ecosystem_controller", None)
        if ctrl is None:
            return {"total_signals": 0}
        return ctrl.fraud.stats()

    @app.get("/api/v1/ecosystem/training", dependencies=deps)
    async def ecosystem_training() -> dict[str, Any]:
        """Security-Training Status."""
        ctrl = getattr(gateway, "_ecosystem_controller", None)
        if ctrl is None:
            return {"total_modules": 0}
        return ctrl.trainer.stats()

    @app.get("/api/v1/ecosystem/trust", dependencies=deps)
    async def ecosystem_trust() -> dict[str, Any]:
        """Trust-Boundary Statistiken."""
        ctrl = getattr(gateway, "_ecosystem_controller", None)
        if ctrl is None:
            return {"total_boundaries": 0}
        return ctrl.trust.stats()

    # -- Performance-Manager (Phase 37) -----------------------------------

    @app.get("/api/v1/performance/health", dependencies=deps)
    async def perf_health() -> dict[str, Any]:
        """Performance Health-Status."""
        pm = getattr(gateway, "_perf_manager", None)
        if pm is None:
            return {"vector_store": {"entries": 0}}
        return pm.health()

    @app.get("/api/v1/performance/latency", dependencies=deps)
    async def perf_latency() -> dict[str, Any]:
        """Latenz-Statistiken."""
        pm = getattr(gateway, "_perf_manager", None)
        if pm is None:
            return {"total_samples": 0}
        return pm.latency.stats()

    @app.get("/api/v1/performance/resources", dependencies=deps)
    async def perf_resources() -> dict[str, Any]:
        """Ressourcen-Auslastung."""
        pm = getattr(gateway, "_perf_manager", None)
        if pm is None:
            return {"snapshots": 0}
        return pm.optimizer.stats()


# ======================================================================
# Portal routes (end-user portal)
# ======================================================================


def _register_portal_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """End-user portal, consent management."""

    @app.get("/api/v1/portal/stats", dependencies=deps)
    async def portal_stats() -> dict[str, Any]:
        """Endnutzer-Portal Statistiken."""
        up = getattr(gateway, "_user_portal", None)
        if up is None:
            return {"consents": {"total_users": 0}}
        return up.stats()

    @app.get("/api/v1/portal/consents", dependencies=deps)
    async def portal_consents() -> dict[str, Any]:
        """Consent-Management Status."""
        up = getattr(gateway, "_user_portal", None)
        if up is None:
            return {"total_users": 0}
        return up.consents.stats()


# ======================================================================
# UI-specific routes (Control Center frontend)
# ======================================================================


def _register_ui_routes(
    app: Any,
    deps: list[Any],
    config_manager: ConfigManager,
    gateway: Any,
) -> None:
    """Endpoints consumed by the Cognithor Control Center React UI.

    Covers system lifecycle, agent/binding persistence, prompts,
    cron-jobs, MCP servers, and A2A configuration.
    """

    cognithor_home = config_manager.config.cognithor_home
    agents_path = cognithor_home / "agents.yaml"
    bindings_path = cognithor_home / "bindings.yaml"

    def _load_yaml(path: Path) -> Any:
        if not path.exists():
            return {}
        try:
            return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}

    def _save_yaml(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.dump(data, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )

    # -- 3.1: System Status -----------------------------------------------

    @app.get("/api/v1/system/status", dependencies=deps)
    async def ui_system_status() -> dict[str, Any]:
        """Returns running status for the UI status indicator."""
        return {
            "status": "running",
            "timestamp": time.time(),
            "config_version": config_manager.config.version,
            "owner": config_manager.config.owner_name,
        }

    # -- 3.2: System Start / Stop ----------------------------------------

    @app.post("/api/v1/system/start", dependencies=deps)
    async def ui_system_start() -> dict[str, Any]:
        """Reloads configuration (logical start)."""
        try:
            config_manager.reload()
            return {"status": "ok", "message": "System gestartet (Config neu geladen)"}
        except Exception as exc:
            log.error("system_start_failed", error=str(exc))
            return {"error": "System konnte nicht gestartet werden", "status": 500}

    @app.post("/api/v1/system/stop", dependencies=deps)
    async def ui_system_stop() -> dict[str, Any]:
        """Initiates graceful shutdown if gateway is available."""
        try:
            if gateway is not None and hasattr(gateway, "shutdown"):
                asyncio.create_task(gateway.shutdown())  # noqa: RUF006
                return {"status": "ok", "message": "Shutdown eingeleitet"}
            return {"status": "ok", "message": "Kein Gateway — nur Config-Server aktiv"}
        except Exception as exc:
            log.error("system_stop_failed", error=str(exc))
            return {"error": "Shutdown fehlgeschlagen", "status": 500}

    # -- System Detector (hardware profiling) -----------------------------

    @app.get("/api/v1/system/profile", dependencies=deps)
    async def get_system_profile() -> dict[str, Any]:
        """Get hardware/software system profile."""
        profile = getattr(gateway, "_system_profile", None)
        if not profile:
            return {"error": "System profile not available"}
        return profile.to_dict()

    @app.post("/api/v1/system/rescan", dependencies=deps)
    async def rescan_system() -> dict[str, Any]:
        """Force a full system re-scan."""
        try:
            from cognithor.system.detector import SystemDetector

            detector = SystemDetector()
            profile = detector.run_full_scan()
            cache = config_manager.config.cognithor_home / "system_profile.json"
            profile.save(cache)
            if gateway:
                gateway._system_profile = profile
            return profile.to_dict()
        except Exception as exc:
            return {"error": str(exc)}

    # -- Per-Agent Budget + Resource Monitor (Phase 3) --------------------

    @app.get("/api/v1/budget/agents", dependencies=deps)
    async def get_agent_budgets() -> dict[str, Any]:
        """Per-agent cost breakdown and budget status."""
        tracker = getattr(gateway, "_cost_tracker", None)
        if not tracker:
            return {"agents": {}, "message": "Cost tracking not available"}
        costs_today = tracker.get_agent_costs(days=1)
        costs_week = tracker.get_agent_costs(days=7)
        costs_month = tracker.get_agent_costs(days=30)
        # Check budgets from config
        evo_config = getattr(config_manager.config, "evolution", None)
        agent_budgets = getattr(evo_config, "agent_budgets", {}) if evo_config else {}
        statuses = {}
        for agent_name in set(list(costs_today.keys()) + list(agent_budgets.keys())):
            limit = agent_budgets.get(agent_name, 0.0)
            status = tracker.check_agent_budget(agent_name, limit)
            statuses[agent_name] = {
                "daily_cost_usd": status.daily_cost_usd,
                "daily_limit_usd": status.daily_limit_usd,
                "ok": status.ok,
                "warning": status.warning,
            }
        return {
            "agents_today": costs_today,
            "agents_week": costs_week,
            "agents_month": costs_month,
            "budgets": statuses,
        }

    @app.get("/api/v1/system/resources", dependencies=deps)
    async def get_system_resources() -> dict[str, Any]:
        """Current system resource usage (CPU/RAM/GPU)."""
        monitor = getattr(gateway, "_resource_monitor", None)
        if not monitor:
            return {"error": "Resource monitor not available"}
        try:
            snap = await monitor.sample()
            return snap.to_dict()
        except Exception as exc:
            return {"error": str(exc)}

    @app.get("/api/v1/evolution/stats", dependencies=deps)
    async def get_evolution_stats() -> dict[str, Any]:
        """Evolution loop statistics including resource, budget, and checkpoint info."""
        loop = getattr(gateway, "_evolution_loop", None)
        if not loop:
            return {"enabled": False, "message": "Evolution loop not active"}
        stats = loop.stats()
        # Enrich with resume state if checkpoint store available
        store = getattr(gateway, "_checkpoint_store", None)
        if store:
            try:
                from cognithor.evolution.resume import EvolutionResumer

                resumer = EvolutionResumer(store)
                latest = resumer.get_latest_cycle_id()
                if latest is not None:
                    rs = resumer.get_resume_state(latest)
                    stats["resume"] = rs.to_dict()
                else:
                    stats["resume"] = None
            except Exception:
                stats["resume"] = None
        return stats

    @app.post("/api/v1/evolution/resume", dependencies=deps)
    async def resume_evolution_cycle() -> dict[str, Any]:
        """Manually resume the last interrupted evolution cycle."""
        loop = getattr(gateway, "_evolution_loop", None)
        store = getattr(gateway, "_checkpoint_store", None)
        if not loop:
            return {"error": "Evolution loop not active"}
        if not store:
            return {"error": "Checkpoint store not available"}
        try:
            from cognithor.evolution.resume import EvolutionResumer

            resumer = EvolutionResumer(store)
            latest = resumer.get_latest_cycle_id()
            if latest is None:
                return {"error": "No checkpoints found", "resumed": False}
            state = resumer.get_resume_state(latest)
            if state.is_complete:
                return {
                    "resumed": False,
                    "reason": "Cycle already complete",
                    "cycle_id": latest,
                }
            if not state.has_checkpoint:
                return {"resumed": False, "reason": "No checkpoint to resume from"}
            # Trigger a new cycle (the loop will pick up from where it left off)
            result = await loop.run_cycle()
            return {
                "resumed": True,
                "cycle_id": result.cycle_id,
                "steps_completed": result.steps_completed,
                "skill_created": result.skill_created,
            }
        except Exception as exc:
            return {"error": str(exc), "resumed": False}

    @app.get("/api/v1/evolution/goals", dependencies=deps)
    async def get_evolution_goals() -> dict[str, Any]:
        """Get user-defined learning goals as structured objects."""
        # Primary source: GoalManager (has real-time progress from Evolution Engine)
        evo_loop = getattr(gateway, "_evolution_loop", None)
        gm = getattr(evo_loop, "_goal_manager", None) if evo_loop else None
        if gm is not None:
            from dataclasses import asdict

            return {"goals": [asdict(g) for g in gm._goals.values()]}

        # Fallback: config.yaml (no live progress)
        evo = getattr(config_manager.config, "evolution", None)
        raw = getattr(evo, "learning_goals", []) if evo else []
        goals = []
        for i, g in enumerate(raw):
            if isinstance(g, str):
                goals.append(
                    {
                        "id": f"goal_{i}",
                        "title": g,
                        "description": "",
                        "status": "active",
                        "priority": 3,
                        "progress": 0.0,
                    }
                )
            elif isinstance(g, dict):
                g.setdefault("id", f"goal_{i}")
                g.setdefault("status", "active")
                g.setdefault("priority", 3)
                g.setdefault("progress", 0.0)
                g.setdefault("description", "")
                goals.append(g)
        return {"goals": goals}

    def _save_goals(goals: list[dict[str, Any]]) -> None:
        config_manager.update_section("evolution", {"learning_goals": goals})
        config_manager.save()
        loop = getattr(gateway, "_evolution_loop", None)
        if loop and loop._config:
            loop._config.learning_goals = [g.get("title", "") for g in goals if isinstance(g, dict)]

    @app.put("/api/v1/evolution/goals", dependencies=deps)
    async def set_evolution_goals(request: Request) -> dict[str, Any]:
        """Replace all learning goals."""
        try:
            body = await request.json()
            goals = body.get("goals", [])
            if not isinstance(goals, list):
                return {"error": "goals must be a list"}
            # Normalize strings to dicts
            normalized = []
            for i, g in enumerate(goals):
                if isinstance(g, str):
                    normalized.append(
                        {
                            "id": f"goal_{i}",
                            "title": g,
                            "description": "",
                            "status": "active",
                            "priority": 3,
                            "progress": 0.0,
                        }
                    )
                elif isinstance(g, dict):
                    normalized.append(g)
            _save_goals(normalized)
            return {"goals": normalized, "count": len(normalized)}
        except Exception as exc:
            return {"error": str(exc)}

    @app.post("/api/v1/evolution/goals", dependencies=deps)
    async def add_evolution_goal(request: Request) -> dict[str, Any]:
        """Add a single learning goal."""
        try:
            body = await request.json()
            title = body.get("title", "").strip()
            if not title:
                return {"error": "title is required"}
            # Load existing
            evo = getattr(config_manager.config, "evolution", None)
            raw = getattr(evo, "learning_goals", []) if evo else []
            existing = []
            for i, g in enumerate(raw):
                if isinstance(g, str):
                    existing.append(
                        {
                            "id": f"goal_{i}",
                            "title": g,
                            "description": "",
                            "status": "active",
                            "priority": 3,
                            "progress": 0.0,
                        }
                    )
                elif isinstance(g, dict):
                    existing.append(g)
            # Add new
            import uuid

            new_goal = {
                "id": uuid.uuid4().hex[:12],
                "title": title,
                "description": body.get("description", ""),
                "status": "active",
                "priority": body.get("priority", 3),
                "progress": 0.0,
            }
            existing.append(new_goal)
            _save_goals(existing)
            return new_goal
        except Exception as exc:
            import traceback

            traceback.print_exc()
            return {"error": str(exc)}

    @app.patch("/api/v1/evolution/goals/{goal_id}", dependencies=deps)
    async def update_evolution_goal(goal_id: str, request: Request) -> dict[str, Any]:
        """Update a single learning goal."""
        try:
            body = await request.json()
            evo = getattr(config_manager.config, "evolution", None)
            raw = getattr(evo, "learning_goals", []) if evo else []
            goals = []
            for i, g in enumerate(raw):
                if isinstance(g, str):
                    goals.append(
                        {
                            "id": f"goal_{i}",
                            "title": g,
                            "description": "",
                            "status": "active",
                            "priority": 3,
                            "progress": 0.0,
                        }
                    )
                elif isinstance(g, dict):
                    goals.append(g)
            updated = None
            for g in goals:
                if g.get("id") == goal_id:
                    for k, v in body.items():
                        g[k] = v
                    updated = g
                    break
            if updated is None:
                return {"error": "Goal not found"}
            _save_goals(goals)
            return updated
        except Exception as exc:
            return {"error": str(exc)}

    @app.delete("/api/v1/evolution/goals/{goal_id}", dependencies=deps)
    async def delete_evolution_goal(goal_id: str) -> dict[str, Any]:
        """Delete a single learning goal."""
        try:
            evo = getattr(config_manager.config, "evolution", None)
            raw = getattr(evo, "learning_goals", []) if evo else []
            goals = [g for g in raw if isinstance(g, dict)]
            filtered = [g for g in goals if g.get("id") != goal_id]
            if len(filtered) == len(goals):
                return {"error": "Goal not found"}
            _save_goals(filtered)
            return {"deleted": goal_id}
        except Exception as exc:
            return {"error": str(exc)}

    @app.get("/api/v1/evolution/journal", dependencies=deps)
    async def get_evolution_journal(days: int = 7) -> dict[str, Any]:
        """Get evolution journal from recent cycle results and vault entries."""
        try:
            from datetime import UTC, datetime, timedelta
            from pathlib import Path as _P

            lines: list[str] = []
            cutoff = datetime.now(UTC) - timedelta(days=days)

            # Source 1: Evolution vault entries
            vault_dir = _P(config_manager.config.cognithor_home) / "vault" / "wissen"
            if vault_dir.exists():
                for f in sorted(
                    vault_dir.glob("*.md"), key=lambda x: x.stat().st_mtime, reverse=True
                )[:30]:
                    mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=UTC)
                    if mtime < cutoff:
                        continue
                    title = f.stem.replace("-", " ")[:80]
                    size_kb = f.stat().st_size / 1024
                    lines.append(
                        f"[{mtime.strftime('%Y-%m-%d %H:%M')}]"
                        f" Researched: {title} ({size_kb:.1f} KB)"
                    )

            # Source 2: Evolution loop stats
            evo_loop = getattr(gateway, "_evolution_loop", None)
            if evo_loop:
                stats = evo_loop.stats()
                lines.insert(0, "## Evolution Engine Status")
                lines.insert(1, f"- Cycles today: {stats.get('cycles_today', 0)}")
                lines.insert(2, f"- Total cycles: {stats.get('total_cycles', 0)}")
                lines.insert(3, f"- Status: {'Running' if stats.get('running') else 'Stopped'}")
                lines.insert(4, "")

            # Source 3: Deep learner plan progress
            dl = getattr(gateway, "_deep_learner", None)
            if dl:
                plans = dl.list_plans()
                if plans:
                    lines.append("")
                    lines.append("## Learning Plans Progress")
                    for p in plans:
                        passed = sum(
                            1 for sg in p.sub_goals if sg.status in ("verified", "completed")
                        )
                        total = len(p.sub_goals)
                        lines.append(f"- {p.goal[:60]}: {passed}/{total} sub-goals")

            content = "\n".join(lines) if lines else ""
            return {"content": content}
        except Exception as exc:
            log.error("evolution_journal_failed", error=str(exc))
            return {"content": "", "error": str(exc)}

    @app.get("/api/v1/evolution/plans", dependencies=deps)
    async def list_evolution_plans() -> dict[str, Any]:
        """List all learning plans."""
        dl = getattr(gateway, "_deep_learner", None)
        if not dl:
            return {"plans": [], "message": "DeepLearner not available"}
        plans = dl.list_plans()
        return {"plans": [p.to_summary_dict() for p in plans]}

    @app.get("/api/v1/evolution/plans/{plan_id}", dependencies=deps)
    async def get_evolution_plan(plan_id: str) -> dict[str, Any]:
        """Get detailed plan with SubGoals."""
        dl = getattr(gateway, "_deep_learner", None)
        if not dl:
            return {"error": "DeepLearner not available"}
        plan = dl.get_plan(plan_id)
        if not plan:
            return {"error": "Plan not found"}
        return plan.to_dict()

    @app.post("/api/v1/evolution/plans", dependencies=deps)
    async def create_evolution_plan(request: Request) -> dict[str, Any]:
        """Create a new learning plan from a goal."""
        dl = getattr(gateway, "_deep_learner", None)
        if not dl:
            return {"error": "DeepLearner not available"}
        try:
            body = await request.json()
            goal = body.get("goal", "")
            if not goal:
                return {"error": "goal is required"}
            seeds_raw = body.get("seed_sources", [])
            seeds = []
            for s in seeds_raw:
                from cognithor.evolution.models import SeedSource

                seeds.append(
                    SeedSource(
                        content_type=s.get("content_type", "hint"),
                        value=s.get("value", ""),
                        title=s.get("title", ""),
                    )
                )
            plan = await dl.create_plan(goal, seed_sources=seeds if seeds else None)
            return plan.to_summary_dict()
        except Exception as exc:
            return {"error": str(exc)}

    @app.patch("/api/v1/evolution/plans/{plan_id}", dependencies=deps)
    async def update_evolution_plan(plan_id: str, request: Request) -> dict[str, Any]:
        """Update plan status (pause/resume/delete)."""
        dl = getattr(gateway, "_deep_learner", None)
        if not dl:
            return {"error": "DeepLearner not available"}
        try:
            body = await request.json()
            action = body.get("action", "")
            if action == "delete":
                dl.delete_plan(plan_id)
                return {"deleted": True}
            elif action in ("pause", "resume", "complete"):
                status_map = {"pause": "paused", "resume": "active", "complete": "completed"}
                ok = dl.update_plan_status(plan_id, status_map[action])
                return {"updated": ok, "status": status_map[action]}
            return {"error": f"Unknown action: {action}"}
        except Exception as exc:
            return {"error": str(exc)}

    @app.get("/api/v1/evolution/plans/{plan_id}/index", dependencies=deps)
    async def get_plan_index_stats(plan_id: str) -> dict[str, Any]:
        """Get per-goal index statistics."""
        dl = getattr(gateway, "_deep_learner", None)
        if not dl:
            return {"error": "DeepLearner not available"}
        plan = dl.get_plan(plan_id)
        if not plan:
            return {"error": "Plan not found"}
        try:
            from cognithor.evolution.goal_index import GoalScopedIndex

            index_base = dl._plans_dir.parent / "indexes"
            idx = GoalScopedIndex(goal_slug=plan.goal_slug, base_dir=index_base)
            stats = idx.stats()
            idx.close()
            return stats
        except Exception as exc:
            return {"error": str(exc)}

    @app.get("/api/v1/evolution/claims", dependencies=deps)
    async def get_evolution_claims(goal_slug: str = "") -> dict[str, Any]:
        """Get knowledge claims table with confidence scores."""
        dl = getattr(gateway, "_deep_learner", None)
        if not dl or not dl._knowledge_validator:
            return {"claims": [], "summary": {}, "message": "KnowledgeValidator not available"}
        summary = dl._knowledge_validator.get_claims_summary()
        claims = dl._knowledge_validator.get_claims(limit=100)
        return {
            "summary": summary,
            "claims": [c.to_dict() for c in claims],
        }

    # -- 3.4: POST /agents/{name} ----------------------------------------

    @app.post("/api/v1/agents/{name}", dependencies=deps)
    async def ui_upsert_agent(name: str, request: Request) -> dict[str, Any]:
        """Creates or updates an agent profile in agents.yaml."""
        try:
            from cognithor.gateway.config_api import AgentProfileDTO

            body = await request.json()
            body["name"] = name
            validated = AgentProfileDTO(**body).model_dump(exclude_unset=False)
            data = _load_yaml(agents_path)
            agents = data.get("agents", [])
            if not isinstance(agents, list):
                agents = []
            # Upsert by name
            found = False
            for i, a in enumerate(agents):
                if isinstance(a, dict) and a.get("name") == name:
                    agents[i] = validated
                    found = True
                    break
            if not found:
                agents.append(validated)
            data["agents"] = agents
            _save_yaml(agents_path, data)
            return {"status": "ok", "agent": name}
        except Exception as exc:
            log.error("agent_upsert_failed", agent=name, error=str(exc))
            return {"error": "Agent konnte nicht gespeichert werden", "status": 400}

    # -- 3.5: POST /bindings/{name} --------------------------------------

    @app.post("/api/v1/bindings/{name}", dependencies=deps)
    async def ui_upsert_binding(name: str, request: Request) -> dict[str, Any]:
        """Creates or updates a binding rule in bindings.yaml."""
        try:
            from cognithor.gateway.config_api import BindingRuleDTO

            body = await request.json()
            body["name"] = name
            validated = BindingRuleDTO(**body).model_dump(exclude_unset=False)
            data = _load_yaml(bindings_path)
            bindings = data.get("bindings", [])
            if not isinstance(bindings, list):
                bindings = []
            found = False
            for i, b in enumerate(bindings):
                if isinstance(b, dict) and b.get("name") == name:
                    bindings[i] = validated
                    found = True
                    break
            if not found:
                bindings.append(validated)
            data["bindings"] = bindings
            _save_yaml(bindings_path, data)
            return {"status": "ok", "binding": name}
        except Exception as exc:
            log.error("binding_upsert_failed", binding=name, error=str(exc))
            return {"error": "Binding konnte nicht gespeichert werden", "status": 400}

    # -- 3.6: Prompts GET / PUT ------------------------------------------

    @app.get("/api/v1/prompts", dependencies=deps)
    async def ui_get_prompts() -> dict[str, Any]:
        """Reads prompt/policy files for the Prompts & Policies page."""
        cfg = config_manager.config
        prompts_dir = cognithor_home / "prompts"
        result: dict[str, str] = {}

        # coreMd
        try:
            core_path = cfg.core_memory_file
            result["coreMd"] = core_path.read_text(encoding="utf-8") if core_path.exists() else ""
        except Exception:
            result["coreMd"] = ""

        # plannerSystem (.md bevorzugt, .txt als Migration-Fallback)
        try:
            from cognithor.core.planner import SYSTEM_PROMPT

            content = ""
            for fname in ("SYSTEM_PROMPT.md", "SYSTEM_PROMPT.txt"):
                sys_path = prompts_dir / fname
                if sys_path.exists():
                    content = sys_path.read_text(encoding="utf-8").strip()
                    if content:
                        break
            if not content:
                content = SYSTEM_PROMPT
            result["plannerSystem"] = content
        except Exception:
            result["plannerSystem"] = ""

        # replanPrompt
        try:
            from cognithor.core.planner import REPLAN_PROMPT

            content = ""
            for fname in ("REPLAN_PROMPT.md", "REPLAN_PROMPT.txt"):
                rp_path = prompts_dir / fname
                if rp_path.exists():
                    content = rp_path.read_text(encoding="utf-8").strip()
                    if content:
                        break
            if not content:
                content = REPLAN_PROMPT
            result["replanPrompt"] = content
        except Exception:
            result["replanPrompt"] = ""

        # escalationPrompt
        try:
            from cognithor.core.planner import ESCALATION_PROMPT

            content = ""
            for fname in ("ESCALATION_PROMPT.md", "ESCALATION_PROMPT.txt"):
                ep_path = prompts_dir / fname
                if ep_path.exists():
                    content = ep_path.read_text(encoding="utf-8").strip()
                    if content:
                        break
            if not content:
                content = ESCALATION_PROMPT
            result["escalationPrompt"] = content
        except Exception:
            result["escalationPrompt"] = ""

        # policyYaml
        try:
            policy_path = cfg.policies_dir / "default.yaml"
            content = policy_path.read_text(encoding="utf-8") if policy_path.exists() else ""
            result["policyYaml"] = content
        except Exception:
            result["policyYaml"] = ""

        # heartbeatMd
        try:
            hb_path = cognithor_home / cfg.heartbeat.checklist_file
            result["heartbeatMd"] = hb_path.read_text(encoding="utf-8") if hb_path.exists() else ""
        except Exception:
            result["heartbeatMd"] = ""

        return result

    @app.put("/api/v1/prompts", dependencies=deps)
    async def ui_put_prompts(request: Request) -> dict[str, Any]:
        """Writes prompt/policy files from the UI."""
        try:
            body = await request.json()
            cfg = config_manager.config
            prompts_dir = cognithor_home / "prompts"
            prompts_dir.mkdir(parents=True, exist_ok=True)
            written: list[str] = []

            if "coreMd" in body:
                cfg.core_memory_file.parent.mkdir(parents=True, exist_ok=True)
                cfg.core_memory_file.write_text(body["coreMd"], encoding="utf-8")
                written.append("coreMd")

            if "plannerSystem" in body:
                (prompts_dir / "SYSTEM_PROMPT.md").write_text(
                    body["plannerSystem"], encoding="utf-8"
                )
                written.append("plannerSystem")

            if "replanPrompt" in body:
                (prompts_dir / "REPLAN_PROMPT.md").write_text(
                    body["replanPrompt"], encoding="utf-8"
                )
                written.append("replanPrompt")

            if "escalationPrompt" in body:
                (prompts_dir / "ESCALATION_PROMPT.md").write_text(
                    body["escalationPrompt"], encoding="utf-8"
                )
                written.append("escalationPrompt")

            if "policyYaml" in body:
                cfg.policies_dir.mkdir(parents=True, exist_ok=True)
                (cfg.policies_dir / "default.yaml").write_text(body["policyYaml"], encoding="utf-8")
                written.append("policyYaml")

            if "heartbeatMd" in body:
                hb_path = cognithor_home / cfg.heartbeat.checklist_file
                hb_path.parent.mkdir(parents=True, exist_ok=True)
                hb_path.write_text(body["heartbeatMd"], encoding="utf-8")
                written.append("heartbeatMd")

            # Live-Reload: Gateway-Komponenten sofort aktualisieren
            if gateway is not None and hasattr(gateway, "reload_components") and written:
                reload_flags: dict[str, bool] = {}
                if any(k in written for k in ("plannerSystem", "replanPrompt", "escalationPrompt")):
                    reload_flags["prompts"] = True
                if "policyYaml" in written:
                    reload_flags["policies"] = True
                if "coreMd" in written:
                    reload_flags["core_memory"] = True
                if reload_flags:
                    gateway.reload_components(**reload_flags)

            return {"status": "ok", "written": written}
        except Exception as exc:
            log.error("prompts_put_failed", error=str(exc))
            return {"error": "Prompts konnten nicht gespeichert werden", "status": 400}

    # -- 3.7: Cron Jobs GET / PUT ----------------------------------------

    @app.get("/api/v1/cron-jobs", dependencies=deps)
    async def ui_get_cron_jobs() -> dict[str, Any]:
        """Reads cron jobs via JobStore."""
        try:
            from cognithor.cron.jobs import JobStore

            store = JobStore(config_manager.config.cron_config_file)
            jobs = store.load()
            return {
                "jobs": [
                    {
                        "name": j.name,
                        "schedule": j.schedule,
                        "prompt": j.prompt,
                        "channel": j.channel,
                        "model": j.model,
                        "enabled": j.enabled,
                        "agent": j.agent,
                    }
                    for j in jobs.values()
                ],
            }
        except Exception as exc:
            log.error("cron_jobs_get_failed", error=str(exc))
            return {"jobs": [], "error": "Cron-Jobs konnten nicht geladen werden"}

    @app.put("/api/v1/cron-jobs", dependencies=deps)
    async def ui_put_cron_jobs(request: Request) -> dict[str, Any]:
        """Writes cron jobs via JobStore."""
        try:
            from cognithor.cron.jobs import JobStore
            from cognithor.models import CronJob

            body = await request.json()
            store = JobStore(config_manager.config.cron_config_file)
            store.load()
            store.jobs = {}
            for j in body.get("jobs", []):
                if not isinstance(j, dict) or "name" not in j:
                    continue
                store.jobs[j["name"]] = CronJob(**j)
            store._save()
            return {"status": "ok", "count": len(store.jobs)}
        except Exception as exc:
            log.error("cron_jobs_put_failed", error=str(exc))
            return {"error": "Cron-Jobs konnten nicht gespeichert werden", "status": 400}

    # -- 3.7b: Cron-Jobs toggle + enriched list ---------------------------

    @app.patch("/api/v1/cron-jobs/{job_name}/toggle", dependencies=deps)
    async def ui_toggle_cron_job(job_name: str) -> dict[str, Any]:
        """Toggle a cron job enabled/disabled."""
        try:
            from cognithor.cron.jobs import JobStore

            store = JobStore(config_manager.config.cron_config_file)
            store.load()
            if job_name not in store.jobs:
                return {"error": f"Job '{job_name}' not found", "status": 404}
            new_enabled = not store.jobs[job_name].enabled
            store.toggle_job(job_name, new_enabled)
            # Also update the live scheduler if available
            gw = getattr(config_manager, "_gateway", None)
            cron_engine = getattr(gw, "_cron_engine", None) if gw else None
            if cron_engine and hasattr(cron_engine, "job_store"):
                cron_engine.job_store.jobs[job_name] = store.jobs[job_name]
            return {"name": job_name, "enabled": new_enabled}
        except Exception as exc:
            log.error("cron_job_toggle_failed", error=str(exc))
            return {"error": str(exc), "status": 500}

    @app.get("/api/v1/cron-jobs/enriched", dependencies=deps)
    async def ui_get_cron_jobs_enriched() -> dict[str, Any]:
        """Returns cron jobs with next_run times and last_run info."""
        try:
            from cognithor.cron.jobs import JobStore

            store = JobStore(config_manager.config.cron_config_file)
            jobs = store.load()

            # Try to get next_run from live engine
            gw = getattr(config_manager, "_gateway", None)
            cron_engine = getattr(gw, "_cron_engine", None) if gw else None
            next_runs: dict[str, Any] = {}
            if cron_engine:
                try:
                    next_runs = cron_engine.get_next_run_times()
                except Exception:
                    log.debug("cron_next_run_times_fetch_failed", exc_info=True)

            result = []
            for j in jobs.values():
                nr = next_runs.get(j.name)
                result.append(
                    {
                        "name": j.name,
                        "schedule": j.schedule,
                        "prompt": j.prompt,
                        "channel": j.channel,
                        "model": j.model,
                        "enabled": j.enabled,
                        "agent": j.agent,
                        "next_run": nr.isoformat() if nr else None,
                    }
                )
            return {"jobs": result}
        except Exception as exc:
            log.error("cron_jobs_enriched_failed", error=str(exc))
            return {"jobs": [], "error": str(exc)}

    # -- 3.8: MCP Servers GET / PUT --------------------------------------

    @app.get("/api/v1/mcp-servers", dependencies=deps)
    async def ui_get_mcp_servers() -> dict[str, Any]:
        """Reads MCP server config, flattened for the UI."""
        try:
            mcp_path = config_manager.config.mcp_config_file
            data = _load_yaml(mcp_path)
            sm = data.get("server_mode", {})
            servers_raw = data.get("servers", {})
            # servers can be dict (name→config) or list; normalize to dict
            if isinstance(servers_raw, list):
                servers_dict = {
                    s.get("name", f"server_{i}"): s
                    for i, s in enumerate(servers_raw)
                    if isinstance(s, dict)
                }
            elif isinstance(servers_raw, dict):
                servers_dict = servers_raw
            else:
                servers_dict = {}
            # Flatten server_mode fields into response + external_servers as dict
            result: dict[str, Any] = {
                "mode": sm.get("mode", "disabled"),
                "http_host": sm.get("http_host", "127.0.0.1"),
                "http_port": sm.get("http_port", 3001),
                "server_name": sm.get("server_name", "jarvis"),
                "require_auth": sm.get("require_auth", False),
                "auth_token": "***" if sm.get("auth_token") else "",
                "expose_tools": sm.get("expose_tools", True),
                "expose_resources": sm.get("expose_resources", True),
                "expose_prompts": sm.get("expose_prompts", False),
                "enable_sampling": sm.get("enable_sampling", False),
                "external_servers": servers_dict,
            }
            return result
        except Exception as exc:
            log.error("mcp_servers_load_failed", error=str(exc))
            return {
                "mode": "disabled",
                "external_servers": {},
                "error": "MCP-Server-Konfiguration konnte nicht geladen werden",
            }

    @app.put("/api/v1/mcp-servers", dependencies=deps)
    async def ui_put_mcp_servers(request: Request) -> dict[str, Any]:
        """Writes MCP config from flat UI format, preserving a2a and built_in_tools."""
        try:
            body = await request.json()
            mcp_path = config_manager.config.mcp_config_file
            data = _load_yaml(mcp_path)
            # Reconstruct server_mode from flat fields
            sm_keys = (
                "mode",
                "http_host",
                "http_port",
                "server_name",
                "require_auth",
                "auth_token",
                "expose_tools",
                "expose_resources",
                "expose_prompts",
                "enable_sampling",
            )
            sm = data.get("server_mode", {})
            for k in sm_keys:
                if k in body:
                    sm[k] = body[k]
            data["server_mode"] = sm
            # external_servers → servers
            if "external_servers" in body:
                data["servers"] = body["external_servers"]
            _save_yaml(mcp_path, data)
            return {"status": "ok"}
        except Exception as exc:
            log.error("mcp_servers_put_failed", error=str(exc))
            return {
                "error": "MCP-Server-Konfiguration konnte nicht gespeichert werden",
                "status": 400,
            }

    # -- 3.9: A2A GET / PUT ----------------------------------------------

    @app.get("/api/v1/a2a", dependencies=deps)
    async def ui_get_a2a() -> dict[str, Any]:
        """Reads a2a section from MCP config."""
        try:
            mcp_path = config_manager.config.mcp_config_file
            data = _load_yaml(mcp_path)
            a2a = data.get("a2a", {})
            # Provide sensible defaults
            return {
                "enabled": a2a.get("enabled", False),
                "host": a2a.get("host", "0.0.0.0"),
                "port": a2a.get("port", 8742),
                "agent_name": a2a.get("agent_name", "jarvis"),
                **{
                    k: v
                    for k, v in a2a.items()
                    if k not in ("enabled", "host", "port", "agent_name")
                },
            }
        except Exception as exc:
            log.error("a2a_get_failed", error=str(exc))
            return {"enabled": False, "error": "A2A-Konfiguration konnte nicht geladen werden"}

    @app.put("/api/v1/a2a", dependencies=deps)
    async def ui_put_a2a(request: Request) -> dict[str, Any]:
        """Writes a2a section, preserving other MCP config sections."""
        try:
            body = await request.json()
            mcp_path = config_manager.config.mcp_config_file
            data = _load_yaml(mcp_path)
            data["a2a"] = body
            _save_yaml(mcp_path, data)
            return {"status": "ok"}
        except Exception as exc:
            log.error("a2a_config_save_failed", error=str(exc))
            return {"error": "A2A-Konfiguration konnte nicht gespeichert werden", "status": 400}


# ======================================================================
# Workflow Execution Graph API
# ======================================================================


def _register_workflow_graph_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """REST endpoints for workflow execution graph visualization."""

    def _get_engines() -> tuple[Any, Any, Any]:
        """Return (simple_engine, dag_engine, template_library) from gateway."""
        simple = getattr(gateway, "_workflow_engine", None) if gateway else None
        dag = getattr(gateway, "_dag_workflow_engine", None) if gateway else None
        tmpl = getattr(gateway, "_template_library", None) if gateway else None
        return simple, dag, tmpl

    # -- Templates ---------------------------------------------------------

    @app.get("/api/v1/workflows/templates", dependencies=deps)
    async def wf_list_templates() -> dict[str, Any]:
        """List all available workflow templates."""
        _, _, tmpl = _get_engines()
        if not tmpl:
            return {"templates": [], "count": 0}
        return {"templates": tmpl.list_all(), "count": tmpl.template_count}

    @app.get("/api/v1/workflows/templates/{template_id}", dependencies=deps)
    async def wf_get_template(template_id: str) -> dict[str, Any]:
        _, _, tmpl = _get_engines()
        if not tmpl:
            return {"error": "Template library unavailable", "status": 503}
        t = tmpl.get(template_id)
        if not t:
            return {"error": "Template not found", "status": 404}
        return t.to_dict()

    # -- Simple workflow instances -----------------------------------------

    @app.get("/api/v1/workflows/instances", dependencies=deps)
    async def wf_list_instances() -> dict[str, Any]:
        """List all workflow instances (simple engine)."""
        simple, _, _ = _get_engines()
        if not simple:
            return {"instances": [], "stats": {}}
        all_inst = list(simple._instances.values())
        return {
            "instances": [i.to_dict() for i in all_inst],
            "stats": simple.stats(),
        }

    @app.get("/api/v1/workflows/instances/{instance_id}", dependencies=deps)
    async def wf_get_instance(instance_id: str) -> dict[str, Any]:
        simple, _, tmpl = _get_engines()
        if not simple:
            return {"error": "Workflow engine unavailable", "status": 503}
        inst = simple.get(instance_id)
        if not inst:
            return {"error": "Instance not found", "status": 404}
        result = inst.to_dict()
        result["step_results"] = inst.step_results
        if tmpl:
            t = tmpl.get(inst.template_id)
            if t:
                result["steps"] = [s.to_dict() for s in t.steps]
        return result

    @app.post("/api/v1/workflows/instances", dependencies=deps)
    async def wf_start_instance(request: Request) -> dict[str, Any]:
        """Start a new workflow from a template."""
        simple, _, tmpl = _get_engines()
        if not simple or not tmpl:
            return {"error": "Workflow engine unavailable", "status": 503}
        body = await request.json()
        template_id = body.get("template_id", "")
        t = tmpl.get(template_id)
        if not t:
            return {"error": f"Template '{template_id}' not found", "status": 404}
        inst = simple.start(t, created_by=body.get("created_by", "ui"))
        return {"status": "ok", "instance": inst.to_dict()}

    # -- DAG workflow runs -------------------------------------------------

    @app.get("/api/v1/workflows/dag/runs", dependencies=deps)
    async def wf_list_dag_runs() -> dict[str, Any]:
        """List DAG workflow runs (checkpoint-based)."""
        _, dag, _ = _get_engines()
        if not dag or not dag._checkpoint_dir:
            return {"runs": []}
        cp_dir = dag._checkpoint_dir
        runs = []
        if cp_dir.exists():
            for cp_file in sorted(cp_dir.glob("*.json"), reverse=True):
                try:
                    data = json.loads(cp_file.read_text(encoding="utf-8"))
                    runs.append(
                        {
                            "id": data.get("id", ""),
                            "workflow_id": data.get("workflow_id", ""),
                            "workflow_name": data.get("workflow_name", ""),
                            "status": data.get("status", ""),
                            "started_at": data.get("started_at"),
                            "completed_at": data.get("completed_at"),
                            "node_count": len(data.get("node_results", {})),
                        }
                    )
                except Exception:
                    continue
        return {"runs": runs}

    @app.get("/api/v1/workflows/dag/runs/{run_id}", dependencies=deps)
    async def wf_get_dag_run(run_id: str) -> dict[str, Any]:
        """Get full DAG workflow run with node graph data."""
        _, dag, _ = _get_engines()
        if not dag or not dag._checkpoint_dir:
            return {"error": "DAG engine unavailable", "status": 503}
        cp_file = (dag._checkpoint_dir / f"{run_id}.json").resolve()
        try:
            cp_file.relative_to(dag._checkpoint_dir.resolve())
        except ValueError:
            return {"error": "Invalid run_id (Path-Traversal)", "status": 400}
        if not cp_file.exists():
            return {"error": "Run not found", "status": 404}
        try:
            return json.loads(cp_file.read_text(encoding="utf-8"))
        except Exception as exc:
            log.error("wf_dag_run_read_failed", run_id=run_id, error=str(exc))
            return {"error": "DAG-Run konnte nicht geladen werden", "status": 500}

    @app.get("/api/v1/workflows/dag/runs/{run_id}/nodes/{node_id}", dependencies=deps)
    async def wf_get_dag_node_detail(run_id: str, node_id: str) -> dict[str, Any]:
        """Get detailed execution data for a single DAG node."""
        _, dag, _ = _get_engines()
        if not dag or not dag._checkpoint_dir:
            return {"error": "DAG engine unavailable", "status": 503}
        cp_file = (dag._checkpoint_dir / f"{run_id}.json").resolve()
        try:
            cp_file.relative_to(dag._checkpoint_dir.resolve())
        except ValueError:
            return {"error": "Invalid run_id (Path-Traversal)", "status": 400}
        if not cp_file.exists():
            return {"error": "Run not found", "status": 404}
        try:
            data = json.loads(cp_file.read_text(encoding="utf-8"))
            node_results = data.get("node_results", {})
            if node_id not in node_results:
                return {"error": f"Node '{node_id}' not found in run", "status": 404}
            return {"node_id": node_id, "run_id": run_id, **node_results[node_id]}
        except json.JSONDecodeError:
            return {"error": "Invalid run data", "status": 500}

    # -- Combined stats ----------------------------------------------------

    @app.get("/api/v1/workflows/stats", dependencies=deps)
    async def wf_stats() -> dict[str, Any]:
        """Combined workflow stats."""
        simple, dag, tmpl = _get_engines()
        result: dict[str, Any] = {"templates": 0, "simple": {}, "dag_runs": 0}
        if tmpl:
            result["templates"] = tmpl.template_count
        if simple:
            result["simple"] = simple.stats()
        if dag and dag._checkpoint_dir and dag._checkpoint_dir.exists():
            result["dag_runs"] = len(list(dag._checkpoint_dir.glob("*.json")))
        return result


# ======================================================================
# Learning / Curiosity / Confidence routes
# ======================================================================


def _register_learning_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """REST endpoints for Active Learning, Curiosity Engine, and Confidence Manager."""

    def _get_learner() -> Any:
        return getattr(gateway, "_active_learner", None) if gateway else None

    def _get_curiosity() -> Any:
        return getattr(gateway, "_curiosity_engine", None) if gateway else None

    def _get_confidence() -> Any:
        return getattr(gateway, "_confidence_manager", None) if gateway else None

    # -- Stats -------------------------------------------------------------

    @app.get("/api/v1/learning/stats", dependencies=deps)
    async def learning_stats() -> dict[str, Any]:
        """Combined learning statistics."""
        result: dict[str, Any] = {}

        learner = _get_learner()
        if learner:
            result["active_learner"] = learner.stats()

        curiosity = _get_curiosity()
        if curiosity:
            result["curiosity"] = {
                "total_gaps": len(curiosity.gaps),
                "open_gaps": curiosity.open_gap_count,
            }

        confidence = _get_confidence()
        if confidence:
            result["confidence"] = confidence.stats()

        if not result:
            result["message"] = "Learning subsystem not initialized"

        return result

    # -- Knowledge gaps ----------------------------------------------------

    @app.get("/api/v1/learning/gaps", dependencies=deps)
    async def learning_gaps() -> dict[str, Any]:
        """List detected knowledge gaps."""
        curiosity = _get_curiosity()
        if not curiosity:
            return {"gaps": [], "count": 0}

        gaps = curiosity.gaps
        return {
            "gaps": [
                {
                    "id": g.id,
                    "question": g.question,
                    "topic": g.topic,
                    "importance": g.importance,
                    "curiosity": g.curiosity,
                    "status": g.status,
                    "created_at": g.created_at.isoformat(),
                    "suggested_sources": g.suggested_sources,
                }
                for g in gaps
            ],
            "count": len(gaps),
            "open": sum(1 for g in gaps if g.status == "open"),
        }

    @app.post("/api/v1/learning/gaps/{gap_id}/dismiss", dependencies=deps)
    async def learning_dismiss_gap(gap_id: str) -> dict[str, Any]:
        """Dismiss a knowledge gap."""
        curiosity = _get_curiosity()
        if not curiosity:
            return {"error": "Curiosity engine not initialized", "status": 503}

        found = curiosity.dismiss_gap(gap_id)
        if not found:
            return {"error": "Gap not found", "status": 404}
        return {"status": "dismissed", "gap_id": gap_id}

    # -- Confidence history ------------------------------------------------

    @app.get("/api/v1/learning/confidence/history", dependencies=deps)
    async def learning_confidence_history() -> dict[str, Any]:
        """Return recent confidence changes."""
        confidence = _get_confidence()
        if not confidence:
            return {"history": [], "stats": {}}

        history = confidence.history
        # Return last 100 entries
        recent = history[-100:]
        return {
            "history": [
                {
                    "entity_id": h.entity_id,
                    "old_confidence": round(h.old_confidence, 4),
                    "new_confidence": round(h.new_confidence, 4),
                    "reason": h.reason,
                    "timestamp": h.timestamp.isoformat(),
                }
                for h in recent
            ],
            "stats": confidence.stats(),
        }

    @app.post("/api/v1/learning/confidence/{entity_id}/feedback", dependencies=deps)
    async def learning_confidence_feedback(entity_id: str, request: Request) -> dict[str, Any]:
        """Apply feedback to an entity's confidence."""
        confidence = _get_confidence()
        if not confidence:
            return {"error": "Confidence manager not initialized", "status": 503}

        body = await request.json()
        feedback_type = body.get("type", "")
        if feedback_type not in ("positive", "negative", "correction"):
            return {"error": "Invalid feedback type. Must be: positive, negative, correction"}

        # Read current confidence from entity DB
        current = 0.5  # fallback
        mm = getattr(gateway, "_memory_manager", None)
        idx = getattr(mm, "_index", None) if mm else None
        if idx:
            try:
                ent = idx.get_entity_by_id(entity_id)
                if ent:
                    current = ent.confidence
            except Exception:
                log.debug("entity_confidence_read_failed", exc_info=True)

        new_conf = confidence.apply_feedback(entity_id, current, feedback_type)

        # Persist updated confidence to database
        if idx:
            with contextlib.suppress(Exception):
                idx.update_entity_confidence(entity_id, new_conf)

        return {
            "entity_id": entity_id,
            "old_confidence": round(current, 4),
            "new_confidence": round(new_conf, 4),
            "feedback_type": feedback_type,
        }

    # -- Exploration queue -------------------------------------------------

    @app.get("/api/v1/learning/queue", dependencies=deps)
    async def learning_queue() -> dict[str, Any]:
        """Return the exploration task queue."""
        curiosity = _get_curiosity()
        if not curiosity:
            return {"tasks": [], "count": 0}

        tasks = curiosity.propose_exploration()
        return {
            "tasks": [
                {
                    "gap_id": t.gap_id,
                    "query": t.query,
                    "sources": t.sources,
                    "priority": t.priority,
                    "max_depth": t.max_depth,
                }
                for t in tasks
            ],
            "count": len(tasks),
        }

    @app.post("/api/v1/learning/explore", dependencies=deps)
    async def learning_explore(request: Request) -> dict[str, Any]:
        """Trigger exploration of a specific gap."""
        curiosity = _get_curiosity()
        if not curiosity:
            return {"error": "Curiosity engine not initialized", "status": 503}

        body = await request.json()
        gap_id = body.get("gap_id", "")

        if not gap_id:
            return {"error": "gap_id is required"}

        found = curiosity.mark_exploring(gap_id)
        if not found:
            return {"error": "Gap not found", "status": 404}

        return {"status": "exploring", "gap_id": gap_id}

    # -- Watch directories -------------------------------------------------

    @app.get("/api/v1/learning/directories", dependencies=deps)
    async def learning_directories() -> dict[str, Any]:
        """Return watched directories configuration."""
        learner = _get_learner()
        if not learner:
            return {"directories": []}
        dirs = learner.stats().get("watch_dirs", [])
        return {"directories": dirs}

    @app.post("/api/v1/learning/directories", dependencies=deps)
    async def learning_update_directories(request: Request) -> dict[str, Any]:
        """Update watched directories (enable/disable, add new)."""
        learner = _get_learner()
        if not learner:
            return {"error": "Active learner not initialized", "status": 503}
        body = await request.json()
        dirs = body.get("directories", [])
        for d in dirs:
            path = d.get("path", "")
            enabled = d.get("enabled", True)
            if path:
                learner.add_directory(path, enabled=enabled)
        return {"status": "updated", "count": len(dirs)}

    # -- Q&A Knowledge Base ------------------------------------------------

    def _get_qa() -> Any:
        return getattr(gateway, "_knowledge_qa", None) if gateway else None

    @app.get("/api/v1/learning/qa", dependencies=deps)
    async def learning_qa_list(request: Request) -> dict[str, Any]:
        """List or search Q&A pairs."""
        qa_store = _get_qa()
        if not qa_store:
            return {"error": "QA store not initialized", "status": 503}

        query = request.query_params.get("q", "")
        limit = int(request.query_params.get("limit", "50"))
        offset = int(request.query_params.get("offset", "0"))

        if query:
            pairs = qa_store.search(query, limit=limit)
        else:
            pairs = qa_store.list_all(limit=limit, offset=offset)

        return {
            "pairs": [
                {
                    "id": p.id,
                    "question": p.question,
                    "answer": p.answer,
                    "topic": p.topic,
                    "confidence": round(p.confidence, 4),
                    "source": p.source,
                    "entity_id": p.entity_id,
                    "created_at": p.created_at,
                    "last_verified": p.last_verified,
                    "verification_count": p.verification_count,
                }
                for p in pairs
            ],
            "count": len(pairs),
            "stats": qa_store.stats(),
        }

    @app.post("/api/v1/learning/qa", dependencies=deps)
    async def learning_qa_add(request: Request) -> dict[str, Any]:
        """Add a new Q&A pair."""
        qa_store = _get_qa()
        if not qa_store:
            return {"error": "QA store not initialized", "status": 503}

        body = await request.json()
        question = body.get("question", "").strip()
        answer = body.get("answer", "").strip()
        if not question or not answer:
            return {"error": "question and answer are required"}

        pair = qa_store.add(
            question,
            answer,
            topic=body.get("topic", ""),
            confidence=float(body.get("confidence", 0.5)),
            source=body.get("source", ""),
            entity_id=body.get("entity_id", ""),
        )
        return {
            "id": pair.id,
            "question": pair.question,
            "answer": pair.answer,
            "topic": pair.topic,
            "confidence": pair.confidence,
            "created_at": pair.created_at,
        }

    @app.post(
        "/api/v1/learning/qa/{qa_id}/verify",
        dependencies=deps,
    )
    async def learning_qa_verify(qa_id: str) -> dict[str, Any]:
        """Verify a Q&A pair, boosting its confidence."""
        qa_store = _get_qa()
        if not qa_store:
            return {"error": "QA store not initialized", "status": 503}

        found = qa_store.verify(qa_id)
        if not found:
            return {"error": "QA pair not found", "status": 404}
        return {"status": "verified", "id": qa_id}

    @app.delete(
        "/api/v1/learning/qa/{qa_id}",
        dependencies=deps,
    )
    async def learning_qa_delete(qa_id: str) -> dict[str, Any]:
        """Delete a Q&A pair."""
        qa_store = _get_qa()
        if not qa_store:
            return {"error": "QA store not initialized", "status": 503}

        found = qa_store.delete(qa_id)
        if not found:
            return {"error": "QA pair not found", "status": 404}
        return {"status": "deleted", "id": qa_id}

    # -- Knowledge Lineage -------------------------------------------------

    def _get_lineage() -> Any:
        return getattr(gateway, "_knowledge_lineage", None) if gateway else None

    @app.get(
        "/api/v1/learning/lineage/{entity_id}",
        dependencies=deps,
    )
    async def learning_lineage_entity(
        entity_id: str,
        request: Request,
    ) -> dict[str, Any]:
        """Get lineage entries for a specific entity."""
        tracker = _get_lineage()
        if not tracker:
            return {"error": "Lineage tracker not initialized", "status": 503}

        limit = int(request.query_params.get("limit", "50"))
        entries = tracker.get_entity_lineage(
            entity_id,
            limit=limit,
        )
        return {
            "entity_id": entity_id,
            "entries": [
                {
                    "id": e.id,
                    "source_type": e.source_type,
                    "source_path": e.source_path,
                    "action": e.action,
                    "old_value": e.old_value,
                    "new_value": e.new_value,
                    "confidence_before": e.confidence_before,
                    "confidence_after": e.confidence_after,
                    "timestamp": e.timestamp,
                }
                for e in entries
            ],
            "count": len(entries),
        }

    @app.get("/api/v1/learning/lineage", dependencies=deps)
    async def learning_lineage_recent(
        request: Request,
    ) -> dict[str, Any]:
        """Get recent lineage entries."""
        tracker = _get_lineage()
        if not tracker:
            return {"error": "Lineage tracker not initialized", "status": 503}

        limit = int(request.query_params.get("limit", "100"))
        entries = tracker.get_recent(limit=limit)
        return {
            "entries": [
                {
                    "id": e.id,
                    "entity_id": e.entity_id,
                    "source_type": e.source_type,
                    "source_path": e.source_path,
                    "action": e.action,
                    "old_value": e.old_value,
                    "new_value": e.new_value,
                    "confidence_before": e.confidence_before,
                    "confidence_after": e.confidence_after,
                    "timestamp": e.timestamp,
                }
                for e in entries
            ],
            "count": len(entries),
            "stats": tracker.stats(),
        }

    # -- Batch Exploration -------------------------------------------------

    def _get_explorer() -> Any:
        return getattr(gateway, "_exploration_executor", None) if gateway else None

    @app.post(
        "/api/v1/learning/explore/run",
        dependencies=deps,
    )
    async def learning_explore_run(
        request: Request,
    ) -> dict[str, Any]:
        """Trigger a batch of exploration tasks."""
        explorer = _get_explorer()
        if not explorer:
            return {
                "error": "Exploration executor not initialized",
                "status": 503,
            }

        body = await request.json()
        max_tasks = int(body.get("max_tasks", 3))
        max_tasks = max(1, min(max_tasks, 10))

        results = await explorer.execute_batch(
            max_tasks=max_tasks,
        )
        return {
            "results": [
                {
                    "gap_id": r.gap_id,
                    "query": r.query,
                    "found_answer": r.found_answer,
                    "answer_summary": r.answer_summary,
                    "sources_checked": r.sources_checked,
                    "entities_updated": r.entities_updated,
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in results
            ],
            "count": len(results),
            "stats": explorer.stats(),
        }


# ======================================================================
# Knowledge Ingestion routes (file upload, URL, YouTube)
# ======================================================================


def _register_ingest_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """REST endpoints for knowledge ingestion (files, URLs, YouTube)."""

    def _get_ingest() -> Any:
        return getattr(gateway, "_knowledge_ingest", None) if gateway else None

    # -- File upload --------------------------------------------------------

    @app.post("/api/v1/learn/file", dependencies=deps)
    async def learn_file(request: Request) -> dict[str, Any]:
        """Ingest a file upload (multipart/form-data).

        Expects field 'file' with the document/image.
        Optional field 'description' with context text.
        """
        svc = _get_ingest()
        if not svc:
            return {"error": "Knowledge ingest service not initialized", "status": 503}

        try:
            form = await request.form()
            file_field = form.get("file")
            if file_field is None:
                return {"error": "Field 'file' is required", "code": "MISSING_FIELD"}

            file_bytes = await file_field.read()
            if not file_bytes:
                return {"error": "Empty file", "code": "EMPTY_FILE"}

            filename = "upload"
            if hasattr(file_field, "filename") and file_field.filename:
                filename = file_field.filename

            priority_str = str(form.get("priority", "normal") or "normal")
            from cognithor.learning.knowledge_ingest import Priority

            priority = Priority.from_string(priority_str)
            result = await svc.ingest_file(filename, file_bytes, priority=priority)

            return {
                "id": result.id,
                "source_type": result.source_type,
                "source_name": result.source_name,
                "status": result.status,
                "chunks_created": result.chunks_created,
                "chunks": result.chunks,
                "deep_learn_status": result.deep_learn_status,
                "text_length": result.text_length,
                "error": result.error,
                "created_at": result.created_at.isoformat(),
            }
        except Exception as exc:
            log.error("learn_file_error", error=str(exc))
            return {"error": "File ingestion failed", "code": "INTERNAL_ERROR"}

    # -- URL ingestion ------------------------------------------------------

    @app.post("/api/v1/learn/url", dependencies=deps)
    async def learn_url(request: Request) -> dict[str, Any]:
        """Ingest a website URL.

        JSON body: {"url": "https://...", "description": "optional", "priority": "normal"}
        """
        svc = _get_ingest()
        if not svc:
            return {"error": "Knowledge ingest service not initialized", "status": 503}

        try:
            body = await request.json()
            url = body.get("url", "").strip()
            if not url:
                return {"error": "Field 'url' is required", "code": "MISSING_FIELD"}

            priority_str = body.get("priority", "normal")
            from cognithor.learning.knowledge_ingest import Priority

            priority = Priority.from_string(priority_str)
            result = await svc.ingest_url(url, priority=priority)

            return {
                "id": result.id,
                "source_type": result.source_type,
                "source_name": result.source_name,
                "status": result.status,
                "chunks_created": result.chunks_created,
                "chunks": result.chunks,
                "deep_learn_status": result.deep_learn_status,
                "text_length": result.text_length,
                "error": result.error,
                "created_at": result.created_at.isoformat(),
            }
        except Exception as exc:
            log.error("learn_url_error", error=str(exc))
            return {"error": "URL ingestion failed", "code": "INTERNAL_ERROR"}

    # -- YouTube ingestion --------------------------------------------------

    @app.post("/api/v1/learn/youtube", dependencies=deps)
    async def learn_youtube(request: Request) -> dict[str, Any]:
        """Ingest a YouTube video transcript.

        JSON body: {"url": "https://youtube.com/watch?v=...", "priority": "normal"}
        """
        svc = _get_ingest()
        if not svc:
            return {"error": "Knowledge ingest service not initialized", "status": 503}

        try:
            body = await request.json()
            url = body.get("url", "").strip()
            if not url:
                return {"error": "Field 'url' is required", "code": "MISSING_FIELD"}

            priority_str = body.get("priority", "normal")
            from cognithor.learning.knowledge_ingest import Priority

            priority = Priority.from_string(priority_str)
            result = await svc.ingest_youtube(url, priority=priority)

            return {
                "id": result.id,
                "source_type": result.source_type,
                "source_name": result.source_name,
                "status": result.status,
                "chunks_created": result.chunks_created,
                "chunks": result.chunks,
                "deep_learn_status": result.deep_learn_status,
                "text_length": result.text_length,
                "error": result.error,
                "created_at": result.created_at.isoformat(),
            }
        except Exception as exc:
            log.error("learn_youtube_error", error=str(exc))
            return {"error": "YouTube ingestion failed", "code": "INTERNAL_ERROR"}

    # -- Queue status -------------------------------------------------------

    @app.get("/api/v1/learn/queue", dependencies=deps)
    async def learn_queue() -> dict[str, Any]:
        """Show pending deep-learn tasks."""
        svc = _get_ingest()
        if not svc:
            return {"error": "Knowledge ingest service not initialized", "status": 503}
        return {"queue": svc._queue.pending(), "size": len(svc._queue)}

    # -- History & Stats ----------------------------------------------------

    @app.get("/api/v1/learn/history", dependencies=deps)
    async def learn_history(request: Request) -> dict[str, Any]:
        """List ingestion results."""
        svc = _get_ingest()
        if not svc:
            return {"error": "Knowledge ingest service not initialized", "status": 503}

        limit = int(request.query_params.get("limit", "50"))
        results = svc.results
        # Return most recent first
        recent = list(reversed(results))[:limit]

        return {
            "results": [
                {
                    "id": r.id,
                    "source_type": r.source_type,
                    "source_name": r.source_name,
                    "status": r.status,
                    "chunks_created": r.chunks_created,
                    "text_length": r.text_length,
                    "error": r.error,
                    "created_at": r.created_at.isoformat(),
                }
                for r in recent
            ],
            "count": len(results),
            "stats": svc.stats(),
        }

    @app.get("/api/v1/learn/stats", dependencies=deps)
    async def learn_stats() -> dict[str, Any]:
        """Return ingestion statistics."""
        svc = _get_ingest()
        if not svc:
            return {"error": "Knowledge ingest service not initialized", "status": 503}

        return svc.stats()


# ======================================================================
# Hermes / agentskills.io compatibility routes
# ======================================================================


def _register_skill_registry_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """Expose the built-in skill registry (not marketplace) via REST."""

    def _get_registry() -> Any:
        return getattr(gateway, "_skill_registry", None) if gateway else None

    @app.get("/api/v1/skill-registry/list", dependencies=deps)
    async def list_registry_skills() -> dict[str, Any]:
        reg = _get_registry()
        if not reg:
            return {"installed": [], "count": 0}
        skills = reg.list_all()
        return {
            "installed": [
                {
                    "name": s.name,
                    "slug": s.slug,
                    "description": s.description,
                    "category": s.category,
                    "enabled": s.enabled,
                    "source": getattr(s, "source", "builtin"),
                    "version": getattr(s, "version", "1.0.0"),
                    "author": getattr(s, "author", ""),
                    "total_uses": s.total_uses,
                    "success_rate": round(s.success_rate, 2) if s.total_uses > 0 else None,
                }
                for s in skills
            ],
            "count": len(skills),
        }

    @app.get("/api/v1/skill-registry/{slug}", dependencies=deps)
    async def get_skill_detail(slug: str) -> dict[str, Any]:
        """Get full skill detail including body."""
        reg = _get_registry()
        if not reg:
            raise HTTPException(404, "Skill registry not available")
        skill = reg.get(slug)
        if not skill:
            raise HTTPException(404, f"Skill '{slug}' not found")
        return {
            "name": skill.name,
            "slug": skill.slug,
            "description": skill.description,
            "category": skill.category,
            "trigger_keywords": skill.trigger_keywords,
            "tools_required": skill.tools_required,
            "priority": skill.priority,
            "enabled": skill.enabled,
            "model_preference": skill.model_preference,
            "agent": skill.agent,
            "body": skill.body,
            "source": getattr(skill, "source", "builtin"),
            "file_path": str(skill.file_path),
            "total_uses": skill.total_uses,
            "success_count": skill.success_count,
            "failure_count": skill.failure_count,
            "success_rate": round(skill.success_rate, 2) if skill.total_uses > 0 else None,
            "avg_score": skill.avg_score,
            "last_used": skill.last_used,
        }

    @app.post("/api/v1/skill-registry/create", dependencies=deps)
    async def create_skill(request: Request) -> dict[str, Any]:
        """Create a new skill from JSON body."""
        import re
        from pathlib import Path

        import yaml

        body = await request.json()
        name = body.get("name", "").strip()
        if not name:
            raise HTTPException(400, "Name is required")

        reg = _get_registry()
        if not reg:
            raise HTTPException(503, "Skill registry not available")

        # Generate slug from name
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        if not slug:
            raise HTTPException(400, "Invalid skill name")

        # Check for duplicate
        if reg.get(slug):
            raise HTTPException(409, f"Skill '{slug}' already exists")

        # Build skill file content
        frontmatter = {
            "name": name,
            "description": body.get("description", ""),
            "category": body.get("category", "general"),
            "trigger_keywords": body.get("trigger_keywords", []),
            "tools_required": body.get("tools_required", []),
            "priority": body.get("priority", 0),
            "enabled": body.get("enabled", True),
        }
        if body.get("model_preference"):
            frontmatter["model_preference"] = body["model_preference"]
        if body.get("agent"):
            frontmatter["agent"] = body["agent"]

        skill_body = body.get("body", "")
        front_yaml = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)
        content = f"---\n{front_yaml}---\n\n{skill_body}\n"

        # Determine save directory
        config = getattr(gateway, "_config", None)
        cognithor_home = Path(getattr(config, "cognithor_home", Path.home() / ".cognithor"))
        skills_dir = cognithor_home / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        file_path = skills_dir / f"{slug}.md"

        file_path.write_text(content, encoding="utf-8")

        # Reload registry
        reg.load_from_directories([skills_dir, Path("data/procedures")])

        return {"status": "created", "slug": slug, "file_path": str(file_path)}

    @app.put("/api/v1/skill-registry/{slug}", dependencies=deps)
    async def update_skill(slug: str, request: Request) -> dict[str, Any]:
        """Update an existing skill (metadata and/or body)."""
        import yaml

        reg = _get_registry()
        if not reg:
            raise HTTPException(503, "Skill registry not available")

        skill = reg.get(slug)
        if not skill:
            raise HTTPException(404, f"Skill '{slug}' not found")

        body = await request.json()

        # Read current file
        file_path = skill.file_path
        if not file_path.exists():
            raise HTTPException(500, "Skill file not found on disk")

        # Build updated frontmatter
        frontmatter = {
            "name": body.get("name", skill.name),
            "description": body.get("description", skill.description),
            "category": body.get("category", skill.category),
            "trigger_keywords": body.get("trigger_keywords", skill.trigger_keywords),
            "tools_required": body.get("tools_required", skill.tools_required),
            "priority": body.get("priority", skill.priority),
            "enabled": body.get("enabled", skill.enabled),
        }
        if skill.model_preference or body.get("model_preference"):
            frontmatter["model_preference"] = body.get("model_preference", skill.model_preference)
        if skill.agent or body.get("agent"):
            frontmatter["agent"] = body.get("agent", skill.agent)
        # Preserve stats
        frontmatter["success_count"] = skill.success_count
        frontmatter["failure_count"] = skill.failure_count
        frontmatter["total_uses"] = skill.total_uses
        frontmatter["avg_score"] = skill.avg_score
        if skill.last_used:
            frontmatter["last_used"] = skill.last_used

        skill_body = body.get("body", skill.body)
        front_yaml = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)
        content = f"---\n{front_yaml}---\n\n{skill_body}\n"

        file_path.write_text(content, encoding="utf-8")

        # Reload registry from the skill's parent directory
        config = getattr(gateway, "_config", None)
        cognithor_home = Path(getattr(config, "cognithor_home", Path.home() / ".cognithor"))
        reg.load_from_directories([cognithor_home / "skills", Path("data/procedures")])

        return {"status": "updated", "slug": slug}

    @app.delete("/api/v1/skill-registry/{slug}", dependencies=deps)
    async def delete_skill(slug: str) -> dict[str, Any]:
        """Delete a skill file from disk and reload registry."""
        reg = _get_registry()
        if not reg:
            raise HTTPException(503, "Skill registry not available")

        skill = reg.get(slug)
        if not skill:
            raise HTTPException(404, f"Skill '{slug}' not found")

        # Don't allow deleting built-in procedure skills
        file_path = skill.file_path
        if "data/procedures" in str(file_path):
            raise HTTPException(403, "Cannot delete built-in procedure skills")

        if file_path.exists():
            file_path.unlink()

        # Reload registry
        config = getattr(gateway, "_config", None)
        cognithor_home = Path(getattr(config, "cognithor_home", Path.home() / ".cognithor"))
        reg.load_from_directories([cognithor_home / "skills", Path("data/procedures")])

        return {"status": "deleted", "slug": slug}

    @app.put("/api/v1/skill-registry/{slug}/toggle", dependencies=deps)
    async def toggle_skill(slug: str) -> dict[str, Any]:
        """Enable or disable a skill."""
        reg = _get_registry()
        if not reg:
            raise HTTPException(503, "Skill registry not available")

        skill = reg.get(slug)
        if not skill:
            raise HTTPException(404, f"Skill '{slug}' not found")

        if skill.enabled:
            reg.disable(slug)
            return {"slug": slug, "enabled": False}
        else:
            reg.enable(slug)
            return {"slug": slug, "enabled": True}

    @app.get("/api/v1/skill-registry/{slug}/export", dependencies=deps)
    async def export_skill_md(slug: str) -> dict[str, Any]:
        """Export a skill in SKILL.md (agentskills.io) format."""
        reg = _get_registry()
        if not reg:
            raise HTTPException(503, "Skill registry not available")

        skill = reg.get(slug)
        if not skill:
            raise HTTPException(404, f"Skill '{slug}' not found")

        try:
            from cognithor.skills.hermes_compat import HermesCompatLayer

            skill_dict = {
                "name": skill.name,
                "description": skill.description,
                "tags": skill.trigger_keywords,
                "prompt": skill.body,
                "version": "1.0.0",
            }
            hermes = HermesCompatLayer.cognithor_to_hermes(skill_dict)
            content = HermesCompatLayer.to_skill_md(hermes)
            return {"slug": slug, "skill_md": content}
        except Exception as e:
            raise HTTPException(500, f"Export failed: {e}") from e


def _register_hermes_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """REST endpoints for Hermes SKILL.md import/export."""

    def _get_compat() -> Any:
        return getattr(gateway, "_hermes_compat", None) if gateway else None

    def _get_registry() -> Any:
        return getattr(gateway, "_skill_registry", None) if gateway else None

    @app.get("/api/v1/skills/hermes/export/{skill_name}", dependencies=deps)
    async def hermes_export_skill(skill_name: str) -> dict[str, Any]:
        """Export a Cognithor skill to SKILL.md format."""
        compat = _get_compat()
        if not compat:
            return {"error": "Hermes compatibility layer not initialized", "status": 503}

        registry = _get_registry()
        if not registry:
            return {"error": "Skill registry not initialized", "status": 503}

        skill = registry.get(skill_name)
        if not skill:
            return {"error": f"Skill '{skill_name}' not found", "status": 404}

        # Convert Cognithor Skill dataclass to dict for conversion
        skill_dict = {
            "name": skill.name,
            "description": skill.description,
            "tags": [],
            "prompt": skill.body,
            "version": "1.0.0",
            "author": "cognithor",
        }
        if skill.manifest:
            skill_dict["version"] = skill.manifest.version
            skill_dict["author"] = skill.manifest.author_github or "cognithor"

        hermes_skill = compat.cognithor_to_hermes(skill_dict)
        content = compat.to_skill_md(hermes_skill)
        return {"skill_name": skill_name, "skill_md": content}

    @app.post("/api/v1/skills/hermes/import", dependencies=deps)
    async def hermes_import_skill(request: Request) -> dict[str, Any]:
        """Import a SKILL.md into Cognithor."""
        compat = _get_compat()
        if not compat:
            return {"error": "Hermes compatibility layer not initialized", "status": 503}

        body = await request.json()
        content = body.get("content", "")
        if not content:
            return {"error": "Missing 'content' field with SKILL.md text", "status": 400}

        try:
            hermes_skill = compat.parse_skill_md(content)
        except ValueError as exc:
            return {"error": f"Invalid SKILL.md format: {exc}", "status": 400}

        cognithor_dict = compat.hermes_to_cognithor(hermes_skill)
        return {
            "imported": True,
            "skill": cognithor_dict,
        }


# ======================================================================
# Self-improvement routes
# ======================================================================


def _register_self_improvement_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """REST endpoints for the self-improvement engine."""

    def _get_improver() -> Any:
        return getattr(gateway, "_self_improver", None) if gateway else None

    @app.get("/api/v1/learning/self-improvement/stats", dependencies=deps)
    async def self_improvement_stats() -> dict[str, Any]:
        """Return self-improvement engine statistics."""
        improver = _get_improver()
        if not improver:
            return {"error": "Self-improvement engine not initialized", "status": 503}
        return improver.stats()

    @app.get("/api/v1/learning/self-improvement/pending", dependencies=deps)
    async def self_improvement_pending() -> dict[str, Any]:
        """Return pending improvement proposals."""
        improver = _get_improver()
        if not improver:
            return {"error": "Self-improvement engine not initialized", "status": 503}

        pending = improver.pending_improvements
        return {
            "pending": [
                {
                    "id": imp.id,
                    "pattern_id": imp.pattern_id,
                    "improvement_type": imp.improvement_type,
                    "before": imp.before,
                    "after": imp.after,
                    "confidence": round(imp.confidence, 3),
                    "created_at": imp.created_at.isoformat(),
                }
                for imp in pending
            ],
            "count": len(pending),
        }

    @app.post(
        "/api/v1/learning/self-improvement/{improvement_id}/apply",
        dependencies=deps,
    )
    async def self_improvement_apply(improvement_id: str) -> dict[str, Any]:
        """Apply a pending improvement."""
        improver = _get_improver()
        if not improver:
            return {"error": "Self-improvement engine not initialized", "status": 503}

        success = improver.apply_improvement(improvement_id)
        if not success:
            return {"error": f"Improvement '{improvement_id}' not found", "status": 404}

        return {"applied": True, "improvement_id": improvement_id}


# ======================================================================
# GEPA Evolution routes
# ======================================================================


def _register_gepa_evolution_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """REST endpoints for GEPA (Guided Evolution through Pattern Analysis)."""

    def _get_orch() -> Any:
        return getattr(gateway, "_evolution_orchestrator", None) if gateway else None

    def _get_trace_store() -> Any:
        return getattr(gateway, "_trace_store", None) if gateway else None

    def _get_proposal_store() -> Any:
        return getattr(gateway, "_proposal_store", None) if gateway else None

    @app.get("/api/v1/evolution/status", dependencies=deps)
    async def get_evolution_status() -> dict[str, Any]:
        orch = _get_orch()
        if not orch:
            return {"enabled": False, "message": "GEPA not enabled"}
        return orch.get_status()

    @app.get("/api/v1/learning/gepa/status", dependencies=deps)
    async def gepa_status() -> dict[str, Any]:
        """Get GEPA evolution cycle status (alias under /learning/)."""
        orch = _get_orch()
        if orch is None:
            return {"status": "not_initialized"}
        try:
            status = orch.get_status()
            return {"status": "ok", **status}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    @app.get("/api/v1/evolution/proposals", dependencies=deps)
    async def list_evolution_proposals(status: str = "all") -> dict[str, Any]:
        ps = _get_proposal_store()
        if not ps:
            return {"proposals": []}
        if status == "all":
            proposals = ps.get_history(limit=50)
        else:
            proposals = ps.get_by_status(status)
        return {"proposals": [_proposal_to_dict(p) for p in proposals]}

    @app.get("/api/v1/evolution/proposals/{proposal_id}", dependencies=deps)
    async def get_evolution_proposal(proposal_id: str) -> dict[str, Any]:
        orch = _get_orch()
        if not orch:
            return {"error": "GEPA not enabled", "status": 404}
        detail = orch.get_proposal_detail(proposal_id)
        if not detail:
            return {"error": "Proposal not found", "status": 404}
        return detail

    @app.post("/api/v1/evolution/proposals/{proposal_id}/apply", dependencies=deps)
    async def apply_evolution_proposal(proposal_id: str) -> dict[str, Any]:
        orch = _get_orch()
        if not orch:
            return {"error": "GEPA not enabled", "status": 404}
        ok = orch.apply_proposal(proposal_id)
        return {"applied": ok, "proposal_id": proposal_id}

    @app.post("/api/v1/evolution/proposals/{proposal_id}/reject", dependencies=deps)
    async def reject_evolution_proposal(proposal_id: str) -> dict[str, Any]:
        orch = _get_orch()
        if not orch:
            return {"error": "GEPA not enabled", "status": 404}
        ok = orch.reject_proposal(proposal_id)
        return {"rejected": ok, "proposal_id": proposal_id}

    @app.post("/api/v1/evolution/proposals/{proposal_id}/rollback", dependencies=deps)
    async def rollback_evolution_proposal(proposal_id: str) -> dict[str, Any]:
        orch = _get_orch()
        if not orch:
            return {"error": "GEPA not enabled", "status": 404}
        ok = orch.rollback_proposal(proposal_id)
        return {"rolled_back": ok, "proposal_id": proposal_id}

    @app.get("/api/v1/evolution/traces", dependencies=deps)
    async def list_evolution_traces(limit: int = 20) -> dict[str, Any]:
        ts = _get_trace_store()
        if not ts:
            return {"traces": []}
        traces = ts.get_recent_traces(limit=min(limit, 100))
        return {"traces": [_trace_to_dict(t) for t in traces]}

    @app.post("/api/v1/evolution/run", dependencies=deps)
    async def trigger_evolution_cycle() -> dict[str, Any]:
        orch = _get_orch()
        if not orch:
            return {"error": "GEPA not enabled", "status": 404}
        result = orch.run_evolution_cycle()
        return {
            "cycle_id": result.cycle_id,
            "traces_analyzed": result.traces_analyzed,
            "findings": result.findings_count,
            "proposals_generated": result.proposals_generated,
            "proposal_applied": result.proposal_applied,
            "auto_rollbacks": result.auto_rollbacks,
            "duration_ms": result.duration_ms,
        }


def _proposal_to_dict(p: Any) -> dict[str, Any]:
    return {
        "proposal_id": p.proposal_id,
        "optimization_type": p.optimization_type,
        "target": p.target,
        "description": p.description,
        "confidence": p.confidence,
        "estimated_impact": p.estimated_impact,
        "failure_category": p.failure_category,
        "tool_name": p.tool_name,
        "status": p.status,
        "created_at": p.created_at,
        "applied_at": p.applied_at,
    }


def _trace_to_dict(t: Any) -> dict[str, Any]:
    return {
        "trace_id": t.trace_id,
        "session_id": t.session_id,
        "goal": t.goal[:200],
        "success_score": t.success_score,
        "model_used": t.model_used,
        "total_duration_ms": t.total_duration_ms,
        "step_count": len(t.steps),
        "failed_steps": len(t.failed_steps),
        "tool_sequence": t.tool_sequence,
        "created_at": t.created_at,
    }


# ======================================================================
# Backend status / switch routes
# ======================================================================


def _register_backend_routes(
    app: Any,
    deps: list[Any],
    config_manager: ConfigManager,
    gateway: Any,
) -> None:
    """Endpoints for querying LLM backend availability and switching backends."""

    import shutil

    @app.get("/api/v1/backend/status", dependencies=deps)
    async def get_backend_status() -> dict[str, Any]:
        """Check which LLM backends are available and authenticated."""
        results: dict[str, Any] = {}

        # Claude Code CLI
        claude_path = shutil.which("claude")
        results["claude-code"] = {
            "installed": claude_path is not None,
            "path": claude_path or "",
            "authenticated": False,
            "models": [],
        }
        if claude_path:
            try:
                proc = await asyncio.create_subprocess_exec(
                    claude_path,
                    "--version",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
                if proc.returncode == 0:
                    results["claude-code"]["authenticated"] = True
                    results["claude-code"]["version"] = stdout.decode().strip()
                    results["claude-code"]["models"] = [
                        "opus",
                        "sonnet",
                        "haiku",
                    ]
            except Exception:
                log.debug("claude_code_provider_check_failed", exc_info=True)

        # Ollama
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                resp = await client.get("http://localhost:11434/api/tags", timeout=3)
                if resp.status_code == 200:
                    models = [m["name"] for m in resp.json().get("models", [])]
                    results["ollama"] = {
                        "installed": True,
                        "authenticated": True,
                        "models": models,
                    }
                else:
                    results["ollama"] = {
                        "installed": True,
                        "authenticated": False,
                        "models": [],
                    }
        except Exception:
            results["ollama"] = {
                "installed": False,
                "authenticated": False,
                "models": [],
            }

        # OpenAI (check if key is set)
        cfg = config_manager.config if config_manager else None
        if cfg:
            has_openai = bool(getattr(cfg, "openai_api_key", ""))
            results["openai"] = {
                "installed": True,
                "authenticated": has_openai,
                "models": [],
            }

            has_anthropic = bool(getattr(cfg, "anthropic_api_key", ""))
            results["anthropic"] = {
                "installed": True,
                "authenticated": has_anthropic,
                "models": [],
            }

            has_openrouter = bool(getattr(cfg, "openrouter_api_key", ""))
            results["openrouter"] = {
                "installed": True,
                "authenticated": has_openrouter,
                "models": [],
            }

        # Current backend
        current = getattr(cfg, "llm_backend_type", "ollama") if cfg else "ollama"

        return {"backends": results, "current": current}

    @app.post("/api/v1/backend/switch", dependencies=deps)
    async def switch_backend(request: Request) -> dict[str, Any]:
        """Switch the LLM backend type."""
        body = await request.json()
        new_backend = body.get("backend", "")

        # Derive valid backends from the config Literal type (single source of truth)
        from typing import get_args, get_type_hints

        from cognithor.config import CognithorConfig

        _hints = get_type_hints(CognithorConfig, include_extras=True)
        valid = list(get_args(_hints["llm_backend_type"]))
        if new_backend not in valid:
            raise HTTPException(400, f"Invalid backend: {new_backend}. Valid: {valid}")

        # Update config
        if config_manager:
            config_manager.config.llm_backend_type = new_backend
            # Save to config.yaml
            with contextlib.suppress(Exception):
                config_manager.save()

        return {
            "status": "switched",
            "backend": new_backend,
            "note": "Restart required for full effect",
        }


# ======================================================================
# Autonomous Task Orchestration routes
# ======================================================================


def _register_autonomous_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """Endpoints for querying autonomous task execution status."""

    @app.get("/api/v1/autonomous/tasks", dependencies=deps)
    async def list_autonomous_tasks() -> dict[str, Any]:
        """List active autonomous tasks."""
        if not hasattr(gateway, "_autonomous_orchestrator"):
            return {"tasks": []}
        return {"tasks": gateway._autonomous_orchestrator.get_active_tasks()}


# ======================================================================
# Feedback routes (thumbs up/down)
# ======================================================================


def _register_feedback_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """REST endpoints for user feedback (thumbs up/down)."""

    @app.post("/api/v1/feedback", dependencies=deps)
    async def submit_feedback(request: Request) -> dict[str, Any]:
        """Submit thumbs up/down feedback for a message."""
        body = await request.json()
        feedback_store = getattr(gateway, "_feedback_store", None)
        if not feedback_store:
            return {"error": "Feedback system not initialized"}

        rating = body.get("rating", 0)
        if rating not in (1, -1):
            return {"error": "rating must be 1 (thumbs up) or -1 (thumbs down)"}

        feedback_id = feedback_store.submit(
            session_id=body.get("session_id", ""),
            message_id=body.get("message_id", ""),
            rating=rating,
            comment=body.get("comment", ""),
            agent_name=body.get("agent_name", "jarvis"),
            channel=body.get("channel", "webui"),
            user_message=body.get("user_message", ""),
            assistant_response=body.get("assistant_response", ""),
            tool_calls=body.get("tool_calls", ""),
        )
        return {"status": "ok", "feedback_id": feedback_id}

    @app.patch("/api/v1/feedback/{feedback_id}", dependencies=deps)
    async def update_feedback_comment(feedback_id: str, request: Request) -> dict[str, Any]:
        """Add comment to existing feedback (after follow-up question)."""
        body = await request.json()
        feedback_store = getattr(gateway, "_feedback_store", None)
        if not feedback_store:
            return {"error": "Feedback system not initialized"}

        ok = feedback_store.add_comment(feedback_id, body.get("comment", ""))
        return {"status": "ok" if ok else "not_found"}

    @app.get("/api/v1/feedback/stats", dependencies=deps)
    async def feedback_stats(agent_name: str = "", hours: int = 0) -> dict[str, Any]:
        """Get feedback statistics."""
        feedback_store = getattr(gateway, "_feedback_store", None)
        if not feedback_store:
            return {"total": 0, "positive": 0, "negative": 0, "satisfaction_rate": 0}
        return feedback_store.get_stats(agent_name=agent_name, hours=hours)

    @app.get("/api/v1/feedback/recent", dependencies=deps)
    async def recent_feedback(limit: int = 50) -> dict[str, Any]:
        """Get recent feedback entries."""
        feedback_store = getattr(gateway, "_feedback_store", None)
        if not feedback_store:
            return {"entries": []}
        return {"entries": feedback_store.get_recent(limit=limit)}

    # ── Chat Tree / Branching ────────────────────────────────────────

    @app.get("/api/v1/chat/tree/latest", dependencies=deps)
    async def get_latest_chat_tree(session_id: str = "") -> dict[str, Any]:
        """Get the most recent conversation tree.

        If session_id is provided, look up the session's persisted
        conversation_id first so the correct tree is returned.
        """
        tree = getattr(gateway, "_conversation_tree", None)
        if not tree:
            return {"nodes": [], "conversation_id": None}

        conv_id = None

        # Try session-specific conversation first
        if session_id:
            store = getattr(gateway, "_session_store", None)
            if store:
                session = store.load_session_by_id(session_id)
                if session and getattr(session, "conversation_id", ""):
                    conv_id = session.conversation_id

        # Fallback: most recent conversation
        if not conv_id:
            with tree._conn() as conn:
                row = conn.execute(
                    "SELECT id FROM conversations ORDER BY updated_at DESC LIMIT 1"
                ).fetchone()
                if row:
                    conv_id = row["id"]

        if not conv_id:
            return {"nodes": [], "conversation_id": None}
        return tree.get_tree_structure(conv_id)

    @app.get("/api/v1/chat/tree/{conversation_id}", dependencies=deps)
    async def get_chat_tree(conversation_id: str) -> dict[str, Any]:
        """Get full conversation tree structure."""
        tree = getattr(gateway, "_conversation_tree", None)
        if not tree:
            return {"error": "Conversation tree not available"}
        return tree.get_tree_structure(conversation_id)

    @app.get("/api/v1/chat/path/{conversation_id}/{leaf_id}", dependencies=deps)
    async def get_chat_path(conversation_id: str, leaf_id: str) -> dict[str, Any]:
        """Get path from root to a specific leaf."""
        tree = getattr(gateway, "_conversation_tree", None)
        if not tree:
            return {"error": "Conversation tree not available"}
        path = tree.get_path_to_root(leaf_id)
        return {"path": path, "count": len(path)}

    @app.post("/api/v1/chat/branch", dependencies=deps)
    async def create_chat_branch(request: Request) -> dict[str, Any]:
        """Create a branch at a specific node."""
        body = await request.json()
        tree = getattr(gateway, "_conversation_tree", None)
        if not tree:
            return {"error": "Conversation tree not available"}
        conv_id = body.get("conversation_id", "")
        parent_id = body.get("parent_id", "")
        text = body.get("text", "")
        role = body.get("role", "user")
        if not conv_id or not text:
            return {"error": "conversation_id and text required"}
        node_id = tree.add_node(conv_id, role=role, text=text, parent_id=parent_id or None)
        return {
            "node_id": node_id,
            "branch_index": tree.get_branch_index(node_id),
        }


# ======================================================================
# Social / Reddit Lead Hunter routes
# ======================================================================


def _register_social_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """REST endpoints for Reddit Lead Hunter."""

    def _get_service() -> Any:
        # Prefer the new source-agnostic LeadService; fall back to legacy alias.
        return (
            (
                getattr(gateway, "_leads_service", None)
                or getattr(gateway, "_reddit_lead_service", None)
            )
            if gateway
            else None
        )

    @app.get("/api/v1/leads/engine-status", dependencies=deps)
    async def leads_engine_status() -> dict[str, Any]:
        """Return which lead sources are enabled. Frontend uses this to gate the sidebar tab."""
        social_cfg = getattr(getattr(gateway, "_config", None), "social", None)
        if social_cfg is None:
            return {"enabled": False, "sources": {}}
        return {
            "enabled": bool(getattr(social_cfg, "leads_engine_enabled", False)),
            "sources": {
                "reddit": bool(getattr(social_cfg, "reddit_scan_enabled", False)),
                "hackernews": bool(getattr(social_cfg, "hn_enabled", False)),
                "discord": bool(getattr(social_cfg, "discord_scanner_enabled", False)),
                "rss": bool(getattr(social_cfg, "rss_enabled", False)),
            },
        }

    @app.get("/api/v1/packs/loaded", dependencies=deps)
    async def list_loaded_packs() -> dict[str, Any]:
        """Return currently loaded packs for Flutter tab gating."""
        loader = getattr(gateway, "_pack_loader", None)
        if loader is None:
            return {"packs": []}
        return {
            "packs": [
                {
                    "qualified_id": p.manifest.qualified_id,
                    "version": p.manifest.version,
                    "display_name": p.manifest.display_name,
                    "tools": p.manifest.tools,
                    "lead_sources": p.manifest.lead_sources,
                }
                for p in loader.loaded()
            ]
        }

    @app.get("/api/v1/leads/sources", dependencies=deps)
    async def list_lead_sources() -> dict[str, Any]:
        """Return registered LeadSource metadata.

        Feeds Flutter's LeadsScreen + locked-pack-card UX. An empty list
        means the backend has no lead sources and the sidebar tab should
        be hidden by the frontend.
        """
        svc = _get_service()
        if svc is None:
            return {"sources": []}
        # LeadService.list_sources() returns registered LeadSource instances.
        sources: list[dict[str, Any]] = []
        try:
            for source in svc.list_sources():
                sources.append(
                    {
                        "source_id": source.source_id,
                        "display_name": source.display_name,
                        "icon": getattr(source, "icon", ""),
                        "color": getattr(source, "color", ""),
                        "capabilities": sorted(getattr(source, "capabilities", [])),
                    }
                )
        except Exception:
            pass
        return {"sources": sources}

    @app.post("/api/v1/leads/scan/rss", dependencies=deps)
    async def scan_leads_rss(request: Request) -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead service not initialized", "status": 503}
        try:
            body = await request.json()
        except Exception:
            body = {}
        social_cfg = getattr(getattr(gateway, "_config", None), "social", None)
        feeds = body.get("feeds") or (
            list(getattr(social_cfg, "rss_feeds", [])) if social_cfg else []
        )
        min_score = int(
            body.get("min_score")
            or (getattr(social_cfg, "rss_min_score", 60) if social_cfg else 60)
        )
        if not feeds:
            return {"error": "No RSS feeds configured", "leads_found": 0, "posts_checked": 0}
        # Delegate to the rss-lead-hunter pack source (source_id="rss") if registered.
        try:
            result = await svc.scan(
                source_id="rss",
                min_score=min_score,
                config={"rss": {"feeds": feeds}},
                trigger="ui",
            )
            return {
                "id": result.id,
                "summary": result.summary(),
                "posts_checked": result.posts_checked,
                "leads_found": result.leads_found,
            }
        except ValueError:
            return {
                "error": "RSS source not registered — install rss-lead-hunter pack",
                "leads_found": 0,
                "posts_checked": 0,
            }

    @app.post("/api/v1/leads/scan", dependencies=deps)
    async def scan_leads(request: Request) -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead Service not initialized", "status": 503}
        try:
            body = await request.json()
        except Exception:
            body = {}
        min_score = int(body.get("min_score") or 0)
        source_id = body.get("source_id") or body.get("platform") or None
        social_cfg = getattr(getattr(gateway, "_config", None), "social", None)
        product = body.get("product") or getattr(social_cfg, "reddit_product_name", "") or ""
        result = await svc.scan(
            source_id=source_id,
            min_score=min_score,
            product=product,
            trigger="ui",
        )
        return {
            "id": result.id,
            "summary": result.summary(),
            "posts_checked": result.posts_checked,
            "leads_found": result.leads_found,
        }

    @app.get("/api/v1/leads", dependencies=deps)
    async def list_leads(
        status: str | None = None,
        min_score: int = 0,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead Service not initialized", "status": 503}
        from cognithor.leads.models import LeadStatus

        status_filter = None
        if status and status in [s.value for s in LeadStatus]:
            status_filter = LeadStatus(status)
        leads = svc.get_leads(status=status_filter, min_score=min_score, limit=limit, offset=offset)
        return {
            "leads": [l.to_dict() for l in leads],
            "count": len(leads),
        }

    @app.get("/api/v1/leads/stats", dependencies=deps)
    async def lead_stats() -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead Service not initialized", "status": 503}
        stats = svc.get_stats()
        history = svc.get_scan_history(limit=10)
        return {
            "stats": {
                "total": stats.total,
                "new": stats.new,
                "reviewed": stats.reviewed,
                "replied": stats.replied,
                "archived": stats.archived,
                "avg_score": stats.avg_score,
                "top_subreddits": stats.top_subreddits,
                "total_scans": stats.total_scans,
            },
            "recent_scans": history,
        }

    @app.post("/api/v1/leads/discover-subreddits", dependencies=deps)
    async def discover_subreddits(request: Request) -> dict[str, Any]:
        # Subreddit discovery is provided by the reddit-lead-hunter-pro pack.
        # Without the pack installed, this endpoint returns an empty suggestion list.
        return {
            "suggestions": [],
            "note": "Install reddit-lead-hunter-pro pack to enable subreddit discovery.",
        }

    @app.get("/api/v1/leads/templates", dependencies=deps)
    async def list_templates(subreddit: str = "") -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead Service not initialized", "status": 503}
        return {"templates": svc.get_templates(subreddit)}

    @app.post("/api/v1/leads/templates", dependencies=deps)
    async def create_template(request: Request) -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead Service not initialized", "status": 503}
        body = await request.json()
        tid = svc.create_template(
            name=body.get("name", ""),
            text=body.get("text", ""),
            subreddit=body.get("subreddit", ""),
            style=body.get("style", ""),
        )
        return {"id": tid, "status": "created"}

    @app.delete("/api/v1/leads/templates/{template_id}", dependencies=deps)
    async def delete_template(template_id: str) -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead Service not initialized", "status": 503}
        svc.delete_template(template_id)
        return {"status": "deleted"}

    @app.get("/api/v1/leads/{lead_id}", dependencies=deps)
    async def get_lead(lead_id: str) -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead Service not initialized", "status": 503}
        lead = svc.get_lead(lead_id)
        if lead is None:
            raise HTTPException(404, "Lead not found")
        return lead.to_dict()

    @app.patch("/api/v1/leads/{lead_id}", dependencies=deps)
    async def update_lead(lead_id: str, request: Request) -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead Service not initialized", "status": 503}
        body = await request.json()
        from cognithor.leads.models import LeadStatus

        status = LeadStatus(body["status"]) if "status" in body else None
        reply_final = body.get("reply_final")
        lead = svc.update_lead(lead_id, status=status, reply_final=reply_final)
        if lead is None:
            raise HTTPException(404, "Lead not found")
        return lead.to_dict()

    @app.post("/api/v1/leads/{lead_id}/reply", dependencies=deps)
    async def reply_to_lead(lead_id: str, request: Request) -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead Service not initialized", "status": 503}
        try:
            body = await request.json()
        except Exception:
            body = {}
        mode = body.get("mode", "clipboard")
        result = svc.post_reply(lead_id, mode=mode)
        return {
            "success": result.success,
            "mode": result.mode.value if hasattr(result.mode, "value") else result.mode,
            "error": result.error,
        }

    @app.post("/api/v1/leads/{lead_id}/refine", dependencies=deps)
    async def refine_lead(lead_id: str, request: Request) -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead Service not initialized", "status": 503}
        try:
            body = await request.json()
        except Exception:
            body = {}
        hint = body.get("hint", "")
        variants = body.get("variants", 0)
        result = await svc.refine_reply(lead_id, hint=hint, variants=variants)
        if result is None:
            raise HTTPException(404, "Lead not found")
        if isinstance(result, list):
            return {"variants": [{"text": r.text, "style": r.style} for r in result]}
        return {"text": result.text, "style": result.style, "changes": result.changes_summary}

    @app.get("/api/v1/leads/{lead_id}/performance", dependencies=deps)
    async def get_lead_performance(lead_id: str) -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead Service not initialized", "status": 503}
        perf = svc.get_performance(lead_id)
        if perf is None:
            return {"performance": None}

        def _engagement_score(upvotes: int, replies: int, author_replied: bool, tag: str) -> float:
            """Inline engagement score — simple weighted heuristic."""
            score = min(upvotes * 0.3 + replies * 0.5, 80.0)
            if author_replied:
                score += 15.0
            if tag == "good":
                score += 5.0
            elif tag == "bad":
                score -= 10.0
            return round(max(0.0, min(score, 100.0)), 1)

        perf["engagement_score"] = _engagement_score(
            perf.get("reply_upvotes", 0),
            perf.get("reply_replies", 0),
            bool(perf.get("author_replied", 0)),
            perf.get("feedback_tag", ""),
        )
        return {"performance": perf}

    @app.patch("/api/v1/leads/{lead_id}/feedback", dependencies=deps)
    async def set_lead_feedback(lead_id: str, request: Request) -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead Service not initialized", "status": 503}
        body = await request.json()
        svc.set_feedback(lead_id, tag=body.get("tag", ""), note=body.get("note", ""))
        return {"status": "ok"}
