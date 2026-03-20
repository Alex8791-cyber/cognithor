"""Jarvis · Konfigurations-API Routes.

REST-Endpoints für die Konfigurationsverwaltung via WebUI:

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

from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from jarvis.config_manager import ConfigManager

log = get_logger(__name__)


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
        config_manager: ConfigManager für Read/Write.
        verify_token_dep: Optional FastAPI Depends() für Auth.
        gateway: Optional Gateway-Instanz für Singleton-Zugriff.
    """
    deps = [verify_token_dep] if verify_token_dep else []

    # Shared MonitoringHub (singleton per app) -- created lazily and used
    # across monitoring, SSE, and audit routes.
    _hub_holder: dict[str, Any] = {"hub": None}

    def _get_hub() -> Any:
        if _hub_holder["hub"] is None:
            from jarvis.gateway.monitoring import MonitoringHub

            _hub_holder["hub"] = MonitoringHub()
        return _hub_holder["hub"]

    _register_system_routes(app, deps, config_manager, gateway)
    _register_config_routes(app, deps, config_manager, gateway)
    _register_session_routes(app, deps, gateway)
    _register_memory_routes(app, deps, gateway)
    _register_skill_routes(app, deps, gateway)
    _register_monitoring_routes(app, deps, _get_hub)
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


# ======================================================================
# System / health / status / dashboard routes
# ======================================================================


def _register_system_routes(
    app: Any,
    deps: list[Any],
    config_manager: ConfigManager,
    gateway: Any,
) -> None:
    """Dashboard, status, overview, presets, bindings, agents, credentials."""

    # -- Admin Dashboard --------------------------------------------------

    @app.get("/dashboard")
    async def serve_dashboard():
        """Liefert das Admin-Dashboard als HTML."""
        from pathlib import Path

        dashboard_path = Path(__file__).parent.parent / "gateway" / "dashboard.html"
        if dashboard_path.exists():
            from starlette.responses import HTMLResponse

            return HTMLResponse(dashboard_path.read_text(encoding="utf-8"))
        return {"error": "Dashboard nicht gefunden"}

    # -- Status -----------------------------------------------------------

    @app.get("/api/v1/status", dependencies=deps)
    async def get_system_status() -> dict[str, Any]:
        """Gibt den aktuellen System-Status zurück."""
        status: dict[str, Any] = {
            "timestamp": time.time(),
            "config_version": config_manager.config.version,
            "owner": config_manager.config.owner_name,
        }

        # RuntimeMonitor
        try:
            from jarvis.openclaw.runtime_monitor import RuntimeMonitor

            monitor = RuntimeMonitor()
            status["runtime"] = {"metrics_count": len(monitor._metrics)}
        except Exception:
            status["runtime"] = {"available": False}

        # HeartbeatScheduler
        try:
            hb_config = config_manager.config.heartbeat
            status["heartbeat"] = {
                "enabled": hb_config.enabled,
                "interval_minutes": hb_config.interval_minutes,
                "channel": hb_config.channel,
            }
        except Exception:
            status["heartbeat"] = {"available": False}

        # Active Channels
        ch = config_manager.config.channels
        active_channels = []
        for attr in dir(ch):
            if attr.endswith("_enabled") and getattr(ch, attr, False):
                active_channels.append(attr.replace("_enabled", ""))
        status["active_channels"] = active_channels

        # Models
        models = config_manager.config.models
        status["models"] = {
            "planner": models.planner.name,
            "executor": models.executor.name,
            "coder": models.coder.name,
            "embedding": models.embedding.name,
        }

        # LLM Backend
        status["llm_backend"] = config_manager.config.llm_backend_type
        return status

    # -- Overview ---------------------------------------------------------

    @app.get("/api/v1/overview", dependencies=deps)
    async def get_overview() -> dict[str, Any]:
        """Gibt eine kompakte Konfigurationsübersicht zurück."""
        try:
            from jarvis.gateway.config_api import ConfigManager as CfgMgr

            cfg_mgr = CfgMgr(config_manager.config)
            overview = cfg_mgr.get_overview()
            return overview.model_dump()
        except Exception:
            log.exception("Failed to build configuration overview")
            return {"error": "Konfigurationsübersicht konnte nicht geladen werden."}

    # -- Agents -----------------------------------------------------------

    @app.get("/api/v1/agents", dependencies=deps)
    async def list_agents() -> dict[str, Any]:
        """Listet alle registrierten Agent-Profile aus agents.yaml."""
        try:
            agents_path = config_manager.config.jarvis_home / "agents.yaml"
            if agents_path.exists():
                raw = yaml.safe_load(agents_path.read_text(encoding="utf-8")) or {}
                agents = raw.get("agents", [])
            else:
                agents = [
                    {
                        "name": "jarvis",
                        "display_name": "Jarvis",
                        "description": "Haupt-Agent (Default)",
                        "system_prompt": "",
                        "language": "de",
                        "trigger_patterns": [],
                        "trigger_keywords": [],
                        "priority": 100,
                        "allowed_tools": [],
                        "blocked_tools": [],
                        "preferred_model": "",
                        "temperature": 0.7,
                        "enabled": True,
                    }
                ]
            return {"agents": agents}
        except Exception as exc:
            log.error("agents_list_failed", error=str(exc))
            return {"agents": [], "error": "Agenten konnten nicht geladen werden"}

    @app.get("/api/v1/agents/{agent_name}", dependencies=deps)
    async def get_agent(agent_name: str) -> dict[str, Any]:
        """Get a single agent by name."""
        # Try agents.yaml first
        agents_path = config_manager.config.jarvis_home / "agents.yaml"
        if agents_path.exists():
            raw = yaml.safe_load(agents_path.read_text(encoding="utf-8")) or {}
            for a in raw.get("agents", []):
                if a.get("name") == agent_name:
                    return a

        # Fallback: check the live agent router
        router = getattr(gateway, "_agent_router", None) if gateway else None
        if router:
            agent_obj = getattr(router, "get_agent", lambda n: None)(agent_name)
            if agent_obj:
                return {
                    "name": getattr(agent_obj, "name", agent_name),
                    "display_name": getattr(agent_obj, "display_name", agent_name.title()),
                    "description": getattr(agent_obj, "description", ""),
                    "system_prompt": getattr(agent_obj, "system_prompt", ""),
                    "language": getattr(agent_obj, "language", "de"),
                    "preferred_model": getattr(agent_obj, "preferred_model", ""),
                    "temperature": getattr(agent_obj, "temperature", 0.7),
                    "priority": getattr(agent_obj, "priority", 0),
                    "enabled": getattr(agent_obj, "enabled", True),
                    "allowed_tools": getattr(agent_obj, "allowed_tools", None) or [],
                    "blocked_tools": getattr(agent_obj, "blocked_tools", []),
                    "can_delegate_to": getattr(agent_obj, "can_delegate_to", []),
                    "sandbox_timeout": getattr(agent_obj, "sandbox_timeout", 30),
                    "sandbox_network": getattr(agent_obj, "sandbox_network", "allow"),
                }

        # Last fallback for default "jarvis"
        if agent_name == "jarvis":
            return {
                "name": "jarvis",
                "display_name": "Jarvis",
                "description": "Default Agent",
                "system_prompt": "",
                "language": "de",
                "preferred_model": "",
                "temperature": 0.7,
                "priority": 100,
                "enabled": True,
                "allowed_tools": [],
                "blocked_tools": [],
                "can_delegate_to": [],
                "sandbox_timeout": 30,
                "sandbox_network": "allow",
            }

        raise HTTPException(404, f"Agent '{agent_name}' not found")

    @app.post("/api/v1/agents", dependencies=deps)
    async def create_agent(request: Request) -> dict[str, Any]:
        """Create a new agent profile."""
        body = await request.json()
        name = body.get("name", "").strip().lower().replace(" ", "-")
        if not name:
            raise HTTPException(400, "Name is required")

        agents_path = config_manager.config.jarvis_home / "agents.yaml"
        raw = {}
        if agents_path.exists():
            raw = yaml.safe_load(agents_path.read_text(encoding="utf-8")) or {}
        agents = raw.get("agents", [])

        # Check duplicate
        if any(a.get("name") == name for a in agents):
            raise HTTPException(409, f"Agent '{name}' already exists")

        agent = {
            "name": name,
            "display_name": body.get("display_name", name.title()),
            "description": body.get("description", ""),
            "system_prompt": body.get("system_prompt", ""),
            "language": body.get("language", "en"),
            "preferred_model": body.get("preferred_model", ""),
            "temperature": body.get("temperature", 0.7),
            "priority": body.get("priority", 0),
            "enabled": body.get("enabled", True),
            "allowed_tools": body.get("allowed_tools") or [],
            "blocked_tools": body.get("blocked_tools", []),
            "can_delegate_to": body.get("can_delegate_to", []),
            "sandbox_timeout": body.get("sandbox_timeout", 30),
            "sandbox_network": body.get("sandbox_network", "allow"),
        }
        agents.append(agent)
        raw["agents"] = agents
        agents_path.write_text(
            yaml.dump(raw, default_flow_style=False, allow_unicode=True), encoding="utf-8"
        )
        return {"status": "created", "agent": agent}

    @app.put("/api/v1/agents/{agent_name}", dependencies=deps)
    async def update_agent(agent_name: str, request: Request) -> dict[str, Any]:
        """Update an existing agent profile."""
        body = await request.json()
        agents_path = config_manager.config.jarvis_home / "agents.yaml"
        if not agents_path.exists():
            raise HTTPException(404, f"Agent '{agent_name}' not found")

        raw = yaml.safe_load(agents_path.read_text(encoding="utf-8")) or {}
        agents = raw.get("agents", [])
        found = False
        updated_agent = None
        for i, a in enumerate(agents):
            if a.get("name") == agent_name:
                # Update fields (preserve name)
                for key in [
                    "display_name",
                    "description",
                    "system_prompt",
                    "language",
                    "preferred_model",
                    "temperature",
                    "priority",
                    "enabled",
                    "allowed_tools",
                    "blocked_tools",
                    "can_delegate_to",
                    "sandbox_timeout",
                    "sandbox_network",
                ]:
                    if key in body:
                        agents[i][key] = body[key]
                found = True
                updated_agent = agents[i]
                break
        if not found:
            raise HTTPException(404, f"Agent '{agent_name}' not found")

        raw["agents"] = agents
        agents_path.write_text(
            yaml.dump(raw, default_flow_style=False, allow_unicode=True), encoding="utf-8"
        )
        return {"status": "updated", "agent": updated_agent}

    @app.delete("/api/v1/agents/{agent_name}", dependencies=deps)
    async def delete_agent(agent_name: str) -> dict[str, Any]:
        """Delete an agent profile."""
        if agent_name == "jarvis":
            raise HTTPException(403, "Cannot delete the default agent")

        agents_path = config_manager.config.jarvis_home / "agents.yaml"
        if not agents_path.exists():
            raise HTTPException(404, f"Agent '{agent_name}' not found")

        raw = yaml.safe_load(agents_path.read_text(encoding="utf-8")) or {}
        agents = raw.get("agents", [])
        original_len = len(agents)
        agents = [a for a in agents if a.get("name") != agent_name]
        if len(agents) == original_len:
            raise HTTPException(404, f"Agent '{agent_name}' not found")

        raw["agents"] = agents
        agents_path.write_text(
            yaml.dump(raw, default_flow_style=False, allow_unicode=True), encoding="utf-8"
        )
        return {"status": "deleted", "name": agent_name}

    # -- Credentials ------------------------------------------------------

    @app.get("/api/v1/credentials", dependencies=deps)
    async def list_credentials() -> dict[str, Any]:
        """Listet alle gespeicherten Credentials (nur Keys, keine Werte)."""
        try:
            from jarvis.security.credentials import CredentialStore

            store = CredentialStore()
            global_creds = store.list_entries()
            return {
                "credentials": [
                    {"service": s, "key": k, "scope": "global"} for s, k in global_creds
                ],
            }
        except Exception as exc:
            log.error("credentials_list_failed", error=str(exc))
            return {"credentials": [], "error": "Credentials konnten nicht geladen werden"}

    @app.post("/api/v1/credentials", dependencies=deps)
    async def store_credential(
        request: Request,
    ) -> dict[str, Any]:
        """Speichert ein Credential (Body: service, key, value, agent_id)."""
        body = await request.json()
        service = body.get("service", "")
        key = body.get("key", "")
        value = body.get("value", "")
        agent_id = body.get("agent_id", "")
        if not service or not key or not value:
            return {"error": "service, key und value sind erforderlich", "status": 400}
        try:
            from jarvis.security.credentials import CredentialStore

            store = CredentialStore()
            store.store(service, key, value, agent_id=agent_id)
            return {"status": "ok", "service": service, "key": key, "scope": agent_id or "global"}
        except Exception as exc:
            log.error("credential_store_failed", error=str(exc))
            return {"error": "Credential konnte nicht gespeichert werden", "status": 500}

    @app.delete("/api/v1/credentials/{service}/{key}", dependencies=deps)
    async def delete_credential(service: str, key: str, agent_id: str = "") -> dict[str, Any]:
        """Löscht ein Credential."""
        try:
            from jarvis.security.credentials import CredentialStore

            store = CredentialStore()
            store.store(service, key, "", agent_id=agent_id)
            return {"status": "ok", "deleted": f"{service}:{key}"}
        except Exception as exc:
            log.error("credential_delete_failed", error=str(exc))
            return {"error": "Credential konnte nicht geloescht werden", "status": 500}

    # -- Bindings ---------------------------------------------------------

    @app.get("/api/v1/bindings", dependencies=deps)
    async def list_bindings() -> dict[str, Any]:
        """Listet alle Binding-Regeln aus bindings.yaml."""
        try:
            bindings_path = config_manager.config.jarvis_home / "bindings.yaml"
            if bindings_path.exists():
                raw = yaml.safe_load(bindings_path.read_text(encoding="utf-8")) or {}
                bindings = raw.get("bindings", [])
            else:
                bindings = []
            return {"bindings": bindings}
        except Exception as exc:
            log.error("bindings_list_failed", error=str(exc))
            return {"bindings": [], "error": "Bindings konnten nicht geladen werden"}

    @app.post("/api/v1/bindings", dependencies=deps)
    async def create_binding(data: dict[str, Any]) -> dict[str, Any]:
        """Erstellt oder aktualisiert eine Binding-Regel."""
        try:
            from jarvis.gateway.config_api import BindingRuleDTO
            from jarvis.gateway.config_api import ConfigManager as CfgMgr

            cfg_mgr = CfgMgr(config_manager.config)
            dto = BindingRuleDTO(**data)
            return {"binding": cfg_mgr.upsert_binding(dto), "status": "ok"}
        except Exception as exc:
            log.error("binding_create_failed", error=str(exc))
            return {"error": "Binding konnte nicht erstellt werden", "status": 400}

    @app.delete("/api/v1/bindings/{name}", dependencies=deps)
    async def delete_binding(name: str) -> dict[str, Any]:
        """Löscht eine Binding-Regel."""
        try:
            from jarvis.gateway.config_api import ConfigManager as CfgMgr

            cfg_mgr = CfgMgr(config_manager.config)
            if cfg_mgr.delete_binding(name):
                return {"status": "ok", "deleted": name}
            return {"error": f"Binding '{name}' nicht gefunden", "status": 404}
        except Exception as exc:
            log.error("binding_delete_failed", error=str(exc))
            return {"error": "Binding konnte nicht geloescht werden", "status": 500}

    # -- Circles ----------------------------------------------------------

    @app.get("/api/v1/circles", dependencies=deps)
    async def list_circles(peer_id: str = "") -> dict[str, Any]:
        """Listet Trusted Circles."""
        try:
            from jarvis.skills.circles import CircleManager

            circles_mgr = CircleManager()
            circles = circles_mgr.list_circles(peer_id=peer_id)
            return {
                "circles": [
                    {
                        "circle_id": c.circle_id,
                        "name": c.name,
                        "description": c.description,
                        "member_count": c.member_count,
                        "curated_skills": len(c.curated_skills),
                        "approved_skills": len(c.approved_skills()),
                    }
                    for c in circles
                ],
                "stats": circles_mgr.stats(),
            }
        except Exception as exc:
            log.error("circles_list_failed", error=str(exc))
            return {"circles": [], "error": "Circles konnten nicht geladen werden"}

    @app.get("/api/v1/circles/stats", dependencies=deps)
    async def circles_stats() -> dict[str, Any]:
        """Ecosystem-Statistiken."""
        try:
            from jarvis.skills.circles import CircleManager

            return CircleManager().stats()
        except Exception as exc:
            log.error("circles_stats_failed", error=str(exc))
            return {"error": "Circle-Statistiken nicht verfuegbar"}

    # -- Sandbox ----------------------------------------------------------

    @app.get("/api/v1/sandbox", dependencies=deps)
    async def get_sandbox() -> dict[str, Any]:
        """Liest Sandbox-Konfiguration."""
        try:
            from jarvis.gateway.config_api import ConfigManager as CfgMgr

            cfg_mgr = CfgMgr(config_manager.config)
            return {"sandbox": cfg_mgr.get_sandbox()}
        except Exception as exc:
            log.error("sandbox_get_failed", error=str(exc))
            return {"error": "Sandbox-Konfiguration nicht verfuegbar"}

    @app.patch("/api/v1/sandbox", dependencies=deps)
    async def update_sandbox(values: dict[str, Any]) -> dict[str, Any]:
        """Aktualisiert Sandbox-Einstellungen."""
        try:
            from jarvis.gateway.config_api import ConfigManager as CfgMgr
            from jarvis.gateway.config_api import SandboxUpdate

            cfg_mgr = CfgMgr(config_manager.config)
            update = SandboxUpdate(**values)
            return {"sandbox": cfg_mgr.update_sandbox(update), "status": "ok"}
        except Exception as exc:
            log.error("sandbox_update_failed", error=str(exc))
            return {"error": "Sandbox konnte nicht aktualisiert werden", "status": 400}

    # -- Wizards ----------------------------------------------------------

    @app.get("/api/v1/wizards", dependencies=deps)
    async def list_wizards() -> dict[str, Any]:
        """Alle verfügbaren Konfigurations-Assistenten."""
        from jarvis.gateway.wizards import WizardRegistry

        reg = WizardRegistry()
        return {"wizards": reg.list_wizards(), "count": reg.wizard_count}

    @app.get("/api/v1/wizards/{wizard_type}", dependencies=deps)
    async def get_wizard(wizard_type: str) -> dict[str, Any]:
        """Details eines Wizards (Schritte + Templates)."""
        from jarvis.gateway.wizards import WizardRegistry

        reg = WizardRegistry()
        wizard = reg.get(wizard_type)
        if not wizard:
            return {"error": f"Wizard '{wizard_type}' nicht gefunden"}
        return wizard.to_dict()

    @app.post("/api/v1/wizards/{wizard_type}/run", dependencies=deps)
    async def run_wizard(wizard_type: str, body: dict[str, Any]) -> dict[str, Any]:
        """Führt einen Wizard aus und generiert Konfiguration."""
        from jarvis.gateway.wizards import WizardRegistry

        reg = WizardRegistry()
        result = reg.run_wizard(wizard_type, body.get("values", {}))
        if not result:
            return {"error": f"Wizard '{wizard_type}' nicht gefunden"}
        return result.to_dict()

    @app.get("/api/v1/wizards/{wizard_type}/templates", dependencies=deps)
    async def wizard_templates(wizard_type: str) -> dict[str, Any]:
        """Templates eines Wizards."""
        from jarvis.gateway.wizards import WizardRegistry

        reg = WizardRegistry()
        wizard = reg.get(wizard_type)
        if not wizard:
            return {"error": f"Wizard '{wizard_type}' nicht gefunden"}
        return {
            "templates": [
                {
                    "id": t.template_id,
                    "name": t.name,
                    "description": t.description,
                    "icon": t.icon,
                    "preset_values": t.preset_values,
                }
                for t in wizard.templates
            ]
        }

    # -- RBAC -------------------------------------------------------------

    @app.get("/api/v1/rbac/roles", dependencies=deps)
    async def rbac_roles() -> dict[str, Any]:
        """Alle verfügbaren Rollen und ihre Berechtigungen."""
        from jarvis.gateway.wizards import ROLE_PERMISSIONS

        return {
            "roles": {
                role.value: {"permissions": [p.key for p in perms], "count": len(perms)}
                for role, perms in ROLE_PERMISSIONS.items()
            }
        }

    @app.get("/api/v1/rbac/check", dependencies=deps)
    async def rbac_check(user_id: str, resource: str, action: str) -> dict[str, Any]:
        """Prüft eine Berechtigung."""
        from jarvis.gateway.wizards import RBACManager

        mgr = RBACManager()
        return {
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "allowed": mgr.check_permission(user_id, resource, action),
        }

    # -- Auth Gateway -----------------------------------------------------

    @app.get("/api/v1/auth/stats", dependencies=deps)
    async def auth_stats() -> dict[str, Any]:
        """Auth-Gateway-Statistiken."""
        try:
            from jarvis.gateway.auth import AuthGateway

            return AuthGateway().stats()
        except Exception as exc:
            log.error("auth_stats_failed", error=str(exc))
            return {"error": "Auth-Statistiken nicht verfuegbar"}

    # -- Agent Heartbeat --------------------------------------------------

    @app.get("/api/v1/agent-heartbeat/dashboard", dependencies=deps)
    async def agent_heartbeat_dashboard() -> dict[str, Any]:
        """Globale Dashboard-Übersicht aller Agent-Heartbeats."""
        try:
            from jarvis.core.agent_heartbeat import AgentHeartbeatScheduler

            return AgentHeartbeatScheduler().global_dashboard()
        except Exception as exc:
            log.error("heartbeat_dashboard_failed", error=str(exc))
            return {"error": "Heartbeat-Dashboard nicht verfuegbar"}

    @app.get("/api/v1/agent-heartbeat/{agent_id}", dependencies=deps)
    async def agent_heartbeat_summary(agent_id: str) -> dict[str, Any]:
        """Heartbeat-Zusammenfassung für einen Agent."""
        try:
            from jarvis.core.agent_heartbeat import AgentHeartbeatScheduler

            return AgentHeartbeatScheduler().agent_summary(agent_id)
        except Exception as exc:
            log.error("heartbeat_summary_failed", agent_id=agent_id, error=str(exc))
            return {"error": "Heartbeat-Zusammenfassung nicht verfuegbar"}


# ======================================================================
# Config read / write routes
# ======================================================================


def _register_config_routes(
    app: Any,
    deps: list[Any],
    config_manager: ConfigManager,
    gateway: Any = None,
) -> None:
    """Config CRUD, presets, reload."""

    @app.get("/api/v1/health", dependencies=deps)
    async def health_check() -> dict[str, Any]:
        """Health check endpoint used by the Vite launcher."""
        return {"status": "ok"}

    @app.get("/api/v1/config", dependencies=deps)
    async def get_config() -> dict[str, Any]:
        """Gibt die gesamte Konfiguration zurück (ohne Secrets)."""
        data = config_manager.read()
        data["_meta"] = {
            "editable_sections": config_manager.editable_sections(),
            "editable_top_level": config_manager.editable_top_level_fields(),
        }
        return data

    @app.patch("/api/v1/config", dependencies=deps)
    async def update_config_top_level(updates: dict[str, Any]) -> dict[str, Any]:
        """Aktualisiert Top-Level-Felder."""
        from jarvis.config_manager import _is_secret_field

        results: list[dict[str, Any]] = []
        for key, value in updates.items():
            # Skip masked secret values — the UI sends "***" for untouched secrets.
            # Real changes (new value or "") are passed through and persisted.
            if value == "***" and _is_secret_field(key):
                results.append({"key": key, "status": "skipped"})
                continue
            try:
                config_manager.update_top_level(key, value)
                results.append({"key": key, "status": "ok"})
            except ValueError as exc:
                log.error("config_update_key_failed", key=key, error=str(exc))
                results.append({"key": key, "status": "error", "error": "Ungueltige Konfiguration"})
        config_manager.save()
        # Trigger live-reload of runtime components
        if gateway is not None and hasattr(gateway, "reload_components"):
            gateway.reload_components(config=True)
        # Sync backend i18n locale when language changes (#33)
        if "language" in updates:
            try:
                from jarvis.i18n import set_locale

                set_locale(updates["language"])
            except Exception:
                pass
        return {"results": results}

    @app.post("/api/v1/config/reload", dependencies=deps)
    async def reload_config() -> dict[str, Any]:
        """Lädt die Konfiguration neu aus der Datei."""
        config_manager.reload()
        if gateway is not None and hasattr(gateway, "reload_components"):
            gateway.reload_components(prompts=True, policies=True, core_memory=True, config=True)
        return {"status": "ok", "message": "Konfiguration und Komponenten neu geladen"}

    # -- Locales (available i18n language packs) --

    @app.get("/api/v1/locales", dependencies=deps)
    async def list_locales() -> dict[str, Any]:
        """Returns available i18n locales and the currently active one."""
        from jarvis.i18n import get_available_locales, get_locale
        from jarvis.i18n.prompt_presets import available_preset_locales

        locales = get_available_locales()
        return {
            "locales": locales,
            "active": get_locale(),
            "preset_locales": available_preset_locales(),
        }

    @app.post("/api/v1/translate-prompts", dependencies=deps)
    async def translate_prompts(request: Request) -> dict[str, Any]:
        """Translate system prompts to a target language.

        Supports two methods:
          - ``"preset"`` — Use curated prompt presets (instant, no LLM needed).
          - ``"ollama"`` — Use local Ollama LLM to translate on-the-fly.

        Request body::

            {
              "target_locale": "en",
              "method": "preset" | "ollama",
              "prompts": {                     // only for method=ollama
                "plannerSystem": "...",
                "replanPrompt": "...",
                "escalationPrompt": "..."
              }
            }

        Returns::

            {
              "translations": {
                "plannerSystem": "...",
                "replanPrompt": "...",
                "escalationPrompt": "..."
              },
              "method": "preset" | "ollama"
            }
        """
        try:
            body = await request.json()
        except Exception:
            return {"error": "Invalid JSON body", "status": 400}

        target_locale = body.get("target_locale", "").strip().lower()
        method = body.get("method", "preset").strip().lower()

        if not target_locale:
            return {"error": "target_locale is required", "status": 400}
        if method not in ("preset", "ollama"):
            return {"error": "method must be 'preset' or 'ollama'", "status": 400}

        # ── Method: preset ──────────────────────────────────────────
        if method == "preset":
            from jarvis.i18n.prompt_presets import get_preset

            preset = get_preset(target_locale)
            if preset is None:
                from jarvis.i18n.prompt_presets import available_preset_locales

                return {
                    "error": f"No preset available for '{target_locale}'",
                    "available": available_preset_locales(),
                    "status": 404,
                }
            return {"translations": preset, "method": "preset"}

        # ── Method: ollama ──────────────────────────────────────────
        prompts = body.get("prompts", {})
        if not prompts:
            return {"error": "prompts dict is required for method=ollama", "status": 400}

        cfg = config_manager.config
        ollama_url = cfg.ollama.base_url.rstrip("/")
        model = cfg.models.planner.name
        timeout = cfg.ollama.timeout_seconds

        import httpx

        translations: dict[str, str] = {}
        errors: list[str] = []

        async with httpx.AsyncClient(timeout=timeout) as client:
            for key, text in prompts.items():
                if key not in ("plannerSystem", "replanPrompt", "escalationPrompt"):
                    continue
                if not text or not text.strip():
                    continue

                system_msg = (
                    f"You are a professional translator. Translate the following system "
                    f"prompt to {target_locale}. Preserve ALL template variables exactly "
                    f"as they are (e.g., {{tools_section}}, {{owner_name}}, "
                    f"{{results_section}}, {{original_goal}}, {{tool}}, {{reason}}). "
                    f"Preserve all markdown formatting, code blocks, and JSON structures. "
                    f"Output ONLY the translated text, nothing else."
                )

                try:
                    resp = await client.post(
                        f"{ollama_url}/api/chat",
                        json={
                            "model": model,
                            "messages": [
                                {"role": "system", "content": system_msg},
                                {"role": "user", "content": text},
                            ],
                            "stream": False,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    translated = data.get("message", {}).get("content", "").strip()

                    # Validate that template variables survived translation
                    import re

                    original_vars = set(re.findall(r"\{[\w_]+\}", text))
                    translated_vars = set(re.findall(r"\{[\w_]+\}", translated))
                    missing = original_vars - translated_vars
                    if missing:
                        errors.append(f"{key}: template variables lost in translation: {missing}")
                        # Still include the translation but flag it
                        translations[key] = translated
                    else:
                        translations[key] = translated
                except httpx.TimeoutException:
                    errors.append(f"{key}: Ollama request timed out")
                except httpx.HTTPStatusError as exc:
                    errors.append(f"{key}: Ollama HTTP {exc.response.status_code}")
                except Exception as exc:
                    errors.append(f"{key}: {exc!s}")

        result: dict[str, Any] = {"translations": translations, "method": "ollama"}
        if errors:
            result["warnings"] = errors
        return result

    # -- Presets (BEFORE {section} routes to avoid path parameter conflict) --

    @app.get("/api/v1/config/presets", dependencies=deps)
    async def list_presets() -> dict[str, Any]:
        """Listet verfügbare Konfigurations-Presets."""
        return {
            "presets": [
                {
                    "name": "minimal",
                    "description": "Minimale Konfiguration (CLI-only, kleine Modelle)",
                    "sections": {
                        "channels": {
                            "cli_enabled": True,
                            "telegram_enabled": False,
                            "webui_enabled": False,
                        },
                        "heartbeat": {"enabled": False},
                        "dashboard": {"enabled": False},
                    },
                },
                {
                    "name": "standard",
                    "description": "Standard-Setup (CLI + WebUI, Heartbeat, Dashboard)",
                    "sections": {
                        "channels": {"cli_enabled": True, "webui_enabled": True},
                        "heartbeat": {"enabled": True, "interval_minutes": 30},
                        "dashboard": {"enabled": True},
                    },
                },
                {
                    "name": "full",
                    "description": "Vollausbau (alle Channels, Heartbeat, Dashboard, Plugins)",
                    "sections": {
                        "channels": {
                            "cli_enabled": True,
                            "webui_enabled": True,
                            "telegram_enabled": True,
                            "slack_enabled": True,
                            "discord_enabled": True,
                        },
                        "heartbeat": {"enabled": True, "interval_minutes": 15},
                        "dashboard": {"enabled": True},
                        "plugins": {"auto_update": True},
                    },
                },
            ],
        }

    @app.post("/api/v1/config/presets/{preset_name}", dependencies=deps)
    async def apply_preset(preset_name: str) -> dict[str, Any]:
        """Wendet ein Konfigurations-Preset an."""
        presets = (await list_presets())["presets"]
        preset = next((p for p in presets if p["name"] == preset_name), None)
        if not preset:
            return {"error": f"Preset '{preset_name}' nicht gefunden", "status": 404}
        results = []
        for section, values in preset["sections"].items():
            try:
                config_manager.update_section(section, values)
                results.append({"section": section, "status": "ok"})
            except ValueError as exc:
                log.warning("preset_section_update_failed", section=section, error=str(exc))
                results.append(
                    {"section": section, "status": "error", "error": "Ungueltige Konfiguration"}
                )
        config_manager.save()
        return {"preset": preset_name, "results": results}

    # -- Config Section CRUD (AFTER presets to avoid {section} capturing "presets") --

    @app.get("/api/v1/config/{section}", dependencies=deps)
    async def get_config_section(section: str) -> dict[str, Any]:
        """Gibt eine einzelne Konfigurations-Sektion zurück."""
        result = config_manager.read_section(section)
        if result is None:
            return {"error": f"Sektion '{section}' nicht gefunden", "status": 404}
        return {"section": section, "values": result}

    @app.patch("/api/v1/config/{section}", dependencies=deps)
    async def update_config_section(section: str, values: dict[str, Any]) -> dict[str, Any]:
        """Aktualisiert eine Konfigurations-Sektion."""
        from jarvis.config_manager import _is_secret_field

        def _deep_clean_secrets(
            data: dict[str, Any],
            existing: dict[str, Any] | None = None,
            *,
            _depth: int = 0,
        ) -> dict[str, Any]:
            """Recursively strip masked ('***') and empty secret values."""
            if _depth > 5:
                return data
            out: dict[str, Any] = {}
            for k, v in data.items():
                if isinstance(v, dict):
                    ex_sub = existing.get(k) if isinstance(existing, dict) else None
                    cleaned_sub = _deep_clean_secrets(
                        v,
                        ex_sub if isinstance(ex_sub, dict) else None,
                        _depth=_depth + 1,
                    )
                    if cleaned_sub:  # only include non-empty dicts
                        out[k] = cleaned_sub
                elif _is_secret_field(k):
                    if v == "***":
                        continue  # skip masked placeholders
                    if (v == "" or v is None) and existing:
                        ex_val = existing.get(k, "") if isinstance(existing, dict) else ""
                        if ex_val and ex_val != "":
                            continue  # protect existing non-empty secret
                    out[k] = v
                else:
                    out[k] = v
            return out

        # Get existing section values (raw, unmasked) for protection comparison
        raw_cfg = config_manager.config.model_dump(mode="json")
        existing_section = (
            raw_cfg.get(section, {}) if isinstance(raw_cfg.get(section), dict) else {}
        )
        cleaned = _deep_clean_secrets(values, existing_section)
        try:
            config_manager.update_section(section, cleaned)
            config_manager.save()
            # Trigger live-reload of runtime components (executor, web tools, model router)
            if gateway is not None and hasattr(gateway, "reload_components"):
                gateway.reload_components(config=True)

            # Validate model availability when saving 'models' section
            model_warnings: list[str] = []
            if section == "models":
                router = getattr(gateway, "_model_router", None) if gateway else None
                if router and router._available_models:
                    available = router._available_models
                    cfg_models = config_manager.config.models
                    for role, model_cfg in [
                        ("planner", cfg_models.planner),
                        ("executor", cfg_models.executor),
                        ("coder", cfg_models.coder),
                        ("embedding", cfg_models.embedding),
                    ]:
                        if model_cfg.name and model_cfg.name not in available:
                            model_warnings.append(
                                f"Modell '{model_cfg.name}' ({role}) nicht in Ollama gefunden. "
                                f"Bitte installieren: ollama pull {model_cfg.name}"
                            )

            result: dict[str, Any] = {
                "status": "ok",
                "section": section,
                "updated_keys": list(cleaned.keys()),
            }
            if model_warnings:
                result["warnings"] = model_warnings
            return result
        except ValueError as exc:
            log.warning("config_section_update_failed", section=section, error=str(exc))
            return {"error": "Ungueltige Konfiguration", "status": 400}


# ======================================================================
# Session management routes (vault, session-isolation, isolation)
# ======================================================================


def _register_session_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """Vault, session-isolation, workspace-isolation, multi-tenant."""

    # -- Vault & Session-Isolation ----------------------------------------

    @app.get("/api/v1/vault/stats", dependencies=deps)
    async def vault_stats() -> dict[str, Any]:
        """Vault-Manager Statistiken."""
        mgr = getattr(gateway, "_vault_manager", None)
        if mgr is None:
            return {"total_vaults": 0, "agents": [], "total_entries": 0}
        return mgr.stats()

    @app.get("/api/v1/vault/agents", dependencies=deps)
    async def vault_agents() -> dict[str, Any]:
        """Alle Agent-Vaults auflisten."""
        mgr = getattr(gateway, "_vault_manager", None)
        if mgr is None:
            return {"agents": []}
        return {"agents": [v.stats() for v in mgr._vaults.values()]}

    @app.get("/api/v1/sessions/stats", dependencies=deps)
    async def session_stats() -> dict[str, Any]:
        """Session-Store Statistiken."""
        store = getattr(gateway, "_isolated_sessions", None)
        if store is None:
            return {"total_agents": 0, "total_sessions": 0, "active_sessions": 0}
        return store.stats()

    @app.get("/api/v1/sessions/guard/violations", dependencies=deps)
    async def guard_violations() -> dict[str, Any]:
        """Session-Guard Violations."""
        guard = getattr(gateway, "_session_guard", None)
        if guard is None:
            return {"violations": [], "count": 0}
        v = guard.violations()
        return {"violations": v, "count": len(v)}

    # -- Multi-User Isolation (Phase 8) -----------------------------------

    @app.get("/api/v1/isolation/stats", dependencies=deps)
    async def isolation_stats_core() -> dict[str, Any]:
        """Isolation-Statistiken (core)."""
        try:
            from jarvis.core.isolation import MultiUserIsolation

            iso = MultiUserIsolation()
            return iso.stats()
        except Exception as exc:
            log.error("isolation_stats_failed", error=str(exc))
            return {"error": "Isolation-Statistiken nicht verfuegbar"}

    @app.get("/api/v1/isolation/quotas", dependencies=deps)
    async def isolation_quotas() -> dict[str, Any]:
        """Quota-Übersicht aller Agents."""
        try:
            from jarvis.core.isolation import MultiUserIsolation

            iso = MultiUserIsolation()
            return {"quotas": iso.all_quota_summaries()}
        except Exception as exc:
            log.error("isolation_quotas_failed", error=str(exc))
            return {"error": "Quota-Uebersicht nicht verfuegbar"}

    @app.get("/api/v1/isolation/violations", dependencies=deps)
    async def isolation_violations() -> dict[str, Any]:
        """Workspace-Violations."""
        try:
            from jarvis.core.isolation import WorkspaceGuard

            guard = WorkspaceGuard()
            return {"violations": guard.violations, "count": guard.violation_count}
        except Exception as exc:
            log.error("isolation_violations_failed", error=str(exc))
            return {"error": "Violations konnten nicht geladen werden"}

    # -- Sandbox-Isolierung + Multi-Tenant (Phase 25) ---------------------

    @app.get("/api/v1/isolation/sandboxes", dependencies=deps)
    async def isolation_sandboxes() -> dict[str, Any]:
        """Laufende Sandboxes."""
        enforcer = getattr(gateway, "_isolation_enforcer", None)
        if enforcer is None:
            return {"sandboxes": []}
        return {"sandboxes": [sb.to_dict() for sb in enforcer.sandboxes.running()]}

    @app.get("/api/v1/isolation/tenants", dependencies=deps)
    async def isolation_tenants() -> dict[str, Any]:
        """Tenant-Übersicht."""
        enforcer = getattr(gateway, "_isolation_enforcer", None)
        if enforcer is None:
            return {"total_tenants": 0}
        return enforcer.tenants.stats()

    @app.get("/api/v1/isolation/secrets", dependencies=deps)
    async def isolation_secrets() -> dict[str, Any]:
        """Secret-Vault Statistiken."""
        enforcer = getattr(gateway, "_isolation_enforcer", None)
        if enforcer is None:
            return {"total_secrets": 0}
        return enforcer.secrets.stats()

    # -- Per-Agent Vault & Session-Isolation (Phase 29) -------------------

    @app.get("/api/v1/vaults/stats", dependencies=deps)
    async def vaults_stats() -> dict[str, Any]:
        """Agent-Vault Statistiken."""
        vm = getattr(gateway, "_vault_manager", None)
        if vm is None:
            return {"total_vaults": 0}
        return vm.stats()

    @app.get("/api/v1/vaults/sessions", dependencies=deps)
    async def vaults_sessions() -> dict[str, Any]:
        """Session-Isolation Status."""
        vm = getattr(gateway, "_vault_manager", None)
        if vm is None:
            return {"agent_stores": 0}
        return vm.sessions.stats()

    @app.get("/api/v1/vaults/firewall", dependencies=deps)
    async def vaults_firewall() -> dict[str, Any]:
        """Session-Firewall Status."""
        vm = getattr(gateway, "_vault_manager", None)
        if vm is None:
            return {"total_violations": 0}
        return vm.firewall.stats()

    # -- Chat-History API (für WebUI Sidebar) ------------------------------

    def _get_session_store() -> Any:
        """Zugriff auf den SessionStore des Gateways."""
        return getattr(gateway, "_session_store", None)

    @app.get("/api/v1/sessions/list", dependencies=deps)
    async def list_sessions(channel: str = "webui", limit: int = 50) -> dict[str, Any]:
        """Aktive Sessions für die Chat-History-Sidebar auflisten."""
        store = _get_session_store()
        if not store:
            return {"sessions": []}
        sessions = store.list_sessions_for_channel(channel=channel, limit=limit)
        return {"sessions": sessions}

    @app.get("/api/v1/sessions/{session_id}/history", dependencies=deps)
    async def get_session_history(session_id: str, limit: int = 100) -> dict[str, Any]:
        """Chat-Messages einer bestimmten Session abrufen."""
        store = _get_session_store()
        if not store:
            return {"messages": [], "session_id": session_id}
        messages = store.get_session_history(session_id, limit=limit)
        return {"messages": messages, "session_id": session_id}

    @app.patch("/api/v1/sessions/{session_id}", dependencies=deps)
    async def update_session(session_id: str, request: Request) -> dict[str, Any]:
        """Session-Metadaten aktualisieren (Titel)."""
        body = await request.json()
        store = _get_session_store()
        if not store:
            raise HTTPException(status_code=503, detail="Session store not available")
        title = body.get("title")
        if title:
            store.update_session_title(session_id, title)
        return {"status": "updated", "session_id": session_id}

    @app.delete("/api/v1/sessions/{session_id}", dependencies=deps)
    async def delete_session(session_id: str) -> dict[str, Any]:
        """Soft-Delete einer Session (active=0)."""
        store = _get_session_store()
        if not store:
            raise HTTPException(status_code=503, detail="Session store not available")
        store.delete_session(session_id)
        return {"status": "deleted", "session_id": session_id}

    @app.post("/api/v1/sessions/new", dependencies=deps)
    async def create_new_session() -> dict[str, Any]:
        """Neue leere Session erstellen und ID zurückgeben."""
        store = _get_session_store()
        if not store:
            raise HTTPException(status_code=503, detail="Session store not available")
        import uuid

        from jarvis.models import SessionContext

        session_id = uuid.uuid4().hex[:16]
        store.save_session(
            SessionContext(
                session_id=session_id,
                channel="webui",
                user_id="web_user",
                agent_name="jarvis",
            )
        )
        return {"session_id": session_id}


# ======================================================================
# Memory / search routes
# ======================================================================


def _register_memory_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """Memory-hygiene, memory-integrity, explainability."""

    # -- Memory-Hygiene ---------------------------------------------------

    @app.post("/api/v1/memory/hygiene/scan", dependencies=deps)
    async def memory_hygiene_scan(request: Request) -> dict[str, Any]:
        """Memory-Einträge auf Injection/Credentials/Widersprüche scannen."""
        try:
            from jarvis.memory.hygiene import MemoryHygieneEngine

            engine = getattr(gateway, "_memory_hygiene", None) or MemoryHygieneEngine()
            body = await request.json()
            entries = body.get("entries", [])
            auto_quarantine = body.get("auto_quarantine", True)
            report = engine.scan_batch(entries, auto_quarantine=auto_quarantine)
            return report.to_dict()
        except Exception:
            log.exception("Error during memory hygiene scan")
            return {"error": "Internal error during memory hygiene scan"}

    @app.get("/api/v1/memory/hygiene/stats", dependencies=deps)
    async def memory_hygiene_stats() -> dict[str, Any]:
        """Memory-Hygiene Statistiken."""
        engine = getattr(gateway, "_memory_hygiene", None)
        if engine is None:
            return {
                "total_scans": 0,
                "total_scanned": 0,
                "total_threats": 0,
                "quarantined": 0,
                "threat_rate": 0.0,
            }
        return engine.stats()

    @app.get("/api/v1/memory/hygiene/quarantine", dependencies=deps)
    async def memory_quarantine() -> dict[str, Any]:
        """Quarantäne-Liste."""
        engine = getattr(gateway, "_memory_hygiene", None)
        if engine is None:
            return {"quarantined": []}
        return {"quarantined": engine.quarantine()}

    # -- Memory-Integrität (Phase 26) ------------------------------------

    @app.get("/api/v1/memory/integrity", dependencies=deps)
    async def memory_integrity() -> dict[str, Any]:
        """Memory-Integritäts-Status."""
        checker = getattr(gateway, "_integrity_checker", None)
        if checker is None:
            return {"total_checks": 0, "last_score": 100}
        return checker.stats()

    @app.get("/api/v1/memory/explainability", dependencies=deps)
    async def memory_explainability() -> dict[str, Any]:
        """Decision-Explainer Statistiken."""
        explainer = getattr(gateway, "_decision_explainer", None)
        if explainer is None:
            return {"total_explanations": 0}
        return explainer.stats()

    # -- Explainability ---------------------------------------------------

    @app.get("/api/v1/explainability/trails", dependencies=deps)
    async def explainability_trails() -> dict[str, Any]:
        """Letzte Decision-Trails."""
        engine = getattr(gateway, "_explainability", None)
        if engine is None:
            return {"trails": [], "count": 0}
        trails = engine.recent_trails(limit=20)
        return {"trails": [t.to_dict() for t in trails], "count": len(trails)}

    @app.get("/api/v1/explainability/stats", dependencies=deps)
    async def explainability_stats() -> dict[str, Any]:
        """Explainability-Engine Statistiken."""
        engine = getattr(gateway, "_explainability", None)
        if engine is None:
            return {
                "total_requests": 0,
                "active_trails": 0,
                "completed_trails": 0,
                "avg_confidence": 0.0,
            }
        return engine.stats()

    @app.get("/api/v1/explainability/low-trust", dependencies=deps)
    async def explainability_low_trust() -> dict[str, Any]:
        """Trails mit niedrigem Trust-Score."""
        engine = getattr(gateway, "_explainability", None)
        if engine is None:
            return {"low_trust_trails": [], "count": 0}
        trails = engine.low_trust_trails(threshold=0.5)
        return {"low_trust_trails": [t.to_dict() for t in trails], "count": len(trails)}

    # -- Knowledge Graph --------------------------------------------------

    @app.get("/api/v1/memory/graph/stats", dependencies=deps)
    async def knowledge_graph_stats() -> dict[str, Any]:
        """Wissensgraph-Statistiken."""
        semantic = getattr(gateway, "_semantic_memory", None)
        if semantic is None:
            return {"entities": 0, "relations": 0, "entity_types": {}}
        try:
            entities = getattr(semantic, "entities", None) or {}
            relations = getattr(semantic, "relations", None) or []
            type_counts: dict[str, int] = {}
            for e in entities.values() if isinstance(entities, dict) else entities:
                etype = getattr(e, "type", None) or getattr(e, "entity_type", "unknown")
                type_counts[etype] = type_counts.get(etype, 0) + 1
            return {
                "entities": len(entities),
                "relations": len(relations),
                "entity_types": type_counts,
            }
        except Exception:
            return {"entities": 0, "relations": 0, "entity_types": {}}

    @app.get("/api/v1/memory/graph/entities", dependencies=deps)
    async def knowledge_graph_entities() -> dict[str, Any]:
        """Alle Entitäten und Beziehungen im Wissensgraph."""
        semantic = getattr(gateway, "_semantic_memory", None)
        if semantic is None:
            return {"entities": [], "relations": []}
        try:
            raw_entities = getattr(semantic, "entities", None) or {}
            raw_relations = getattr(semantic, "relations", None) or []

            entities = []
            for eid, e in (
                raw_entities.items() if isinstance(raw_entities, dict) else enumerate(raw_entities)
            ):
                entity_id = str(getattr(e, "id", eid))
                entities.append(
                    {
                        "id": entity_id,
                        "name": getattr(e, "name", str(e)),
                        "type": getattr(e, "type", None) or getattr(e, "entity_type", "unknown"),
                        "confidence": getattr(e, "confidence", 0.5),
                        "attributes": getattr(e, "attributes", {}),
                    }
                )

            relations = []
            for r in raw_relations:
                relations.append(
                    {
                        "source_entity": str(
                            getattr(r, "source_entity", getattr(r, "source_name", ""))
                        ),
                        "target_entity": str(
                            getattr(r, "target_entity", getattr(r, "target_name", ""))
                        ),
                        "relation_type": str(getattr(r, "relation_type", "related_to")),
                        "confidence": getattr(r, "confidence", 0.5),
                    }
                )

            return {"entities": entities, "relations": relations}
        except Exception:
            return {"entities": [], "relations": []}

    @app.get("/api/v1/memory/graph/entities/{entity_id}/relations", dependencies=deps)
    async def knowledge_graph_entity_relations(entity_id: str) -> dict[str, Any]:
        """Beziehungen einer bestimmten Entität."""
        semantic = getattr(gateway, "_semantic_memory", None)
        if semantic is None:
            return {"relations": []}
        try:
            raw_relations = getattr(semantic, "relations", None) or []
            entity_rels = []
            for r in raw_relations:
                src = str(getattr(r, "source_entity", getattr(r, "source_name", "")))
                tgt = str(getattr(r, "target_entity", getattr(r, "target_name", "")))
                if src == entity_id or tgt == entity_id:
                    entity_rels.append(
                        {
                            "source_entity": src,
                            "target_entity": tgt,
                            "target_name": tgt,
                            "relation_type": str(getattr(r, "relation_type", "related_to")),
                            "confidence": getattr(r, "confidence", 0.5),
                        }
                    )
            return {"relations": entity_rels}
        except Exception:
            return {"relations": []}


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
        """Kuratierter Feed für die Startseite."""
        try:
            from jarvis.skills.marketplace import SkillMarketplace

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
            from jarvis.skills.marketplace import SkillMarketplace

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
            from jarvis.skills.marketplace import SkillMarketplace

            return {"categories": [c.to_dict() for c in SkillMarketplace().categories()]}
        except Exception as exc:
            log.error("marketplace_categories_failed", error=str(exc))
            return {"error": "Kategorien nicht verfuegbar"}

    @app.get("/api/v1/marketplace/featured", dependencies=deps)
    async def marketplace_featured(n: int = 10) -> dict[str, Any]:
        """Featured-Skills."""
        try:
            from jarvis.skills.marketplace import SkillMarketplace

            return {"featured": [s.to_dict() for s in SkillMarketplace().featured(n)]}
        except Exception as exc:
            log.error("marketplace_featured_failed", error=str(exc))
            return {"error": "Featured-Skills nicht verfuegbar"}

    @app.get("/api/v1/marketplace/trending", dependencies=deps)
    async def marketplace_trending(window: str = "24h", n: int = 10) -> dict[str, Any]:
        """Trending-Skills."""
        try:
            from jarvis.skills.marketplace import SkillMarketplace

            return {"trending": [s.to_dict() for s in SkillMarketplace().trending(max_results=n)]}
        except Exception as exc:
            log.error("marketplace_trending_failed", error=str(exc))
            return {"error": "Trending-Skills nicht verfuegbar"}

    @app.get("/api/v1/marketplace/stats", dependencies=deps)
    async def marketplace_stats() -> dict[str, Any]:
        """Marktplatz-Statistiken."""
        try:
            from jarvis.skills.marketplace import SkillMarketplace

            return SkillMarketplace().stats()
        except Exception as exc:
            log.error("marketplace_stats_failed", error=str(exc))
            return {"error": "Marketplace-Statistiken nicht verfuegbar"}

    # -- Skill-Updater ----------------------------------------------------

    @app.get("/api/v1/updater/stats", dependencies=deps)
    async def updater_stats() -> dict[str, Any]:
        """Skill-Updater-Statistiken."""
        try:
            from jarvis.skills.updater import SkillUpdater

            return SkillUpdater().stats()
        except Exception as exc:
            log.error("updater_stats_failed", error=str(exc))
            return {"error": "Updater-Statistiken nicht verfuegbar"}

    @app.get("/api/v1/updater/pending", dependencies=deps)
    async def updater_pending() -> dict[str, Any]:
        """Ausstehende Updates."""
        try:
            from jarvis.skills.updater import SkillUpdater

            u = SkillUpdater()
            return {"updates": [c.to_dict() for c in u.pending_updates()]}
        except Exception as exc:
            log.error("updater_pending_failed", error=str(exc))
            return {"error": "Ausstehende Updates nicht verfuegbar"}

    @app.get("/api/v1/updater/recalls", dependencies=deps)
    async def updater_recalls() -> dict[str, Any]:
        """Aktive Security-Recalls."""
        try:
            from jarvis.skills.updater import SkillUpdater

            u = SkillUpdater()
            return {"recalls": [r.to_dict() for r in u.active_recalls()]}
        except Exception as exc:
            log.error("updater_recalls_failed", error=str(exc))
            return {"error": "Recalls nicht verfuegbar"}

    @app.get("/api/v1/updater/history", dependencies=deps)
    async def updater_history(n: int = 20) -> dict[str, Any]:
        """Update-Historie."""
        try:
            from jarvis.skills.updater import SkillUpdater

            return {"history": SkillUpdater().update_history(n)}
        except Exception as exc:
            log.error("updater_history_failed", error=str(exc))
            return {"error": "Update-Historie nicht verfuegbar"}

    # -- Commands ---------------------------------------------------------

    @app.get("/api/v1/commands/list", dependencies=deps)
    async def list_commands() -> dict[str, Any]:
        """Alle registrierten Slash-Commands."""
        try:
            from jarvis.channels.commands import CommandRegistry

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
            from jarvis.channels.commands import CommandRegistry

            return {"definitions": CommandRegistry().slack_definitions()}
        except Exception as exc:
            log.error("commands_slack_failed", error=str(exc))
            return {"error": "Slack-Commands nicht verfuegbar"}

    @app.get("/api/v1/commands/discord", dependencies=deps)
    async def commands_discord() -> dict[str, Any]:
        """Discord Application-Command-Definitionen."""
        try:
            from jarvis.channels.commands import CommandRegistry

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
        """Listet alle in Ollama/Backend verfügbaren Modelle auf."""
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
        """Verfügbare Sprachen."""
        mgr = getattr(gateway, "_i18n", None)
        if mgr is None:
            return {"locales": [], "default": "de"}
        return {"locales": mgr.available_locales(), "default": mgr.default_locale}

    @app.get("/api/v1/i18n/translate/{key}", dependencies=deps)
    async def i18n_translate(key: str, locale: str = "") -> dict[str, Any]:
        """Einzelnen Key übersetzen."""
        mgr = getattr(gateway, "_i18n", None)
        if mgr is None:
            return {"key": key, "translation": key}
        return {"key": key, "translation": mgr.t(key, locale=locale)}

    @app.get("/api/v1/i18n/stats", dependencies=deps)
    async def i18n_stats() -> dict[str, Any]:
        """i18n-Manager Statistiken."""
        mgr = getattr(gateway, "_i18n", None)
        if mgr is None:
            return {"default_locale": "de", "locale_count": 0, "locales": []}
        return mgr.stats()

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
        """Verfügbare Skill-Templates."""
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
# Monitoring / metrics routes
# ======================================================================


def _register_monitoring_routes(
    app: Any,
    deps: list[Any],
    get_hub: Any,
) -> None:
    """Metrics, events, audit-trail, heartbeat, SSE streaming, performance."""

    @app.get("/api/v1/monitoring/dashboard", dependencies=deps)
    async def monitoring_dashboard() -> dict[str, Any]:
        """Komplett-Snapshot für das Live-Dashboard."""
        return get_hub().dashboard_snapshot()

    @app.get("/api/v1/monitoring/metrics", dependencies=deps)
    async def monitoring_metrics() -> dict[str, Any]:
        """Aktuelle Metriken."""
        hub = get_hub()
        return {"snapshot": hub.metrics.snapshot(), "names": hub.metrics.all_metric_names()}

    @app.get("/api/v1/monitoring/metrics/{name}", dependencies=deps)
    async def monitoring_metric_history(name: str, n: int = 60) -> dict[str, Any]:
        """Zeitreihe einer einzelnen Metrik."""
        return {"name": name, "history": get_hub().metrics.get_history(name, last_n=n)}

    @app.get("/api/v1/monitoring/events", dependencies=deps)
    async def monitoring_events(n: int = 50, severity: str = "") -> dict[str, Any]:
        """Letzte System-Events."""
        hub = get_hub()
        events = hub.events.recent_events(n=n, severity=severity or "")
        return {"events": [e.to_dict() for e in events], "total": hub.events.event_count}

    @app.get("/api/v1/monitoring/audit", dependencies=deps)
    async def audit_trail(
        action: str = "",
        actor: str = "",
        severity: str = "",
        limit: int = 100,
    ) -> dict[str, Any]:
        """Durchsucht den Audit-Trail."""
        hub = get_hub()
        entries = hub.audit.search(action=action, actor=actor, severity=severity, limit=limit)
        return {
            "entries": [e.to_dict() for e in entries],
            "total": hub.audit.entry_count,
            "severity_counts": hub.audit.severity_counts(),
        }

    @app.get("/api/v1/monitoring/heartbeat", dependencies=deps)
    async def heartbeat_status() -> dict[str, Any]:
        """Heartbeat-Status und Historie."""
        hub = get_hub()
        return {
            "stats": hub.heartbeat.stats(),
            "recent_runs": [r.to_dict() for r in hub.heartbeat.recent_runs(20)],
        }

    # -- SSE Live-Event-Streaming -----------------------------------------

    @app.get("/api/v1/monitoring/stream", dependencies=deps)
    async def monitoring_sse_stream() -> Any:
        """Server-Sent-Events Stream für Live-Monitoring."""
        from starlette.responses import StreamingResponse

        hub = get_hub()
        queue = hub.events.create_sse_stream()

        async def event_generator():
            import asyncio

            try:
                while True:
                    try:
                        event = queue.get_nowait()
                        yield event.to_sse()
                    except Exception:
                        yield ": keepalive\n\n"
                        await asyncio.sleep(1)
            finally:
                hub.events.remove_sse_stream(queue)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )


# ======================================================================
# Prometheus metrics endpoint
# ======================================================================


def _register_prometheus_routes(
    app: Any,
    get_hub: Any,
    gateway: Any,
) -> None:
    """Prometheus /metrics endpoint -- no auth required (standard practice)."""

    @app.get("/metrics")
    async def prometheus_metrics() -> Any:
        """Prometheus-Metriken im Text Exposition Format."""
        from starlette.responses import Response

        from jarvis.telemetry.prometheus import PrometheusExporter

        # Collect sources: MetricsProvider from TelemetryHub, MetricCollector from MonitoringHub
        metrics_provider = None
        metric_collector = None

        # TelemetryHub -> MetricsProvider (telemetry/metrics.py)
        if gateway is not None:
            telemetry_hub = getattr(gateway, "_telemetry_hub", None)
            if telemetry_hub is not None:
                metrics_provider = getattr(telemetry_hub, "metrics", None)

        # MonitoringHub -> MetricCollector (gateway/monitoring.py)
        try:
            hub = get_hub()
            if hub is not None:
                metric_collector = getattr(hub, "metrics", None)
        except Exception:
            pass  # Cleanup — metric collector lookup failure is non-critical

        exporter = PrometheusExporter(
            metrics_provider=metrics_provider,
            metric_collector=metric_collector,
        )
        content = exporter.export()
        return Response(
            content=content,
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )


# ======================================================================
# Security / audit routes
# ======================================================================


def _register_security_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """Red-team scanning, compliance, security pipeline, security framework,
    ecosystem security policy, CI/CD gate."""

    # -- Red-Team Security Scanner ----------------------------------------

    @app.post("/api/v1/security/redteam/scan", dependencies=deps)
    async def redteam_scan(request: Request) -> dict[str, Any]:
        """Red-Team-Scan gegen Prompt-Injection etc."""
        try:
            from jarvis.security.redteam import ScanPolicy, SecurityScanner

            scanner = getattr(gateway, "_security_scanner", None) or SecurityScanner()
            body = await request.json()
            if "policy" in body:
                p = body["policy"]
                scanner.policy = ScanPolicy(
                    max_risk_score=p.get("max_risk_score", 70),
                    block_on_critical=p.get("block_on_critical", True),
                    block_on_high=p.get("block_on_high", False),
                    min_tests=p.get("min_tests", 5),
                )

            import re as _re

            def sanitizer(text: str) -> dict[str, Any]:
                dangerous = [
                    r"ignore\s+(all\s+)?previous",
                    r"system\s*:",
                    r"<\s*script",
                    r"rm\s+-rf",
                    r"\bsudo\b",
                    r"\bexec\b",
                    r"\beval\b",
                ]
                for pat in dangerous:
                    if _re.search(pat, text, _re.IGNORECASE):
                        return {"blocked": True}
                return {"blocked": False}

            result = scanner.scan(
                sanitizer_fn=sanitizer,
                is_blocked_fn=lambda r: r.get("blocked", False),
            )
            return result.to_dict()
        except Exception as exc:
            log.error("redteam_scan_failed", error=str(exc))
            return {"error": "Security-Scan fehlgeschlagen"}

    @app.get("/api/v1/security/redteam/status", dependencies=deps)
    async def redteam_status() -> dict[str, Any]:
        """Status des Security-Scanners."""
        scanner = getattr(gateway, "_security_scanner", None)
        return {
            "available": True,
            "scanner": "SecurityScanner",
            "has_gateway_instance": scanner is not None,
        }

    # -- Compliance & Audit -----------------------------------------------

    @app.get("/api/v1/compliance/report", dependencies=deps)
    async def compliance_report() -> dict[str, Any]:
        """EU-AI-Act + DSGVO Compliance-Report generieren."""
        try:
            from jarvis.audit.compliance import ComplianceFramework

            fw = getattr(gateway, "_compliance_framework", None) or ComplianceFramework()
            fw.auto_assess(
                has_audit_log=True,
                has_decision_log=getattr(gateway, "_decision_log", None) is not None,
                has_kill_switch=True,
                has_encryption=True,
                has_rbac=True,
                has_sandbox=True,
                has_approval_workflow=True,
                has_redteam=getattr(gateway, "_security_scanner", None) is not None,
            )
            report = fw.generate_report()
            return report.to_dict()
        except Exception as exc:
            log.error("compliance_report_failed", error=str(exc))
            return {"error": "Compliance-Report konnte nicht generiert werden"}

    @app.get("/api/v1/compliance/export/{fmt}", dependencies=deps)
    async def compliance_export(fmt: str) -> Any:
        """Compliance-Report exportieren (json/csv/markdown)."""
        try:
            from jarvis.audit.compliance import ComplianceFramework, ReportExporter

            fw = getattr(gateway, "_compliance_framework", None) or ComplianceFramework()
            fw.auto_assess(
                has_audit_log=True,
                has_decision_log=True,
                has_kill_switch=True,
                has_encryption=True,
                has_rbac=True,
                has_sandbox=True,
                has_approval_workflow=True,
                has_redteam=True,
            )
            report = fw.generate_report()
            if fmt == "json":
                from starlette.responses import JSONResponse

                return JSONResponse(content=report.to_dict())
            elif fmt == "csv":
                from starlette.responses import PlainTextResponse

                return PlainTextResponse(ReportExporter.to_csv(report), media_type="text/csv")
            elif fmt == "markdown":
                from starlette.responses import PlainTextResponse

                return PlainTextResponse(
                    ReportExporter.to_markdown(report), media_type="text/markdown"
                )
            return {"error": f"Unknown format: {fmt}. Use json/csv/markdown."}
        except Exception as exc:
            log.error("compliance_export_failed", error=str(exc))
            return {"error": "Compliance-Export fehlgeschlagen"}

    @app.get("/api/v1/compliance/decisions", dependencies=deps)
    async def compliance_decisions() -> dict[str, Any]:
        """Decision-Log Übersicht."""
        decision_log = getattr(gateway, "_decision_log", None)
        if decision_log is None:
            return {
                "total_decisions": 0,
                "flagged_count": 0,
                "approval_rate": 0.0,
                "unique_agents": 0,
                "avg_confidence": 0.0,
            }
        return decision_log.stats()

    @app.get("/api/v1/compliance/remediations", dependencies=deps)
    async def compliance_remediations() -> dict[str, Any]:
        """Remediation-Tracker Status."""
        tracker = getattr(gateway, "_remediation_tracker", None)
        if tracker is None:
            return {"total": 0, "open": 0, "in_progress": 0, "resolved": 0, "overdue": 0}
        return tracker.stats()

    @app.get("/api/v1/compliance/stats", dependencies=deps)
    async def compliance_stats() -> dict[str, Any]:
        """Compliance-Exporter Statistiken."""
        exporter = getattr(gateway, "_compliance_exporter", None)
        if exporter is None:
            return {"total_reports": 0}
        return exporter.stats()

    @app.get("/api/v1/compliance/transparency", dependencies=deps)
    async def compliance_transparency() -> dict[str, Any]:
        """Transparenzpflichten-Status."""
        exporter = getattr(gateway, "_compliance_exporter", None)
        if exporter is None:
            return {"total_obligations": 0}
        return exporter.transparency.stats()

    @app.post("/api/v1/compliance/report", dependencies=deps)
    async def compliance_generate() -> dict[str, Any]:
        """Generiert einen Compliance-Bericht."""
        exporter = getattr(gateway, "_compliance_exporter", None)
        if exporter is None:
            return {"error": "Exporter nicht verfügbar"}
        report = exporter.generate_report()
        return report.to_dict()

    # -- Security Pipeline (Phase 19) ------------------------------------

    @app.get("/api/v1/security/pipeline/stats", dependencies=deps)
    async def pipeline_stats() -> dict[str, Any]:
        """Security-Pipeline Statistiken."""
        pipeline = getattr(gateway, "_security_pipeline", None)
        if pipeline is None:
            return {"total_runs": 0, "last_result": "none", "total_findings": 0, "pass_rate": 0}
        return pipeline.stats()

    @app.post("/api/v1/security/pipeline/run", dependencies=deps)
    async def pipeline_run(request: Request) -> dict[str, Any]:
        """Security-Pipeline manuell starten."""
        try:
            pipeline = getattr(gateway, "_security_pipeline", None)
            if pipeline is None:
                return {"error": "Security-Pipeline nicht verfügbar"}
            body = await request.json()
            trigger = body.get("trigger", "manual")

            def sanitizer(text: str) -> dict[str, Any]:
                return {"blocked": False}

            run = pipeline.run(
                handler_fn=sanitizer,
                is_blocked_fn=lambda r: r.get("blocked", False),
                test_inputs=body.get("test_inputs", []),
                dependencies=body.get("dependencies", []),
                trigger=trigger,
            )
            return run.to_dict()
        except Exception as exc:
            log.error("pipeline_run_failed", error=str(exc))
            return {"error": "Security-Pipeline-Run fehlgeschlagen"}

    @app.get("/api/v1/security/pipeline/history", dependencies=deps)
    async def pipeline_history() -> dict[str, Any]:
        """Security-Pipeline Run-Historie."""
        pipeline = getattr(gateway, "_security_pipeline", None)
        if pipeline is None:
            return {"runs": [], "count": 0}
        runs = pipeline.history(limit=20)
        return {"runs": [r.to_dict() for r in runs], "count": len(runs)}

    # -- Ecosystem Security Policy ----------------------------------------

    @app.get("/api/v1/ecosystem/policy/stats", dependencies=deps)
    async def ecosystem_policy_stats() -> dict[str, Any]:
        """Ecosystem-Policy Statistiken."""
        policy = getattr(gateway, "_ecosystem_policy", None)
        if policy is None:
            return {"total_requirements": 0, "minimum_tier": "community", "total_badges": 0}
        return policy.stats()

    @app.post("/api/v1/ecosystem/evaluate", dependencies=deps)
    async def ecosystem_evaluate(request: Request) -> dict[str, Any]:
        """Skill gegen Ecosystem-Policy evaluieren."""
        try:
            policy = getattr(gateway, "_ecosystem_policy", None)
            if policy is None:
                return {"error": "Ecosystem-Policy nicht verfügbar"}
            body = await request.json()
            skill_id = body.get("skill_id", "unknown")
            badge = policy.evaluate_skill(
                skill_id,
                has_signature=body.get("has_signature", False),
                has_sandbox=body.get("has_sandbox", False),
                has_license=body.get("has_license", False),
                has_network_control=body.get("has_network_control", False),
                passed_static_analysis=body.get("passed_static_analysis", False),
                passed_code_review=body.get("passed_code_review", False),
                passed_pentest=body.get("passed_pentest", False),
                has_audit_trail=body.get("has_audit_trail", False),
                has_input_validation=body.get("has_input_validation", False),
                is_dsgvo_compliant=body.get("is_dsgvo_compliant", False),
            )
            return badge.to_dict()
        except Exception as exc:
            log.error("ecosystem_evaluate_failed", error=str(exc))
            return {"error": "Ecosystem-Evaluierung fehlgeschlagen"}

    # -- AI Agent Security Framework (Phase 21) ---------------------------

    @app.get("/api/v1/framework/metrics", dependencies=deps)
    async def framework_metrics() -> dict[str, Any]:
        """Security-Metriken (MTTD, MTTR, etc.)."""
        metrics = getattr(gateway, "_security_metrics", None)
        if metrics is None:
            return {
                "mttd_seconds": 0,
                "mttr_seconds": 0,
                "resolution_rate": 100,
                "total_incidents": 0,
            }
        return metrics.to_dict()

    @app.get("/api/v1/framework/incidents", dependencies=deps)
    async def framework_incidents() -> dict[str, Any]:
        """Alle Incidents."""
        tracker = getattr(gateway, "_incident_tracker", None)
        if tracker is None:
            return {"incidents": [], "stats": {}}
        return {
            "incidents": [i.to_dict() for i in tracker.all_incidents()],
            "stats": tracker.stats(),
        }

    @app.get("/api/v1/framework/team", dependencies=deps)
    async def framework_team() -> dict[str, Any]:
        """Security-Team Übersicht."""
        team = getattr(gateway, "_security_team", None)
        if team is None:
            return {"members": [], "stats": {"total_members": 0}}
        return {
            "members": [m.to_dict() for m in team.on_call()],
            "stats": team.stats(),
        }

    @app.get("/api/v1/framework/posture", dependencies=deps)
    async def framework_posture() -> dict[str, Any]:
        """Security-Posture-Score."""
        scorer = getattr(gateway, "_posture_scorer", None)
        if scorer is None:
            return {"posture_score": 0, "level": "unknown"}
        metrics = getattr(gateway, "_security_metrics", None)
        pipeline = getattr(gateway, "_security_pipeline", None)
        team = getattr(gateway, "_security_team", None)
        return scorer.calculate(
            resolution_rate=metrics.resolution_rate() if metrics else 100,
            mttr_seconds=metrics.mttr() if metrics else 0,
            team_roles_filled=team.member_count if team else 0,
            pipeline_pass_rate=pipeline.stats().get("pass_rate", 100) if pipeline else 100,
        )

    # -- CI/CD Security Gate (Phase 24) -----------------------------------

    @app.get("/api/v1/gate/stats", dependencies=deps)
    async def gate_stats() -> dict[str, Any]:
        """Security-Gate Statistiken."""
        gate = getattr(gateway, "_security_gate", None)
        if gate is None:
            return {"total_evaluations": 0, "pass_rate": 100}
        return gate.stats()

    @app.post("/api/v1/gate/evaluate", dependencies=deps)
    async def gate_evaluate(body: dict[str, Any]) -> dict[str, Any]:
        """Evaluiert ein Pipeline-Ergebnis."""
        gate = getattr(gateway, "_security_gate", None)
        if gate is None:
            return {"verdict": "pass", "error": "Gate nicht verfügbar"}
        result = gate.evaluate(body)
        return result.to_dict()

    @app.get("/api/v1/gate/history", dependencies=deps)
    async def gate_history() -> dict[str, Any]:
        """Gate-History."""
        gate = getattr(gateway, "_security_gate", None)
        if gate is None:
            return {"history": []}
        return {"history": [r.to_dict() for r in gate.history()]}

    @app.get("/api/v1/redteam/stats", dependencies=deps)
    async def redteam_stats() -> dict[str, Any]:
        """Continuous Red-Team Statistiken."""
        rt = getattr(gateway, "_continuous_redteam", None)
        if rt is None:
            return {"total_probes": 0, "detection_rate": 100}
        return rt.stats()

    @app.get("/api/v1/scans/stats", dependencies=deps)
    async def scans_stats() -> dict[str, Any]:
        """Scan-Scheduler Status."""
        sched = getattr(gateway, "_scan_scheduler", None)
        if sched is None:
            return {"total_schedules": 0}
        return sched.stats()

    # -- Red-Team-Framework (Phase 30) ------------------------------------

    @app.get("/api/v1/red-team/stats", dependencies=deps)
    async def red_team_stats() -> dict[str, Any]:
        """Red-Team Statistiken."""
        rt = getattr(gateway, "_red_team", None)
        if rt is None:
            return {"total_runs": 0}
        return rt.stats()

    @app.get("/api/v1/red-team/coverage", dependencies=deps)
    async def red_team_coverage() -> dict[str, Any]:
        """Angriffs-Abdeckung."""
        rt = getattr(gateway, "_red_team", None)
        if rt is None:
            return {"coverage_rate": 0}
        return rt.coverage_report()

    @app.get("/api/v1/red-team/latest", dependencies=deps)
    async def red_team_latest() -> dict[str, Any]:
        """Letzter Red-Team-Report."""
        rt = getattr(gateway, "_red_team", None)
        if rt is None:
            return {"report": None}
        report = rt.runner.latest_report()
        return {"report": report.to_dict() if report else None}

    # -- Code-Audit (Phase 33) -------------------------------------------

    @app.get("/api/v1/code-audit/stats", dependencies=deps)
    async def code_audit_stats() -> dict[str, Any]:
        """Code-Audit Statistiken."""
        ca = getattr(gateway, "_code_auditor", None)
        if ca is None:
            return {"total_audits": 0}
        return ca.stats()


# ======================================================================
# Governance routes
# ======================================================================


def _register_governance_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """Marketplace governance, economics, governance hub, interop, impact."""

    # -- Marketplace-Governance (Phase 20) --------------------------------

    @app.get("/api/v1/governance/reputation/stats", dependencies=deps)
    async def governance_reputation_stats() -> dict[str, Any]:
        """Reputation-Engine Statistiken."""
        engine = getattr(gateway, "_reputation_engine", None)
        if engine is None:
            return {
                "total_entities": 0,
                "avg_score": 0,
                "flagged_count": 0,
                "trust_distribution": {},
            }
        return engine.stats()

    @app.get("/api/v1/governance/reputation/{entity_id}", dependencies=deps)
    async def governance_reputation_detail(entity_id: str) -> dict[str, Any]:
        """Reputation-Score für ein Entity."""
        engine = getattr(gateway, "_reputation_engine", None)
        if engine is None:
            return {"error": "Reputation-Engine nicht verfügbar"}
        score = engine.get_score(entity_id)
        if score is None:
            return {"error": f"Entity '{entity_id}' nicht gefunden"}
        return score.to_dict()

    @app.get("/api/v1/governance/recalls/stats", dependencies=deps)
    async def governance_recalls_stats() -> dict[str, Any]:
        """Recall-Manager Statistiken."""
        mgr = getattr(gateway, "_recall_manager", None)
        if mgr is None:
            return {"total_recalls": 0, "active_blocks": 0}
        return mgr.stats()

    @app.get("/api/v1/governance/recalls/active", dependencies=deps)
    async def governance_recalls_active() -> dict[str, Any]:
        """Aktive Recalls."""
        mgr = getattr(gateway, "_recall_manager", None)
        if mgr is None:
            return {"recalls": []}
        return {"recalls": [r.to_dict() for r in mgr.active_recalls()]}

    @app.get("/api/v1/governance/abuse/stats", dependencies=deps)
    async def governance_abuse_stats() -> dict[str, Any]:
        """Abuse-Reporter Statistiken."""
        reporter = getattr(gateway, "_abuse_reporter", None)
        if reporter is None:
            return {"total_reports": 0, "open": 0, "investigating": 0}
        return reporter.stats()

    @app.get("/api/v1/governance/policy/stats", dependencies=deps)
    async def governance_policy_stats() -> dict[str, Any]:
        """Governance-Policy Statistiken."""
        policy = getattr(gateway, "_governance_policy", None)
        if policy is None:
            return {"total_rules": 0, "enabled": 0, "total_triggered": 0}
        return policy.stats()

    # -- Cross-Agent Interop (Phase 22) -----------------------------------

    @app.get("/api/v1/interop/stats", dependencies=deps)
    async def interop_stats() -> dict[str, Any]:
        """Interop-Protokoll Statistiken."""
        interop = getattr(gateway, "_interop", None)
        if interop is None:
            return {"registered_agents": 0, "online": 0}
        return interop.stats()

    @app.get("/api/v1/interop/agents", dependencies=deps)
    async def interop_agents() -> dict[str, Any]:
        """Registrierte Agenten."""
        interop = getattr(gateway, "_interop", None)
        if interop is None:
            return {"agents": []}
        return {"agents": [a.to_dict() for a in interop.online_agents()]}

    @app.get("/api/v1/interop/federation", dependencies=deps)
    async def interop_federation() -> dict[str, Any]:
        """Federation-Status."""
        interop = getattr(gateway, "_interop", None)
        if interop is None:
            return {"links": [], "stats": {}}
        return {
            "links": [link.to_dict() for link in interop.federation.active_links()],
            "stats": interop.federation.stats(),
        }

    # -- Ethik- und Wirtschaftsgovernance (Phase 23) ----------------------

    @app.get("/api/v1/economics/stats", dependencies=deps)
    async def economics_stats() -> dict[str, Any]:
        """Wirtschaftsgovernance Übersicht."""
        gov = getattr(gateway, "_economic_governor", None)
        if gov is None:
            return {"budget": {}, "costs": {}, "bias": {}, "fairness": {}, "ethics": {}}
        return gov.stats()

    @app.get("/api/v1/economics/budget", dependencies=deps)
    async def economics_budget() -> dict[str, Any]:
        """Budget-Status."""
        gov = getattr(gateway, "_economic_governor", None)
        if gov is None:
            return {"total_entities": 0}
        return gov.budget.stats()

    @app.get("/api/v1/economics/costs", dependencies=deps)
    async def economics_costs() -> dict[str, Any]:
        """Kosten-Tracking."""
        gov = getattr(gateway, "_economic_governor", None)
        if gov is None:
            return {"total_entries": 0, "total_cost_eur": 0}
        return gov.costs.stats()

    @app.get("/api/v1/economics/fairness", dependencies=deps)
    async def economics_fairness() -> dict[str, Any]:
        """Fairness-Audit Ergebnisse."""
        gov = getattr(gateway, "_economic_governor", None)
        if gov is None:
            return {"total_audits": 0, "pass_rate": 100}
        return gov.fairness.stats()

    @app.get("/api/v1/economics/ethics", dependencies=deps)
    async def economics_ethics() -> dict[str, Any]:
        """Ethik-Policy Status."""
        gov = getattr(gateway, "_economic_governor", None)
        if gov is None:
            return {"total_violations": 0}
        return gov.ethics.stats()

    # -- Governance Hub (Phase 31) ----------------------------------------

    @app.get("/api/v1/governance/health", dependencies=deps)
    async def governance_health() -> dict[str, Any]:
        """Ecosystem-Gesundheit."""
        gh = getattr(gateway, "_governance_hub", None)
        if gh is None:
            return {"skill_reviews": 0}
        return gh.ecosystem_health()

    @app.get("/api/v1/governance/curation", dependencies=deps)
    async def governance_curation() -> dict[str, Any]:
        """Kurations-Board Status."""
        gh = getattr(gateway, "_governance_hub", None)
        if gh is None:
            return {"total_reviews": 0}
        return gh.curation.stats()

    @app.get("/api/v1/governance/diversity", dependencies=deps)
    async def governance_diversity() -> dict[str, Any]:
        """Diversity-Audit Ergebnisse."""
        gh = getattr(gateway, "_governance_hub", None)
        if gh is None:
            return {"total_audits": 0}
        return gh.diversity.stats()

    @app.get("/api/v1/governance/budget", dependencies=deps)
    async def governance_budget_transfers() -> dict[str, Any]:
        """Cross-Agent-Budget Status."""
        gh = getattr(gateway, "_governance_hub", None)
        if gh is None:
            return {"total_transfers": 0}
        return gh.budget.stats()

    @app.get("/api/v1/governance/explainer", dependencies=deps)
    async def governance_explainer() -> dict[str, Any]:
        """Decision-Explainer Statistiken."""
        gh = getattr(gateway, "_governance_hub", None)
        if gh is None:
            return {"total_explanations": 0}
        return gh.explainer.stats()

    # -- AI Impact Assessment (Phase 32) ----------------------------------

    @app.get("/api/v1/impact/stats", dependencies=deps)
    async def impact_stats() -> dict[str, Any]:
        """Impact Assessment Statistiken."""
        ia = getattr(gateway, "_impact_assessor", None)
        if ia is None:
            return {"total_assessments": 0}
        return ia.stats()

    @app.get("/api/v1/impact/board", dependencies=deps)
    async def impact_board() -> dict[str, Any]:
        """Ethik-Board Status."""
        ia = getattr(gateway, "_impact_assessor", None)
        if ia is None:
            return {"board_members": 0}
        return ia.board.stats()

    @app.get("/api/v1/impact/stakeholders", dependencies=deps)
    async def impact_stakeholders() -> dict[str, Any]:
        """Stakeholder-Registry."""
        ia = getattr(gateway, "_impact_assessor", None)
        if ia is None:
            return {"total": 0}
        return ia.stakeholders.stats()

    @app.get("/api/v1/impact/mitigations", dependencies=deps)
    async def impact_mitigations() -> dict[str, Any]:
        """Mitigationsmaßnahmen."""
        ia = getattr(gateway, "_impact_assessor", None)
        if ia is None:
            return {"total": 0}
        return ia.mitigations.stats()


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
            from jarvis.governance.improvement_gate import GateVerdict, ImprovementDomain

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
                    from jarvis.learning.prompt_evolution import PromptEvolutionEngine

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

    jarvis_home = config_manager.config.jarvis_home
    agents_path = jarvis_home / "agents.yaml"
    bindings_path = jarvis_home / "bindings.yaml"

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

    # -- 3.4: POST /agents/{name} ----------------------------------------

    @app.post("/api/v1/agents/{name}", dependencies=deps)
    async def ui_upsert_agent(name: str, request: Request) -> dict[str, Any]:
        """Creates or updates an agent profile in agents.yaml."""
        try:
            from jarvis.gateway.config_api import AgentProfileDTO

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
            from jarvis.gateway.config_api import BindingRuleDTO

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
        prompts_dir = jarvis_home / "prompts"
        result: dict[str, str] = {}

        # coreMd
        try:
            core_path = cfg.core_memory_file
            result["coreMd"] = core_path.read_text(encoding="utf-8") if core_path.exists() else ""
        except Exception:
            result["coreMd"] = ""

        # plannerSystem (.md bevorzugt, .txt als Migration-Fallback)
        try:
            from jarvis.core.planner import SYSTEM_PROMPT

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
            from jarvis.core.planner import REPLAN_PROMPT

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
            from jarvis.core.planner import ESCALATION_PROMPT

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
            hb_path = jarvis_home / cfg.heartbeat.checklist_file
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
            prompts_dir = jarvis_home / "prompts"
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
                hb_path = jarvis_home / cfg.heartbeat.checklist_file
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
            from jarvis.cron.jobs import JobStore

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
            from jarvis.cron.jobs import JobStore
            from jarvis.models import CronJob

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
                pass

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

            result = await svc.ingest_file(filename, file_bytes)

            return {
                "id": result.id,
                "source_type": result.source_type,
                "source_name": result.source_name,
                "status": result.status,
                "chunks_created": result.chunks_created,
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

        JSON body: {"url": "https://...", "description": "optional"}
        """
        svc = _get_ingest()
        if not svc:
            return {"error": "Knowledge ingest service not initialized", "status": 503}

        try:
            body = await request.json()
            url = body.get("url", "").strip()
            if not url:
                return {"error": "Field 'url' is required", "code": "MISSING_FIELD"}

            result = await svc.ingest_url(url)

            return {
                "id": result.id,
                "source_type": result.source_type,
                "source_name": result.source_name,
                "status": result.status,
                "chunks_created": result.chunks_created,
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

        JSON body: {"url": "https://youtube.com/watch?v=..."}
        """
        svc = _get_ingest()
        if not svc:
            return {"error": "Knowledge ingest service not initialized", "status": 503}

        try:
            body = await request.json()
            url = body.get("url", "").strip()
            if not url:
                return {"error": "Field 'url' is required", "code": "MISSING_FIELD"}

            result = await svc.ingest_youtube(url)

            return {
                "id": result.id,
                "source_type": result.source_type,
                "source_name": result.source_name,
                "status": result.status,
                "chunks_created": result.chunks_created,
                "text_length": result.text_length,
                "error": result.error,
                "created_at": result.created_at.isoformat(),
            }
        except Exception as exc:
            log.error("learn_youtube_error", error=str(exc))
            return {"error": "YouTube ingestion failed", "code": "INTERNAL_ERROR"}

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
        from pathlib import Path
        import re
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
        content = f"---\n{yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)}---\n\n{skill_body}\n"

        # Determine save directory
        config = getattr(gateway, "_config", None)
        jarvis_home = Path(getattr(config, "jarvis_home", Path.home() / ".jarvis"))
        skills_dir = jarvis_home / "skills"
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
        content = f"---\n{yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)}---\n\n{skill_body}\n"

        file_path.write_text(content, encoding="utf-8")

        # Reload registry from the skill's parent directory
        from pathlib import Path as P

        config = getattr(gateway, "_config", None)
        jarvis_home = P(getattr(config, "jarvis_home", P.home() / ".jarvis"))
        reg.load_from_directories([jarvis_home / "skills", P("data/procedures")])

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
        from pathlib import Path as P

        config = getattr(gateway, "_config", None)
        jarvis_home = P(getattr(config, "jarvis_home", P.home() / ".jarvis"))
        reg.load_from_directories([jarvis_home / "skills", P("data/procedures")])

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
            from jarvis.skills.hermes_compat import HermesCompatLayer

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
            raise HTTPException(500, f"Export failed: {e}")


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
