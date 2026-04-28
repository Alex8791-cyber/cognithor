"""Cognithor · System / health / status / dashboard routes.

Sub-Modul des `config_routes`-Pakets (siehe
`docs/superpowers/plans/2026-04-29-config-routes-split.md`). Enthaelt
`_register_system_routes()` — registriert Dashboard, Status, Overview,
Agents, Credentials, Bindings, Circles, Sandbox, Wizards, RBAC,
Auth-Stats und Agent-Heartbeat-Endpoints.
"""

from __future__ import annotations

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

from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.config_manager import ConfigManager

log = get_logger(__name__)


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
        """Gibt den aktuellen System-Status zurueck."""
        status: dict[str, Any] = {
            "timestamp": time.time(),
            "config_version": config_manager.config.version,
            "owner": config_manager.config.owner_name,
        }

        # RuntimeMonitor
        try:
            from cognithor.openclaw.runtime_monitor import RuntimeMonitor

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
        """Gibt eine kompakte Konfigurationsuebersicht zurueck."""
        try:
            from cognithor.gateway.config_api import ConfigManager as CfgMgr

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
            agents_path = config_manager.config.cognithor_home / "agents.yaml"
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
                        "top_p": None,
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
        agents_path = config_manager.config.cognithor_home / "agents.yaml"
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
                    "top_p": getattr(agent_obj, "top_p", None),
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
                "top_p": None,
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

        agents_path = config_manager.config.cognithor_home / "agents.yaml"
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
            "top_p": body.get("top_p"),
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
        # Live-reload agent router
        router = getattr(gateway, "_agent_router", None)
        if router and hasattr(router, "reload_from_yaml"):
            router.reload_from_yaml(agents_path)
        return {"status": "created", "agent": agent}

    @app.put("/api/v1/agents/{agent_name}", dependencies=deps)
    async def update_agent(agent_name: str, request: Request) -> dict[str, Any]:
        """Update an existing agent profile."""
        body = await request.json()
        agents_path = config_manager.config.cognithor_home / "agents.yaml"
        if not agents_path.exists():
            raise HTTPException(404, f"Agent '{agent_name}' not found")

        raw = yaml.safe_load(agents_path.read_text(encoding="utf-8")) or {}
        agents = raw.get("agents", [])
        found = False
        updated_agent = None
        for i, a in enumerate(agents):
            if a.get("name") == agent_name:
                # Update fields (including name rename)
                for key in [
                    "name",
                    "display_name",
                    "description",
                    "system_prompt",
                    "language",
                    "preferred_model",
                    "temperature",
                    "top_p",
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
        # Live-reload agent router
        router = getattr(gateway, "_agent_router", None)
        if router and hasattr(router, "reload_from_yaml"):
            router.reload_from_yaml(agents_path)
        return {"status": "updated", "agent": updated_agent}

    @app.delete("/api/v1/agents/{agent_name}", dependencies=deps)
    async def delete_agent(agent_name: str) -> dict[str, Any]:
        """Delete an agent profile."""
        if agent_name == "jarvis":
            raise HTTPException(403, "Cannot delete the default agent")

        agents_path = config_manager.config.cognithor_home / "agents.yaml"
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
        # Live-reload agent router
        router = getattr(gateway, "_agent_router", None)
        if router and hasattr(router, "reload_from_yaml"):
            router.reload_from_yaml(agents_path)
        return {"status": "deleted", "name": agent_name}

    # -- Credentials ------------------------------------------------------

    @app.get("/api/v1/credentials", dependencies=deps)
    async def list_credentials() -> dict[str, Any]:
        """Listet alle gespeicherten Credentials (nur Keys, keine Werte)."""
        try:
            from cognithor.security.credentials import CredentialStore

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
            from cognithor.security.credentials import CredentialStore

            store = CredentialStore()
            store.store(service, key, value, agent_id=agent_id)
            return {"status": "ok", "service": service, "key": key, "scope": agent_id or "global"}
        except Exception as exc:
            log.error("credential_store_failed", error=str(exc))
            return {"error": "Credential konnte nicht gespeichert werden", "status": 500}

    @app.delete("/api/v1/credentials/{service}/{key}", dependencies=deps)
    async def delete_credential(service: str, key: str, agent_id: str = "") -> dict[str, Any]:
        """Loescht ein Credential."""
        try:
            from cognithor.security.credentials import CredentialStore

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
            bindings_path = config_manager.config.cognithor_home / "bindings.yaml"
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
            from cognithor.gateway.config_api import BindingRuleDTO
            from cognithor.gateway.config_api import ConfigManager as CfgMgr

            cfg_mgr = CfgMgr(config_manager.config)
            dto = BindingRuleDTO(**data)
            return {"binding": cfg_mgr.upsert_binding(dto), "status": "ok"}
        except Exception as exc:
            log.error("binding_create_failed", error=str(exc))
            return {"error": "Binding konnte nicht erstellt werden", "status": 400}

    @app.delete("/api/v1/bindings/{name}", dependencies=deps)
    async def delete_binding(name: str) -> dict[str, Any]:
        """Loescht eine Binding-Regel."""
        try:
            from cognithor.gateway.config_api import ConfigManager as CfgMgr

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
            from cognithor.skills.circles import CircleManager

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
            from cognithor.skills.circles import CircleManager

            return CircleManager().stats()
        except Exception as exc:
            log.error("circles_stats_failed", error=str(exc))
            return {"error": "Circle-Statistiken nicht verfuegbar"}

    # -- Sandbox ----------------------------------------------------------

    @app.get("/api/v1/sandbox", dependencies=deps)
    async def get_sandbox() -> dict[str, Any]:
        """Liest Sandbox-Konfiguration."""
        try:
            from cognithor.gateway.config_api import ConfigManager as CfgMgr

            cfg_mgr = CfgMgr(config_manager.config)
            return {"sandbox": cfg_mgr.get_sandbox()}
        except Exception as exc:
            log.error("sandbox_get_failed", error=str(exc))
            return {"error": "Sandbox-Konfiguration nicht verfuegbar"}

    @app.patch("/api/v1/sandbox", dependencies=deps)
    async def update_sandbox(values: dict[str, Any]) -> dict[str, Any]:
        """Aktualisiert Sandbox-Einstellungen."""
        try:
            from cognithor.gateway.config_api import ConfigManager as CfgMgr
            from cognithor.gateway.config_api import SandboxUpdate

            cfg_mgr = CfgMgr(config_manager.config)
            update = SandboxUpdate(**values)
            return {"sandbox": cfg_mgr.update_sandbox(update), "status": "ok"}
        except Exception as exc:
            log.error("sandbox_update_failed", error=str(exc))
            return {"error": "Sandbox konnte nicht aktualisiert werden", "status": 400}

    # -- Wizards ----------------------------------------------------------

    @app.get("/api/v1/wizards", dependencies=deps)
    async def list_wizards() -> dict[str, Any]:
        """Alle verfuegbaren Konfigurations-Assistenten."""
        from cognithor.gateway.wizards import WizardRegistry

        reg = WizardRegistry()
        return {"wizards": reg.list_wizards(), "count": reg.wizard_count}

    @app.get("/api/v1/wizards/{wizard_type}", dependencies=deps)
    async def get_wizard(wizard_type: str) -> dict[str, Any]:
        """Details eines Wizards (Schritte + Templates)."""
        from cognithor.gateway.wizards import WizardRegistry

        reg = WizardRegistry()
        wizard = reg.get(wizard_type)
        if not wizard:
            return {"error": f"Wizard '{wizard_type}' nicht gefunden"}
        return wizard.to_dict()

    @app.post("/api/v1/wizards/{wizard_type}/run", dependencies=deps)
    async def run_wizard(wizard_type: str, body: dict[str, Any]) -> dict[str, Any]:
        """Fuehrt einen Wizard aus und generiert Konfiguration."""
        from cognithor.gateway.wizards import WizardRegistry

        reg = WizardRegistry()
        result = reg.run_wizard(wizard_type, body.get("values", {}))
        if not result:
            return {"error": f"Wizard '{wizard_type}' nicht gefunden"}
        return result.to_dict()

    @app.get("/api/v1/wizards/{wizard_type}/templates", dependencies=deps)
    async def wizard_templates(wizard_type: str) -> dict[str, Any]:
        """Templates eines Wizards."""
        from cognithor.gateway.wizards import WizardRegistry

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
        """Alle verfuegbaren Rollen und ihre Berechtigungen."""
        from cognithor.gateway.wizards import ROLE_PERMISSIONS

        return {
            "roles": {
                role.value: {"permissions": [p.key for p in perms], "count": len(perms)}
                for role, perms in ROLE_PERMISSIONS.items()
            }
        }

    @app.get("/api/v1/rbac/check", dependencies=deps)
    async def rbac_check(user_id: str, resource: str, action: str) -> dict[str, Any]:
        """Prueft eine Berechtigung."""
        from cognithor.gateway.wizards import RBACManager

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
            from cognithor.gateway.auth import AuthGateway

            return AuthGateway().stats()
        except Exception as exc:
            log.error("auth_stats_failed", error=str(exc))
            return {"error": "Auth-Statistiken nicht verfuegbar"}

    # -- Agent Heartbeat --------------------------------------------------

    @app.get("/api/v1/agent-heartbeat/dashboard", dependencies=deps)
    async def agent_heartbeat_dashboard() -> dict[str, Any]:
        """Globale Dashboard-Uebersicht aller Agent-Heartbeats."""
        try:
            from cognithor.core.agent_heartbeat import AgentHeartbeatScheduler

            return AgentHeartbeatScheduler().global_dashboard()
        except Exception as exc:
            log.error("heartbeat_dashboard_failed", error=str(exc))
            return {"error": "Heartbeat-Dashboard nicht verfuegbar"}

    @app.get("/api/v1/agent-heartbeat/{agent_id}", dependencies=deps)
    async def agent_heartbeat_summary(agent_id: str) -> dict[str, Any]:
        """Heartbeat-Zusammenfassung fuer einen Agent."""
        try:
            from cognithor.core.agent_heartbeat import AgentHeartbeatScheduler

            return AgentHeartbeatScheduler().agent_summary(agent_id)
        except Exception as exc:
            log.error("heartbeat_summary_failed", agent_id=agent_id, error=str(exc))
            return {"error": "Heartbeat-Zusammenfassung nicht verfuegbar"}
