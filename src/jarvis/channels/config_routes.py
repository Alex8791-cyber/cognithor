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
import json
import time
from pathlib import Path
from typing import Any

import yaml

try:
    from starlette.requests import Request
except ImportError:
    Request = Any  # type: ignore[assignment,misc]

from jarvis.config_manager import ConfigManager
from jarvis.utils.logging import get_logger

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
    _register_config_routes(app, deps, config_manager)
    _register_session_routes(app, deps, gateway)
    _register_memory_routes(app, deps, gateway)
    _register_skill_routes(app, deps, gateway)
    _register_monitoring_routes(app, deps, _get_hub)
    _register_security_routes(app, deps, gateway)
    _register_governance_routes(app, deps, gateway)
    _register_infrastructure_routes(app, deps, gateway)
    _register_portal_routes(app, deps, gateway)
    _register_ui_routes(app, deps, config_manager, gateway)


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
                agents = [{
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
                }]
            return {"agents": agents}
        except Exception as exc:
            return {"agents": [], "error": str(exc)}

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
                    {"service": s, "key": k, "scope": "global"}
                    for s, k in global_creds
                ],
            }
        except Exception as exc:
            return {"credentials": [], "error": str(exc)}

    @app.post("/api/v1/credentials", dependencies=deps)
    async def store_credential(
        service: str, key: str, value: str, agent_id: str = "",
    ) -> dict[str, Any]:
        """Speichert ein Credential."""
        try:
            from jarvis.security.credentials import CredentialStore
            store = CredentialStore()
            store.store(service, key, value, agent_id=agent_id)
            return {"status": "ok", "service": service, "key": key, "scope": agent_id or "global"}
        except Exception as exc:
            return {"error": str(exc), "status": 500}

    @app.delete("/api/v1/credentials/{service}/{key}", dependencies=deps)
    async def delete_credential(service: str, key: str, agent_id: str = "") -> dict[str, Any]:
        """Löscht ein Credential."""
        try:
            from jarvis.security.credentials import CredentialStore
            store = CredentialStore()
            store.store(service, key, "", agent_id=agent_id)
            return {"status": "ok", "deleted": f"{service}:{key}"}
        except Exception as exc:
            return {"error": str(exc), "status": 500}

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
            return {"bindings": [], "error": str(exc)}

    @app.post("/api/v1/bindings", dependencies=deps)
    async def create_binding(data: dict[str, Any]) -> dict[str, Any]:
        """Erstellt oder aktualisiert eine Binding-Regel."""
        try:
            from jarvis.gateway.config_api import BindingRuleDTO, ConfigManager as CfgMgr
            cfg_mgr = CfgMgr(config_manager.config)
            dto = BindingRuleDTO(**data)
            return {"binding": cfg_mgr.upsert_binding(dto), "status": "ok"}
        except Exception as exc:
            return {"error": str(exc), "status": 400}

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
            return {"error": str(exc), "status": 500}

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
            return {"circles": [], "error": str(exc)}

    @app.get("/api/v1/circles/stats", dependencies=deps)
    async def circles_stats() -> dict[str, Any]:
        """Ecosystem-Statistiken."""
        try:
            from jarvis.skills.circles import CircleManager
            return CircleManager().stats()
        except Exception as exc:
            return {"error": str(exc)}

    # -- Sandbox ----------------------------------------------------------

    @app.get("/api/v1/sandbox", dependencies=deps)
    async def get_sandbox() -> dict[str, Any]:
        """Liest Sandbox-Konfiguration."""
        try:
            from jarvis.gateway.config_api import ConfigManager as CfgMgr
            cfg_mgr = CfgMgr(config_manager.config)
            return {"sandbox": cfg_mgr.get_sandbox()}
        except Exception as exc:
            return {"error": str(exc)}

    @app.patch("/api/v1/sandbox", dependencies=deps)
    async def update_sandbox(values: dict[str, Any]) -> dict[str, Any]:
        """Aktualisiert Sandbox-Einstellungen."""
        try:
            from jarvis.gateway.config_api import ConfigManager as CfgMgr, SandboxUpdate
            cfg_mgr = CfgMgr(config_manager.config)
            update = SandboxUpdate(**values)
            return {"sandbox": cfg_mgr.update_sandbox(update), "status": "ok"}
        except Exception as exc:
            return {"error": str(exc), "status": 400}

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
        return {"templates": [
            {"id": t.template_id, "name": t.name, "description": t.description,
             "icon": t.icon, "preset_values": t.preset_values}
            for t in wizard.templates
        ]}

    # -- RBAC -------------------------------------------------------------

    @app.get("/api/v1/rbac/roles", dependencies=deps)
    async def rbac_roles() -> dict[str, Any]:
        """Alle verfügbaren Rollen und ihre Berechtigungen."""
        from jarvis.gateway.wizards import ROLE_PERMISSIONS
        return {"roles": {
            role.value: {"permissions": [p.key for p in perms], "count": len(perms)}
            for role, perms in ROLE_PERMISSIONS.items()
        }}

    @app.get("/api/v1/rbac/check", dependencies=deps)
    async def rbac_check(user_id: str, resource: str, action: str) -> dict[str, Any]:
        """Prüft eine Berechtigung."""
        from jarvis.gateway.wizards import RBACManager
        mgr = RBACManager()
        return {
            "user_id": user_id, "resource": resource, "action": action,
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
            return {"error": str(exc)}

    # -- Agent Heartbeat --------------------------------------------------

    @app.get("/api/v1/agent-heartbeat/dashboard", dependencies=deps)
    async def agent_heartbeat_dashboard() -> dict[str, Any]:
        """Globale Dashboard-Übersicht aller Agent-Heartbeats."""
        try:
            from jarvis.core.agent_heartbeat import AgentHeartbeatScheduler
            return AgentHeartbeatScheduler().global_dashboard()
        except Exception as exc:
            return {"error": str(exc)}

    @app.get("/api/v1/agent-heartbeat/{agent_id}", dependencies=deps)
    async def agent_heartbeat_summary(agent_id: str) -> dict[str, Any]:
        """Heartbeat-Zusammenfassung für einen Agent."""
        try:
            from jarvis.core.agent_heartbeat import AgentHeartbeatScheduler
            return AgentHeartbeatScheduler().agent_summary(agent_id)
        except Exception as exc:
            return {"error": str(exc)}


# ======================================================================
# Config read / write routes
# ======================================================================


def _register_config_routes(
    app: Any,
    deps: list[Any],
    config_manager: ConfigManager,
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
                results.append({"key": key, "status": "error", "error": str(exc)})
        config_manager.save()
        return {"results": results}

    @app.post("/api/v1/config/reload", dependencies=deps)
    async def reload_config() -> dict[str, Any]:
        """Lädt die Konfiguration neu aus der Datei."""
        config_manager.reload()
        return {"status": "ok", "message": "Konfiguration neu geladen"}

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
                        "channels": {"cli_enabled": True, "telegram_enabled": False, "webui_enabled": False},
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
                results.append({"section": section, "status": "error", "error": str(exc)})
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
        existing_section = raw_cfg.get(section, {}) if isinstance(raw_cfg.get(section), dict) else {}
        cleaned = _deep_clean_secrets(values, existing_section)
        try:
            config_manager.update_section(section, cleaned)
            config_manager.save()
            return {"status": "ok", "section": section, "updated_keys": list(cleaned.keys())}
        except ValueError as exc:
            return {"error": str(exc), "status": 400}


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
            return {"error": str(exc)}

    @app.get("/api/v1/isolation/quotas", dependencies=deps)
    async def isolation_quotas() -> dict[str, Any]:
        """Quota-Übersicht aller Agents."""
        try:
            from jarvis.core.isolation import MultiUserIsolation
            iso = MultiUserIsolation()
            return {"quotas": iso.all_quota_summaries()}
        except Exception as exc:
            return {"error": str(exc)}

    @app.get("/api/v1/isolation/violations", dependencies=deps)
    async def isolation_violations() -> dict[str, Any]:
        """Workspace-Violations."""
        try:
            from jarvis.core.isolation import WorkspaceGuard
            guard = WorkspaceGuard()
            return {"violations": guard.violations, "count": guard.violation_count}
        except Exception as exc:
            return {"error": str(exc)}

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
        except Exception as exc:
            log.exception("Error during memory hygiene scan")
            return {"error": "Internal error during memory hygiene scan"}

    @app.get("/api/v1/memory/hygiene/stats", dependencies=deps)
    async def memory_hygiene_stats() -> dict[str, Any]:
        """Memory-Hygiene Statistiken."""
        engine = getattr(gateway, "_memory_hygiene", None)
        if engine is None:
            return {"total_scans": 0, "total_scanned": 0, "total_threats": 0, "quarantined": 0, "threat_rate": 0.0}
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
            return {"total_requests": 0, "active_trails": 0, "completed_trails": 0, "avg_confidence": 0.0}
        return engine.stats()

    @app.get("/api/v1/explainability/low-trust", dependencies=deps)
    async def explainability_low_trust() -> dict[str, Any]:
        """Trails mit niedrigem Trust-Score."""
        engine = getattr(gateway, "_explainability", None)
        if engine is None:
            return {"low_trust_trails": [], "count": 0}
        trails = engine.low_trust_trails(threshold=0.5)
        return {"low_trust_trails": [t.to_dict() for t in trails], "count": len(trails)}


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
            return {"error": str(exc)}

    @app.get("/api/v1/marketplace/search", dependencies=deps)
    async def marketplace_search(
        q: str = "", category: str = "", verified_only: bool = False,
        sort_by: str = "relevance", max_results: int = 20,
    ) -> dict[str, Any]:
        """Durchsucht den Skill-Marktplatz."""
        try:
            from jarvis.skills.marketplace import SkillMarketplace
            mp = SkillMarketplace()
            results = mp.search(
                query=q, category=category,
                verified_only=verified_only,
                sort_by=sort_by, max_results=max_results,
            )
            return {"results": [r.to_dict() for r in results], "count": len(results)}
        except Exception as exc:
            return {"error": str(exc)}

    @app.get("/api/v1/marketplace/categories", dependencies=deps)
    async def marketplace_categories() -> dict[str, Any]:
        """Alle Skill-Kategorien mit Counts."""
        try:
            from jarvis.skills.marketplace import SkillMarketplace
            return {"categories": [c.to_dict() for c in SkillMarketplace().categories()]}
        except Exception as exc:
            return {"error": str(exc)}

    @app.get("/api/v1/marketplace/featured", dependencies=deps)
    async def marketplace_featured(n: int = 10) -> dict[str, Any]:
        """Featured-Skills."""
        try:
            from jarvis.skills.marketplace import SkillMarketplace
            return {"featured": [s.to_dict() for s in SkillMarketplace().featured(n)]}
        except Exception as exc:
            return {"error": str(exc)}

    @app.get("/api/v1/marketplace/trending", dependencies=deps)
    async def marketplace_trending(window: str = "24h", n: int = 10) -> dict[str, Any]:
        """Trending-Skills."""
        try:
            from jarvis.skills.marketplace import SkillMarketplace
            return {"trending": [s.to_dict() for s in SkillMarketplace().trending(window, n)]}
        except Exception as exc:
            return {"error": str(exc)}

    @app.get("/api/v1/marketplace/stats", dependencies=deps)
    async def marketplace_stats() -> dict[str, Any]:
        """Marktplatz-Statistiken."""
        try:
            from jarvis.skills.marketplace import SkillMarketplace
            return SkillMarketplace().stats()
        except Exception as exc:
            return {"error": str(exc)}

    # -- Skill-Updater ----------------------------------------------------

    @app.get("/api/v1/updater/stats", dependencies=deps)
    async def updater_stats() -> dict[str, Any]:
        """Skill-Updater-Statistiken."""
        try:
            from jarvis.skills.updater import SkillUpdater
            return SkillUpdater().stats()
        except Exception as exc:
            return {"error": str(exc)}

    @app.get("/api/v1/updater/pending", dependencies=deps)
    async def updater_pending() -> dict[str, Any]:
        """Ausstehende Updates."""
        try:
            from jarvis.skills.updater import SkillUpdater
            u = SkillUpdater()
            return {"updates": [c.to_dict() for c in u.pending_updates()]}
        except Exception as exc:
            return {"error": str(exc)}

    @app.get("/api/v1/updater/recalls", dependencies=deps)
    async def updater_recalls() -> dict[str, Any]:
        """Aktive Security-Recalls."""
        try:
            from jarvis.skills.updater import SkillUpdater
            u = SkillUpdater()
            return {"recalls": [r.to_dict() for r in u.active_recalls()]}
        except Exception as exc:
            return {"error": str(exc)}

    @app.get("/api/v1/updater/history", dependencies=deps)
    async def updater_history(n: int = 20) -> dict[str, Any]:
        """Update-Historie."""
        try:
            from jarvis.skills.updater import SkillUpdater
            return {"history": SkillUpdater().update_history(n)}
        except Exception as exc:
            return {"error": str(exc)}

    # -- Commands ---------------------------------------------------------

    @app.get("/api/v1/commands/list", dependencies=deps)
    async def list_commands() -> dict[str, Any]:
        """Alle registrierten Slash-Commands."""
        try:
            from jarvis.channels.commands import CommandRegistry
            reg = CommandRegistry()
            return {"commands": [c.to_dict() for c in reg.list_commands()], "count": reg.command_count}
        except Exception as exc:
            return {"error": str(exc)}

    @app.get("/api/v1/commands/slack", dependencies=deps)
    async def commands_slack() -> dict[str, Any]:
        """Slack Slash-Command-Definitionen."""
        try:
            from jarvis.channels.commands import CommandRegistry
            return {"definitions": CommandRegistry().slack_definitions()}
        except Exception as exc:
            return {"error": str(exc)}

    @app.get("/api/v1/commands/discord", dependencies=deps)
    async def commands_discord() -> dict[str, Any]:
        """Discord Application-Command-Definitionen."""
        try:
            from jarvis.channels.commands import CommandRegistry
            return {"definitions": CommandRegistry().discord_definitions()}
        except Exception as exc:
            return {"error": str(exc)}

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
            return {"total_connectors": 0, "connectors": [], "scope_guard": {"policies": 0, "violations": 0}}
        return reg.stats()

    # -- Workflows --------------------------------------------------------

    @app.get("/api/v1/workflows/templates", dependencies=deps)
    async def workflow_templates() -> dict[str, Any]:
        """Alle verfügbaren Workflow-Templates."""
        lib = getattr(gateway, "_template_library", None)
        if lib is None:
            return {"templates": [], "count": 0}
        return {"templates": lib.list_all(), "count": lib.template_count}

    @app.get("/api/v1/workflows/templates/categories", dependencies=deps)
    async def workflow_categories() -> dict[str, Any]:
        """Workflow-Kategorien."""
        lib = getattr(gateway, "_template_library", None)
        if lib is None:
            return {"categories": []}
        return {"categories": lib.categories()}

    @app.post("/api/v1/workflows/start", dependencies=deps)
    async def workflow_start(request: Request) -> dict[str, Any]:
        """Workflow-Instanz starten."""
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
            return {"error": str(exc)}

    @app.get("/api/v1/workflows/instances", dependencies=deps)
    async def workflow_instances() -> dict[str, Any]:
        """Laufende Workflow-Instanzen."""
        engine = getattr(gateway, "_workflow_engine", None)
        if engine is None:
            return {"instances": [], "count": 0}
        active = engine.active_instances()
        return {"instances": [i.to_dict() for i in active], "count": len(active)}

    @app.get("/api/v1/workflows/stats", dependencies=deps)
    async def workflow_stats() -> dict[str, Any]:
        """Workflow-Engine Statistiken."""
        engine = getattr(gateway, "_workflow_engine", None)
        if engine is None:
            return {"total": 0, "running": 0, "completed": 0, "failed": 0}
        return engine.stats()

    # -- Models -----------------------------------------------------------

    @app.get("/api/v1/models/list", dependencies=deps)
    async def model_list() -> dict[str, Any]:
        """Alle registrierten ML-Modelle."""
        reg = getattr(gateway, "_model_registry", None)
        if reg is None:
            return {"models": [], "count": 0}
        return {"models": reg.list_all(), "count": reg.model_count}

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
        action: str = "", actor: str = "", severity: str = "", limit: int = 100,
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
            from jarvis.security.redteam import SecurityScanner, ScanPolicy

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
            return {"error": str(exc)}

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
            return {"error": str(exc)}

    @app.get("/api/v1/compliance/export/{fmt}", dependencies=deps)
    async def compliance_export(fmt: str) -> Any:
        """Compliance-Report exportieren (json/csv/markdown)."""
        try:
            from jarvis.audit.compliance import ComplianceFramework, ReportExporter
            fw = getattr(gateway, "_compliance_framework", None) or ComplianceFramework()
            fw.auto_assess(
                has_audit_log=True, has_decision_log=True,
                has_kill_switch=True, has_encryption=True,
                has_rbac=True, has_sandbox=True,
                has_approval_workflow=True, has_redteam=True,
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
                return PlainTextResponse(ReportExporter.to_markdown(report), media_type="text/markdown")
            return {"error": f"Unknown format: {fmt}. Use json/csv/markdown."}
        except Exception as exc:
            return {"error": str(exc)}

    @app.get("/api/v1/compliance/decisions", dependencies=deps)
    async def compliance_decisions() -> dict[str, Any]:
        """Decision-Log Übersicht."""
        decision_log = getattr(gateway, "_decision_log", None)
        if decision_log is None:
            return {"total_decisions": 0, "flagged_count": 0, "approval_rate": 0.0, "unique_agents": 0, "avg_confidence": 0.0}
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
            return {"error": str(exc)}

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
            return {"error": str(exc)}

    # -- AI Agent Security Framework (Phase 21) ---------------------------

    @app.get("/api/v1/framework/metrics", dependencies=deps)
    async def framework_metrics() -> dict[str, Any]:
        """Security-Metriken (MTTD, MTTR, etc.)."""
        metrics = getattr(gateway, "_security_metrics", None)
        if metrics is None:
            return {"mttd_seconds": 0, "mttr_seconds": 0, "resolution_rate": 100, "total_incidents": 0}
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
            return {"total_entities": 0, "avg_score": 0, "flagged_count": 0, "trust_distribution": {}}
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
            "links": [l.to_dict() for l in interop.federation.active_links()],
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
            return {"error": str(exc), "status": 500}

    @app.post("/api/v1/system/stop", dependencies=deps)
    async def ui_system_stop() -> dict[str, Any]:
        """Initiates graceful shutdown if gateway is available."""
        try:
            if gateway is not None and hasattr(gateway, "shutdown"):
                asyncio.create_task(gateway.shutdown())
                return {"status": "ok", "message": "Shutdown eingeleitet"}
            return {"status": "ok", "message": "Kein Gateway — nur Config-Server aktiv"}
        except Exception as exc:
            return {"error": str(exc), "status": 500}

    # -- 3.4: POST /agents/{name} ----------------------------------------

    @app.post("/api/v1/agents/{name}", dependencies=deps)
    async def ui_upsert_agent(name: str, request: Request) -> dict[str, Any]:
        """Creates or updates an agent profile in agents.yaml."""
        try:
            body = await request.json()
            data = _load_yaml(agents_path)
            agents = data.get("agents", [])
            if not isinstance(agents, list):
                agents = []
            # Upsert by name
            found = False
            for i, a in enumerate(agents):
                if isinstance(a, dict) and a.get("name") == name:
                    agents[i] = body
                    found = True
                    break
            if not found:
                body["name"] = name
                agents.append(body)
            data["agents"] = agents
            _save_yaml(agents_path, data)
            return {"status": "ok", "agent": name}
        except Exception as exc:
            return {"error": str(exc), "status": 400}

    # -- 3.5: POST /bindings/{name} --------------------------------------

    @app.post("/api/v1/bindings/{name}", dependencies=deps)
    async def ui_upsert_binding(name: str, request: Request) -> dict[str, Any]:
        """Creates or updates a binding rule in bindings.yaml."""
        try:
            body = await request.json()
            data = _load_yaml(bindings_path)
            bindings = data.get("bindings", [])
            if not isinstance(bindings, list):
                bindings = []
            found = False
            for i, b in enumerate(bindings):
                if isinstance(b, dict) and b.get("name") == name:
                    bindings[i] = body
                    found = True
                    break
            if not found:
                body["name"] = name
                bindings.append(body)
            data["bindings"] = bindings
            _save_yaml(bindings_path, data)
            return {"status": "ok", "binding": name}
        except Exception as exc:
            return {"error": str(exc), "status": 400}

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

        # plannerSystem (fallback to Python constant if file missing or empty)
        try:
            sys_path = prompts_dir / "SYSTEM_PROMPT.txt"
            content = sys_path.read_text(encoding="utf-8").strip() if sys_path.exists() else ""
            if not content:
                from jarvis.core.planner import SYSTEM_PROMPT
                content = SYSTEM_PROMPT
            result["plannerSystem"] = content
        except Exception:
            result["plannerSystem"] = ""

        # replanPrompt
        try:
            replan_path = prompts_dir / "REPLAN_PROMPT.txt"
            content = replan_path.read_text(encoding="utf-8").strip() if replan_path.exists() else ""
            if not content:
                from jarvis.core.planner import REPLAN_PROMPT
                content = REPLAN_PROMPT
            result["replanPrompt"] = content
        except Exception:
            result["replanPrompt"] = ""

        # escalationPrompt
        try:
            esc_path = prompts_dir / "ESCALATION_PROMPT.txt"
            content = esc_path.read_text(encoding="utf-8").strip() if esc_path.exists() else ""
            if not content:
                from jarvis.core.planner import ESCALATION_PROMPT
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
                (prompts_dir / "SYSTEM_PROMPT.txt").write_text(body["plannerSystem"], encoding="utf-8")
                written.append("plannerSystem")

            if "replanPrompt" in body:
                (prompts_dir / "REPLAN_PROMPT.txt").write_text(body["replanPrompt"], encoding="utf-8")
                written.append("replanPrompt")

            if "escalationPrompt" in body:
                (prompts_dir / "ESCALATION_PROMPT.txt").write_text(body["escalationPrompt"], encoding="utf-8")
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

            return {"status": "ok", "written": written}
        except Exception as exc:
            return {"error": str(exc), "status": 400}

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
            return {"jobs": [], "error": str(exc)}

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
            return {"error": str(exc), "status": 400}

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
                servers_dict = {s.get("name", f"server_{i}"): s for i, s in enumerate(servers_raw) if isinstance(s, dict)}
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
                "auth_token": sm.get("auth_token", ""),
                "expose_tools": sm.get("expose_tools", True),
                "expose_resources": sm.get("expose_resources", True),
                "expose_prompts": sm.get("expose_prompts", False),
                "enable_sampling": sm.get("enable_sampling", False),
                "external_servers": servers_dict,
            }
            return result
        except Exception as exc:
            return {"mode": "disabled", "external_servers": {}, "error": str(exc)}

    @app.put("/api/v1/mcp-servers", dependencies=deps)
    async def ui_put_mcp_servers(request: Request) -> dict[str, Any]:
        """Writes MCP config from flat UI format, preserving a2a and built_in_tools."""
        try:
            body = await request.json()
            mcp_path = config_manager.config.mcp_config_file
            data = _load_yaml(mcp_path)
            # Reconstruct server_mode from flat fields
            sm_keys = ("mode", "http_host", "http_port", "server_name",
                       "require_auth", "auth_token", "expose_tools",
                       "expose_resources", "expose_prompts", "enable_sampling")
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
            return {"error": str(exc), "status": 400}

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
                **{k: v for k, v in a2a.items() if k not in ("enabled", "host", "port", "agent_name")},
            }
        except Exception as exc:
            return {"enabled": False, "error": str(exc)}

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
            return {"error": str(exc), "status": 400}
