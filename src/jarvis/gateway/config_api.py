"""Config-API: REST-Endpoints für Jarvis-Konfiguration.

Stellt CRUD-Endpoints bereit für:
  - Heartbeat-Konfiguration
  - Agent-Profile
  - Binding-Regeln
  - Sandbox-Einstellungen
  - Konfigurations-Presets

Wird als FastAPI-Router in die WebUI gemountet.
Die Konfiguration wird in config.yaml persistiert.

Bibel-Referenz: §12 (Konfigurationsmanagement)
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from jarvis.utils.logging import get_logger

log = get_logger(__name__)


# ============================================================================
# API-Models (Pydantic — für Request/Response-Validierung)
# ============================================================================


class HeartbeatUpdate(BaseModel):
    """Heartbeat-Konfiguration aktualisieren."""

    enabled: bool | None = None
    interval_minutes: int | None = Field(default=None, ge=1, le=1440)
    checklist_file: str | None = None
    channel: str | None = None
    model: str | None = None


class AgentProfileDTO(BaseModel):
    """Agent-Profil anlegen/aktualisieren."""

    name: str
    display_name: str = ""
    description: str = ""
    system_prompt: str = ""
    language: str = "de"
    trigger_patterns: list[str] = Field(default_factory=list)
    trigger_keywords: list[str] = Field(default_factory=list)
    priority: int = 0
    allowed_tools: list[str] | None = None
    blocked_tools: list[str] = Field(default_factory=list)
    preferred_model: str = ""
    temperature: float | None = None
    workspace_subdir: str = ""
    shared_workspace: bool = False
    sandbox_network: str = "allow"
    sandbox_max_memory_mb: int = 512
    sandbox_max_processes: int = 64
    sandbox_timeout: int = 30
    can_delegate_to: list[str] = Field(default_factory=list)
    max_delegation_depth: int = 2
    credential_scope: str = ""
    credential_mappings: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True


class BindingRuleDTO(BaseModel):
    """Binding-Regel anlegen/aktualisieren."""

    name: str
    target_agent: str
    priority: int = 100
    description: str = ""
    channels: list[str] | None = None
    user_ids: list[str] | None = None
    command_prefixes: list[str] | None = None
    message_patterns: list[str] | None = None
    metadata_conditions: dict[str, str] | None = None
    negate: bool = False
    stop_processing: bool = True
    enabled: bool = True


class SandboxUpdate(BaseModel):
    """Sandbox-Einstellungen aktualisieren."""

    enabled: bool | None = None
    network: str | None = None  # "allow" | "block"
    max_memory_mb: int | None = Field(default=None, ge=64, le=8192)
    max_processes: int | None = Field(default=None, ge=1, le=512)
    timeout_seconds: int | None = Field(default=None, ge=5, le=600)
    allowed_paths: list[str] | None = None
    blocked_paths: list[str] | None = None


class PresetInfo(BaseModel):
    """Konfigurations-Preset."""

    name: str
    description: str
    agents: list[str] = Field(default_factory=list)
    heartbeat_enabled: bool = False


class ConfigOverview(BaseModel):
    """Gesamtübersicht der Konfiguration."""

    version: str
    owner_name: str
    llm_backend: str
    heartbeat_enabled: bool
    heartbeat_interval: int
    agent_count: int
    binding_count: int
    sandbox_enabled: bool
    channels_active: list[str]


# ============================================================================
# Presets
# ============================================================================

PRESETS: dict[str, dict[str, Any]] = {
    "office": {
        "name": "Büro-Assistent",
        "description": "Tägliche Briefings, E-Mail-Übersicht, Kalender-Check",
        "heartbeat": {"enabled": True, "interval_minutes": 30, "channel": "telegram"},
        "agents": [
            {
                "name": "jarvis",
                "display_name": "Jarvis",
                "description": "Allgemeiner Assistent",
                "trigger_keywords": [],
                "priority": 0,
            },
        ],
        "bindings": [
            {
                "name": "slash_briefing",
                "target_agent": "jarvis",
                "command_prefixes": ["/briefing", "/morgen"],
                "description": "Morning-Briefing per Slash-Command",
            },
        ],
    },
    "developer": {
        "name": "Developer",
        "description": "Code-Assistent mit Research-Agent und Sandbox",
        "heartbeat": {"enabled": False},
        "agents": [
            {
                "name": "jarvis",
                "display_name": "Jarvis",
                "description": "Allgemeiner Assistent",
                "priority": 0,
            },
            {
                "name": "coder",
                "display_name": "Coder",
                "description": "Code-Spezialist",
                "trigger_keywords": ["code", "python", "debug", "function"],
                "priority": 10,
                "sandbox_network": "block",
                "credential_scope": "coder",
            },
            {
                "name": "researcher",
                "display_name": "Researcher",
                "description": "Web-Recherche und Analyse",
                "trigger_keywords": ["recherche", "suche", "finde"],
                "priority": 5,
            },
        ],
        "bindings": [
            {
                "name": "slash_code",
                "target_agent": "coder",
                "command_prefixes": ["/code", "/debug", "/fix"],
            },
            {
                "name": "slash_research",
                "target_agent": "researcher",
                "command_prefixes": ["/research", "/suche"],
            },
        ],
    },
    "family": {
        "name": "Familien-Planer",
        "description": "Termine, Einkaufslisten, Familien-Koordination",
        "heartbeat": {"enabled": True, "interval_minutes": 60, "channel": "telegram"},
        "agents": [
            {
                "name": "jarvis",
                "display_name": "Jarvis",
                "description": "Familien-Assistent",
                "trigger_keywords": [],
                "priority": 0,
            },
        ],
        "bindings": [
            {
                "name": "slash_einkauf",
                "target_agent": "jarvis",
                "command_prefixes": ["/einkauf", "/liste"],
                "description": "Einkaufsliste verwalten",
            },
        ],
    },
}


# ============================================================================
# ConfigManager: Zentrale Config-Verwaltung
# ============================================================================


class ConfigManager:
    """Verwaltet Jarvis-Konfiguration mit CRUD-Operationen.

    Hält eine In-Memory-Kopie der Konfiguration und kann
    Änderungen in config.yaml persistieren.
    Thread-safe über einfache Methoden-Granularität.
    """

    def __init__(self, config: Any) -> None:
        """Initialisiert den ConfigManager.

        Args:
            config: JarvisConfig-Instanz (oder kompatibles Objekt).
        """
        self._config = config
        self._agents: dict[str, dict[str, Any]] = {}
        self._bindings: list[dict[str, Any]] = []
        self._load_from_config()

    def _load_from_config(self) -> None:
        """Lädt Agent-Profile und Bindings aus der Konfiguration."""
        # Agents aus Config extrahieren
        if hasattr(self._config, "agents"):
            for agent in self._config.agents:
                name = getattr(agent, "name", str(agent))
                self._agents[name] = self._agent_to_dict(agent)

    @staticmethod
    def _agent_to_dict(agent: Any) -> dict[str, Any]:
        """Konvertiert AgentProfile → dict."""
        if isinstance(agent, dict):
            return agent
        return {
            f: getattr(agent, f, None)
            for f in AgentProfileDTO.model_fields
            if hasattr(agent, f)
        }

    # ------------------------------------------------------------------
    # Übersicht
    # ------------------------------------------------------------------

    def get_overview(self) -> ConfigOverview:
        """Gibt eine Konfigurationsübersicht zurück."""
        return ConfigOverview(
            version=getattr(self._config, "version", "0.0.0"),
            owner_name=getattr(self._config, "owner_name", "User"),
            llm_backend=getattr(self._config, "llm_backend_type", "ollama"),
            heartbeat_enabled=self.get_heartbeat().get("enabled", False),
            heartbeat_interval=self.get_heartbeat().get("interval_minutes", 30),
            agent_count=len(self._agents),
            binding_count=len(self._bindings),
            sandbox_enabled=getattr(
                getattr(self._config, "sandbox", None), "enabled", True
            ),
            channels_active=self._get_active_channels(),
        )

    def _get_active_channels(self) -> list[str]:
        """Ermittelt aktive Channels."""
        channels = []
        ch_config = getattr(self._config, "channels", None)
        if ch_config:
            for attr in ["cli_enabled", "telegram_enabled", "webui_enabled",
                         "slack_enabled", "discord_enabled"]:
                if getattr(ch_config, attr, False):
                    channels.append(attr.replace("_enabled", ""))
        return channels or ["cli"]

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    def get_heartbeat(self) -> dict[str, Any]:
        """Liest Heartbeat-Konfiguration."""
        hb = getattr(self._config, "heartbeat", None)
        if not hb:
            return {"enabled": False, "interval_minutes": 30, "channel": "cli"}
        return {
            "enabled": hb.enabled,
            "interval_minutes": hb.interval_minutes,
            "checklist_file": hb.checklist_file,
            "channel": hb.channel,
            "model": hb.model,
        }

    def update_heartbeat(self, update: HeartbeatUpdate) -> dict[str, Any]:
        """Aktualisiert Heartbeat-Konfiguration."""
        hb = getattr(self._config, "heartbeat", None)
        if not hb:
            return self.get_heartbeat()

        for field_name, value in update.model_dump(exclude_none=True).items():
            if hasattr(hb, field_name):
                setattr(hb, field_name, value)

        log.info("config_heartbeat_updated", changes=update.model_dump(exclude_none=True))
        return self.get_heartbeat()

    # ------------------------------------------------------------------
    # Agents
    # ------------------------------------------------------------------

    def list_agents(self) -> list[dict[str, Any]]:
        """Listet alle Agent-Profile."""
        return list(self._agents.values())

    def get_agent(self, name: str) -> dict[str, Any] | None:
        """Liest ein Agent-Profil."""
        return self._agents.get(name)

    def upsert_agent(self, dto: AgentProfileDTO) -> dict[str, Any]:
        """Erstellt oder aktualisiert ein Agent-Profil."""
        data = dto.model_dump()
        self._agents[dto.name] = data
        log.info("config_agent_upserted", name=dto.name)
        return data

    def delete_agent(self, name: str) -> bool:
        """Löscht ein Agent-Profil."""
        if name == "jarvis":
            return False  # Default-Agent kann nicht gelöscht werden
        if name in self._agents:
            del self._agents[name]
            log.info("config_agent_deleted", name=name)
            return True
        return False

    # ------------------------------------------------------------------
    # Bindings
    # ------------------------------------------------------------------

    def list_bindings(self) -> list[dict[str, Any]]:
        """Listet alle Binding-Regeln."""
        return list(self._bindings)

    def get_binding(self, name: str) -> dict[str, Any] | None:
        """Liest eine Binding-Regel."""
        for b in self._bindings:
            if b.get("name") == name:
                return b
        return None

    def upsert_binding(self, dto: BindingRuleDTO) -> dict[str, Any]:
        """Erstellt oder aktualisiert eine Binding-Regel."""
        data = dto.model_dump()
        # Ersetze existierendes Binding oder füge neues hinzu
        for i, b in enumerate(self._bindings):
            if b.get("name") == dto.name:
                self._bindings[i] = data
                log.info("config_binding_updated", name=dto.name)
                return data
        self._bindings.append(data)
        log.info("config_binding_created", name=dto.name)
        return data

    def delete_binding(self, name: str) -> bool:
        """Löscht eine Binding-Regel."""
        for i, b in enumerate(self._bindings):
            if b.get("name") == name:
                self._bindings.pop(i)
                log.info("config_binding_deleted", name=name)
                return True
        return False

    # ------------------------------------------------------------------
    # Sandbox
    # ------------------------------------------------------------------

    def get_sandbox(self) -> dict[str, Any]:
        """Liest Sandbox-Konfiguration."""
        sb = getattr(self._config, "sandbox", None)
        if not sb:
            return {"enabled": True, "network": "allow", "max_memory_mb": 512}
        result: dict[str, Any] = {}
        for attr in ["enabled", "network", "max_memory_mb", "max_processes",
                      "timeout_seconds", "allowed_paths", "blocked_paths"]:
            if hasattr(sb, attr):
                result[attr] = getattr(sb, attr)
        return result

    def update_sandbox(self, update: SandboxUpdate) -> dict[str, Any]:
        """Aktualisiert Sandbox-Konfiguration."""
        sb = getattr(self._config, "sandbox", None)
        if not sb:
            return self.get_sandbox()

        for field_name, value in update.model_dump(exclude_none=True).items():
            if hasattr(sb, field_name):
                setattr(sb, field_name, value)

        log.info("config_sandbox_updated", changes=update.model_dump(exclude_none=True))
        return self.get_sandbox()

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    def list_presets(self) -> list[PresetInfo]:
        """Listet verfügbare Presets."""
        return [
            PresetInfo(
                name=key,
                description=preset["description"],
                agents=[a["name"] for a in preset.get("agents", [])],
                heartbeat_enabled=preset.get("heartbeat", {}).get("enabled", False),
            )
            for key, preset in PRESETS.items()
        ]

    def apply_preset(self, preset_name: str) -> dict[str, Any]:
        """Wendet ein Preset an.

        Returns:
            Dict mit angewandten Änderungen oder Fehler.
        """
        preset = PRESETS.get(preset_name)
        if not preset:
            return {"error": f"Preset '{preset_name}' nicht gefunden"}

        changes: dict[str, Any] = {"preset": preset_name, "applied": []}

        # Heartbeat
        hb_data = preset.get("heartbeat", {})
        if hb_data:
            self.update_heartbeat(HeartbeatUpdate(**hb_data))
            changes["applied"].append("heartbeat")

        # Agents
        for agent_data in preset.get("agents", []):
            self.upsert_agent(AgentProfileDTO(**agent_data))
            changes["applied"].append(f"agent:{agent_data['name']}")

        # Bindings
        for binding_data in preset.get("bindings", []):
            self.upsert_binding(BindingRuleDTO(**binding_data))
            changes["applied"].append(f"binding:{binding_data['name']}")

        log.info("config_preset_applied", preset=preset_name, changes=changes)
        return changes

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def export_config(self) -> dict[str, Any]:
        """Exportiert die gesamte Konfiguration als Dict."""
        return {
            "heartbeat": self.get_heartbeat(),
            "agents": self.list_agents(),
            "bindings": self.list_bindings(),
            "sandbox": self.get_sandbox(),
        }
