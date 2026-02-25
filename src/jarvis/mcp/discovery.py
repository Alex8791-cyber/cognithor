"""MCP Discovery: Agent Card und Service-Discovery.

Implementiert das /.well-known/agent.json-Muster für Agent-Discovery,
kompatibel mit dem MCP-Ökosystem und vorbereitet auf A2A-Integration.

Features:
  - AgentCard: JSON-Beschreibung der Jarvis-Instanz
  - Capability-Announcements
  - Health-Endpoint
  - Versionierung und Kompatibilitäts-Flags

Die Agent Card wird als REST-Endpoint exponiert und kann von
anderen Agenten, MCP-Clients und Discovery-Services abgefragt werden.

Bibel-Referenz: §5.5.4 (MCP Discovery)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from jarvis.utils.logging import get_logger

log = get_logger(__name__)


# ============================================================================
# Agent Card (MCP + A2A kompatibel)
# ============================================================================


@dataclass
class AgentSkill:
    """Eine einzelne Fähigkeit des Agenten."""

    id: str
    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
        }
        if self.tags:
            result["tags"] = self.tags
        if self.examples:
            result["examples"] = self.examples
        return result


@dataclass
class AgentAuth:
    """Authentifizierungs-Schema."""

    schemes: list[str] = field(default_factory=lambda: ["bearer"])
    credentials: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"schemes": self.schemes}
        if self.credentials:
            result["credentials"] = self.credentials
        return result


@dataclass
class AgentCard:
    """Vollständige Agent Card für Discovery.

    Kompatibel mit:
    - MCP Server Discovery
    - A2A Agent Cards (/.well-known/agent.json)
    - Jarvis Interop Protocol (JAIP)
    """

    name: str = "Jarvis"
    description: str = "Lokaler KI-Assistent mit Multi-Agent-Fähigkeiten"
    version: str = "15.0.0"
    protocol_version: str = "2025-11-25"
    url: str = ""
    endpoint: str = ""  # MCP-Endpoint URL
    owner: str = ""
    icon_url: str = ""

    # Capabilities
    capabilities: list[str] = field(default_factory=lambda: [
        "tools",
        "resources",
        "prompts",
        "sampling",
    ])

    # Skills
    skills: list[AgentSkill] = field(default_factory=list)

    # Auth
    authentication: AgentAuth = field(default_factory=AgentAuth)

    # Supported modalities
    default_input_modes: list[str] = field(
        default_factory=lambda: ["text/plain", "application/json"]
    )
    default_output_modes: list[str] = field(
        default_factory=lambda: ["text/plain", "application/json"]
    )

    # Metadata
    tags: list[str] = field(default_factory=lambda: [
        "local-first",
        "privacy",
        "german",
        "insurance",
        "multi-agent",
    ])
    languages: list[str] = field(default_factory=lambda: ["de", "en"])
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert in JSON-serialisierbares Dict."""
        card: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "protocolVersion": self.protocol_version,
        }

        if self.url:
            card["url"] = self.url
        if self.endpoint:
            card["endpoint"] = self.endpoint
        if self.owner:
            card["owner"] = self.owner
        if self.icon_url:
            card["iconUrl"] = self.icon_url

        card["capabilities"] = {cap: True for cap in self.capabilities}

        if self.skills:
            card["skills"] = [s.to_dict() for s in self.skills]

        card["authentication"] = self.authentication.to_dict()

        card["defaultInputModes"] = self.default_input_modes
        card["defaultOutputModes"] = self.default_output_modes

        card["tags"] = self.tags
        card["languages"] = self.languages

        if self.created_at:
            card["createdAt"] = self.created_at
        if self.updated_at:
            card["updatedAt"] = self.updated_at

        return card


# ============================================================================
# Discovery Manager
# ============================================================================


class DiscoveryManager:
    """Verwaltet die Agent Card und Discovery-Endpoints.

    Erstellt und aktualisiert die Agent Card basierend auf den
    tatsächlich registrierten Tools, Resources und Prompts.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 3001,
        owner: str = "",
    ) -> None:
        self._host = host
        self._port = port
        self._owner = owner
        self._card: AgentCard | None = None
        self._boot_time = time.time()
        self._health_checks = 0

    def build_card(
        self,
        tool_names: list[str] | None = None,
        resource_count: int = 0,
        prompt_count: int = 0,
        server_mode: str = "disabled",
    ) -> AgentCard:
        """Erstellt die Agent Card basierend auf aktuellem Stand.

        Args:
            tool_names: Liste der verfügbaren Tool-Namen
            resource_count: Anzahl registrierter Resources
            prompt_count: Anzahl registrierter Prompts
            server_mode: MCP-Server-Modus

        Returns:
            Konfigurierte AgentCard
        """
        tool_names = tool_names or []

        # Endpoint bestimmen
        endpoint = ""
        if server_mode in ("http", "both"):
            endpoint = f"http://{self._host}:{self._port}/mcp"

        # Capabilities bestimmen
        caps = []
        if tool_names:
            caps.append("tools")
        if resource_count > 0:
            caps.append("resources")
        if prompt_count > 0:
            caps.append("prompts")

        # Skills aus Tool-Namen ableiten
        skills = self._derive_skills(tool_names)

        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()

        self._card = AgentCard(
            name="Jarvis",
            description=(
                "Lokaler KI-Agent mit 5-Tier-Memory, Browser-Automatisierung, "
                "Sandbox-Isolation und EU-AI-Act-Compliance. "
                "Spezialisiert auf Versicherungsberatung und Workflow-Automatisierung."
            ),
            version="15.0.0",
            endpoint=endpoint,
            owner=self._owner,
            capabilities=caps,
            skills=skills,
            created_at=now,
            updated_at=now,
        )

        log.info(
            "agent_card_built",
            tools=len(tool_names),
            resources=resource_count,
            prompts=prompt_count,
            skills=len(skills),
        )

        return self._card

    def get_card(self) -> dict[str, Any]:
        """Gibt die aktuelle Agent Card als Dict zurück."""
        if self._card is None:
            self.build_card()
        return self._card.to_dict()  # type: ignore[union-attr]

    def health(self) -> dict[str, Any]:
        """Health-Check-Endpoint."""
        self._health_checks += 1
        uptime = time.time() - self._boot_time
        return {
            "status": "healthy",
            "uptime_seconds": round(uptime, 1),
            "health_checks": self._health_checks,
            "agent": "jarvis",
            "version": "15.0.0",
            "mcp_protocol": "2025-11-25",
        }

    # ── Skill-Ableitung ──────────────────────────────────────────

    def _derive_skills(self, tool_names: list[str]) -> list[AgentSkill]:
        """Leitet Skills aus den Tool-Namen ab."""
        skills: list[AgentSkill] = []

        # Gruppen-Mapping
        skill_groups: dict[str, dict[str, Any]] = {
            "file_management": {
                "name": "Dateiverwaltung",
                "description": "Lesen, Schreiben und Bearbeiten von Dateien",
                "tags": ["filesystem", "io"],
                "tools": ["read_file", "write_file", "edit_file", "list_directory"],
            },
            "web_research": {
                "name": "Web-Recherche",
                "description": "Websuche und Seiteninhalt-Extraktion",
                "tags": ["web", "search", "research"],
                "tools": ["web_search", "web_fetch"],
            },
            "browser_automation": {
                "name": "Browser-Automatisierung",
                "description": "Headless-Browser: Navigation, Screenshots, Formulare",
                "tags": ["browser", "automation", "playwright"],
                "tools": ["browse_url", "browse_screenshot", "browse_click", "browse_fill", "browse_execute_js"],
            },
            "memory_system": {
                "name": "Gedächtnissystem",
                "description": "5-Tier-Memory mit Hybrid-Suche, Wissens-Graph und Episodischem Speicher",
                "tags": ["memory", "knowledge", "graph"],
                "tools": ["search_memory", "save_to_memory", "get_entity", "add_entity"],
            },
            "code_execution": {
                "name": "Code-Ausführung",
                "description": "Sandbox-isolierte Shell-Befehle (bwrap/firejail)",
                "tags": ["shell", "code", "sandbox"],
                "tools": ["exec_command"],
            },
            "media_processing": {
                "name": "Medienverarbeitung",
                "description": "Audio-Transkription, Bild-Analyse, Dokumenten-Extraktion, TTS",
                "tags": ["media", "audio", "image", "ocr"],
                "tools": ["media_transcribe_audio", "media_analyze_image", "media_extract_text", "media_tts"],
            },
        }

        tool_set = set(tool_names)

        for skill_id, group in skill_groups.items():
            matching_tools = [t for t in group["tools"] if t in tool_set]
            if matching_tools:
                skills.append(AgentSkill(
                    id=skill_id,
                    name=group["name"],
                    description=group["description"],
                    tags=group["tags"],
                    examples=matching_tools[:3],
                ))

        return skills

    def stats(self) -> dict[str, Any]:
        """Statistiken des Discovery-Managers."""
        return {
            "card_built": self._card is not None,
            "skills": len(self._card.skills) if self._card else 0,
            "capabilities": self._card.capabilities if self._card else [],
            "health_checks": self._health_checks,
            "endpoint": self._card.endpoint if self._card else "",
        }
