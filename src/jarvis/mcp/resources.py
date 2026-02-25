"""MCP Resources: Exponiert Jarvis-Daten als MCP-Resources.

Resources sind read-only Datenquellen die externe MCP-Clients
abfragen können -- im Gegensatz zu Tools, die Aktionen ausführen.

Registrierte Resources:
  - jarvis://memory/core         → Core Memory (CORE.md)
  - jarvis://memory/episodes     → Letzte episodische Einträge
  - jarvis://memory/stats        → Memory-Statistiken
  - jarvis://memory/entity/{id}  → Einzelne Entität (Template)
  - jarvis://config/status       → System-Status
  - jarvis://config/tools        → Verfügbare Tools
  - jarvis://config/agents       → Registrierte Agenten
  - jarvis://workspace/files     → Workspace-Verzeichnisbaum

OPTIONAL: Wird nur registriert wenn MCP-Server-Modus aktiviert ist.

Bibel-Referenz: §5.5.2 (MCP Resources)
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from jarvis.mcp.server import (
    JarvisMCPServer,
    MCPResource,
    MCPResourceTemplate,
)
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.config import JarvisConfig
    from jarvis.memory.manager import MemoryManager

log = get_logger(__name__)


class JarvisResourceProvider:
    """Stellt Jarvis-Daten als MCP-Resources bereit.

    Verbindet sich mit dem Memory-Manager und der Config,
    um Daten über das MCP-Resource-Protokoll verfügbar zu machen.
    """

    def __init__(
        self,
        config: JarvisConfig | None = None,
        memory: MemoryManager | None = None,
    ) -> None:
        self._config = config
        self._memory = memory
        self._registered_count = 0
        self._boot_time = time.time()

    def register_all(self, server: JarvisMCPServer) -> int:
        """Registriert alle Resources beim MCP-Server.

        Returns:
            Anzahl registrierter Resources.
        """
        count = 0

        # ── Memory Resources ────────────────────────────────────
        server.register_resource(MCPResource(
            uri="jarvis://memory/core",
            name="Core Memory",
            description="Langfristige Kern-Erinnerungen und Persönlichkeit von Jarvis",
            mime_type="text/markdown",
            handler=self._read_core_memory,
        ))
        count += 1

        server.register_resource(MCPResource(
            uri="jarvis://memory/episodes",
            name="Recent Episodes",
            description="Die letzten episodischen Einträge (Tageslog)",
            mime_type="application/json",
            handler=self._read_episodes,
        ))
        count += 1

        server.register_resource(MCPResource(
            uri="jarvis://memory/stats",
            name="Memory Statistics",
            description="Gesamtstatistiken des 5-Tier Memory-Systems",
            mime_type="application/json",
            handler=self._read_memory_stats,
        ))
        count += 1

        # Template für einzelne Entitäten
        server.register_resource_template(MCPResourceTemplate(
            uri_template="jarvis://memory/entity/{entity_id}",
            name="Memory Entity",
            description="Eine einzelne Entität aus dem Wissens-Graphen",
            mime_type="application/json",
            handler=self._read_entity,
        ))
        count += 1

        # ── Config / Status Resources ───────────────────────────
        server.register_resource(MCPResource(
            uri="jarvis://config/status",
            name="System Status",
            description="Aktueller Systemstatus von Jarvis",
            mime_type="application/json",
            handler=self._read_status,
        ))
        count += 1

        server.register_resource(MCPResource(
            uri="jarvis://config/tools",
            name="Available Tools",
            description="Liste aller verfügbaren MCP-Tools",
            mime_type="application/json",
            handler=lambda **_: self._read_tools(server),
        ))
        count += 1

        server.register_resource(MCPResource(
            uri="jarvis://config/capabilities",
            name="Capabilities",
            description="Fähigkeiten und Konfiguration dieser Jarvis-Instanz",
            mime_type="application/json",
            handler=self._read_capabilities,
        ))
        count += 1

        # ── Workspace Resources ─────────────────────────────────
        server.register_resource(MCPResource(
            uri="jarvis://workspace/files",
            name="Workspace Files",
            description="Verzeichnisbaum des Jarvis-Workspace",
            mime_type="application/json",
            handler=self._read_workspace_files,
        ))
        count += 1

        self._registered_count = count
        log.info("mcp_resources_registered", count=count)
        return count

    # ── Handler ──────────────────────────────────────────────────

    def _read_core_memory(self, **kwargs: Any) -> str:
        """Liest die Core Memory."""
        if self._memory is None:
            return "# Core Memory\n\n(Memory-Manager nicht initialisiert)"

        try:
            core = self._memory.get_core_memory()
            if hasattr(core, "content"):
                return core.content
            return str(core)
        except Exception as exc:
            return f"# Fehler beim Lesen der Core Memory\n\n{exc}"

    def _read_episodes(self, **kwargs: Any) -> str:
        """Liest die letzten episodischen Einträge."""
        if self._memory is None:
            return json.dumps({"episodes": [], "error": "Memory nicht initialisiert"})

        try:
            episodes = self._memory.get_recent_episodes(days=7)
            if isinstance(episodes, list):
                items = []
                for ep in episodes[:20]:
                    if hasattr(ep, "to_dict"):
                        items.append(ep.to_dict())
                    elif isinstance(ep, dict):
                        items.append(ep)
                    else:
                        items.append({"text": str(ep)})
                return json.dumps({"episodes": items, "count": len(items)}, ensure_ascii=False)
            return json.dumps({"episodes": str(episodes)})
        except Exception as exc:
            return json.dumps({"episodes": [], "error": str(exc)})

    def _read_memory_stats(self, **kwargs: Any) -> str:
        """Liest Memory-Statistiken."""
        if self._memory is None:
            return json.dumps({"error": "Memory nicht initialisiert"})

        try:
            stats = self._memory.stats()
            if isinstance(stats, dict):
                return json.dumps(stats, ensure_ascii=False, default=str)
            return json.dumps({"stats": str(stats)})
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    def _read_entity(self, **kwargs: Any) -> str:
        """Liest eine einzelne Entität."""
        uri = kwargs.get("uri", "")
        entity_id = uri.rsplit("/", 1)[-1] if "/" in uri else ""

        if not entity_id or self._memory is None:
            return json.dumps({"error": "Entität nicht gefunden"})

        try:
            entity = self._memory.get_entity(entity_id)
            if entity and hasattr(entity, "to_dict"):
                return json.dumps(entity.to_dict(), ensure_ascii=False, default=str)
            elif entity:
                return json.dumps({"entity": str(entity)})
            return json.dumps({"error": f"Entität '{entity_id}' nicht gefunden"})
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    def _read_status(self, **kwargs: Any) -> str:
        """Liest den Systemstatus."""
        uptime = time.time() - self._boot_time
        status = {
            "status": "running",
            "uptime_seconds": round(uptime, 1),
            "mcp_resources": self._registered_count,
            "memory_available": self._memory is not None,
            "config_loaded": self._config is not None,
        }

        if self._config:
            status["jarvis_home"] = str(self._config.jarvis_home)
            status["model"] = getattr(self._config, "default_model", "unknown")

        return json.dumps(status, ensure_ascii=False, default=str)

    def _read_tools(self, server: JarvisMCPServer) -> str:
        """Liest die Tool-Liste vom Server."""
        tool_list = []
        for name, tool in server._tools.items():
            tool_list.append({
                "name": name,
                "description": tool.description,
                "annotations": tool.annotations,
            })
        return json.dumps({"tools": tool_list, "count": len(tool_list)}, ensure_ascii=False)

    def _read_capabilities(self, **kwargs: Any) -> str:
        """Liest die Capabilities der Jarvis-Instanz."""
        caps = {
            "agent_name": "Jarvis",
            "version": "15.0.0",
            "capabilities": [
                "task_execution",
                "web_search",
                "file_management",
                "code_execution",
                "memory_system",
                "browser_automation",
                "media_processing",
                "multi_agent",
            ],
            "languages": ["de", "en"],
            "mcp_protocol_version": "2025-11-25",
            "security_features": [
                "sandbox_isolation",
                "agent_vault",
                "session_firewall",
                "red_team_ci",
                "eu_ai_act_compliance",
            ],
        }
        return json.dumps(caps, ensure_ascii=False)

    def _read_workspace_files(self, **kwargs: Any) -> str:
        """Liest den Workspace-Verzeichnisbaum."""
        if self._config is None:
            return json.dumps({"error": "Config nicht verfügbar"})

        import os
        workspace = self._config.jarvis_home / "workspace"
        if not workspace.exists():
            return json.dumps({"files": [], "workspace": str(workspace)})

        files = []
        try:
            for root, dirs, filenames in os.walk(workspace):
                # Max 2 Ebenen tief
                depth = root.replace(str(workspace), "").count(os.sep)
                if depth > 2:
                    dirs.clear()
                    continue
                for fname in filenames[:50]:
                    rel_path = os.path.relpath(os.path.join(root, fname), workspace)
                    files.append(rel_path)
                if len(files) > 200:
                    break
        except Exception:
            pass

        return json.dumps({
            "workspace": str(workspace),
            "files": files[:200],
            "count": len(files),
        }, ensure_ascii=False)

    def stats(self) -> dict[str, Any]:
        """Statistiken des Resource-Providers."""
        return {
            "registered_resources": self._registered_count,
            "memory_available": self._memory is not None,
            "config_available": self._config is not None,
        }
