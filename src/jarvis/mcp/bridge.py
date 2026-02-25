"""MCP Bridge: Verbindet bestehende Builtin-Tools mit dem MCP-Server.

Dieses Modul ist die zentrale Brücke zwischen dem bestehenden
register_builtin_handler()-System und dem neuen MCP-Server-Modus.

ARCHITEKTUR:
  - Ohne MCP-Server: Tools laufen wie bisher über register_builtin_handler()
  - Mit MCP-Server: Tools werden ZUSÄTZLICH über den MCP-Server exponiert
  - Der MCP-Server ist rein additiv -- er ersetzt nichts

Verantwortlich für:
  1. Bestehende Builtin-Handler in MCPToolDefs konvertieren
  2. Tool-Annotations hinzufügen (readOnly, destructive, etc.)
  3. Resources und Prompts beim Server registrieren
  4. Discovery/Agent-Card aufbauen
  5. HTTP-Endpoints bereitstellen (wenn HTTP-Modus aktiv)

Bibel-Referenz: §5.5.5 (MCP Bridge)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from jarvis.mcp.discovery import DiscoveryManager
from jarvis.mcp.prompts import JarvisPromptProvider
from jarvis.mcp.resources import JarvisResourceProvider
from jarvis.mcp.server import (
    JarvisMCPServer,
    MCPServerConfig,
    MCPServerMode,
    MCPToolDef,
    ToolAnnotationKey,
)
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.config import JarvisConfig
    from jarvis.mcp.client import JarvisMCPClient
    from jarvis.memory.manager import MemoryManager

log = get_logger(__name__)


# ============================================================================
# Tool Annotation Mappings
# ============================================================================

# Welche Tools sind read-only (ändern nichts am System)?
READ_ONLY_TOOLS = frozenset({
    "read_file",
    "list_directory",
    "web_search",
    "web_fetch",
    "search_memory",
    "get_entity",
    "get_core_memory",
    "get_recent_episodes",
    "search_procedures",
    "memory_stats",
    "browse_url",
    "browse_screenshot",
    "browse_page_info",
    "media_transcribe_audio",
    "media_analyze_image",
    "media_extract_text",
})

# Welche Tools sind destruktiv (können Daten löschen/überschreiben)?
DESTRUCTIVE_TOOLS = frozenset({
    "write_file",
    "edit_file",
    "exec_command",
    "browse_fill",
    "browse_click",
    "browse_execute_js",
})

# Welche Tools sind idempotent (mehrfach aufrufen = gleich)?
IDEMPOTENT_TOOLS = frozenset({
    "read_file",
    "list_directory",
    "web_search",
    "web_fetch",
    "search_memory",
    "get_entity",
    "get_core_memory",
    "get_recent_episodes",
    "memory_stats",
    "browse_url",
    "browse_screenshot",
})


def _build_annotations(tool_name: str) -> dict[str, Any]:
    """Erzeugt MCP-Annotations für ein Tool basierend auf seinem Namen."""
    annotations: dict[str, Any] = {}

    if tool_name in READ_ONLY_TOOLS:
        annotations[ToolAnnotationKey.READ_ONLY_HINT.value] = True

    if tool_name in DESTRUCTIVE_TOOLS:
        annotations[ToolAnnotationKey.DESTRUCTIVE_HINT.value] = True

    if tool_name in IDEMPOTENT_TOOLS:
        annotations[ToolAnnotationKey.IDEMPOTENT_HINT.value] = True

    return annotations


# ============================================================================
# MCP Bridge
# ============================================================================


class MCPBridge:
    """Zentrale Brücke zwischen Builtin-Handlers und MCP-Server.

    Nutzung:
        bridge = MCPBridge(config)
        bridge.setup(mcp_client, memory_manager)
        await bridge.start()  # Startet MCP-Server falls konfiguriert
    """

    def __init__(self, config: JarvisConfig) -> None:
        self._config = config
        self._server: JarvisMCPServer | None = None
        self._resource_provider: JarvisResourceProvider | None = None
        self._prompt_provider: JarvisPromptProvider | None = None
        self._discovery: DiscoveryManager | None = None
        self._server_config: MCPServerConfig | None = None
        self._enabled = False
        self._setup_time: float = 0

    def setup(
        self,
        mcp_client: JarvisMCPClient,
        memory: MemoryManager | None = None,
    ) -> bool:
        """Richtet den MCP-Server-Modus ein (falls konfiguriert).

        Liest die MCP-Server-Config, konvertiert bestehende Builtin-Tools
        in MCPToolDefs und registriert Resources + Prompts.

        Args:
            mcp_client: Der bestehende JarvisMCPClient mit registrierten Tools
            memory: MemoryManager für Resource-Zugriff

        Returns:
            True wenn MCP-Server-Modus aktiviert wurde, False sonst.
        """
        start = time.time()

        # Server-Config aus Jarvis-Config laden
        self._server_config = self._load_server_config()

        if self._server_config.mode == MCPServerMode.DISABLED:
            log.info("mcp_bridge_disabled", reason="server_mode=disabled")
            return False

        # MCP-Server erstellen
        self._server = JarvisMCPServer(self._server_config)

        # 1. Bestehende Builtin-Tools konvertieren
        tools_count = self._bridge_builtin_tools(mcp_client)

        # 2. Resources registrieren
        self._resource_provider = JarvisResourceProvider(
            config=self._config,
            memory=memory,
        )
        resources_count = self._resource_provider.register_all(self._server)

        # 3. Prompts registrieren
        self._prompt_provider = JarvisPromptProvider()
        prompts_count = self._prompt_provider.register_all(self._server)

        # 4. Discovery/Agent-Card
        self._discovery = DiscoveryManager(
            host=self._server_config.http_host,
            port=self._server_config.http_port,
            owner=getattr(self._config, "owner_name", ""),
        )
        self._discovery.build_card(
            tool_names=mcp_client.get_tool_list(),
            resource_count=resources_count,
            prompt_count=prompts_count,
            server_mode=self._server_config.mode.value,
        )

        self._enabled = True
        self._setup_time = time.time() - start

        log.info(
            "mcp_bridge_setup_complete",
            mode=self._server_config.mode.value,
            tools=tools_count,
            resources=resources_count,
            prompts=prompts_count,
            setup_ms=round(self._setup_time * 1000),
        )

        return True

    async def start(self) -> None:
        """Startet den MCP-Server (falls aktiviert)."""
        if self._server and self._enabled:
            await self._server.start()

    async def stop(self) -> None:
        """Stoppt den MCP-Server."""
        if self._server:
            await self._server.stop()

    # ── Properties ───────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        """Ist der MCP-Server-Modus aktiv?"""
        return self._enabled

    @property
    def server(self) -> JarvisMCPServer | None:
        """Der MCP-Server (für HTTP-Endpoint-Registrierung)."""
        return self._server

    @property
    def discovery(self) -> DiscoveryManager | None:
        """Der Discovery-Manager (für Agent-Card-Endpoint)."""
        return self._discovery

    # ── Bridge Logic ─────────────────────────────────────────────

    def _bridge_builtin_tools(self, mcp_client: JarvisMCPClient) -> int:
        """Konvertiert bestehende Builtin-Handler in MCPToolDefs.

        Liest alle registrierten Tools aus dem MCP-Client und
        erstellt für jeden eine MCPToolDef mit Annotations.

        Returns:
            Anzahl konvertierter Tools.
        """
        if self._server is None:
            return 0

        count = 0
        schemas = mcp_client.get_tool_schemas()

        for tool_name, schema in schemas.items():
            # Handler aus dem Client holen
            handler = mcp_client._builtin_handlers.get(tool_name)
            if handler is None:
                continue

            # MCPToolDef erstellen
            tool_def = MCPToolDef(
                name=tool_name,
                description=schema.get("description", ""),
                input_schema=schema.get("inputSchema", {}),
                handler=handler,  # Originaler Handler
                annotations=_build_annotations(tool_name),
            )

            self._server.register_tool(tool_def)
            count += 1

        log.info("mcp_bridge_tools_converted", count=count)
        return count

    # ── Config Loading ───────────────────────────────────────────

    def _load_server_config(self) -> MCPServerConfig:
        """Lädt die MCP-Server-Konfiguration.

        Prüft zuerst die Jarvis-Config, dann die MCP-Config-YAML.
        Default: DISABLED.
        """
        import yaml
        from pathlib import Path

        config = MCPServerConfig()  # Default: disabled

        # Aus MCP-Config-YAML laden
        mcp_config_path = self._config.mcp_config_file
        if mcp_config_path.exists():
            try:
                with open(mcp_config_path, encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}

                server_section = data.get("server_mode", {})
                if isinstance(server_section, dict):
                    mode_str = server_section.get("mode", "disabled")
                    try:
                        config.mode = MCPServerMode(mode_str)
                    except ValueError:
                        config.mode = MCPServerMode.DISABLED

                    config.http_host = server_section.get("http_host", config.http_host)
                    config.http_port = server_section.get("http_port", config.http_port)
                    config.server_name = server_section.get("server_name", config.server_name)
                    config.require_auth = server_section.get("require_auth", config.require_auth)
                    config.auth_token = server_section.get("auth_token", config.auth_token)
                    config.enable_sampling = server_section.get("enable_sampling", config.enable_sampling)
                    config.expose_tools = server_section.get("expose_tools", config.expose_tools)
                    config.expose_resources = server_section.get("expose_resources", config.expose_resources)
                    config.expose_prompts = server_section.get("expose_prompts", config.expose_prompts)

            except Exception as exc:
                log.warning("mcp_server_config_load_error", error=str(exc))

        return config

    # ── HTTP Request Handler ─────────────────────────────────────

    async def handle_mcp_request(
        self,
        body: dict[str, Any],
        auth_header: str = "",
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Verarbeitet einen eingehenden MCP HTTP-Request.

        Wird von config_routes.py aufgerufen.

        Args:
            body: JSON-RPC-Message(s)
            auth_header: Authorization-Header-Wert

        Returns:
            JSON-RPC-Response(s)
        """
        if self._server is None:
            return {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32000, "message": "MCP Server not running"},
            }

        # Token aus "Bearer xxx" extrahieren
        token = None
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]

        return await self._server.handle_http_request(body, auth_token=token)

    def get_agent_card(self) -> dict[str, Any]:
        """Gibt die Agent Card zurück (für /.well-known/agent.json)."""
        if self._discovery:
            return self._discovery.get_card()
        return {"error": "Discovery not initialized"}

    def get_health(self) -> dict[str, Any]:
        """Health-Check."""
        if self._discovery:
            return self._discovery.health()
        return {"status": "disabled"}

    # ── Stats ────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Gesamtstatistiken der MCP-Bridge."""
        result: dict[str, Any] = {
            "enabled": self._enabled,
            "setup_time_ms": round(self._setup_time * 1000) if self._setup_time else 0,
        }

        if self._server:
            result["server"] = self._server.stats()

        if self._resource_provider:
            result["resources"] = self._resource_provider.stats()

        if self._prompt_provider:
            result["prompts"] = self._prompt_provider.stats()

        if self._discovery:
            result["discovery"] = self._discovery.stats()

        return result
