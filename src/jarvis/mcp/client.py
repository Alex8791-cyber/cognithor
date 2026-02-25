"""MCP-Client: Verbindet sich mit mehreren MCP-Servern gleichzeitig.

Verantwortlich für:
  - Start und Verwaltung aller konfigurierten MCP-Server
  - Sammeln aller Tool-Schemas aus allen Servern
  - Dispatching von Tool-Calls an den richtigen Server
  - Reconnect bei Server-Absturz

Bibel-Referenz: §5.2 (Jarvis als MCP-Client), §5.4 (Server-Konfiguration)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import yaml

from jarvis.models import MCPServerConfig, MCPToolInfo
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.config import JarvisConfig

log = get_logger(__name__)

# Zeitlimits (Sekunden)
STDIO_CONNECT_TIMEOUT = 30.0
SESSION_INIT_TIMEOUT = 30.0
CLIENT_SESSION_TIMEOUT = 10.0
PROCESS_TERMINATION_TIMEOUT = 5.0

# Groessenlimits
MAX_CONTENT_LENGTH = 1_048_576  # 1 MB
MAX_CONFIG_FILE_SIZE = 1_048_576  # 1 MB

__all__ = [
    "JarvisMCPClient",
    "MCPClientError",
    "ServerConnection",
    "ToolCallResult",
]


@dataclass
class ToolCallResult:
    """Ergebnis eines MCP-Tool-Calls."""

    content: str = ""
    is_error: bool = False


@dataclass
class ServerConnection:
    """Aktive Verbindung zu einem MCP-Server."""

    name: str
    config: MCPServerConfig
    session: Any = None  # mcp.client.ClientSession
    read_stream: Any = None
    write_stream: Any = None
    process: asyncio.subprocess.Process | None = None
    tools: dict[str, MCPToolInfo] = field(default_factory=dict)
    connected: bool = False


class MCPClientError(Exception):
    """Fehler im MCP-Client."""


class JarvisMCPClient:
    """Multi-Server MCP-Client. [B§5.2]

    Verbindet sich mit allen konfigurierten MCP-Servern,
    sammelt deren Tool-Schemas, und dispatcht Tool-Calls.
    """

    def __init__(self, config: JarvisConfig) -> None:
        """Initialisiert den MCP-Client mit leerer Server- und Tool-Registry."""
        self._config = config
        self._servers: dict[str, ServerConnection] = {}
        self._tool_registry: dict[str, MCPToolInfo] = {}
        self._builtin_handlers: dict[str, Any] = {}
        self._subscriptions: dict[str, dict[str, list[Any]]] = {}

    async def connect_all(self) -> None:
        """Startet und verbindet alle konfigurierten MCP-Server.

        Liest die Server-Konfiguration aus ~/.jarvis/mcp/config.yaml
        und startet jeden Server als Subprocess (stdio) oder HTTP-Client.
        """
        server_configs = self._load_server_configs()

        for name, server_config in server_configs.items():
            if not server_config.enabled:
                log.info("mcp_server_disabled", server=name)
                continue

            try:
                await self._connect_server(name, server_config)
                log.info(
                    "mcp_server_connected",
                    server=name,
                    tools=list(self._servers[name].tools.keys()),
                )
            except Exception as exc:
                log.error(
                    "mcp_server_connect_failed",
                    server=name,
                    error=str(exc),
                )

        log.info(
            "mcp_client_ready",
            servers=len(self._servers),
            total_tools=len(self._tool_registry),
            tool_names=sorted(self._tool_registry.keys()),
        )

    def register_builtin_handler(
        self,
        tool_name: str,
        handler: Any,
        description: str = "",
        input_schema: dict[str, Any] | None = None,
    ) -> None:
        """Registriert einen eingebauten Tool-Handler (ohne MCP-Server).

        Nützlich für Phase 1 wo wir MCP-Server noch nicht als
        separate Prozesse starten, sondern direkt einbetten.
        """
        self._builtin_handlers[tool_name] = handler
        self._tool_registry[tool_name] = MCPToolInfo(
            name=tool_name,
            server="builtin",
            description=description,
            input_schema=input_schema or {},
        )
        log.debug("builtin_tool_registered", tool=tool_name)

    async def call_tool(
        self,
        name: str,
        params: dict[str, Any],
    ) -> ToolCallResult:
        """Ruft ein Tool auf dem zuständigen MCP-Server auf.

        Args:
            name: Tool-Name (z.B. "read_file")
            params: Tool-Parameter als Dict

        Returns:
            ToolCallResult mit Content und Error-Status.

        Raises:
            MCPClientError: Wenn Tool nicht gefunden oder Server nicht erreichbar.
        """
        # Zuerst eingebaute Handler prüfen
        if name in self._builtin_handlers:
            return await self._call_builtin(name, params)

        # Dann MCP-Server
        tool_info = self._tool_registry.get(name)
        if tool_info is None:
            available = ", ".join(sorted(self._tool_registry.keys()))
            return ToolCallResult(
                content=f"Tool '{name}' nicht gefunden. Verfügbar: {available}",
                is_error=True,
            )

        server = self._servers.get(tool_info.server)
        if server is None or not server.connected:
            return ToolCallResult(
                content=f"MCP-Server '{tool_info.server}' nicht verbunden",
                is_error=True,
            )

        try:
            result = await server.session.call_tool(name, arguments=params)
            # MCP SDK gibt Ergebnis als Liste von Content-Blöcken zurück
            content_parts = []
            total_len = 0
            max_content_len = MAX_CONTENT_LENGTH
            for block in result.content:
                text = block.text if hasattr(block, "text") else str(block)
                total_len += len(text)
                if total_len > max_content_len:
                    content_parts.append("[... Ausgabe gekuerzt (>1MB)]")
                    break
                content_parts.append(text)

            return ToolCallResult(
                content="\n".join(content_parts),
                is_error=getattr(result, "isError", False),
            )
        except Exception as exc:
            log.error(
                "mcp_tool_call_failed",
                tool=name,
                server=tool_info.server,
                error=str(exc),
            )
            return ToolCallResult(
                content=f"Tool-Fehler: {exc}",
                is_error=True,
            )

    def get_tool_schemas(self) -> dict[str, Any]:
        """Gibt alle verfügbaren Tool-Schemas zurück.

        Returns:
            Dict von Tool-Name → Schema (für den Planner-Prompt).
        """
        schemas = {}
        for name, info in self._tool_registry.items():
            schemas[name] = {
                "name": name,
                "description": info.description,
                "inputSchema": info.input_schema,
            }
        return schemas

    def get_tool_list(self) -> list[str]:
        """Gibt eine sortierte Liste aller Tool-Namen zurück."""
        return sorted(self._tool_registry.keys())

    @property
    def tool_count(self) -> int:
        """Anzahl registrierter Tools."""
        return len(self._tool_registry)

    @property
    def server_count(self) -> int:
        """Anzahl verbundener MCP-Server."""
        return len(self._servers)

    async def disconnect_all(self) -> None:
        """Trennt alle Server-Verbindungen und beendet Subprozesse."""
        for name, server in self._servers.items():
            try:
                # Properly close session context manager
                if server.session is not None:
                    try:
                        await server.session.__aexit__(None, None, None)
                    except Exception:
                        pass
                    server.session = None
                # Close stdio streams context manager
                if server.read_stream is not None:
                    try:
                        # The stdio_client returns streams that should be closed
                        if hasattr(server.read_stream, "aclose"):
                            await server.read_stream.aclose()
                    except Exception:
                        pass
                    server.read_stream = None
                if server.write_stream is not None:
                    try:
                        if hasattr(server.write_stream, "aclose"):
                            await server.write_stream.aclose()
                    except Exception:
                        pass
                    server.write_stream = None
                # Terminate subprocess
                if server.process and server.process.returncode is None:
                    server.process.terminate()
                    try:
                        await asyncio.wait_for(server.process.wait(), timeout=PROCESS_TERMINATION_TIMEOUT)
                    except TimeoutError:
                        server.process.kill()
                server.connected = False
                log.info("mcp_server_disconnected", server=name)
            except Exception as exc:
                log.warning("mcp_server_disconnect_error", server=name, error=str(exc))

        self._servers.clear()
        self._tool_registry.clear()

    # =========================================================================
    # Private Methoden
    # =========================================================================

    async def _call_builtin(
        self,
        name: str,
        params: dict[str, Any],
    ) -> ToolCallResult:
        """Ruft einen eingebauten Handler auf."""
        handler = self._builtin_handlers[name]
        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(**params)
            else:
                result = handler(**params)
            return ToolCallResult(content=str(result), is_error=False)
        except Exception as exc:
            return ToolCallResult(
                content=f"Builtin-Tool-Fehler: {exc}",
                is_error=True,
            )

    async def _connect_server(
        self,
        name: str,
        config: MCPServerConfig,
    ) -> None:
        """Verbindet sich mit einem einzelnen MCP-Server."""
        if config.transport == "stdio":
            await self._connect_stdio_server(name, config)
        elif config.transport == "http":
            log.warning("mcp_http_not_yet_implemented", server=name)
        else:
            log.error("mcp_unknown_transport", server=name, transport=config.transport)

    async def _connect_stdio_server(
        self,
        name: str,
        config: MCPServerConfig,
    ) -> None:
        """Verbindet sich mit einem stdio-basierten MCP-Server.

        Startet den Server als Subprocess und nutzt das MCP SDK
        für die Kommunikation über stdin/stdout.
        """
        try:
            from mcp.client import ClientSession  # type: ignore[attr-defined]
            from mcp.client.stdio import StdioServerParameters, stdio_client
        except ImportError:
            log.warning(
                "mcp_sdk_not_available",
                message="MCP SDK nicht installiert. Verwende pip install mcp",
            )
            return

        cmd = config.command
        if not cmd:
            log.error("mcp_no_command", server=name)
            return
        args = config.args

        # Python-Module als eigenen Interpreter starten
        server_params = StdioServerParameters(
            command=cmd,
            args=args,
            env={**config.env} if config.env else None,
        )

        try:
            read_stream, write_stream = await asyncio.wait_for(
                stdio_client(server_params).__aenter__(),
                timeout=STDIO_CONNECT_TIMEOUT,
            )

            session = await asyncio.wait_for(
                ClientSession(read_stream, write_stream).__aenter__(),
                timeout=CLIENT_SESSION_TIMEOUT,
            )

            await asyncio.wait_for(session.initialize(), timeout=SESSION_INIT_TIMEOUT)

            # Tools sammeln
            tools_result = await session.list_tools()
            tools = {}
            for tool in tools_result.tools:
                tool_info = MCPToolInfo(
                    name=tool.name,
                    server=name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema if hasattr(tool, "inputSchema") else {},
                )
                tools[tool.name] = tool_info
                self._tool_registry[tool.name] = tool_info

            conn = ServerConnection(
                name=name,
                config=config,
                session=session,
                read_stream=read_stream,
                write_stream=write_stream,
                tools=tools,
                connected=True,
            )
            self._servers[name] = conn

        except TimeoutError:
            log.error("mcp_server_timeout", server=name)
            raise MCPClientError(f"Timeout beim Verbinden mit MCP-Server '{name}'") from None
        except Exception as exc:
            log.error("mcp_server_error", server=name, error=str(exc))
            raise

    def _load_server_configs(self) -> dict[str, MCPServerConfig]:
        """Lädt MCP-Server-Konfiguration aus YAML."""
        configs: dict[str, MCPServerConfig] = {}

        config_path = self._config.mcp_config_file
        if not config_path.exists():
            log.info("mcp_config_not_found", path=str(config_path))
            return configs

        try:
            # Config-Datei-Groesse begrenzen
            config_size = config_path.stat().st_size
            if config_size > MAX_CONFIG_FILE_SIZE:
                log.error("mcp_config_too_large", size=config_size)
                return configs

            with open(config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not isinstance(data, dict) or "servers" not in data:
                return configs

            for name, server_data in data["servers"].items():
                if not isinstance(server_data, dict):
                    continue
                try:
                    configs[name] = MCPServerConfig(**server_data)
                except Exception as exc:
                    log.warning(
                        "mcp_server_config_invalid",
                        server=name,
                        error=str(exc),
                    )
        except Exception as exc:
            log.error("mcp_config_load_failed", error=str(exc))

        return configs

    async def subscribe_resource(
        self, server_name: str, uri: str, callback: Any,
    ) -> bool:
        """Abonniert Änderungen an einer MCP-Ressource."""
        conn = self._servers.get(server_name)
        if not conn or not conn.session:
            log.warning("mcp_subscribe_no_connection", server=server_name)
            return False
        try:
            await conn.session.subscribe_resource(uri)
            self._subscriptions.setdefault(server_name, {}).setdefault(uri, []).append(callback)
            log.info("mcp_subscribed", server=server_name, uri=uri)
            return True
        except Exception as exc:
            log.warning("mcp_subscribe_failed", server=server_name, error=str(exc))
            return False

    async def unsubscribe_resource(self, server_name: str, uri: str) -> bool:
        """Beendet ein Ressourcen-Abonnement."""
        conn = self._servers.get(server_name)
        if not conn or not conn.session:
            return False
        try:
            await conn.session.unsubscribe_resource(uri)
            if server_name in self._subscriptions:
                self._subscriptions[server_name].pop(uri, None)
            return True
        except Exception as exc:
            log.warning("mcp_unsubscribe_failed", error=str(exc))
            return False
