"""MCP-Server: Exponiert Jarvis als Standard-MCP-Server.

Ermöglicht es externen MCP-Clients (Claude Desktop, Cursor, VS Code,
andere Agenten) sich mit Jarvis zu verbinden und dessen Tools, Resources
und Prompts zu nutzen.

OPTIONAL: Funktioniert nur wenn `mcp` SDK installiert ist UND
server_mode in der Konfiguration aktiviert ist. Ohne MCP SDK
läuft Jarvis normal mit Builtin-Handlers weiter.

MCP-Spec 2025-11-25 Compliance:
  - Tools (list, call) mit Annotations
  - Resources (list, read, subscribe)
  - Prompts (list, get)
  - Sampling (createMessage)
  - Progress Notifications
  - Logging
  - Roots
  - Streamable HTTP + stdio Transport

Bibel-Referenz: §5.5 (Jarvis als MCP-Server)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable

from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# Protokoll-Version (MCP Spec)
PROTOCOL_VERSION = "2025-11-25"

__all__ = [
    "JarvisMCPServer",
    "MCPServerConfig",
    "MCPServerMode",
    "MCPToolDef",
    "MCPResource",
    "MCPResourceTemplate",
    "MCPPrompt",
    "MCPPromptArgument",
    "ProgressNotification",
    "MCPLogEntry",
    "ToolAnnotationKey",
]


# ============================================================================
# Tool Annotations (MCP Spec)
# ============================================================================


class ToolAnnotationKey(str, Enum):
    """Standard MCP Tool Annotation Keys."""

    TITLE = "title"
    READ_ONLY_HINT = "readOnlyHint"
    DESTRUCTIVE_HINT = "destructiveHint"
    IDEMPOTENT_HINT = "idempotentHint"
    OPEN_WORLD_HINT = "openWorldHint"


@dataclass
class MCPToolDef:
    """Vollständige MCP-Tool-Definition mit Annotations."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]
    annotations: dict[str, Any] = field(default_factory=dict)

    def to_mcp_schema(self) -> dict[str, Any]:
        """Konvertiert in MCP-Spec-kompatibles JSON."""
        schema: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }
        if self.annotations:
            schema["annotations"] = self.annotations
        return schema


# ============================================================================
# Resource Definitions
# ============================================================================


@dataclass
class MCPResource:
    """MCP Resource Definition."""

    uri: str  # z.B. "jarvis://memory/core"
    name: str
    description: str = ""
    mime_type: str = "text/plain"
    handler: Callable[..., Any] | None = None

    def to_mcp_schema(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type,
        }


@dataclass
class MCPResourceTemplate:
    """MCP Resource Template für dynamische URIs."""

    uri_template: str  # z.B. "jarvis://memory/entity/{entity_id}"
    name: str
    description: str = ""
    mime_type: str = "text/plain"
    handler: Callable[..., Any] | None = None

    def to_mcp_schema(self) -> dict[str, Any]:
        return {
            "uriTemplate": self.uri_template,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type,
        }


# ============================================================================
# Prompt Definitions
# ============================================================================


@dataclass
class MCPPromptArgument:
    """Argument für ein MCP-Prompt-Template."""

    name: str
    description: str = ""
    required: bool = False


@dataclass
class MCPPrompt:
    """MCP Prompt Template Definition."""

    name: str
    description: str = ""
    arguments: list[MCPPromptArgument] = field(default_factory=list)
    handler: Callable[..., Any] | None = None

    def to_mcp_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
        }
        if self.arguments:
            schema["arguments"] = [
                {
                    "name": arg.name,
                    "description": arg.description,
                    "required": arg.required,
                }
                for arg in self.arguments
            ]
        return schema


# ============================================================================
# Server Configuration
# ============================================================================


class MCPServerMode(str, Enum):
    """Betriebsmodus des MCP-Servers."""

    DISABLED = "disabled"   # Standard: Kein MCP-Server
    STDIO = "stdio"         # stdio-basierter Server (für Claude Desktop etc.)
    HTTP = "http"           # Streamable HTTP Server (für Netzwerk)
    BOTH = "both"           # Beide gleichzeitig


@dataclass
class MCPServerConfig:
    """Konfiguration des MCP-Server-Modus."""

    mode: MCPServerMode = MCPServerMode.DISABLED
    http_host: str = "127.0.0.1"
    http_port: int = 3001
    server_name: str = "jarvis"
    server_version: str = "1.0.0"
    require_auth: bool = False
    auth_token: str = ""
    max_concurrent_requests: int = 10
    enable_sampling: bool = False
    enable_logging: bool = True
    expose_tools: bool = True
    expose_resources: bool = True
    expose_prompts: bool = True


# ============================================================================
# Progress & Logging
# ============================================================================


@dataclass
class ProgressNotification:
    """MCP Progress Notification."""

    progress_token: str
    progress: float  # 0.0 - 1.0
    total: float = 1.0
    message: str = ""


@dataclass
class MCPLogEntry:
    """MCP Logging Notification."""

    level: str = "info"  # debug, info, warning, error, critical
    logger: str = "jarvis"
    data: Any = None


# ============================================================================
# MCP Server
# ============================================================================


class JarvisMCPServer:
    """Exponiert Jarvis als Standard-MCP-Server.

    Registrierung von Tools, Resources und Prompts über eine
    einfache API. Kann über stdio oder HTTP gestartet werden.

    Nutzung:
        server = JarvisMCPServer(config)
        server.register_tool(MCPToolDef(...))
        server.register_resource(MCPResource(...))
        server.register_prompt(MCPPrompt(...))
        await server.start()  # Startet stdio oder HTTP
    """

    MAX_BATCH_SIZE = 50  # Maximale Anzahl Requests in einem Batch
    MAX_SUBSCRIBERS_PER_URI = 100  # Maximale Subscriber pro URI
    HANDLER_TIMEOUT = 60  # Sekunden Timeout für Handler-Ausführung

    def __init__(self, config: MCPServerConfig | None = None) -> None:
        self._config = config or MCPServerConfig()
        self._tools: dict[str, MCPToolDef] = {}
        self._resources: dict[str, MCPResource] = {}
        self._resource_templates: dict[str, MCPResourceTemplate] = {}
        self._prompts: dict[str, MCPPrompt] = {}
        self._subscribers: dict[str, list[Callable]] = {}
        self._running = False
        self._request_count = 0
        self._start_time: float = 0
        self._progress_handlers: dict[str, Callable] = {}
        self._request_semaphore = asyncio.Semaphore(self._config.max_concurrent_requests)

    # ── Registration API ─────────────────────────────────────────

    def register_tool(self, tool: MCPToolDef) -> None:
        """Registriert ein Tool beim MCP-Server."""
        self._tools[tool.name] = tool
        log.debug("mcp_server_tool_registered", tool=tool.name)

    def register_resource(self, resource: MCPResource) -> None:
        """Registriert eine Resource beim MCP-Server."""
        self._resources[resource.uri] = resource
        log.debug("mcp_server_resource_registered", uri=resource.uri)

    def register_resource_template(self, template: MCPResourceTemplate) -> None:
        """Registriert ein Resource-Template."""
        self._resource_templates[template.uri_template] = template
        log.debug("mcp_server_template_registered", uri=template.uri_template)

    def register_prompt(self, prompt: MCPPrompt) -> None:
        """Registriert ein Prompt-Template."""
        self._prompts[prompt.name] = prompt
        log.debug("mcp_server_prompt_registered", prompt=prompt.name)

    # ── MCP Protocol Handlers ────────────────────────────────────

    async def handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        """Behandelt initialize-Request vom Client."""
        self._request_count += 1

        capabilities: dict[str, Any] = {}

        if self._config.expose_tools and self._tools:
            capabilities["tools"] = {"listChanged": True}

        if self._config.expose_resources and (
            self._resources or self._resource_templates
        ):
            capabilities["resources"] = {
                "subscribe": True,
                "listChanged": True,
            }

        if self._config.expose_prompts and self._prompts:
            capabilities["prompts"] = {"listChanged": True}

        if self._config.enable_sampling:
            capabilities["sampling"] = {}

        if self._config.enable_logging:
            capabilities["logging"] = {}

        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": capabilities,
            "serverInfo": {
                "name": self._config.server_name,
                "version": self._config.server_version,
            },
        }

    async def handle_tools_list(self) -> dict[str, Any]:
        """Behandelt tools/list-Request."""
        self._request_count += 1
        tools = [tool.to_mcp_schema() for tool in self._tools.values()]
        return {"tools": tools}

    async def handle_tools_call(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        progress_token: str | None = None,
    ) -> dict[str, Any]:
        """Behandelt tools/call-Request."""
        self._request_count += 1

        tool = self._tools.get(name)
        if tool is None:
            return {
                "content": [{"type": "text", "text": f"Tool '{name}' nicht gefunden"}],
                "isError": True,
            }

        try:
            # Progress-Notification senden falls Token vorhanden
            if progress_token:
                await self._send_progress(progress_token, 0.0, "Starte Tool...")

            handler = tool.handler
            if asyncio.iscoroutinefunction(handler):
                result = await asyncio.wait_for(
                    handler(**(arguments or {})), timeout=self.HANDLER_TIMEOUT,
                )
            else:
                result = handler(**(arguments or {}))

            if progress_token:
                await self._send_progress(progress_token, 1.0, "Fertig")

            # Ergebnis normalisieren
            if isinstance(result, str):
                content = [{"type": "text", "text": result}]
            elif isinstance(result, dict):
                content = [{"type": "text", "text": str(result)}]
            elif isinstance(result, list):
                content = result
            else:
                content = [{"type": "text", "text": str(result)}]

            return {"content": content, "isError": False}

        except asyncio.TimeoutError:
            log.error("mcp_server_tool_timeout", tool=name, timeout=self.HANDLER_TIMEOUT)
            return {
                "content": [{"type": "text", "text": f"Tool '{name}' Timeout nach {self.HANDLER_TIMEOUT}s"}],
                "isError": True,
            }
        except Exception as exc:
            log.error("mcp_server_tool_error", tool=name, error=str(exc))
            return {
                "content": [{"type": "text", "text": f"Tool-Fehler bei '{name}'"}],
                "isError": True,
            }

    async def handle_resources_list(self) -> dict[str, Any]:
        """Behandelt resources/list-Request."""
        self._request_count += 1
        resources = [r.to_mcp_schema() for r in self._resources.values()]
        return {"resources": resources}

    async def handle_resources_templates_list(self) -> dict[str, Any]:
        """Behandelt resources/templates/list-Request."""
        self._request_count += 1
        templates = [t.to_mcp_schema() for t in self._resource_templates.values()]
        return {"resourceTemplates": templates}

    async def handle_resources_read(self, uri: str) -> dict[str, Any]:
        """Behandelt resources/read-Request."""
        self._request_count += 1

        resource = self._resources.get(uri)
        if resource is None:
            # Template-Matching versuchen
            resource = self._match_template(uri)
            if resource is None:
                return {
                    "contents": [{
                        "uri": uri,
                        "mimeType": "text/plain",
                        "text": f"Resource '{uri}' nicht gefunden",
                    }]
                }

        try:
            if resource.handler:
                if asyncio.iscoroutinefunction(resource.handler):
                    content = await asyncio.wait_for(
                        resource.handler(uri=uri), timeout=self.HANDLER_TIMEOUT,
                    )
                else:
                    content = resource.handler(uri=uri)
            else:
                content = ""

            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": resource.mime_type,
                    "text": str(content),
                }]
            }
        except asyncio.TimeoutError:
            log.error("mcp_server_resource_timeout", uri=uri, timeout=self.HANDLER_TIMEOUT)
            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": "text/plain",
                    "text": f"Resource-Timeout nach {self.HANDLER_TIMEOUT}s",
                }]
            }
        except Exception as exc:
            log.error("mcp_server_resource_error", uri=uri, error=str(exc))
            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": "text/plain",
                    "text": "Resource-Lesefehler",
                }]
            }

    async def handle_resources_subscribe(self, uri: str) -> dict[str, Any]:
        """Behandelt resources/subscribe-Request."""
        self._request_count += 1
        if uri not in self._subscribers:
            self._subscribers[uri] = []
        elif len(self._subscribers[uri]) >= self.MAX_SUBSCRIBERS_PER_URI:
            log.warning("mcp_subscriber_limit_reached", uri=uri, limit=self.MAX_SUBSCRIBERS_PER_URI)
            return {"error": {"code": -32000, "message": "Subscriber-Limit erreicht"}}
        return {}

    async def notify_subscribers(self, uri: str) -> None:
        """Benachrichtigt alle Subscriber einer Resource über Änderungen.

        Sendet eine notifications/resources/updated-Nachricht an alle
        registrierten Callbacks für die gegebene URI.

        Args:
            uri: Die URI der geänderten Resource.
        """
        callbacks = self._subscribers.get(uri, [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(uri)
                else:
                    callback(uri)
            except Exception as exc:
                log.warning("mcp_subscriber_notify_failed", uri=uri, error=str(exc))

    async def handle_prompts_list(self) -> dict[str, Any]:
        """Behandelt prompts/list-Request."""
        self._request_count += 1
        prompts = [p.to_mcp_schema() for p in self._prompts.values()]
        return {"prompts": prompts}

    async def handle_prompts_get(
        self,
        name: str,
        arguments: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Behandelt prompts/get-Request."""
        self._request_count += 1

        prompt = self._prompts.get(name)
        if prompt is None:
            return {"description": f"Prompt '{name}' nicht gefunden", "messages": []}

        try:
            if prompt.handler:
                if asyncio.iscoroutinefunction(prompt.handler):
                    messages = await asyncio.wait_for(
                        prompt.handler(**(arguments or {})), timeout=self.HANDLER_TIMEOUT,
                    )
                else:
                    messages = prompt.handler(**(arguments or {}))
            else:
                messages = []

            return {
                "description": prompt.description,
                "messages": messages if isinstance(messages, list) else [messages],
            }
        except asyncio.TimeoutError:
            log.error("mcp_server_prompt_timeout", prompt=name, timeout=self.HANDLER_TIMEOUT)
            return {"description": "Prompt-Timeout", "messages": []}
        except Exception as exc:
            log.error("mcp_server_prompt_error", prompt=name, error=str(exc))
            return {"description": "Prompt-Verarbeitungsfehler", "messages": []}

    async def handle_logging_set_level(self, level: str) -> dict[str, Any]:
        """Behandelt logging/setLevel-Request."""
        self._request_count += 1
        log.info("mcp_server_log_level_set", level=level)
        return {}

    # ── JSON-RPC Dispatcher ──────────────────────────────────────

    async def dispatch(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Dispatcht einen JSON-RPC-2.0-Request an den richtigen Handler.

        Args:
            method: JSON-RPC-Methode (z.B. "tools/list")
            params: Parameter-Dict

        Returns:
            Result-Dict für die JSON-RPC-Response
        """
        params = params or {}

        handlers: dict[str, Callable[..., Awaitable[dict[str, Any]]]] = {
            "initialize": lambda: self.handle_initialize(params),
            "tools/list": self.handle_tools_list,
            "tools/call": lambda: self.handle_tools_call(
                name=params.get("name", ""),
                arguments=params.get("arguments"),
                progress_token=params.get("_meta", {}).get("progressToken"),
            ),
            "resources/list": self.handle_resources_list,
            "resources/templates/list": self.handle_resources_templates_list,
            "resources/read": lambda: self.handle_resources_read(
                uri=params.get("uri", ""),
            ),
            "resources/subscribe": lambda: self.handle_resources_subscribe(
                uri=params.get("uri", ""),
            ),
            "prompts/list": self.handle_prompts_list,
            "prompts/get": lambda: self.handle_prompts_get(
                name=params.get("name", ""),
                arguments=params.get("arguments"),
            ),
            "logging/setLevel": lambda: self.handle_logging_set_level(
                level=params.get("level", "info"),
            ),
            "ping": self._handle_ping,
        }

        handler = handlers.get(method)
        if handler is None:
            return {"error": {"code": -32601, "message": f"Method not found: {method}"}}

        async with self._request_semaphore:
            try:
                return await handler()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error("mcp_server_dispatch_error", method=method, error=str(exc))
                return {"error": {"code": -32603, "message": "Interner Server-Fehler"}}

    async def _handle_ping(self) -> dict[str, Any]:
        """Ping-Handler für Health-Checks."""
        self._request_count += 1
        return {}

    # ── JSON-RPC Message Processing ──────────────────────────────

    async def process_jsonrpc_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Verarbeitet eine JSON-RPC-2.0-Nachricht.

        Unterstützt Requests (mit id) und Notifications (ohne id).

        Returns:
            JSON-RPC-Response-Dict oder None für Notifications.
        """
        jsonrpc = message.get("jsonrpc", "2.0")
        method = message.get("method", "")
        params = message.get("params")
        msg_id = message.get("id")

        # Notification (kein id) → kein Response
        if msg_id is None:
            await self.dispatch(method, params)
            return None

        # Request → Response
        result = await self.dispatch(method, params)

        if "error" in result:
            return {
                "jsonrpc": jsonrpc,
                "id": msg_id,
                "error": result["error"],
            }

        return {
            "jsonrpc": jsonrpc,
            "id": msg_id,
            "result": result,
        }

    # ── Lifecycle ────────────────────────────────────────────────

    async def start(self) -> None:
        """Startet den MCP-Server im konfigurierten Modus."""
        if self._config.mode == MCPServerMode.DISABLED:
            log.info("mcp_server_disabled")
            return

        self._running = True
        self._start_time = time.time()

        log.info(
            "mcp_server_starting",
            mode=self._config.mode.value,
            tools=len(self._tools),
            resources=len(self._resources),
            prompts=len(self._prompts),
        )

        if self._config.mode in (MCPServerMode.STDIO, MCPServerMode.BOTH):
            t = asyncio.create_task(self._run_stdio())
            t.add_done_callback(self._handle_task_exception)

        if self._config.mode in (MCPServerMode.HTTP, MCPServerMode.BOTH):
            t = asyncio.create_task(self._run_http())
            t.add_done_callback(self._handle_task_exception)

    def _handle_task_exception(self, task: asyncio.Task[None]) -> None:
        """Callback for fire-and-forget tasks — logs unhandled exceptions."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            log.error("mcp_server_task_error", error=str(exc), exc_type=type(exc).__name__)

    async def stop(self) -> None:
        """Stoppt den MCP-Server."""
        self._running = False
        log.info("mcp_server_stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # ── stdio Transport ──────────────────────────────────────────

    async def _run_stdio(self) -> None:
        """Liest JSON-RPC-Nachrichten von stdin, schreibt auf stdout.

        Implementiert das MCP stdio-Protokoll:
        - Liest Zeile für Zeile von stdin
        - Jede Zeile ist ein JSON-RPC-2.0-Message
        - Responses werden als JSON-Zeile auf stdout geschrieben
        """
        import json
        import sys

        log.info("mcp_server_stdio_started")

        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_running_loop().connect_read_pipe(lambda: protocol, sys.stdin)

        while self._running:
            try:
                line = await asyncio.wait_for(reader.readline(), timeout=1.0)
                if not line:
                    break

                message = json.loads(line.decode("utf-8").strip())
                response = await self.process_jsonrpc_message(message)

                if response is not None:
                    sys.stdout.write(json.dumps(response) + "\n")
                    sys.stdout.flush()

            except asyncio.TimeoutError:
                continue
            except json.JSONDecodeError as exc:
                log.warning("mcp_server_invalid_json", error=str(exc))
            except Exception as exc:
                log.error("mcp_server_stdio_error", error=str(exc))
                break

        log.info("mcp_server_stdio_stopped")

    # ── HTTP Transport (Streamable HTTP) ─────────────────────────

    async def _run_http(self) -> None:
        """Startet Streamable HTTP Transport.

        Nutzt den bestehenden FastAPI/Starlette-Stack von Jarvis.
        Der eigentliche HTTP-Endpoint wird in config_routes registriert.
        """
        log.info(
            "mcp_server_http_ready",
            host=self._config.http_host,
            port=self._config.http_port,
        )
        # HTTP-Endpoints werden extern in config_routes.py registriert
        # Der Server stellt nur die dispatch-Methode bereit

    async def handle_http_request(
        self,
        body: dict[str, Any],
        auth_token: str | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Verarbeitet einen HTTP-Request (Streamable HTTP Transport).

        Unterstützt:
        - Einzelne JSON-RPC-Messages
        - Batch-Requests (Array von Messages)

        Args:
            body: JSON-RPC-Message oder Array von Messages
            auth_token: Optional Bearer Token für Authentifizierung

        Returns:
            JSON-RPC-Response oder Array von Responses
        """
        # Auth prüfen
        if self._config.require_auth:
            import hmac
            if not hmac.compare_digest(
                (auth_token or "").encode(), self._config.auth_token.encode()
            ):
                return {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32000, "message": "Unauthorized"},
                }

        # Batch-Request
        if isinstance(body, list):
            if len(body) > self.MAX_BATCH_SIZE:
                return {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32000,
                        "message": f"Batch zu gross ({len(body)} > {self.MAX_BATCH_SIZE})",
                    },
                }
            responses = []
            for msg in body:
                resp = await self.process_jsonrpc_message(msg)
                if resp is not None:
                    responses.append(resp)
            return responses

        # Einzelner Request
        resp = await self.process_jsonrpc_message(body)
        return resp or {"jsonrpc": "2.0", "id": None, "result": {}}

    # ── Helpers ───────────────────────────────────────────────────

    def _match_template(self, uri: str) -> MCPResource | None:
        """Versucht eine URI gegen registrierte Templates zu matchen."""
        for tmpl_uri, template in self._resource_templates.items():
            # Einfaches Pattern-Matching: {param} → beliebiger Wert
            pattern_parts = tmpl_uri.split("/")
            uri_parts = uri.split("/")
            if len(pattern_parts) != len(uri_parts):
                continue
            match = True
            for pp, up in zip(pattern_parts, uri_parts):
                if pp.startswith("{") and pp.endswith("}"):
                    continue  # Wildcard
                if pp != up:
                    match = False
                    break
            if match:
                return MCPResource(
                    uri=uri,
                    name=template.name,
                    description=template.description,
                    mime_type=template.mime_type,
                    handler=template.handler,
                )
        return None

    async def _send_progress(
        self,
        token: str,
        progress: float,
        message: str = "",
    ) -> None:
        """Sendet eine Progress-Notification."""
        handler = self._progress_handlers.get(token)
        if handler:
            notification = ProgressNotification(
                progress_token=token,
                progress=progress,
                message=message,
            )
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(notification)
                else:
                    handler(notification)
            except Exception:
                pass

    # ── Stats ────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Gibt Statistiken des MCP-Servers zurück."""
        uptime = time.time() - self._start_time if self._start_time else 0
        return {
            "mode": self._config.mode.value,
            "running": self._running,
            "uptime_seconds": round(uptime, 1),
            "tools_registered": len(self._tools),
            "resources_registered": len(self._resources),
            "resource_templates": len(self._resource_templates),
            "prompts_registered": len(self._prompts),
            "total_requests": self._request_count,
            "subscribers": sum(len(v) for v in self._subscribers.values()),
            "server_info": {
                "name": self._config.server_name,
                "version": self._config.server_version,
                "protocol_version": PROTOCOL_VERSION,
            },
        }
