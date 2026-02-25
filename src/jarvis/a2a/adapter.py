"""A2A Adapter: Brücke zwischen Jarvis-Interop (JAIP) und A2A RC v1.0.

Verbindet InteropProtocol mit dem standardisierten A2A-Protokoll.
Verantwortlich für:
  1. JAIP-Capabilities → A2A-Skills
  2. JAIP-AgentIdentity → A2A-AgentCard
  3. Eingehende A2A-Tasks → Jarvis Message-Handler
  4. Gateway-Integration (Boot/Shutdown)

OPTIONAL: Aktiviert sich nur wenn A2A in Config eingeschaltet ist.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from jarvis.a2a.client import A2AClient
from jarvis.a2a.server import A2AServer, A2AServerConfig
from jarvis.a2a.types import (
    A2A_PROTOCOL_VERSION,
    A2AAgentCapabilities,
    A2AAgentCard,
    A2AInterface,
    A2AProvider,
    A2ASkill,
    Artifact,
    Message,
    MessageRole,
    Task,
    TaskState,
    TextPart,
)
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.config import JarvisConfig
    from jarvis.core.interop import AgentCapability, InteropProtocol

log = get_logger(__name__)


# ── Capability → Skill Mapping ───────────────────────────────────

CAPABILITY_TO_SKILL: dict[str, dict[str, str]] = {
    "task_execution": {"name": "Aufgabenausführung", "description": "Allgemeine Aufgaben planen und ausführen"},
    "data_analysis": {"name": "Datenanalyse", "description": "Strukturierte Datenanalyse und Visualisierung"},
    "web_search": {"name": "Web-Recherche", "description": "Internet-Suche und Informationsextraktion"},
    "code_generation": {"name": "Code-Generierung", "description": "Programmcode erstellen und debuggen"},
    "document_processing": {"name": "Dokumentenverarbeitung", "description": "PDFs, DOCX analysieren"},
    "image_analysis": {"name": "Bildanalyse", "description": "Bilder beschreiben und analysieren"},
    "translation": {"name": "Übersetzung", "description": "Texte zwischen Sprachen übersetzen"},
    "scheduling": {"name": "Terminplanung", "description": "Termine koordinieren"},
    "email": {"name": "E-Mail", "description": "E-Mails verwalten"},
    "crm": {"name": "CRM", "description": "Kundenverwaltung"},
}


def capabilities_to_skills(capabilities: list[AgentCapability]) -> list[A2ASkill]:
    skills: list[A2ASkill] = []
    for cap in capabilities:
        cap_type = cap.capability_type.value
        mapping = CAPABILITY_TO_SKILL.get(cap_type, {})
        skills.append(A2ASkill(
            id=cap_type,
            name=mapping.get("name", cap_type),
            description=cap.description or mapping.get("description", ""),
            tags=[cap_type] + cap.languages,
        ))
    return skills


# ── A2A Adapter ──────────────────────────────────────────────────

class A2AAdapter:
    """Zentrale Brücke zwischen JAIP und A2A RC v1.0."""

    def __init__(self, config: JarvisConfig) -> None:
        self._config = config
        self._server: A2AServer | None = None
        self._client: A2AClient | None = None
        self._interop: InteropProtocol | None = None
        self._enabled = False
        self._setup_time: float = 0
        self._message_handler: Any = None

    def setup(self, interop: InteropProtocol | None = None,
              message_handler: Any = None) -> bool:
        start = time.time()
        server_config = self._load_config()
        if not server_config.enabled:
            log.info("a2a_adapter_disabled")
            return False

        self._interop = interop
        self._message_handler = message_handler

        self._server = A2AServer(server_config)
        self._server.set_task_handler(self._handle_incoming_task)

        if interop:
            card = self._build_card_from_interop(interop, server_config)
            self._server.set_agent_card(card)

        self._client = A2AClient()
        self._enabled = True
        self._setup_time = time.time() - start

        log.info("a2a_adapter_setup_complete", host=server_config.host,
                 port=server_config.port, version=A2A_PROTOCOL_VERSION)
        return True

    async def start(self) -> None:
        if self._server and self._enabled:
            await self._server.start()

    async def stop(self) -> None:
        if self._server:
            await self._server.stop()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def server(self) -> A2AServer | None:
        return self._server

    @property
    def client(self) -> A2AClient | None:
        return self._client

    # ── Incoming Task Processing ─────────────────────────────────

    async def _handle_incoming_task(self, task: Task) -> Task:
        user_text = ""
        for msg in reversed(task.messages):
            if msg.role == MessageRole.USER and msg.text:
                user_text = msg.text
                break

        if not user_text:
            task.transition(TaskState.FAILED, Message(
                role=MessageRole.AGENT, parts=[TextPart(text="No text message in task.")],
            ))
            return task

        if self._message_handler:
            try:
                import asyncio
                if asyncio.iscoroutinefunction(self._message_handler):
                    result = await self._message_handler(
                        user_text, session_id=f"a2a-{task.id}", channel="a2a",
                    )
                else:
                    result = self._message_handler(
                        user_text, session_id=f"a2a-{task.id}", channel="a2a",
                    )

                result_text = str(result) if result else "Task processed (no result)"
                task.artifacts.append(Artifact(
                    parts=[TextPart(text=result_text)], name="response",
                ))
                task.transition(TaskState.COMPLETED, Message(
                    role=MessageRole.AGENT, parts=[TextPart(text=result_text[:500])],
                ))
            except Exception as exc:
                task.transition(TaskState.FAILED, Message(
                    role=MessageRole.AGENT, parts=[TextPart(text=f"Error: {exc}")],
                ))
        else:
            task.artifacts.append(Artifact(
                parts=[TextPart(text=f"Echo: {user_text}")], name="echo",
            ))
            task.transition(TaskState.COMPLETED, Message(
                role=MessageRole.AGENT, parts=[TextPart(text=f"Task received: {user_text[:200]}")],
            ))
        return task

    # ── Outgoing Delegation ──────────────────────────────────────

    async def delegate_task(self, endpoint: str, text: str,
                            context_id: str | None = None,
                            metadata: dict[str, Any] | None = None) -> Task | None:
        if not self._client:
            return None
        return await self._client.send_message(
            endpoint=endpoint, text=text, context_id=context_id, metadata=metadata,
        )

    async def discover_remote(self, endpoint: str) -> A2AAgentCard | None:
        if not self._client:
            return None
        return await self._client.discover(endpoint)

    # ── Agent Card Builder ───────────────────────────────────────

    def _build_card_from_interop(self, interop: InteropProtocol,
                                 server_config: A2AServerConfig) -> A2AAgentCard:
        all_caps: list[AgentCapability] = []
        local_agent = interop.get_agent(interop.local_agent_id)
        if local_agent:
            all_caps.extend(interop.capabilities.get_capabilities(interop.local_agent_id))

        skills = capabilities_to_skills(all_caps)
        base_url = f"http://{server_config.host}:{server_config.port}"

        return A2AAgentCard(
            name="Jarvis",
            description="Lokaler KI-Agent mit Memory, Browser-Automatisierung und EU-AI-Act-Compliance.",
            url=base_url,
            version="16.0.0",
            provider=A2AProvider(organization=getattr(self._config, "owner_name", "Jarvis")),
            capabilities=A2AAgentCapabilities(
                streaming=server_config.enable_streaming,
                push_notifications=server_config.enable_push,
                state_transition_history=True,
            ),
            skills=skills,
            interfaces=[
                A2AInterface(protocol="jsonrpc", url=f"{base_url}/a2a"),
            ],
            tags=["jarvis", "local-first", "privacy", "german", "insurance"],
        )

    # ── Config Loading ───────────────────────────────────────────

    def _load_config(self) -> A2AServerConfig:
        import yaml
        config = A2AServerConfig()
        mcp_config_path = self._config.mcp_config_file
        if mcp_config_path.exists():
            try:
                with open(mcp_config_path, encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                a2a = data.get("a2a", {})
                if isinstance(a2a, dict):
                    config.enabled = a2a.get("enabled", False)
                    config.host = a2a.get("host", config.host)
                    config.port = a2a.get("port", config.port)
                    config.agent_name = a2a.get("agent_name", config.agent_name)
                    config.require_auth = a2a.get("require_auth", config.require_auth)
                    config.auth_token = a2a.get("auth_token", config.auth_token)
                    config.max_tasks = a2a.get("max_tasks", config.max_tasks)
                    config.enable_streaming = a2a.get("enable_streaming", False)
                    config.enable_push = a2a.get("enable_push", False)
            except Exception as exc:
                log.warning("a2a_config_load_error", error=str(exc))
        return config

    # ── HTTP Handlers ────────────────────────────────────────────

    async def handle_a2a_request(self, body: dict[str, Any], auth_header: str = "",
                                 client_version: str = "") -> dict[str, Any]:
        if self._server is None:
            return {"jsonrpc": "2.0", "id": body.get("id"),
                    "error": {"code": -32000, "message": "A2A Server not running"}}
        token = auth_header[7:] if auth_header.startswith("Bearer ") else None
        return await self._server.handle_http_request(body, auth_token=token,
                                                       client_version=client_version)

    async def handle_stream_request(self, body: dict[str, Any],
                                     auth_token: str | None = None):
        """Streaming-Endpoint: Leitet an Server weiter."""
        if self._server is None:
            yield 'event: error\ndata: {"code": -32000, "message": "A2A Server not running"}\n\n'
            return
        async for event in self._server.handle_stream_request(body, auth_token):
            yield event

    def get_agent_card(self) -> dict[str, Any]:
        if self._server:
            return self._server.get_agent_card()
        return {"error": "A2A not initialized"}

    def stats(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "enabled": self._enabled,
            "protocol_version": A2A_PROTOCOL_VERSION,
            "setup_time_ms": round(self._setup_time * 1000) if self._setup_time else 0,
        }
        if self._server:
            result["server"] = self._server.stats()
        if self._client:
            result["client"] = self._client.stats()
        return result
