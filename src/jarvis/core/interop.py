"""Jarvis · Cross-Agent Interoperability Protocol (JAIP).

Standardisiertes Kommunikationsprotokoll für mehrere Jarvis-Instanzen:

  - AgentIdentity:         Eindeutige Identität einer Jarvis-Instanz
  - AgentCapability:       Gemeldete Fähigkeiten einer Instanz
  - InteropMessage:        Strukturierte Nachricht zwischen Agenten
  - MessageRouter:         Routing von Nachrichten an registrierte Agenten
  - CapabilityRegistry:    Discovery: Welcher Agent kann was?
  - FederationManager:     Verwaltung von Agent-Verbünden
  - InteropProtocol:       Hauptklasse, orchestriert alles

Architektur-Bibel: §12.1 (Multi-Agent), §15.3 (Federation)
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


# ============================================================================
# Identity & Capabilities
# ============================================================================


@dataclass
class AgentIdentity:
    """Eindeutige Identität einer Jarvis-Instanz."""

    agent_id: str
    instance_name: str
    version: str = "1.0.0"
    endpoint: str = ""            # URL oder lokale Adresse
    public_key: str = ""          # Für signierte Nachrichten
    owner: str = ""
    registered_at: str = ""
    last_seen: str = ""
    status: str = "online"        # online, offline, busy, maintenance

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "instance_name": self.instance_name,
            "version": self.version,
            "endpoint": self.endpoint,
            "status": self.status,
            "owner": self.owner,
            "last_seen": self.last_seen,
        }


class CapabilityType(Enum):
    TASK_EXECUTION = "task_execution"
    DATA_ANALYSIS = "data_analysis"
    WEB_SEARCH = "web_search"
    CODE_GENERATION = "code_generation"
    DOCUMENT_PROCESSING = "document_processing"
    IMAGE_ANALYSIS = "image_analysis"
    TRANSLATION = "translation"
    SCHEDULING = "scheduling"
    EMAIL = "email"
    CRM = "crm"
    CUSTOM = "custom"


@dataclass
class AgentCapability:
    """Eine gemeldete Fähigkeit einer Instanz."""

    capability_type: CapabilityType
    description: str = ""
    languages: list[str] = field(default_factory=lambda: ["de", "en"])
    max_concurrent: int = 5
    avg_response_ms: float = 0.0
    cost_per_call: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.capability_type.value,
            "description": self.description,
            "languages": self.languages,
            "max_concurrent": self.max_concurrent,
            "avg_response_ms": round(self.avg_response_ms, 1),
        }


# ============================================================================
# Messages
# ============================================================================


class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    HEARTBEAT = "heartbeat"
    CAPABILITY_ANNOUNCE = "capability_announce"
    TASK_DELEGATE = "task_delegate"
    TASK_RESULT = "task_result"
    ERROR = "error"


class MessagePriority(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class InteropMessage:
    """Strukturierte Nachricht zwischen Agenten."""

    message_id: str
    msg_type: MessageType
    sender_id: str
    receiver_id: str            # "" = broadcast
    payload: dict[str, Any] = field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: str = ""
    correlation_id: str = ""    # Für Request/Response-Paare
    ttl_seconds: int = 300      # Time-to-Live
    signature: str = ""         # Optionale Signatur

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "type": self.msg_type.value,
            "sender": self.sender_id,
            "receiver": self.receiver_id,
            "priority": self.priority.value,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "payload_keys": list(self.payload.keys()),
        }


# ============================================================================
# Capability Registry (Discovery)
# ============================================================================


class CapabilityRegistry:
    """Zentrales Register: Welcher Agent kann was?"""

    def __init__(self) -> None:
        self._capabilities: dict[str, list[AgentCapability]] = {}  # agent_id → caps

    def register(self, agent_id: str, capabilities: list[AgentCapability]) -> None:
        self._capabilities[agent_id] = capabilities

    def unregister(self, agent_id: str) -> bool:
        if agent_id in self._capabilities:
            del self._capabilities[agent_id]
            return True
        return False

    def find_agents(self, capability_type: CapabilityType) -> list[str]:
        """Findet alle Agenten mit einer bestimmten Fähigkeit."""
        return [
            agent_id
            for agent_id, caps in self._capabilities.items()
            if any(c.capability_type == capability_type for c in caps)
        ]

    def find_by_language(self, language: str) -> list[str]:
        return [
            agent_id
            for agent_id, caps in self._capabilities.items()
            if any(language in c.languages for c in caps)
        ]

    def best_agent_for(
        self, capability_type: CapabilityType, *, prefer_fast: bool = True
    ) -> str | None:
        """Findet den besten Agenten für eine Aufgabe."""
        candidates: list[tuple[str, AgentCapability]] = []
        for agent_id, caps in self._capabilities.items():
            for cap in caps:
                if cap.capability_type == capability_type:
                    candidates.append((agent_id, cap))

        if not candidates:
            return None

        if prefer_fast:
            candidates.sort(key=lambda x: x[1].avg_response_ms)
        else:
            candidates.sort(key=lambda x: x[1].cost_per_call)

        return candidates[0][0]

    def get_capabilities(self, agent_id: str) -> list[AgentCapability]:
        return self._capabilities.get(agent_id, [])

    @property
    def agent_count(self) -> int:
        return len(self._capabilities)

    def stats(self) -> dict[str, Any]:
        all_caps: list[AgentCapability] = []
        for caps in self._capabilities.values():
            all_caps.extend(caps)
        return {
            "registered_agents": len(self._capabilities),
            "total_capabilities": len(all_caps),
            "capability_types": list({c.capability_type.value for c in all_caps}),
        }


# ============================================================================
# Message Router
# ============================================================================


class MessageRouter:
    """Routing von Nachrichten an registrierte Agenten."""

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[[InteropMessage], Any]] = {}
        self._message_log: list[InteropMessage] = []
        self._broadcast_handlers: list[Callable[[InteropMessage], Any]] = []

    def register_handler(self, agent_id: str, handler: Callable[[InteropMessage], Any]) -> None:
        self._handlers[agent_id] = handler

    def register_broadcast_handler(self, handler: Callable[[InteropMessage], Any]) -> None:
        self._broadcast_handlers.append(handler)

    def unregister(self, agent_id: str) -> bool:
        if agent_id in self._handlers:
            del self._handlers[agent_id]
            return True
        return False

    def send(self, message: InteropMessage) -> dict[str, Any]:
        """Sendet eine Nachricht an den Ziel-Agenten."""
        self._message_log.append(message)

        if not message.receiver_id:
            # Broadcast
            results = []
            for handler in self._broadcast_handlers:
                try:
                    results.append(handler(message))
                except Exception as e:
                    results.append({"error": str(e)})
            return {"delivered": True, "broadcast": True, "receivers": len(results)}

        handler = self._handlers.get(message.receiver_id)
        if not handler:
            return {"delivered": False, "error": f"Agent '{message.receiver_id}' nicht erreichbar"}

        try:
            result = handler(message)
            return {"delivered": True, "result": result}
        except Exception as e:
            return {"delivered": False, "error": str(e)}

    def create_message(
        self,
        msg_type: MessageType,
        sender_id: str,
        receiver_id: str = "",
        payload: dict[str, Any] | None = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        correlation_id: str = "",
    ) -> InteropMessage:
        msg_id = hashlib.sha256(f"msg:{time.time()}:{sender_id}".encode()).hexdigest()[:16]
        return InteropMessage(
            message_id=msg_id,
            msg_type=msg_type,
            sender_id=sender_id,
            receiver_id=receiver_id,
            payload=payload or {},
            priority=priority,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            correlation_id=correlation_id,
        )

    @property
    def message_count(self) -> int:
        return len(self._message_log)

    def recent_messages(self, limit: int = 20) -> list[InteropMessage]:
        return list(reversed(self._message_log[-limit:]))

    def stats(self) -> dict[str, Any]:
        msgs = self._message_log
        return {
            "total_messages": len(msgs),
            "registered_handlers": len(self._handlers),
            "broadcast_handlers": len(self._broadcast_handlers),
            "by_type": {
                t.value: sum(1 for m in msgs if m.msg_type == t)
                for t in MessageType
                if any(m.msg_type == t for m in msgs)
            },
        }


# ============================================================================
# Federation Manager
# ============================================================================


class FederationStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    REVOKED = "revoked"


@dataclass
class FederationLink:
    """Verbindung zwischen zwei Jarvis-Instanzen."""

    link_id: str
    local_agent_id: str
    remote_agent_id: str
    remote_endpoint: str
    status: FederationStatus = FederationStatus.PENDING
    established_at: str = ""
    allowed_capabilities: list[CapabilityType] = field(default_factory=list)
    max_requests_per_hour: int = 100
    requests_this_hour: int = 0
    _hour_start: float = field(default_factory=time.monotonic)

    def _maybe_reset_hour(self) -> None:
        """Setzt den Stundenzaehler zurueck wenn eine Stunde vergangen ist."""
        if time.monotonic() - self._hour_start >= 3600:
            self.requests_this_hour = 0
            self._hour_start = time.monotonic()

    @property
    def rate_limited(self) -> bool:
        self._maybe_reset_hour()
        return self.requests_this_hour >= self.max_requests_per_hour

    def to_dict(self) -> dict[str, Any]:
        return {
            "link_id": self.link_id,
            "local": self.local_agent_id,
            "remote": self.remote_agent_id,
            "status": self.status.value,
            "allowed_capabilities": [c.value for c in self.allowed_capabilities],
            "rate_limited": self.rate_limited,
        }


class FederationManager:
    """Verwaltung von Agent-Verbünden (Federation)."""

    def __init__(self) -> None:
        self._links: dict[str, FederationLink] = {}

    def propose(
        self,
        local_agent_id: str,
        remote_agent_id: str,
        remote_endpoint: str,
        allowed_capabilities: list[CapabilityType] | None = None,
    ) -> FederationLink:
        link_id = hashlib.sha256(
            f"fed:{local_agent_id}:{remote_agent_id}:{time.time()}".encode()
        ).hexdigest()[:12]
        link = FederationLink(
            link_id=link_id,
            local_agent_id=local_agent_id,
            remote_agent_id=remote_agent_id,
            remote_endpoint=remote_endpoint,
            allowed_capabilities=allowed_capabilities or list(CapabilityType),
        )
        self._links[link_id] = link
        return link

    def accept(self, link_id: str) -> FederationLink | None:
        link = self._links.get(link_id)
        if link and link.status == FederationStatus.PENDING:
            link.status = FederationStatus.ACTIVE
            link.established_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            return link
        return None

    def suspend(self, link_id: str) -> bool:
        link = self._links.get(link_id)
        if link:
            link.status = FederationStatus.SUSPENDED
            return True
        return False

    def revoke(self, link_id: str) -> bool:
        link = self._links.get(link_id)
        if link:
            link.status = FederationStatus.REVOKED
            return True
        return False

    def active_links(self) -> list[FederationLink]:
        return [l for l in self._links.values() if l.status == FederationStatus.ACTIVE]

    def get_link(self, link_id: str) -> FederationLink | None:
        return self._links.get(link_id)

    def links_for_agent(self, agent_id: str) -> list[FederationLink]:
        return [
            l for l in self._links.values()
            if l.local_agent_id == agent_id or l.remote_agent_id == agent_id
        ]

    def can_delegate(self, link_id: str, capability: CapabilityType) -> bool:
        """Prüft ob eine Capability über einen Federation-Link delegiert werden darf."""
        link = self._links.get(link_id)
        if not link or link.status != FederationStatus.ACTIVE:
            return False
        if link.rate_limited:
            return False
        if capability not in link.allowed_capabilities:
            return False
        link.requests_this_hour += 1
        return True

    @property
    def link_count(self) -> int:
        return len(self._links)

    def stats(self) -> dict[str, Any]:
        links = list(self._links.values())
        return {
            "total_links": len(links),
            "active": sum(1 for l in links if l.status == FederationStatus.ACTIVE),
            "pending": sum(1 for l in links if l.status == FederationStatus.PENDING),
            "suspended": sum(1 for l in links if l.status == FederationStatus.SUSPENDED),
            "revoked": sum(1 for l in links if l.status == FederationStatus.REVOKED),
        }


# ============================================================================
# Interop Protocol (Hauptklasse)
# ============================================================================


class InteropProtocol:
    """Hauptklasse: Orchestriert Cross-Agent-Kommunikation.

    Bringt alles zusammen:
      - Agent-Registrierung
      - Capability-Discovery
      - Message-Routing
      - Federation-Management
    """

    def __init__(self, local_agent_id: str = "jarvis-local") -> None:
        self._local_id = local_agent_id
        self._agents: dict[str, AgentIdentity] = {}
        self._capabilities = CapabilityRegistry()
        self._router = MessageRouter()
        self._federation = FederationManager()

    @property
    def local_agent_id(self) -> str:
        return self._local_id

    @property
    def capabilities(self) -> CapabilityRegistry:
        return self._capabilities

    @property
    def router(self) -> MessageRouter:
        return self._router

    @property
    def federation(self) -> FederationManager:
        return self._federation

    def register_agent(
        self,
        identity: AgentIdentity,
        capabilities: list[AgentCapability] | None = None,
        handler: Callable[[InteropMessage], Any] | None = None,
    ) -> None:
        identity.registered_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        identity.last_seen = identity.registered_at
        self._agents[identity.agent_id] = identity
        if capabilities:
            self._capabilities.register(identity.agent_id, capabilities)
        if handler:
            self._router.register_handler(identity.agent_id, handler)

    def unregister_agent(self, agent_id: str) -> bool:
        if agent_id in self._agents:
            del self._agents[agent_id]
            self._capabilities.unregister(agent_id)
            self._router.unregister(agent_id)
            return True
        return False

    def get_agent(self, agent_id: str) -> AgentIdentity | None:
        return self._agents.get(agent_id)

    def online_agents(self) -> list[AgentIdentity]:
        return [a for a in self._agents.values() if a.status == "online"]

    def delegate_task(
        self,
        capability_type: CapabilityType,
        payload: dict[str, Any],
        *,
        prefer_fast: bool = True,
    ) -> dict[str, Any]:
        """Delegiert eine Aufgabe an den besten verfügbaren Agenten."""
        agent_id = self._capabilities.best_agent_for(capability_type, prefer_fast=prefer_fast)
        if not agent_id:
            return {"success": False, "error": f"Kein Agent für {capability_type.value} verfügbar"}

        msg = self._router.create_message(
            msg_type=MessageType.TASK_DELEGATE,
            sender_id=self._local_id,
            receiver_id=agent_id,
            payload=payload,
            priority=MessagePriority.NORMAL,
        )
        result = self._router.send(msg)
        return {"success": result.get("delivered", False), "agent_id": agent_id, "result": result}

    def broadcast(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Sendet eine Broadcast-Nachricht an alle Agenten."""
        msg = self._router.create_message(
            msg_type=MessageType.BROADCAST,
            sender_id=self._local_id,
            payload=payload,
        )
        return self._router.send(msg)

    @property
    def agent_count(self) -> int:
        return len(self._agents)

    def stats(self) -> dict[str, Any]:
        return {
            "local_agent": self._local_id,
            "registered_agents": len(self._agents),
            "online": sum(1 for a in self._agents.values() if a.status == "online"),
            "capabilities": self._capabilities.stats(),
            "router": self._router.stats(),
            "federation": self._federation.stats(),
        }
