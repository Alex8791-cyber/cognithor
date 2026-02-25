"""Jarvis · A2A Protocol -- RC v1.0 (Linux Foundation).

Agent-zu-Agent-Kommunikation nach offenem Standard.
Ersetzt proprietäres JAIP-Protokoll durch Linux-Foundation-konformes A2A.

Module:
  types        -- Datentypen (Task, Message, AgentCard, Parts, Events)
  server       -- JSON-RPC 2.0 Server (empfängt Remote-Tasks)
  client       -- JSON-RPC 2.0 Client (sendet Tasks an Remote-Agenten)
  adapter      -- Brücke JAIP↔A2A + Gateway-Integration
  http_handler -- HTTP-Transport (FastAPI-Routes)

OPTIONAL: Nur aktiv wenn in config aktiviert. Kein Import-Fehler wenn deaktiviert.
"""

from __future__ import annotations

# Types -- always available (no external deps)
from jarvis.a2a.types import (
    A2A_CONTENT_TYPE,
    A2A_PROTOCOL_VERSION,
    A2A_VERSION_HEADER,
    A2AAgentCapabilities,
    A2AAgentCard,
    A2AErrorCode,
    A2AInterface,
    A2AProvider,
    A2ASecurityScheme,
    A2ASkill,
    Artifact,
    DataPart,
    FilePart,
    Message,
    MessageRole,
    Part,
    PartType,
    PushNotificationAuth,
    PushNotificationConfig,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
    VALID_TRANSITIONS,
    is_valid_transition,
    part_from_dict,
)

# Server + Client -- always available
from jarvis.a2a.server import A2AServer, A2AServerConfig
from jarvis.a2a.client import A2AClient, RemoteAgent

# Adapter -- always available
from jarvis.a2a.adapter import A2AAdapter

__all__ = [
    # Types
    "A2A_CONTENT_TYPE", "A2A_PROTOCOL_VERSION", "A2A_VERSION_HEADER",
    "A2AAgentCapabilities", "A2AAgentCard", "A2AErrorCode", "A2AInterface",
    "A2AProvider", "A2ASecurityScheme", "A2ASkill", "Artifact", "DataPart",
    "FilePart", "Message", "MessageRole", "Part", "PartType",
    "PushNotificationAuth", "PushNotificationConfig",
    "Task", "TaskArtifactUpdateEvent", "TaskState", "TaskStatus",
    "TaskStatusUpdateEvent", "TextPart", "VALID_TRANSITIONS",
    "is_valid_transition", "part_from_dict",
    # Server
    "A2AServer", "A2AServerConfig",
    # Client
    "A2AClient", "RemoteAgent",
    # Adapter
    "A2AAdapter",
]
