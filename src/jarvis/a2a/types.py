"""A2A Protocol Types -- RC v1.0 (Linux Foundation).

Standardisierte Datentypen für Agent-zu-Agent-Kommunikation.
Neu in RC v1.0: contextId, messageId, AgentInterface, streaming events,
push notification CRUD, A2A-Version header, application/a2a+json.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Constants ────────────────────────────────────────────────────

A2A_PROTOCOL_VERSION = "1.0"
A2A_CONTENT_TYPE = "application/a2a+json"
A2A_VERSION_HEADER = "A2A-Version"


# ── Parts ────────────────────────────────────────────────────────

class PartType(str, Enum):
    TEXT = "text"
    FILE = "file"
    DATA = "data"


@dataclass
class TextPart:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    type: str = "text"

    def to_dict(self) -> dict[str, Any]:
        r: dict[str, Any] = {"type": self.type, "text": self.text}
        if self.metadata:
            r["metadata"] = self.metadata
        return r


@dataclass
class FilePart:
    name: str = ""
    mime_type: str = "application/octet-stream"
    uri: str = ""
    data: str = ""  # base64
    type: str = "file"

    def to_dict(self) -> dict[str, Any]:
        r: dict[str, Any] = {
            "type": self.type,
            "file": {"name": self.name, "mimeType": self.mime_type},
        }
        if self.uri:
            r["file"]["uri"] = self.uri
        if self.data:
            r["file"]["bytes"] = self.data
        return r


@dataclass
class DataPart:
    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    type: str = "data"

    def to_dict(self) -> dict[str, Any]:
        r: dict[str, Any] = {"type": self.type, "data": self.data}
        if self.metadata:
            r["metadata"] = self.metadata
        return r


Part = TextPart | FilePart | DataPart


def part_from_dict(p: dict[str, Any]) -> Part:
    ptype = p.get("type", "text")
    if ptype == "text":
        return TextPart(text=p.get("text", ""), metadata=p.get("metadata", {}))
    elif ptype == "file":
        f = p.get("file", {})
        return FilePart(
            name=f.get("name", ""), mime_type=f.get("mimeType", "application/octet-stream"),
            uri=f.get("uri", ""), data=f.get("bytes", ""),
        )
    elif ptype == "data":
        return DataPart(data=p.get("data", {}), metadata=p.get("metadata", {}))
    return TextPart(text=str(p))


# ── Task States ──────────────────────────────────────────────────

class TaskState(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    REJECTED = "rejected"
    AUTH_REQUIRED = "auth-required"

    @property
    def is_terminal(self) -> bool:
        return self in (TaskState.COMPLETED, TaskState.FAILED,
                        TaskState.CANCELED, TaskState.REJECTED)

    @property
    def is_active(self) -> bool:
        return self in (TaskState.SUBMITTED, TaskState.WORKING,
                        TaskState.INPUT_REQUIRED, TaskState.AUTH_REQUIRED)


VALID_TRANSITIONS: dict[TaskState, set[TaskState]] = {
    TaskState.SUBMITTED: {TaskState.WORKING, TaskState.FAILED, TaskState.CANCELED,
                          TaskState.REJECTED, TaskState.AUTH_REQUIRED},
    TaskState.WORKING: {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED,
                        TaskState.INPUT_REQUIRED, TaskState.REJECTED},
    TaskState.INPUT_REQUIRED: {TaskState.WORKING, TaskState.FAILED, TaskState.CANCELED},
    TaskState.AUTH_REQUIRED: {TaskState.WORKING, TaskState.FAILED, TaskState.CANCELED},
    TaskState.COMPLETED: set(),
    TaskState.FAILED: set(),
    TaskState.CANCELED: set(),
    TaskState.REJECTED: set(),
}


def is_valid_transition(from_state: TaskState, to_state: TaskState) -> bool:
    return to_state in VALID_TRANSITIONS.get(from_state, set())


# ── Messages ─────────────────────────────────────────────────────

class MessageRole(str, Enum):
    USER = "user"
    AGENT = "agent"


@dataclass
class Message:
    role: MessageRole
    parts: list[Part] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    message_id: str = ""
    context_id: str = ""
    task_id: str = ""
    extensions: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.message_id:
            self.message_id = uuid.uuid4().hex[:16]

    def to_dict(self) -> dict[str, Any]:
        r: dict[str, Any] = {
            "role": self.role.value,
            "parts": [p.to_dict() for p in self.parts],
            "messageId": self.message_id,
        }
        if self.context_id:
            r["contextId"] = self.context_id
        if self.task_id:
            r["taskId"] = self.task_id
        if self.metadata:
            r["metadata"] = self.metadata
        if self.extensions:
            r["extensions"] = self.extensions
        return r

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        return cls(
            role=MessageRole(data.get("role", "user")),
            parts=[part_from_dict(p) for p in data.get("parts", [])],
            metadata=data.get("metadata", {}),
            message_id=data.get("messageId", ""),
            context_id=data.get("contextId", ""),
            task_id=data.get("taskId", ""),
            extensions=data.get("extensions", {}),
        )

    @property
    def text(self) -> str:
        for part in self.parts:
            if isinstance(part, TextPart):
                return part.text
        return ""


# ── Artifacts ────────────────────────────────────────────────────

@dataclass
class Artifact:
    parts: list[Part] = field(default_factory=list)
    artifact_id: str = ""
    name: str = ""
    description: str = ""
    index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.artifact_id:
            self.artifact_id = uuid.uuid4().hex[:16]

    def to_dict(self) -> dict[str, Any]:
        r: dict[str, Any] = {
            "artifactId": self.artifact_id,
            "parts": [p.to_dict() for p in self.parts],
            "index": self.index,
        }
        if self.name:
            r["name"] = self.name
        if self.description:
            r["description"] = self.description
        if self.metadata:
            r["metadata"] = self.metadata
        return r

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Artifact:
        return cls(
            parts=[part_from_dict(p) for p in data.get("parts", [])],
            artifact_id=data.get("artifactId", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            index=data.get("index", 0),
            metadata=data.get("metadata", {}),
        )


# ── Task ─────────────────────────────────────────────────────────

@dataclass
class TaskStatus:
    state: TaskState
    message: Message | None = None
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def to_dict(self) -> dict[str, Any]:
        r: dict[str, Any] = {"state": self.state.value, "timestamp": self.timestamp}
        if self.message:
            r["message"] = self.message.to_dict()
        return r


@dataclass
class Task:
    id: str
    context_id: str
    status: TaskStatus
    messages: list[Message] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    history: list[TaskStatus] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        r: dict[str, Any] = {
            "id": self.id, "contextId": self.context_id,
            "status": self.status.to_dict(),
        }
        if self.messages:
            r["messages"] = [m.to_dict() for m in self.messages]
        if self.artifacts:
            r["artifacts"] = [a.to_dict() for a in self.artifacts]
        if self.metadata:
            r["metadata"] = self.metadata
        if self.history:
            r["history"] = [h.to_dict() for h in self.history]
        return r

    def transition(self, new_state: TaskState, message: Message | None = None) -> bool:
        if not is_valid_transition(self.status.state, new_state):
            return False
        self.history.append(self.status)
        self.status = TaskStatus(state=new_state, message=message)
        return True

    @property
    def state(self) -> TaskState:
        return self.status.state

    @property
    def is_complete(self) -> bool:
        return self.status.state.is_terminal

    @property
    def is_active(self) -> bool:
        return self.status.state.is_active

    @classmethod
    def create(cls, task_id: str | None = None, context_id: str | None = None,
               message: Message | None = None) -> Task:
        task = cls(
            id=task_id or uuid.uuid4().hex[:16],
            context_id=context_id or uuid.uuid4().hex[:16],
            status=TaskStatus(state=TaskState.SUBMITTED),
        )
        if message:
            task.messages.append(message)
        return task


# ── Streaming Events (RC v1.0) ───────────────────────────────────

@dataclass
class TaskStatusUpdateEvent:
    task_id: str
    context_id: str
    status: TaskStatus
    final: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "taskId": self.task_id, "contextId": self.context_id,
            "status": self.status.to_dict(), "final": self.final,
        }

    def to_sse(self) -> str:
        import json
        return f"event: status\ndata: {json.dumps(self.to_dict())}\n\n"


@dataclass
class TaskArtifactUpdateEvent:
    task_id: str
    context_id: str
    artifact: Artifact
    last_chunk: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "taskId": self.task_id, "contextId": self.context_id,
            "artifact": self.artifact.to_dict(), "lastChunk": self.last_chunk,
        }

    def to_sse(self) -> str:
        import json
        return f"event: artifact\ndata: {json.dumps(self.to_dict())}\n\n"


# ── Push Notification Config (RC v1.0) ───────────────────────────

@dataclass
class PushNotificationAuth:
    type: str = "bearer"
    credentials: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "credentials": self.credentials}


@dataclass
class PushNotificationConfig:
    task_id: str
    url: str
    config_id: str = ""
    authentication: PushNotificationAuth | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.config_id:
            self.config_id = uuid.uuid4().hex[:12]

    def to_dict(self) -> dict[str, Any]:
        r: dict[str, Any] = {
            "taskId": self.task_id, "url": self.url, "configId": self.config_id,
        }
        if self.authentication:
            r["authentication"] = self.authentication.to_dict()
        if self.metadata:
            r["metadata"] = self.metadata
        return r


# ── Agent Card (RC v1.0) ─────────────────────────────────────────

@dataclass
class A2ASkill:
    id: str
    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        r: dict[str, Any] = {"id": self.id, "name": self.name, "description": self.description}
        if self.tags:
            r["tags"] = self.tags
        if self.examples:
            r["examples"] = self.examples
        return r


@dataclass
class A2AProvider:
    organization: str = "Jarvis"
    url: str = ""

    def to_dict(self) -> dict[str, Any]:
        r: dict[str, Any] = {"organization": self.organization}
        if self.url:
            r["url"] = self.url
        return r


@dataclass
class A2AAgentCapabilities:
    streaming: bool = False
    push_notifications: bool = False
    state_transition_history: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "streaming": self.streaming,
            "pushNotifications": self.push_notifications,
            "stateTransitionHistory": self.state_transition_history,
        }


@dataclass
class A2AInterface:
    protocol: str = "jsonrpc"  # jsonrpc, rest, grpc
    url: str = ""
    content_type: str = A2A_CONTENT_TYPE

    def to_dict(self) -> dict[str, Any]:
        return {"protocol": self.protocol, "url": self.url, "contentType": self.content_type}


@dataclass
class A2ASecurityScheme:
    type: str = "http"
    scheme: str = "bearer"
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        r: dict[str, Any] = {"type": self.type, "scheme": self.scheme}
        if self.description:
            r["description"] = self.description
        return r


@dataclass
class A2AAgentCard:
    name: str = "Jarvis"
    description: str = ""
    url: str = ""
    version: str = "16.0.0"
    protocol_version: str = A2A_PROTOCOL_VERSION
    provider: A2AProvider = field(default_factory=A2AProvider)
    capabilities: A2AAgentCapabilities = field(default_factory=A2AAgentCapabilities)
    skills: list[A2ASkill] = field(default_factory=list)
    interfaces: list[A2AInterface] = field(default_factory=list)
    security_schemes: list[A2ASecurityScheme] = field(default_factory=list)
    default_input_modes: list[str] = field(default_factory=lambda: ["text/plain", "application/json"])
    default_output_modes: list[str] = field(default_factory=lambda: ["text/plain", "application/json"])
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        card: dict[str, Any] = {
            "name": self.name, "description": self.description,
            "version": self.version, "protocolVersion": self.protocol_version,
            "provider": self.provider.to_dict(),
            "capabilities": self.capabilities.to_dict(),
        }
        if self.url:
            card["url"] = self.url
        if self.skills:
            card["skills"] = [s.to_dict() for s in self.skills]
        if self.interfaces:
            card["interfaces"] = [i.to_dict() for i in self.interfaces]
        if self.security_schemes:
            card["securitySchemes"] = {f"scheme_{i}": s.to_dict() for i, s in enumerate(self.security_schemes)}
        card["defaultInputModes"] = self.default_input_modes
        card["defaultOutputModes"] = self.default_output_modes
        if self.tags:
            card["tags"] = self.tags
        return card

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> A2AAgentCard:
        skills = [A2ASkill(id=s.get("id", ""), name=s.get("name", ""),
                           description=s.get("description", ""), tags=s.get("tags", []),
                           examples=s.get("examples", [])) for s in data.get("skills", [])]
        interfaces = [A2AInterface(protocol=i.get("protocol", "jsonrpc"), url=i.get("url", ""),
                                   content_type=i.get("contentType", A2A_CONTENT_TYPE))
                      for i in data.get("interfaces", [])]
        prov = data.get("provider", {})
        cap = data.get("capabilities", {})
        return cls(
            name=data.get("name", "Unknown"), description=data.get("description", ""),
            url=data.get("url", ""), version=data.get("version", ""),
            protocol_version=data.get("protocolVersion", A2A_PROTOCOL_VERSION),
            provider=A2AProvider(organization=prov.get("organization", ""), url=prov.get("url", "")),
            capabilities=A2AAgentCapabilities(
                streaming=cap.get("streaming", False),
                push_notifications=cap.get("pushNotifications", False),
                state_transition_history=cap.get("stateTransitionHistory", True),
            ),
            skills=skills, interfaces=interfaces,
            default_input_modes=data.get("defaultInputModes", ["text/plain"]),
            default_output_modes=data.get("defaultOutputModes", ["text/plain"]),
            tags=data.get("tags", []),
        )


# ── A2A Error Codes (RC v1.0) ────────────────────────────────────

class A2AErrorCode:
    """Standard A2A + JSON-RPC error codes."""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    # A2A-specific
    TASK_NOT_FOUND = -32001
    TASK_NOT_CANCELABLE = -32002
    PUSH_NOT_SUPPORTED = -32003
    UNAUTHORIZED = -32004
    INCOMPATIBLE_VERSION = -32005
    CONTENT_TYPE_NOT_SUPPORTED = -32006
