"""
Jarvis · Central data models.

All Pydantic models used across modules.
Architecture Bible: §3.1, §3.2, §3.3, §4, §5, §6, §8, §9, §10

Design principles:
  - Immutable (frozen) where sensible (Audit, Decisions, Messages)
  - Mutable where necessary (WorkingMemory, SessionContext)
  - Strict validation (no invalid state possible)
  - JSON-serializable (for logging, persistence, MCP transport)
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

# ============================================================================
# Hilfsfunktionen
# ============================================================================


def _utc_now() -> datetime:
    """Aktuelle Zeit in UTC. Einheitlich im gesamten System."""
    return datetime.now(UTC)


def _new_id() -> str:
    """Neue UUID als String. Für alle IDs im System."""
    return uuid.uuid4().hex


# ============================================================================
# Enums
# ============================================================================


class RiskLevel(StrEnum):
    """Risk level of a planned action. [B§3.2]

    GREEN:  Execute automatically (read memory, search, computation)
    YELLOW: Execute + inform user (create file, appointment)
    ORANGE: User must confirm (send email, delete file)
    RED:    Blocked (system commands, credentials, unknown hosts)
    """

    GREEN = "green"
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"


class GateStatus(StrEnum):
    """Gatekeeper decision. [B§3.2]

    ALLOW:   Execute immediately.
    INFORM:  Execute, user is informed.
    APPROVE: Wait for user confirmation.
    BLOCK:   Do not execute. Never.
    MASK:    Execute, but mask credential values.
    """

    ALLOW = "ALLOW"
    INFORM = "INFORM"
    APPROVE = "APPROVE"
    BLOCK = "BLOCK"
    MASK = "MASK"


class MessageRole(StrEnum):
    """Role of a message in the chat history."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class MemoryTier(StrEnum):
    """The 5 memory tiers. [B§4.1]"""

    CORE = "core"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"


class OperationMode(StrEnum):
    """Operating mode of the system.

    OFFLINE:  Local LLM (Ollama), no cloud LLM.
              Web research (web_search, web_fetch) remains allowed.
    ONLINE:   Cloud LLM (OpenAI/Anthropic) + all network tools.
    HYBRID:   Local LLM + selective network tools.
    """

    OFFLINE = "offline"
    ONLINE = "online"
    HYBRID = "hybrid"


class SandboxLevel(StrEnum):
    """Isolations-Stufe der Sandbox. [B§3.3]"""

    PROCESS = "process"
    NAMESPACE = "namespace"
    CONTAINER = "container"
    JOBOBJECT = "jobobject"  # Windows Job Objects


# ============================================================================
# Nachrichten (E3: Alles ist eine Message)
# ============================================================================


class Message(BaseModel, frozen=True):
    """A single message in the chat history.

    Used everywhere: chat history, tool results, system prompts.
    Immutable -- once created, not modifiable.
    """

    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=_utc_now)
    name: str | None = None  # Tool-Name bei role=TOOL
    tool_call_id: str | None = None  # Referenz auf den Tool-Call
    channel: str | None = None  # Herkunfts-Channel (cli, telegram, …)


class IncomingMessage(BaseModel, frozen=True):
    """Nachricht die vom Channel hereinkommt. [B§9.1]"""

    id: str = Field(default_factory=_new_id)
    channel: str  # "cli", "telegram", "webui", "api", "voice"
    user_id: str
    text: str
    attachments: list[str] = Field(default_factory=list)
    session_id: str | None = None  # Optional: existierende Session fortführen
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=_utc_now)


class OutgoingMessage(BaseModel, frozen=True):
    """Antwort die an den Channel zurückgeht. [B§9.1]"""

    channel: str
    text: str
    session_id: str = ""  # Session-Referenz
    is_final: bool = False  # True = letzte Nachricht dieser Anfrage
    reply_to: str | None = None  # ID der IncomingMessage
    attachments: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=_utc_now)


# ============================================================================
# PGE-Trinität: Planner → Gatekeeper → Executor [B§3]
# ============================================================================


class PlannedAction(BaseModel, frozen=True):
    """Ein einzelner Schritt im Plan. [B§3.1]

    Der Planner erstellt diese. Der Gatekeeper prüft jede einzeln.
    Der Executor führt nur genehmigte aus.
    """

    tool: str  # MCP-Tool-Name (z.B. "read_file", "exec_command")
    params: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""
    depends_on: list[int] = Field(default_factory=list)
    risk_estimate: RiskLevel = RiskLevel.GREEN
    rollback: str | None = None


class ActionPlan(BaseModel, frozen=True):
    """Was der Planner dem Gatekeeper übergibt. [B§3.1]

    Enthält das Ziel, die Begründung, und eine geordnete Liste von Schritten.
    Kann auch eine direkte Antwort ohne Schritte enthalten (einfache Fragen).
    """

    goal: str
    reasoning: str = ""
    steps: list[PlannedAction] = Field(default_factory=list)
    direct_response: str | None = None
    memory_context: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    iteration: int = Field(default=0, ge=0)

    @property
    def is_direct_response(self) -> bool:
        """True wenn der Planner direkt antwortet ohne Tools."""
        return self.direct_response is not None and len(self.steps) == 0

    @property
    def requires_tools(self) -> bool:
        """True wenn der Plan Tool-Calls enthält."""
        return len(self.steps) > 0

    @property
    def has_actions(self) -> bool:
        """Alias für requires_tools – True wenn Steps vorhanden."""
        return len(self.steps) > 0


class GateDecision(BaseModel, frozen=True):
    """Entscheidung des Gatekeepers für eine einzelne Aktion. [B§3.2]

    Immutable – einmal getroffen, nie änderbar.
    """

    status: GateStatus
    risk_level: RiskLevel = RiskLevel.GREEN
    reason: str = ""
    policy_name: str = ""  # Name der auslösenden Policy-Regel
    original_action: PlannedAction | None = None  # Referenz auf geprüfte Aktion
    masked_params: dict[str, Any] | None = None  # Params nach Credential-Maskierung
    timestamp: datetime = Field(default_factory=_utc_now)

    @property
    def matched_policy(self) -> str | None:
        """Alias für policy_name (Rückwärtskompatibilität)."""
        return self.policy_name or None

    @property
    def is_allowed(self) -> bool:
        """True wenn die Aktion ausgeführt werden darf."""
        return self.status in (GateStatus.ALLOW, GateStatus.INFORM, GateStatus.MASK)

    @property
    def needs_approval(self) -> bool:
        """True wenn User-Bestätigung erforderlich ist."""
        return self.status == GateStatus.APPROVE

    @property
    def is_blocked(self) -> bool:
        """True wenn die Aktion blockiert ist."""
        return self.status == GateStatus.BLOCK


class ToolResult(BaseModel, frozen=True):
    """Ergebnis einer Tool-Ausführung. [B§3.3]"""

    tool_name: str
    content: str = ""
    is_error: bool = False
    error_message: str | None = None
    error_type: str | None = None  # Fehler-Klasse (z.B. TimeoutError, GatekeeperBlock)
    duration_ms: int = 0
    truncated: bool = False
    timestamp: datetime = Field(default_factory=_utc_now)

    @property
    def success(self) -> bool:
        """True wenn die Ausführung erfolgreich war."""
        return not self.is_error


class AuditEntry(BaseModel, frozen=True):
    """Unveränderlicher Audit-Log-Eintrag. [B§3.2]

    Jede Gatekeeper-Entscheidung wird protokolliert.
    JSONL in gatekeeper.jsonl. Nur Append, nie geändert.
    Flache Struktur für performante Log-Analyse.
    """

    id: str = Field(default_factory=_new_id)
    timestamp: datetime = Field(default_factory=_utc_now)
    session_id: str
    # Flache Action-Felder (statt geschachteltem PlannedAction)
    action_tool: str
    action_params_hash: str  # SHA-256 Hash der Params (keine Klartexte im Log)
    # Flache Decision-Felder (statt geschachteltem GateDecision)
    decision_status: GateStatus
    decision_reason: str = ""
    risk_level: RiskLevel = RiskLevel.GREEN
    policy_name: str = ""
    # Optional
    user_override: bool = False
    execution_result: str | None = None
    error: str | None = None


# ============================================================================
# Sandbox [B§3.3]
# ============================================================================


class SandboxConfig(BaseModel):
    """Konfiguration der Ausführungs-Sandbox. [B§3.3]"""

    level: SandboxLevel = SandboxLevel.PROCESS
    timeout_seconds: int = Field(default=30, ge=1, le=600)
    max_memory_mb: int = Field(default=512, ge=64, le=8192)
    max_cpu_seconds: int = Field(default=10, ge=1, le=300)
    allowed_paths: list[str] = Field(
        default_factory=lambda: ["~/.jarvis/workspace/", "/tmp/jarvis/"]
    )
    network_access: bool = False
    env_vars: dict[str, str] = Field(default_factory=dict)


# ============================================================================
# Session & Kontext [B§9]
# ============================================================================


class SessionContext(BaseModel):
    """Kontext einer laufenden Sitzung. [B§9.1]

    Mutable – wird während der Session aktualisiert.
    Jede Session gehört zu genau einem Agenten. Verschiedene Agenten
    haben getrennte Sessions und Working Memories.
    """

    session_id: str = Field(default_factory=_new_id)
    user_id: str = "default"
    channel: str = "cli"
    agent_name: str = "jarvis"  # Zugeordneter Agent
    started_at: datetime = Field(default_factory=_utc_now)
    last_activity: datetime = Field(default_factory=_utc_now)
    message_count: int = 0
    plan_iterations: int = 0
    active: bool = True
    # Agent-Loop Steuerung [B§3.4]
    max_iterations: int = 10
    iteration_count: int = 0
    _blocked_tools: dict[str, int] = {}  # Tool → Block-Counter (private, set in __init__)

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data: Any) -> None:
        """Initialisiert die SessionContext mit User- und System-Nachrichten."""
        super().__init__(**data)
        object.__setattr__(self, "_blocked_tools", {})

    def touch(self) -> None:
        """Aktualisiert Zeitstempel und Zähler."""
        self.last_activity = _utc_now()
        self.message_count += 1

    def reset_iteration(self) -> None:
        """Setzt den Iterations-Zähler zurück (neue Anfrage)."""
        self.iteration_count = 0
        object.__setattr__(self, "_blocked_tools", {})

    def record_block(self, tool: str) -> int:
        """Zählt einen Block für ein Tool und gibt den neuen Zählerstand zurück."""
        blocked = self._blocked_tools
        blocked[tool] = blocked.get(tool, 0) + 1
        return blocked[tool]

    @property
    def iterations_exhausted(self) -> bool:
        """True wenn das Iterationslimit erreicht ist."""
        return self.iteration_count >= self.max_iterations


# ============================================================================
# Kognitives Memory [B§4]
# ============================================================================


class Chunk(BaseModel, frozen=True):
    """Ein Text-Chunk für den Memory-Index. [B§4.8]"""

    id: str = Field(default_factory=_new_id)
    text: str
    source_path: str
    line_start: int = 0
    line_end: int = 0
    content_hash: str = ""  # SHA-256 für Embedding-Cache
    memory_tier: MemoryTier = MemoryTier.SEMANTIC
    entities: list[str] = Field(default_factory=list)
    timestamp: datetime | None = None
    token_count: int = 0


class Entity(BaseModel):
    """Eine Entität im Wissens-Graphen. [B§4.4]"""

    id: str = Field(default_factory=_new_id)
    type: str  # "person", "company", "product", "project"
    name: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    source_file: str = ""
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class Relation(BaseModel):
    """Eine Beziehung zwischen zwei Entitäten. [B§4.4]"""

    id: str = Field(default_factory=_new_id)
    source_entity: str  # Entity-ID
    relation_type: str  # "hat_police", "arbeitet_bei", ...
    target_entity: str  # Entity-ID
    attributes: dict[str, Any] = Field(default_factory=dict)
    source_file: str = ""
    created_at: datetime = Field(default_factory=_utc_now)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ProcedureMetadata(BaseModel):
    """Metadaten einer gelernten Prozedur. [B§6.3]"""

    name: str
    trigger_keywords: list[str] = Field(default_factory=list)
    tools_required: list[str] = Field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    total_uses: int = 0
    avg_score: float = Field(default=0.0, ge=0.0, le=1.0)
    last_used: datetime | None = None
    learned_from: list[str] = Field(default_factory=list)
    failure_patterns: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    source_file: str = ""

    @property
    def success_rate(self) -> float:
        """Berechnet die Erfolgsrate (0.0–1.0)."""
        if self.total_uses == 0:
            return 0.0
        return self.success_count / self.total_uses

    @property
    def is_reliable(self) -> bool:
        """10+ Nutzungen, >80% Erfolg. [B§6.3]"""
        return self.total_uses >= 10 and self.success_rate > 0.8

    @property
    def needs_review(self) -> bool:
        """5+ Fehler in Folge, <30% Erfolg."""
        return self.failure_count >= 5 and self.success_rate < 0.3


class MemorySearchResult(BaseModel, frozen=True):
    """Ergebnis einer Hybrid-Suche. [B§4.7]"""

    chunk: Chunk
    score: float = 0.0
    bm25_score: float = 0.0
    vector_score: float = 0.0
    graph_score: float = 0.0
    recency_factor: float = 1.0


class WorkingMemory(BaseModel):
    """Aktiver Session-Kontext. Tier 5. Flüchtig. [B§4.6]

    Lebt nur im RAM. Pre-Compaction Flush bei >80%.
    """

    session_id: str = Field(default_factory=_new_id)
    chat_history: list[Message] = Field(default_factory=list)
    active_plan: ActionPlan | None = None
    tool_results: list[ToolResult] = Field(default_factory=list)
    injected_memories: list[MemorySearchResult] = Field(default_factory=list)
    injected_procedures: list[str] = Field(default_factory=list)  # Relevante Prozeduren als Text
    core_memory_text: str = ""  # CORE.md Inhalt
    token_count: int = 0
    max_tokens: int = 32768  # Qwen3-32B Default

    @property
    def usage_ratio(self) -> float:
        """Wie viel Prozent des Token-Budgets verbraucht sind."""
        if self.max_tokens == 0:
            return 1.0
        return self.token_count / self.max_tokens

    @property
    def needs_compaction(self) -> bool:
        """True wenn Pre-Compaction Flush nötig ist (>80%). [B§4.6]"""
        return self.usage_ratio > 0.80

    def add_message(self, msg: Message) -> None:
        """Fügt eine Nachricht zum Session-Kontext hinzu."""
        self.chat_history.append(msg)

    def add_tool_result(self, result: ToolResult) -> None:
        """Fügt ein Tool-Ergebnis zum Session-Kontext hinzu."""
        self.tool_results.append(result)

    def clear_for_compaction(self, keep_last_n: int = 4) -> list[Message]:
        """Entfernt alte Messages, gibt entfernte zurück.

        System-Messages bleiben immer. Behält die letzten N nicht-System-Messages.

        Returns:
            Entfernte Messages (für Zusammenfassung durch Reflector).
        """
        if len(self.chat_history) <= keep_last_n:
            return []

        system_msgs = [m for m in self.chat_history if m.role == MessageRole.SYSTEM]
        non_system = [m for m in self.chat_history if m.role != MessageRole.SYSTEM]

        if len(non_system) <= keep_last_n:
            return []

        removed = non_system[:-keep_last_n]
        kept = non_system[-keep_last_n:]
        self.chat_history = system_msgs + kept
        self.tool_results = []
        return removed

    def clear_for_new_request(self) -> None:
        """Räumt Working Memory für eine neue Anfrage auf.

        Behält Chat-History und Core-Memory, löscht temporäre Daten.
        """
        self.tool_results = []
        self.active_plan = None
        self.injected_memories = []
        self.injected_procedures = []


# ============================================================================
# Model-Router [B§8]
# ============================================================================


class ModelConfig(BaseModel):
    """Konfiguration eines LLM-Modells. [B§8.1]"""

    name: str  # Ollama-Modellname
    context_window: int = 32768
    vram_gb: float = 0.0
    strengths: list[str] = Field(default_factory=list)
    speed: Literal["fast", "medium", "slow"] = "medium"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    embedding_dimensions: int = 768  # Vektor-Dimensionen für Embedding-Modelle


# ============================================================================
# MCP [B§5]
# ============================================================================


class MCPServerConfig(BaseModel):
    """Konfiguration eines MCP-Servers. [B§5.4]"""

    transport: Literal["stdio", "http"] = "stdio"
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    sandbox: SandboxLevel = SandboxLevel.PROCESS
    enabled: bool = True


class MCPToolInfo(BaseModel, frozen=True):
    """Registriertes MCP-Tool. [B§5.2]"""

    name: str
    server: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Policy [B§3.2]
# ============================================================================


class PolicyParamMatch(BaseModel, frozen=True):
    """Bedingung für einen einzelnen Parameter. [B§3.2]

    Alle Felder sind optional – nur gesetzte werden geprüft.
    Alle müssen matchen (AND-Verknüpfung).
    """

    regex: str | None = None
    startswith: str | list[str] | None = None
    not_startswith: str | list[str] | None = None
    contains: str | list[str] | None = None
    contains_pattern: str | None = None
    equals: str | None = None


class PolicyMatch(BaseModel, frozen=True):
    """Match-Kriterium einer Policy-Regel. [B§3.2]

    tool: MCP-Tool-Name oder "*" für alle.
    params: Dict von Param-Name → PolicyParamMatch.
            Key "*" matcht gegen alle Params (Wildcard).
    """

    tool: str = "*"
    params: dict[str, PolicyParamMatch] = Field(default_factory=dict)


class PolicyRule(BaseModel, frozen=True):
    """Gatekeeper-Policy-Regel. [B§3.2]

    Matcht auf Tool-Name + Param-Bedingungen → GateStatus.
    """

    name: str
    match: PolicyMatch = Field(default_factory=PolicyMatch)
    action: GateStatus = GateStatus.BLOCK
    reason: str = ""
    priority: int = Field(default=0, ge=0)


# ============================================================================
# Cron [B§10]
# ============================================================================


class CronJob(BaseModel):
    """Ein geplanter Cron-Job. [B§10.1]

    Jobs können optional einem bestimmten Agenten zugeordnet werden.
    Ohne agent-Feld wird die Nachricht normal geroutet.
    """

    name: str
    schedule: str  # Cron-Expression
    prompt: str
    channel: str = "telegram"
    model: str = "qwen3:8b"
    enabled: bool = True
    agent: str = ""  # Ziel-Agent (leer = normales Routing)


# ============================================================================
# Reflexion & Prozedurales Lernen [B§6]
# ============================================================================


class ExtractedFact(BaseModel, frozen=True):
    """Ein aus einer Session extrahierter Fakt. [B§6.1]

    Wird vom Reflector erzeugt und ins Semantic Memory geschrieben.
    """

    entity_name: str
    entity_type: str = "unknown"
    attribute_key: str = ""
    attribute_value: str = ""
    relation_type: str | None = None
    relation_target: str | None = None
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    source_session: str = ""


class ProcedureCandidate(BaseModel, frozen=True):
    """Kandidat für eine neue oder aktualisierte Prozedur. [B§6.3]

    Der Reflector erkennt wiederholbare Muster und erzeugt diese.
    """

    name: str
    trigger_keywords: list[str] = Field(default_factory=list)
    prerequisite_text: str = ""
    steps_text: str = ""
    learned_text: str = ""
    failure_patterns: list[str] = Field(default_factory=list)
    tools_required: list[str] = Field(default_factory=list)
    is_update: bool = False  # True → existierende Prozedur verbessern


class SessionSummary(BaseModel, frozen=True):
    """Zusammenfassung einer abgeschlossenen Session. [B§6.1]

    Wird ins Episodic Memory als Tageslog-Eintrag geschrieben.
    """

    goal: str
    outcome: str
    key_decisions: list[str] = Field(default_factory=list)
    open_items: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    duration_ms: int = 0


class ReflectionResult(BaseModel):
    """Vollständiges Ergebnis einer Reflexion über eine Session. [B§6.1]

    Wird vom Reflector erzeugt nach Ende einer Session.
    Enthält Evaluation, extrahierte Fakten, Prozedur-Kandidaten und
    eine Session-Zusammenfassung für das Episodic Memory.
    """

    session_id: str
    success_score: float = Field(default=0.5, ge=0.0, le=1.0)
    evaluation: str = ""
    extracted_facts: list[ExtractedFact] = Field(default_factory=list)
    procedure_candidate: ProcedureCandidate | None = None
    session_summary: SessionSummary | None = None
    failure_analysis: str = ""
    improvement_suggestions: list[str] = Field(default_factory=list)
    reflected_at: datetime = Field(default_factory=_utc_now)

    @property
    def was_successful(self) -> bool:
        """Gilt als erfolgreich wenn Score >= 0.6."""
        return self.success_score >= 0.6

    @property
    def has_procedure(self) -> bool:
        """Ob ein wiederholbares Muster erkannt wurde."""
        return self.procedure_candidate is not None

    @property
    def has_facts(self) -> bool:
        """Ob neue Fakten extrahiert wurden."""
        return len(self.extracted_facts) > 0


# ============================================================================
# Agent-Ergebnis
# ============================================================================


class AgentResult(BaseModel):
    """Vollständiges Ergebnis eines Agent-Zyklus."""

    response: str
    plans: list[ActionPlan] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    audit_entries: list[AuditEntry] = Field(default_factory=list)
    total_iterations: int = 0
    total_duration_ms: int = 0
    model_used: str = ""
    success: bool = True
    error: str | None = None
    reflection: ReflectionResult | None = None


# ============================================================================
# Phase 5: Multi-Agent & Sicherheit [B§7, B§11]
# ============================================================================


class AgentType(StrEnum):
    """Spezialisierte Agent-Typen. [B§7.3]"""

    PLANNER = "planner"
    WORKER = "worker"
    CODER = "coder"
    RESEARCHER = "researcher"


class SubAgentConfig(BaseModel):
    """Konfiguration zum Spawnen eines Sub-Agents. [B§7.2]"""

    task: str
    agent_type: AgentType = AgentType.WORKER
    model: str = "qwen3:8b"
    tools: list[str] | None = None
    timeout_seconds: int = Field(default=300, ge=1, le=3600)
    max_iterations: int = Field(default=5, ge=1, le=20)
    depth: int = Field(default=0, ge=0, description="Aktuelle Tiefe (0 = top-level)")


class AgentHandle(BaseModel):
    """Handle auf einen laufenden Sub-Agent. [B§7.2]"""

    agent_id: str
    config: SubAgentConfig
    session_id: str = ""
    depth: int = 0  # Aktuelle Tiefe des Agents
    status: str = "pending"  # pending | running | completed | failed | timeout
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: AgentResult | None = None


class SanitizeResult(BaseModel):
    """Ergebnis der Input-Sanitization. [B§11.3]"""

    original_length: int
    sanitized_length: int
    patterns_found: list[str] = Field(default_factory=list)
    was_modified: bool = False
    sanitized_text: str = ""


class CredentialEntry(BaseModel):
    """Eintrag im Credential-Store. [B§11.2]"""

    service: str
    key: str
    encrypted: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_accessed: datetime | None = None


# ============================================================================
# v20 Intelligence Features
# ============================================================================


# --- Feature 1: Episodic Store ---


class EpisodicEntry(BaseModel, frozen=True):
    """Strukturierter Eintrag in der episodischen Datenbank."""

    id: str = Field(default_factory=_new_id)
    session_id: str
    timestamp: datetime = Field(default_factory=_utc_now)
    topic: str
    content: str
    outcome: str = ""
    tool_sequence: list[str] = Field(default_factory=list)
    success_score: float = 0.0
    tags: list[str] = Field(default_factory=list)


class EpisodicSummary(BaseModel, frozen=True):
    """Zusammenfassung ueber einen Zeitraum."""

    id: str = Field(default_factory=_new_id)
    period: Literal["day", "week", "month"]
    start_date: date
    end_date: date
    summary: str
    key_learnings: list[str] = Field(default_factory=list)


# --- Feature 2: Weight Optimizer ---


class SearchOutcome(BaseModel, frozen=True):
    """Ergebnis einer Suche fuer Weight-Optimierung."""

    query_hash: str
    feedback_score: float  # 0-1
    channel_contributions: dict[str, float] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=_utc_now)


# --- Feature 3: Self-Profiling ---


class ToolProfile(BaseModel, frozen=True):
    """Profil eines einzelnen Tools."""

    tool_name: str
    avg_latency_ms: float
    p95_latency_ms: float
    success_rate: float
    call_count: int
    error_types: dict[str, int] = Field(default_factory=dict)


class TaskProfile(BaseModel, frozen=True):
    """Profil einer Task-Kategorie."""

    category: str
    avg_score: float
    success_rate: float
    avg_duration_seconds: float
    common_tools: list[str] = Field(default_factory=list)
    task_count: int = 0


class CapabilityProfile(BaseModel, frozen=True):
    """Gesamtprofil der Faehigkeiten."""

    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    tool_profiles: list[ToolProfile] = Field(default_factory=list)
    task_profiles: list[TaskProfile] = Field(default_factory=list)
    overall_success_rate: float = 0.0


# --- Feature 4: Agent Kernel ---


class KernelState(StrEnum):
    """Zustaende der Agent-Kernel State Machine."""

    IDLE = "idle"
    ROUTING = "routing"
    PLANNING = "planning"
    GATING = "gating"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    DONE = "done"
    ERROR = "error"


class PlanNode(BaseModel, frozen=True):
    """Knoten im Plan-DAG."""

    id: str = Field(default_factory=_new_id)
    tool: str
    params: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    estimated_risk: RiskLevel = RiskLevel.GREEN


class Checkpoint(BaseModel):
    """Snapshot des Kernel-Zustands fuer Rollback."""

    id: str = Field(default_factory=_new_id)
    session_id: str
    kernel_state: KernelState
    timestamp: datetime = Field(default_factory=_utc_now)
    working_memory_snapshot: dict[str, Any] = Field(default_factory=dict)
    completed_nodes: list[str] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)


# --- Feature 5: Telemetry Layer ---


class TaskTelemetryRecord(BaseModel, frozen=True):
    """Einzelne Task-Telemetrie-Messung."""

    session_id: str
    timestamp: datetime = Field(default_factory=_utc_now)
    success: bool
    duration_ms: float
    tools_used: list[str] = Field(default_factory=list)
    error_type: str = ""
    error_message: str = ""


class ErrorCluster(BaseModel, frozen=True):
    """Cluster aehnlicher Fehler."""

    pattern: str
    count: int
    examples: list[str] = Field(default_factory=list)
    first_seen: datetime
    last_seen: datetime
    severity: str = "medium"


# --- Feature 6: Causal Learning ---


class ToolSequenceRecord(BaseModel, frozen=True):
    """Aufzeichnung einer Tool-Sequenz mit Erfolgs-Score."""

    session_id: str
    tool_sequence: list[str]
    success_score: float
    timestamp: datetime = Field(default_factory=_utc_now)


class SequenceScore(BaseModel, frozen=True):
    """Bewertung einer Tool-Subsequenz."""

    subsequence: tuple[str, ...]
    avg_score: float
    occurrence_count: int
    confidence: float  # occurrence_count / total_sequences


# --- Feature 7: Refactoring Agent ---


class CodeSmell(BaseModel, frozen=True):
    """Erkannter Code-Smell."""

    file_path: str
    line: int
    smell_type: str  # "long_function", "deep_nesting", "too_many_params", "god_class"
    severity: str  # "info", "warning", "error"
    message: str
    suggestion: str = ""


class ArchitectureFinding(BaseModel, frozen=True):
    """Architektur-Befund."""

    finding_type: str  # "circular_import", "layer_violation", "high_coupling"
    severity: str
    modules: list[str]
    message: str


class RefactoringReport(BaseModel, frozen=True):
    """Gesamtbericht einer Refactoring-Analyse."""

    timestamp: datetime = Field(default_factory=_utc_now)
    code_smells: list[CodeSmell] = Field(default_factory=list)
    architecture_findings: list[ArchitectureFinding] = Field(default_factory=list)
    summary: str = ""
    total_files_analyzed: int = 0


# --- Feature 8: Sandbox Capabilities ---


class ToolCapability(StrEnum):
    """Feingranulare Faehigkeiten eines Tools."""

    FS_READ = "fs_read"
    FS_WRITE = "fs_write"
    NETWORK_HTTP = "network_http"
    NETWORK_WS = "network_ws"
    EXEC_PROCESS = "exec_process"
    EXEC_SCRIPT = "exec_script"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    CREDENTIAL_ACCESS = "credential_access"
    SYSTEM_INFO = "system_info"


class ToolCapabilitySpec(BaseModel, frozen=True):
    """Spezifikation der Faehigkeiten eines Tools."""

    tool_name: str
    capabilities: frozenset[ToolCapability]
    max_memory_mb: int = 512
    max_timeout_seconds: int = 30
    network_domains: list[str] = Field(default_factory=list)


class PolicyDecision(BaseModel, frozen=True):
    """Entscheidung der Capability-Policy-Evaluierung."""

    allowed: bool
    violations: list[str] = Field(default_factory=list)
    suggested_profile: str = ""


# ============================================================================
# Phase 2 Intelligence: Cost Tracking, Forensik, Governance
# ============================================================================


# --- Cost Tracking ---


class CostRecord(BaseModel, frozen=True):
    """Einzelner LLM-Kosten-Eintrag."""

    id: str = Field(default_factory=_new_id)
    timestamp: datetime = Field(default_factory=_utc_now)
    session_id: str = ""
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class BudgetStatus(BaseModel, frozen=True):
    """Budget-Pruefungsergebnis."""

    ok: bool = True
    daily_remaining: float = -1.0   # -1 = kein Limit
    monthly_remaining: float = -1.0
    warning: str = ""


class CostReport(BaseModel, frozen=True):
    """Aggregierter Kosten-Report."""

    total_cost_usd: float = 0.0
    total_calls: int = 0
    cost_by_model: dict[str, float] = Field(default_factory=dict)
    cost_by_day: dict[str, float] = Field(default_factory=dict)
    avg_cost_per_call: float = 0.0


# --- Run Recording + Replay ---


class RunRecord(BaseModel):
    """Kompletter aufgezeichneter Agent-Run."""

    id: str = Field(default_factory=_new_id)
    session_id: str
    timestamp: datetime = Field(default_factory=_utc_now)
    user_message: str = ""
    operation_mode: str = ""
    success: bool = False
    final_response: str = ""
    duration_ms: float = 0.0
    plans: list[ActionPlan] = Field(default_factory=list)
    gate_decisions: list[list[GateDecision]] = Field(default_factory=list)
    tool_results: list[list[ToolResult]] = Field(default_factory=list)
    reflection: ReflectionResult | None = None
    policy_snapshot: dict[str, Any] = Field(default_factory=dict)


class RunSummary(BaseModel, frozen=True):
    """Zusammenfassung eines Runs."""

    id: str
    session_id: str
    timestamp: datetime
    user_message_preview: str = ""
    success: bool = False
    duration_ms: float = 0.0
    tool_count: int = 0


class DecisionDivergence(BaseModel, frozen=True):
    """Abweichung zwischen Original- und Replay-Entscheidung."""

    step_index: int
    tool_name: str
    original_status: str
    replayed_status: str
    original_reason: str
    replayed_reason: str


class ReplayResult(BaseModel, frozen=True):
    """Ergebnis eines Replays."""

    run_id: str
    divergences: list[DecisionDivergence] = Field(default_factory=list)
    original_success: bool = False
    would_have_succeeded: bool | None = None
    policy_variant_name: str = ""


# --- Governance ---


class PolicyProposal(BaseModel):
    """Vorschlag fuer eine Policy-Aenderung."""

    id: int = 0  # SQLite auto-increment ID (0 = noch nicht persistiert)
    timestamp: datetime = Field(default_factory=_utc_now)
    category: str = ""     # "error_rate", "budget", "recurring_error", "tool_latency", "unused_tool"
    title: str = ""
    description: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)
    suggested_change: dict[str, Any] = Field(default_factory=dict)
    status: str = "pending"  # "pending", "approved", "rejected"
    decision_reason: str = ""


class PolicyChange(BaseModel, frozen=True):
    """Genehmigte Policy-Aenderung."""

    proposal_id: int
    category: str = ""     # Bestimmt Ziel-Datei: {category}.yaml
    title: str = ""
    change: dict[str, Any] = Field(default_factory=dict)  # Die eigentliche Aenderung
    timestamp: datetime = Field(default_factory=_utc_now)
