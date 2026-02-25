"""Human-in-the-Loop Types -- v20.

Datenmodelle für HITL-Workflows auf Graph-Ebene:
  - ApprovalRequest:   Anfrage an menschlichen Reviewer
  - ApprovalResponse:  Entscheidung des Reviewers
  - EscalationPolicy:  Eskalations-Regeln (Timeout, Delegation)
  - ReviewTask:        Vollständiger Review-Auftrag mit Kontext
  - HITLConfig:        Konfiguration pro HITL-Knoten
  - NotificationChannel: Benachrichtigungs-Kanal-Definition

Integriert sich mit:
  - v18 Graph Orchestrator (Pause/Resume an Knoten)
  - v19 OpenTelemetry (Tracing von HITL-Wartezeiten)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable


# ── Enums ────────────────────────────────────────────────────────

class ApprovalStatus(str, Enum):
    """Status einer Approval-Anfrage."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    TIMED_OUT = "timed_out"
    CANCELED = "canceled"
    DELEGATED = "delegated"


class ReviewPriority(str, Enum):
    """Priorität eines Reviews."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class EscalationAction(str, Enum):
    """Aktion bei Eskalation."""
    AUTO_APPROVE = "auto_approve"
    AUTO_REJECT = "auto_reject"
    DELEGATE = "delegate"
    NOTIFY_SUPERVISOR = "notify_supervisor"
    PAUSE_INDEFINITELY = "pause_indefinitely"


class NotificationType(str, Enum):
    """Art der Benachrichtigung."""
    IN_APP = "in_app"
    WEBHOOK = "webhook"
    CALLBACK = "callback"
    EMAIL = "email"
    LOG = "log"


class HITLNodeKind(str, Enum):
    """Art des HITL-Knotens."""
    APPROVAL = "approval"       # Ja/Nein-Entscheidung
    REVIEW = "review"           # Prüfung mit Kommentar
    INPUT = "input"             # Menschliche Eingabe benötigt
    GATE = "gate"               # Bedingter Stopp (nur bei Risiko)
    EDIT = "edit"               # Mensch editiert State-Daten
    SELECTION = "selection"     # Auswahl aus Optionen


# ── Notification Channel ─────────────────────────────────────────

@dataclass
class NotificationChannel:
    """Definition eines Benachrichtigungs-Kanals."""
    channel_type: NotificationType = NotificationType.LOG
    endpoint: str = ""          # URL für webhook, Callback-ID, E-Mail
    template: str = ""          # Nachricht-Template mit {placeholders}
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.channel_type.value,
            "endpoint": self.endpoint,
            "enabled": self.enabled,
        }


# ── Escalation Policy ───────────────────────────────────────────

@dataclass
class EscalationPolicy:
    """Regeln für Timeout und Eskalation.

    Definiert was passiert wenn innerhalb von timeout_seconds
    keine Antwort kommt.
    """
    timeout_seconds: int = 3600             # Default: 1 Stunde
    action: EscalationAction = EscalationAction.PAUSE_INDEFINITELY
    delegate_to: str = ""                   # User/Rolle für Delegation
    max_escalations: int = 3
    reminder_interval_seconds: int = 900    # Alle 15 Min Erinnerung
    auto_approve_conditions: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeout_seconds": self.timeout_seconds,
            "action": self.action.value,
            "delegate_to": self.delegate_to,
            "max_escalations": self.max_escalations,
            "reminder_interval_seconds": self.reminder_interval_seconds,
        }


# ── HITL Config ──────────────────────────────────────────────────

@dataclass
class HITLConfig:
    """Konfiguration für einen HITL-Knoten."""
    node_kind: HITLNodeKind = HITLNodeKind.APPROVAL
    title: str = ""
    description: str = ""
    instructions: str = ""
    assignees: list[str] = field(default_factory=list)
    priority: ReviewPriority = ReviewPriority.NORMAL
    required_approvals: int = 1         # Wie viele Approvals nötig
    escalation: EscalationPolicy = field(default_factory=EscalationPolicy)
    notifications: list[NotificationChannel] = field(default_factory=list)
    context_keys: list[str] = field(default_factory=list)  # State-Keys die angezeigt werden
    options: list[str] = field(default_factory=list)  # Für SELECTION-Typ
    editable_keys: list[str] = field(default_factory=list)  # Für EDIT-Typ
    auto_approve_fn: Callable[[dict[str, Any]], bool] | None = None  # Gate: Auto-Skip
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.node_kind.value,
            "title": self.title,
            "description": self.description,
            "assignees": self.assignees,
            "priority": self.priority.value,
            "required_approvals": self.required_approvals,
            "escalation": self.escalation.to_dict(),
            "options": self.options,
        }


# ── Approval Request ─────────────────────────────────────────────

@dataclass
class ApprovalRequest:
    """Anfrage an menschlichen Reviewer."""
    request_id: str = ""
    execution_id: str = ""
    graph_name: str = ""
    node_name: str = ""
    config: HITLConfig = field(default_factory=HITLConfig)
    context: dict[str, Any] = field(default_factory=dict)
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: str = ""
    updated_at: str = ""
    expires_at: str = ""
    checkpoint_id: str = ""
    escalation_count: int = 0
    reminders_sent: int = 0

    def __post_init__(self) -> None:
        if not self.request_id:
            self.request_id = f"apr_{uuid.uuid4().hex[:10]}"
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    @property
    def is_pending(self) -> bool:
        return self.status == ApprovalStatus.PENDING

    @property
    def is_resolved(self) -> bool:
        return self.status in (
            ApprovalStatus.APPROVED, ApprovalStatus.REJECTED,
            ApprovalStatus.TIMED_OUT, ApprovalStatus.CANCELED,
        )

    @property
    def age_seconds(self) -> float:
        try:
            import calendar
            created = calendar.timegm(time.strptime(self.created_at, "%Y-%m-%dT%H:%M:%SZ"))
            return time.time() - created
        except (ValueError, OverflowError):
            return 0.0

    @property
    def is_expired(self) -> bool:
        if not self.config.escalation:
            return False
        return self.age_seconds > self.config.escalation.timeout_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "execution_id": self.execution_id,
            "graph_name": self.graph_name,
            "node_name": self.node_name,
            "config": self.config.to_dict(),
            "context": self.context,
            "status": self.status.value,
            "created_at": self.created_at,
            "age_seconds": round(self.age_seconds),
            "escalation_count": self.escalation_count,
        }


# ── Approval Response ────────────────────────────────────────────

@dataclass
class ApprovalResponse:
    """Antwort eines menschlichen Reviewers."""
    response_id: str = ""
    request_id: str = ""
    decision: ApprovalStatus = ApprovalStatus.APPROVED
    reviewer: str = ""
    comment: str = ""
    modifications: dict[str, Any] = field(default_factory=dict)
    selected_option: str = ""   # Für SELECTION-Typ
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.response_id:
            self.response_id = f"res_{uuid.uuid4().hex[:10]}"
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "response_id": self.response_id,
            "request_id": self.request_id,
            "decision": self.decision.value,
            "reviewer": self.reviewer,
            "timestamp": self.timestamp,
        }
        if self.comment:
            d["comment"] = self.comment
        if self.modifications:
            d["modifications"] = self.modifications
        if self.selected_option:
            d["selected_option"] = self.selected_option
        return d


# ── Review Task ──────────────────────────────────────────────────

@dataclass
class ReviewTask:
    """Vollständiger Review-Auftrag mit Kontext für UI/API."""
    request: ApprovalRequest
    responses: list[ApprovalResponse] = field(default_factory=list)
    notifications_sent: int = 0
    last_notification_at: str = ""

    @property
    def approval_count(self) -> int:
        return sum(1 for r in self.responses if r.decision == ApprovalStatus.APPROVED)

    @property
    def rejection_count(self) -> int:
        return sum(1 for r in self.responses if r.decision == ApprovalStatus.REJECTED)

    @property
    def is_fully_approved(self) -> bool:
        return self.approval_count >= self.request.config.required_approvals

    @property
    def is_rejected(self) -> bool:
        return self.rejection_count > 0

    @property
    def needs_more_approvals(self) -> bool:
        return (not self.is_rejected and
                self.approval_count < self.request.config.required_approvals)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "responses": [r.to_dict() for r in self.responses],
            "approval_count": self.approval_count,
            "rejection_count": self.rejection_count,
            "is_fully_approved": self.is_fully_approved,
            "notifications_sent": self.notifications_sent,
        }
