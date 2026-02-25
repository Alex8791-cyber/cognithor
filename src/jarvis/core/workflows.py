"""Jarvis · Workflow-Templates & Ecosystem Security Policy.

Punkt 6: Beispiel-Workflows für Endanwender
  - WorkflowTemplate:  Vordefinierte Abläufe (Team-Onboarding, Sales-Pipeline, etc.)
  - WorkflowEngine:    Ausführung & Tracking von Workflow-Instanzen
  - TemplateLibrary:   Katalog aller verfügbaren Templates

Punkt 8: Zentrale Sicherheitsrichtlinien
  - EcosystemPolicy:   Mindestanforderungen für das Skill-Ecosystem
  - PolicyEnforcer:     Erzwingt Standards bei Skill-Installation
  - ComplianceBadge:    Zertifizierungs-System für Skills

Architektur-Bibel: §9.4 (Workflows), §11.5 (Ecosystem-Security)
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ============================================================================
# TEIL 1: Workflow-Templates
# ============================================================================


class WorkflowStatus(Enum):
    """Status einer Workflow-Instanz."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowStep:
    """Ein einzelner Schritt in einem Workflow."""

    step_id: str
    name: str
    description: str
    action_type: str  # "agent_task", "approval", "notification", "wait", "condition"
    config: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300
    required: bool = True
    next_steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "name": self.name,
            "description": self.description,
            "action_type": self.action_type,
            "timeout_seconds": self.timeout_seconds,
            "required": self.required,
        }


@dataclass
class WorkflowTemplate:
    """Vordefinierter Workflow-Ablauf."""

    template_id: str
    name: str
    description: str
    category: str  # "onboarding", "sales", "support", "devops", "hr"
    steps: list[WorkflowStep] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    estimated_minutes: int = 0
    icon: str = "📋"

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "steps": [s.to_dict() for s in self.steps],
            "step_count": self.step_count,
            "tags": self.tags,
            "estimated_minutes": self.estimated_minutes,
            "icon": self.icon,
        }


# === Built-in Templates ===

def _team_onboarding_template() -> WorkflowTemplate:
    return WorkflowTemplate(
        template_id="wf-onboarding",
        name="Team-Onboarding",
        description="Neuen Mitarbeiter ins Team einführen: Accounts, Zugänge, Einarbeitung.",
        category="onboarding",
        icon="👋",
        estimated_minutes=120,
        tags=["hr", "onboarding", "team"],
        steps=[
            WorkflowStep("s1", "Willkommens-Nachricht", "Begrüßung via Teams/Slack senden", "notification"),
            WorkflowStep("s2", "Accounts erstellen", "E-Mail, Jira, Git, CRM-Zugänge anlegen", "agent_task",
                         config={"tools": ["jira", "teams", "crm"]}),
            WorkflowStep("s3", "Einarbeitungsplan", "Persönlichen Einarbeitungsplan erstellen", "agent_task"),
            WorkflowStep("s4", "Mentor zuweisen", "Buddy/Mentor-Vorschlag und Benachrichtigung", "approval"),
            WorkflowStep("s5", "Check-in planen", "30-Tage Check-in Termin erstellen", "agent_task"),
        ],
    )

def _sales_pipeline_template() -> WorkflowTemplate:
    return WorkflowTemplate(
        template_id="wf-sales-pipeline",
        name="Sales-Pipeline Follow-up",
        description="Automatisiertes Follow-up für offene Leads: Erinnerungen, Angebote, Nachverfolgung.",
        category="sales",
        icon="💰",
        estimated_minutes=30,
        tags=["sales", "crm", "follow-up"],
        steps=[
            WorkflowStep("s1", "Lead-Daten sammeln", "CRM-Daten des Leads abrufen", "agent_task",
                         config={"tools": ["crm"]}),
            WorkflowStep("s2", "Follow-up E-Mail", "Personalisierte Follow-up-Mail generieren", "agent_task"),
            WorkflowStep("s3", "Genehmigung", "Vertriebsleiter genehmigt den Entwurf", "approval"),
            WorkflowStep("s4", "E-Mail senden", "Genehmigte E-Mail versenden", "agent_task"),
            WorkflowStep("s5", "Reminder setzen", "7-Tage Follow-up Reminder im CRM", "agent_task"),
        ],
    )

def _incident_response_template() -> WorkflowTemplate:
    return WorkflowTemplate(
        template_id="wf-incident",
        name="Incident-Response",
        description="Strukturierter Ablauf bei System-Incidents: Melden, Eskalieren, Lösen, Dokumentieren.",
        category="support",
        icon="🚨",
        estimated_minutes=60,
        tags=["ops", "incident", "servicenow"],
        steps=[
            WorkflowStep("s1", "Incident erstellen", "ServiceNow-Ticket automatisch anlegen", "agent_task",
                         config={"tools": ["servicenow"]}),
            WorkflowStep("s2", "Team benachrichtigen", "On-Call-Team via Teams/Slack informieren", "notification"),
            WorkflowStep("s3", "Diagnose", "Agent sammelt Logs und analysiert", "agent_task"),
            WorkflowStep("s4", "Lösung vorschlagen", "Agent schlägt Remediation vor", "agent_task"),
            WorkflowStep("s5", "Bestätigung", "Mensch bestätigt die Lösung", "approval"),
            WorkflowStep("s6", "Post-Mortem", "Automatisches Post-Mortem-Dokument erstellen", "agent_task"),
        ],
    )

def _code_review_template() -> WorkflowTemplate:
    return WorkflowTemplate(
        template_id="wf-code-review",
        name="Automatischer Code-Review",
        description="Agent prüft Code-Änderungen auf Qualität, Sicherheit und Standards.",
        category="devops",
        icon="🔍",
        estimated_minutes=15,
        tags=["dev", "code-review", "ci-cd"],
        steps=[
            WorkflowStep("s1", "Diff laden", "Code-Änderungen aus Git laden", "agent_task"),
            WorkflowStep("s2", "Sicherheits-Check", "Red-Team-Scanner auf neuen Code anwenden", "agent_task"),
            WorkflowStep("s3", "Style-Check", "Code-Style und Best-Practices prüfen", "agent_task"),
            WorkflowStep("s4", "Review-Kommentar", "Zusammenfassung als Jira-Kommentar posten", "agent_task",
                         config={"tools": ["jira"]}),
        ],
    )


# ============================================================================
# Workflow-Engine
# ============================================================================


@dataclass
class WorkflowInstance:
    """Eine laufende Instanz eines Workflows."""

    instance_id: str
    template_id: str
    template_name: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    current_step: int = 0
    total_steps: int = 0
    started_at: str = ""
    completed_at: str = ""
    step_results: dict[str, Any] = field(default_factory=dict)
    created_by: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "template_id": self.template_id,
            "template_name": self.template_name,
            "status": self.status.value,
            "progress": f"{self.current_step}/{self.total_steps}",
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class WorkflowEngine:
    """Ausführung und Tracking von Workflow-Instanzen."""

    def __init__(self) -> None:
        self._instances: dict[str, WorkflowInstance] = {}

    def start(self, template: WorkflowTemplate, *, created_by: str = "") -> WorkflowInstance:
        instance_id = hashlib.sha256(f"{template.template_id}:{time.time()}".encode()).hexdigest()[:12]
        instance = WorkflowInstance(
            instance_id=instance_id,
            template_id=template.template_id,
            template_name=template.name,
            status=WorkflowStatus.RUNNING,
            total_steps=template.step_count,
            started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            created_by=created_by,
        )
        self._instances[instance_id] = instance
        return instance

    def advance(self, instance_id: str, step_result: Any = None) -> WorkflowInstance | None:
        inst = self._instances.get(instance_id)
        if not inst or inst.status != WorkflowStatus.RUNNING:
            return None
        inst.step_results[str(inst.current_step)] = step_result
        inst.current_step += 1
        if inst.current_step >= inst.total_steps:
            inst.status = WorkflowStatus.COMPLETED
            inst.completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return inst

    def pause(self, instance_id: str) -> bool:
        inst = self._instances.get(instance_id)
        if inst and inst.status == WorkflowStatus.RUNNING:
            inst.status = WorkflowStatus.PAUSED
            return True
        return False

    def cancel(self, instance_id: str) -> bool:
        inst = self._instances.get(instance_id)
        if inst and inst.status in (WorkflowStatus.RUNNING, WorkflowStatus.PAUSED):
            inst.status = WorkflowStatus.CANCELLED
            return True
        return False

    def get(self, instance_id: str) -> WorkflowInstance | None:
        return self._instances.get(instance_id)

    @property
    def instance_count(self) -> int:
        return len(self._instances)

    def active_instances(self) -> list[WorkflowInstance]:
        return [i for i in self._instances.values() if i.status == WorkflowStatus.RUNNING]

    def stats(self) -> dict[str, Any]:
        instances = list(self._instances.values())
        return {
            "total": len(instances),
            "running": sum(1 for i in instances if i.status == WorkflowStatus.RUNNING),
            "completed": sum(1 for i in instances if i.status == WorkflowStatus.COMPLETED),
            "failed": sum(1 for i in instances if i.status == WorkflowStatus.FAILED),
        }


# ============================================================================
# Template-Library
# ============================================================================


class TemplateLibrary:
    """Katalog aller verfügbaren Workflow-Templates."""

    def __init__(self, *, load_builtins: bool = True) -> None:
        self._templates: dict[str, WorkflowTemplate] = {}
        if load_builtins:
            for t in [
                _team_onboarding_template(),
                _sales_pipeline_template(),
                _incident_response_template(),
                _code_review_template(),
            ]:
                self._templates[t.template_id] = t

    def add(self, template: WorkflowTemplate) -> None:
        self._templates[template.template_id] = template

    def get(self, template_id: str) -> WorkflowTemplate | None:
        return self._templates.get(template_id)

    def search(self, *, category: str = "", tag: str = "") -> list[WorkflowTemplate]:
        results = list(self._templates.values())
        if category:
            results = [t for t in results if t.category == category]
        if tag:
            results = [t for t in results if tag in t.tags]
        return results

    @property
    def template_count(self) -> int:
        return len(self._templates)

    def categories(self) -> list[str]:
        return list(set(t.category for t in self._templates.values()))

    def list_all(self) -> list[dict[str, Any]]:
        return [t.to_dict() for t in self._templates.values()]


# ============================================================================
# TEIL 2: Ecosystem Security Policy
# ============================================================================


class SecurityTier(Enum):
    """Sicherheitsstufe eines Skills."""

    UNVERIFIED = "unverified"     # Keine Prüfung
    COMMUNITY = "community"       # Community-geprüft
    REVIEWED = "reviewed"         # Code-Review bestanden
    CERTIFIED = "certified"       # Vollständig zertifiziert
    TRUSTED = "trusted"           # Offizieller Jarvis-Partner


@dataclass
class SkillSecurityRequirement:
    """Mindestanforderung für Skill-Sicherheit."""

    requirement_id: str
    name: str
    description: str
    tier: SecurityTier  # Ab welcher Stufe erforderlich
    check_fn_name: str = ""  # Name der Prüffunktion
    mandatory: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "name": self.name,
            "tier": self.tier.value,
            "mandatory": self.mandatory,
        }


# Standard-Anforderungen
_ECOSYSTEM_REQUIREMENTS: list[SkillSecurityRequirement] = [
    SkillSecurityRequirement("ESR-001", "Code-Signatur", "Skill muss digital signiert sein", SecurityTier.COMMUNITY),
    SkillSecurityRequirement("ESR-002", "Keine Netzwerk-Calls", "Kein unkontrollierter Netzwerkzugriff", SecurityTier.COMMUNITY),
    SkillSecurityRequirement("ESR-003", "Sandbox-kompatibel", "Muss innerhalb der Sandbox laufen", SecurityTier.COMMUNITY),
    SkillSecurityRequirement("ESR-004", "Lizenz-Deklaration", "Open-Source-Lizenz muss angegeben sein", SecurityTier.COMMUNITY),
    SkillSecurityRequirement("ESR-005", "Static-Analysis", "Keine bekannten CVEs in Dependencies", SecurityTier.REVIEWED),
    SkillSecurityRequirement("ESR-006", "Input-Validation", "Alle Inputs müssen validiert werden", SecurityTier.REVIEWED),
    SkillSecurityRequirement("ESR-007", "Code-Review", "Manuelles Code-Review durch Kurator", SecurityTier.REVIEWED),
    SkillSecurityRequirement("ESR-008", "Penetration-Test", "Red-Team-Test bestanden", SecurityTier.CERTIFIED),
    SkillSecurityRequirement("ESR-009", "Audit-Trail", "Alle Aktionen werden geloggt", SecurityTier.CERTIFIED),
    SkillSecurityRequirement("ESR-010", "DSGVO-Konformität", "Keine unerlaubte Datenverarbeitung", SecurityTier.CERTIFIED),
]


@dataclass
class ComplianceBadge:
    """Zertifizierungs-Badge für einen Skill."""

    badge_id: str
    skill_id: str
    tier: SecurityTier
    issued_at: str
    expires_at: str = ""
    issuer: str = "jarvis-ecosystem"
    requirements_met: list[str] = field(default_factory=list)
    requirements_failed: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        if not self.expires_at:
            return True
        return self.expires_at > time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def to_dict(self) -> dict[str, Any]:
        return {
            "badge_id": self.badge_id,
            "skill_id": self.skill_id,
            "tier": self.tier.value,
            "issued_at": self.issued_at,
            "valid": self.is_valid,
            "requirements_met": self.requirements_met,
            "requirements_failed": self.requirements_failed,
        }


class EcosystemPolicy:
    """Zentrale Sicherheitsrichtlinien für das Skill-Ecosystem."""

    def __init__(
        self,
        *,
        requirements: list[SkillSecurityRequirement] | None = None,
        minimum_tier: SecurityTier = SecurityTier.COMMUNITY,
    ) -> None:
        self._requirements = requirements or list(_ECOSYSTEM_REQUIREMENTS)
        self._minimum_tier = minimum_tier
        self._badges: dict[str, ComplianceBadge] = {}

    @property
    def requirement_count(self) -> int:
        return len(self._requirements)

    @property
    def minimum_tier(self) -> SecurityTier:
        return self._minimum_tier

    @minimum_tier.setter
    def minimum_tier(self, value: SecurityTier) -> None:
        self._minimum_tier = value

    def requirements_for_tier(self, tier: SecurityTier) -> list[SkillSecurityRequirement]:
        tier_order = [SecurityTier.UNVERIFIED, SecurityTier.COMMUNITY, SecurityTier.REVIEWED,
                      SecurityTier.CERTIFIED, SecurityTier.TRUSTED]
        tier_idx = tier_order.index(tier)
        return [r for r in self._requirements if tier_order.index(r.tier) <= tier_idx]

    def evaluate_skill(
        self,
        skill_id: str,
        *,
        has_signature: bool = False,
        has_sandbox: bool = False,
        has_license: bool = False,
        has_network_control: bool = False,
        passed_static_analysis: bool = False,
        passed_code_review: bool = False,
        passed_pentest: bool = False,
        has_audit_trail: bool = False,
        has_input_validation: bool = False,
        is_dsgvo_compliant: bool = False,
    ) -> ComplianceBadge:
        """Bewertet einen Skill und vergibt ein Badge."""
        checks = {
            "ESR-001": has_signature,
            "ESR-002": has_network_control,
            "ESR-003": has_sandbox,
            "ESR-004": has_license,
            "ESR-005": passed_static_analysis,
            "ESR-006": has_input_validation,
            "ESR-007": passed_code_review,
            "ESR-008": passed_pentest,
            "ESR-009": has_audit_trail,
            "ESR-010": is_dsgvo_compliant,
        }

        met = [rid for rid, passed in checks.items() if passed]
        failed = [rid for rid, passed in checks.items() if not passed]

        # Höchste erreichbare Stufe bestimmen
        tier_order = [SecurityTier.UNVERIFIED, SecurityTier.COMMUNITY, SecurityTier.REVIEWED,
                      SecurityTier.CERTIFIED, SecurityTier.TRUSTED]
        achieved_tier = SecurityTier.UNVERIFIED

        for tier in tier_order[1:]:
            reqs = self.requirements_for_tier(tier)
            if all(checks.get(r.requirement_id, False) for r in reqs if r.mandatory):
                achieved_tier = tier
            else:
                break

        badge = ComplianceBadge(
            badge_id=hashlib.sha256(f"{skill_id}:{time.time()}".encode()).hexdigest()[:12],
            skill_id=skill_id,
            tier=achieved_tier,
            issued_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            requirements_met=met,
            requirements_failed=failed,
        )
        self._badges[skill_id] = badge
        return badge

    def meets_minimum(self, skill_id: str) -> bool:
        """Prüft ob ein Skill die Mindestanforderungen erfüllt."""
        badge = self._badges.get(skill_id)
        if not badge:
            return False
        tier_order = [SecurityTier.UNVERIFIED, SecurityTier.COMMUNITY, SecurityTier.REVIEWED,
                      SecurityTier.CERTIFIED, SecurityTier.TRUSTED]
        return tier_order.index(badge.tier) >= tier_order.index(self._minimum_tier)

    def get_badge(self, skill_id: str) -> ComplianceBadge | None:
        return self._badges.get(skill_id)

    def stats(self) -> dict[str, Any]:
        badges = list(self._badges.values())
        tier_counts = {}
        for b in badges:
            tier_counts[b.tier.value] = tier_counts.get(b.tier.value, 0) + 1
        return {
            "total_requirements": len(self._requirements),
            "minimum_tier": self._minimum_tier.value,
            "total_badges": len(badges),
            "tier_distribution": tier_counts,
        }
