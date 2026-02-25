"""Jarvis · AI Agent Security Framework.

Enterprise-Security-Metriken und Team-Rollenverteilung:

  - SecurityMetrics:       MTTD, MTTR, Incident-Rate, Posture-Score
  - SecurityIncident:      Strukturierte Incident-Erfassung mit Lifecycle
  - IncidentTracker:       Verwaltung aller Security-Incidents
  - SecurityTeam:          Rollenverteilung (ML, Dev, Security, Compliance)
  - PostureScorer:         Gesamt-Sicherheitsbewertung des Systems
  - SecurityDashboardData: Aggregierte Daten für Echtzeit-Dashboard

Architektur-Bibel: §11.7 (Security-Framework), §14.4 (Metriken)
"""

from __future__ import annotations

import calendar
import hashlib
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ============================================================================
# Security Incidents
# ============================================================================


class IncidentSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IncidentStatus(Enum):
    OPEN = "open"
    DETECTED = "detected"
    INVESTIGATING = "investigating"
    CONTAINED = "contained"
    RESOLVED = "resolved"
    CLOSED = "closed"


class IncidentCategory(Enum):
    PROMPT_INJECTION = "prompt_injection"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    MODEL_INVERSION = "model_inversion"
    MEMORY_POISONING = "memory_poisoning"
    DENIAL_OF_SERVICE = "denial_of_service"
    CREDENTIAL_LEAK = "credential_leak"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    BIAS_VIOLATION = "bias_violation"
    POLICY_VIOLATION = "policy_violation"


@dataclass
class SecurityIncident:
    """Ein strukturierter Security-Incident mit vollständigem Lifecycle."""

    incident_id: str
    title: str
    category: IncidentCategory
    severity: IncidentSeverity
    status: IncidentStatus = IncidentStatus.OPEN
    description: str = ""
    agent_id: str = ""
    source: str = ""  # Welches Modul hat erkannt
    occurred_at: str = ""
    detected_at: str = ""
    contained_at: str = ""
    resolved_at: str = ""
    closed_at: str = ""
    assigned_to: str = ""
    assigned_role: str = ""
    remediation: str = ""
    root_cause: str = ""
    findings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def time_to_detect_seconds(self) -> float | None:
        """MTTD: Zeit von Auftreten bis Erkennung."""
        if self.occurred_at and self.detected_at:
            try:
                t1 = calendar.timegm(time.strptime(self.occurred_at, "%Y-%m-%dT%H:%M:%SZ"))
                t2 = calendar.timegm(time.strptime(self.detected_at, "%Y-%m-%dT%H:%M:%SZ"))
                return max(0, t2 - t1)
            except (ValueError, OverflowError):
                return None
        return None

    @property
    def time_to_resolve_seconds(self) -> float | None:
        """MTTR: Zeit von Erkennung bis Lösung."""
        if self.detected_at and self.resolved_at:
            try:
                t1 = calendar.timegm(time.strptime(self.detected_at, "%Y-%m-%dT%H:%M:%SZ"))
                t2 = calendar.timegm(time.strptime(self.resolved_at, "%Y-%m-%dT%H:%M:%SZ"))
                return max(0, t2 - t1)
            except (ValueError, OverflowError):
                return None
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "title": self.title,
            "category": self.category.value,
            "severity": self.severity.value,
            "status": self.status.value,
            "agent_id": self.agent_id,
            "source": self.source,
            "assigned_to": self.assigned_to,
            "assigned_role": self.assigned_role,
            "ttd_seconds": self.time_to_detect_seconds,
            "ttr_seconds": self.time_to_resolve_seconds,
        }


# ============================================================================
# Incident Tracker
# ============================================================================


class IncidentTracker:
    """Verwaltet alle Security-Incidents mit Lifecycle-Tracking."""

    def __init__(self) -> None:
        self._incidents: dict[str, SecurityIncident] = {}
        self._counter = 0

    def create(
        self,
        title: str,
        category: IncidentCategory,
        severity: IncidentSeverity,
        *,
        description: str = "",
        agent_id: str = "",
        source: str = "",
        occurred_at: str = "",
    ) -> SecurityIncident:
        self._counter += 1
        incident_id = f"INC-{self._counter:05d}"
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        incident = SecurityIncident(
            incident_id=incident_id,
            title=title,
            category=category,
            severity=severity,
            description=description,
            agent_id=agent_id,
            source=source,
            occurred_at=occurred_at or now,
            detected_at=now,
            status=IncidentStatus.DETECTED,
        )
        self._incidents[incident_id] = incident
        return incident

    def get(self, incident_id: str) -> SecurityIncident | None:
        return self._incidents.get(incident_id)

    def transition(self, incident_id: str, new_status: IncidentStatus) -> SecurityIncident | None:
        inc = self._incidents.get(incident_id)
        if not inc:
            return None
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        inc.status = new_status
        if new_status == IncidentStatus.CONTAINED:
            inc.contained_at = now
        elif new_status == IncidentStatus.RESOLVED:
            inc.resolved_at = now
        elif new_status == IncidentStatus.CLOSED:
            inc.closed_at = now
        return inc

    def assign(self, incident_id: str, person: str, role: str = "") -> bool:
        inc = self._incidents.get(incident_id)
        if inc:
            inc.assigned_to = person
            inc.assigned_role = role
            return True
        return False

    def open_incidents(self) -> list[SecurityIncident]:
        return [i for i in self._incidents.values()
                if i.status not in (IncidentStatus.RESOLVED, IncidentStatus.CLOSED)]

    def by_severity(self, severity: IncidentSeverity) -> list[SecurityIncident]:
        return [i for i in self._incidents.values() if i.severity == severity]

    def by_category(self, category: IncidentCategory) -> list[SecurityIncident]:
        return [i for i in self._incidents.values() if i.category == category]

    @property
    def count(self) -> int:
        return len(self._incidents)

    def all_incidents(self) -> list[SecurityIncident]:
        return list(self._incidents.values())

    def stats(self) -> dict[str, Any]:
        incidents = list(self._incidents.values())
        return {
            "total": len(incidents),
            "open": sum(1 for i in incidents if i.status not in (IncidentStatus.RESOLVED, IncidentStatus.CLOSED)),
            "resolved": sum(1 for i in incidents if i.status == IncidentStatus.RESOLVED),
            "by_severity": {
                sev.value: sum(1 for i in incidents if i.severity == sev)
                for sev in IncidentSeverity
                if any(i.severity == sev for i in incidents)
            },
            "by_category": {
                cat.value: sum(1 for i in incidents if i.category == cat)
                for cat in IncidentCategory
                if any(i.category == cat for i in incidents)
            },
        }


# ============================================================================
# Security Metrics (MTTD / MTTR)
# ============================================================================


class SecurityMetrics:
    """Berechnet Enterprise-Security-KPIs aus Incident-Daten.

    Metriken:
      - MTTD (Mean Time to Detect): Durchschnittliche Erkennungszeit
      - MTTR (Mean Time to Resolve): Durchschnittliche Lösungszeit
      - Incident Rate: Incidents pro Zeiteinheit
      - Resolution Rate: % gelöster Incidents
      - Severity Distribution: Verteilung nach Schweregrad
    """

    def __init__(self, tracker: IncidentTracker) -> None:
        self._tracker = tracker

    def mttd(self) -> float:
        """Mean Time to Detect (Sekunden). 0.0 wenn keine Daten."""
        times = [
            i.time_to_detect_seconds
            for i in self._tracker.all_incidents()
            if i.time_to_detect_seconds is not None
        ]
        return statistics.mean(times) if times else 0.0

    def mttr(self) -> float:
        """Mean Time to Resolve (Sekunden). 0.0 wenn keine Daten."""
        times = [
            i.time_to_resolve_seconds
            for i in self._tracker.all_incidents()
            if i.time_to_resolve_seconds is not None
        ]
        return statistics.mean(times) if times else 0.0

    def resolution_rate(self) -> float:
        """Prozentsatz gelöster Incidents."""
        total = self._tracker.count
        if total == 0:
            return 100.0
        resolved = sum(
            1 for i in self._tracker.all_incidents()
            if i.status in (IncidentStatus.RESOLVED, IncidentStatus.CLOSED)
        )
        return resolved / total * 100

    def incident_rate(self, period_hours: float = 24.0) -> float:
        """Incidents pro Zeitperiode (Standard: 24h)."""
        if self._tracker.count == 0:
            return 0.0
        incidents = self._tracker.all_incidents()
        try:
            times = [
                calendar.timegm(time.strptime(i.detected_at, "%Y-%m-%dT%H:%M:%SZ"))
                for i in incidents if i.detected_at
            ]
            if len(times) < 2:
                return float(len(times))
            span_hours = (max(times) - min(times)) / 3600
            if span_hours == 0:
                return float(len(times))
            return len(times) / span_hours * period_hours
        except (ValueError, OverflowError):
            return float(self._tracker.count)

    def severity_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {}
        for i in self._tracker.all_incidents():
            dist[i.severity.value] = dist.get(i.severity.value, 0) + 1
        return dist

    def category_heatmap(self) -> dict[str, int]:
        heatmap: dict[str, int] = {}
        for i in self._tracker.all_incidents():
            heatmap[i.category.value] = heatmap.get(i.category.value, 0) + 1
        return heatmap

    def to_dict(self) -> dict[str, Any]:
        return {
            "mttd_seconds": round(self.mttd(), 1),
            "mttr_seconds": round(self.mttr(), 1),
            "resolution_rate": round(self.resolution_rate(), 1),
            "incident_rate_24h": round(self.incident_rate(), 2),
            "total_incidents": self._tracker.count,
            "open_incidents": len(self._tracker.open_incidents()),
            "severity_distribution": self.severity_distribution(),
            "category_heatmap": self.category_heatmap(),
        }


# ============================================================================
# Security Team Roles
# ============================================================================


class TeamRole(Enum):
    ML_ENGINEER = "ml_engineer"
    DEVELOPER = "developer"
    SECURITY_ANALYST = "security_analyst"
    COMPLIANCE_OFFICER = "compliance_officer"
    INCIDENT_RESPONDER = "incident_responder"
    DATA_PROTECTION = "data_protection"


@dataclass
class TeamMember:
    """Mitglied des Security-Teams."""

    member_id: str
    name: str
    role: TeamRole
    email: str = ""
    on_call: bool = False
    specialties: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "member_id": self.member_id,
            "name": self.name,
            "role": self.role.value,
            "on_call": self.on_call,
            "specialties": self.specialties,
        }


# Automatische Zuweisungsregeln
ROLE_ASSIGNMENT_RULES: dict[IncidentCategory, TeamRole] = {
    IncidentCategory.PROMPT_INJECTION: TeamRole.SECURITY_ANALYST,
    IncidentCategory.DATA_EXFILTRATION: TeamRole.SECURITY_ANALYST,
    IncidentCategory.MODEL_INVERSION: TeamRole.ML_ENGINEER,
    IncidentCategory.MEMORY_POISONING: TeamRole.ML_ENGINEER,
    IncidentCategory.PRIVILEGE_ESCALATION: TeamRole.DEVELOPER,
    IncidentCategory.DENIAL_OF_SERVICE: TeamRole.INCIDENT_RESPONDER,
    IncidentCategory.CREDENTIAL_LEAK: TeamRole.SECURITY_ANALYST,
    IncidentCategory.UNAUTHORIZED_ACCESS: TeamRole.SECURITY_ANALYST,
    IncidentCategory.BIAS_VIOLATION: TeamRole.COMPLIANCE_OFFICER,
    IncidentCategory.POLICY_VIOLATION: TeamRole.COMPLIANCE_OFFICER,
}


class SecurityTeam:
    """Verwaltet Security-Team-Mitglieder und automatische Zuweisung."""

    def __init__(self) -> None:
        self._members: dict[str, TeamMember] = {}

    def add_member(self, member: TeamMember) -> None:
        self._members[member.member_id] = member

    def remove_member(self, member_id: str) -> bool:
        if member_id in self._members:
            del self._members[member_id]
            return True
        return False

    def get_member(self, member_id: str) -> TeamMember | None:
        return self._members.get(member_id)

    def by_role(self, role: TeamRole) -> list[TeamMember]:
        return [m for m in self._members.values() if m.role == role]

    def on_call(self) -> list[TeamMember]:
        return [m for m in self._members.values() if m.on_call]

    def auto_assign(self, incident: SecurityIncident) -> TeamMember | None:
        """Weist automatisch den richtigen Experten zu."""
        target_role = ROLE_ASSIGNMENT_RULES.get(incident.category)
        if not target_role:
            return None
        # Bevorzuge On-Call-Mitglieder
        candidates = self.by_role(target_role)
        on_call = [m for m in candidates if m.on_call]
        chosen = on_call[0] if on_call else (candidates[0] if candidates else None)
        if chosen:
            incident.assigned_to = chosen.name
            incident.assigned_role = chosen.role.value
        return chosen

    @property
    def member_count(self) -> int:
        return len(self._members)

    def stats(self) -> dict[str, Any]:
        members = list(self._members.values())
        return {
            "total_members": len(members),
            "on_call": sum(1 for m in members if m.on_call),
            "by_role": {
                role.value: sum(1 for m in members if m.role == role)
                for role in TeamRole
                if any(m.role == role for m in members)
            },
        }


# ============================================================================
# Security Posture Scorer
# ============================================================================


class PostureScorer:
    """Berechnet einen Gesamt-Sicherheits-Posture-Score.

    Faktoren (gewichtet):
      - Incident-Resolution-Rate (25%)
      - MTTR < Schwellenwert (20%)
      - Abdeckung Security-Team (15%)
      - Pipeline-Pass-Rate (20%)
      - Compliance-Score (20%)
    """

    WEIGHTS = {
        "resolution_rate": 0.25,
        "mttr_health": 0.20,
        "team_coverage": 0.15,
        "pipeline_pass": 0.20,
        "compliance": 0.20,
    }

    def __init__(self, mttr_threshold_seconds: float = 3600.0) -> None:
        self._mttr_threshold = mttr_threshold_seconds

    def calculate(
        self,
        *,
        resolution_rate: float = 100.0,
        mttr_seconds: float = 0.0,
        team_roles_filled: int = 0,
        team_roles_total: int = 6,
        pipeline_pass_rate: float = 100.0,
        compliance_score: float = 100.0,
    ) -> dict[str, Any]:
        # Normalisiere alle Faktoren auf 0-100
        f_resolution = min(100, resolution_rate)
        f_mttr = 100.0 if mttr_seconds == 0 else max(0, 100 - (mttr_seconds / self._mttr_threshold * 100))
        f_team = (team_roles_filled / max(1, team_roles_total)) * 100
        f_pipeline = min(100, pipeline_pass_rate)
        f_compliance = min(100, compliance_score)

        total = (
            f_resolution * self.WEIGHTS["resolution_rate"]
            + f_mttr * self.WEIGHTS["mttr_health"]
            + f_team * self.WEIGHTS["team_coverage"]
            + f_pipeline * self.WEIGHTS["pipeline_pass"]
            + f_compliance * self.WEIGHTS["compliance"]
        )

        level = (
            "excellent" if total >= 90 else
            "good" if total >= 70 else
            "moderate" if total >= 50 else
            "poor" if total >= 30 else
            "critical"
        )

        return {
            "posture_score": round(total, 1),
            "level": level,
            "breakdown": {
                "resolution_rate": round(f_resolution, 1),
                "mttr_health": round(f_mttr, 1),
                "team_coverage": round(f_team, 1),
                "pipeline_pass": round(f_pipeline, 1),
                "compliance": round(f_compliance, 1),
            },
            "weights": dict(self.WEIGHTS),
        }
