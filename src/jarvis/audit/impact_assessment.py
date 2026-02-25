"""Jarvis · AI Impact Assessment & Partizipative Governance.

Datenschutz-Folgenabschätzung (DPIA) für KI-Systeme:

  - ImpactDimension:       Bewertungsdimensionen (Grundrechte, Umwelt, Gesellschaft...)
  - ImpactAssessment:      Strukturierte Folgenabschätzung nach EU-AI-Act Art. 9
  - StakeholderRegistry:   Betroffene Parteien registrieren und einbinden
  - EthicsBoard:           Ethik-Gremium mit Abstimmungen und Veto-Recht
  - MitigationTracker:     Maßnahmen zur Risikominderung verfolgen
  - ImpactAssessor:        Hauptklasse

Architektur-Bibel: §16.5 (Impact Assessment), §16.6 (Partizipative Governance)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ============================================================================
# Impact Dimensions
# ============================================================================


class ImpactDimension(Enum):
    FUNDAMENTAL_RIGHTS = "fundamental_rights"    # Grundrechte
    PRIVACY = "privacy"                          # Datenschutz
    SAFETY = "safety"                            # Sicherheit
    TRANSPARENCY = "transparency"                # Transparenz
    DISCRIMINATION = "discrimination"            # Diskriminierung
    AUTONOMY = "autonomy"                        # Menschliche Autonomie
    ENVIRONMENT = "environment"                  # Umweltauswirkung
    LABOR = "labor"                              # Arbeitsmarkt
    DEMOCRACY = "democracy"                      # Demokratie
    CHILDREN = "children"                        # Kinderschutz


class ImpactSeverity(Enum):
    NEGLIGIBLE = "negligible"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ImpactLikelihood(Enum):
    RARE = "rare"
    UNLIKELY = "unlikely"
    POSSIBLE = "possible"
    LIKELY = "likely"
    ALMOST_CERTAIN = "almost_certain"


# ============================================================================
# Impact Assessment
# ============================================================================


@dataclass
class DimensionScore:
    """Bewertung einer einzelnen Impact-Dimension."""

    dimension: ImpactDimension
    severity: ImpactSeverity
    likelihood: ImpactLikelihood
    description: str = ""
    affected_groups: list[str] = field(default_factory=list)
    existing_mitigations: list[str] = field(default_factory=list)
    residual_risk: str = "medium"

    @property
    def risk_score(self) -> int:
        """Risiko-Matrix: Severity × Likelihood → 1-25."""
        s_map = {"negligible": 1, "low": 2, "moderate": 3, "high": 4, "critical": 5}
        l_map = {"rare": 1, "unlikely": 2, "possible": 3, "likely": 4, "almost_certain": 5}
        return s_map.get(self.severity.value, 1) * l_map.get(self.likelihood.value, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "severity": self.severity.value,
            "likelihood": self.likelihood.value,
            "risk_score": self.risk_score,
            "affected_groups": self.affected_groups,
            "residual_risk": self.residual_risk,
        }


@dataclass
class ImpactAssessment:
    """Vollständige Folgenabschätzung für ein KI-System."""

    assessment_id: str
    system_name: str
    purpose: str
    assessor: str
    scores: list[DimensionScore] = field(default_factory=list)
    stakeholder_consultations: list[str] = field(default_factory=list)
    conclusion: str = ""
    approved: bool = False
    approved_by: str = ""
    created_at: str = ""
    reviewed_at: str = ""

    @property
    def overall_risk(self) -> int:
        """Höchster Einzelrisiko-Score."""
        return max((s.risk_score for s in self.scores), default=0)

    @property
    def risk_level(self) -> str:
        score = self.overall_risk
        if score >= 20:
            return "critical"
        if score >= 12:
            return "high"
        if score >= 6:
            return "moderate"
        if score >= 2:
            return "low"
        return "negligible"

    @property
    def high_risk_dimensions(self) -> list[DimensionScore]:
        return [s for s in self.scores if s.risk_score >= 12]

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "system": self.system_name,
            "purpose": self.purpose,
            "overall_risk": self.overall_risk,
            "risk_level": self.risk_level,
            "dimensions_assessed": len(self.scores),
            "high_risk_count": len(self.high_risk_dimensions),
            "stakeholders_consulted": len(self.stakeholder_consultations),
            "approved": self.approved,
        }


# ============================================================================
# Stakeholder Registry
# ============================================================================


class StakeholderRole(Enum):
    AFFECTED_PERSON = "affected_person"      # Direkt Betroffene
    DATA_SUBJECT = "data_subject"            # Datensubjekte
    OPERATOR = "operator"                    # Betreiber
    DEVELOPER = "developer"                  # Entwickler
    REGULATOR = "regulator"                  # Aufsichtsbehörde
    CIVIL_SOCIETY = "civil_society"          # Zivilgesellschaft
    ETHICS_EXPERT = "ethics_expert"          # Ethik-Expert:in
    DOMAIN_EXPERT = "domain_expert"          # Fachexpert:in
    WORKER_REPRESENTATIVE = "worker_rep"     # Arbeitnehmervertretung


@dataclass
class Stakeholder:
    """Ein:e Stakeholder:in im Impact-Assessment-Prozess."""

    stakeholder_id: str
    name: str
    role: StakeholderRole
    organization: str = ""
    email: str = ""
    consulted: bool = False
    feedback: str = ""
    consultation_date: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.stakeholder_id,
            "name": self.name,
            "role": self.role.value,
            "organization": self.organization,
            "consulted": self.consulted,
        }


class StakeholderRegistry:
    """Registrierung und Einbindung von Stakeholdern."""

    def __init__(self) -> None:
        self._stakeholders: dict[str, Stakeholder] = {}
        self._counter = 0

    def register(
        self,
        name: str,
        role: StakeholderRole,
        *,
        organization: str = "",
        email: str = "",
    ) -> Stakeholder:
        self._counter += 1
        sh = Stakeholder(
            stakeholder_id=f"SH-{self._counter:04d}",
            name=name,
            role=role,
            organization=organization,
            email=email,
        )
        self._stakeholders[sh.stakeholder_id] = sh
        return sh

    def record_consultation(
        self, stakeholder_id: str, feedback: str
    ) -> bool:
        sh = self._stakeholders.get(stakeholder_id)
        if not sh:
            return False
        sh.consulted = True
        sh.feedback = feedback
        sh.consultation_date = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return True

    def by_role(self, role: StakeholderRole) -> list[Stakeholder]:
        return [s for s in self._stakeholders.values() if s.role == role]

    def unconsulted(self) -> list[Stakeholder]:
        return [s for s in self._stakeholders.values() if not s.consulted]

    @property
    def count(self) -> int:
        return len(self._stakeholders)

    def consultation_rate(self) -> float:
        if not self._stakeholders:
            return 0.0
        consulted = sum(1 for s in self._stakeholders.values() if s.consulted)
        return round(consulted / len(self._stakeholders) * 100, 1)

    def stats(self) -> dict[str, Any]:
        all_sh = list(self._stakeholders.values())
        return {
            "total": len(all_sh),
            "consulted": sum(1 for s in all_sh if s.consulted),
            "consultation_rate": self.consultation_rate(),
            "by_role": {
                r.value: sum(1 for s in all_sh if s.role == r)
                for r in StakeholderRole
                if any(s.role == r for s in all_sh)
            },
        }


# ============================================================================
# Ethics Board (Partizipative Governance)
# ============================================================================


class VoteOption(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"
    CONDITIONAL = "conditional"   # Bedingte Zustimmung


@dataclass
class BoardVote:
    """Eine Abstimmung im Ethik-Gremium."""

    voter_id: str
    voter_name: str
    vote: VoteOption
    comment: str = ""
    conditions: list[str] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "voter": self.voter_name,
            "vote": self.vote.value,
            "has_conditions": bool(self.conditions),
        }


@dataclass
class BoardDecision:
    """Entscheidung des Ethik-Gremiums."""

    decision_id: str
    subject: str
    votes: list[BoardVote] = field(default_factory=list)
    final_decision: str = ""   # approved, rejected, deferred
    requires_conditions: list[str] = field(default_factory=list)
    decided_at: str = ""

    @property
    def vote_count(self) -> dict[str, int]:
        return {
            opt.value: sum(1 for v in self.votes if v.vote == opt)
            for opt in VoteOption
        }

    @property
    def has_veto(self) -> bool:
        """Veto = mindestens 2 Reject-Stimmen."""
        return sum(1 for v in self.votes if v.vote == VoteOption.REJECT) >= 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "subject": self.subject,
            "votes": self.vote_count,
            "final": self.final_decision,
            "veto": self.has_veto,
            "conditions": self.requires_conditions,
        }


class EthicsBoard:
    """Ethik-Gremium mit Abstimmungen und Veto-Recht.

    Entscheidungen über Hochrisiko-KI werden demokratisch getroffen.
    Mindestens 3 Board-Mitglieder müssen abstimmen.
    """

    def __init__(self) -> None:
        self._members: list[Stakeholder] = []
        self._decisions: list[BoardDecision] = []
        self._counter = 0
        self._min_quorum = 3

    def add_member(self, member: Stakeholder) -> None:
        self._members.append(member)

    def remove_member(self, stakeholder_id: str) -> bool:
        before = len(self._members)
        self._members = [m for m in self._members if m.stakeholder_id != stakeholder_id]
        return len(self._members) < before

    def create_decision(self, subject: str) -> BoardDecision:
        self._counter += 1
        decision = BoardDecision(
            decision_id=f"BD-{self._counter:04d}",
            subject=subject,
        )
        self._decisions.append(decision)
        return decision

    def cast_vote(
        self,
        decision_id: str,
        voter_id: str,
        voter_name: str,
        vote: VoteOption,
        *,
        comment: str = "",
        conditions: list[str] | None = None,
    ) -> bool:
        decision = next((d for d in self._decisions if d.decision_id == decision_id), None)
        if not decision:
            return False
        # Keine Doppelabstimmung
        if any(v.voter_id == voter_id for v in decision.votes):
            return False
        decision.votes.append(BoardVote(
            voter_id=voter_id,
            voter_name=voter_name,
            vote=vote,
            comment=comment,
            conditions=conditions or [],
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        ))
        return True

    def finalize(self, decision_id: str) -> BoardDecision | None:
        """Finalisiert eine Abstimmung."""
        decision = next((d for d in self._decisions if d.decision_id == decision_id), None)
        if not decision:
            return None
        if len(decision.votes) < self._min_quorum:
            decision.final_decision = "deferred_no_quorum"
            return decision

        if decision.has_veto:
            decision.final_decision = "rejected_veto"
        else:
            approvals = sum(1 for v in decision.votes if v.vote in (VoteOption.APPROVE, VoteOption.CONDITIONAL))
            if approvals > len(decision.votes) / 2:
                # Bedingungen sammeln
                for v in decision.votes:
                    if v.vote == VoteOption.CONDITIONAL:
                        decision.requires_conditions.extend(v.conditions)
                decision.final_decision = "approved" if not decision.requires_conditions else "approved_conditional"
            else:
                decision.final_decision = "rejected"

        decision.decided_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return decision

    @property
    def member_count(self) -> int:
        return len(self._members)

    def all_decisions(self) -> list[BoardDecision]:
        return list(self._decisions)

    def stats(self) -> dict[str, Any]:
        decisions = self._decisions
        finalized = [d for d in decisions if d.final_decision]
        return {
            "board_members": len(self._members),
            "total_decisions": len(decisions),
            "finalized": len(finalized),
            "approved": sum(1 for d in finalized if "approved" in d.final_decision),
            "rejected": sum(1 for d in finalized if "rejected" in d.final_decision),
            "vetoed": sum(1 for d in finalized if d.has_veto),
            "min_quorum": self._min_quorum,
        }


# ============================================================================
# Mitigation Tracker
# ============================================================================


class MitigationStatus(Enum):
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    IMPLEMENTED = "implemented"
    VERIFIED = "verified"
    INEFFECTIVE = "ineffective"


@dataclass
class Mitigation:
    """Eine Risikominderungsmaßnahme."""

    mitigation_id: str
    assessment_id: str
    dimension: ImpactDimension
    description: str
    status: MitigationStatus = MitigationStatus.PLANNED
    responsible: str = ""
    deadline: str = ""
    effectiveness: float = 0.0   # 0-1
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.mitigation_id,
            "dimension": self.dimension.value,
            "description": self.description,
            "status": self.status.value,
            "effectiveness": self.effectiveness,
        }


class MitigationTracker:
    """Verfolgt Maßnahmen zur Risikominderung."""

    def __init__(self) -> None:
        self._mitigations: dict[str, Mitigation] = {}
        self._counter = 0

    def add(
        self,
        assessment_id: str,
        dimension: ImpactDimension,
        description: str,
        *,
        responsible: str = "",
        deadline: str = "",
    ) -> Mitigation:
        self._counter += 1
        m = Mitigation(
            mitigation_id=f"MIT-{self._counter:04d}",
            assessment_id=assessment_id,
            dimension=dimension,
            description=description,
            responsible=responsible,
            deadline=deadline,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self._mitigations[m.mitigation_id] = m
        return m

    def update_status(self, mitigation_id: str, status: MitigationStatus, effectiveness: float = 0.0) -> bool:
        m = self._mitigations.get(mitigation_id)
        if not m:
            return False
        m.status = status
        m.effectiveness = effectiveness
        return True

    def by_assessment(self, assessment_id: str) -> list[Mitigation]:
        return [m for m in self._mitigations.values() if m.assessment_id == assessment_id]

    def overdue(self) -> list[Mitigation]:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return [
            m for m in self._mitigations.values()
            if m.deadline and m.deadline < now and m.status in (MitigationStatus.PLANNED, MitigationStatus.IN_PROGRESS)
        ]

    @property
    def count(self) -> int:
        return len(self._mitigations)

    def completion_rate(self) -> float:
        if not self._mitigations:
            return 0.0
        done = sum(1 for m in self._mitigations.values() if m.status in (MitigationStatus.IMPLEMENTED, MitigationStatus.VERIFIED))
        return round(done / len(self._mitigations) * 100, 1)

    def stats(self) -> dict[str, Any]:
        all_m = list(self._mitigations.values())
        return {
            "total": len(all_m),
            "completion_rate": self.completion_rate(),
            "by_status": {
                s.value: sum(1 for m in all_m if m.status == s)
                for s in MitigationStatus
                if any(m.status == s for m in all_m)
            },
        }


# ============================================================================
# Impact Assessor (Hauptklasse)
# ============================================================================


class ImpactAssessor:
    """Hauptklasse: AI Impact Assessment + Partizipative Governance."""

    def __init__(self) -> None:
        self._assessments: list[ImpactAssessment] = []
        self._stakeholders = StakeholderRegistry()
        self._board = EthicsBoard()
        self._mitigations = MitigationTracker()
        self._counter = 0

    @property
    def stakeholders(self) -> StakeholderRegistry:
        return self._stakeholders

    @property
    def board(self) -> EthicsBoard:
        return self._board

    @property
    def mitigations(self) -> MitigationTracker:
        return self._mitigations

    def create_assessment(
        self,
        system_name: str,
        purpose: str,
        assessor: str,
        scores: list[DimensionScore] | None = None,
    ) -> ImpactAssessment:
        self._counter += 1
        assessment = ImpactAssessment(
            assessment_id=f"IA-{self._counter:04d}",
            system_name=system_name,
            purpose=purpose,
            assessor=assessor,
            scores=scores or [],
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self._assessments.append(assessment)
        return assessment

    def assess_jarvis_insurance(self) -> ImpactAssessment:
        """Vordefinierte Folgenabschätzung für Jarvis im Versicherungskontext."""
        scores = [
            DimensionScore(
                ImpactDimension.FUNDAMENTAL_RIGHTS, ImpactSeverity.MODERATE, ImpactLikelihood.POSSIBLE,
                "KI-gestützte Versicherungsempfehlungen können Zugang zu Versicherungsschutz beeinflussen",
                ["Versicherungsnehmer", "Antragsteller"],
                ["Human-in-the-Loop", "Transparenzpflicht"],
                "medium",
            ),
            DimensionScore(
                ImpactDimension.PRIVACY, ImpactSeverity.HIGH, ImpactLikelihood.LIKELY,
                "Verarbeitung sensibler Gesundheitsdaten bei BU-Beratung",
                ["Kunden", "Interessenten"],
                ["DSGVO-Konformität", "Datensparsamkeit", "Verschlüsselung"],
                "low",
            ),
            DimensionScore(
                ImpactDimension.DISCRIMINATION, ImpactSeverity.HIGH, ImpactLikelihood.POSSIBLE,
                "Risiko der Benachteiligung bestimmter Berufsgruppen oder Altersklassen",
                ["Ältere Antragsteller", "Risikoberufe"],
                ["Fairness-Audits", "Bias-Detektion"],
                "medium",
            ),
            DimensionScore(
                ImpactDimension.TRANSPARENCY, ImpactSeverity.MODERATE, ImpactLikelihood.LIKELY,
                "Kunden müssen verstehen warum eine bestimmte Versicherung empfohlen wird",
                ["Alle Kunden"],
                ["DecisionExplainer", "Art. 52 Transparenz"],
                "low",
            ),
            DimensionScore(
                ImpactDimension.AUTONOMY, ImpactSeverity.LOW, ImpactLikelihood.POSSIBLE,
                "Empfehlungen könnten Entscheidungsfreiheit einschränken",
                ["Kunden"],
                ["Alternativen-Anzeige", "Keine automatischen Abschlüsse"],
                "low",
            ),
        ]
        return self.create_assessment(
            "Jarvis Versicherungsberater",
            "KI-gestützte Beratung für BU, bAV und weitere Versicherungsprodukte",
            "system",
            scores,
        )

    def all_assessments(self) -> list[ImpactAssessment]:
        return list(self._assessments)

    def high_risk_assessments(self) -> list[ImpactAssessment]:
        return [a for a in self._assessments if a.risk_level in ("high", "critical")]

    def stats(self) -> dict[str, Any]:
        return {
            "total_assessments": len(self._assessments),
            "high_risk": len(self.high_risk_assessments()),
            "stakeholders": self._stakeholders.stats(),
            "board": self._board.stats(),
            "mitigations": self._mitigations.stats(),
        }
