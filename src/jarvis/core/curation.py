"""Jarvis · Zentrales Kurations-Board & Erweiterte Governance.

Strengere Ecosystem-Kontrolle und erweiterte Transparenz:

  - SkillReview:           Pull-Request-basiertes Skill-Review
  - CurationBoard:         Zentrales Kurations-Board mit Audit-Trail
  - DiversityAuditor:      Diversity-Audits für Agent-Entscheidungen
  - CrossAgentBudget:      Finanzflüsse zwischen föderierten Agenten
  - DecisionExplainer:     Zeigt Alternativen und Risiken für Entscheidungen
  - GovernanceHub:         Hauptklasse

Architektur-Bibel: §15.2 (Ecosystem-Governance), §16.3 (Transparency)
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ============================================================================
# Skill Review (PR-basiert)
# ============================================================================


class ReviewStatus(Enum):
    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    CHANGES_REQUESTED = "changes_requested"
    QUARANTINED = "quarantined"


class ReviewFlag(Enum):
    SECURITY_RISK = "security_risk"
    PRIVACY_CONCERN = "privacy_concern"
    QUALITY_ISSUE = "quality_issue"
    LICENSE_VIOLATION = "license_violation"
    MALWARE_DETECTED = "malware_detected"
    EXCESSIVE_PERMISSIONS = "excessive_permissions"
    UNDOCUMENTED_BEHAVIOR = "undocumented_behavior"


@dataclass
class ReviewComment:
    """Ein Kommentar in einem Skill-Review."""

    author: str
    text: str
    timestamp: str = ""
    is_blocking: bool = False


@dataclass
class SkillReview:
    """Ein PR-basiertes Review für einen Skill."""

    review_id: str
    skill_id: str
    skill_name: str
    submitter: str
    status: ReviewStatus = ReviewStatus.PENDING
    flags: list[ReviewFlag] = field(default_factory=list)
    comments: list[ReviewComment] = field(default_factory=list)
    reviewers: list[str] = field(default_factory=list)
    auto_scan_passed: bool = False
    manual_review_required: bool = True
    submitted_at: str = ""
    decided_at: str = ""
    decision_by: str = ""

    @property
    def blocking_comments(self) -> list[ReviewComment]:
        return [c for c in self.comments if c.is_blocking]

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_id": self.review_id,
            "skill": self.skill_name,
            "submitter": self.submitter,
            "status": self.status.value,
            "flags": [f.value for f in self.flags],
            "comments": len(self.comments),
            "blocking": len(self.blocking_comments),
            "auto_scan": self.auto_scan_passed,
        }


# ============================================================================
# Curation Board
# ============================================================================


class CurationBoard:
    """Zentrales Kurations-Board für das Skill-Ecosystem.

    Jeder neue Skill durchläuft:
    1. Automatischen Security-Scan
    2. Lizenz-Prüfung
    3. Manuelles Review durch Board-Mitglieder
    4. Freigabe oder Ablehnung
    """

    def __init__(self) -> None:
        self._reviews: dict[str, SkillReview] = {}
        self._board_members: list[str] = []
        self._counter = 0

    def submit(
        self,
        skill_id: str,
        skill_name: str,
        submitter: str,
        *,
        auto_scan_result: bool = False,
    ) -> SkillReview:
        """Reicht einen neuen Skill zur Prüfung ein."""
        self._counter += 1
        review = SkillReview(
            review_id=f"REV-{self._counter:04d}",
            skill_id=skill_id,
            skill_name=skill_name,
            submitter=submitter,
            auto_scan_passed=auto_scan_result,
            submitted_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        # Auto-Flagging
        if not auto_scan_result:
            review.flags.append(ReviewFlag.SECURITY_RISK)
            review.manual_review_required = True

        self._reviews[review.review_id] = review
        return review

    def assign_reviewer(self, review_id: str, reviewer: str) -> bool:
        review = self._reviews.get(review_id)
        if not review:
            return False
        review.reviewers.append(reviewer)
        review.status = ReviewStatus.IN_REVIEW
        return True

    def add_comment(
        self, review_id: str, author: str, text: str, *, blocking: bool = False
    ) -> bool:
        review = self._reviews.get(review_id)
        if not review:
            return False
        review.comments.append(ReviewComment(
            author=author,
            text=text,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            is_blocking=blocking,
        ))
        return True

    def add_flag(self, review_id: str, flag: ReviewFlag) -> bool:
        review = self._reviews.get(review_id)
        if not review:
            return False
        if flag not in review.flags:
            review.flags.append(flag)
        return True

    def approve(self, review_id: str, approved_by: str) -> bool:
        review = self._reviews.get(review_id)
        if not review:
            return False
        if review.blocking_comments:
            return False  # Kann nicht freigegeben werden mit blockierenden Kommentaren
        review.status = ReviewStatus.APPROVED
        review.decided_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        review.decision_by = approved_by
        return True

    def reject(self, review_id: str, rejected_by: str, reason: str = "") -> bool:
        review = self._reviews.get(review_id)
        if not review:
            return False
        review.status = ReviewStatus.REJECTED
        review.decided_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        review.decision_by = rejected_by
        if reason:
            review.comments.append(ReviewComment(
                author=rejected_by, text=f"Abgelehnt: {reason}",
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), is_blocking=True,
            ))
        return True

    def quarantine(self, review_id: str, by: str) -> bool:
        review = self._reviews.get(review_id)
        if not review:
            return False
        review.status = ReviewStatus.QUARANTINED
        review.decided_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        review.decision_by = by
        return True

    def pending_reviews(self) -> list[SkillReview]:
        return [r for r in self._reviews.values() if r.status in (ReviewStatus.PENDING, ReviewStatus.IN_REVIEW)]

    def flagged_reviews(self) -> list[SkillReview]:
        return [r for r in self._reviews.values() if r.flags]

    def get(self, review_id: str) -> SkillReview | None:
        return self._reviews.get(review_id)

    def add_board_member(self, member: str) -> None:
        if member not in self._board_members:
            self._board_members.append(member)

    @property
    def review_count(self) -> int:
        return len(self._reviews)

    def stats(self) -> dict[str, Any]:
        reviews = list(self._reviews.values())
        return {
            "total_reviews": len(reviews),
            "pending": sum(1 for r in reviews if r.status == ReviewStatus.PENDING),
            "approved": sum(1 for r in reviews if r.status == ReviewStatus.APPROVED),
            "rejected": sum(1 for r in reviews if r.status == ReviewStatus.REJECTED),
            "quarantined": sum(1 for r in reviews if r.status == ReviewStatus.QUARANTINED),
            "flagged": sum(1 for r in reviews if r.flags),
            "board_members": len(self._board_members),
            "approval_rate": (
                round(sum(1 for r in reviews if r.status == ReviewStatus.APPROVED)
                      / len(reviews) * 100, 1) if reviews else 0
            ),
        }


# ============================================================================
# Diversity Auditor
# ============================================================================


class DiversityDimension(Enum):
    GENDER = "gender"
    AGE = "age"
    ETHNICITY = "ethnicity"
    LANGUAGE = "language"
    DISABILITY = "disability"
    SOCIOECONOMIC = "socioeconomic"
    GEOGRAPHIC = "geographic"


@dataclass
class DiversityAuditResult:
    """Ergebnis eines Diversity-Audits."""

    audit_id: str
    dimension: DiversityDimension
    score: float            # 0-100 (100 = perfekte Gleichbehandlung)
    findings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    sample_size: int = 0
    timestamp: str = ""

    @property
    def passed(self) -> bool:
        return self.score >= 70.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "dimension": self.dimension.value,
            "score": self.score,
            "passed": self.passed,
            "findings": len(self.findings),
            "sample_size": self.sample_size,
        }


class DiversityAuditor:
    """Prüft Agent-Entscheidungen auf Gleichbehandlung.

    Geht über einfache Bias-Detektion hinaus: prüft ob verschiedene
    demografische Gruppen gleich behandelt werden.
    """

    def __init__(self) -> None:
        self._audits: list[DiversityAuditResult] = []
        self._counter = 0

    def audit_responses(
        self,
        dimension: DiversityDimension,
        group_a_scores: list[float],
        group_b_scores: list[float],
        *,
        label_a: str = "Gruppe A",
        label_b: str = "Gruppe B",
    ) -> DiversityAuditResult:
        """Vergleicht Antwortqualität zwischen zwei Gruppen."""
        self._counter += 1

        if not group_a_scores or not group_b_scores:
            return DiversityAuditResult(
                audit_id=f"DIV-{self._counter:04d}",
                dimension=dimension,
                score=0,
                findings=["Keine Daten für Vergleich"],
                sample_size=0,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )

        avg_a = sum(group_a_scores) / len(group_a_scores)
        avg_b = sum(group_b_scores) / len(group_b_scores)
        max_avg = max(avg_a, avg_b)
        diff = abs(avg_a - avg_b)

        # Score: 100 = identisch, 0 = komplett unterschiedlich
        score = max(0, 100 - (diff / max_avg * 100)) if max_avg > 0 else 100

        findings = []
        recommendations = []
        if diff > 0.2 * max_avg:  # > 20% Differenz
            findings.append(
                f"Signifikante Differenz: {label_a} ({avg_a:.2f}) vs {label_b} ({avg_b:.2f})"
            )
            recommendations.append(
                f"Ursachenanalyse für {dimension.value}-Ungleichbehandlung durchführen"
            )
        if diff > 0.4 * max_avg:
            findings.append("Schwerwiegende Ungleichbehandlung erkannt")
            recommendations.append("Sofortige Korrekturmaßnahmen erforderlich")

        result = DiversityAuditResult(
            audit_id=f"DIV-{self._counter:04d}",
            dimension=dimension,
            score=round(score, 1),
            findings=findings,
            recommendations=recommendations,
            sample_size=len(group_a_scores) + len(group_b_scores),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self._audits.append(result)
        return result

    def all_audits(self) -> list[DiversityAuditResult]:
        return list(self._audits)

    @property
    def audit_count(self) -> int:
        return len(self._audits)

    def overall_score(self) -> float:
        if not self._audits:
            return 0.0
        return round(sum(a.score for a in self._audits) / len(self._audits), 1)

    def stats(self) -> dict[str, Any]:
        audits = self._audits
        return {
            "total_audits": len(audits),
            "overall_score": self.overall_score(),
            "passed": sum(1 for a in audits if a.passed),
            "failed": sum(1 for a in audits if not a.passed),
            "by_dimension": {
                d.value: round(
                    sum(a.score for a in audits if a.dimension == d) /
                    max(sum(1 for a in audits if a.dimension == d), 1), 1
                )
                for d in DiversityDimension
                if any(a.dimension == d for a in audits)
            },
        }


# ============================================================================
# Cross-Agent Budget (Finanzflüsse)
# ============================================================================


@dataclass
class BudgetTransfer:
    """Ein Finanztransfer zwischen Agenten."""

    transfer_id: str
    from_agent: str
    to_agent: str
    amount_eur: float
    reason: str
    timestamp: str = ""
    approved: bool = False
    approved_by: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "transfer_id": self.transfer_id,
            "from": self.from_agent,
            "to": self.to_agent,
            "amount": self.amount_eur,
            "reason": self.reason,
            "approved": self.approved,
        }


class CrossAgentBudget:
    """Verwaltet Finanzflüsse zwischen föderierten Agenten."""

    def __init__(self, max_single_transfer: float = 50.0, daily_limit: float = 200.0) -> None:
        self._max_single = max_single_transfer
        self._daily_limit = daily_limit
        self._transfers: list[BudgetTransfer] = []
        self._counter = 0

    def request_transfer(
        self,
        from_agent: str,
        to_agent: str,
        amount_eur: float,
        reason: str,
    ) -> BudgetTransfer:
        """Fordert einen Transfer an."""
        self._counter += 1
        transfer = BudgetTransfer(
            transfer_id=f"TRF-{self._counter:04d}",
            from_agent=from_agent,
            to_agent=to_agent,
            amount_eur=amount_eur,
            reason=reason,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        # Auto-Approve wenn unter Limit
        if amount_eur <= self._max_single:
            daily_spent = self._daily_spent(from_agent)
            if daily_spent + amount_eur <= self._daily_limit:
                transfer.approved = True
                transfer.approved_by = "auto"

        self._transfers.append(transfer)
        return transfer

    def approve(self, transfer_id: str, approved_by: str) -> bool:
        for t in self._transfers:
            if t.transfer_id == transfer_id and not t.approved:
                t.approved = True
                t.approved_by = approved_by
                return True
        return False

    def _daily_spent(self, agent_id: str) -> float:
        today = time.strftime("%Y-%m-%d", time.gmtime())
        return sum(
            t.amount_eur for t in self._transfers
            if t.from_agent == agent_id and t.approved and t.timestamp.startswith(today)
        )

    def transfers_by_agent(self, agent_id: str) -> list[BudgetTransfer]:
        return [t for t in self._transfers if t.from_agent == agent_id or t.to_agent == agent_id]

    def pending_transfers(self) -> list[BudgetTransfer]:
        return [t for t in self._transfers if not t.approved]

    @property
    def transfer_count(self) -> int:
        return len(self._transfers)

    def stats(self) -> dict[str, Any]:
        transfers = self._transfers
        return {
            "total_transfers": len(transfers),
            "approved": sum(1 for t in transfers if t.approved),
            "pending": sum(1 for t in transfers if not t.approved),
            "total_volume_eur": round(sum(t.amount_eur for t in transfers if t.approved), 2),
            "max_single_limit": self._max_single,
            "daily_limit": self._daily_limit,
        }


# ============================================================================
# Decision Explainer (Punkt 9: Alternativen + Risiken)
# ============================================================================


@dataclass
class DecisionAlternative:
    """Eine alternative Entscheidungsoption."""

    option_id: str
    description: str
    confidence: float       # 0-1
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)
    risk_level: str = "low"  # low, medium, high
    estimated_cost_eur: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "option_id": self.option_id,
            "description": self.description,
            "confidence": self.confidence,
            "pros": self.pros,
            "cons": self.cons,
            "risk": self.risk_level,
            "cost": self.estimated_cost_eur,
        }


@dataclass
class DecisionExplanation:
    """Vollständige Erklärung einer Agent-Entscheidung."""

    decision_id: str
    question: str
    chosen_option: DecisionAlternative
    alternatives: list[DecisionAlternative] = field(default_factory=list)
    reasoning: str = ""
    data_sources: list[str] = field(default_factory=list)
    confidence: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "question": self.question,
            "chosen": self.chosen_option.to_dict(),
            "alternatives": [a.to_dict() for a in self.alternatives],
            "reasoning": self.reasoning,
            "sources": self.data_sources,
            "confidence": self.confidence,
        }


class DecisionExplainer:
    """Macht Agent-Entscheidungen transparent mit Alternativen und Risiken."""

    def __init__(self) -> None:
        self._explanations: list[DecisionExplanation] = []
        self._counter = 0

    def explain(
        self,
        question: str,
        chosen: DecisionAlternative,
        alternatives: list[DecisionAlternative] | None = None,
        *,
        reasoning: str = "",
        sources: list[str] | None = None,
    ) -> DecisionExplanation:
        """Erstellt eine Entscheidungserklärung."""
        self._counter += 1
        explanation = DecisionExplanation(
            decision_id=f"DEC-{self._counter:04d}",
            question=question,
            chosen_option=chosen,
            alternatives=alternatives or [],
            reasoning=reasoning,
            data_sources=sources or [],
            confidence=chosen.confidence,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self._explanations.append(explanation)
        return explanation

    def all_explanations(self) -> list[DecisionExplanation]:
        return list(self._explanations)

    @property
    def explanation_count(self) -> int:
        return len(self._explanations)

    def avg_confidence(self) -> float:
        if not self._explanations:
            return 0.0
        return round(sum(e.confidence for e in self._explanations) / len(self._explanations), 3)

    def stats(self) -> dict[str, Any]:
        return {
            "total_explanations": len(self._explanations),
            "avg_confidence": self.avg_confidence(),
            "with_alternatives": sum(1 for e in self._explanations if e.alternatives),
        }


# ============================================================================
# Governance Hub (Hauptklasse)
# ============================================================================


class GovernanceHub:
    """Hauptklasse: Kurations-Board + Diversity + Budget + Explainability."""

    def __init__(self) -> None:
        self._curation = CurationBoard()
        self._diversity = DiversityAuditor()
        self._budget = CrossAgentBudget()
        self._explainer = DecisionExplainer()

    @property
    def curation(self) -> CurationBoard:
        return self._curation

    @property
    def diversity(self) -> DiversityAuditor:
        return self._diversity

    @property
    def budget(self) -> CrossAgentBudget:
        return self._budget

    @property
    def explainer(self) -> DecisionExplainer:
        return self._explainer

    def ecosystem_health(self) -> dict[str, Any]:
        """Gesamt-Gesundheit des Ecosystems."""
        curation = self._curation.stats()
        diversity = self._diversity.stats()
        budget = self._budget.stats()

        return {
            "skill_reviews": curation["total_reviews"],
            "approval_rate": curation["approval_rate"],
            "diversity_score": diversity["overall_score"],
            "budget_volume": budget["total_volume_eur"],
            "pending_reviews": curation["pending"],
            "pending_transfers": budget["pending"],
        }

    def stats(self) -> dict[str, Any]:
        return {
            "curation": self._curation.stats(),
            "diversity": self._diversity.stats(),
            "budget": self._budget.stats(),
            "explainer": self._explainer.stats(),
        }
