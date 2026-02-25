"""Jarvis · Marketplace-Governance.

Strengeres Bewertungs-/Reputationssystem für das Skill-Ecosystem:

  - ReputationEngine:    Trust-Score pro Publisher + Skill
  - SkillRecallManager:  Zentraler Rückruf bösartiger Skills
  - AbuseReporter:       Melde-System für verdächtige Skills
  - GovernancePolicy:    Regel-Engine für automatische Aktionen
  - ReviewQueue:         Warteschlange für manuelle Reviews

Architektur-Bibel: §7.4 (Marketplace-Security), §14.2 (Supply-Chain)
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ============================================================================
# Trust & Reputation
# ============================================================================


class TrustLevel(Enum):
    UNTRUSTED = "untrusted"       # Score 0-20
    LOW = "low"                    # Score 21-40
    MODERATE = "moderate"          # Score 41-60
    HIGH = "high"                  # Score 61-80
    VERIFIED = "verified"          # Score 81-100


@dataclass
class ReputationScore:
    """Reputations-Score für Publisher oder Skill."""

    entity_id: str
    entity_type: str  # "publisher" oder "skill"
    score: float = 50.0  # 0-100
    total_reviews: int = 0
    positive_reviews: int = 0
    negative_reviews: int = 0
    abuse_reports: int = 0
    recalls: int = 0
    skills_published: int = 0
    failed_security: bool = False
    last_updated: str = ""

    @property
    def trust_level(self) -> TrustLevel:
        if self.score >= 81:
            return TrustLevel.VERIFIED
        elif self.score >= 61:
            return TrustLevel.HIGH
        elif self.score >= 41:
            return TrustLevel.MODERATE
        elif self.score >= 21:
            return TrustLevel.LOW
        return TrustLevel.UNTRUSTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "score": round(self.score, 1),
            "trust_level": self.trust_level.value,
            "total_reviews": self.total_reviews,
            "positive_reviews": self.positive_reviews,
            "abuse_reports": self.abuse_reports,
            "recalls": self.recalls,
        }


class ReputationEngine:
    """Berechnet und verwaltet Trust-Scores."""

    # Score-Auswirkungen
    POSITIVE_REVIEW = +2.0
    NEGATIVE_REVIEW = -3.0
    ABUSE_REPORT = -10.0
    RECALL_PENALTY = -25.0
    SUCCESSFUL_INSTALL = +0.5
    PASSED_SECURITY_SCAN = +5.0
    FAILED_SECURITY_SCAN = -15.0

    def __init__(self) -> None:
        self._scores: dict[str, ReputationScore] = {}

    def get_or_create(self, entity_id: str, entity_type: str = "skill") -> ReputationScore:
        if entity_id not in self._scores:
            self._scores[entity_id] = ReputationScore(
                entity_id=entity_id,
                entity_type=entity_type,
                last_updated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
        return self._scores[entity_id]

    def add_review(self, entity_id: str, positive: bool) -> ReputationScore:
        score = self.get_or_create(entity_id)
        score.total_reviews += 1
        if positive:
            score.positive_reviews += 1
            score.score = min(100, score.score + self.POSITIVE_REVIEW)
        else:
            score.negative_reviews += 1
            score.score = max(0, score.score + self.NEGATIVE_REVIEW)
        score.last_updated = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return score

    def report_abuse(self, entity_id: str) -> ReputationScore:
        score = self.get_or_create(entity_id)
        score.abuse_reports += 1
        score.score = max(0, score.score + self.ABUSE_REPORT)
        score.last_updated = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return score

    def apply_recall(self, entity_id: str) -> ReputationScore:
        score = self.get_or_create(entity_id)
        score.recalls += 1
        score.score = max(0, score.score + self.RECALL_PENALTY)
        score.last_updated = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return score

    def apply_security_result(self, entity_id: str, passed: bool) -> ReputationScore:
        score = self.get_or_create(entity_id)
        delta = self.PASSED_SECURITY_SCAN if passed else self.FAILED_SECURITY_SCAN
        score.score = max(0, min(100, score.score + delta))
        score.last_updated = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return score

    def record_install(self, entity_id: str) -> ReputationScore:
        score = self.get_or_create(entity_id)
        score.score = min(100, score.score + self.SUCCESSFUL_INSTALL)
        return score

    def get_score(self, entity_id: str) -> ReputationScore | None:
        return self._scores.get(entity_id)

    def top_rated(self, n: int = 10, entity_type: str = "") -> list[ReputationScore]:
        scores = list(self._scores.values())
        if entity_type:
            scores = [s for s in scores if s.entity_type == entity_type]
        return sorted(scores, key=lambda s: s.score, reverse=True)[:n]

    def flagged(self, threshold: float = 30.0) -> list[ReputationScore]:
        return [s for s in self._scores.values() if s.score < threshold]

    @property
    def entity_count(self) -> int:
        return len(self._scores)

    def stats(self) -> dict[str, Any]:
        scores = list(self._scores.values())
        return {
            "total_entities": len(scores),
            "avg_score": sum(s.score for s in scores) / len(scores) if scores else 0,
            "flagged_count": len(self.flagged()),
            "trust_distribution": {
                level.value: sum(1 for s in scores if s.trust_level == level)
                for level in TrustLevel
            },
        }


# ============================================================================
# Skill Recall Manager
# ============================================================================


class RecallReason(Enum):
    SECURITY_VULNERABILITY = "security_vulnerability"
    MALICIOUS_BEHAVIOR = "malicious_behavior"
    DATA_THEFT = "data_theft"
    CRYPTO_MINING = "crypto_mining"
    PRIVACY_VIOLATION = "privacy_violation"
    LICENSE_VIOLATION = "license_violation"
    QUALITY_ISSUE = "quality_issue"


@dataclass
class RecallNotice:
    """Rückruf-Benachrichtigung für einen Skill."""

    recall_id: str
    skill_id: str
    skill_name: str
    reason: RecallReason
    severity: str  # critical, high, medium
    description: str
    issued_at: str
    issued_by: str = "jarvis-governance"
    affected_versions: list[str] = field(default_factory=lambda: ["all"])
    action_required: str = "Sofort deinstallieren"

    def to_dict(self) -> dict[str, Any]:
        return {
            "recall_id": self.recall_id,
            "skill_id": self.skill_id,
            "skill_name": self.skill_name,
            "reason": self.reason.value,
            "severity": self.severity,
            "description": self.description,
            "issued_at": self.issued_at,
            "action_required": self.action_required,
            "affected_versions": self.affected_versions,
        }


class SkillRecallManager:
    """Zentraler Rückruf-Manager für bösartige Skills."""

    def __init__(self, reputation: ReputationEngine | None = None) -> None:
        self._reputation = reputation or ReputationEngine()
        self._recalls: dict[str, RecallNotice] = {}
        self._blocked_skills: set[str] = set()

    def issue_recall(
        self,
        skill_id: str,
        skill_name: str,
        reason: RecallReason,
        description: str,
        severity: str = "high",
    ) -> RecallNotice:
        """Gibt einen Rückruf für einen Skill heraus."""
        recall_id = hashlib.sha256(f"recall:{skill_id}:{time.time()}".encode()).hexdigest()[:12]
        notice = RecallNotice(
            recall_id=recall_id,
            skill_id=skill_id,
            skill_name=skill_name,
            reason=reason,
            severity=severity,
            description=description,
            issued_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self._recalls[recall_id] = notice
        self._blocked_skills.add(skill_id)
        self._reputation.apply_recall(skill_id)
        return notice

    def is_recalled(self, skill_id: str) -> bool:
        return skill_id in self._blocked_skills

    def is_installable(self, skill_id: str) -> bool:
        return skill_id not in self._blocked_skills

    def lift_recall(self, skill_id: str) -> bool:
        if skill_id in self._blocked_skills:
            self._blocked_skills.discard(skill_id)
            return True
        return False

    def active_recalls(self) -> list[RecallNotice]:
        return [
            r for r in self._recalls.values()
            if r.skill_id in self._blocked_skills
        ]

    def all_recalls(self) -> list[RecallNotice]:
        return list(self._recalls.values())

    @property
    def recall_count(self) -> int:
        return len(self._recalls)

    @property
    def blocked_count(self) -> int:
        return len(self._blocked_skills)

    def stats(self) -> dict[str, Any]:
        recalls = list(self._recalls.values())
        return {
            "total_recalls": len(recalls),
            "active_blocks": len(self._blocked_skills),
            "by_reason": {
                reason.value: sum(1 for r in recalls if r.reason == reason)
                for reason in RecallReason
                if any(r.reason == reason for r in recalls)
            },
            "by_severity": {
                sev: sum(1 for r in recalls if r.severity == sev)
                for sev in ("critical", "high", "medium")
                if any(r.severity == sev for r in recalls)
            },
        }


# ============================================================================
# Abuse Reporter
# ============================================================================


@dataclass
class AbuseReport:
    """Missbrauchs-Meldung."""

    report_id: str
    skill_id: str
    reporter: str
    category: str  # "malware", "crypto", "spam", "data_theft", "other"
    description: str
    submitted_at: str
    status: str = "open"  # open, investigating, confirmed, dismissed
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "skill_id": self.skill_id,
            "category": self.category,
            "status": self.status,
            "submitted_at": self.submitted_at,
        }


class AbuseReporter:
    """Melde-System für verdächtige Skills."""

    AUTO_INVESTIGATE_THRESHOLD = 3  # Ab 3 Meldungen → auto-investigate

    def __init__(self, reputation: ReputationEngine | None = None) -> None:
        self._reputation = reputation or ReputationEngine()
        self._reports: list[AbuseReport] = []

    def submit(
        self,
        skill_id: str,
        reporter: str,
        category: str,
        description: str,
        evidence: str = "",
    ) -> AbuseReport:
        report = AbuseReport(
            report_id=hashlib.sha256(f"abuse:{skill_id}:{time.time()}".encode()).hexdigest()[:12],
            skill_id=skill_id,
            reporter=reporter,
            category=category,
            description=description,
            evidence=evidence,
            submitted_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self._reports.append(report)
        self._reputation.report_abuse(skill_id)

        # Auto-Investigate bei wiederholten Meldungen
        skill_reports = [r for r in self._reports if r.skill_id == skill_id and r.status == "open"]
        if len(skill_reports) >= self.AUTO_INVESTIGATE_THRESHOLD:
            for r in skill_reports:
                r.status = "investigating"

        return report

    def reports_for_skill(self, skill_id: str) -> list[AbuseReport]:
        return [r for r in self._reports if r.skill_id == skill_id]

    def open_reports(self) -> list[AbuseReport]:
        return [r for r in self._reports if r.status in ("open", "investigating")]

    def resolve(self, report_id: str, status: str) -> bool:
        for r in self._reports:
            if r.report_id == report_id:
                r.status = status
                return True
        return False

    @property
    def report_count(self) -> int:
        return len(self._reports)

    def stats(self) -> dict[str, Any]:
        reports = self._reports
        return {
            "total_reports": len(reports),
            "open": sum(1 for r in reports if r.status == "open"),
            "investigating": sum(1 for r in reports if r.status == "investigating"),
            "confirmed": sum(1 for r in reports if r.status == "confirmed"),
            "dismissed": sum(1 for r in reports if r.status == "dismissed"),
        }


# ============================================================================
# Governance Policy (automatische Aktionen)
# ============================================================================


@dataclass
class GovernanceRule:
    """Automatische Governance-Regel."""

    rule_id: str
    name: str
    condition: str  # "abuse_reports >= 5", "score < 20", etc.
    action: str     # "block", "recall", "flag", "notify"
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "condition": self.condition,
            "action": self.action,
            "enabled": self.enabled,
        }


class GovernancePolicy:
    """Automatische Governance-Regeln für das Ecosystem.

    Regeln werden bei jeder Reputation-Änderung geprüft.
    """

    DEFAULT_RULES = [
        GovernanceRule("GOV-001", "Auto-Block bei Score < 10", "score < 10", "block"),
        GovernanceRule("GOV-002", "Auto-Flag bei 3+ Abuse Reports", "abuse_reports >= 3", "flag"),
        GovernanceRule("GOV-003", "Auto-Recall bei Security-Fail", "failed_security", "recall"),
        GovernanceRule("GOV-004", "Notify bei Score < 30", "score < 30", "notify"),
    ]

    def __init__(self, rules: list[GovernanceRule] | None = None) -> None:
        self._rules = rules if rules is not None else list(self.DEFAULT_RULES)
        self._triggered: list[dict[str, Any]] = []

    def add_rule(self, rule: GovernanceRule) -> None:
        self._rules.append(rule)

    def evaluate(self, score: ReputationScore) -> list[dict[str, Any]]:
        """Prüft alle Regeln gegen einen Score."""
        actions: list[dict[str, Any]] = []
        for rule in self._rules:
            if not rule.enabled:
                continue
            triggered = False
            if "score < " in rule.condition:
                threshold = float(rule.condition.split("< ")[1])
                triggered = score.score < threshold
            elif "abuse_reports >= " in rule.condition:
                threshold = int(rule.condition.split(">= ")[1])
                triggered = score.abuse_reports >= threshold
            elif rule.condition == "failed_security":
                triggered = score.failed_security if hasattr(score, "failed_security") else False

            if triggered:
                action = {
                    "rule_id": rule.rule_id,
                    "rule_name": rule.name,
                    "action": rule.action,
                    "entity_id": score.entity_id,
                    "triggered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                actions.append(action)
                self._triggered.append(action)

        return actions

    @property
    def rule_count(self) -> int:
        return len(self._rules)

    def triggered_actions(self) -> list[dict[str, Any]]:
        return list(self._triggered)

    def stats(self) -> dict[str, Any]:
        return {
            "total_rules": len(self._rules),
            "enabled": sum(1 for r in self._rules if r.enabled),
            "total_triggered": len(self._triggered),
        }
