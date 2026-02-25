"""Jarvis · Ecosystem-Kontrolle & Security-Training.

Strenge Kuration und Notfall-Updates für das Skill-Ökosystem:

  - SkillCurator:          Manuelle und automatische Skill-Kuration
  - EmergencyUpdater:      Zentralisierte Notfall-Patches
  - FraudDetector:         Erkennt betrügerische/schädliche Skills
  - SecurityTrainer:       Trainingsmaterialien für Agent-Security
  - TrustBoundaryManager:  Vertrauensgrenzen zwischen Agenten
  - EcosystemController:   Hauptklasse, orchestriert alles

Architektur-Bibel: §10.2 (Marketplace), §16.2 (Ecosystem)
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ============================================================================
# Skill Curation
# ============================================================================


class CurationStatus(Enum):
    PENDING = "pending"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUSPENDED = "suspended"
    EMERGENCY_BLOCK = "emergency_block"


class ReviewCriteria(Enum):
    CODE_QUALITY = "code_quality"
    SECURITY_SCAN = "security_scan"
    PRIVACY_CHECK = "privacy_check"
    PERFORMANCE = "performance"
    DOCUMENTATION = "documentation"
    LICENSE = "license"
    DEPENDENCY_AUDIT = "dependency_audit"


@dataclass
class CurationReview:
    """Ergebnis einer Skill-Kuration."""

    review_id: str
    skill_id: str
    status: CurationStatus
    reviewer: str = ""
    criteria_results: dict[str, bool] = field(default_factory=dict)
    comments: str = ""
    reviewed_at: str = ""
    auto_review: bool = False

    @property
    def all_passed(self) -> bool:
        return all(self.criteria_results.values()) if self.criteria_results else False

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_id": self.review_id,
            "skill_id": self.skill_id,
            "status": self.status.value,
            "reviewer": self.reviewer,
            "all_passed": self.all_passed,
            "criteria": self.criteria_results,
            "auto_review": self.auto_review,
        }


class SkillCurator:
    """Manuelle und automatische Skill-Kuration."""

    REQUIRED_CRITERIA = [
        ReviewCriteria.SECURITY_SCAN,
        ReviewCriteria.PRIVACY_CHECK,
        ReviewCriteria.DEPENDENCY_AUDIT,
    ]

    def __init__(self, require_manual_for_new: bool = True) -> None:
        self._require_manual = require_manual_for_new
        self._reviews: dict[str, CurationReview] = {}
        self._counter = 0

    def submit_for_review(self, skill_id: str) -> CurationReview:
        self._counter += 1
        review = CurationReview(
            review_id=f"CUR-{self._counter:04d}",
            skill_id=skill_id,
            status=CurationStatus.PENDING,
        )
        self._reviews[skill_id] = review
        return review

    def auto_review(self, skill_id: str, criteria_results: dict[str, bool]) -> CurationReview:
        """Automatische Prüfung aller Kriterien."""
        self._counter += 1
        all_passed = all(criteria_results.values())
        status = CurationStatus.APPROVED if all_passed and not self._require_manual else CurationStatus.UNDER_REVIEW
        if not all_passed:
            status = CurationStatus.REJECTED

        review = CurationReview(
            review_id=f"CUR-{self._counter:04d}",
            skill_id=skill_id,
            status=status,
            criteria_results=criteria_results,
            reviewed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            auto_review=True,
        )
        self._reviews[skill_id] = review
        return review

    def manual_approve(self, skill_id: str, reviewer: str, comments: str = "") -> CurationReview | None:
        review = self._reviews.get(skill_id)
        if review:
            review.status = CurationStatus.APPROVED
            review.reviewer = reviewer
            review.comments = comments
            review.reviewed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return review

    def manual_reject(self, skill_id: str, reviewer: str, reason: str = "") -> CurationReview | None:
        review = self._reviews.get(skill_id)
        if review:
            review.status = CurationStatus.REJECTED
            review.reviewer = reviewer
            review.comments = reason
            review.reviewed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return review

    def suspend(self, skill_id: str) -> bool:
        review = self._reviews.get(skill_id)
        if review:
            review.status = CurationStatus.SUSPENDED
            return True
        return False

    def get_status(self, skill_id: str) -> CurationStatus | None:
        review = self._reviews.get(skill_id)
        return review.status if review else None

    def is_approved(self, skill_id: str) -> bool:
        review = self._reviews.get(skill_id)
        return review.status == CurationStatus.APPROVED if review else False

    def pending_reviews(self) -> list[CurationReview]:
        return [r for r in self._reviews.values() if r.status in (CurationStatus.PENDING, CurationStatus.UNDER_REVIEW)]

    @property
    def review_count(self) -> int:
        return len(self._reviews)

    def stats(self) -> dict[str, Any]:
        reviews = list(self._reviews.values())
        return {
            "total_reviews": len(reviews),
            "pending": sum(1 for r in reviews if r.status in (CurationStatus.PENDING, CurationStatus.UNDER_REVIEW)),
            "approved": sum(1 for r in reviews if r.status == CurationStatus.APPROVED),
            "rejected": sum(1 for r in reviews if r.status == CurationStatus.REJECTED),
            "suspended": sum(1 for r in reviews if r.status == CurationStatus.SUSPENDED),
            "require_manual": self._require_manual,
        }


# ============================================================================
# Emergency Updater
# ============================================================================


class UpdateSeverity(Enum):
    ROUTINE = "routine"
    IMPORTANT = "important"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class EmergencyPatch:
    """Notfall-Patch für das Ecosystem."""

    patch_id: str
    severity: UpdateSeverity
    affected_skills: list[str]
    description: str
    action: str  # "block", "update", "rollback", "notify"
    issued_at: str = ""
    applied_at: str = ""
    applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "patch_id": self.patch_id,
            "severity": self.severity.value,
            "affected_skills": self.affected_skills,
            "description": self.description,
            "action": self.action,
            "applied": self.applied,
        }


class EmergencyUpdater:
    """Zentralisierte Notfall-Patches für das Skill-Ecosystem."""

    def __init__(self) -> None:
        self._patches: list[EmergencyPatch] = []
        self._counter = 0

    def issue_patch(
        self,
        severity: UpdateSeverity,
        affected_skills: list[str],
        description: str,
        action: str = "block",
    ) -> EmergencyPatch:
        self._counter += 1
        patch = EmergencyPatch(
            patch_id=f"EP-{self._counter:04d}",
            severity=severity,
            affected_skills=affected_skills,
            description=description,
            action=action,
            issued_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self._patches.append(patch)
        return patch

    def apply_patch(self, patch_id: str) -> bool:
        for p in self._patches:
            if p.patch_id == patch_id:
                p.applied = True
                p.applied_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                return True
        return False

    def pending_patches(self) -> list[EmergencyPatch]:
        return [p for p in self._patches if not p.applied]

    def is_blocked(self, skill_id: str) -> bool:
        return any(
            skill_id in p.affected_skills and p.action == "block" and p.applied
            for p in self._patches
        )

    @property
    def patch_count(self) -> int:
        return len(self._patches)

    def stats(self) -> dict[str, Any]:
        return {
            "total_patches": len(self._patches),
            "pending": sum(1 for p in self._patches if not p.applied),
            "applied": sum(1 for p in self._patches if p.applied),
            "by_severity": {
                sev.value: sum(1 for p in self._patches if p.severity == sev)
                for sev in UpdateSeverity
                if any(p.severity == sev for p in self._patches)
            },
        }


# ============================================================================
# Fraud Detector
# ============================================================================


@dataclass
class FraudSignal:
    """Ein Betrugs-Signal für einen Skill."""

    signal_id: str
    skill_id: str
    signal_type: str  # "name_squatting", "crypto_mining", "data_theft", "malware", "reputation_gaming"
    confidence: float
    evidence: str = ""
    detected_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "skill_id": self.skill_id,
            "type": self.signal_type,
            "confidence": round(self.confidence, 2),
            "evidence": self.evidence[:100],
        }


class FraudDetector:
    """Erkennt betrügerische oder schädliche Skills.

    Prüft auf:
      - Name-Squatting (ähnliche Namen wie populäre Skills)
      - Crypto-Mining-Patterns
      - Data-Theft-Indikatoren
      - Reputation-Gaming
    """

    CRYPTO_PATTERNS = [
        r"crypto\s*min", r"bitcoin\s*min", r"ethereum\s*min",
        r"monero", r"coinhive", r"wasm.*min",
    ]

    DATA_THEFT_PATTERNS = [
        r"exfiltrat", r"phone\s*home", r"send.*secret",
        r"upload.*credential", r"steal.*data",
    ]

    MALWARE_PATTERNS = [
        r"eval\s*\(", r"exec\s*\(", r"__import__\s*\(",
        r"subprocess\s*\.", r"os\.system", r"shutil\.rmtree",
    ]

    def __init__(self) -> None:
        self._signals: list[FraudSignal] = []
        self._counter = 0
        self._known_popular: set[str] = {
            "code-formatter", "text-analyzer", "web-scraper",
            "data-processor", "file-converter", "task-manager",
        }

    def scan(self, skill_id: str, code: str, metadata: dict[str, Any] | None = None) -> list[FraudSignal]:
        findings: list[FraudSignal] = []
        meta = metadata or {}

        # Name-Squatting
        for popular in self._known_popular:
            if skill_id != popular and self._similar_name(skill_id, popular):
                self._counter += 1
                findings.append(FraudSignal(
                    signal_id=f"FS-{self._counter:04d}",
                    skill_id=skill_id,
                    signal_type="name_squatting",
                    confidence=0.7,
                    evidence=f"Ähnlich zu '{popular}'",
                    detected_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                ))

        # Pattern-Scans
        for pattern_set, signal_type in [
            (self.CRYPTO_PATTERNS, "crypto_mining"),
            (self.DATA_THEFT_PATTERNS, "data_theft"),
            (self.MALWARE_PATTERNS, "malware"),
        ]:
            for pattern in pattern_set:
                if re.search(pattern, code, re.IGNORECASE):
                    self._counter += 1
                    findings.append(FraudSignal(
                        signal_id=f"FS-{self._counter:04d}",
                        skill_id=skill_id,
                        signal_type=signal_type,
                        confidence=0.8,
                        evidence=f"Pattern: {pattern}",
                        detected_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    ))
                    break  # Eine pro Kategorie reicht

        # Reputation-Gaming
        reviews = meta.get("reviews_count", 0)
        stars = meta.get("avg_stars", 0)
        age_days = meta.get("age_days", 365)
        if reviews > 100 and stars > 4.9 and age_days < 7:
            self._counter += 1
            findings.append(FraudSignal(
                signal_id=f"FS-{self._counter:04d}",
                skill_id=skill_id,
                signal_type="reputation_gaming",
                confidence=0.9,
                evidence=f"{reviews} Reviews mit {stars}★ in {age_days} Tagen",
                detected_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            ))

        self._signals.extend(findings)
        return findings

    @staticmethod
    def _similar_name(a: str, b: str) -> bool:
        """Erkennt Name-Squatting durch Typo-Varianten (echte Levenshtein-Distanz)."""
        a_clean = a.replace("-", "").replace("_", "").lower()
        b_clean = b.replace("-", "").replace("_", "").lower()
        if a_clean == b_clean:
            return True
        # Echte Levenshtein-Distanz (DP)
        if abs(len(a_clean) - len(b_clean)) > 2:
            return False
        m, n = len(a_clean), len(b_clean)
        if m < n:
            a_clean, b_clean = b_clean, a_clean
            m, n = n, m
        row = list(range(n + 1))
        for i in range(1, m + 1):
            prev, row[0] = row[0], i
            for j in range(1, n + 1):
                old = row[j]
                row[j] = min(
                    row[j] + 1,
                    row[j - 1] + 1,
                    prev + (0 if a_clean[i - 1] == b_clean[j - 1] else 1),
                )
                prev = old
        return row[n] <= 2

    @property
    def signal_count(self) -> int:
        return len(self._signals)

    def stats(self) -> dict[str, Any]:
        signals = self._signals
        return {
            "total_signals": len(signals),
            "by_type": {
                t: sum(1 for s in signals if s.signal_type == t)
                for t in ("name_squatting", "crypto_mining", "data_theft", "malware", "reputation_gaming")
                if any(s.signal_type == t for s in signals)
            },
        }


# ============================================================================
# Security Trainer
# ============================================================================


@dataclass
class TrainingModule:
    """Ein Trainingsmodul für Security-Teams."""

    module_id: str
    title: str
    category: str  # "prompt_injection", "tool_abuse", "privilege_escalation", "general"
    difficulty: str  # "beginner", "intermediate", "advanced"
    duration_minutes: int
    content_topics: list[str] = field(default_factory=list)
    quiz_questions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_id": self.module_id,
            "title": self.title,
            "category": self.category,
            "difficulty": self.difficulty,
            "duration_min": self.duration_minutes,
            "topics": self.content_topics,
        }


@dataclass
class TrainingProgress:
    """Fortschritt eines Team-Mitglieds."""

    user_id: str
    completed_modules: list[str] = field(default_factory=list)
    quiz_scores: dict[str, float] = field(default_factory=dict)

    @property
    def avg_score(self) -> float:
        if not self.quiz_scores:
            return 0.0
        return sum(self.quiz_scores.values()) / len(self.quiz_scores)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "completed": len(self.completed_modules),
            "avg_score": round(self.avg_score, 1),
        }


class SecurityTrainer:
    """Trainingsmaterialien für Agent-spezifische Security-Risiken."""

    DEFAULT_MODULES = [
        TrainingModule("SM-001", "Prompt-Injection Grundlagen", "prompt_injection", "beginner", 30,
                       ["Was ist Prompt-Injection?", "Beispiele", "Erkennung", "Abwehr"], 10),
        TrainingModule("SM-002", "Tool-Missbrauch erkennen", "tool_abuse", "intermediate", 45,
                       ["Tool-Chaining-Angriffe", "Privilege-Escalation via Tools", "Monitoring"], 15),
        TrainingModule("SM-003", "Memory-Poisoning", "memory_poisoning", "advanced", 60,
                       ["Angriffsvektoren", "Integritäts-Checks", "Versionskontrolle", "Incident-Response"], 20),
        TrainingModule("SM-004", "Agent-zu-Agent Sicherheit", "cross_agent", "advanced", 45,
                       ["Vertrauensgrenzen", "Message-Validierung", "Federation-Security"], 15),
        TrainingModule("SM-005", "EU AI Act für Betreiber", "compliance", "beginner", 30,
                       ["Risikoklassen", "Transparenzpflichten", "Dokumentation", "Audit-Trails"], 10),
    ]

    def __init__(self) -> None:
        self._modules = {m.module_id: m for m in self.DEFAULT_MODULES}
        self._progress: dict[str, TrainingProgress] = {}

    def get_module(self, module_id: str) -> TrainingModule | None:
        return self._modules.get(module_id)

    def all_modules(self) -> list[TrainingModule]:
        return list(self._modules.values())

    def complete_module(self, user_id: str, module_id: str, quiz_score: float = 0.0) -> bool:
        if module_id not in self._modules:
            return False
        if user_id not in self._progress:
            self._progress[user_id] = TrainingProgress(user_id=user_id)
        prog = self._progress[user_id]
        if module_id not in prog.completed_modules:
            prog.completed_modules.append(module_id)
        prog.quiz_scores[module_id] = quiz_score
        return True

    def get_progress(self, user_id: str) -> TrainingProgress | None:
        return self._progress.get(user_id)

    def team_completion_rate(self) -> float:
        if not self._progress:
            return 0.0
        total_modules = len(self._modules)
        rates = [len(p.completed_modules) / total_modules * 100 for p in self._progress.values()]
        return sum(rates) / len(rates)

    @property
    def module_count(self) -> int:
        return len(self._modules)

    def stats(self) -> dict[str, Any]:
        return {
            "total_modules": len(self._modules),
            "total_users": len(self._progress),
            "team_completion_rate": round(self.team_completion_rate(), 1),
            "modules": [m.to_dict() for m in self._modules.values()],
        }


# ============================================================================
# Trust Boundary Manager
# ============================================================================


class TrustLevel(Enum):
    UNTRUSTED = "untrusted"
    RESTRICTED = "restricted"
    STANDARD = "standard"
    ELEVATED = "elevated"
    TRUSTED = "trusted"


@dataclass
class TrustBoundary:
    """Vertrauensgrenze zwischen zwei Agenten."""

    boundary_id: str
    local_agent: str
    remote_agent: str
    trust_level: TrustLevel = TrustLevel.RESTRICTED
    budget_limit_eur: float = 10.0
    allowed_operations: set[str] = field(default_factory=lambda: {"read"})
    max_requests_per_minute: int = 10
    requires_approval: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary_id": self.boundary_id,
            "local": self.local_agent,
            "remote": self.remote_agent,
            "trust_level": self.trust_level.value,
            "budget_limit": self.budget_limit_eur,
            "allowed_ops": sorted(self.allowed_operations),
            "requires_approval": self.requires_approval,
        }


class TrustBoundaryManager:
    """Definiert und erzwingt Vertrauensgrenzen zwischen Agenten."""

    def __init__(self) -> None:
        self._boundaries: dict[str, TrustBoundary] = {}

    def set_boundary(
        self,
        local_agent: str,
        remote_agent: str,
        trust_level: TrustLevel,
        *,
        budget_limit: float = 10.0,
        allowed_ops: set[str] | None = None,
        requires_approval: bool = True,
    ) -> TrustBoundary:
        boundary_id = f"{local_agent}↔{remote_agent}"
        boundary = TrustBoundary(
            boundary_id=boundary_id,
            local_agent=local_agent,
            remote_agent=remote_agent,
            trust_level=trust_level,
            budget_limit_eur=budget_limit,
            allowed_operations=allowed_ops or {"read"},
            requires_approval=requires_approval,
        )
        self._boundaries[boundary_id] = boundary
        return boundary

    def check_operation(self, local_agent: str, remote_agent: str, operation: str) -> dict[str, Any]:
        boundary_id = f"{local_agent}↔{remote_agent}"
        boundary = self._boundaries.get(boundary_id)

        if not boundary:
            return {"allowed": False, "reason": "Keine Vertrauensgrenze definiert"}

        if boundary.trust_level == TrustLevel.UNTRUSTED:
            return {"allowed": False, "reason": "Agent ist untrusted"}

        if operation not in boundary.allowed_operations and "*" not in boundary.allowed_operations:
            return {"allowed": False, "reason": f"Operation '{operation}' nicht erlaubt"}

        return {
            "allowed": True,
            "requires_approval": boundary.requires_approval,
            "trust_level": boundary.trust_level.value,
        }

    def get_boundary(self, local_agent: str, remote_agent: str) -> TrustBoundary | None:
        return self._boundaries.get(f"{local_agent}↔{remote_agent}")

    @property
    def boundary_count(self) -> int:
        return len(self._boundaries)

    def stats(self) -> dict[str, Any]:
        boundaries = list(self._boundaries.values())
        return {
            "total_boundaries": len(boundaries),
            "by_trust_level": {
                level.value: sum(1 for b in boundaries if b.trust_level == level)
                for level in TrustLevel
                if any(b.trust_level == level for b in boundaries)
            },
        }


# ============================================================================
# Ecosystem Controller (Hauptklasse)
# ============================================================================


class EcosystemController:
    """Orchestriert alle Ecosystem-Kontrollen."""

    def __init__(self) -> None:
        self._curator = SkillCurator()
        self._updater = EmergencyUpdater()
        self._fraud = FraudDetector()
        self._trainer = SecurityTrainer()
        self._trust = TrustBoundaryManager()

    @property
    def curator(self) -> SkillCurator:
        return self._curator

    @property
    def updater(self) -> EmergencyUpdater:
        return self._updater

    @property
    def fraud(self) -> FraudDetector:
        return self._fraud

    @property
    def trainer(self) -> SecurityTrainer:
        return self._trainer

    @property
    def trust(self) -> TrustBoundaryManager:
        return self._trust

    def full_skill_review(self, skill_id: str, code: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Vollständige Prüfung eines Skills: Kuration + Fraud-Scan."""
        fraud_signals = self._fraud.scan(skill_id, code, metadata)
        has_fraud = len(fraud_signals) > 0

        criteria = {
            "security_scan": not any(s.signal_type == "malware" for s in fraud_signals),
            "privacy_check": not any(s.signal_type == "data_theft" for s in fraud_signals),
            "dependency_audit": True,
            "no_fraud": not has_fraud,
        }
        review = self._curator.auto_review(skill_id, criteria)

        return {
            "skill_id": skill_id,
            "curation": review.to_dict(),
            "fraud_signals": [s.to_dict() for s in fraud_signals],
            "approved": review.status == CurationStatus.APPROVED,
        }

    def stats(self) -> dict[str, Any]:
        return {
            "curator": self._curator.stats(),
            "updater": self._updater.stats(),
            "fraud": self._fraud.stats(),
            "trainer": self._trainer.stats(),
            "trust": self._trust.stats(),
        }
