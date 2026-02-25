"""Jarvis · Memory-Integrität & erweiterte Explainability.

Prüft die Integrität des Knowledge-Speichers:

  - MemoryEntry:           Einzelner Eintrag mit Hash und Version
  - IntegrityChecker:      Erkennt Manipulationen durch Hash-Verifikation
  - DuplicateDetector:     Findet Dubletten in Erinnerungen
  - ContradictionDetector: Erkennt widersprüchliche Fakten
  - MemoryVersionControl:  Versionskontrolle für Memory-Änderungen
  - PlausibilityChecker:   Plausibilitäts-Checks auf neue Einträge
  - DecisionExplainer:     Erweiterte Explainability mit Quellen-Tracking

Architektur-Bibel: §7.3 (Memory), §13.3 (Explainability)
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ============================================================================
# Memory Entry
# ============================================================================


@dataclass
class MemoryEntry:
    """Einzelner Memory-Eintrag mit Integritäts-Hash."""

    entry_id: str
    content: str
    source: str = ""          # Woher stammt das Wissen
    agent_id: str = ""
    category: str = ""        # "fact", "preference", "context", "skill"
    confidence: float = 1.0   # 0-1
    created_at: str = ""
    updated_at: str = ""
    version: int = 1
    content_hash: str = ""
    tags: list[str] = field(default_factory=list)

    def compute_hash(self) -> str:
        data = f"{self.entry_id}:{self.content}:{self.version}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def verify_integrity(self) -> bool:
        return self.content_hash == self.compute_hash()

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "content": self.content[:100],
            "source": self.source,
            "confidence": self.confidence,
            "version": self.version,
            "integrity_ok": self.verify_integrity(),
            "tags": self.tags,
        }


# ============================================================================
# Integrity Checker
# ============================================================================


class IntegrityStatus(Enum):
    INTACT = "intact"
    TAMPERED = "tampered"
    MISSING_HASH = "missing_hash"
    UNKNOWN = "unknown"


@dataclass
class IntegrityReport:
    """Ergebnis einer Integritäts-Prüfung."""

    total_entries: int
    intact: int
    tampered: int
    missing_hash: int
    tampered_ids: list[str] = field(default_factory=list)
    timestamp: str = ""

    @property
    def integrity_score(self) -> float:
        if self.total_entries == 0:
            return 100.0
        return self.intact / self.total_entries * 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_entries": self.total_entries,
            "intact": self.intact,
            "tampered": self.tampered,
            "missing_hash": self.missing_hash,
            "integrity_score": round(self.integrity_score, 1),
            "tampered_ids": self.tampered_ids,
        }


class IntegrityChecker:
    """Erkennt Manipulationen durch Hash-Verifikation."""

    def __init__(self) -> None:
        self._history: list[IntegrityReport] = []

    def check(self, entries: list[MemoryEntry]) -> IntegrityReport:
        intact = 0
        tampered = 0
        missing = 0
        tampered_ids: list[str] = []

        for entry in entries:
            if not entry.content_hash:
                missing += 1
            elif entry.verify_integrity():
                intact += 1
            else:
                tampered += 1
                tampered_ids.append(entry.entry_id)

        report = IntegrityReport(
            total_entries=len(entries),
            intact=intact,
            tampered=tampered,
            missing_hash=missing,
            tampered_ids=tampered_ids,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self._history.append(report)
        return report

    def last_report(self) -> IntegrityReport | None:
        return self._history[-1] if self._history else None

    def stats(self) -> dict[str, Any]:
        return {
            "total_checks": len(self._history),
            "last_score": round(self._history[-1].integrity_score, 1) if self._history else 100.0,
        }


# ============================================================================
# Duplicate Detector
# ============================================================================


@dataclass
class DuplicateGroup:
    """Gruppe von Dubletten."""

    group_id: str
    entries: list[str]  # entry_ids
    similarity: float
    recommended_action: str = "merge"

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "entries": self.entries,
            "similarity": round(self.similarity, 2),
            "action": self.recommended_action,
        }


class DuplicateDetector:
    """Findet Dubletten in Erinnerungen."""

    def __init__(self, similarity_threshold: float = 0.85) -> None:
        self._threshold = similarity_threshold

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower().strip())

    @staticmethod
    def _simple_similarity(a: str, b: str) -> float:
        """Jaccard-Ähnlichkeit auf Wort-Ebene."""
        words_a = set(a.split())
        words_b = set(b.split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    def detect(self, entries: list[MemoryEntry]) -> list[DuplicateGroup]:
        groups: list[DuplicateGroup] = []
        seen: set[str] = set()
        group_counter = 0

        for i, entry_a in enumerate(entries):
            if entry_a.entry_id in seen:
                continue
            norm_a = self._normalize(entry_a.content)
            duplicates = [entry_a.entry_id]

            for j in range(i + 1, len(entries)):
                entry_b = entries[j]
                if entry_b.entry_id in seen:
                    continue
                norm_b = self._normalize(entry_b.content)
                sim = self._simple_similarity(norm_a, norm_b)
                if sim >= self._threshold:
                    duplicates.append(entry_b.entry_id)
                    seen.add(entry_b.entry_id)

            if len(duplicates) > 1:
                group_counter += 1
                seen.add(entry_a.entry_id)
                groups.append(DuplicateGroup(
                    group_id=f"DUP-{group_counter:04d}",
                    entries=duplicates,
                    similarity=self._threshold,
                ))

        return groups

    def stats(self, groups: list[DuplicateGroup]) -> dict[str, Any]:
        return {
            "duplicate_groups": len(groups),
            "total_duplicates": sum(len(g.entries) for g in groups),
        }


# ============================================================================
# Contradiction Detector
# ============================================================================


@dataclass
class Contradiction:
    """Ein erkannter Widerspruch zwischen zwei Einträgen."""

    contradiction_id: str
    entry_a_id: str
    entry_b_id: str
    entry_a_content: str
    entry_b_content: str
    reason: str
    confidence: float = 0.8

    def to_dict(self) -> dict[str, Any]:
        return {
            "contradiction_id": self.contradiction_id,
            "entry_a": self.entry_a_id,
            "entry_b": self.entry_b_id,
            "reason": self.reason,
            "confidence": self.confidence,
        }


class ContradictionDetector:
    """Erkennt widersprüchliche Fakten im Memory.

    Einfache Heuristiken (Produktionsversion → NLI-Modell):
      - Negations-Erkennung bei gleichem Subjekt
      - Numerische Widersprüche (x ist 5 vs x ist 10)
      - Gegenteil-Paare
    """

    OPPOSITES = [
        ("aktiviert", "deaktiviert"), ("erlaubt", "verboten"),
        ("wahr", "falsch"), ("ja", "nein"), ("erfolgreich", "fehlgeschlagen"),
        ("sicher", "unsicher"), ("online", "offline"),
    ]

    def __init__(self) -> None:
        self._counter = 0

    def detect(self, entries: list[MemoryEntry]) -> list[Contradiction]:
        contradictions: list[Contradiction] = []

        for i, a in enumerate(entries):
            for j in range(i + 1, len(entries)):
                b = entries[j]
                reason = self._check_contradiction(a.content, b.content)
                if reason:
                    self._counter += 1
                    contradictions.append(Contradiction(
                        contradiction_id=f"CONTR-{self._counter:04d}",
                        entry_a_id=a.entry_id,
                        entry_b_id=b.entry_id,
                        entry_a_content=a.content[:100],
                        entry_b_content=b.content[:100],
                        reason=reason,
                    ))

        return contradictions

    def _check_contradiction(self, text_a: str, text_b: str) -> str:
        a_lower = text_a.lower()
        b_lower = text_b.lower()

        # Check opposites
        for pos, neg in self.OPPOSITES:
            if (pos in a_lower and neg in b_lower) or (neg in a_lower and pos in b_lower):
                return f"Gegenteil-Paar: {pos}/{neg}"

        # Negation + same subject
        if ("nicht" in a_lower and "nicht" not in b_lower) or ("nicht" not in a_lower and "nicht" in b_lower):
            words_a = set(re.findall(r"\w+", a_lower)) - {"nicht", "kein", "keine"}
            words_b = set(re.findall(r"\w+", b_lower)) - {"nicht", "kein", "keine"}
            overlap = words_a & words_b
            if len(overlap) >= 3:
                return f"Negations-Widerspruch (Überlappung: {len(overlap)} Wörter)"

        # Numeric contradiction
        nums_a = re.findall(r"\b(\d+(?:\.\d+)?)\b", a_lower)
        nums_b = re.findall(r"\b(\d+(?:\.\d+)?)\b", b_lower)
        if nums_a and nums_b:
            words_a = set(re.findall(r"[a-zäöüß]+", a_lower))
            words_b = set(re.findall(r"[a-zäöüß]+", b_lower))
            if len(words_a & words_b) >= 2 and set(nums_a) != set(nums_b):
                return f"Numerischer Widerspruch: {nums_a} vs {nums_b}"

        return ""


# ============================================================================
# Memory Version Control
# ============================================================================


@dataclass
class MemoryVersion:
    """Eine Version eines Memory-Eintrags."""

    entry_id: str
    version: int
    content: str
    changed_by: str = ""
    changed_at: str = ""
    change_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "version": self.version,
            "changed_by": self.changed_by,
            "changed_at": self.changed_at,
            "reason": self.change_reason,
        }


class MemoryVersionControl:
    """Versionskontrolle für Memory-Änderungen."""

    def __init__(self) -> None:
        self._versions: dict[str, list[MemoryVersion]] = {}  # entry_id → versions

    def record(self, entry: MemoryEntry, changed_by: str = "", reason: str = "") -> MemoryVersion:
        if entry.entry_id not in self._versions:
            self._versions[entry.entry_id] = []

        version = MemoryVersion(
            entry_id=entry.entry_id,
            version=entry.version,
            content=entry.content,
            changed_by=changed_by,
            changed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            change_reason=reason,
        )
        self._versions[entry.entry_id].append(version)
        return version

    def get_history(self, entry_id: str) -> list[MemoryVersion]:
        return self._versions.get(entry_id, [])

    def get_version(self, entry_id: str, version: int) -> MemoryVersion | None:
        for v in self._versions.get(entry_id, []):
            if v.version == version:
                return v
        return None

    def rollback(self, entry_id: str, to_version: int) -> MemoryVersion | None:
        """Rollt einen Eintrag auf eine frühere Version zurück."""
        return self.get_version(entry_id, to_version)

    @property
    def tracked_entries(self) -> int:
        return len(self._versions)

    @property
    def total_versions(self) -> int:
        return sum(len(v) for v in self._versions.values())

    def stats(self) -> dict[str, Any]:
        return {
            "tracked_entries": self.tracked_entries,
            "total_versions": self.total_versions,
            "avg_versions": round(self.total_versions / max(1, self.tracked_entries), 1),
        }


# ============================================================================
# Plausibility Checker
# ============================================================================


class PlausibilityResult(Enum):
    PLAUSIBLE = "plausible"
    SUSPICIOUS = "suspicious"
    IMPLAUSIBLE = "implausible"


@dataclass
class PlausibilityCheck:
    """Ergebnis einer Plausibilitäts-Prüfung."""

    entry_id: str
    result: PlausibilityResult
    reasons: list[str] = field(default_factory=list)
    score: float = 100.0  # 0-100

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "result": self.result.value,
            "score": round(self.score, 1),
            "reasons": self.reasons,
        }


class PlausibilityChecker:
    """Plausibilitäts-Checks auf neue Memory-Einträge.

    Prüft:
      - Länge (zu kurz/lang?)
      - Sprache (enthält Injection-Patterns?)
      - Confidence (zu niedrig?)
      - Content-Type-Konsistenz
    """

    INJECTION_PATTERNS = [
        r"ignore\s+(previous|all|above)",
        r"system\s*prompt",
        r"you\s+are\s+now",
        r"<\s*script",
        r"\beval\b.*\(",
    ]

    def check(self, entry: MemoryEntry) -> PlausibilityCheck:
        reasons: list[str] = []
        score = 100.0

        # Länge
        if len(entry.content) < 3:
            reasons.append("Inhalt zu kurz")
            score -= 30
        if len(entry.content) > 10000:
            reasons.append("Inhalt ungewöhnlich lang")
            score -= 20

        # Injection-Patterns
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, entry.content, re.IGNORECASE):
                reasons.append(f"Injection-Pattern erkannt: {pattern}")
                score -= 40

        # Confidence
        if entry.confidence < 0.3:
            reasons.append(f"Sehr niedrige Confidence: {entry.confidence}")
            score -= 20

        # Source
        if not entry.source:
            reasons.append("Keine Quelle angegeben")
            score -= 10

        score = max(0, score)
        result = (
            PlausibilityResult.PLAUSIBLE if score >= 70 else
            PlausibilityResult.SUSPICIOUS if score >= 40 else
            PlausibilityResult.IMPLAUSIBLE
        )

        return PlausibilityCheck(
            entry_id=entry.entry_id,
            result=result,
            reasons=reasons,
            score=score,
        )


# ============================================================================
# Decision Explainer
# ============================================================================


@dataclass
class DecisionExplanation:
    """Erklärung einer Agent-Entscheidung mit Quellen."""

    decision_id: str
    question: str
    answer: str
    sources: list[dict[str, str]] = field(default_factory=list)
    reasoning_steps: list[str] = field(default_factory=list)
    confidence: float = 0.0
    alternative_answers: list[str] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "question": self.question[:100],
            "answer": self.answer[:200],
            "sources": self.sources,
            "steps": len(self.reasoning_steps),
            "confidence": self.confidence,
            "alternatives": len(self.alternative_answers),
        }


class DecisionExplainer:
    """Erweiterte Explainability mit Quellen-Tracking."""

    def __init__(self) -> None:
        self._explanations: list[DecisionExplanation] = []
        self._counter = 0

    def explain(
        self,
        question: str,
        answer: str,
        *,
        sources: list[dict[str, str]] | None = None,
        reasoning_steps: list[str] | None = None,
        confidence: float = 0.8,
        alternatives: list[str] | None = None,
    ) -> DecisionExplanation:
        self._counter += 1
        explanation = DecisionExplanation(
            decision_id=f"DEC-{self._counter:05d}",
            question=question,
            answer=answer,
            sources=sources or [],
            reasoning_steps=reasoning_steps or [],
            confidence=confidence,
            alternative_answers=alternatives or [],
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self._explanations.append(explanation)
        return explanation

    def recent(self, limit: int = 10) -> list[DecisionExplanation]:
        return list(reversed(self._explanations[-limit:]))

    @property
    def explanation_count(self) -> int:
        return len(self._explanations)

    def avg_confidence(self) -> float:
        if not self._explanations:
            return 0.0
        return sum(e.confidence for e in self._explanations) / len(self._explanations)

    def stats(self) -> dict[str, Any]:
        return {
            "total_explanations": len(self._explanations),
            "avg_confidence": round(self.avg_confidence(), 2),
            "with_sources": sum(1 for e in self._explanations if e.sources),
            "with_alternatives": sum(1 for e in self._explanations if e.alternative_answers),
        }
