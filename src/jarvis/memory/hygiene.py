"""Jarvis · Memory-Hygiene-Framework.

Schutz des RAG-Gedächtnisses vor Manipulation:

  - InjectionScanner:      Erkennt Prompt-Injections in Memory-Einträgen
  - ContradictionChecker:  Findet widersprüchliche Fakten
  - CredentialLeakDetector: Erkennt versehentlich gespeicherte Secrets
  - IntegrityVerifier:     Prüft Hashes und Zeitstempel-Konsistenz
  - MemoryHygieneEngine:   Orchestriert alle Checks

Architektur-Bibel: §7 (Memory-Schichten), §14.7 (Memory-Integrität)

Memory-Poisoning ist ein realer Angriffsvektor bei RAG-Systemen:
  - Bösartige Einträge können Agent-Verhalten manipulieren
  - Widersprüchliche Fakten degradieren die Antwortqualität
  - Credentials in Episoden-Logs sind ein Datenleck-Risiko
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ============================================================================
# Enums & Data Classes
# ============================================================================


class ThreatType(Enum):
    """Art der erkannten Bedrohung."""

    INJECTION = "injection"
    CONTRADICTION = "contradiction"
    CREDENTIAL_LEAK = "credential_leak"
    INTEGRITY_VIOLATION = "integrity_violation"
    SUSPICIOUS_SOURCE = "suspicious_source"
    STALE_DATA = "stale_data"


class ThreatSeverity(Enum):
    """Schweregrad der Bedrohung."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MemoryThreat:
    """Eine erkannte Bedrohung in einem Memory-Eintrag."""

    threat_id: str
    threat_type: ThreatType
    severity: ThreatSeverity
    description: str
    entry_content: str
    matched_pattern: str = ""
    recommended_action: str = "quarantine"
    detected_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "threat_id": self.threat_id,
            "threat_type": self.threat_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "entry_preview": self.entry_content[:100],
            "matched_pattern": self.matched_pattern,
            "recommended_action": self.recommended_action,
            "detected_at": self.detected_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


@dataclass
class HygieneReport:
    """Ergebnis eines Memory-Hygiene-Scans."""

    report_id: str
    scanned_entries: int = 0
    clean_entries: int = 0
    threats_found: int = 0
    threats: list[MemoryThreat] = field(default_factory=list)
    quarantined: int = 0
    scan_duration_ms: int = 0
    timestamp: str = ""

    @property
    def threat_rate(self) -> float:
        if self.scanned_entries == 0:
            return 0.0
        return round((self.threats_found / self.scanned_entries) * 100, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "scanned_entries": self.scanned_entries,
            "clean_entries": self.clean_entries,
            "threats_found": self.threats_found,
            "quarantined": self.quarantined,
            "threat_rate": self.threat_rate,
            "scan_duration_ms": self.scan_duration_ms,
            "timestamp": self.timestamp,
            "threats": [t.to_dict() for t in self.threats],
        }


# ============================================================================
# Injection-Scanner
# ============================================================================

# Patterns die auf Prompt-Injection in Memory-Einträgen hindeuten
_INJECTION_PATTERNS: list[tuple[str, ThreatSeverity, str]] = [
    # System-Override-Versuche
    (r"(?i)ignore\s+(all\s+)?previous\s+instructions", ThreatSeverity.CRITICAL, "System-Override"),
    (r"(?i)disregard\s+(all\s+)?(safety|security|rules)", ThreatSeverity.CRITICAL, "Safety-Override"),
    (r"(?i)\bsystem\s*(override|prompt|directive)\b", ThreatSeverity.HIGH, "System-Directive"),
    (r"(?i)you\s+are\s+now\s+(a|an)\s+unrestricted", ThreatSeverity.CRITICAL, "Persona-Switch"),

    # Tag-basierte Injections
    (r"<\s*(system|hidden|secret|admin)\s*>", ThreatSeverity.HIGH, "Hidden-Tag"),
    (r"\[SYSTEM\]", ThreatSeverity.HIGH, "System-Bracket"),
    (r"</s>|</?user>|</?assistant>", ThreatSeverity.HIGH, "Delimiter-Injection"),

    # Command-Injections
    (r"(?i)execute\s+(shell|bash|command|cmd)", ThreatSeverity.CRITICAL, "Shell-Injection"),
    (r"(?i)(rm\s+-rf|curl\s+.+/exfil|wget\s+)", ThreatSeverity.CRITICAL, "Destructive-Command"),
    (r"(?i)(eval|exec)\s*\(", ThreatSeverity.HIGH, "Code-Execution"),

    # Manipulation-Markers
    (r"(?i)always\s+(include|output|send|respond\s+with)", ThreatSeverity.MEDIUM, "Behavior-Override"),
    (r"(?i)never\s+(mention|reveal|block|filter)", ThreatSeverity.MEDIUM, "Suppression-Attempt"),
    (r"(?i)pretend\s+(you|to\s+be)\s+", ThreatSeverity.HIGH, "Impersonation"),

    # Deutsche Injection-Patterns (Jarvis ist primär deutsch)
    (r"(?i)ignoriere?\s+(alle\s+)?(vorherigen?\s+)?anweisungen", ThreatSeverity.CRITICAL, "DE-System-Override"),
    (r"(?i)vergiss\s+(alles|alle\s+regeln)", ThreatSeverity.CRITICAL, "DE-Memory-Override"),
    (r"(?i)du\s+bist\s+(jetzt|ab\s+sofort)\s+(ein|eine)", ThreatSeverity.HIGH, "DE-Persona-Switch"),
    (r"(?i)neue\s+(system|sicherheits)?\s*anweisungen?", ThreatSeverity.HIGH, "DE-System-Directive"),
    (r"(?i)(führe|starte|öffne)\s+(den\s+)?(befehl|kommando|shell)", ThreatSeverity.CRITICAL, "DE-Shell-Injection"),
    (r"(?i)admin\s*zugriff|root\s*zugang|wartungsmodus", ThreatSeverity.HIGH, "DE-Authority-Claim"),
]


class InjectionScanner:
    """Erkennt Prompt-Injections in Memory-Einträgen.

    Scannt Memory-Content auf bekannte Injection-Patterns
    und markiert verdächtige Einträge zur Quarantäne.
    """

    def __init__(
        self,
        *,
        extra_patterns: list[tuple[str, ThreatSeverity, str]] | None = None,
    ) -> None:
        self._patterns = list(_INJECTION_PATTERNS)
        if extra_patterns:
            self._patterns.extend(extra_patterns)
        self._compiled = [(re.compile(p), sev, desc) for p, sev, desc in self._patterns]

    def scan(self, content: str) -> list[MemoryThreat]:
        """Scannt einen Memory-Eintrag auf Injections."""
        threats: list[MemoryThreat] = []
        for pattern, severity, description in self._compiled:
            match = pattern.search(content)
            if match:
                threats.append(MemoryThreat(
                    threat_id=f"INJ-{hashlib.md5(match.group().encode()).hexdigest()[:8]}",
                    threat_type=ThreatType.INJECTION,
                    severity=severity,
                    description=f"Injection erkannt: {description}",
                    entry_content=content,
                    matched_pattern=match.group()[:60],
                    recommended_action="quarantine" if severity in (ThreatSeverity.CRITICAL, ThreatSeverity.HIGH) else "flag",
                ))
        return threats

    def is_clean(self, content: str) -> bool:
        """Schnellprüfung: True wenn keine Injection erkannt."""
        for pattern, _, _ in self._compiled:
            if pattern.search(content):
                return False
        return True


# ============================================================================
# Credential-Leak-Detector
# ============================================================================

_CREDENTIAL_PATTERNS: list[tuple[str, ThreatSeverity, str]] = [
    # API-Keys
    (r"(?i)(api[_-]?key|api[_-]?secret)\s*[=:]\s*\S{10,}", ThreatSeverity.CRITICAL, "API-Key"),
    (r"sk-[a-zA-Z0-9]{20,}", ThreatSeverity.CRITICAL, "OpenAI-Key"),
    (r"ghp_[a-zA-Z0-9]{36}", ThreatSeverity.CRITICAL, "GitHub-Token"),
    (r"xoxb-[0-9]{10,}-[a-zA-Z0-9]+", ThreatSeverity.CRITICAL, "Slack-Bot-Token"),

    # Passwords & Secrets
    (r"(?i)(password|passwd|secret)\s*[=:]\s*\S{6,}", ThreatSeverity.HIGH, "Password"),
    (r"(?i)bearer\s+[a-zA-Z0-9._\-]{20,}", ThreatSeverity.HIGH, "Bearer-Token"),

    # Connection-Strings
    (r"(?i)(mysql|postgres|mongodb)://\S+:\S+@", ThreatSeverity.HIGH, "DB-Connection-String"),
    (r"(?i)-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----", ThreatSeverity.CRITICAL, "Private-Key"),

    # AWS
    (r"AKIA[0-9A-Z]{16}", ThreatSeverity.CRITICAL, "AWS-Access-Key"),
]


class CredentialLeakDetector:
    """Erkennt versehentlich gespeicherte Credentials in Memory.

    Secrets in Memory-Einträgen sind ein Datenleck-Risiko,
    da sie bei RAG-Retrieval exponiert werden können.
    """

    def __init__(
        self,
        *,
        extra_patterns: list[tuple[str, ThreatSeverity, str]] | None = None,
    ) -> None:
        patterns = list(_CREDENTIAL_PATTERNS)
        if extra_patterns:
            patterns.extend(extra_patterns)
        self._compiled = [(re.compile(p), sev, desc) for p, sev, desc in patterns]

    def scan(self, content: str) -> list[MemoryThreat]:
        threats: list[MemoryThreat] = []
        for pattern, severity, description in self._compiled:
            match = pattern.search(content)
            if match:
                threats.append(MemoryThreat(
                    threat_id=f"CRED-{hashlib.md5(match.group()[:20].encode()).hexdigest()[:8]}",
                    threat_type=ThreatType.CREDENTIAL_LEAK,
                    severity=severity,
                    description=f"Credential-Leak: {description}",
                    entry_content=content,
                    matched_pattern="[REDACTED]",  # Credential nicht im Report anzeigen!
                    recommended_action="redact_and_quarantine",
                ))
        return threats

    def has_credentials(self, content: str) -> bool:
        for pattern, _, _ in self._compiled:
            if pattern.search(content):
                return True
        return False


# ============================================================================
# Contradiction-Checker
# ============================================================================


@dataclass
class FactAssertion:
    """Eine extrahierte Fakten-Behauptung aus einem Memory-Eintrag."""

    subject: str
    predicate: str
    value: str
    source_entry_id: str = ""
    timestamp: str = ""

    @property
    def key(self) -> str:
        return f"{self.subject.lower()}:{self.predicate.lower()}"


class ContradictionChecker:
    """Findet widersprüchliche Fakten im Memory.

    Wenn zwei Memory-Einträge entgegengesetzte Behauptungen
    über dasselbe Subjekt machen, wird ein Widerspruch markiert.
    """

    def __init__(self) -> None:
        self._facts: dict[str, list[FactAssertion]] = {}

    def add_fact(self, fact: FactAssertion) -> list[MemoryThreat]:
        """Fügt einen Fakt hinzu und prüft auf Widersprüche."""
        threats: list[MemoryThreat] = []

        key = fact.key
        existing = self._facts.get(key, [])

        for ex in existing:
            if ex.value.lower() != fact.value.lower():
                threats.append(MemoryThreat(
                    threat_id=f"CONTRA-{hashlib.md5(key.encode()).hexdigest()[:8]}",
                    threat_type=ThreatType.CONTRADICTION,
                    severity=ThreatSeverity.MEDIUM,
                    description=(
                        f"Widerspruch: '{fact.subject} {fact.predicate}' ist "
                        f"'{ex.value}' vs. '{fact.value}'"
                    ),
                    entry_content=f"Existing: {ex.value}, New: {fact.value}",
                    recommended_action="review",
                ))

        if key not in self._facts:
            self._facts[key] = []
        self._facts[key].append(fact)
        return threats

    def check_consistency(self) -> list[MemoryThreat]:
        """Prüft alle gespeicherten Fakten auf Widersprüche."""
        threats: list[MemoryThreat] = []
        for key, facts in self._facts.items():
            values = set(f.value.lower() for f in facts)
            if len(values) > 1:
                threats.append(MemoryThreat(
                    threat_id=f"CONTRA-{hashlib.md5(key.encode()).hexdigest()[:8]}",
                    threat_type=ThreatType.CONTRADICTION,
                    severity=ThreatSeverity.MEDIUM,
                    description=f"Widersprüchliche Werte für '{key}': {values}",
                    entry_content=str(values),
                    recommended_action="review",
                ))
        return threats

    @property
    def fact_count(self) -> int:
        return sum(len(v) for v in self._facts.values())

    @property
    def unique_subjects(self) -> int:
        return len(self._facts)

    def clear(self) -> None:
        self._facts.clear()


# ============================================================================
# Integrity-Verifier
# ============================================================================


class IntegrityVerifier:
    """Prüft die Integrität von Memory-Einträgen.

    Verifiziert Hashes, Zeitstempel-Konsistenz und
    erkennt manipulierte Einträge.
    """

    def __init__(self) -> None:
        self._hashes: dict[str, str] = {}  # entry_id -> hash

    def register_entry(self, entry_id: str, content: str) -> str:
        """Registriert einen Eintrag und speichert seinen Hash."""
        h = hashlib.sha256(content.encode()).hexdigest()
        self._hashes[entry_id] = h
        return h

    def verify_entry(self, entry_id: str, content: str) -> bool:
        """Prüft ob ein Eintrag seit der Registrierung verändert wurde."""
        expected = self._hashes.get(entry_id)
        if not expected:
            return True  # Unbekannter Eintrag = nicht verifizierbar
        actual = hashlib.sha256(content.encode()).hexdigest()
        return actual == expected

    def check_integrity(self, entry_id: str, content: str) -> MemoryThreat | None:
        """Prüft Integrität und gibt ggf. eine Bedrohung zurück."""
        if not self.verify_entry(entry_id, content):
            return MemoryThreat(
                threat_id=f"INT-{entry_id[:8]}",
                threat_type=ThreatType.INTEGRITY_VIOLATION,
                severity=ThreatSeverity.HIGH,
                description=f"Hash-Mismatch für Eintrag '{entry_id}'",
                entry_content=content,
                recommended_action="quarantine",
            )
        return None

    @property
    def registered_count(self) -> int:
        return len(self._hashes)


# ============================================================================
# Memory-Hygiene-Engine: Orchestriert alle Checks
# ============================================================================


class MemoryHygieneEngine:
    """Orchestriert alle Memory-Hygiene-Checks.

    Kombiniert Injection-Scanner, Credential-Detector,
    Contradiction-Checker und Integrity-Verifier zu einem
    einheitlichen Scan-Prozess.
    """

    def __init__(self) -> None:
        self.injection_scanner = InjectionScanner()
        self.credential_detector = CredentialLeakDetector()
        self.contradiction_checker = ContradictionChecker()
        self.integrity_verifier = IntegrityVerifier()
        self._quarantine: list[dict[str, Any]] = []
        self._scan_history: list[HygieneReport] = []

    def scan_entry(self, entry_id: str, content: str, *, source: str = "") -> list[MemoryThreat]:
        """Scannt einen einzelnen Memory-Eintrag auf alle Bedrohungen."""
        threats: list[MemoryThreat] = []

        # 1. Injection-Check
        threats.extend(self.injection_scanner.scan(content))

        # 2. Credential-Check
        threats.extend(self.credential_detector.scan(content))

        # 3. Integrity-Check
        integrity_threat = self.integrity_verifier.check_integrity(entry_id, content)
        if integrity_threat:
            threats.append(integrity_threat)

        return threats

    def scan_batch(
        self,
        entries: list[dict[str, Any]],
        *,
        auto_quarantine: bool = True,
    ) -> HygieneReport:
        """Scannt eine Batch von Memory-Einträgen.

        Args:
            entries: Liste von Dicts mit 'id' und 'content' Keys.
            auto_quarantine: Bei True werden gefährliche Einträge quarantäniert.
        """
        start = time.time()
        report = HygieneReport(
            report_id=hashlib.sha256(str(time.time()).encode()).hexdigest()[:16],
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        for entry in entries:
            entry_id = entry.get("id", "unknown")
            content = entry.get("content", "")
            report.scanned_entries += 1

            threats = self.scan_entry(entry_id, content, source=entry.get("source", ""))

            if threats:
                report.threats_found += len(threats)
                report.threats.extend(threats)

                if auto_quarantine and any(
                    t.severity in (ThreatSeverity.CRITICAL, ThreatSeverity.HIGH)
                    for t in threats
                ):
                    self._quarantine.append(entry)
                    report.quarantined += 1
            else:
                report.clean_entries += 1

        report.scan_duration_ms = int((time.time() - start) * 1000)
        self._scan_history.append(report)
        return report

    def quarantine(self) -> list[dict[str, Any]]:
        """Gibt quarantänierte Einträge zurück."""
        return list(self._quarantine)

    def release_from_quarantine(self, entry_id: str) -> bool:
        """Gibt einen Eintrag aus der Quarantäne frei."""
        for i, entry in enumerate(self._quarantine):
            if entry.get("id") == entry_id:
                self._quarantine.pop(i)
                return True
        return False

    def scan_history(self) -> list[HygieneReport]:
        return list(self._scan_history)

    def stats(self) -> dict[str, Any]:
        total_scanned = sum(r.scanned_entries for r in self._scan_history)
        total_threats = sum(r.threats_found for r in self._scan_history)
        return {
            "total_scans": len(self._scan_history),
            "total_scanned": total_scanned,
            "total_threats": total_threats,
            "quarantined": len(self._quarantine),
            "integrity_entries": self.integrity_verifier.registered_count,
            "contradiction_facts": self.contradiction_checker.fact_count,
            "threat_rate": round((total_threats / total_scanned * 100), 2) if total_scanned else 0.0,
        }


# ============================================================================
# Memory Version Control (Punkt 3: Versionskontrolle)
# ============================================================================


@dataclass
class MemorySnapshot:
    """Ein Snapshot des Memory-Stores zu einem bestimmten Zeitpunkt."""

    snapshot_id: str
    timestamp: str
    entry_count: int
    total_size_bytes: int
    content_hash: str       # SHA-256 des gesamten Inhalts
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp,
            "entries": self.entry_count,
            "size_bytes": self.total_size_bytes,
            "hash": self.content_hash[:16] + "...",
        }


class MemoryVersionControl:
    """Versionskontrolle für den Memory-Store.

    Erstellt periodische Snapshots und erkennt unerwartete Änderungen.
    """

    def __init__(self) -> None:
        self._snapshots: list[MemorySnapshot] = []
        self._counter = 0

    def snapshot(self, entries: list[dict[str, Any]]) -> MemorySnapshot:
        """Erstellt einen Snapshot des aktuellen Zustands."""
        self._counter += 1
        import json
        content = json.dumps(entries, sort_keys=True, ensure_ascii=False)
        snap = MemorySnapshot(
            snapshot_id=f"SNAP-{self._counter:04d}",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            entry_count=len(entries),
            total_size_bytes=len(content.encode()),
            content_hash=hashlib.sha256(content.encode()).hexdigest(),
        )
        self._snapshots.append(snap)
        return snap

    def diff(self, snap_a: str, snap_b: str) -> dict[str, Any]:
        """Vergleicht zwei Snapshots."""
        a = self._by_id(snap_a)
        b = self._by_id(snap_b)
        if not a or not b:
            return {"error": "Snapshot nicht gefunden"}
        return {
            "entries_diff": b.entry_count - a.entry_count,
            "size_diff_bytes": b.total_size_bytes - a.total_size_bytes,
            "hash_changed": a.content_hash != b.content_hash,
            "from": a.timestamp,
            "to": b.timestamp,
        }

    def detect_drift(self, max_change_rate: float = 20.0) -> dict[str, Any]:
        """Erkennt unerwarteten Drift zwischen aufeinanderfolgenden Snapshots.

        Args:
            max_change_rate: Max. erlaubte Änderungsrate in % pro Snapshot.

        Returns:
            Drift-Analyse mit Warnungen.
        """
        if len(self._snapshots) < 2:
            return {"drift_detected": False, "message": "Zu wenige Snapshots"}

        warnings: list[str] = []
        for i in range(1, len(self._snapshots)):
            prev = self._snapshots[i - 1]
            curr = self._snapshots[i]
            if prev.entry_count > 0:
                change_rate = abs(curr.entry_count - prev.entry_count) / prev.entry_count * 100
                if change_rate > max_change_rate:
                    warnings.append(
                        f"{prev.snapshot_id}→{curr.snapshot_id}: {change_rate:.1f}% Änderung "
                        f"({prev.entry_count}→{curr.entry_count} Einträge)"
                    )

        return {
            "drift_detected": len(warnings) > 0,
            "warnings": warnings,
            "total_snapshots": len(self._snapshots),
        }

    def _by_id(self, snapshot_id: str) -> MemorySnapshot | None:
        return next((s for s in self._snapshots if s.snapshot_id == snapshot_id), None)

    @property
    def snapshot_count(self) -> int:
        return len(self._snapshots)

    def latest(self) -> MemorySnapshot | None:
        return self._snapshots[-1] if self._snapshots else None

    def all_snapshots(self) -> list[MemorySnapshot]:
        return list(self._snapshots)


# ============================================================================
# Duplicate Detector (Punkt 3: Dublettenerkennung)
# ============================================================================


class DuplicateDetector:
    """Erkennt Dubletten und quasi-identische Einträge im Memory-Store."""

    @staticmethod
    def find_duplicates(
        entries: list[dict[str, Any]],
        *,
        key_field: str = "content",
        threshold: float = 0.9,
    ) -> list[tuple[int, int, float]]:
        """Findet Duplikate basierend auf Textähnlichkeit.

        Returns:
            Liste von (index_a, index_b, similarity) Tupeln.
        """
        duplicates: list[tuple[int, int, float]] = []
        contents = [str(e.get(key_field, "")) for e in entries]

        for i in range(len(contents)):
            for j in range(i + 1, len(contents)):
                sim = DuplicateDetector._similarity(contents[i], contents[j])
                if sim >= threshold:
                    duplicates.append((i, j, round(sim, 3)))

        return duplicates

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """Einfache Jaccard-Ähnlichkeit auf Wort-Ebene."""
        if not a or not b:
            return 0.0
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)


# ============================================================================
# Poisoning Prevention (Punkt 3: automatische Erkennung)
# ============================================================================


class PoisoningIndicator(Enum):
    SUDDEN_TOPIC_SHIFT = "sudden_topic_shift"
    INSTRUCTION_PATTERN = "instruction_pattern"
    AUTHORITY_CLAIM = "authority_claim"
    REPETITIVE_INJECTION = "repetitive_injection"
    CONTRADICTS_BASELINE = "contradicts_baseline"
    SPAM_CONTENT = "spam_content"


@dataclass
class PoisoningAlert:
    """Ein Alarm bei erkanntem Poisoning-Versuch."""

    alert_id: str
    indicator: PoisoningIndicator
    severity: str           # low, medium, high, critical
    entry_index: int
    evidence: str
    auto_quarantined: bool = False
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "indicator": self.indicator.value,
            "severity": self.severity,
            "entry_index": self.entry_index,
            "quarantined": self.auto_quarantined,
        }


class PoisoningPreventor:
    """Erkennt und verhindert Memory-Poisoning-Angriffe.

    Prüft neue Einträge auf verdächtige Muster:
    - Plötzliche Themenwechsel
    - Eingebettete Anweisungen
    - Autoritäts-Behauptungen
    - Wiederholte Injections
    - Widersprüche zur Baseline
    - Spam-Inhalte
    """

    INSTRUCTION_KEYWORDS = [
        "ignore previous", "new instructions", "system prompt",
        "override", "admin access", "execute command",
        "you must now", "forget everything", "from now on",
    ]

    AUTHORITY_KEYWORDS = [
        "as an admin", "i am the developer", "maintenance mode",
        "debugging session", "authorized override", "root access",
    ]

    SPAM_INDICATORS = [
        "buy now", "limited offer", "click here", "free money",
        "congratulations you won", "act fast",
    ]

    def __init__(self) -> None:
        self._alerts: list[PoisoningAlert] = []
        self._counter = 0
        self._baseline_topics: set[str] = set()

    def set_baseline(self, topics: list[str]) -> None:
        """Setzt die erlaubten Basis-Themen."""
        self._baseline_topics = set(t.lower() for t in topics)

    def scan_entry(self, content: str, entry_index: int = 0) -> list[PoisoningAlert]:
        """Scannt einen einzelnen Eintrag auf Poisoning-Indikatoren."""
        alerts: list[PoisoningAlert] = []
        content_lower = content.lower()

        # Instruction-Patterns
        for kw in self.INSTRUCTION_KEYWORDS:
            if kw in content_lower:
                alerts.append(self._create_alert(
                    PoisoningIndicator.INSTRUCTION_PATTERN, "critical",
                    entry_index, f"Instruction-Pattern: '{kw}'",
                ))
                break

        # Authority-Claims
        for kw in self.AUTHORITY_KEYWORDS:
            if kw in content_lower:
                alerts.append(self._create_alert(
                    PoisoningIndicator.AUTHORITY_CLAIM, "high",
                    entry_index, f"Authority-Claim: '{kw}'",
                ))
                break

        # Spam
        spam_count = sum(1 for kw in self.SPAM_INDICATORS if kw in content_lower)
        if spam_count >= 2:
            alerts.append(self._create_alert(
                PoisoningIndicator.SPAM_CONTENT, "medium",
                entry_index, f"{spam_count} Spam-Indikatoren erkannt",
            ))

        return alerts

    def scan_batch(self, entries: list[dict[str, Any]], key: str = "content") -> list[PoisoningAlert]:
        """Scannt eine Batch von Einträgen."""
        all_alerts = []
        for i, entry in enumerate(entries):
            content = str(entry.get(key, ""))
            alerts = self.scan_entry(content, i)
            all_alerts.extend(alerts)
        return all_alerts

    def _create_alert(
        self, indicator: PoisoningIndicator, severity: str,
        entry_index: int, evidence: str,
    ) -> PoisoningAlert:
        self._counter += 1
        alert = PoisoningAlert(
            alert_id=f"POI-{self._counter:04d}",
            indicator=indicator,
            severity=severity,
            entry_index=entry_index,
            evidence=evidence,
            auto_quarantined=severity in ("critical", "high"),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self._alerts.append(alert)
        return alert

    @property
    def alert_count(self) -> int:
        return len(self._alerts)

    def critical_alerts(self) -> list[PoisoningAlert]:
        return [a for a in self._alerts if a.severity == "critical"]

    def stats(self) -> dict[str, Any]:
        return {
            "total_alerts": len(self._alerts),
            "critical": sum(1 for a in self._alerts if a.severity == "critical"),
            "high": sum(1 for a in self._alerts if a.severity == "high"),
            "auto_quarantined": sum(1 for a in self._alerts if a.auto_quarantined),
        }


# ============================================================================
# Source Integrity Checker (Quellenintegrität)
# ============================================================================


@dataclass
class SourceTrust:
    """Vertrauenswürdigkeit einer Wissensquelle."""

    source_id: str
    name: str
    trust_score: float = 1.0   # 0-1 (1 = voll vertrauenswürdig)
    verified: bool = False
    total_entries: int = 0
    flagged_entries: int = 0

    @property
    def reliability(self) -> str:
        if self.trust_score >= 0.9:
            return "excellent"
        if self.trust_score >= 0.7:
            return "good"
        if self.trust_score >= 0.5:
            return "moderate"
        return "unreliable"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "name": self.name,
            "trust": self.trust_score,
            "reliability": self.reliability,
            "verified": self.verified,
        }


class SourceIntegrityChecker:
    """Bewertet die Vertrauenswürdigkeit von Wissensquellen."""

    def __init__(self) -> None:
        self._sources: dict[str, SourceTrust] = {}

    def register_source(self, source_id: str, name: str, *, verified: bool = False) -> SourceTrust:
        source = SourceTrust(source_id=source_id, name=name, verified=verified)
        if verified:
            source.trust_score = 1.0
        self._sources[source_id] = source
        return source

    def report_entry(self, source_id: str, flagged: bool = False) -> None:
        source = self._sources.get(source_id)
        if not source:
            return
        source.total_entries += 1
        if flagged:
            source.flagged_entries += 1
        # Trust-Score anpassen
        if source.total_entries > 0:
            flag_rate = source.flagged_entries / source.total_entries
            source.trust_score = round(max(0, 1 - flag_rate * 2), 3)

    def get_source(self, source_id: str) -> SourceTrust | None:
        return self._sources.get(source_id)

    def unreliable_sources(self) -> list[SourceTrust]:
        return [s for s in self._sources.values() if s.trust_score < 0.5]

    @property
    def source_count(self) -> int:
        return len(self._sources)

    def stats(self) -> dict[str, Any]:
        sources = list(self._sources.values())
        return {
            "total_sources": len(sources),
            "verified": sum(1 for s in sources if s.verified),
            "unreliable": len(self.unreliable_sources()),
            "avg_trust": round(sum(s.trust_score for s in sources) / len(sources), 3) if sources else 0,
        }
