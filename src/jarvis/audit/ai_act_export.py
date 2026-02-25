"""Jarvis · EU AI Act Compliance-Export.

Standardisierte Export-Funktionen für gesetzliche Vorgaben:

  - RiskClassification:    AI-Act Risikoklassen (Unacceptable/High/Limited/Minimal)
  - RiskAssessment:        Strukturierte Risikobewertung
  - AIActReport:           Vollständiger Compliance-Bericht
  - MitigationPlan:        Dokumentierte Gegenmaßnahmen
  - TransparencyObligation: Transparenzpflichten
  - ComplianceExporter:    Export als JSON/Markdown/HTML
  - AuditTrailExporter:    Entscheidungs-Log für Behörden

Architektur-Bibel: §13.2 (Compliance), §16.3 (EU AI Act)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ============================================================================
# Risk Classification (EU AI Act Art. 6)
# ============================================================================


class RiskLevel(Enum):
    UNACCEPTABLE = "unacceptable"  # Art. 5: Verboten
    HIGH = "high"                   # Art. 6: Strenge Auflagen
    LIMITED = "limited"             # Art. 50: Transparenzpflichten
    MINIMAL = "minimal"             # Keine Auflagen


class SystemCategory(Enum):
    BIOMETRIC = "biometric_identification"
    CRITICAL_INFRASTRUCTURE = "critical_infrastructure"
    EDUCATION = "education_training"
    EMPLOYMENT = "employment"
    ESSENTIAL_SERVICES = "essential_services"
    LAW_ENFORCEMENT = "law_enforcement"
    MIGRATION = "migration_border"
    JUSTICE = "justice_democracy"
    GENERAL_PURPOSE = "general_purpose_ai"
    CHATBOT = "chatbot"
    CONTENT_GENERATION = "content_generation"
    OTHER = "other"


@dataclass
class RiskAssessment:
    """Strukturierte Risikobewertung nach EU AI Act."""

    assessment_id: str
    system_name: str
    system_version: str
    category: SystemCategory
    risk_level: RiskLevel
    intended_purpose: str = ""
    risk_factors: list[str] = field(default_factory=list)
    affected_persons: list[str] = field(default_factory=list)
    data_types: list[str] = field(default_factory=list)
    assessed_by: str = ""
    assessed_at: str = ""
    valid_until: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "system_name": self.system_name,
            "version": self.system_version,
            "category": self.category.value,
            "risk_level": self.risk_level.value,
            "intended_purpose": self.intended_purpose,
            "risk_factors": self.risk_factors,
            "affected_persons": self.affected_persons,
            "data_types": self.data_types,
            "assessed_by": self.assessed_by,
            "assessed_at": self.assessed_at,
        }


class RiskClassifier:
    """Klassifiziert AI-Systeme nach EU AI Act Risikoklassen."""

    CATEGORY_RISK_MAP: dict[SystemCategory, RiskLevel] = {
        SystemCategory.BIOMETRIC: RiskLevel.UNACCEPTABLE,
        SystemCategory.LAW_ENFORCEMENT: RiskLevel.HIGH,
        SystemCategory.CRITICAL_INFRASTRUCTURE: RiskLevel.HIGH,
        SystemCategory.EDUCATION: RiskLevel.HIGH,
        SystemCategory.EMPLOYMENT: RiskLevel.HIGH,
        SystemCategory.ESSENTIAL_SERVICES: RiskLevel.HIGH,
        SystemCategory.MIGRATION: RiskLevel.HIGH,
        SystemCategory.JUSTICE: RiskLevel.HIGH,
        SystemCategory.CHATBOT: RiskLevel.LIMITED,
        SystemCategory.CONTENT_GENERATION: RiskLevel.LIMITED,
        SystemCategory.GENERAL_PURPOSE: RiskLevel.LIMITED,
        SystemCategory.OTHER: RiskLevel.MINIMAL,
    }

    def __init__(self) -> None:
        self._assessments: list[RiskAssessment] = []
        self._counter = 0

    def assess(
        self,
        system_name: str,
        system_version: str,
        category: SystemCategory,
        *,
        intended_purpose: str = "",
        risk_factors: list[str] | None = None,
        affected_persons: list[str] | None = None,
        data_types: list[str] | None = None,
        assessed_by: str = "",
    ) -> RiskAssessment:
        self._counter += 1
        risk_level = self.CATEGORY_RISK_MAP.get(category, RiskLevel.MINIMAL)

        assessment = RiskAssessment(
            assessment_id=f"RA-{self._counter:04d}",
            system_name=system_name,
            system_version=system_version,
            category=category,
            risk_level=risk_level,
            intended_purpose=intended_purpose,
            risk_factors=risk_factors or [],
            affected_persons=affected_persons or [],
            data_types=data_types or [],
            assessed_by=assessed_by,
            assessed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self._assessments.append(assessment)
        return assessment

    def latest(self) -> RiskAssessment | None:
        return self._assessments[-1] if self._assessments else None

    def all_assessments(self) -> list[RiskAssessment]:
        return list(self._assessments)

    def stats(self) -> dict[str, Any]:
        return {
            "total_assessments": len(self._assessments),
            "by_risk_level": {
                level.value: sum(1 for a in self._assessments if a.risk_level == level)
                for level in RiskLevel
                if any(a.risk_level == level for a in self._assessments)
            },
        }


# ============================================================================
# Mitigation Plans
# ============================================================================


class MitigationStatus(Enum):
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    IMPLEMENTED = "implemented"
    VERIFIED = "verified"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class MitigationMeasure:
    """Dokumentierte Gegenmaßnahme."""

    measure_id: str
    risk_factor: str
    description: str
    status: MitigationStatus = MitigationStatus.PLANNED
    responsible: str = ""
    deadline: str = ""
    evidence: str = ""
    verified_by: str = ""
    verified_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "measure_id": self.measure_id,
            "risk_factor": self.risk_factor,
            "description": self.description,
            "status": self.status.value,
            "responsible": self.responsible,
            "deadline": self.deadline,
            "verified": bool(self.verified_by),
        }


class MitigationTracker:
    """Verfolgt und dokumentiert Gegenmaßnahmen."""

    def __init__(self) -> None:
        self._measures: list[MitigationMeasure] = []
        self._counter = 0

    def add(
        self,
        risk_factor: str,
        description: str,
        *,
        responsible: str = "",
        deadline: str = "",
    ) -> MitigationMeasure:
        self._counter += 1
        measure = MitigationMeasure(
            measure_id=f"MIT-{self._counter:04d}",
            risk_factor=risk_factor,
            description=description,
            responsible=responsible,
            deadline=deadline,
        )
        self._measures.append(measure)
        return measure

    def update_status(self, measure_id: str, status: MitigationStatus) -> bool:
        for m in self._measures:
            if m.measure_id == measure_id:
                m.status = status
                return True
        return False

    def verify(self, measure_id: str, verified_by: str) -> bool:
        for m in self._measures:
            if m.measure_id == measure_id:
                m.status = MitigationStatus.VERIFIED
                m.verified_by = verified_by
                m.verified_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                return True
        return False

    def all_measures(self) -> list[MitigationMeasure]:
        return list(self._measures)

    def completion_rate(self) -> float:
        if not self._measures:
            return 100.0
        done = sum(1 for m in self._measures
                   if m.status in (MitigationStatus.IMPLEMENTED, MitigationStatus.VERIFIED))
        return done / len(self._measures) * 100

    @property
    def measure_count(self) -> int:
        return len(self._measures)

    def stats(self) -> dict[str, Any]:
        return {
            "total_measures": len(self._measures),
            "completion_rate": round(self.completion_rate(), 1),
            "by_status": {
                st.value: sum(1 for m in self._measures if m.status == st)
                for st in MitigationStatus
                if any(m.status == st for m in self._measures)
            },
        }


# ============================================================================
# Transparency Obligations (Art. 50)
# ============================================================================


@dataclass
class TransparencyObligation:
    """Transparenzpflicht nach EU AI Act Art. 50."""

    obligation_id: str
    article: str        # z.B. "Art. 50(1)"
    description: str
    implemented: bool = False
    evidence: str = ""
    last_checked: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "obligation_id": self.obligation_id,
            "article": self.article,
            "description": self.description,
            "implemented": self.implemented,
            "evidence": self.evidence,
        }


class TransparencyChecker:
    """Prüft ob Transparenzpflichten erfüllt sind."""

    DEFAULT_OBLIGATIONS = [
        TransparencyObligation("T-001", "Art. 50(1)",
            "Nutzer müssen informiert werden, dass sie mit einem AI-System interagieren."),
        TransparencyObligation("T-002", "Art. 50(2)",
            "AI-generierte Inhalte müssen als solche gekennzeichnet werden."),
        TransparencyObligation("T-003", "Art. 50(3)",
            "Deep-Fakes müssen als künstlich generiert gekennzeichnet werden."),
        TransparencyObligation("T-004", "Art. 50(4)",
            "Texte zu Themen öffentlichen Interesses müssen als AI-generiert markiert werden."),
        TransparencyObligation("T-005", "Art. 13",
            "Entscheidungen müssen nachvollziehbar und erklärbar sein."),
        TransparencyObligation("T-006", "Art. 14",
            "Menschliche Aufsicht muss sichergestellt sein."),
    ]

    def __init__(self, obligations: list[TransparencyObligation] | None = None) -> None:
        self._obligations = obligations if obligations is not None else list(self.DEFAULT_OBLIGATIONS)

    def check_all(self) -> dict[str, Any]:
        total = len(self._obligations)
        implemented = sum(1 for o in self._obligations if o.implemented)
        return {
            "total_obligations": total,
            "implemented": implemented,
            "compliance_rate": round(implemented / max(1, total) * 100, 1),
            "missing": [o.to_dict() for o in self._obligations if not o.implemented],
        }

    def mark_implemented(self, obligation_id: str, evidence: str = "") -> bool:
        for o in self._obligations:
            if o.obligation_id == obligation_id:
                o.implemented = True
                o.evidence = evidence
                o.last_checked = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                return True
        return False

    def all_obligations(self) -> list[TransparencyObligation]:
        return list(self._obligations)

    def stats(self) -> dict[str, Any]:
        return self.check_all()


# ============================================================================
# Compliance Report
# ============================================================================


@dataclass
class AIActReport:
    """Vollständiger Compliance-Bericht nach EU AI Act."""

    report_id: str
    system_name: str
    system_version: str
    generated_at: str
    risk_assessment: dict[str, Any] = field(default_factory=dict)
    mitigations: list[dict[str, Any]] = field(default_factory=list)
    transparency: dict[str, Any] = field(default_factory=dict)
    security_summary: dict[str, Any] = field(default_factory=dict)
    ethics_summary: dict[str, Any] = field(default_factory=dict)
    audit_trail_entries: int = 0
    overall_compliance: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "system_name": self.system_name,
            "version": self.system_version,
            "generated_at": self.generated_at,
            "risk_assessment": self.risk_assessment,
            "mitigations": self.mitigations,
            "transparency": self.transparency,
            "security_summary": self.security_summary,
            "ethics_summary": self.ethics_summary,
            "audit_trail_entries": self.audit_trail_entries,
            "overall_compliance": round(self.overall_compliance, 1),
        }


# ============================================================================
# Compliance Exporter
# ============================================================================


class ComplianceExporter:
    """Exportiert Compliance-Berichte in verschiedenen Formaten."""

    def __init__(self) -> None:
        self._classifier = RiskClassifier()
        self._mitigations = MitigationTracker()
        self._transparency = TransparencyChecker()
        self._reports: list[AIActReport] = []

    @property
    def classifier(self) -> RiskClassifier:
        return self._classifier

    @property
    def mitigations(self) -> MitigationTracker:
        return self._mitigations

    @property
    def transparency(self) -> TransparencyChecker:
        return self._transparency

    def generate_report(
        self,
        system_name: str = "Jarvis",
        system_version: str = "11.0",
        *,
        security_summary: dict[str, Any] | None = None,
        ethics_summary: dict[str, Any] | None = None,
        audit_entries: int = 0,
    ) -> AIActReport:
        """Generiert einen vollständigen Compliance-Bericht."""
        risk = self._classifier.latest()
        trans = self._transparency.check_all()
        mits = [m.to_dict() for m in self._mitigations.all_measures()]

        # Compliance-Score berechnen
        scores = [trans.get("compliance_rate", 0), self._mitigations.completion_rate()]
        if risk:
            scores.append(100.0 if risk.risk_level != RiskLevel.UNACCEPTABLE else 0.0)
        overall = sum(scores) / len(scores) if scores else 0.0

        report = AIActReport(
            report_id=f"AIAR-{int(time.time())}",
            system_name=system_name,
            system_version=system_version,
            generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            risk_assessment=risk.to_dict() if risk else {},
            mitigations=mits,
            transparency=trans,
            security_summary=security_summary or {},
            ethics_summary=ethics_summary or {},
            audit_trail_entries=audit_entries,
            overall_compliance=overall,
        )
        self._reports.append(report)
        return report

    def export_json(self, report: AIActReport) -> str:
        return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)

    def export_markdown(self, report: AIActReport) -> str:
        d = report.to_dict()
        lines = [
            f"# EU AI Act Compliance Report — {d['system_name']} v{d['version']}",
            f"",
            f"**Report-ID:** {d['report_id']}",
            f"**Erstellt:** {d['generated_at']}",
            f"**Gesamt-Compliance:** {d['overall_compliance']}%",
            f"",
            f"## 1. Risikobewertung",
        ]

        ra = d.get("risk_assessment", {})
        if ra:
            lines.append(f"- **Kategorie:** {ra.get('category', 'N/A')}")
            lines.append(f"- **Risikoklasse:** {ra.get('risk_level', 'N/A')}")
            lines.append(f"- **Zweck:** {ra.get('intended_purpose', 'N/A')}")
        else:
            lines.append("_Keine Risikobewertung durchgeführt._")

        lines.extend(["", "## 2. Transparenzpflichten"])
        trans = d.get("transparency", {})
        lines.append(f"- **Erfüllungsquote:** {trans.get('compliance_rate', 0)}%")
        lines.append(f"- **Gesamt:** {trans.get('total_obligations', 0)}")
        lines.append(f"- **Umgesetzt:** {trans.get('implemented', 0)}")

        lines.extend(["", "## 3. Gegenmaßnahmen"])
        for m in d.get("mitigations", []):
            lines.append(f"- [{m.get('status', '')}] **{m.get('risk_factor', '')}**: {m.get('description', '')}")

        lines.extend([
            "", "## 4. Audit-Trail",
            f"- **Protokollierte Entscheidungen:** {d.get('audit_trail_entries', 0)}",
            "",
            "---",
            f"_Bericht generiert am {d['generated_at']} durch Jarvis ComplianceExporter._",
        ])

        return "\n".join(lines)

    def latest_report(self) -> AIActReport | None:
        return self._reports[-1] if self._reports else None

    @property
    def report_count(self) -> int:
        return len(self._reports)

    def stats(self) -> dict[str, Any]:
        return {
            "total_reports": len(self._reports),
            "risk_assessments": self._classifier.stats(),
            "mitigations": self._mitigations.stats(),
            "transparency": self._transparency.stats(),
            "last_compliance": round(
                self._reports[-1].overall_compliance, 1
            ) if self._reports else None,
        }
