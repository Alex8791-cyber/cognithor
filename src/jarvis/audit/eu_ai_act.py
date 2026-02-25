"""Jarvis · EU-AI-Act Compliance & Audit-Berichte.

Standardisierte Compliance-Dokumentation:

  - RiskClassifier:        Klassifiziert Systeme nach EU-AI-Act Risikolevels
  - ComplianceDocument:    Strukturierte Pflichtdokumentation
  - ComplianceReport:      Audit-Berichte generieren
  - TechnicalDocumentation: Technische Dokumentation nach Art. 11
  - TransparencyRegister:  Transparenzregister nach Art. 52
  - TrainingMaterial:      Schulungsmaterial für Sicherheitsteams
  - EUAIActGovernor:       Hauptklasse

Architektur-Bibel: §16.1 (Regulierung), §16.4 (EU-AI-Act)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ============================================================================
# Risk Classification (Art. 6, Annex III)
# ============================================================================

from jarvis.audit.compliance import RiskLevel


class SystemCategory(Enum):
    """Kategorien nach Annex III."""
    BIOMETRIC = "biometric_identification"
    CRITICAL_INFRASTRUCTURE = "critical_infrastructure"
    EDUCATION = "education_training"
    EMPLOYMENT = "employment_hr"
    ESSENTIAL_SERVICES = "essential_services"
    LAW_ENFORCEMENT = "law_enforcement"
    BORDER_CONTROL = "border_control"
    JUSTICE = "justice_democratic"
    GENERAL_PURPOSE = "general_purpose_ai"
    INSURANCE_ADVISORY = "insurance_advisory"
    CUSTOM = "custom"


@dataclass
class RiskAssessment:
    """Ergebnis einer Risikobewertung."""

    assessment_id: str
    system_name: str
    risk_level: RiskLevel
    category: SystemCategory
    description: str = ""
    obligations: list[str] = field(default_factory=list)
    mitigation_measures: list[str] = field(default_factory=list)
    assessed_at: str = ""
    assessed_by: str = ""
    valid_until: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "system": self.system_name,
            "risk_level": self.risk_level.value,
            "category": self.category.value,
            "obligations": self.obligations,
            "mitigations": self.mitigation_measures,
            "assessed_at": self.assessed_at,
        }


class RiskClassifier:
    """Klassifiziert KI-Systeme nach EU-AI-Act Risikolevels."""

    # Pflichten pro Risikolevel
    OBLIGATIONS: dict[RiskLevel, list[str]] = {
        RiskLevel.UNACCEPTABLE: [
            "System darf NICHT in der EU betrieben werden (Art. 5)",
        ],
        RiskLevel.HIGH: [
            "Art. 9: Risikomanagement-System einführen",
            "Art. 10: Daten-Governance sicherstellen",
            "Art. 11: Technische Dokumentation erstellen",
            "Art. 12: Record-Keeping / Logging aktivieren",
            "Art. 13: Transparenz für Nutzer gewährleisten",
            "Art. 14: Menschliche Aufsicht sicherstellen",
            "Art. 15: Genauigkeit, Robustheit, Cybersecurity",
            "Art. 17: Qualitätsmanagementsystem",
            "Art. 62: Schwerwiegende Vorfälle melden",
        ],
        RiskLevel.LIMITED: [
            "Art. 52: Transparenzpflicht (KI-Nutzung offenlegen)",
            "Art. 52(1): Nutzer über KI-Interaktion informieren",
        ],
        RiskLevel.MINIMAL: [
            "Freiwillige Verhaltenskodizes (Art. 69)",
        ],
    }

    def __init__(self) -> None:
        self._assessments: list[RiskAssessment] = []
        self._counter = 0

    def classify(
        self,
        system_name: str,
        category: SystemCategory,
        *,
        description: str = "",
        assessed_by: str = "system",
    ) -> RiskAssessment:
        """Klassifiziert ein System basierend auf seiner Kategorie."""
        self._counter += 1

        # Risikolevel bestimmen
        if category in (SystemCategory.BIOMETRIC, SystemCategory.LAW_ENFORCEMENT):
            risk = RiskLevel.HIGH
        elif category in (
            SystemCategory.CRITICAL_INFRASTRUCTURE,
            SystemCategory.EMPLOYMENT,
            SystemCategory.ESSENTIAL_SERVICES,
            SystemCategory.JUSTICE,
            SystemCategory.EDUCATION,
            SystemCategory.BORDER_CONTROL,
        ):
            risk = RiskLevel.HIGH
        elif category == SystemCategory.INSURANCE_ADVISORY:
            risk = RiskLevel.HIGH  # Art. 6(2) + Annex III Nr. 5(b)
        elif category == SystemCategory.GENERAL_PURPOSE:
            risk = RiskLevel.LIMITED
        else:
            risk = RiskLevel.MINIMAL

        assessment = RiskAssessment(
            assessment_id=f"RA-{self._counter:04d}",
            system_name=system_name,
            risk_level=risk,
            category=category,
            description=description,
            obligations=list(self.OBLIGATIONS.get(risk, [])),
            assessed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            assessed_by=assessed_by,
        )
        self._assessments.append(assessment)
        return assessment

    def all_assessments(self) -> list[RiskAssessment]:
        return list(self._assessments)

    def high_risk_systems(self) -> list[RiskAssessment]:
        return [a for a in self._assessments if a.risk_level == RiskLevel.HIGH]

    @property
    def assessment_count(self) -> int:
        return len(self._assessments)

    def stats(self) -> dict[str, Any]:
        a = self._assessments
        return {
            "total_assessments": len(a),
            "by_risk_level": {
                level.value: sum(1 for x in a if x.risk_level == level)
                for level in RiskLevel
                if any(x.risk_level == level for x in a)
            },
        }

    def classify_skill(
        self,
        skill_name: str,
        *,
        accesses_pii: bool = False,
        makes_decisions: bool = False,
        interacts_with_users: bool = False,
        uses_external_apis: bool = False,
        description: str = "",
    ) -> RiskAssessment:
        """Klassifiziert einen einzelnen Skill nach seinem Risikoprofil.

        Automatische Einstufung basierend auf Skill-Eigenschaften:
        - PII-Zugriff → HIGH
        - Entscheidungsfindung → HIGH
        - User-Interaktion → LIMITED
        - Nur intern → MINIMAL
        """
        if accesses_pii or makes_decisions:
            category = SystemCategory.ESSENTIAL_SERVICES
        elif interacts_with_users:
            category = SystemCategory.GENERAL_PURPOSE
        else:
            category = SystemCategory.CUSTOM

        assessment = self.classify(
            f"Skill: {skill_name}",
            category,
            description=description or f"Skill '{skill_name}' Risikobewertung",
            assessed_by="skill_classifier",
        )

        # Zusätzliche Mitigationsmaßnahmen
        if accesses_pii:
            assessment.mitigation_measures.append("DSGVO-Konformität sicherstellen")
            assessment.mitigation_measures.append("Datenminimierung anwenden")
        if uses_external_apis:
            assessment.mitigation_measures.append("API-Zugriff über Sandbox leiten")
        if makes_decisions:
            assessment.mitigation_measures.append("Human-in-the-Loop für kritische Entscheidungen")

        return assessment


# ============================================================================
# Compliance Documentation (Art. 11)
# ============================================================================


class DocumentType(Enum):
    TECHNICAL_DOC = "technical_documentation"    # Art. 11
    RISK_ASSESSMENT = "risk_assessment"          # Art. 9
    QUALITY_MANAGEMENT = "quality_management"    # Art. 17
    CONFORMITY = "conformity_assessment"         # Art. 43
    INCIDENT_REPORT = "incident_report"          # Art. 62
    TRANSPARENCY = "transparency_notice"         # Art. 52
    DATA_GOVERNANCE = "data_governance"          # Art. 10
    MONITORING_PLAN = "monitoring_plan"          # Art. 61
    TRAINING_RECORD = "training_record"


@dataclass
class ComplianceDocument:
    """Strukturierte Pflichtdokumentation."""

    doc_id: str
    doc_type: DocumentType
    title: str
    system_name: str
    content: dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"
    created_at: str = ""
    updated_at: str = ""
    approved_by: str = ""
    status: str = "draft"  # draft, review, approved, archived

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "type": self.doc_type.value,
            "title": self.title,
            "system": self.system_name,
            "version": self.version,
            "status": self.status,
            "created_at": self.created_at,
            "sections": list(self.content.keys()),
        }


class ComplianceDocManager:
    """Verwaltet alle Compliance-Dokumente."""

    def __init__(self) -> None:
        self._documents: dict[str, ComplianceDocument] = {}
        self._counter = 0

    def create(
        self,
        doc_type: DocumentType,
        title: str,
        system_name: str,
        content: dict[str, Any] | None = None,
    ) -> ComplianceDocument:
        self._counter += 1
        doc = ComplianceDocument(
            doc_id=f"DOC-{self._counter:04d}",
            doc_type=doc_type,
            title=title,
            system_name=system_name,
            content=content or {},
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self._documents[doc.doc_id] = doc
        return doc

    def generate_technical_doc(self, system_name: str, system_info: dict[str, Any]) -> ComplianceDocument:
        """Generiert technische Dokumentation nach Art. 11."""
        content = {
            "1_system_description": {
                "name": system_name,
                "intended_purpose": system_info.get("purpose", ""),
                "provider": system_info.get("provider", ""),
                "version": system_info.get("version", ""),
            },
            "2_design_specifications": {
                "architecture": system_info.get("architecture", "Multi-Agent-System"),
                "models_used": system_info.get("models", []),
                "input_types": system_info.get("inputs", []),
                "output_types": system_info.get("outputs", []),
            },
            "3_development_process": {
                "methodology": "Agile + Security-by-Design",
                "testing": system_info.get("test_count", 0),
                "security_measures": system_info.get("security", []),
            },
            "4_monitoring": {
                "metrics": ["MTTD", "MTTR", "Resolution-Rate", "Posture-Score"],
                "alerting": "Webhook + Dashboard",
                "human_oversight": True,
            },
            "5_risk_management": {
                "risk_level": system_info.get("risk_level", ""),
                "mitigations": system_info.get("mitigations", []),
            },
            "6_data_governance": {
                "data_sources": system_info.get("data_sources", []),
                "pii_handling": "DSGVO-konform, Anonymisierung",
                "retention": system_info.get("retention_days", 90),
            },
        }
        return self.create(DocumentType.TECHNICAL_DOC, f"Technische Dokumentation: {system_name}", system_name, content)

    def generate_incident_report(
        self, system_name: str, incident: dict[str, Any]
    ) -> ComplianceDocument:
        """Generiert einen Incident-Report nach Art. 62."""
        content = {
            "incident_id": incident.get("id", ""),
            "severity": incident.get("severity", ""),
            "description": incident.get("description", ""),
            "affected_persons": incident.get("affected", 0),
            "root_cause": incident.get("root_cause", ""),
            "corrective_actions": incident.get("actions", []),
            "timeline": incident.get("timeline", {}),
            "notification_to_authority": incident.get("notified", False),
        }
        return self.create(DocumentType.INCIDENT_REPORT, f"Incident-Report: {incident.get('id', '')}", system_name, content)

    def get(self, doc_id: str) -> ComplianceDocument | None:
        return self._documents.get(doc_id)

    def by_type(self, doc_type: DocumentType) -> list[ComplianceDocument]:
        return [d for d in self._documents.values() if d.doc_type == doc_type]

    def all_documents(self) -> list[ComplianceDocument]:
        return list(self._documents.values())

    @property
    def doc_count(self) -> int:
        return len(self._documents)

    def stats(self) -> dict[str, Any]:
        docs = list(self._documents.values())
        return {
            "total_documents": len(docs),
            "by_type": {
                t.value: sum(1 for d in docs if d.doc_type == t)
                for t in DocumentType
                if any(d.doc_type == t for d in docs)
            },
            "by_status": {
                s: sum(1 for d in docs if d.status == s)
                for s in ("draft", "review", "approved", "archived")
                if any(d.status == s for d in docs)
            },
        }


# ============================================================================
# Transparency Register (Art. 52)
# ============================================================================


@dataclass
class TransparencyEntry:
    """Eintrag im Transparenzregister."""

    entry_id: str
    system_name: str
    purpose: str
    is_ai_disclosed: bool = True
    disclosure_text: str = ""
    target_audience: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "system": self.system_name,
            "purpose": self.purpose,
            "ai_disclosed": self.is_ai_disclosed,
            "audience": self.target_audience,
        }


class TransparencyRegister:
    """Verwaltet Transparenzpflichten nach Art. 52."""

    def __init__(self) -> None:
        self._entries: dict[str, TransparencyEntry] = {}
        self._counter = 0

    def register(
        self,
        system_name: str,
        purpose: str,
        *,
        disclosure_text: str = "",
        target_audience: str = "Endnutzer",
    ) -> TransparencyEntry:
        self._counter += 1
        entry = TransparencyEntry(
            entry_id=f"TRANS-{self._counter:04d}",
            system_name=system_name,
            purpose=purpose,
            disclosure_text=disclosure_text or (
                f"Dieses System ({system_name}) nutzt künstliche Intelligenz. "
                f"Zweck: {purpose}. Sie haben das Recht, dies zu erfahren (Art. 52 EU-AI-Act)."
            ),
            target_audience=target_audience,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self._entries[entry.entry_id] = entry
        return entry

    def all_entries(self) -> list[TransparencyEntry]:
        return list(self._entries.values())

    @property
    def entry_count(self) -> int:
        return len(self._entries)


# ============================================================================
# Training Material (Punkt 6: Sicherheitsteam-Schulung)
# ============================================================================


class TrainingTopic(Enum):
    PROMPT_INJECTION = "prompt_injection"
    TOOL_MISUSE = "tool_misuse"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    MEMORY_POISONING = "memory_poisoning"
    MODEL_INVERSION = "model_inversion"
    SOCIAL_ENGINEERING = "social_engineering"
    SUPPLY_CHAIN = "supply_chain_attack"
    COMPLIANCE = "regulatory_compliance"


@dataclass
class TrainingModule:
    """Ein Schulungsmodul für das Sicherheitsteam."""

    module_id: str
    topic: TrainingTopic
    title: str
    description: str
    difficulty: str = "intermediate"   # beginner, intermediate, advanced
    duration_minutes: int = 60
    content_sections: list[str] = field(default_factory=list)
    exercises: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_id": self.module_id,
            "topic": self.topic.value,
            "title": self.title,
            "difficulty": self.difficulty,
            "duration_min": self.duration_minutes,
            "sections": len(self.content_sections),
            "exercises": len(self.exercises),
        }


class TrainingCatalog:
    """Katalog von Schulungsmodulen für Agent-Security."""

    BUILT_IN_MODULES = [
        TrainingModule(
            "TRN-001", TrainingTopic.PROMPT_INJECTION,
            "Prompt-Injection verstehen & abwehren",
            "Wie Angreifer Agent-Prompts manipulieren und wie man sich schützt.",
            "intermediate", 90,
            [
                "Was ist Prompt-Injection?",
                "Direkte vs. indirekte Injection",
                "Jarvis-Schutzmaßnahmen: InputSanitizer, GateKeeper",
                "Red-Team-Übungen mit dem PromptFuzzer",
                "Incident-Response bei erkannter Injection",
            ],
            [
                "Übung: Versuche 10 bekannte Injection-Techniken gegen Jarvis",
                "Übung: Konfiguriere den InputSanitizer für einen neuen Angriffsvektor",
            ],
            ["OWASP LLM Top 10", "Jarvis Architektur-Bibel §5.1"],
        ),
        TrainingModule(
            "TRN-002", TrainingTopic.TOOL_MISUSE,
            "Tool-Missbrauch erkennen & verhindern",
            "Wie Agenten Tools auf unbeabsichtigte Weise nutzen können.",
            "advanced", 60,
            [
                "Gefährliche Tool-Kombinationen",
                "Exfiltration über erlaubte Kanäle",
                "ResourceQuota & AgentPermissions",
                "Sandbox-Konfiguration",
            ],
            ["Übung: Identifiziere 5 potenzielle Tool-Missbrauchsszenarien"],
            ["Jarvis Architektur-Bibel §7.2"],
        ),
        TrainingModule(
            "TRN-003", TrainingTopic.MEMORY_POISONING,
            "Memory-Poisoning: Angriffe auf den Wissensspeicher",
            "Wie Angreifer Memory-Stores manipulieren und Gegenmaßnahmen.",
            "advanced", 75,
            [
                "Angriffsvektoren auf Memory-Systeme",
                "Plausibilitäts-Checks und Drift-Detection",
                "Memory-Hygiene-Modul",
                "Versions-Kontrolle für Wissensdaten",
            ],
            ["Übung: Simuliere einen Memory-Poisoning-Angriff und erkenne ihn"],
            ["Jarvis Memory-Hygiene-Modul", "Architektur-Bibel §8.3"],
        ),
        TrainingModule(
            "TRN-004", TrainingTopic.COMPLIANCE,
            "EU-AI-Act: Pflichten für KI-Agenten",
            "Regulatorische Anforderungen und Jarvis-Compliance.",
            "beginner", 120,
            [
                "Überblick EU-AI-Act",
                "Risikoklassifizierung",
                "Pflichten für High-Risk-Systeme",
                "Technische Dokumentation (Art. 11)",
                "Meldepflichten (Art. 62)",
                "Transparenz (Art. 52)",
                "Jarvis Compliance-Module",
            ],
            ["Übung: Erstelle eine Risikobewertung für ein Versicherungs-Beratungssystem"],
            ["EU-AI-Act Volltext", "Jarvis ComplianceDocManager"],
        ),
    ]

    def __init__(self, load_defaults: bool = True) -> None:
        self._modules: dict[str, TrainingModule] = {}
        if load_defaults:
            for m in self.BUILT_IN_MODULES:
                self._modules[m.module_id] = m

    def add(self, module: TrainingModule) -> None:
        self._modules[module.module_id] = module

    def get(self, module_id: str) -> TrainingModule | None:
        return self._modules.get(module_id)

    def by_topic(self, topic: TrainingTopic) -> list[TrainingModule]:
        return [m for m in self._modules.values() if m.topic == topic]

    def by_difficulty(self, difficulty: str) -> list[TrainingModule]:
        return [m for m in self._modules.values() if m.difficulty == difficulty]

    def all_modules(self) -> list[TrainingModule]:
        return list(self._modules.values())

    @property
    def module_count(self) -> int:
        return len(self._modules)

    def stats(self) -> dict[str, Any]:
        modules = list(self._modules.values())
        return {
            "total_modules": len(modules),
            "total_hours": round(sum(m.duration_minutes for m in modules) / 60, 1),
            "by_difficulty": {
                d: sum(1 for m in modules if m.difficulty == d)
                for d in ("beginner", "intermediate", "advanced")
                if any(m.difficulty == d for m in modules)
            },
        }


# ============================================================================
# EU AI Act Governor (Hauptklasse)
# ============================================================================


class EUAIActGovernor:
    """Hauptklasse: Orchestriert EU-AI-Act Compliance."""

    def __init__(self) -> None:
        self._classifier = RiskClassifier()
        self._documents = ComplianceDocManager()
        self._transparency = TransparencyRegister()
        self._training = TrainingCatalog()

    @property
    def classifier(self) -> RiskClassifier:
        return self._classifier

    @property
    def documents(self) -> ComplianceDocManager:
        return self._documents

    @property
    def transparency(self) -> TransparencyRegister:
        return self._transparency

    @property
    def training(self) -> TrainingCatalog:
        return self._training

    def classify_jarvis(self) -> RiskAssessment:
        """Klassifiziert das Jarvis-System selbst."""
        return self._classifier.classify(
            "Jarvis AI Agent",
            SystemCategory.INSURANCE_ADVISORY,
            description="KI-Agent-System für Versicherungsberatung und -verwaltung",
            assessed_by="system",
        )

    def compliance_status(self) -> dict[str, Any]:
        """Gesamtstatus der EU-AI-Act Compliance."""
        assessments = self._classifier.all_assessments()
        high_risk = [a for a in assessments if a.risk_level == RiskLevel.HIGH]
        docs = self._documents.all_documents()
        approved_docs = [d for d in docs if d.status == "approved"]

        return {
            "total_assessments": len(assessments),
            "high_risk_systems": len(high_risk),
            "total_documents": len(docs),
            "approved_documents": len(approved_docs),
            "transparency_entries": self._transparency.entry_count,
            "training_modules": self._training.module_count,
            "compliance_score": self._calculate_score(high_risk, docs, approved_docs),
        }

    def _calculate_score(
        self,
        high_risk: list[RiskAssessment],
        docs: list[ComplianceDocument],
        approved: list[ComplianceDocument],
    ) -> float:
        """Berechnet einen Compliance-Score (0-100)."""
        if not high_risk:
            return 100.0
        # Jedes High-Risk-System braucht Tech-Doc + Risk-Assessment
        needed = len(high_risk) * 2
        have = sum(1 for d in docs if d.doc_type in (DocumentType.TECHNICAL_DOC, DocumentType.RISK_ASSESSMENT))
        doc_ratio = min(have / needed * 100, 100) if needed > 0 else 100
        # Approved-Rate
        approved_ratio = (len(approved) / len(docs) * 100) if docs else 0
        return round((doc_ratio * 0.6 + approved_ratio * 0.4), 1)

    def stats(self) -> dict[str, Any]:
        return {
            "classifier": self._classifier.stats(),
            "documents": self._documents.stats(),
            "transparency": self._transparency.entry_count,
            "training": self._training.stats(),
            "compliance": self.compliance_status(),
        }
