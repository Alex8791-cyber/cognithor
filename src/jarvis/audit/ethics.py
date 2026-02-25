"""Jarvis · Ethik- und Wirtschaftsgovernance.

Budget-Limits, Bias-Checks und Fairness-Audits:

  - BudgetManager:     Budget-Limits pro Agent, Team, Gesamt
  - CostTracker:       Echtzeit-Kostenerfassung pro API-Call
  - BiasDetector:      Erkennt systematische Verzerrungen in Agent-Outputs
  - FairnessAuditor:   Auditiert Entscheidungen auf Fairness-Kriterien
  - EthicsPolicy:      Konfigurierbare Ethik-Regeln
  - EconomicGovernor:  Hauptklasse, orchestriert alles

Architektur-Bibel: §13.1 (Governance), §15.1 (Wirtschaftlichkeit)
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ============================================================================
# Budget Management
# ============================================================================


@dataclass
class BudgetLimit:
    """Budget-Limit für eine Entität (Agent, Team, System)."""

    entity_id: str
    entity_type: str  # "agent", "team", "system"
    daily_limit: float = 50.0       # EUR pro Tag
    monthly_limit: float = 1000.0   # EUR pro Monat
    per_request_limit: float = 5.0  # EUR pro Einzelanfrage
    spent_today: float = 0.0
    spent_this_month: float = 0.0
    last_reset_day: str = ""
    last_reset_month: str = ""

    @property
    def daily_remaining(self) -> float:
        return max(0, self.daily_limit - self.spent_today)

    @property
    def monthly_remaining(self) -> float:
        return max(0, self.monthly_limit - self.spent_this_month)

    @property
    def daily_utilization(self) -> float:
        return (self.spent_today / self.daily_limit * 100) if self.daily_limit > 0 else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "daily_limit": self.daily_limit,
            "monthly_limit": self.monthly_limit,
            "per_request_limit": self.per_request_limit,
            "spent_today": round(self.spent_today, 4),
            "spent_this_month": round(self.spent_this_month, 4),
            "daily_remaining": round(self.daily_remaining, 4),
            "monthly_remaining": round(self.monthly_remaining, 4),
            "daily_utilization": round(self.daily_utilization, 1),
        }


class BudgetManager:
    """Verwaltet Budget-Limits und prüft Ausgaben."""

    def __init__(self) -> None:
        self._limits: dict[str, BudgetLimit] = {}

    def set_limit(
        self,
        entity_id: str,
        entity_type: str = "agent",
        *,
        daily: float = 50.0,
        monthly: float = 1000.0,
        per_request: float = 5.0,
    ) -> BudgetLimit:
        limit = BudgetLimit(
            entity_id=entity_id,
            entity_type=entity_type,
            daily_limit=daily,
            monthly_limit=monthly,
            per_request_limit=per_request,
        )
        self._limits[entity_id] = limit
        return limit

    def get_limit(self, entity_id: str) -> BudgetLimit | None:
        return self._limits.get(entity_id)

    def can_spend(self, entity_id: str, amount: float) -> dict[str, Any]:
        """Prüft ob eine Ausgabe erlaubt ist."""
        limit = self._limits.get(entity_id)
        if not limit:
            return {"allowed": True, "reason": "Kein Limit konfiguriert"}

        if amount > limit.per_request_limit:
            return {"allowed": False, "reason": f"Einzelanfrage {amount:.2f}€ > Limit {limit.per_request_limit:.2f}€"}
        if limit.spent_today + amount > limit.daily_limit:
            return {"allowed": False, "reason": f"Tagesbudget erschöpft ({limit.daily_remaining:.2f}€ übrig)"}
        if limit.spent_this_month + amount > limit.monthly_limit:
            return {"allowed": False, "reason": f"Monatsbudget erschöpft ({limit.monthly_remaining:.2f}€ übrig)"}

        return {"allowed": True, "reason": "OK"}

    def record_spend(self, entity_id: str, amount: float) -> bool:
        """Bucht eine Ausgabe."""
        limit = self._limits.get(entity_id)
        if not limit:
            return True  # Kein Limit = immer erlaubt
        limit.spent_today += amount
        limit.spent_this_month += amount
        return True

    def reset_daily(self, entity_id: str) -> None:
        limit = self._limits.get(entity_id)
        if limit:
            limit.spent_today = 0.0
            limit.last_reset_day = time.strftime("%Y-%m-%d", time.gmtime())

    def reset_monthly(self, entity_id: str) -> None:
        limit = self._limits.get(entity_id)
        if limit:
            limit.spent_this_month = 0.0
            limit.last_reset_month = time.strftime("%Y-%m", time.gmtime())

    @property
    def entity_count(self) -> int:
        return len(self._limits)

    def over_budget(self) -> list[BudgetLimit]:
        return [l for l in self._limits.values() if l.daily_utilization >= 90]

    def stats(self) -> dict[str, Any]:
        limits = list(self._limits.values())
        return {
            "total_entities": len(limits),
            "total_spent_today": round(sum(l.spent_today for l in limits), 4),
            "total_spent_month": round(sum(l.spent_this_month for l in limits), 4),
            "over_budget_count": len(self.over_budget()),
        }


# ============================================================================
# Cost Tracker
# ============================================================================


@dataclass
class CostEntry:
    """Einzelner Kosteneintrag."""

    entry_id: str
    agent_id: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_eur: float = 0.0
    timestamp: str = ""
    task_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "agent_id": self.agent_id,
            "model": self.model,
            "tokens": self.input_tokens + self.output_tokens,
            "cost_eur": round(self.cost_eur, 6),
            "task_type": self.task_type,
        }


# Standard-Preise pro 1K Tokens (EUR, gerundet)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 0.0023, "output": 0.0092},
    "gpt-4o-mini": {"input": 0.00014, "output": 0.00055},
    "claude-3.5-sonnet": {"input": 0.0028, "output": 0.0138},
    "claude-3-haiku": {"input": 0.00023, "output": 0.00115},
    "llama-3.1-70b": {"input": 0.0, "output": 0.0},  # Lokal = kostenlos
    "mistral-7b": {"input": 0.0, "output": 0.0},
}


class CostTracker:
    """Echtzeit-Kostenerfassung pro API-Call."""

    def __init__(self, budget_manager: BudgetManager | None = None) -> None:
        self._budget = budget_manager or BudgetManager()
        self._entries: list[CostEntry] = []
        self._counter = 0

    @property
    def budget_manager(self) -> BudgetManager:
        return self._budget

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Berechnet die Kosten für einen API-Call."""
        pricing = MODEL_PRICING.get(model, {"input": 0.003, "output": 0.015})
        return (input_tokens / 1000 * pricing["input"]) + (output_tokens / 1000 * pricing["output"])

    def track(
        self,
        agent_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        task_type: str = "",
    ) -> CostEntry:
        """Erfasst einen API-Call und bucht die Kosten."""
        self._counter += 1
        cost = self.calculate_cost(model, input_tokens, output_tokens)

        entry = CostEntry(
            entry_id=f"COST-{self._counter:06d}",
            agent_id=agent_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_eur=cost,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            task_type=task_type,
        )
        self._entries.append(entry)
        self._budget.record_spend(agent_id, cost)
        return entry

    def total_cost(self) -> float:
        return sum(e.cost_eur for e in self._entries)

    def cost_by_agent(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for e in self._entries:
            result[e.agent_id] = result.get(e.agent_id, 0) + e.cost_eur
        return {k: round(v, 4) for k, v in result.items()}

    def cost_by_model(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for e in self._entries:
            result[e.model] = result.get(e.model, 0) + e.cost_eur
        return {k: round(v, 4) for k, v in result.items()}

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def stats(self) -> dict[str, Any]:
        return {
            "total_entries": len(self._entries),
            "total_cost_eur": round(self.total_cost(), 4),
            "total_tokens": sum(e.input_tokens + e.output_tokens for e in self._entries),
            "cost_by_agent": self.cost_by_agent(),
            "cost_by_model": self.cost_by_model(),
        }


# ============================================================================
# Bias Detector
# ============================================================================


class BiasCategory(Enum):
    GENDER = "gender"
    AGE = "age"
    ETHNICITY = "ethnicity"
    LANGUAGE = "language"
    SOCIOECONOMIC = "socioeconomic"
    DISABILITY = "disability"
    RELIGION = "religion"


@dataclass
class BiasReport:
    """Ergebnis einer Bias-Prüfung."""

    report_id: str
    category: BiasCategory
    severity: str  # low, medium, high
    description: str
    evidence: str = ""
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "category": self.category.value,
            "severity": self.severity,
            "description": self.description,
            "recommendation": self.recommendation,
        }


class BiasDetector:
    """Erkennt systematische Verzerrungen in Agent-Outputs.

    Prüft:
      - Stereotypen in generierten Texten
      - Ungleiche Behandlung verschiedener Gruppen
      - Sprachliche Verzerrungen
    """

    # Einfache Keyword-basierte Bias-Indikatoren (Produktionsversion → ML-Modell)
    GENDER_INDICATORS = {
        "positive_male": ["er ist kompetent", "er ist der experte", "der chef"],
        "positive_female": ["sie ist kompetent", "sie ist die expertin", "die chefin"],
        "negative_female": ["sie ist emotional", "sie ist hysterisch", "typisch frau"],
        "negative_male": ["er ist gefühllos", "typisch mann"],
    }

    AGEISM_INDICATORS = [
        "zu alt für", "digital native", "die jungen verstehen",
        "in seinem alter", "alte generation",
    ]

    def __init__(self) -> None:
        self._reports: list[BiasReport] = []
        self._counter = 0

    def check(self, text: str, context: str = "") -> list[BiasReport]:
        """Prüft einen Text auf Bias-Indikatoren."""
        text_lower = text.lower()
        findings: list[BiasReport] = []

        # Gender-Bias
        neg_female = sum(1 for p in self.GENDER_INDICATORS["negative_female"] if p in text_lower)
        neg_male = sum(1 for p in self.GENDER_INDICATORS["negative_male"] if p in text_lower)
        if neg_female > 0:
            self._counter += 1
            r = BiasReport(
                report_id=f"BIAS-{self._counter:04d}",
                category=BiasCategory.GENDER,
                severity="medium" if neg_female == 1 else "high",
                description=f"Gender-Stereotyp erkannt ({neg_female} Indikatoren)",
                evidence=text[:200],
                recommendation="Gender-neutrale Formulierung verwenden.",
            )
            findings.append(r)
            self._reports.append(r)

        if neg_male > 0:
            self._counter += 1
            r = BiasReport(
                report_id=f"BIAS-{self._counter:04d}",
                category=BiasCategory.GENDER,
                severity="low",
                description=f"Gender-Stereotyp (männl.) erkannt",
                evidence=text[:200],
                recommendation="Stereotyp-freie Formulierung verwenden.",
            )
            findings.append(r)
            self._reports.append(r)

        # Age-Bias
        age_hits = sum(1 for p in self.AGEISM_INDICATORS if p in text_lower)
        if age_hits > 0:
            self._counter += 1
            r = BiasReport(
                report_id=f"BIAS-{self._counter:04d}",
                category=BiasCategory.AGE,
                severity="medium",
                description=f"Alters-Diskriminierung erkannt ({age_hits} Indikatoren)",
                evidence=text[:200],
                recommendation="Alters-neutrale Formulierung verwenden.",
            )
            findings.append(r)
            self._reports.append(r)

        return findings

    def all_reports(self) -> list[BiasReport]:
        return list(self._reports)

    @property
    def report_count(self) -> int:
        return len(self._reports)

    def stats(self) -> dict[str, Any]:
        reports = self._reports
        return {
            "total_checks": self._counter,
            "total_findings": len(reports),
            "by_category": {
                cat.value: sum(1 for r in reports if r.category == cat)
                for cat in BiasCategory
                if any(r.category == cat for r in reports)
            },
            "by_severity": {
                sev: sum(1 for r in reports if r.severity == sev)
                for sev in ("low", "medium", "high")
                if any(r.severity == sev for r in reports)
            },
        }


# ============================================================================
# Fairness Auditor
# ============================================================================


class FairnessMetric(Enum):
    EQUAL_TREATMENT = "equal_treatment"
    PROPORTIONAL_ALLOCATION = "proportional_allocation"
    RESPONSE_QUALITY_PARITY = "response_quality_parity"
    LATENCY_PARITY = "latency_parity"
    ERROR_RATE_PARITY = "error_rate_parity"


@dataclass
class FairnessAuditResult:
    """Ergebnis eines Fairness-Audits."""

    audit_id: str
    metric: FairnessMetric
    score: float  # 0-100, 100 = perfekt fair
    passed: bool
    threshold: float
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "metric": self.metric.value,
            "score": round(self.score, 1),
            "passed": self.passed,
            "threshold": self.threshold,
            "details": self.details,
        }


class FairnessAuditor:
    """Auditiert Agent-Entscheidungen auf Fairness-Kriterien."""

    DEFAULT_THRESHOLD = 80.0  # Minimum-Score für "Fair"

    def __init__(self, threshold: float = 80.0) -> None:
        self._threshold = threshold
        self._results: list[FairnessAuditResult] = []
        self._counter = 0

    def audit_response_times(self, times_by_group: dict[str, list[float]]) -> FairnessAuditResult:
        """Prüft ob Antwortzeiten zwischen Gruppen fair verteilt sind."""
        self._counter += 1
        if not times_by_group:
            return self._create_result(FairnessMetric.LATENCY_PARITY, 100.0, {})

        averages = {g: sum(t) / len(t) for g, t in times_by_group.items() if t}
        if len(averages) < 2:
            return self._create_result(FairnessMetric.LATENCY_PARITY, 100.0, averages)

        max_avg = max(averages.values())
        min_avg = min(averages.values())
        ratio = (min_avg / max_avg * 100) if max_avg > 0 else 100

        return self._create_result(FairnessMetric.LATENCY_PARITY, ratio, averages)

    def audit_error_rates(self, errors_by_group: dict[str, tuple[int, int]]) -> FairnessAuditResult:
        """Prüft ob Fehlerraten zwischen Gruppen fair verteilt sind.

        Args:
            errors_by_group: {group: (errors, total)}
        """
        self._counter += 1
        if not errors_by_group:
            return self._create_result(FairnessMetric.ERROR_RATE_PARITY, 100.0, {})

        rates = {
            g: (errs / total * 100) if total > 0 else 0
            for g, (errs, total) in errors_by_group.items()
        }

        if len(rates) < 2:
            return self._create_result(FairnessMetric.ERROR_RATE_PARITY, 100.0, rates)

        max_rate = max(rates.values())
        min_rate = min(rates.values())
        diff = max_rate - min_rate
        score = max(0, 100 - diff * 10)  # 10% Diff = 0 Score

        return self._create_result(FairnessMetric.ERROR_RATE_PARITY, score, rates)

    def audit_allocation(self, allocated_by_group: dict[str, float]) -> FairnessAuditResult:
        """Prüft ob Ressourcen-Allokation proportional fair ist."""
        self._counter += 1
        if not allocated_by_group:
            return self._create_result(FairnessMetric.PROPORTIONAL_ALLOCATION, 100.0, {})

        values = list(allocated_by_group.values())
        if len(values) < 2:
            return self._create_result(FairnessMetric.PROPORTIONAL_ALLOCATION, 100.0, allocated_by_group)

        avg = sum(values) / len(values)
        if avg == 0:
            return self._create_result(FairnessMetric.PROPORTIONAL_ALLOCATION, 100.0, allocated_by_group)

        max_deviation = max(abs(v - avg) / avg * 100 for v in values)
        score = max(0, 100 - max_deviation)

        return self._create_result(FairnessMetric.PROPORTIONAL_ALLOCATION, score, allocated_by_group)

    def _create_result(
        self, metric: FairnessMetric, score: float, details: dict[str, Any]
    ) -> FairnessAuditResult:
        result = FairnessAuditResult(
            audit_id=f"FAIR-{self._counter:04d}",
            metric=metric,
            score=score,
            passed=score >= self._threshold,
            threshold=self._threshold,
            details=details,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self._results.append(result)
        return result

    def all_results(self) -> list[FairnessAuditResult]:
        return list(self._results)

    @property
    def result_count(self) -> int:
        return len(self._results)

    def pass_rate(self) -> float:
        if not self._results:
            return 100.0
        return sum(1 for r in self._results if r.passed) / len(self._results) * 100

    def stats(self) -> dict[str, Any]:
        results = self._results
        return {
            "total_audits": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "pass_rate": round(self.pass_rate(), 1),
            "avg_score": round(
                sum(r.score for r in results) / len(results), 1
            ) if results else 0,
        }


# ============================================================================
# Ethics Policy
# ============================================================================


class EthicsViolationType(Enum):
    BIAS = "bias"
    FAIRNESS = "fairness"
    BUDGET_OVERRUN = "budget_overrun"
    PRIVACY = "privacy"
    TRANSPARENCY = "transparency"
    ACCOUNTABILITY = "accountability"


@dataclass
class EthicsViolation:
    """Eine Ethik-Verletzung."""

    violation_id: str
    violation_type: EthicsViolationType
    severity: str
    description: str
    agent_id: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "violation_id": self.violation_id,
            "type": self.violation_type.value,
            "severity": self.severity,
            "description": self.description,
            "agent_id": self.agent_id,
        }


class EthicsPolicy:
    """Konfigurierbare Ethik-Regeln."""

    def __init__(
        self,
        *,
        max_bias_findings_per_day: int = 5,
        min_fairness_score: float = 80.0,
        require_transparency: bool = True,
        require_human_oversight_for_critical: bool = True,
    ) -> None:
        self._max_bias = max_bias_findings_per_day
        self._min_fairness = min_fairness_score
        self._require_transparency = require_transparency
        self._require_human_oversight = require_human_oversight_for_critical
        self._violations: list[EthicsViolation] = []
        self._counter = 0

    def check_bias(self, bias_findings: int, agent_id: str = "") -> EthicsViolation | None:
        if bias_findings > self._max_bias:
            self._counter += 1
            v = EthicsViolation(
                violation_id=f"ETH-{self._counter:04d}",
                violation_type=EthicsViolationType.BIAS,
                severity="high" if bias_findings > self._max_bias * 2 else "medium",
                description=f"{bias_findings} Bias-Findings überschreiten Limit von {self._max_bias}/Tag",
                agent_id=agent_id,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            self._violations.append(v)
            return v
        return None

    def check_fairness(self, score: float, agent_id: str = "") -> EthicsViolation | None:
        if score < self._min_fairness:
            self._counter += 1
            v = EthicsViolation(
                violation_id=f"ETH-{self._counter:04d}",
                violation_type=EthicsViolationType.FAIRNESS,
                severity="high" if score < 50 else "medium",
                description=f"Fairness-Score {score:.1f} unter Minimum {self._min_fairness}",
                agent_id=agent_id,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            self._violations.append(v)
            return v
        return None

    def all_violations(self) -> list[EthicsViolation]:
        return list(self._violations)

    @property
    def violation_count(self) -> int:
        return len(self._violations)

    def stats(self) -> dict[str, Any]:
        violations = self._violations
        return {
            "total_violations": len(violations),
            "by_type": {
                t.value: sum(1 for v in violations if v.violation_type == t)
                for t in EthicsViolationType
                if any(v.violation_type == t for v in violations)
            },
            "config": {
                "max_bias_per_day": self._max_bias,
                "min_fairness": self._min_fairness,
                "transparency": self._require_transparency,
                "human_oversight": self._require_human_oversight,
            },
        }


# ============================================================================
# Economic Governor (Hauptklasse)
# ============================================================================


class EconomicGovernor:
    """Hauptklasse: Orchestriert Ethik- und Wirtschaftsgovernance."""

    def __init__(self) -> None:
        self._budget = BudgetManager()
        self._costs = CostTracker(self._budget)
        self._bias = BiasDetector()
        self._fairness = FairnessAuditor()
        self._ethics = EthicsPolicy()

    @property
    def budget(self) -> BudgetManager:
        return self._budget

    @property
    def costs(self) -> CostTracker:
        return self._costs

    @property
    def bias(self) -> BiasDetector:
        return self._bias

    @property
    def fairness(self) -> FairnessAuditor:
        return self._fairness

    @property
    def ethics(self) -> EthicsPolicy:
        return self._ethics

    def pre_flight_check(self, agent_id: str, estimated_cost: float) -> dict[str, Any]:
        """Vorab-Prüfung vor einer Agent-Aktion."""
        budget_check = self._budget.can_spend(agent_id, estimated_cost)
        bias_count = sum(1 for r in self._bias.all_reports() if r.severity in ("medium", "high"))
        ethics_check = self._ethics.check_bias(bias_count, agent_id)

        return {
            "approved": budget_check["allowed"] and ethics_check is None,
            "budget": budget_check,
            "ethics_violation": ethics_check.to_dict() if ethics_check else None,
        }

    def stats(self) -> dict[str, Any]:
        return {
            "budget": self._budget.stats(),
            "costs": self._costs.stats(),
            "bias": self._bias.stats(),
            "fairness": self._fairness.stats(),
            "ethics": self._ethics.stats(),
        }
