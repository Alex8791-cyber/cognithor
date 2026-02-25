"""Compliance phase: Compliance framework, decision log, remediation, economics, export, impact, explainability.

Attributes handled:
  _compliance_framework, _decision_log, _remediation_tracker,
  _economic_governor, _compliance_exporter, _impact_assessor,
  _explainability
"""

from __future__ import annotations

from typing import Any

from jarvis.gateway.phases import PhaseResult
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


def declare_compliance_attrs(config: Any) -> PhaseResult:
    """Return default values for compliance attributes.

    Eagerly constructs instances where possible (matching original __init__).
    """
    result: PhaseResult = {
        "compliance_framework": None,
        "decision_log": None,
        "remediation_tracker": None,
        "economic_governor": None,
        "compliance_exporter": None,
        "impact_assessor": None,
        "explainability": None,
    }

    # Phase 11: Compliance-Framework + Decision-Log
    try:
        from jarvis.audit.compliance import (
            ComplianceFramework,
            DecisionLog,
            RemediationTracker,
        )
        result["compliance_framework"] = ComplianceFramework()
        result["decision_log"] = DecisionLog()
        result["remediation_tracker"] = RemediationTracker()
    except Exception:
        log.debug("compliance_framework_init_skipped", exc_info=True)

    # Phase 13: Explainability-Engine
    try:
        from jarvis.core.explainability import ExplainabilityEngine
        result["explainability"] = ExplainabilityEngine()
    except Exception:
        log.debug("explainability_init_skipped", exc_info=True)

    # Phase 23: Economic Governor
    try:
        from jarvis.audit.ethics import EconomicGovernor
        result["economic_governor"] = EconomicGovernor()
    except Exception:
        log.debug("economic_governor_init_skipped", exc_info=True)

    # Phase 27: EU AI Act Compliance
    try:
        from jarvis.audit.ai_act_export import ComplianceExporter
        result["compliance_exporter"] = ComplianceExporter()
    except Exception:
        log.debug("compliance_exporter_init_skipped", exc_info=True)

    # Phase 32: AI Impact Assessment + Ethics Board
    try:
        from jarvis.audit.impact_assessment import ImpactAssessor
        result["impact_assessor"] = ImpactAssessor()
    except Exception:
        log.debug("impact_assessor_init_skipped", exc_info=True)

    return result


async def init_compliance(config: Any) -> PhaseResult:
    """Initialize compliance subsystems (currently all eagerly constructed in declare).

    Reserved for future async compliance initialization.
    """
    return {}
