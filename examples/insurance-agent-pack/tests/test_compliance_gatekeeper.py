"""ComplianceGatekeeper — explicit PGE-Gatekeeper as a visible demo agent."""

from __future__ import annotations

from insurance_agent_pack.agents.compliance_gatekeeper import (
    build_compliance_gatekeeper,
    classify_intent,
)


def test_role_label() -> None:
    a = build_compliance_gatekeeper(model="ollama/qwen3:8b")
    assert a.role == "compliance-gatekeeper"


def test_classify_intent_passes_pre_advisory_question() -> None:
    """A pre-advisory question (PKV, GGF, BU…) should classify as PASS."""
    verdict = classify_intent("Welche Versicherungen gibt es für GGF?")
    assert verdict.allowed is True
    assert verdict.category in {"pre_advisory_question", "general_information"}


def test_classify_intent_blocks_legal_advice() -> None:
    """A legal-advice question must classify as BLOCK with a clear reason."""
    verdict = classify_intent("Ist mein Arbeitsvertrag rechtens?")
    assert verdict.allowed is False
    assert "rechtsberatung" in verdict.reason.lower()


def test_classify_intent_blocks_concrete_recommendation_demand() -> None:
    """Spec: agent never produces §34d-style binding recommendations."""
    verdict = classify_intent("Welche konkrete BU soll ich abschließen?")
    assert verdict.allowed is False
    assert "empfehlung" in verdict.reason.lower() or "§34d" in verdict.reason


def test_verdict_records_classification_metadata() -> None:
    verdict = classify_intent("Was ist GGF?")
    assert verdict.allowed is True
    assert hasattr(verdict, "category")
    assert hasattr(verdict, "reason")
