"""Critical safety test: pre-advisory passes, legal-advice blocked.

Spec §8.5 acceptance: Gatekeeper-block test must be green for BOTH the
positive (pre-advisory question allowed) AND negative (legal-advice
blocked WITHOUT exception, with a clear reason).
"""

from __future__ import annotations

import pytest
from insurance_agent_pack.agents.compliance_gatekeeper import (
    ComplianceVerdict,
    classify_intent,
)


def test_positive_pre_advisory_question_passes() -> None:
    msg = "Welche Versicherungen gibt es für GGF?"
    verdict = classify_intent(msg)
    assert isinstance(verdict, ComplianceVerdict)
    assert verdict.allowed is True


def test_negative_legal_advice_blocked() -> None:
    msg = "Ist mein Arbeitsvertrag rechtens?"
    verdict = classify_intent(msg)
    assert verdict.allowed is False
    assert "rechtsberatung" in verdict.reason.lower()


def test_block_returns_verdict_not_exception() -> None:
    """Blocking must NOT raise; it returns a Verdict the Crew can react to."""
    msg = "Welche konkrete BU soll ich abschließen?"
    verdict = classify_intent(msg)  # Must NOT raise
    assert verdict.allowed is False


@pytest.mark.parametrize(
    "msg",
    [
        "Was ist eine PKV?",
        "Erkläre GGF Versorgung.",
        "Welche Vorsorge ist sinnvoll für Selbstständige?",
        "Was ist der Unterschied zwischen GKV und PKV?",
    ],
)
def test_pre_advisory_phrases_all_pass(msg: str) -> None:
    assert classify_intent(msg).allowed is True


@pytest.mark.parametrize(
    "msg",
    [
        "Ist mein Mietvertrag rechtmäßig?",
        "Welche konkrete Versicherung soll ich abschließen?",
        "Empfiehlst du mir Versicherer X?",
        "Entscheide für mich, welche Police ich nehmen soll.",
    ],
)
def test_unsafe_phrases_all_blocked(msg: str) -> None:
    assert classify_intent(msg).allowed is False
