"""ComplianceGatekeeper — explicit pre-advisory compliance check.

This is a thin RULE-BASED classifier. It is INTENTIONALLY not LLM-backed
in the v0.94.0 reference pack — keeping the safety boundary inspectable
and deterministic. An LLM-augmented variant is post-v0.94.0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from cognithor.crew import CrewAgent

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "compliance_gatekeeper.md"


@dataclass(frozen=True)
class ComplianceVerdict:
    allowed: bool
    category: str
    reason: str


_LEGAL_ADVICE_PATTERNS = (
    r"\barbeitsvertrag\b.*\brechtens\b",
    r"\brechtens\b",
    r"\bjuristisch\b",
    r"\berbrecht\b",
    r"\bmietrecht\b",
    r"\bist .* rechtmäßig\b",
)

_CONCRETE_RECOMMENDATION_PATTERNS = (
    r"\bsoll(en)? ich\b.*\babschließen\b",
    r"\bwelche konkrete\b.*\b(versicherung|police|tarif|bu|pkv|bav)\b",
    r"\bempfiehlst du mir\b",
    r"\bentscheide für mich\b",
)

_PRE_ADVISORY_PATTERNS = (
    r"\b(pkv|ggf|bav|bav-|bu|berufsunfähigkeit|haftpflicht|hausrat)\b",
    r"\bwelche versicherungen gibt es\b",
    r"\bwas ist .*\b(pkv|ggf|bav|bu)\b",
)


def classify_intent(message: str) -> ComplianceVerdict:
    text = message.lower().strip()

    for pat in _LEGAL_ADVICE_PATTERNS:
        if re.search(pat, text):
            return ComplianceVerdict(
                allowed=False,
                category="legal_advice_request",
                reason="Diese Frage berührt Rechtsberatung; bitte einen Anwalt konsultieren.",
            )

    for pat in _CONCRETE_RECOMMENDATION_PATTERNS:
        if re.search(pat, text):
            return ComplianceVerdict(
                allowed=False,
                category="concrete_recommendation_demand",
                reason=(
                    "Eine konkrete Produkt-Empfehlung erfordert eine §34d-konforme "
                    "Beratung; der Pack ist Pre-Beratung, keine Beratung."
                ),
            )

    for pat in _PRE_ADVISORY_PATTERNS:
        if re.search(pat, text):
            return ComplianceVerdict(
                allowed=True,
                category="pre_advisory_question",
                reason="Allgemeinbildende Pre-Beratungsfrage.",
            )

    return ComplianceVerdict(
        allowed=True,
        category="general_information",
        reason="Keine §34d-relevanten Empfehlungs-Token erkannt.",
    )


def build_compliance_gatekeeper(*, model: str) -> CrewAgent:
    backstory = _PROMPT_PATH.read_text(encoding="utf-8")
    return CrewAgent(
        role="compliance-gatekeeper",
        goal=(
            "Block legal-advice and concrete-recommendation requests; allow pre-advisory questions."
        ),
        backstory=backstory,
        tools=[],
        llm=model,
        allow_delegation=False,
        max_iter=5,
        memory=False,
        verbose=False,
    )
