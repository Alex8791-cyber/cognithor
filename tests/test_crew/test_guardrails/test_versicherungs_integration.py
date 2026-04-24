"""Task 31 — versicherungs-vergleich Feature-4 integration.

Spec §4.5 AC 5: ``chain(no_pii(), StringGuardrail("..."))`` must catch BOTH
PII leakage (regex-based) AND Tarif-Empfehlung violations (LLM-validated).
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from cognithor.core.observer import ResponseEnvelope
from cognithor.crew import Crew, CrewAgent, CrewTask
from cognithor.crew.errors import GuardrailFailure
from cognithor.crew.guardrails import StringGuardrail, chain, no_pii


def _mock_ollama_client(validator_verdict: dict) -> MagicMock:
    """Build an OllamaClient-shaped mock returning a JSON-wrapped verdict."""
    client = MagicMock()
    client.chat = AsyncMock(return_value={"message": {"content": json.dumps(validator_verdict)}})
    return client


async def test_versicherungs_crew_blocks_pii_output():
    agent = CrewAgent(role="analyst", goal="compare PKV tariffs", llm="ollama/qwen3:8b")
    # Validator LLM (OllamaClient stand-in) passes every check — but no_pii
    # runs first and short-circuits the chain.
    ollama = _mock_ollama_client({"passed": True, "feedback": None})

    neutral_rule = StringGuardrail(
        "Output darf keine Tarif-Empfehlung enthalten, nur neutralen Vergleich",
        llm_client=ollama,
        model="ollama/qwen3:8b",
    )
    task = CrewTask(
        description="Compare",
        expected_output="Tabular comparison",
        agent=agent,
        guardrail=chain(no_pii(), neutral_rule),
        max_retries=0,
    )

    mock_planner = MagicMock()
    mock_planner._ollama = ollama
    mock_planner.formulate_response = AsyncMock(
        return_value=ResponseEnvelope(
            content="Kontakt: sachbearbeiter@versicherer.de zur Beratung.",
            directive=None,
        )
    )

    crew = Crew(agents=[agent], tasks=[task], planner=mock_planner)

    with pytest.raises(GuardrailFailure, match="PII erkannt"):
        await crew.kickoff_async()


async def test_versicherungs_crew_blocks_tarif_recommendation():
    """The string guardrail catches outputs that make recommendations."""
    agent = CrewAgent(role="analyst", goal="compare PKV tariffs", llm="ollama/qwen3:8b")
    ollama = _mock_ollama_client(
        {
            "passed": False,
            "feedback": (
                "Output enthält eine Empfehlung ('empfehle Tarif A'); nur Vergleich erlaubt."
            ),
        }
    )

    neutral_rule = StringGuardrail(
        "Output darf keine Tarif-Empfehlung enthalten, nur neutralen Vergleich",
        llm_client=ollama,
        model="ollama/qwen3:8b",
    )
    task = CrewTask(
        description="Compare",
        expected_output="Tabular comparison",
        agent=agent,
        guardrail=chain(no_pii(), neutral_rule),
        max_retries=0,
    )

    mock_planner = MagicMock()
    mock_planner._ollama = ollama
    # Clean output (no PII) that recommends a tariff — no_pii passes,
    # string-guard fails.
    mock_planner.formulate_response = AsyncMock(
        return_value=ResponseEnvelope(
            content="Ich empfehle Tarif A fuer Ihre Situation.",
            directive=None,
        )
    )

    crew = Crew(agents=[agent], tasks=[task], planner=mock_planner)

    with pytest.raises(GuardrailFailure, match="Empfehlung"):
        await crew.kickoff_async()
