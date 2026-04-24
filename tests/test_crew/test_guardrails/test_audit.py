"""Task 30 — guardrail verdicts recorded in Hashline-Guard audit chain."""

from unittest.mock import AsyncMock, MagicMock

from cognithor.core.observer import ResponseEnvelope
from cognithor.crew import Crew, CrewAgent, CrewTask
from cognithor.crew.guardrails.base import GuardrailResult


async def test_guardrail_pass_audited(monkeypatch):
    agent = CrewAgent(role="writer", goal="write")
    task = CrewTask(
        description="x",
        expected_output="y",
        agent=agent,
        guardrail=lambda o: GuardrailResult(passed=True, feedback=None),
    )

    events: list = []

    def spy(name, **fields):
        events.append((name, fields))

    mock_planner = MagicMock()
    mock_planner.formulate_response = AsyncMock(
        return_value=ResponseEnvelope(content="ok", directive=None),
    )

    crew = Crew(agents=[agent], tasks=[task], planner=mock_planner)

    monkeypatch.setattr("cognithor.crew.compiler.append_audit", spy)
    result = await crew.kickoff_async()

    guardrail_events = [e for e in events if "guardrail" in e[0]]
    assert guardrail_events
    assert any(fields.get("verdict") == "pass" for _name, fields in guardrail_events)
    # Parent correlation: every guardrail event carries the kickoff's trace_id
    for name, fields in guardrail_events:
        assert fields.get("trace_id") == result.trace_id, (
            f"Guardrail event '{name}' lost parent trace_id — "
            f"expected {result.trace_id}, got {fields.get('trace_id')}"
        )
