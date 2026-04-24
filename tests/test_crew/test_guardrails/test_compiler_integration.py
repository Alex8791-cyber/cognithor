"""Task 29 — guardrail execution in the compiler (retry + GuardrailFailure)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from cognithor.core.observer import ResponseEnvelope
from cognithor.crew import Crew, CrewAgent, CrewTask
from cognithor.crew.errors import GuardrailFailure
from cognithor.crew.guardrails.base import GuardrailResult


async def test_guardrail_failure_retries_then_raises():
    agent = CrewAgent(role="writer", goal="write")

    def fail_always(_out):
        return GuardrailResult(passed=False, feedback="zu kurz")

    task = CrewTask(
        description="write",
        expected_output="long text",
        agent=agent,
        guardrail=fail_always,
        max_retries=2,
    )

    call_count = {"n": 0}

    async def fake(user_message, results, working_memory):
        call_count["n"] += 1
        return ResponseEnvelope(content=f"attempt-{call_count['n']}", directive=None)

    mock_planner = MagicMock()
    mock_planner.formulate_response = AsyncMock(side_effect=fake)
    crew = Crew(agents=[agent], tasks=[task], planner=mock_planner)

    with pytest.raises(GuardrailFailure) as exc_info:
        await crew.kickoff_async()

    # Initial try + max_retries == 3 attempts total
    assert call_count["n"] == 3
    assert exc_info.value.attempts == 3
    assert "zu kurz" in exc_info.value.reason
    assert "after 3 attempt(s)" in str(exc_info.value)


async def test_guardrail_passes_after_retry():
    agent = CrewAgent(role="writer", goal="write")
    attempts_counter = {"n": 0}

    def pass_on_second(_out):
        attempts_counter["n"] += 1
        return GuardrailResult(passed=(attempts_counter["n"] >= 2), feedback="try again")

    task = CrewTask(
        description="x",
        expected_output="y",
        agent=agent,
        guardrail=pass_on_second,
        max_retries=2,
    )

    mock_planner = MagicMock()
    mock_planner.formulate_response = AsyncMock(
        return_value=ResponseEnvelope(content="text", directive=None),
    )
    crew = Crew(agents=[agent], tasks=[task], planner=mock_planner)
    result = await crew.kickoff_async()

    assert result.tasks_output[0].guardrail_verdict == "pass"
