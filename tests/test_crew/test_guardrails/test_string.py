"""Task 23 — StringGuardrail LLM-validated tests (async)."""

from unittest.mock import AsyncMock, MagicMock

from cognithor.crew.guardrails.string_guardrail import StringGuardrail
from cognithor.crew.output import TaskOutput


async def test_string_guardrail_passes_when_llm_says_yes():
    llm = MagicMock()
    llm.chat = AsyncMock(
        return_value={"message": {"content": '{"passed": true, "feedback": null}'}}
    )
    g = StringGuardrail("Output must be one sentence", llm_client=llm, model="ollama/qwen3:8b")
    r = await g(TaskOutput(task_id="t", agent_role="w", raw="Hello."))
    assert r.passed


async def test_string_guardrail_fails_when_llm_says_no():
    llm = MagicMock()
    llm.chat = AsyncMock(
        return_value={
            "message": {"content": '{"passed": false, "feedback": "more than one sentence"}'}
        }
    )
    g = StringGuardrail("one sentence", llm_client=llm, model="ollama/qwen3:8b")
    r = await g(TaskOutput(task_id="t", agent_role="w", raw="A. B."))
    assert not r.passed
    assert "more than one sentence" in (r.feedback or "")


async def test_string_guardrail_unparseable_llm_response_fails_safe():
    llm = MagicMock()
    llm.chat = AsyncMock(return_value={"message": {"content": "not json"}})
    g = StringGuardrail("x", llm_client=llm, model="ollama/qwen3:8b")
    r = await g(TaskOutput(task_id="t", agent_role="w", raw="y"))
    assert not r.passed
    assert "parse" in (r.feedback or "").lower()


async def test_string_guardrail_llm_unavailable_fails_safe():
    llm = MagicMock()
    llm.chat = AsyncMock(side_effect=ConnectionError("ollama down"))
    g = StringGuardrail("x", llm_client=llm, model="ollama/qwen3:8b")
    r = await g(TaskOutput(task_id="t", agent_role="w", raw="y"))
    assert not r.passed  # fail-safe: production can't silently skip validation
