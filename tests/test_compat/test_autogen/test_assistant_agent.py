"""AssistantAgent behaviour — 1-shot run, message shape, tool-call summary."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognithor.compat.autogen import AssistantAgent
from cognithor.compat.autogen._bridge import TaskResult


@pytest.mark.asyncio
async def test_assistant_agent_run_returns_task_result() -> None:
    fake_output = MagicMock()
    fake_output.raw = "Hello!"
    fake_output.tasks_outputs = []
    fake_output.token_usage = {"total_tokens": 10}

    with patch("cognithor.compat.autogen._bridge.Crew") as crew_cls:
        crew = MagicMock()
        crew.kickoff_async = AsyncMock(return_value=fake_output)
        crew_cls.return_value = crew

        agent = AssistantAgent(
            name="test-bot",
            model_client=MagicMock(),
            description="Friendly bot",
            system_message="Be polite.",
        )
        result = await agent.run(task="Hi")

        assert isinstance(result, TaskResult)
        assert result.messages[-1].source == "test-bot"
        assert "Hello" in str(result.messages[-1].content)


@pytest.mark.asyncio
async def test_assistant_agent_metadata_default_is_empty_dict() -> None:
    agent = AssistantAgent(name="x", model_client=MagicMock())
    assert agent.metadata == {}


@pytest.mark.asyncio
async def test_assistant_agent_max_tool_iterations_default_is_one() -> None:
    agent = AssistantAgent(name="x", model_client=MagicMock())
    assert agent.max_tool_iterations == 1


@pytest.mark.asyncio
async def test_assistant_agent_run_stream_yields_events() -> None:
    fake_output = MagicMock()
    fake_output.raw = "stream-out"
    fake_output.tasks_outputs = []
    fake_output.token_usage = {}

    with patch("cognithor.compat.autogen._bridge.Crew") as crew_cls:
        crew = MagicMock()
        crew.kickoff_async = AsyncMock(return_value=fake_output)
        crew_cls.return_value = crew

        agent = AssistantAgent(name="x", model_client=MagicMock())
        events = []
        async for evt in agent.run_stream(task="hi"):
            events.append(evt)
        assert len(events) >= 1
