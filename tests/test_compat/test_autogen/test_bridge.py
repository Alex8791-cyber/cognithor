"""_bridge — translates AutoGen-shaped calls into cognithor.crew calls."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognithor.compat.autogen import AssistantAgent
from cognithor.compat.autogen._bridge import TaskResult, run_single_task


@pytest.mark.asyncio
async def test_run_single_task_returns_task_result_with_messages() -> None:
    fake_output = MagicMock()
    fake_output.raw = "Hello back."
    fake_output.tasks_outputs = []
    fake_output.token_usage = {"total_tokens": 12}

    with patch("cognithor.compat.autogen._bridge.Crew") as crew_cls:
        crew = MagicMock()
        crew.kickoff_async = AsyncMock(return_value=fake_output)
        crew_cls.return_value = crew

        agent = AssistantAgent(name="bot", model_client=MagicMock())
        result = await run_single_task(agent, "Say hi.")

        assert isinstance(result, TaskResult)
        assert result.messages
        last = result.messages[-1]
        assert last.content == "Hello back."
        assert last.source == "bot"


@pytest.mark.asyncio
async def test_task_result_messages_have_autogen_event_shape() -> None:
    """Each message must carry source / models_usage / metadata / content / type."""
    fake_output = MagicMock()
    fake_output.raw = "OK"
    fake_output.tasks_outputs = []
    fake_output.token_usage = {"total_tokens": 1}

    with patch("cognithor.compat.autogen._bridge.Crew") as crew_cls:
        crew = MagicMock()
        crew.kickoff_async = AsyncMock(return_value=fake_output)
        crew_cls.return_value = crew

        agent = AssistantAgent(name="bot", model_client=MagicMock())
        result = await run_single_task(agent, "test")

        msg = result.messages[-1]
        for attr in ("source", "models_usage", "metadata", "content", "type"):
            assert hasattr(msg, attr), f"event-shape attr missing: {attr}"


@pytest.mark.asyncio
async def test_run_single_task_passes_system_message_into_backstory() -> None:
    fake_output = MagicMock()
    fake_output.raw = "OK"
    fake_output.tasks_outputs = []
    fake_output.token_usage = {}

    captured: dict = {}

    def _capture(agents, tasks, **kwargs):
        captured["agent_backstory"] = agents[0].backstory
        crew = MagicMock()
        crew.kickoff_async = AsyncMock(return_value=fake_output)
        return crew

    with patch("cognithor.compat.autogen._bridge.Crew", side_effect=_capture):
        agent = AssistantAgent(
            name="bot",
            model_client=MagicMock(),
            system_message="You are a careful assistant.",
        )
        await run_single_task(agent, "x")
        assert "careful assistant" in captured["agent_backstory"]
