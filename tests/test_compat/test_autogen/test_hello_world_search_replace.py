# tests/test_compat/test_autogen/test_hello_world_search_replace.py
"""Stage-2 D6 test — AutoGen Hello-World runs through the shim with import swap.

Reference: AutoGen README hello-world (get_current_time tool).
We adapt the shape with Cognithor's mock model client; the goal is to verify
that a user can search-and-replace `from autogen_agentchat.agents` →
`from cognithor.compat.autogen` with no other changes.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognithor.compat.autogen import (
    AssistantAgent,
    OpenAIChatCompletionClient,
    TextMessage,
)


@pytest.mark.asyncio
async def test_autogen_hello_world_runs_through_shim() -> None:
    """The 30-line AutoGen hello-world example, rewritten with our import paths.

    Original AutoGen (referenced for shape — NOT copied):
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        from autogen_agentchat.agents import AssistantAgent

        async def main():
            client = OpenAIChatCompletionClient(model="gpt-4o-mini")
            agent = AssistantAgent("assistant", model_client=client)
            result = await agent.run(task="Say hello.")
            print(result.messages[-1].content)
    """
    fake_output = MagicMock()
    fake_output.raw = "Hello, world!"
    fake_output.tasks_outputs = []
    fake_output.token_usage = {"total_tokens": 5}

    with patch("cognithor.compat.autogen._bridge.Crew") as crew_cls:
        crew = MagicMock()
        crew.kickoff_async = AsyncMock(return_value=fake_output)
        crew_cls.return_value = crew

        client = OpenAIChatCompletionClient(model="ollama/qwen3:8b")
        agent = AssistantAgent("assistant", model_client=client)
        result = await agent.run(task="Say hello.")

        last = result.messages[-1]
        assert isinstance(last, TextMessage) or hasattr(last, "content")
        assert "Hello" in str(last.content)
        assert last.source == "assistant"


@pytest.mark.asyncio
async def test_message_event_shape_matches_autogen_attrs() -> None:
    """Spec §8.4 verhaltensgarantien — events expose source/models_usage/metadata/content/type."""
    fake_output = MagicMock()
    fake_output.raw = "OK"
    fake_output.tasks_outputs = []
    fake_output.token_usage = {"total_tokens": 1}

    with patch("cognithor.compat.autogen._bridge.Crew") as crew_cls:
        crew = MagicMock()
        crew.kickoff_async = AsyncMock(return_value=fake_output)
        crew_cls.return_value = crew

        client = OpenAIChatCompletionClient(model="ollama/qwen3:8b")
        agent = AssistantAgent("assistant", model_client=client)
        result = await agent.run(task="x")
        msg = result.messages[-1]

        for attr in ("source", "models_usage", "metadata", "content", "type"):
            assert hasattr(msg, attr), f"missing event-shape attr: {attr}"


@pytest.mark.asyncio
async def test_autogen_two_agent_round_robin_runs_through_shim() -> None:
    """The minimal RoundRobinGroupChat example, rewritten with our import paths."""
    from cognithor.compat.autogen import (
        MaxMessageTermination,
        RoundRobinGroupChat,
    )

    fake_output_a = MagicMock()
    fake_output_a.raw = "a-says"
    fake_output_a.tasks_outputs = []
    fake_output_a.token_usage = {}

    fake_output_b = MagicMock()
    fake_output_b.raw = "b-says"
    fake_output_b.tasks_outputs = []
    fake_output_b.token_usage = {}

    outputs = [fake_output_a, fake_output_b, fake_output_a, fake_output_b]

    with patch("cognithor.compat.autogen._bridge.Crew") as crew_cls:
        crew = MagicMock()
        crew.kickoff_async = AsyncMock(side_effect=outputs)
        crew_cls.return_value = crew

        client = OpenAIChatCompletionClient(model="ollama/qwen3:8b")
        a = AssistantAgent("a", model_client=client)
        b = AssistantAgent("b", model_client=client)

        team = RoundRobinGroupChat(
            participants=[a, b],
            termination_condition=MaxMessageTermination(2),
        )
        result = await team.run(task="kickoff")
        assert len(result.messages) == 2
        assert result.messages[0].source == "a"
        assert result.messages[1].source == "b"
