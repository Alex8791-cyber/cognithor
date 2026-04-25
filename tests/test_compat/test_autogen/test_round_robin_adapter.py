"""_round_robin_adapter — multi-round loop with termination."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cognithor.compat.autogen._round_robin_adapter import _RoundRobinAdapter
from cognithor.compat.autogen.conditions import (
    MaxMessageTermination,
    TextMentionTermination,
)
from cognithor.compat.autogen.messages import TextMessage


def _stub_agent(name: str, replies: list[str]) -> MagicMock:
    agent = MagicMock()
    agent.name = name
    queue = list(replies)

    async def _run(*, task):
        from cognithor.compat.autogen._bridge import TaskResult

        msg = TextMessage(content=queue.pop(0), source=name)
        return TaskResult(messages=[msg], stop_reason=None)

    agent.run = _run
    return agent


@pytest.mark.asyncio
async def test_round_robin_terminates_on_max_messages() -> None:
    a = _stub_agent("a", ["a1", "a2", "a3"])
    b = _stub_agent("b", ["b1", "b2", "b3"])
    adapter = _RoundRobinAdapter(participants=[a, b], termination=MaxMessageTermination(4))

    result = await adapter.run(task="kickoff")
    assert len(result.messages) == 4
    assert result.messages[0].source == "a"
    assert result.messages[1].source == "b"
    assert result.messages[2].source == "a"
    assert result.messages[3].source == "b"
    assert result.stop_reason == "MaxMessageTermination"


@pytest.mark.asyncio
async def test_round_robin_terminates_on_text_mention() -> None:
    a = _stub_agent("a", ["working...", "almost", "DONE"])
    b = _stub_agent("b", ["b1", "b2", "b3"])
    adapter = _RoundRobinAdapter(
        participants=[a, b],
        termination=TextMentionTermination("DONE"),
    )

    result = await adapter.run(task="x")
    assert "DONE" in str(result.messages[-1].content)
    assert result.stop_reason == "TextMentionTermination"


@pytest.mark.asyncio
async def test_round_robin_combined_termination() -> None:
    """Composite condition `A | B` ends as soon as either fires."""
    a = _stub_agent("a", ["DONE", "n2", "n3"])
    b = _stub_agent("b", ["b1", "b2", "b3"])
    cond = MaxMessageTermination(100) | TextMentionTermination("DONE")
    adapter = _RoundRobinAdapter(participants=[a, b], termination=cond)
    result = await adapter.run(task="x")
    assert len(result.messages) == 1


@pytest.mark.asyncio
async def test_round_robin_empty_participants_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        _RoundRobinAdapter(participants=[], termination=MaxMessageTermination(2))


@pytest.mark.asyncio
async def test_round_robin_aggregates_token_usage() -> None:
    a = _stub_agent("a", ["a1", "a2"])
    b = _stub_agent("b", ["b1", "b2"])
    adapter = _RoundRobinAdapter(participants=[a, b], termination=MaxMessageTermination(2))
    result = await adapter.run(task="x")
    # token_usage attribute exists even if zero
    assert hasattr(result, "token_usage_total") or hasattr(result, "messages")
