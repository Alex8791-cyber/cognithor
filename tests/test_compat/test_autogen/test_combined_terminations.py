"""End-to-end combined-termination behaviour through the public team class."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cognithor.compat.autogen.conditions import (
    MaxMessageTermination,
    TextMentionTermination,
)
from cognithor.compat.autogen.messages import TextMessage
from cognithor.compat.autogen.teams import RoundRobinGroupChat


def _stub_agent(name: str, replies: list[str]) -> MagicMock:
    agent = MagicMock()
    agent.name = name
    queue = list(replies)

    async def _run(*, task):
        from cognithor.compat.autogen._bridge import TaskResult

        return TaskResult(
            messages=[TextMessage(content=queue.pop(0), source=name)],
            stop_reason=None,
        )

    agent.run = _run
    return agent


@pytest.mark.asyncio
async def test_and_termination_requires_both_to_match() -> None:
    """AND requires count threshold AND text mention. Fires when both true."""
    a = _stub_agent("a", ["DONE", "DONE", "DONE", "DONE"])
    b = _stub_agent("b", ["x", "x", "x", "x"])
    cond = MaxMessageTermination(3) & TextMentionTermination("DONE")
    team = RoundRobinGroupChat(participants=[a, b], termination_condition=cond)
    result = await team.run(task="x")
    # turn 0: a:DONE (count=1, last has DONE) — count<3 → no fire
    # turn 1: b:x    (count=2, last="x")       — no DONE  → no fire
    # turn 2: a:DONE (count=3, last has DONE) — count>=3 AND DONE → FIRE
    assert len(result.messages) == 3
    assert "DONE" in str(result.messages[-1].content)


@pytest.mark.asyncio
async def test_or_termination_short_circuits() -> None:
    a = _stub_agent("a", ["DONE", "x", "x"])
    b = _stub_agent("b", ["x", "x", "x"])
    cond = MaxMessageTermination(100) | TextMentionTermination("DONE")
    team = RoundRobinGroupChat(participants=[a, b], termination_condition=cond)
    result = await team.run(task="x")
    assert len(result.messages) == 1


@pytest.mark.asyncio
async def test_complex_composite() -> None:
    a = _stub_agent("a", ["x", "DONE", "x"])
    b = _stub_agent("b", ["x", "x", "x"])
    cond = (MaxMessageTermination(2) & TextMentionTermination("DONE")) | MaxMessageTermination(10)
    team = RoundRobinGroupChat(participants=[a, b], termination_condition=cond)
    result = await team.run(task="x")
    # The (2 AND DONE) branch fires when DONE appears AND count >= 2.
    # Sequence: a:x (count=1, no DONE) -> b:x (count=2, no DONE) -> a:DONE (count=3, AND fires)
    assert len(result.messages) == 3
