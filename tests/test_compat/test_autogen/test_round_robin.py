"""RoundRobinGroupChat — public AutoGen-shaped class."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cognithor.compat.autogen.conditions import MaxMessageTermination
from cognithor.compat.autogen.messages import TextMessage
from cognithor.compat.autogen.teams import RoundRobinGroupChat


def _stub_assistant(name: str, replies: list[str]) -> MagicMock:
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
async def test_round_robin_constructor_and_run() -> None:
    a = _stub_assistant("a", ["msg1", "msg2"])
    b = _stub_assistant("b", ["msg3", "msg4"])
    team = RoundRobinGroupChat(
        participants=[a, b],
        termination_condition=MaxMessageTermination(2),
    )
    result = await team.run(task="hello")
    assert len(result.messages) == 2


def test_round_robin_attributes_match_autogen_signature() -> None:
    """Construction kwargs match `RoundRobinGroupChat(participants=, termination_condition=)`."""
    import inspect

    sig = inspect.signature(RoundRobinGroupChat.__init__)
    params = list(sig.parameters)
    assert "participants" in params
    assert "termination_condition" in params
