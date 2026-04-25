"""RoundRobinGroupChat — AutoGen-shaped public class.

Thin wrapper over _RoundRobinAdapter from cognithor.compat.autogen._round_robin_adapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cognithor.compat.autogen._round_robin_adapter import (
    _RoundRobinAdapter,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cognithor.compat.autogen._round_robin_adapter import _RoundRobinResult
    from cognithor.compat.autogen.agents._assistant_agent import AssistantAgent
    from cognithor.compat.autogen.conditions import _TerminationCondition


class RoundRobinGroupChat:
    """Round-robin team — turns proceed in declaration order until terminated."""

    def __init__(
        self,
        participants: Sequence[AssistantAgent],
        *,
        termination_condition: _TerminationCondition,
        max_turns: int = 50,
    ) -> None:
        self._adapter = _RoundRobinAdapter(
            participants=participants,
            termination=termination_condition,
            max_turns=max_turns,
        )

    async def run(self, *, task: str) -> _RoundRobinResult:
        return await self._adapter.run(task=task)
