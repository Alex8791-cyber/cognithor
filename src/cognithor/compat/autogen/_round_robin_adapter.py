"""Multi-round adapter — drives RoundRobinGroupChat semantics through PGE-Trinity.

Per spec D4 (Hybrid mapping): single-shot AssistantAgent.run uses the cognithor.crew
1-shot path. RoundRobinGroupChat.run goes through THIS adapter, which loops over
participants in order, gathers messages, applies the termination condition, and
emits an AutoGen-shaped TaskResult.

Each participant's `.run(task=...)` is an `AssistantAgent` instance; we feed it
the running conversation as `task=` and collect its message back. The Gatekeeper
intercepts each tool call inside the underlying CrewAgent execution (out of scope
here) — this adapter only orchestrates turn-taking and termination.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cognithor.compat.autogen.messages import TextMessage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cognithor.compat.autogen.agents._assistant_agent import AssistantAgent
    from cognithor.compat.autogen.conditions import _TerminationCondition


@dataclass
class _RoundRobinResult:
    messages: list[TextMessage] = field(default_factory=list)
    stop_reason: str | None = None
    token_usage_total: int = 0


class _RoundRobinAdapter:
    """Round-robin orchestrator — internal helper for RoundRobinGroupChat."""

    def __init__(
        self,
        *,
        participants: Sequence[AssistantAgent],
        termination: _TerminationCondition,
        max_turns: int = 50,
    ) -> None:
        if not participants:
            raise ValueError("RoundRobinGroupChat requires at least one participant")
        self.participants = list(participants)
        self.termination = termination
        self.max_turns = max_turns

    async def run(self, *, task: str) -> _RoundRobinResult:
        result = _RoundRobinResult(messages=[], stop_reason=None, token_usage_total=0)
        running_task = task
        turn = 0
        while turn < self.max_turns:
            participant = self.participants[turn % len(self.participants)]
            sub_result = await participant.run(task=running_task)
            for msg in getattr(sub_result, "messages", []) or []:
                tm = (
                    msg
                    if isinstance(msg, TextMessage)
                    else TextMessage(
                        content=str(getattr(msg, "content", msg)),
                        source=str(getattr(msg, "source", participant.name)),
                    )
                )
                result.messages.append(tm)
                usage = getattr(msg, "models_usage", None)
                if usage:
                    result.token_usage_total += int(usage.get("total_tokens", 0))

            if self.termination.is_terminated(result.messages):
                result.stop_reason = self._termination_label()
                return result

            running_task = self._format_history_as_task(result.messages)
            turn += 1

        result.stop_reason = "MaxTurnsExceeded"
        return result

    def _termination_label(self) -> str:
        """Best-effort name lookup. For composite conditions, prefer reporting
        the inner leaf class names rather than the underscore-prefix wrapper."""
        cls = type(self.termination)
        name = getattr(cls, "__name__", "TerminationCondition")
        if name.startswith("_"):
            # _AndTermination / _OrTermination — duck-type access to .left/.right;
            # base _TerminationCondition lacks these, hence the attr-defined ignores.
            try:
                left = type(self.termination.left).__name__  # type: ignore[attr-defined]
                right = type(self.termination.right).__name__  # type: ignore[attr-defined]
                return f"{name.lstrip('_')}({left},{right})"
            except AttributeError:
                return name
        return name

    @staticmethod
    def _format_history_as_task(messages: Sequence[TextMessage]) -> str:
        return "\n".join(f"[{m.source}] {m.content}" for m in messages)
