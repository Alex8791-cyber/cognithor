"""Internal bridge — translates AssistantAgent calls into cognithor.crew calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cognithor.crew import Crew, CrewAgent, CrewTask

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from cognithor.compat.autogen.agents._assistant_agent import AssistantAgent


@dataclass
class _AutoGenEvent:
    """AutoGen-shaped event — fields match autogen_agentchat.messages.TextMessage."""

    source: str = ""
    content: str = ""
    type: str = "TextMessage"
    models_usage: dict[str, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.content


@dataclass
class TaskResult:
    """AutoGen-shaped run() return — has `messages` list."""

    messages: list[_AutoGenEvent] = field(default_factory=list)
    stop_reason: str | None = None


def _coerce_task_text(task: str | Sequence[Any]) -> str:
    if isinstance(task, str):
        return task
    parts: list[str] = []
    for item in task:
        if isinstance(item, str):
            parts.append(item)
        else:
            parts.append(str(getattr(item, "content", item)))
    return "\n".join(parts)


def _make_cognithor_agent(agent: AssistantAgent) -> CrewAgent:
    backstory = agent.system_message or agent.description or ""
    return CrewAgent(
        role=agent.name,
        goal=agent.description or f"Assistant agent {agent.name}",
        backstory=backstory,
        llm=None,  # model_client wrapper resolved upstream by ModelRouter
        verbose=False,
        memory=False,
    )


async def run_single_task(
    agent: AssistantAgent,
    task: str | Sequence[Any],
) -> TaskResult:
    """1-shot path: create a single-agent, single-task Crew and run it."""
    text = _coerce_task_text(task)
    cognithor_agent = _make_cognithor_agent(agent)
    cognithor_task = CrewTask(
        description=text,
        expected_output="A direct answer to the user's task.",
        agent=cognithor_agent,
    )
    crew = Crew(agents=[cognithor_agent], tasks=[cognithor_task])
    output = await crew.kickoff_async({})

    raw = str(getattr(output, "raw", "") or "")
    usage = getattr(output, "token_usage", None) or {}
    event = _AutoGenEvent(
        source=agent.name,
        content=raw,
        type="TextMessage",
        models_usage={"total_tokens": int(usage.get("total_tokens", 0))} if usage else None,
        metadata={},
    )
    return TaskResult(messages=[event], stop_reason="task_completed")


async def stream_single_task(
    agent: AssistantAgent,
    task: str | Sequence[Any],
) -> AsyncIterator[Any]:  # pragma: no cover - streaming events are wrapper-thin
    """Streaming variant: emit one event per Cognithor-task plus a final stop event."""
    result = await run_single_task(agent, task)
    for msg in result.messages:
        yield msg
    yield result
