"""Crew — top-level orchestration object."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cognithor.crew.agent import CrewAgent, LLMConfig
from cognithor.crew.process import CrewProcess
from cognithor.crew.task import CrewTask

if TYPE_CHECKING:
    from cognithor.crew.output import CrewOutput


class Crew(BaseModel):
    """A Crew is a declarative bundle of agents + tasks + process."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    agents: list[CrewAgent] = Field(..., min_length=1)
    tasks: list[CrewTask] = Field(..., min_length=1)
    process: CrewProcess = CrewProcess.SEQUENTIAL
    verbose: bool = False
    planning: bool = False
    # Spec §1.2 — matches CrewAgent.llm: str | LLMConfig | None.
    manager_llm: str | LLMConfig | None = None

    @model_validator(mode="after")
    def _warn_on_hierarchical_without_manager(self) -> Crew:
        if self.process is CrewProcess.HIERARCHICAL and self.manager_llm is None:
            warnings.warn(
                "CrewProcess.HIERARCHICAL without manager_llm falls back to the "
                "first agent's llm for routing decisions. For production, set "
                "manager_llm explicitly.",
                stacklevel=3,
            )
        return self

    def kickoff(self, inputs: dict[str, Any] | None = None) -> CrewOutput:
        """Synchronous kickoff. Calls the sync compiler directly; Task 9 rewires
        this through asyncio.run(self.kickoff_async(inputs)).
        """
        from cognithor.crew.compiler import compile_and_run_sync
        from cognithor.crew.runtime import get_default_tool_registry

        return compile_and_run_sync(
            agents=self.agents,
            tasks=self.tasks,
            process=self.process,
            inputs=inputs,
            registry=get_default_tool_registry(),
            manager_llm=self.manager_llm if isinstance(self.manager_llm, str) else None,
        )

    async def kickoff_async(self, inputs: dict[str, Any] | None = None) -> CrewOutput:
        """Async kickoff with parallel fan-out for tasks marked async_execution=True."""
        from cognithor.crew.compiler import compile_and_run_async
        from cognithor.crew.runtime import get_default_tool_registry

        return await compile_and_run_async(
            agents=self.agents,
            tasks=self.tasks,
            process=self.process,
            inputs=inputs,
            registry=get_default_tool_registry(),
            manager_llm=self.manager_llm if isinstance(self.manager_llm, str) else None,
        )
