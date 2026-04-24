"""CrewTask — declarative description of a unit of work."""

from __future__ import annotations

import uuid as _uuid
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cognithor.crew.agent import CrewAgent

# A function-based guardrail: takes the raw output string (to keep the public
# API decoupled from TaskOutput) plus a context dict, returns (ok, feedback).
# The detailed GuardrailResult structure lives in Feature 4.
type GuardrailCallable = Callable[[Any], tuple[bool, Any]]


class CrewTask(BaseModel):
    """Declarative unit of work executed by a CrewAgent."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    task_id: str = Field(default_factory=lambda: _uuid.uuid4().hex)
    description: str = Field(..., min_length=1)
    expected_output: str = Field(..., min_length=1)
    agent: CrewAgent
    context: list[CrewTask] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    guardrail: GuardrailCallable | str | None = None
    output_file: str | None = None
    output_json: type[BaseModel] | None = None
    async_execution: bool = False
    max_retries: int = Field(default=2, ge=0, le=10)


# Resolve the self-reference after the class is defined.
CrewTask.model_rebuild()
