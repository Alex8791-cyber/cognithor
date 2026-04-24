"""CrewAgent — declarative Pydantic model for a Crew participant."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Forward-compat stub (Spec §1.2): the spec allows `llm: str | LLMConfig`. For
# v1.0 LLMConfig is an opaque dict — concrete schema (temperature, seed, …)
# lands in a later minor release. Using PEP 695 `type` keeps the public type
# stable so adding a real BaseModel later is a pure type-widening (non-breaking).
type LLMConfig = dict[str, Any]


class CrewAgent(BaseModel):
    """Declarative description of an agent participating in a Crew.

    Concept inspired by CrewAI's Agent; re-implementation in Apache 2.0.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    role: str = Field(..., min_length=1, description="Short role name, used in logs")
    goal: str = Field(..., min_length=1, description="What this agent is trying to accomplish")
    backstory: str = Field(
        default="", description="Context the Planner uses to shape the system prompt"
    )
    tools: list[str] = Field(
        default_factory=list, description="Tool names resolved via MCP registry"
    )
    # Spec §1.2 — widened to str | LLMConfig | None. LLMConfig is currently a
    # dict alias; a future BaseModel swap-in is non-breaking.
    llm: str | LLMConfig | None = Field(
        default=None,
        description="Model spec (e.g. 'ollama/qwen3:32b') or LLMConfig dict",
    )
    allow_delegation: bool = Field(default=False)
    max_iter: int = Field(default=20, ge=1, le=200)
    memory: bool = Field(default=True, description="Enable 6-Tier Cognitive Memory for this agent")
    verbose: bool = Field(default=False)
    metadata: dict[str, Any] = Field(default_factory=dict)
