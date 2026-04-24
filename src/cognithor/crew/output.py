"""Crew output dataclasses — immutable result objects."""

from __future__ import annotations

from typing import Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field, computed_field


class TokenUsageDict(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def empty_token_usage() -> TokenUsageDict:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


class TaskOutput(BaseModel):
    """Result of one CrewTask execution."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    task_id: str
    agent_role: str
    raw: str
    structured: dict[str, Any] | None = None
    duration_ms: float = 0.0
    token_usage: TokenUsageDict = Field(default_factory=empty_token_usage)
    guardrail_verdict: str | None = None  # pass / fail / skipped


class CrewOutput(BaseModel):
    """Aggregate result of one Crew.kickoff()."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    raw: str
    tasks_output: list[TaskOutput]
    trace_id: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def token_usage(self) -> TokenUsageDict:
        # Sum each field independently. total_tokens is summed directly from
        # per-task totals (not recomputed from prompt + completion) so that
        # providers which report additional buckets (e.g. cached tokens) keep
        # their contribution to total_tokens at the aggregate level.
        return {
            "prompt_tokens": sum(t.token_usage["prompt_tokens"] for t in self.tasks_output),
            "completion_tokens": sum(t.token_usage["completion_tokens"] for t in self.tasks_output),
            "total_tokens": sum(t.token_usage["total_tokens"] for t in self.tasks_output),
        }
