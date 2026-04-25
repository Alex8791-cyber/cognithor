"""Adapter Protocol + scenario types."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class ScenarioInput(BaseModel):
    """One scenario row from a JSONL file."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., min_length=1)
    task: str = Field(..., min_length=1)
    expected: str = Field(...)
    timeout_sec: int = Field(default=60, ge=1, le=3600)
    requires: tuple[str, ...] = Field(default_factory=tuple)


class ScenarioResult(BaseModel):
    """One adapter execution result."""

    model_config = ConfigDict(frozen=True)

    id: str
    output: str
    success: bool
    duration_sec: float
    error: str | None = None


@runtime_checkable
class Adapter(Protocol):
    """Pluggable benchmark adapter.

    Implementations:
      - cognithor_bench.adapters.cognithor_adapter.CognithorAdapter (default)
      - cognithor_bench.adapters.autogen_adapter.AutoGenAdapter (opt-in)
    """

    name: str

    async def run(self, scenario: ScenarioInput) -> ScenarioResult: ...
