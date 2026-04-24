"""Cognithor Crew-Layer — high-level Multi-Agent API on top of PGE-Trinity.

Concept inspired by CrewAI (MIT, crewAIInc/crewAI) — re-implementation in
Apache 2.0; no source-level copy.

See docs/superpowers/specs/2026-04-23-cognithor-crew-v1-adoption.md.
"""

from __future__ import annotations

from cognithor.crew.agent import CrewAgent, LLMConfig
from cognithor.crew.crew import Crew
from cognithor.crew.errors import (
    CrewCompilationError,
    CrewError,
    GuardrailFailure,
    ToolNotFoundError,
)
from cognithor.crew.output import CrewOutput, TaskOutput, TokenUsageDict
from cognithor.crew.process import CrewProcess
from cognithor.crew.task import CrewTask, GuardrailCallable

__all__ = [
    "Crew",
    "CrewAgent",
    "CrewCompilationError",
    "CrewError",
    "CrewOutput",
    "CrewProcess",
    "CrewTask",
    "GuardrailCallable",
    "GuardrailFailure",
    "LLMConfig",
    "TaskOutput",
    "TokenUsageDict",
    "ToolNotFoundError",
]
