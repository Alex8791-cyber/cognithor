"""NeedsAssessor — converts interview answers into a structured profile."""

from __future__ import annotations

from pathlib import Path

from cognithor.crew import CrewAgent

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "needs_assessor.md"


def build_needs_assessor(*, model: str) -> CrewAgent:
    backstory = _PROMPT_PATH.read_text(encoding="utf-8")
    return CrewAgent(
        role="needs-assessor",
        goal="Convert interview answers into a structured insurance-needs profile.",
        backstory=backstory,
        tools=[],
        llm=model,
        allow_delegation=False,
        max_iter=20,
        memory=True,
        verbose=False,
    )
