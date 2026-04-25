"""ReportGenerator — final pre-advisory markdown report."""

from __future__ import annotations

from pathlib import Path

from cognithor.crew import CrewAgent

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "report_generator.md"


def build_report_generator(*, model: str) -> CrewAgent:
    backstory = _PROMPT_PATH.read_text(encoding="utf-8")
    return CrewAgent(
        role="report-generator",
        goal="Compose a §34d-NEUTRAL pre-advisory markdown report.",
        backstory=backstory,
        tools=[],
        llm=model,
        allow_delegation=False,
        max_iter=10,
        memory=True,
        verbose=False,
    )
