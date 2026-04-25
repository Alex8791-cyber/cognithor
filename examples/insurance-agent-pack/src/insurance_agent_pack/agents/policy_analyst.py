"""PolicyAnalyst — extracts policy facts from PDF inputs (Cognithor CrewAgent)."""

from __future__ import annotations

from pathlib import Path

from cognithor.crew import CrewAgent

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "policy_analyst.md"


def build_policy_analyst(*, model: str) -> CrewAgent:
    backstory = _PROMPT_PATH.read_text(encoding="utf-8")
    return CrewAgent(
        role="policy-analyst",
        goal="Extract structured facts from German-language insurance policy PDFs.",
        backstory=backstory,
        tools=["pdf_extract_text"],
        llm=model,
        allow_delegation=False,
        max_iter=10,
        memory=True,
        verbose=False,
    )
