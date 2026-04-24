"""Guardrails — retry-with-feedback and GuardrailFailure.

Two scenarios:
  1. A task whose first attempt violates `no_pii()` / `word_count(max_words=10)`,
     then passes on retry because the Planner self-corrects.
  2. A task that fails the guardrail `max_retries + 1` times in a row and
     therefore raises `GuardrailFailure`.

Runs fully offline — the Planner is mocked in the smoke test. The inline
`main()` below talks to the real Ollama-backed default Planner when you run
`python main.py` on a machine with Ollama set up.
"""

from __future__ import annotations

import asyncio

from cognithor.crew import Crew, CrewAgent, CrewTask
from cognithor.crew.errors import GuardrailFailure
from cognithor.crew.guardrails import chain, no_pii, word_count


def build_strict_task(agent: CrewAgent) -> CrewTask:
    """Task with PII + length guardrails that retries twice on failure."""
    return CrewTask(
        description="Fasse zusammen: Was ist Cognithor in einem Satz.",
        expected_output="Ein einziger Satz, maximal 10 Wörter, keine E-Mail-Adressen.",
        agent=agent,
        guardrail=chain(no_pii(), word_count(max_words=10)),
        max_retries=2,
    )


def build_crew() -> Crew:
    writer = CrewAgent(
        role="Writer",
        goal="Schreibe knappe, PII-freie Zusammenfassungen",
        llm="ollama/qwen3:8b",
    )
    return Crew(
        agents=[writer],
        tasks=[build_strict_task(writer)],
    )


async def main() -> None:
    crew = build_crew()
    try:
        result = await crew.kickoff_async()
        print("Final output:", result.raw)
    except GuardrailFailure as exc:
        print(f"Guardrail gave up after {exc.attempts} attempts: {exc.reason}")


if __name__ == "__main__":
    asyncio.run(main())
