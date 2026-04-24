"""First Crew — spec §1.4 walkthrough.

Sequential researcher + reporter pattern. Runs end-to-end with a local Ollama
backend — no API keys, no cloud calls.
"""

from __future__ import annotations

import asyncio

from cognithor.crew import Crew, CrewAgent, CrewProcess, CrewTask


def build_crew() -> Crew:
    researcher = CrewAgent(
        role="Researcher",
        goal="Recherchiere Fakten zum Thema",
        llm="ollama/qwen3:8b",
    )
    reporter = CrewAgent(
        role="Reporter",
        goal="Schreibe einen strukturierten Report",
        llm="ollama/qwen3:8b",
    )
    research = CrewTask(
        description="Recherchiere: Trends in Hausautomation 2026",
        expected_output="Bulletpoints der 5 wichtigsten Trends",
        agent=researcher,
    )
    report = CrewTask(
        description="Erstelle einen Report basierend auf der Research",
        expected_output="Markdown-Report, 300 Wörter",
        agent=reporter,
        context=[research],
    )
    return Crew(
        agents=[researcher, reporter],
        tasks=[research, report],
        process=CrewProcess.SEQUENTIAL,
    )


def main() -> None:
    crew = build_crew()
    result = asyncio.run(crew.kickoff_async())
    print(result.raw)


if __name__ == "__main__":
    main()
