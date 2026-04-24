"""PKV-Report — spec §1.4 runnable example.

Ein realistischer Zwei-Agenten-Sequential-Flow:
  * `PKV-Tarif-Analyst` recherchiert strukturiert.
  * `Kunden-Report-Schreiber` formuliert für Endkunden.

Die Output-Datei `output/pkv_report.md` wird vom zweiten Task geschrieben.
Komplett offline-fähig — Default-LLM ist `ollama/qwen3:*`, keine Cloud-APIs.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from cognithor.crew import Crew, CrewAgent, CrewProcess, CrewTask


def build_crew() -> Crew:
    analyst = CrewAgent(
        role="PKV-Tarif-Analyst",
        goal="Private Krankenversicherungstarife strukturiert vergleichen",
        backstory="Erfahrener Versicherungsmakler mit §34d-Zulassung, DSGVO-bewusst",
        tools=[],  # Echte Crews setzen hier ["web_search", "pdf_reader"]
        llm="ollama/qwen3:32b",
        memory=True,
    )
    writer = CrewAgent(
        role="Kunden-Report-Schreiber",
        goal="Analyst-Ergebnisse in eine kundenverständliche PDF überführen",
        backstory="Spezialist für kundentaugliche Finanzkommunikation",
        llm="ollama/qwen3:8b",
    )
    research = CrewTask(
        description=(
            "Vergleiche die drei Top-PKV-Tarife für einen 42-jährigen GGF mit 95k Jahreseinkommen."
        ),
        expected_output=("Tabellarische Gegenüberstellung mit Beitrag, Leistungen, Ausschlüssen."),
        agent=analyst,
    )
    report = CrewTask(
        description="Erstelle einen Kunden-Report basierend auf der Analyse.",
        expected_output=(
            "PDF-tauglicher Markdown-Text, 500-800 Wörter, keine Fachjargon-Überfrachtung."
        ),
        agent=writer,
        context=[research],
        output_file="output/pkv_report.md",
    )
    return Crew(
        agents=[analyst, writer],
        tasks=[research, report],
        process=CrewProcess.SEQUENTIAL,
        verbose=True,
    )


def main() -> None:
    Path("output").mkdir(exist_ok=True)
    crew = build_crew()
    result = asyncio.run(crew.kickoff_async())
    print(result.raw)
    print("Gesamt-Tokens:", result.token_usage["total_tokens"])
    print("Trace-ID:", result.trace_id)


if __name__ == "__main__":
    main()
