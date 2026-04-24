"""Spec §1.4 — end-to-end PKV (Private Krankenversicherung) example.

Acceptance test from spec §11: a realistic two-agent, two-task sequential
Crew must run end-to-end with a mocked Planner, pass prior-task context
through, aggregate token counts across tasks, and surface a trace_id on
the CrewOutput.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from cognithor.core.observer import ResponseEnvelope
from cognithor.crew import Crew, CrewAgent, CrewProcess, CrewTask


async def test_pkv_example_runs_end_to_end(monkeypatch):
    analyst = CrewAgent(
        role="PKV-Tarif-Analyst",
        goal="Private Krankenversicherungstarife strukturiert vergleichen",
        backstory="Erfahrener Versicherungsmakler mit §34d-Zulassung, DSGVO-bewusst",
        tools=[],
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
        expected_output="Tabellarische Gegenüberstellung mit Beitrag, Leistungen, Ausschlüssen.",
        agent=analyst,
    )
    report = CrewTask(
        description="Erstelle einen Kunden-Report basierend auf der Analyse.",
        expected_output=(
            "PDF-tauglicher Markdown-Text, 500-800 Wörter, keine Fachjargon-Überfrachtung."
        ),
        agent=writer,
        context=[research],
    )

    # CostTracker shim — returns different CostRecord per call to mimic two-task run.
    tracker = MagicMock()
    tracker.last_call = MagicMock(
        side_effect=[
            SimpleNamespace(input_tokens=500, output_tokens=100),
            SimpleNamespace(input_tokens=800, output_tokens=600),
        ]
    )

    mock_planner = MagicMock()
    mock_planner._cost_tracker = tracker
    mock_planner.formulate_response = AsyncMock(
        side_effect=[
            ResponseEnvelope(
                content=(
                    "| Tarif | Beitrag | Leistungen |\n|---|---|---|\n| A | 450€ | Stationär |"
                ),
                directive=None,
            ),
            ResponseEnvelope(
                content=("# PKV-Empfehlung\nBasierend auf der Analyse empfehlen wir..."),
                directive=None,
            ),
        ]
    )

    crew = Crew(
        agents=[analyst, writer],
        tasks=[research, report],
        process=CrewProcess.SEQUENTIAL,
        verbose=True,
        planner=mock_planner,
    )

    result = await crew.kickoff_async()

    assert "PKV-Empfehlung" in result.raw
    assert len(result.tasks_output) == 2
    assert result.trace_id
    # Aggregate token usage: sum of per-task totals (NOT recomputed from
    # prompt+completion). Task 1 used 600 (500+100), Task 2 used 1400 (800+600).
    assert result.token_usage["total_tokens"] == 2000
