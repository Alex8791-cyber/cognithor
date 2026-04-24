# 01 · Erste Crew

Dein erster Multi-Agent-Workflow: Researcher + Reporter arbeiten sequentiell zusammen.

**Voraussetzungen**
- Abgeschlossen: [00 · Installation](00-installation.md)
- `ollama list` zeigt `qwen3:8b`

**Zeitbedarf:** 5 Minuten
**Endzustand:** Du hast eine `main.py` lokal ausgeführt und bekommst einen zweistufigen Markdown-Report als Output.

---

## 1. Projekt anlegen

```bash
mkdir my_first_crew && cd my_first_crew
```

Erstelle `main.py` mit folgendem Inhalt (aus Spec §1.4 — das PKV-Beispiel):

```python
"""Erste Crew — sequentielles Researcher+Reporter-Pattern."""

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
```

## 2. Ausführen

```bash
python main.py
```

Die erste Ausführung kann 30–60 s dauern, weil Ollama das Modell lädt. Nachfolgende Läufe sind deutlich schneller.

Erwartete Ausgabe (gekürzt):

```
# Trends in Hausautomation 2026

- Matter/Thread-Adoption auf >60% neuer Geräte
- Lokale KI-Inferenz (Edge-LLMs) für Sprachsteuerung
- ...
```

## 3. Was ist passiert?

1. **`CrewAgent`** beschreibt nur die Rolle — keine Prompts, kein Code.
2. **`CrewTask`** beschreibt die Arbeit, plus Abhängigkeiten via `context=[research]`.
3. **`Crew(process=SEQUENTIAL)`** kompiliert beide Tasks in einen DAG und führt sie der Reihe nach aus.
4. Jeder Task durchläuft intern die **PGE-Trinity** (Planner → Gatekeeper → Executor) — genauso wie jede andere Cognithor-Aktion. Keine neue Security-Oberfläche.
5. Der finale `result.raw` ist der Output des letzten Tasks. Alle Zwischen-Outputs sind in `result.tasks_output`.

## 4. Lauffähige Version im Repo

Das gleiche Beispiel liegt als eigenständiges Mini-Projekt unter [`examples/quickstart/01_first_crew/`](../../examples/quickstart/01_first_crew/). Dort findest du:

- `main.py` — identisches Skript
- `requirements.txt` — nur `cognithor>=0.93.0`
- `test_example.py` — smoke-test mit gemocktem Planner, läuft im CI

## 5. Varianten

- **Mehr Agenten:** Füge zur `agents=` Liste beliebig weitere `CrewAgent`-Instanzen hinzu.
- **Hierarchical:** Setze `process=CrewProcess.HIERARCHICAL` + `manager=CrewAgent(...)` um einen Manager-Agent Tasks delegieren zu lassen.
- **YAML-Config:** Statt Python-Code kannst du `agents.yaml` + `tasks.yaml` nutzen — siehe `cognithor.crew.yaml_loader`.

---

**Next:** [02 · Eigenes Tool](02-first-tool.md)
