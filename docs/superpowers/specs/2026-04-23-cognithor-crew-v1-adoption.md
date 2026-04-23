# Cognithor · Umsetzungsprompt: Übernahme ausgewählter CrewAI-Konzepte

**Version:** 1.0 · **Zielstand:** Cognithor v1.0.0 Launch (~6 Wochen)
**Adressat:** Coding-Modell mit Zugriff auf das `Alex8791-cyber/cognithor`-Repository
**Lizenzrahmen:** Cognithor ist Apache 2.0, CrewAI ist MIT. Konzepte dürfen frei adaptiert werden; **Code von CrewAI nicht verbatim kopieren.** Die in diesem Dokument referenzierten API-Designs sind Inspiration, keine Vorlage zum 1:1-Abschreiben.

---

## 0. Kontext für das Coding-Modell

Cognithor ist bereits ein voll funktionsfähiges Agent OS. Aktueller Stand laut Repo/PyPI:

- Version ~0.92.x, Python 3.12+, Apache 2.0
- 19 LLM-Provider, 18 Kommunikationskanäle, 145+ MCP-Tools
- PGE-Trinity: `Planner`, `Gatekeeper`, `Executor`, plus `Config`, `Models`, `Reflector`, `Distributed Lock`, `Model Router`, `DAG Engine`, `Delegation`, `Collaboration`, `Agent SDK`, `Workers`, `Personality`, `Sentiment`
- 6-Tier Memory mit 4-Channel Hybrid Search
- Bereits vorhandenes **Agent SDK** mit Decorators: `@agent`, `@tool`, `@hook` sowie Projekt-Scaffolding
- Skill-System: `registry`, `generator`, `marketplace`, `persistence`, `API`, `CLI tools`, `scaffolder`, `linter`, `BaseSkill`, `remote registry`
- Plugin Remote Registry mit SHA-256 Checksums, Dependency Resolution, Install/Update/Rollback
- Flutter Command Center (Dart/Flutter 3.41, Port 8741)
- 13.000+ Tests, 89% Coverage, 0 Lint-Errors, 0 CodeQL-Alerts
- Docker-Compose-Setups inkl. Postgres/Nginx/Prometheus/Grafana Profile
- CLI-Einstiegspunkte: `cognithor`, `python -m cognithor`, Flags: `--lite`, `--no-cli`

**Harte Regeln für alle folgenden Features:**

1. **Keine Breaking Changes** an bestehenden öffentlichen APIs. Alles Neue ist additiv.
2. **Alles baut auf PGE-Trinity auf.** Die Crew-API darunter ruft zwingend Planner → Gatekeeper → Executor, niemals direkt ein LLM.
3. **Test-Coverage darf nicht unter 89% fallen.** Jedes neue Feature kommt mit Tests.
4. **DSGVO-first.** Keine neuen externen Abhängigkeiten, die Daten rausschicken. Kein Cloud-Default.
5. **Existierende Patterns nutzen:** Agent SDK, BaseSkill, Scaffolder, Config-Forms, Skill-Registry – nichts davon neu erfinden.
6. **Sprache:** Code-Kommentare und Docstrings Englisch, Nutzer-facing-Texte Deutsch (mit i18n-Key für Englisch-Switch, analog zur bestehenden Lokalisierung).
7. **Lizenzhygiene:** CrewAI-Konzepte reimplementieren, keine CrewAI-Sourcefiles einfügen oder importieren.

---

## 1. Feature: Crew-Layer als Onboarding-Abstraktion über PGE-Trinity

### 1.1 Ziel

Ein **hochgelegener, deklarativer Syntax-Layer** ("Crew"), der es Nicht-Entwicklern und Cognithor-Neueinsteigern erlaubt, Multi-Agent-Workflows in unter 10 Zeilen Code zu definieren. Der Layer kompiliert darunter auf bestehende PGE-Trinity-Komponenten und das Agent SDK – er ersetzt sie nicht.

### 1.2 API-Design

Neues Modul: `cognithor.crew` (öffentlich exportiert über `cognithor/__init__.py`).

Primitive (Reimplementierung der Konzepte, keine CrewAI-Imports):

- `cognithor.crew.CrewAgent(role: str, goal: str, backstory: str = "", tools: list = None, llm: str | LLMConfig = None, allow_delegation: bool = False, max_iter: int = 20, memory: bool = True, verbose: bool = False)`
- `cognithor.crew.CrewTask(description: str, expected_output: str, agent: CrewAgent, context: list[CrewTask] = None, tools: list = None, guardrail: Callable | str = None, output_file: str = None, output_json: type[BaseModel] = None, async_execution: bool = False)`
- `cognithor.crew.Crew(agents: list[CrewAgent], tasks: list[CrewTask], process: CrewProcess = CrewProcess.SEQUENTIAL, verbose: bool = False, planning: bool = False, manager_llm: str | LLMConfig = None)`
- `cognithor.crew.CrewProcess` — Enum mit `SEQUENTIAL` und `HIERARCHICAL`
- `Crew.kickoff(inputs: dict = None) -> CrewOutput`
- `Crew.kickoff_async(inputs: dict = None) -> Awaitable[CrewOutput]`

Rückgabetyp `CrewOutput` enthält mindestens:
- `.raw: str` (finaler Text-Output)
- `.tasks_output: list[TaskOutput]`
- `.token_usage: TokenUsageDict`
- `.trace_id: str` (verbindet zur Hashline-Guard-Audit-Chain)

### 1.3 Mapping auf PGE-Trinity (normativ)

Dies ist die **einzige korrekte Übersetzung** – das Coding-Modell darf nicht direkt LLMs ansprechen:

| Crew-Konzept | Cognithor-Runtime |
|---|---|
| `Crew.kickoff()` | Erzeugt einen `PlanRequest`, routet durch den Planner |
| `CrewAgent` | Wird intern als Agent-SDK-Objekt registriert (nutzt `@agent`-Registry, keine neue Registry) |
| `CrewAgent.tools` | Werden über die bestehende Tool-Registry aufgelöst (inkl. MCP-Tools) |
| `CrewTask.description` + `expected_output` | Wird zu einem `TaskSpec` für den Planner |
| `CrewProcess.SEQUENTIAL` | Planner erzeugt linearen DAG mit `DAG Engine` |
| `CrewProcess.HIERARCHICAL` | Planner nutzt `Delegation`-Modul, `manager_llm` wird als Router-Model gesetzt |
| `CrewTask.guardrail` | Läuft über `Gatekeeper` als Post-Execution-Check (siehe Feature 4) |
| `CrewAgent.memory=True` | Aktiviert den bestehenden 6-Tier-Memory-Stack für diesen Agent-Scope |
| `Crew.planning=True` | Aktiviert den `Reflector` zwischen Tasks |
| `verbose=True` | Aktiviert strukturierte Events in das bestehende Event-Bus-System, sichtbar im Flutter Command Center |

### 1.4 Beispielnutzung (muss lauffähig werden)

```python
from cognithor.crew import Crew, CrewAgent, CrewTask, CrewProcess

analyst = CrewAgent(
    role="PKV-Tarif-Analyst",
    goal="Private Krankenversicherungstarife strukturiert vergleichen",
    backstory="Erfahrener Versicherungsmakler mit §34d-Zulassung, DSGVO-bewusst",
    tools=["web_search", "pdf_reader"],  # Namen aus der MCP-Tool-Registry
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
    description="Vergleiche die drei Top-PKV-Tarife für einen 42-jährigen GGF mit 95k Jahreseinkommen.",
    expected_output="Tabellarische Gegenüberstellung mit Beitrag, Leistungen, Ausschlüssen.",
    agent=analyst,
)

report = CrewTask(
    description="Erstelle einen Kunden-Report basierend auf der Analyse.",
    expected_output="PDF-tauglicher Markdown-Text, 500-800 Wörter, keine Fachjargon-Überfrachtung.",
    agent=writer,
    context=[research],
    output_file="output/pkv_report.md",
)

crew = Crew(
    agents=[analyst, writer],
    tasks=[research, report],
    process=CrewProcess.SEQUENTIAL,
    verbose=True,
)

result = crew.kickoff()
print(result.raw)
```

### 1.5 YAML-Variante (optional für v1.0, erforderlich für v1.1)

Analog zu CrewAIs `config/agents.yaml` + `config/tasks.yaml` Pattern: Unterstütze das Laden einer Crew aus zwei YAML-Dateien plus Python-Host-Klasse mit Decorators `@cognithor.crew.agent`, `@cognithor.crew.task`, `@cognithor.crew.crew` (nicht zu verwechseln mit den bestehenden `@agent`/`@tool`/`@hook` des Agent-SDK – Namespace über `cognithor.crew.*` hält sie sauber getrennt).

### 1.6 Akzeptanzkriterien

- [ ] Obiges Beispiel läuft mit Ollama-Backend, ohne Cloud-Calls, Default-Installation
- [ ] Jeder `CrewTask` erscheint als eigener Trace-Block in der Hashline-Guard-Audit-Chain
- [ ] Gatekeeper prüft JEDE vom Planner vorgeschlagene Tool-Aktion (auch in Crew-Modus)
- [ ] Fehlerszenario: Wenn ein Tool-Name nicht in der Registry ist, klare Fehlermeldung mit Vorschlag (`"Tool 'web_seach' nicht gefunden. Meintest du 'web_search'?"`)
- [ ] `kickoff()` ist idempotent re-aufrufbar (nutzt bestehende Distributed-Lock-Logik)
- [ ] Neue Tests im `tests/test_crew/`-Baum, mindestens: Sequential-Happy-Path, Hierarchical-Happy-Path, Missing-Tool-Error, Context-Passing, Async-Kickoff, Guardrail-Failure-Retry
- [ ] Bestehende Tests laufen unverändert (kein Refactoring der PGE-Internals)

### 1.7 Nicht-Ziele

- Keine neue Persistenzschicht – Crew-Zustände nutzen den vorhandenen State Store
- Keine eigene Telemetry – alles über das bestehende Event Bus System
- Keine verbatim-Kopie von CrewAI-Prompttemplates

---

## 2. Feature: Quickstart-Dokumentation auf Einsteiger-Niveau

### 2.1 Ziel

Ein neuer Pfad in der Doku, der einen Nicht-Experten in **unter 10 Minuten** vom leeren Terminal zur ersten laufenden Cognithor-Crew bringt.

### 2.2 Umfang

Neues Verzeichnis: `docs/quickstart/` mit folgenden Seiten (alle zweisprachig DE/EN):

1. `00-installation.md` – 3 Wege: Windows One-Click-Installer, `pip install cognithor[all]`, Docker Compose. Jeweils mit Verifikationsschritt (`cognithor --version`, Health-Check-URL).
2. `01-first-crew.md` – Das PKV-Beispiel aus Feature 1.4 Schritt-für-Schritt.
3. `02-first-tool.md` – Eigenes Tool via `@cognithor.tool` registrieren (bereits existierendes SDK nutzen) und in eine Crew einbinden.
4. `03-first-skill.md` – Unterschied Tool vs. Skill, wann was. Nutzt bestehenden Skill-Scaffolder.
5. `04-guardrails.md` – Verweis auf Feature 4.
6. `05-deployment.md` – Lokal, Docker, systemd, headless (`--no-cli`).
7. `06-next-steps.md` – Links zu Memory, Voice, Computer Use, MCP-Tool-Katalog.

### 2.3 Docs-Infrastruktur

- Nutze das bestehende Docs-System (kein Wechsel des Generators)
- Jede Seite am oberen Rand: Voraussetzungen, Zeitbedarf, Endzustand
- Jedes Code-Beispiel läuft auf einem sauberen Python 3.12 Setup mit `pip install cognithor[all]` und Ollama, nichts weiter
- Alle Beispiele sind als Dateien unter `examples/quickstart/` auch im Repo (nicht nur in der Doku), mit einem `README.md` pro Beispiel
- CI-Job `quickstart-examples.yml`: Spielt alle Beispiele in einem Docker-Container mit Ollama-Mock durch (pytest-basiert), verhindert dass Beispiele durch Refactorings brechen

### 2.4 Akzeptanzkriterien

- [ ] Ein externer Testleser (nicht der Autor) schafft `00` → `01` in unter 15 Minuten ohne Rückfragen
- [ ] Alle Beispiele sind lauffähig, unter CI getestet
- [ ] Cross-Links zu bestehender Doku (Status & Maturity, Security, Memory) sind an sinnvollen Stellen gesetzt
- [ ] `cognithor.ai` Startseite verlinkt prominent auf `docs/quickstart/00-installation.md`

### 2.5 Nicht-Ziele

- Keine Video-Tutorials in Scope von v1.0
- Keine Enterprise-Features im Quickstart (RBAC, SSO etc. gehören in die Advanced-Doku)

---

## 3. Feature: `cognithor init` CLI + First-Party Crew-Templates

### 3.1 Ziel

Ein CLI-Kommando, das ein lauffähiges Crew-Projekt aus einem benannten Template erzeugt. **Der bestehende Scaffolder wird dabei erweitert, nicht ersetzt.**

### 3.2 CLI-Spezifikation

Neues Sub-Command im bestehenden Cognithor-CLI:

```bash
cognithor init <project_name> --template <template_name> [--dir <path>] [--lang de|en]
cognithor init --list-templates
cognithor init --help
```

Beispiel:

```bash
cognithor init my_research_crew --template research
# → erzeugt ./my_research_crew/ mit lauffähiger Research-Crew
cd my_research_crew && cognithor run
```

Verhalten:
- `project_name` wird zu Python-kompatibler Snake-Case-ID konvertiert
- Default-Verzeichnis ist `./{project_name}/`
- Existiert das Verzeichnis bereits und ist nicht leer → Abbruch mit klarer Fehlermeldung
- Generiert `pyproject.toml`, `README.md` (DE + EN), `src/<project_name>/`, `config/agents.yaml`, `config/tasks.yaml`, `.env.example`, `main.py`, `tests/test_crew.py`
- `cognithor run` im generierten Projekt startet die Crew

### 3.3 First-Party Templates (v1.0-Umfang)

Fünf Templates im Repo unter `src/cognithor/crew/templates/`:

1. **`research`** – Researcher + Reporter (2 Agenten, sequential). Analog zu CrewAIs Standard-Quickstart, aber mit Cognithors MCP-Websuche statt Serper.
2. **`customer-support`** – Intake-Agent + Klassifikator + Response-Writer (3 Agenten, sequential). Nutzt Memory für Kundenhistorie.
3. **`data-analyst`** – Code-Interpreter-Agent (mit `allow_code_execution=True` in sandboxed Modus) + Visualisierungs-Agent. Nutzt das bestehende Sandbox-Modul.
4. **`content`** – Outline-Agent + Draft-Agent + Editor (3 Agenten, hierarchical mit Manager-LLM).
5. **`versicherungs-vergleich`** *(DACH-Differenzierer)* – PKV/BU-Tarif-Vergleichs-Crew mit drei Agenten: `Tarif-Researcher`, `Kunden-Profiler`, `Empfehlungs-Writer`. Output ist ein DSGVO-konformer Markdown-Report. Enthält explizite Guardrails gegen Angebots-charaktere (§34d-konforme Formulierungen – "Information, keine Beratung"). **Vollständig offline-fähig** mit Ollama, keine externen APIs im Default.

### 3.4 Template-Struktur (pro Template)

```
src/cognithor/crew/templates/<template>/
├── template.yaml            # Metadata: name, description_de, description_en, required_models, tags
├── pyproject.toml.jinja     # Projekt-Manifest
├── README.md.jinja.de
├── README.md.jinja.en
├── src/
│   └── {{project_name}}/
│       ├── __init__.py
│       ├── main.py.jinja
│       └── crew.py.jinja
├── config/
│   ├── agents.yaml.jinja
│   └── tasks.yaml.jinja
├── .env.example
└── tests/
    └── test_crew.py.jinja
```

Rendering über Jinja2 (bereits im Cognithor-Dependency-Baum vorhanden – verifizieren!).

### 3.5 Integration mit bestehendem Scaffolder

Der bestehende Skill-Scaffolder (`skills/scaffolder`) bleibt unverändert. `cognithor init` ist **ein neuer, paralleler** Einstiegspunkt speziell für Crew-Projekte – er darf aber die gleichen Utility-Funktionen wiederverwenden (Template-Rendering, Validation, Name-Sanitization).

### 3.6 Akzeptanzkriterien

- [ ] `cognithor init test_proj --template research` erzeugt lauffähiges Projekt
- [ ] `cognithor init --list-templates` zeigt alle 5 Templates mit DE-Beschreibung
- [ ] Das `versicherungs-vergleich`-Template läuft ohne Cloud-Credentials auf einer reinen Ollama-Installation
- [ ] Alle Templates haben lauffähige Tests (`pytest` im generierten Projekt → grün)
- [ ] CLI-Hilfe (`cognithor init --help`) ist zweisprachig
- [ ] Integration-Test im CI: Für jedes Template wird das Projekt erzeugt, `pytest` läuft, und die Crew einmal mit Mock-LLM ausgeführt
- [ ] Der bestehende Skill-Scaffolder ist weiter verfügbar und ungebrochen

### 3.7 Nicht-Ziele

- Keine Template-Marketplace-Integration in v1.0 (gehört zu "Agent Packs"-Roadmap, siehe Cognithor-Strategiepfad)
- Keine Third-Party-Templates per URL in v1.0

---

## 4. Feature: Task-Level Guardrails als First-Class-Primitive

### 4.1 Ziel

`CrewTask.guardrail` (aus Feature 1.2) als **explizit dokumentiertes, erstklassiges Feature**, nicht als Implementierungs-Detail.

### 4.2 Zwei Guardrail-Typen

Reimplementiere das Konzept, das CrewAI in `docs.crewai.com/en/concepts/tasks` beschreibt – aber komplett über Cognithors Gatekeeper-Infrastruktur:

**A) Function-based Guardrail** – Python-Callable:

```python
def validate_min_length(output: TaskOutput) -> tuple[bool, str | TaskOutput]:
    if len(output.raw.split()) < 150:
        return (False, "Output ist kürzer als 150 Wörter, bitte erweitern.")
    return (True, output)

CrewTask(
    description="...",
    expected_output="...",
    agent=writer,
    guardrail=validate_min_length,
)
```

Signatur verbindlich: `Callable[[TaskOutput], tuple[bool, str | TaskOutput]]`.
- `(True, output)` → Task erfolgreich, Output wandert weiter
- `(False, feedback_string)` → Retry mit Feedback als zusätzlichem Context
- Nach `max_retries` (default 2) Abbruch des Tasks mit `GuardrailFailure`-Exception

**B) String-based Guardrail** – natürliche Sprache, LLM-validiert:

```python
CrewTask(
    description="...",
    expected_output="...",
    agent=blog_agent,
    guardrail="Der Output darf keine konkreten Preise oder Beitragshöhen nennen und muss §34d-neutral formuliert sein.",
)
```

Wird intern zu einem LLM-Prüfungs-Call über den Gatekeeper. Modell-Default: dasselbe LLM wie der Agent; optional überschreibbar via `guardrail_llm` auf Crew-Ebene.

### 4.3 Eingebaute Guardrails

Cognithor liefert mindestens vier vorgefertigte Guardrails (als Factory-Funktionen):

- `cognithor.crew.guardrails.hallucination_check(reference: str)` – vergleicht Output gegen Referenz-Kontext
- `cognithor.crew.guardrails.word_count(min_words: int = None, max_words: int = None)`
- `cognithor.crew.guardrails.no_pii()` – blockt E-Mails, IBANs, Telefonnummern (DE-Format), Steuer-IDs (DSGVO-relevant, DACH-Fokus)
- `cognithor.crew.guardrails.schema(pydantic_model: type[BaseModel])` – erzwingt strukturierte Ausgabe

Kombinierbar via `cognithor.crew.guardrails.chain(*guardrails)`.

### 4.4 Integration mit Gatekeeper

Jede Guardrail-Ausführung ist ein **Gatekeeper-Event** und wird in der Hashline-Guard-Audit-Chain protokolliert mit:
- Task-ID, Guardrail-Typ (function/string), Ergebnis (pass/fail), Retry-Count, Begründung
- Feld `pii_detected: bool` für die DSGVO-spezifischen Guardrails

### 4.5 Akzeptanzkriterien

- [ ] Function-based und String-based Guardrails laufen, Retry-Logik funktioniert
- [ ] Alle vier eingebauten Guardrails haben Unit-Tests mit Edge-Cases (leerer Output, Unicode-Namen, deutsche IBAN, etc.)
- [ ] Guardrail-Events erscheinen in der Audit-Chain und sind über das bestehende Event-Bus-System abrufbar
- [ ] Dokumentationsseite `docs/quickstart/04-guardrails.md` mit lauffähigen Beispielen
- [ ] Das `versicherungs-vergleich`-Template nutzt `no_pii()` und einen custom String-Guardrail ("keine Tarif-Empfehlung, nur Vergleich")
- [ ] `GuardrailFailure`-Exception hat klare, aktionsorientierte Fehlermeldung

### 4.6 Nicht-Ziele

- Keine Content-Moderation-API-Anbindung (Perspective API etc.) in v1.0 – wäre Cloud-Call
- Kein selbstlernendes Guardrail-System

---

## 5. Feature (v1.x, nach Launch): Trace-UI im Flutter Command Center

### 5.1 Ziel

Visuelle Sichtbarmachung der PGE-Trinity + Crew-Execution im bestehenden Flutter Command Center.

### 5.2 Umfang

Neue Screen-Komponente im Flutter-Projekt: `TraceExplorerScreen`. Features:

- Liste aller Crew-Kickoffs der letzten 24h (aus Audit-Chain)
- Klick auf Kickoff → Detail-Screen mit Baumansicht:
  - Planner-Decision (Input, generierter Plan, verwendetes Modell)
  - Gatekeeper-Verdicts pro geplanter Aktion (approved / denied / modified, mit Begründung)
  - Executor-Calls (Tool-Name, Input, Output, Duration, Token-Usage)
  - Task-Output inkl. Guardrail-Verdict
- Filter: nach Agent-Name, Task-ID, Status (success/failed/retry)
- Export: JSON-Download einer kompletten Trace (für Debugging)

### 5.3 Datenquelle

Der FastAPI-Bridge auf Port 8741 erhält neue Endpunkte:

- `GET /api/v1/traces?since=<iso>&limit=<n>` – Liste
- `GET /api/v1/traces/{trace_id}` – Detail
- `GET /api/v1/traces/{trace_id}/export` – JSON-Download

Alle Daten kommen aus der bestehenden Hashline-Guard-Audit-Chain – **keine neue Persistenz.**

### 5.4 Design-Richtlinie

- Bestehende Sci-Fi-UI-Sprache des Command Center beibehalten
- Kein Light-Mode-Fork
- Ladezeiten: Liste < 300ms bei 1000 Traces (mit Pagination)

### 5.5 Akzeptanzkriterien

- [ ] Ein User kann den Trace einer gerade ausgeführten Crew vollständig nachvollziehen
- [ ] Gatekeeper-Denials sind visuell (rote Hervorhebung) sofort erkennbar
- [ ] Export-JSON ist valide und self-contained (alle Referenzen aufgelöst)
- [ ] Mobile + Desktop Flutter-Targets funktionieren
- [ ] Widget-Tests für TraceExplorerScreen in der bestehenden Flutter-Test-Suite

### 5.6 Priorität

**v1.x, nicht v1.0.** Implementierung darf v1.0 nicht blockieren.

---

## 6. Feature (v1.x): Event-driven Flows als Komplement zu Crews

### 6.1 Ziel

Deterministische, ereignisgetriebene Orchestrierungsschicht **über** Crews. Antwort auf reale DACH-Use-Cases wie "Wenn E-Mail eingeht → Research-Crew → bei Freigabe → Response-Crew → senden".

### 6.2 API-Design

Neues Modul: `cognithor.flow`.

Decorators (Reimplementierung der Konzepte aus `docs.crewai.com/en/concepts/flows`):

- `@cognithor.flow.start()` – Einstiegsmethode(n) der Flow-Klasse
- `@cognithor.flow.listen(upstream)` – reagiert auf Methode, Event-Name oder logische Kombination
- `@cognithor.flow.router(upstream)` – gibt String zurück, der zu `@listen("<string>")`-Methoden routet
- `cognithor.flow.or_(*upstreams)`, `cognithor.flow.and_(*upstreams)` – logische Operatoren

State-Management:
- `Flow[TState]` Basisklasse mit generischem Pydantic-State-Modell
- `self.state` ist die einzige Quelle der Wahrheit zwischen Schritten
- State wird automatisch persistiert (nutzt bestehenden State Store, analog zum `@persist`-Konzept)

Human-in-the-Loop:
- `@cognithor.flow.human_approval(channel: str = "default")` – pausiert Flow, pusht Approval-Request über einen der 18 Channels (Telegram/Slack/Web-UI etc.), wartet auf Antwort. State bleibt persistiert während der Wartezeit.

Flow-Ausführung:
- `flow = MyFlow(); flow.kickoff(inputs={...})`
- `flow.plot()` – erzeugt Mermaid-Diagramm der Flow-Struktur (für Doku/Debugging)

### 6.3 Beispielnutzung

```python
from cognithor.flow import Flow, start, listen, router, human_approval
from pydantic import BaseModel

class InsuranceLeadState(BaseModel):
    email_body: str = ""
    lead_score: float = 0.0
    research_result: str = ""
    response_draft: str = ""
    approved: bool = False

class InsuranceLeadFlow(Flow[InsuranceLeadState]):

    @start()
    def ingest_email(self):
        # Parse eingehende E-Mail aus Mail-Channel
        ...

    @listen(ingest_email)
    def score_lead(self):
        ...

    @router(score_lead)
    def route_by_score(self):
        return "qualified" if self.state.lead_score > 0.7 else "unqualified"

    @listen("qualified")
    def run_research_crew(self):
        # Ruft die versicherungs-vergleich Crew auf
        ...

    @listen("qualified")
    @human_approval(channel="telegram")
    def approve_response(self):
        # Wartet auf menschliche Freigabe
        ...

    @listen(approve_response)
    def send_response(self):
        ...
```

### 6.4 Integration

- Flows rufen Crews auf (nicht umgekehrt)
- Flow-State nutzt den bestehenden State Store
- Human-Approval-Pushes nutzen das bestehende Channel-System (18 Channels)
- Flow-Execution ist in derselben Trace-UI sichtbar wie Crew-Execution (Feature 5)

### 6.5 Akzeptanzkriterien

- [ ] Obiges Beispiel läuft end-to-end mit Telegram-Approval
- [ ] `flow.plot()` erzeugt korrektes Mermaid-Diagramm
- [ ] `or_`/`and_` funktionieren mit `@start`, `@listen`, `@router`
- [ ] State-Persistenz: Flow kann nach Prozess-Restart weiterlaufen
- [ ] Dokumentationsseite mit 3 realistischen Beispielen (Lead-Ingestion, Content-Pipeline mit Review, Daten-ETL)

### 6.6 Priorität

**v1.x.** v1.0 hat explizit nur Crews.

### 6.7 Nicht-Ziele

- Keine visuelle Flow-Builder-UI in v1.x (analog zu CrewAI Studio – das wäre eigenes Großprojekt)
- Kein verteiltes Flow-Execution (ein Prozess pro Flow-Instanz)

---

## 7. Feature: Offizieller Konnektor-Katalog (Marketing + Discoverability)

### 7.1 Ziel

Aus den 145+ MCP-Tools eine **kuratierte, marketing-taugliche "Official Integrations"-Ansicht** machen, die Nicht-Techniker auf cognithor.ai sofort verstehen.

### 7.2 Umfang

**Nur Doku- und Website-Arbeit, kein Code-Refactoring an Tools.**

#### 7.2.1 Neue Seite: `cognithor.ai/integrations`

Statische Seite (analog zur bestehenden Agent-Packs-Site im Neural-Noir-Design):

- Grid-Layout mit Logos der großen Integrations
- Kategorien: Produktivität (Outlook, Gmail, Google Calendar, Notion), Kommunikation (Slack, Teams, Telegram, Discord, WhatsApp), CRM (HubSpot, Pipedrive, Salesforce – sofern via MCP), Entwicklung (GitHub, GitLab), Daten (Postgres, SQLite, CSV), **DACH-Spezifika (DATEV, Lexware, sevDesk – sofern MCP-Tools existieren oder via generischem HTTP-Wrapper erreichbar)**
- Pro Integration: Logo, Ein-Satz-Nutzen, Link zur Dokumentationsseite
- Prominente Sektion "Self-hostable via MCP" mit Verweis auf das offene Protokoll

#### 7.2.2 Wahrheitspflicht

- **Nur Integrations listen, die tatsächlich im Repo existieren.** Kein "Coming Soon"-Vapourware.
- DACH-Spezifika: vor Listung prüfen, ob ein entsprechendes MCP-Tool/Skill im aktuellen Repo liegt. Wenn nicht: **in Scope von v1.0 EINES** (Vorschlag: sevDesk oder DATEV REST API) implementieren – oder aus der Seite weglassen.

#### 7.2.3 Datenquelle

Der Konnektor-Katalog wird **automatisch aus dem Repo generiert**. Neues Script `scripts/generate_integrations_catalog.py`:

- Scannt `src/cognithor/` nach MCP-Tool-Registrierungen
- Liest Metadata (Name, Description, Category) aus Tool-Definitionen
- Erzeugt `docs/integrations/catalog.json`
- Die Website rendert aus dieser JSON-Datei

Damit: Nie wieder Drift zwischen beworbener und vorhandener Integration.

### 7.3 Akzeptanzkriterien

- [ ] `cognithor.ai/integrations` Seite ist live
- [ ] Jede gelistete Integration verlinkt auf eine existierende Doku-Seite mit lauffähigem Beispiel
- [ ] `scripts/generate_integrations_catalog.py` läuft in CI und committed `catalog.json` automatisch
- [ ] Mindestens ein DACH-spezifischer Konnektor (sevDesk ODER DATEV ODER Lexware) ist funktional, getestet und in der Liste
- [ ] Keine Listing ohne Repo-Entsprechung (CI-Check)

### 7.4 Nicht-Ziele

- Keine eigene Konnektor-Entwicklung über MCP hinaus – wenn ein Dienst keine offene API hat, nicht listen
- Keine kommerziellen Tiers ("Official vs. Community") in v1.0

---

## 8. Querschnittsanforderungen

### 8.1 Tests

- Jedes neue Modul hat Unit-Tests mit mindestens 85% Line-Coverage
- Integration-Tests für Crew + Guardrails + PGE-Zusammenspiel
- E2E-Tests für mindestens eines der 5 Templates (Research-Crew) auf CI mit Ollama-Container
- Keine Reduktion der Gesamt-Coverage unter 89%

### 8.2 DSGVO / Privacy

- Keine neuen externen HTTP-Aufrufe im Default-Pfad
- Alle neuen Log-Ausgaben gehen durch den bestehenden PII-Sanitizer
- Das `versicherungs-vergleich`-Template muss vollständig offline-fähig sein

### 8.3 Dokumentation

- Jede neue öffentliche API (Klasse, Funktion, Decorator) hat vollständige Docstrings im Google-Style (konsistent zum bestehenden Code)
- Jede neue Doku-Seite ist zweisprachig
- `CHANGELOG.md` wird für jedes Feature separat gepflegt
- Breaking-Change-Sektion mit "keine" bestätigen

### 8.4 Lizenz & Attribution

- `NOTICE` oder `THIRD_PARTY.md` aktualisieren: "Das Crew-API-Design ist inspiriert durch CrewAI (MIT, crewAIInc/crewAI) – Re-Implementierung in Apache 2.0, kein Source-Level-Copy."
- Keine CrewAI-Pakete als Dependency

### 8.5 Performance

- `Crew.kickoff()` Overhead gegenüber direktem PGE-Planner-Call: < 5% zusätzliche Latenz bei identischem Workload (messbar via Benchmark-Suite)
- Template-Generation via `cognithor init` in < 500ms

### 8.6 Kompatibilität

- Python 3.12+ (unverändert)
- Bestehendes Agent SDK (`@agent`, `@tool`, `@hook`) bleibt funktional
- Bestehender Skill-Scaffolder bleibt funktional
- Bestehende CLI-Flags (`--lite`, `--no-cli`) bleiben funktional

---

## 9. Reihenfolge und Abhängigkeiten

**v1.0-Blocker (müssen alle fertig sein vor Release):**

1. Feature 1 (Crew-Layer) – Grundlage für alles
2. Feature 4 (Guardrails) – baut auf Feature 1
3. Feature 3 (`cognithor init` + Templates) – nutzt Feature 1 + 4
4. Feature 2 (Quickstart-Doku) – dokumentiert 1 + 3 + 4
5. Feature 7 (Konnektor-Katalog) – parallel möglich, rein Doku/Website

**v1.x (nach Launch):**

6. Feature 5 (Trace-UI) – setzt voraus, dass Crews Production-Workload sehen
7. Feature 6 (Flows) – größtes Einzelfeature, explizit nicht im v1.0-Scope

**Empfohlene Implementation-Order pro Woche:**

- **Woche 1:** Feature 1 Design + Core-Implementierung + Unit-Tests
- **Woche 2:** Feature 1 Integration mit PGE + Feature 4 (Guardrails)
- **Woche 3:** Feature 3 (Templates + CLI)
- **Woche 4:** Feature 2 (Quickstart-Doku) + alle E2E-Tests
- **Woche 5:** Feature 7 (Konnektor-Katalog) + DACH-Konnektor
- **Woche 6:** Buffer, Bugfixes, Release-Vorbereitung

---

## 10. Quellen (nachprüfbar)

**Cognithor-Specs:**
- Repo: `https://github.com/Alex8791-cyber/cognithor`
- PyPI: `https://pypi.org/project/cognithor/`

**CrewAI-Konzepte (zur API-Referenz, kein Code-Import):**
- Agent-API: `https://docs.crewai.com/en/concepts/agents`
- Task-API + Guardrails: `https://docs.crewai.com/en/concepts/tasks`
- Flow-API: `https://docs.crewai.com/en/concepts/flows`
- Quickstart-Struktur: `https://docs.crewai.com/en/quickstart`
- Lizenz (MIT): `https://github.com/crewAIInc/crewAI/blob/main/LICENSE`

---

## 11. Was dieser Prompt bewusst AUSSCHLIESST

Damit das Coding-Modell nicht in Scope-Creep abrutscht:

- **Kein Cloud-AMP-Äquivalent.** Kein hosted Cognithor, keine Managed-Infrastructure-APIs.
- **Kein Execution-Caps-Pricing.** Die Crew-API hat keine Quoten.
- **Kein Visual Studio Builder.** Drag-and-Drop UI ist nicht im Scope.
- **Kein verbatim CrewAI-Code.** Inspiration ja, Copy-Paste nein.
- **Kein Refactoring der PGE-Trinity.** Der Crew-Layer ist additiv.
- **Kein Ersatz des bestehenden Agent SDK.** Beide APIs koexistieren.
- **Keine Enterprise-Features** (SSO, SAML, RBAC-Erweiterungen) in v1.0.

---

## 12. Sign-off-Kriterien für v1.0-Release

- [ ] Alle Akzeptanzkriterien der Features 1–4 und 7 erfüllt
- [ ] Test-Coverage ≥ 89% bestätigt
- [ ] CI grün über alle Jobs (inkl. neuer `quickstart-examples.yml` und `integrations-catalog.yml`)
- [ ] Mindestens ein externer Nutzer hat den Quickstart erfolgreich durchgespielt
- [ ] Alle 5 Templates laufen auf frischer Docker-Installation ohne manuelle Eingriffe
- [ ] `CHANGELOG.md` vollständig, Migration-Guide (wenn nötig, nicht erwartet) vorhanden
- [ ] `cognithor.ai/integrations` live, Konnektor-Katalog-Generator in CI verankert

---

**Ende des Umsetzungsprompts.**
