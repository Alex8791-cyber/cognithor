# Cognithor × AutoGen Strategy — v0.94.0 Design Spec

**Status:** Draft — 2026-04-25
**Target Release:** v0.94.0
**Source Prompt:** `~/Downloads/cognithor_autogen_strategy_prompt.md`
**Decided in Brainstorm:** 2026-04-25 (auto-mode session)

---

## 1. Executive Summary

Microsoft hat AutoGen offiziell in den **Maintenance-Modus** versetzt und verweist auf das Microsoft Agent Framework (MAF) 1.0 (April 2026, MIT) als Nachfolger. MAF's Programmiermodell ist von conversation-centric zu graph-based gewechselt — eine harte Migration für tausende AutoGen-Nutzer.

Cognithor positioniert sich in diesem Window als **lock-in-freie EU-Alternative** und **AutoGen-Migrations-Onramp**, ohne Cognithor's PGE-Trinity-Architektur zu verbiegen.

Diese Spec definiert **fünf Arbeitspakete (WP1-WP5)** für v0.94.0, geliefert als **vier Pull Requests** plus **direkter Release-Commit auf main** (analog zur v0.93.0-Release-Discipline). Kein Vendoring von AutoGen-Code; API-Shape-Kompatibilität durch Inspection-Tests gegen `autogen-agentchat==0.7.5`.

---

## 2. Brainstorm-Decisions (binding)

Diese Entscheidungen wurden im 2026-04-25-Brainstorm getroffen und sind nicht-revidierbar ohne Spec-Update:

| ID | Frage | Entscheidung |
|----|-------|--------------|
| D1 | Scope-Cut | **Mega-Spec mit allen 5 WPs** (nicht decomposed in Sub-Specs) |
| D2 | Release-Timing | **v0.94.0** = alle 5 WPs (nicht post-v1.0.0 wie im Source-Prompt vorgeschlagen) |
| D3 | WP3-Form | **Standalone Pack** unter `examples/insurance-agent-pack/` (nicht im versicherungs-vergleich-Template aufgehen) |
| D4 | WP2-Approach | **Hybrid-Mapping**: Single-Agent → `cognithor.crew`, Multi-Round → eigener `_RoundRobinAdapter` |
| D5 | WP4-Location | **Submodul im Monorepo** (`cognithor_bench/`, eigenes `pyproject.toml`) |
| D6 | WP2-Tests | **Pure Signatur + Hello-World-Behavior-Test** (nicht Property-Test, nicht Bench-Cross-Cut) |

**Sieben Verbesserungen aus der Design-Review eingearbeitet (siehe §11):**

- **F1:** WP3 NICHT durch `cognithor.packs`-Loader registriert — pure `pip install`
- **F2:** WP3 baut auf v0.93.0's `versicherungs-vergleich`-Template auf, dupliziert es nicht
- **F3:** Kein PR 5 — Release-Bundle direkt auf main committed (4 PRs + 1 Direct-Commit + Tag)
- **F4:** WP2 RoundRobin-Adapter ~250-300 LOC (von "~100 LOC" auf realistisch korrigiert)
- **F5:** Pre-WP1 Gate: 24-48h v0.93.0-Stabilität ohne Hotfix bevor PR 1 startet
- **F6:** Explizite `[project.optional-dependencies] autogen = [...]` in pyproject.toml
- **F7:** Single Pin-Point für `autogen-agentchat==0.7.5` (zentral in pyproject.toml, von WP2 + WP4 referenziert)

---

## 3. Validierte Faktenbasis

Aus dem Source-Prompt §10, hier nur die Kern-Punkte (Quellen sind im Prompt referenziert, nicht hier dupliziert):

### 3.1 AutoGen-Status (Q2 2026)

- `microsoft/autogen` Repo trägt seit ~Oktober 2025 expliziten Maintenance-Hinweis
- Letztes aktives Release: Python `v0.7.5` (Sept 2025)
- Microsoft leitet Nutzer aktiv zu Microsoft Agent Framework (MAF) weiter

### 3.2 AutoGen-Architektur (Layered Design — drei Pakete)

- **`autogen-core`** — Actor-Model-Runtime, `SingleThreadedAgentRuntime`, `RoutedAgent`, `@message_handler`. **OUT-OF-SCOPE** für Cognithor-Compat (zu tief mit Actor-Modell verwoben).
- **`autogen-agentchat`** — High-Level-API. **DAS IST WP2's TARGET.** Klassen: `AssistantAgent`, `BaseChatAgent`, Teams (`RoundRobinGroupChat`, `SelectorGroupChat`, `Swarm`, `MagenticOneGroupChat`, `GraphFlow`), Messages (`TextMessage`, `ToolCallSummaryMessage`, `HandoffMessage`, `StructuredMessage`).
- **`autogen-ext`** — Provider/Tools. WP2 nimmt nur `OpenAIChatCompletionClient` als Wrapper-Target.

### 3.3 `AssistantAgent` Signatur (autogen_agentchat 0.7.5)

Die Pydantic-Konfiguration `AssistantAgentConfig` hat 14 Felder (siehe Source-Prompt §2.3). WP2's `cognithor.compat.autogen.AssistantAgent.__init__` muss diese Felder **byte-genau** spiegeln (Reihenfolge, Defaults, Typen). Verifikation via `inspect.signature`-Diff in Tests.

### 3.4 MIT/Apache-Brücke

AutoGen ist MIT-lizenziert (Code) / CC-BY-4.0 (Docs). Cognithor ist Apache 2.0. **Kompatibel** — `NOTICE` muss AutoGen-Inspiration vermerken (analog zur CrewAI-Attribution aus v0.93.0).

---

## 4. Cognithor-Kontext (post-v0.93.0)

**Repo State nach v0.93.0:** `Alex8791-cyber/cognithor` auf `main` mit:

- 5 v0.93.0-PRs gemergt (#141 F1 Crew-Layer, #142 F4 Guardrails, #143 F3 CLI+Templates, #144 F7 Integrations+sevDesk, #145 F2 Quickstart)
- `cognithor.crew` ist die **production Multi-Agent-API** (Pydantic v2, frozen models, Async-Kickoff, Idempotent-Replay, Distributed-Lock, Audit-Chain, PII-Redaction)
- 5 Templates inkl. `versicherungs-vergleich` (DACH, offline, no_pii + StringGuardrail)
- `cognithor.packs` System (loader/installer für **kommerzielle** Packs aus privatem `cognithor-packs` Repo, NICHT für public references)
- PyPI: `pip install cognithor==0.93.0`

**Hardware-Baseline:** RTX 5090, Ryzen 9 9950X3D, Windows 11, Ollama lokal.

---

## 5. Gesamtziel und Nicht-Ziele

### 5.1 Gesamtziel

Cognithor positionieren als:
1. **lock-in-freie EU-Alternative** zu Microsoft Agent Framework
2. **natürlicher Migrations-Pfad** für AutoGen-Nutzer ohne Azure-Abhängigkeit

### 5.2 Nicht-Ziele

- Keine Runtime-Abhängigkeit auf `autogen-core`, `autogen-agentchat` oder `agent-framework`-Pakete
- Keine MAF-Kompatibilität (Graph-Modell würde PGE-Trinity verbiegen)
- Keine Änderungen an PGE-Trinity selbst
- Keine Anpassungen am Cognithor-Release-Pipeline (re-uses v0.93.0 patterns)
- Keine öffentliche Kommunikation (Blog, Reddit) — Alexander entscheidet
- Keine echten Kundendaten in WP3-Fixtures (alles synthetic)

---

## 6. Globale Randbedingungen

- **Python 3.12+**, Type Hints vollständig, Pydantic v2, `ruff` + `mypy --strict` müssen grün sein
- **Apache 2.0** bleibt Lizenz; AutoGen-Inspiration in `NOTICE` (analog CrewAI)
- **Single Pin-Point:** `autogen-agentchat==0.7.5` als **eine** zentrale Konstante in `pyproject.toml` `[project.optional-dependencies] autogen = [...]`. WP2-Tests + WP4-Adapter referenzieren diese Pin, niemals eigene Versions-Strings
- **DSGVO-First:** Wenn ein WP externe Calls macht, EU-Provider-Variante dokumentieren (Ollama lokal, Mistral, Azure EU)
- **Conventional Commits:** `feat:`, `docs:`, `test:`, `refactor:`
- **Branching:** Jeder PR auf eigenem Feature-Branch `feat/cognithor-autogen-vN-<kurzname>`, einzelne PRs gegen `main`. Keine Sammel-Commits. Cleanup nach Merge in **separater Turn** (per Memory `feedback_pr_merge_never_chain_cleanup.md`)
- **Pre-WP1 Gate:** 24-48h v0.93.0-Stabilität ohne v0.93.1-Hotfix-Bedarf bevor PR 1 startet (verhindert Branch-Konflikte mit Patch-Releases)

---

## 7. Arbeitspaket-Übersicht

| WP  | Titel                                      | PR  | Branch                                          | Tasks | Aufwand    |
|-----|--------------------------------------------|-----|------------------------------------------------|-------|------------|
| WP1 | Competitive Analysis Dokumente             | PR1 | `feat/cognithor-autogen-v1-docs`               | 5     | 1 Tag      |
| WP5 | PGE-Trinity ADR                            | PR1 | (gleiche PR)                                   | 3     | 0.5 Tag    |
| WP4 | `cognithor-bench` Scaffold                 | PR2 | `feat/cognithor-autogen-v2-bench`              | 14    | 4-5 Tage   |
| WP2 | AutoGen-Compatibility-Shim                 | PR3 | `feat/cognithor-autogen-v3-compat`             | 18    | 7-9 Tage   |
| WP3 | Insurance Agent Pack (Standalone)          | PR4 | `feat/cognithor-autogen-v4-insurance`          | 14    | 5-7 Tage   |
| —   | v0.94.0 Release-Bundle                     | DC* | `main` (direct commit + tag)                    | 5     | 1 Tag      |

*DC = Direct Commit auf main (kein PR), pattern aus v0.93.0 reused.

**Total:** ~59 Tasks across 4 PRs + 1 Direct-Commit. Reihenfolge: PR1 → PR2 → PR3 → PR4 → Direct-Commit → Tag-Push.

**Aufwand seriell:** ~17-22 Tage. **Mit Subagent-Parallelisierung:** ~10-14 Tage realistisch.

---

## 8. WP-Details

### 8.1 WP1 — Competitive Analysis Dokumente

**Ziel:** Saubere, zitierfähige Vergleichsdokumente; Grundlage für Marketing + Doku + zukünftige Architektur-Entscheidungen.

**Deliverables:**

```
docs/competitive-analysis/
├── README.md                         # Index, ~150 Wörter
├── autogen.md                        # Cognithor vs AutoGen, ≥400 Wörter
├── microsoft-agent-framework.md      # Cognithor vs MAF, ≥400 Wörter
└── decision-matrix.md                # Side-by-side Markdown-Tabelle
```

**`autogen.md` Pflicht-Sections:**
1. Status von AutoGen Q2 2026 (Faktenbasis, keine Meinung)
2. Architektur-Zusammenfassung (3-Layer-Design)
3. Gemeinsamkeiten mit Cognithor (MCP-Support, Multi-Agent-Pattern, lokale Modelle, Tool-Use)
4. Unterschiede (ehrlich!): AutoGen besser bei cross-language, Community, Docs. Cognithor besser bei PGE-Trinity, DSGVO, Vendor-Neutralität (16 Provider), lokale Inferenz First-Class, Deep Research v2, 6-Tier Memory
5. Konzeptioneller Migrations-Pfad (Details in WP2)

**`microsoft-agent-framework.md` Pflicht-Sections:**
1. Was MAF ist (GA April 2026, MIT, Python+.NET, graph-based)
2. Programmiermodell-Shift (graph statt conversation, `@tool` statt `FunctionTool`)
3. Warum Cognithor dennoch existiert (EU-Souveränität, keine Azure-Abhängigkeit, DSGVO-relevante Features)
4. Kein Framework-War-Narrativ: Anerkennen dass MAF für Azure-zentrierte Enterprise exzellent ist; Cognithor ist komplementär

**`decision-matrix.md` Dimensionen** (Zeilen) × Frameworks (Spalten = Cognithor, AutoGen, MAF, LangGraph, CrewAI):
- Lizenz (Core)
- Host-Region (Default)
- Lokale Inferenz First-Class
- Anzahl LLM-Provider out-of-the-box
- MCP-Client
- A2A-Protokoll
- Multi-Agent-Pattern (Conversation vs Graph vs PGE)
- DSGVO-Compliance (expliziter Claim?)
- Audit-Chain
- Kommerzielle Abhängigkeit (keine / Microsoft / etc.)
- Aktiver Maintenance-Status

**Acceptance Criteria:**
- [ ] Alle drei Dateien existieren und ≥400 Wörter (README darf kürzer sein)
- [ ] Jede faktische Behauptung über AutoGen/MAF hat Quellenlink (Footnote oder inline)
- [ ] Keine Behauptung über Konkurrenten die über öffentliche Docs hinausgeht
- [ ] Tone: sachlich, kein "AutoGen ist tot, Cognithor rettet euch"
- [ ] In `docs/README.md` (oder Hauptindex) verlinkt

**Negative Acceptance:**
- ❌ Keine übernommenen Code-Snippets aus AutoGen/MAF ohne Lizenzvermerk
- ❌ Keine Performance-Claims ohne Benchmark-Grundlage (gibt es bis WP4 nicht)
- ❌ Kein Verweis auf Cognithor-Features die nicht in main gemerged + getestet sind

---

### 8.2 WP5 — PGE-Trinity ADR

**Ziel:** Architecture Decision Record (Nygard-Style), das Cognithor's PGE-Trinity als bewusste Abgrenzung von AutoGens Group-Chat-Muster dokumentiert. Zitierfähige Antwort auf "Warum kein einfacher Group Chat?"

**Deliverables:**

```
docs/adr/
├── README.md                                # ADR-Index + Nygard-Template-Note
└── 0001-pge-trinity-vs-group-chat.md        # Erste ADR (docs/adr/ ist neu)
```

**Content-Struktur (ADR-Format):**

```markdown
# ADR 0001: PGE Trinity als Multi-Agent-Kontrollmodell

## Status
Accepted — 2026-04-XX

## Context
[AutoGen GroupChat-Pattern, Magentic-One-Orchestrator, Cognithor's strikte Trennung]

## Decision
Cognithor nutzt Planner-Gatekeeper-Executor als erzwungene Rollenseparation.
[Was jeder Agent darf/nicht darf, Gatekeeper DSGVO-Checks, Delegation]

## Consequences
### Positiv
- Auditierbarkeit: jede Action durch Gatekeeper (Hashline Guard Chain)
- DSGVO: PII-Filter zentral, nicht in jedem Agenten neu
- Keine "Agent-Drift" durch emergente Rollenverletzung

### Negativ / Trade-offs (≥3 ehrlich!)
- Höhere Latenz als direkter Group Chat (Gatekeeper-Hop)
- Weniger "kreative" emergente Lösungen — by design
- Höhere Einstiegshürde für AutoGen-Migranten

## Alternatives Considered
1. RoundRobinGroupChat-Äquivalent — verworfen: kein Audit-Punkt
2. SelectorGroupChat-Äquivalent — verworfen: LLM als Sicherheitsgrenze nicht tragfähig
3. Pure Handoff/Swarm — verworfen: keine zentrale Policy-Durchsetzung

## References
- AutoGen GroupChat Docs
- Magentic-One Paper (arxiv 2411.04468)
- Cognithor Gatekeeper Code
```

**Acceptance Criteria:**
- [ ] ADR existiert, nummeriert (`0001-`), hat Status + Datum
- [ ] Alle drei AutoGen-Patterns (RoundRobin, Selector, Swarm) sind namentlich erwähnt + abgegrenzt
- [ ] "Consequences" enthält mindestens 3 ehrliche Trade-offs (negativ)
- [ ] Im Haupt-`README.md` unter "Architecture" verlinkt

---

### 8.3 WP4 — `cognithor-bench` Scaffold

**Ziel:** Eigenes Benchmark-Subpaket im Monorepo, reproduzierbare Agent-Benchmarks mit Cognithor. Struktur angelehnt an `agbench`, aber Cognithor-nativ.

**Scope-Check:** Nur Scaffold + Smoke-Test pre-v0.94.0. GAIA/WebArena-Integration ist post-v0.94.0.

**Deliverables:**

```
cognithor_bench/                                 # Monorepo-Submodul, eigenes pyproject
├── README.md
├── pyproject.toml                               # console-scripts: cognithor-bench
├── src/cognithor_bench/
│   ├── __init__.py
│   ├── cli.py                                   # argparse, run/tabulate
│   ├── runner.py                                # core loop, --native default
│   ├── reporter.py                              # markdown table
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base.py                              # Adapter Protocol
│   │   ├── cognithor_adapter.py                 # PGE-Agent wrapper, default
│   │   └── autogen_adapter.py                   # opt-in, ImportError-safe
│   └── scenarios/
│       └── smoke_test.jsonl                     # 3-5 Trivial-Tasks für CI
└── tests/
    ├── test_runner.py
    ├── test_cli.py
    ├── test_adapters.py
    └── fixtures/
```

**CLI-API:**

```bash
cognithor-bench run scenarios/smoke_test.jsonl
cognithor-bench run scenarios/smoke_test.jsonl --repeat 5
cognithor-bench run scenarios/smoke_test.jsonl --subsample 0.5
cognithor-bench run scenarios/smoke_test.jsonl --adapter cognithor    # default
cognithor-bench run scenarios/smoke_test.jsonl --adapter autogen      # opt-in (siehe WP2)
cognithor-bench run scenarios/smoke_test.jsonl --model ollama/qwen3:8b
cognithor-bench run scenarios/smoke_test.jsonl --docker               # opt-in, --native default
cognithor-bench tabulate results/
```

**Scenario-Format (JSONL):**

```json
{"id": "smoke-001", "task": "Was ist 2+2?", "expected": "4", "timeout_sec": 30, "requires": ["no_network"]}
```

**Wichtige Constraints:**
- **Docker-Isolation OPTIONAL** (`--docker` opt-in, `--native` default) — Cognithor läuft oft lokal auf Windows
- **Ollama-Support First-Class** — `--model ollama/...` muss out-of-the-box laufen
- **Keine Runtime-Dep auf `agbench`** — nur Referenz in Doku
- **`autogen_adapter.py`** importiert `autogen_agentchat` lazy + ImportError-safe; nutzt die zentrale Pin aus pyproject.toml `[autogen]` Extra (siehe F7)

**Acceptance Criteria:**
- [ ] `pip install -e ./cognithor_bench` installiert sauber
- [ ] `cognithor-bench --help` zeigt alle Kommandos
- [ ] `cognithor-bench run cognithor_bench/src/cognithor_bench/scenarios/smoke_test.jsonl` läuft (mit Mock-Adapter für CI)
- [ ] `cognithor-bench tabulate <results_dir>` erzeugt Markdown-Tabelle
- [ ] pytest-Coverage ≥80%
- [ ] README erklärt wie neue Scenarios angelegt werden

**Nicht-Ziel:**
- ❌ Integration von GAIA/WebArena/AssistantBench (post-v0.94.0, separates Ticket)
- ❌ Veröffentlichte Benchmark-Zahlen (erst nach verifizierten Runs)

---

### 8.4 WP2 — AutoGen-Compatibility-Shim

**Ziel:** `cognithor.compat.autogen` Submodul, Teilmenge der `autogen-agentchat`-API. Bestehender AutoGen-Code soll mit minimalen Änderungen (Search-&-Replace-Imports) auf Cognithor laufen.

**Scope (Hybrid-Approach per D4):**

| AutoGen-Klasse | Cognithor-Mapping | Anmerkung |
|---|---|---|
| `AssistantAgent.run(task=...)` | `cognithor.crew.Crew(agents=[a], tasks=[t]).kickoff_async()` | 1-shot, perfect match |
| `AssistantAgent.run_stream(...)` | gleicher Path + Event-Reshaping | Streaming-Events in AutoGen-Event-Shape |
| `RoundRobinGroupChat(...).run()` | **Eigener `_RoundRobinAdapter`** (~250-300 LOC) | Multi-Round-Loop, Termination-Conditions, Gatekeeper-wrapped |
| `MaxMessageTermination(N)` | counter im Adapter | trivial |
| `TextMentionTermination("DONE")` | regex-check über letztes raw output | trivial |
| Combined Conditions (`A & B`, `A \| B`) | Adapter unterstützt `__and__`, `__or__` | trivial Operator-Overload |
| Tools (FunctionTool/Workbench) | MCP-Registry via Coercion | bestehende Pipeline aus v0.93.0 |
| `OpenAIChatCompletionClient` | Wrapper auf `cognithor.core.model_router` | 16 Provider transparent |

**Explizit NICHT unterstützt (by design):**
- `SelectorGroupChat` — LLM-Selector konflikt mit Gatekeeper (siehe WP5 ADR)
- `Swarm` — HandoffMessage-Freiheit konflikt mit PGE-Trinity
- `MagenticOneGroupChat` — eigenes Pattern, separates Ticket
- `autogen-core`-Klassen (`RoutedAgent`, `@message_handler`, etc.)

**Verzeichnis-Struktur:**

```
src/cognithor/compat/
├── __init__.py
└── autogen/
    ├── __init__.py                              # Re-exports + DeprecationWarning bei Import
    ├── README.md                                # Migration-Guide mit Side-by-Side Diff
    ├── _bridge.py                               # interner Bridge auf cognithor.crew + PGE
    ├── _round_robin_adapter.py                  # Multi-Round-Loop, Hybrid-Path
    ├── agents/
    │   ├── __init__.py
    │   └── _assistant_agent.py                  # AssistantAgent mit exakter Signatur
    ├── teams/
    │   ├── __init__.py
    │   └── _round_robin.py                      # RoundRobinGroupChat → _round_robin_adapter
    ├── messages/
    │   └── __init__.py                          # TextMessage, HandoffMessage, ToolCallSummaryMessage
    ├── conditions/
    │   └── __init__.py                          # TextMentionTermination, MaxMessageTermination
    └── models/
        └── __init__.py                          # OpenAIChatCompletionClient → cognithor model_router

tests/test_compat/test_autogen/
├── test_signature_compat.py                     # inspect.signature parity (Stufe 1, D6)
├── test_hello_world_search_replace.py           # AutoGen README Hello-World, Mock-LLM (Stufe 2, D6)
├── test_assistant_agent.py
├── test_round_robin.py
├── test_round_robin_adapter.py                  # Multi-Round-Behavior
├── test_combined_terminations.py                # A & B, A | B
└── conftest.py
```

**`AssistantAgent`-Signatur (muss EXAKT spiegeln):**

```python
class AssistantAgent:
    """
    AutoGen-AgentChat-compatible AssistantAgent.

    Signature mirrors autogen_agentchat.agents.AssistantAgent (MIT-licensed).
    Internally delegates to Cognithor's PGE Executor via cognithor.crew.

    Reference: https://microsoft.github.io/autogen/stable//reference/python/autogen_agentchat.agents.html
    """

    def __init__(
        self,
        name: str,
        model_client: Any,
        *,
        tools: Sequence[Any] | None = None,
        workbench: Any | None = None,
        handoffs: list[Any] | None = None,
        model_context: Any | None = None,
        memory: Sequence[Any] | None = None,
        description: str = "An assistant agent.",
        system_message: str | None = None,
        model_client_stream: bool = False,
        reflect_on_tool_use: bool = False,
        tool_call_summary_format: str = "{result}",
        max_tool_iterations: int = 1,
        metadata: dict[str, str] | None = None,
    ) -> None: ...

    async def run(self, *, task: str | Sequence[Any]) -> "TaskResult": ...
    def run_stream(self, *, task: str | Sequence[Any]) -> AsyncIterator[Any]: ...
```

**Verhaltensgarantien:**
- Events von `run_stream` haben denselben Shape wie AutoGens Events: `source`, `models_usage`, `metadata`, `content`, `type`. Cognithor-untaubliche Felder dürfen `None` sein, müssen aber existieren
- Tool-Calls durchlaufen intern den **Gatekeeper**. Abgewiesener Tool-Call → `ToolCallSummaryMessage` mit Gatekeeper-Grund (KEIN Exception-Throw)
- `max_tool_iterations=1` Default (wie AutoGen)
- Import von `cognithor.compat.autogen` triggert eine `DeprecationWarning` mit Link auf Migration-Guide

**Test-Strategie (D6 Approach B):**

1. **Stufe 1 — Signatur-Tests** (`test_signature_compat.py`):
   - Pure `inspect.signature(autogen.AssistantAgent.__init__) == inspect.signature(cognithor.compat.autogen.AssistantAgent.__init__)` für die 14 Felder
   - Importiert nur in Test-Extra (`pip install cognithor[autogen]`), normale Devs brauchen das nicht
   - ~5 Tests, schnell

2. **Stufe 2 — Hello-World-Behavior-Test** (`test_hello_world_search_replace.py`):
   - Das offizielle AutoGen-Hello-World-Script (aus deren README mit `get_current_time`-Tool) läuft via Search-&-Replace-Imports
   - Mocked ModelClient produziert deterministische Outputs
   - Beide Implementierungen kriegen denselben Input, Output-Shape wird verglichen
   - ~7 Tests, ~1 Tag Schreiben

**Migrations-Doku (`src/cognithor/compat/autogen/README.md`):**

Pflicht-Inhalt:
1. Side-by-Side-Diff zwischen 30-Zeilen-AutoGen-Script und Cognithor-Compat-Äquivalent (oft nur Import-Path)
2. Liste der unterstützten / nicht-unterstützten Klassen
3. Warum SelectorGroupChat/Swarm absichtlich nicht unterstützt — Link auf WP5 ADR
4. "Ab wann solltest du raus aus der Compat-Layer?" — Empfehlung: nach Migrations-Stabilität auf native `cognithor.crew` wechseln

**Acceptance Criteria:**
- [ ] AutoGen-Hello-World-Script aus deren README läuft mit nur Import-Änderung (`from autogen_agentchat.agents import AssistantAgent` → `from cognithor.compat.autogen import AssistantAgent`)
- [ ] Tests mit ≥85% Coverage für `compat/autogen/`
- [ ] mypy --strict sauber
- [ ] DeprecationWarning bei Import zeigt Link auf Migration-Guide
- [ ] Keine Runtime-Abhängigkeit auf `autogen-*` Pakete; `autogen-agentchat==0.7.5` als `[autogen]`-Extra in pyproject.toml (NICHT in `[dev]`)
- [ ] `NOTICE` Datei vermerkt AutoGen-MIT-Inspiration (Section "Third-party attributions")
- [ ] Single-Pin-Point: pyproject.toml hat **eine** Stelle mit `autogen-agentchat==0.7.5`, von WP4-`autogen_adapter.py` referenziert (siehe F7)

**Risiken:**
- AutoGen könnte trotz Maintenance-Mode minor-API-Breaks einführen → Pin auf `==0.7.5`
- MIT-Compliance: AutoGen-Lizenzvermerk in `NOTICE` (analog CrewAI aus v0.93.0)

---

### 8.5 WP3 — Insurance Agent Pack (Standalone)

**Ziel:** Prominente, installierbare, konkrete Cognithor-Anwendung — analog zu Magentic-One's Rolle für AutoGen. Demonstriert PGE-Trinity als sichtbares Feature, zielt auf Alexander's WWK-Domäne (DACH-Versicherung).

**Reuse-Strategie (per F2):** WP3 baut auf v0.93.0's `versicherungs-vergleich` Template-Logik **konzeptionell** auf, ist aber **ein eigenständiges Python-Paket** (kein Template, keine Pack-System-Registrierung — siehe F1):

| v0.93.0 versicherungs-vergleich Template | WP3 Insurance Agent Pack |
|---|---|
| Tarif-Researcher | (analog) PolicyAnalyst — **erweitert um PDF-Tool-Use** |
| Kunden-Profiler | NeedsAssessor — gleiche Rolle, weniger Code-Duplication |
| Empfehlungs-Writer | ReportGenerator — gleiche Rolle |
| (kein dezidierter Compliance-Agent) | **NEU: ComplianceGatekeeper** — explizit der PGE-Gatekeeper als Demo |
| `chain(no_pii(), StringGuardrail)` | ✓ identisch verwendet |
| Sequential Process | ✓ identisch |
| Knowledge-Vault Seeds | **erweitert** (PKV/GGF/bAV/BU JSONL-Seeds neu) |

**WP3 baut also nicht das gleiche zweimal**, sondern fokussiert auf:
1. **PolicyAnalyst** (neu — PDF-Read mit Tool-Use)
2. **ComplianceGatekeeper als sichtbarer Demo-Agent** (neu — der Gatekeeper-Block ist die Marketing-Story)
3. **Knowledge-Vault-Seeds** (PKV/GGF/bAV/BU)
4. **Standalone-Installierbarkeit** (`pip install ./examples/insurance-agent-pack/`)
5. **CLI-Entry** (`insurance-agent-pack run --interview`)

**Pack-Registry-Form (per F1):**
- **NICHT** im `cognithor.packs` Loader-System (das ist für kommerzielle private Packs aus `cognithor-packs` Repo, mit EULA + license-key Validation)
- **JA** als pure `pip install ./examples/insurance-agent-pack/` — Public Reference-Implementation, Apache 2.0
- Kein `pack_manifest.json`, keine SHA-256-Sidecars, keine EULA-click-through

**Verzeichnis-Struktur:**

```
examples/insurance-agent-pack/
├── README.md                                    # Marketing + Walkthrough + asciinema-Link
├── LICENSE                                      # Apache 2.0 (verweist auf repo-root)
├── pyproject.toml                               # standalone, depends on cognithor>=0.94.0
├── src/insurance_agent_pack/
│   ├── __init__.py
│   ├── crew.py                                  # @agent decorators für 4 Rollen
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── policy_analyst.py                    # NEU vs versicherungs-vergleich
│   │   ├── needs_assessor.py                    # ähnlich Kunden-Profiler
│   │   ├── compliance_gatekeeper.py             # NEU — Gatekeeper-Demo
│   │   └── report_generator.py                  # ähnlich Empfehlungs-Writer
│   ├── prompts/                                 # DE-localized
│   │   ├── policy_analyst.md
│   │   ├── needs_assessor.md
│   │   ├── compliance_gatekeeper.md
│   │   └── report_generator.md
│   ├── knowledge/                               # Knowledge-Vault Seeds (synthetic)
│   │   ├── pkv_grundlagen.jsonl
│   │   ├── ggf_versorgung.jsonl
│   │   ├── bav_basics.jsonl
│   │   └── bu_grundlagen.jsonl
│   ├── tools/                                   # Custom @tool-decorated functions
│   │   └── pdf_extractor.py
│   └── cli.py                                   # argparse, --interview Modus
├── tests/
│   ├── test_team.py                             # 4-agent flow with mocked planner
│   ├── test_gatekeeper_blocks_legal_advice.py   # critical: positiv + negativ
│   ├── test_audit_chain_intact.py               # Hashline-Guard chain integrity
│   ├── test_local_inference_mode.py             # marked `slow`, ollama-only
│   └── fixtures/
│       └── sample_policy.pdf                    # synthetic, anonymized
└── docs/
    ├── demo_walkthrough.md
    ├── architecture.md                          # PGE-Trinity visibility diagram
    └── DISCLAIMER.md                            # ALEXANDER schreibt PERSÖNLICH (kein AI)
```

**Verhaltens-Tests (Pflicht):**

1. **`test_gatekeeper_blocks_legal_advice.py`:**
   - **Positiv:** "Welche Versicherungen gibt es für GGF?" → OK, Crew kickoff erfolgreich
   - **Negativ:** "Ist mein Arbeitsvertrag rechtens?" → Gatekeeper-Block mit klarer Meldung, keine Exception
2. **`test_audit_chain_intact.py`:** Nach vollständigem Run muss Hashline-Guard-Chain lückenlos sein (alle `crew_kickoff_started` haben matching `crew_kickoff_completed`)
3. **`test_local_inference_mode.py`:** Komplett ohne externe API-Calls (`OLLAMA_HOST=http://localhost:11434` only); markiert mit `@pytest.mark.slow` und in CI als optional

**Marketing-Hooks (nur in Doku):**
- `README.md` zeigt wie das Pack in `cognithor-bench` läuft (Verbindung zu WP4)
- Ein kurzes asciinema-Recording eines Interview-Flows verlinkt
- Insurance-spezifische Einordnung — kein generischer Demo, zielt auf Alexanders B2B-Zielgruppe (Mittelstand-Versicherungsmakler)

**Acceptance Criteria:**
- [ ] `pip install ./examples/insurance-agent-pack/` installiert Package sauber
- [ ] `insurance-agent-pack run --interview` startet Konsolen-Session, spielt vollständiges Needs-Assessment durch
- [ ] Alle 4 Agenten existieren und sind getestet
- [ ] Gatekeeper-Block-Test ist grün (positiv UND negativ)
- [ ] Läuft sowohl mit OpenAI-kompatiblem Remote-Modell als auch mit lokalem Ollama-Modell
- [ ] README verlinkt zurück zu Cognithor-Hauptrepo und WP5-ADR
- [ ] **Rechtlicher Disclaimer** prominent: "Demo-Pack, keine §34d-konforme Beratungssoftware". Alexander reviewt Wortlaut persönlich
- [ ] **Reuse-Note** in README dokumentiert: "Konzeptionelle Verwandtschaft mit `cognithor init --template versicherungs-vergleich` aus v0.93.0; WP3 fokussiert auf PolicyAnalyst-Tool-Use + ComplianceGatekeeper-Demo"

**Nicht-Ziel:**
- ❌ Kein Produkt — Referenz-Implementation
- ❌ Keine echten Kundendaten, niemals — alle Fixtures synthetic
- ❌ Keine Integration in Alexanders produktives WWK-Setup
- ❌ Keine Pack-System-Registrierung (das ist für private commerce-Packs)

---

### 8.6 v0.94.0 Release-Bundle (Direct Commit + Tag, kein PR)

**Pattern reuse:** Identisch zu v0.93.0 release-discipline. Nach merge von PR 4 (WP3):

**Step 1 — Version-Bump-Commit auf main (5 files):**

```
+ pyproject.toml                                    [project] version = "0.94.0"
+ src/cognithor/__init__.py                         __version__ = "0.94.0"
+ flutter_app/pubspec.yaml                          version: 0.94.0+1
+ flutter_app/lib/providers/connection_provider.dart kFrontendVersion = '0.94.0'
+ CHANGELOG.md                                      [Unreleased] → [0.94.0] -- 2026-MM-DD
```

Plus: `NOTICE` append AutoGen-MIT-Attribution Section (analog CrewAI).
Plus: `README.md` Highlights bullet für AutoGen-Compat + cognithor-bench + Insurance-Pack.

**Step 2 — Tag + Push:**

```bash
git tag -a v0.94.0 -m "Cognithor v0.94.0 — AutoGen Strategy Adoption"
git push origin v0.94.0
```

**Step 3 — Workflow-Triggers (parallel via REST API workflow_dispatch):**
- `publish.yml` (PyPI auto-publish)
- `build-windows-installer.yml`
- `build-deb.yml`
- `build-mobile.yml`
- `build-flutter-web.yml`

**Step 4 — Verify:**
- PyPI: https://pypi.org/project/cognithor/0.94.0/ live
- GitHub Release: 6 platform artifacts attached
- Stale-asset cleanup wenn nötig (lesson from v0.93.0)

---

## 9. Test-Strategie

| WP | Test-Location | Coverage-Target | Spezial |
|----|--------------|-----------------|---------|
| WP1 | n/a (docs) | n/a | Markdown-Lint optional |
| WP5 | n/a (docs) | n/a | — |
| WP4 | `cognithor_bench/tests/` | ≥80% | Mock-Adapter für CI |
| WP2 | `tests/test_compat/test_autogen/` | ≥85% | Signatur-Tests + Hello-World-Test (D6) |
| WP3 | `examples/insurance-agent-pack/tests/` | ≥80% | Gatekeeper-Block + Audit-Chain + Local-Inference |

**Pre-PR-Closeout Template** (analog v0.93.0):
- `pytest tests/ -x -q --cov=src/cognithor --cov-fail-under=89`
- `pytest --cov=cognithor.compat.autogen --cov-fail-under=85` (WP2-PR only)
- `ruff check .` + `ruff format --check .`
- `mypy --strict src/cognithor/compat` (WP2-PR only)
- Push + open PR + wait CI green + squash-merge + (separate turn) cleanup

---

## 10. Acceptance Criteria — Gesamt

v0.94.0 ist released wenn:

- [ ] Alle 4 PRs gemergt + cleanup done
- [ ] Direct-Commit-on-main mit Version-Bump in allen 5 Files
- [ ] Tag `v0.94.0` gepusht, alle 5 Release-Workflows grün
- [ ] PyPI hat `cognithor==0.94.0` (wheel + sdist)
- [ ] GitHub Release hat 6 fresh Artifacts (Windows Installer, Launcher, Linux .deb, Android APK, iOS IPA, Flutter Web)
- [ ] Keine stale Artifacts auf der Release
- [ ] `pip install cognithor==0.94.0` works
- [ ] `pip install cognithor[autogen]` installiert `autogen-agentchat==0.7.5` zum Test-Vergleich
- [ ] AutoGen-Hello-World aus deren README läuft via Search-&-Replace-Imports
- [ ] `cognithor-bench --help` works (nach `pip install -e ./cognithor_bench/`)
- [ ] `pip install ./examples/insurance-agent-pack/` works, `insurance-agent-pack run --interview` startet
- [ ] Alle 3 Doku-Files unter `docs/competitive-analysis/` existieren
- [ ] ADR `docs/adr/0001-pge-trinity-vs-group-chat.md` existiert
- [ ] CHANGELOG `[0.94.0]` Section hat alle 5 WPs gelistet
- [ ] `NOTICE` enthält AutoGen-MIT-Attribution

---

## 11. Sieben Design-Verbesserungen (eingearbeitet)

Aus dem 2026-04-25 Spec-Self-Review:

| # | Issue | Fix in Spec |
|---|-------|-------------|
| F1 | WP3 mixt Pack-System-Registrierung mit Public-Demo | §8.5: Pack-System NICHT genutzt; pure `pip install` |
| F2 | WP3 dupliziert versicherungs-vergleich Template | §8.5 Reuse-Tabelle: WP3 fokussiert auf PolicyAnalyst + ComplianceGatekeeper |
| F3 | PR 5 unnötig (10-Zeilen-Bump) | §7 + §8.6: 4 PRs + 1 Direct-Commit-on-main + Tag (v0.93.0-Pattern) |
| F4 | WP2 RoundRobin-Adapter undersized | §7 + §8.4: 6-8 Tage → 7-9 Tage (~250-300 LOC) |
| F5 | Pre-WP1 v0.93.0-Stabilität ungeprüft | §6: 24-48h v0.93.0-Stabilität-Gate vor PR 1 |
| F6 | pyproject `[autogen]` Extra fehlt | §8.4 Acceptance: `[project.optional-dependencies] autogen = ["autogen-agentchat==0.7.5"]` |
| F7 | autogen-Pin doppelt verwaltbar | §6: Single-Pin-Point in pyproject.toml; WP2 + WP4 referenzieren via Extra-Name |

---

## 12. Out-of-Scope (für v0.94.0)

- Keine Arbeiten an MAF-Kompatibilität (Graph-Modell konflikt mit PGE-Trinity)
- Keine Änderungen an PGE-Trinity selbst
- Keine Anpassungen am Cognithor-Release-Pipeline (re-uses v0.93.0)
- Keine Security-Audits des `agbench`-Docker-Isolation-Codes (wir nutzen `--native` Default)
- Keine öffentliche Kommunikation (Blog, Reddit, Discord-Posts) — Alexander entscheidet
- Keine Magentic-One-Reimplementation
- Keine GAIA/WebArena/AssistantBench-Integration in WP4 (post-v0.94.0)
- Keine veröffentlichten Performance-Zahlen (erst nach verifizierten Bench-Runs)

---

## 13. Sequencing & Dependencies

```
v0.93.0 (gestern released)
    ↓
[Pre-WP1 Gate: 24-48h Stabilität ohne v0.93.1-Hotfix]
    ↓
PR 1 (WP1+WP5 docs) — small, fast review
    ↓
PR 2 (WP4 cognithor-bench) — depends on nothing structural
    ↓
PR 3 (WP2 compat-shim) — depends on PR 2's autogen-Pin in pyproject.toml
    ↓
PR 4 (WP3 insurance-pack) — independent, but ships last for clean release
    ↓
Direct-Commit on main (version bump v0.93.0 → v0.94.0)
    ↓
Tag v0.94.0 + push → Release-Workflows
    ↓
v0.94.0 LIVE
```

**Hard Dependencies:**
- PR 3 (WP2) braucht den `[autogen]` Extra aus PR 2 (WP4) — sonst eigene Pin-Verwaltung
- PR 4 (WP3) braucht v0.93.0's `cognithor.crew` (bereits in main)
- Direct-Commit braucht alle 4 PRs gemergt + main CI grün

**Soft Dependencies (parallel-ok):**
- PR 1 (Doku) blockiert keinen Code-PR — könnte theoretisch nach PR 2/3 landen, ist aber als Marketing-Foundation zuerst sinnvoll

---

## 14. References

- Source-Prompt: `~/Downloads/cognithor_autogen_strategy_prompt.md`
- v0.93.0 Spec: `docs/superpowers/specs/2026-04-23-cognithor-crew-v1-adoption.md`
- v0.93.0 Plan: `docs/superpowers/plans/2026-04-24-cognithor-crew-v1.md`
- v0.93.0 Release-Pattern: commits `45abe90` → `b6e9230` → `dff283a` → `db41cde5` → `8d9c8e8f`
- AutoGen Maintenance-Hinweis: github.com/microsoft/autogen
- AutoGen `AssistantAgentConfig`: microsoft.github.io/autogen/stable/_modules/autogen_agentchat/agents/_assistant_agent.html
- Magentic-One Paper: arxiv.org/abs/2411.04468
- MAF Migration-Guide: learn.microsoft.com/en-us/agent-framework/migration-guide/from-autogen/

---

**End of Spec.**
