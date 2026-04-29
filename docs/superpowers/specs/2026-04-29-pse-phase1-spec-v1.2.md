# Cognithor Program Synthesis Engine — Phase 1 Spezifikation (MVP)

**Version:** 1.2.0-draft
**Stand:** April 2026
**Autor:** Alexander (Cognithor Lead)
**Lizenz:** Apache 2.0 (Cognithor-Standard)
**Ziel-Release:** Cognithor v0.80.0
**Implementierungsdauer:** 7 Wochen + 1,5 Tage (1 Solo-Entwickler mit AI-Assistenz)
**Changelog v1.1:** External Review (ChatGPT) übernommen — Higher-Order-Primitive als Phase 1.5 Pflichtteil, WSL2 als Windows-Default, Trace-KPIs als Hard-Gate, Roadmap auf 7 Wochen. Siehe Detail-Notes in den betroffenen Sektionen (§3, §7.5, §11.6, §21, §22).
**Changelog v1.2 (Patch):** Zweites External Review (Sanity-Check 10/10-ARC-Tasks) übernommen — `sort_objects` und `branch` als 4./5. Higher-Order-Primitiv ergänzt (§7.5), K3 ehrlich auf 30 s Default + 60 s Eval-Soft-Cap nachgeschärft (§3), Pattern Completion als expliziter Phase-2-Punkt (§23). Kein Major-Bump, da nur kompakter Patch ohne Architektur-Änderung.

---

## 0. Inhaltsverzeichnis

1. [Executive Summary](#1-executive-summary)
2. [Ziele & Non-Ziele](#2-ziele--non-ziele)
3. [Erfolgs-Kriterien (messbar)](#3-erfolgs-kriterien-messbar)
4. [Architektur-Übersicht Phase 1](#4-architektur-übersicht-phase-1)
5. [Verzeichnis- & Datei-Layout](#5-verzeichnis--datei-layout)
6. [Datenmodell & Typsystem](#6-datenmodell--typsystem)
7. [Die ARC-DSL: ~50 Primitive im Detail (+ Phase 1.5 Higher-Order)](#7-die-arc-dsl-50-primitive-im-detail)
8. [Enumerative Search-Algorithmus](#8-enumerative-search-algorithmus)
9. [Observational Equivalence Pruning](#9-observational-equivalence-pruning)
10. [Verifier-Pipeline](#10-verifier-pipeline)
11. [Execution-Sandbox](#11-execution-sandbox)
12. [Integration in PGE-Trinity](#12-integration-in-pge-trinity)
13. [Capability-Token & Hashline-Guard-Integration](#13-capability-token--hashline-guard-integration)
14. [Tactical-Memory-Integration](#14-tactical-memory-integration)
15. [State-Graph-Navigator-Brücke](#15-state-graph-navigator-brücke)
16. [Test-Strategie](#16-test-strategie)
17. [Telemetrie & Metriken](#17-telemetrie--metriken)
18. [Benchmark-Plan (ARC-AGI-3)](#18-benchmark-plan-arc-agi-3)
19. [Konfiguration & CLI](#19-konfiguration--cli)
20. [Risiken & Mitigationen](#20-risiken--mitigationen)
21. [Roadmap: Wochenplan 1–6](#21-roadmap-wochenplan-16)
22. [Akzeptanzkriterien (Definition of Done)](#22-akzeptanzkriterien-definition-of-done)
23. [Offene Fragen für Phase 2](#23-offene-fragen-für-phase-2)
24. [Anhang A: Beispiel-Programme](#24-anhang-a-beispiel-programme)
25. [Anhang B: Glossar](#25-anhang-b-glossar)
26. [Anhang C: Selbstprüfungs-Checkliste](#26-anhang-c-selbstprüfungs-checkliste)

---

## 1. Executive Summary

Diese Spezifikation beschreibt **Phase 1** der Cognithor Program Synthesis Engine (PSE) — ein neuer Channel, der Programme statt Antworten synthetisiert. Phase 1 liefert ein **funktionales MVP** mit:

- **ARC-DSL** mit ~50 typisierten Basis-Primitiven für Grid-Transformationen
- **Phase 1.5: 3 Higher-Order-Primitive** (`map_objects`, `filter_objects`, `align_to`) inkl. Predicate-Typsystem
- **Enumerative Bottom-Up-Suche** mit Observational-Equivalence-Pruning
- **Cost-Auto-Tuner** auf Benchmark-Daten (rein symbolisch, kein ML)
- **Mehrstufiger Verifier** (Syntax → Typ → Demo → Property → Held-Out)
- **Subprocess-Sandbox** mit Wall-Clock- und Memory-Limits, **WSL2-bevorzugt** unter Windows
- **Vollständige Programm-Traces** als Erstklass-Output (Differenzierungsmerkmal ggü. LLM-only)
- **Integration** in PGE-Trinity, Tactical Memory, Capability-Token-System
- **Brücke** zum bestehenden State-Graph-Navigator und NumPy-Solver

**Kein** LLM-Prior in Phase 1, **kein** MCTS, **kein** Library-Learning — diese Komponenten kommen in Phase 2 / 3. Der MVP ist bewusst rein symbolisch, damit Korrektheit und Performance des Kerns isoliert validierbar sind, bevor neuronale Komponenten als Heuristik addiert werden.

**Differenzierungsmerkmal:** Anders als reine LLM-Lösungen liefert PSE für jede gelöste Task einen **deterministischen, replay-baren, menschenlesbaren Programm-Trace**. Dieser Trace ist ein Erstklass-KPI (siehe K9, K10), nicht nur Nebenprodukt. Er ist die Basis für späteres Library-Learning, Audit-Trail und Erklärbarkeit gegenüber Endnutzern.

**Ziel-Output:** messbare Verbesserung des ARC-AGI-3-Scores ggü. dem aktuellen NumPy-Solver auf mindestens einer Task-Klasse, bei nachvollziehbarer Korrektheit (jedes gelöste Beispiel mit vollständigem Programm-Trace).

**Was sich gegenüber v1.0 dieser Spec geändert hat (Auslöser: External Review):**
1. Phase 1.5 (Higher-Order-Primitive) ist jetzt **Pflichtteil**, nicht optionaler Puffer.
2. Predicate/Lambda-Typsystem ist neu in §6.4 spezifiziert.
3. Cost-Auto-Tuner als kleiner deterministischer Mechanismus eingeführt (§7.6).
4. Windows-Sandbox: WSL2-Worker als Default-Strategie (§11.6); reines Windows nur „Research-Mode" mit Warnung.
5. Erfolgs-Kriterien um Trace-KPIs K9, K10 erweitert (§3); D16 als Release-Hard-Gate (§22).
6. Roadmap von 6 auf 7 Wochen, Woche 3 von 5 auf 8 Tage gestreckt (§21).

---

## 2. Ziele & Non-Ziele

### 2.1 Ziele (Phase 1)

- **Z1** Funktionsfähige Enumerative-Search-Engine, die für ARC-AGI-3-Trainingstasks Programme der Tiefe 1–4 in unter 30 s findet, falls existent.
- **Z2** Vollständige ARC-DSL mit ~50 Basis-Primitiven plus 3 Higher-Order-Primitiven (Phase 1.5), dokumentiert, getestet, typisiert.
- **Z3** Sichere Sandbox: kein DSL-Programm darf den Cognithor-Hauptprozess beeinflussen können (kein FS-, Netzwerk-, oder Memory-Leak). Unter Windows: WSL2-Worker als Default.
- **Z4** Saubere Integration in PGE-Trinity: Planner kann PSE als Tool aufrufen, Gatekeeper validiert, Executor führt aus.
- **Z5** Reproduzierbare Benchmarks gegen ARC-AGI-3-Trainings-Subset (mindestens 100 Tasks).
- **Z6** Test-Coverage ≥ 90 % auf neuem Code (passend zu Cognithor-Standard 89 %).
- **Z7** Vollständige Markdown-Dokumentation pro Modul (Cognithor-Stil).
- **Z8** **Trace-First-Architektur:** Jedes gelöste Programm liefert einen vollständigen, deterministischen, menschenlesbaren Pseudo-Code-Trace. Trace-Erzeugung ist nicht optional, sondern Erstklass-Output.

### 2.2 Non-Ziele (explizit ausgeschlossen)

- **N1** Kein LLM-Prior / Neuro-symbolische Suche → Phase 2.
- **N2** Kein MCTS → Phase 2.
- **N3** Kein Library Learning / DreamCoder-Style → Phase 3.
- **N4** Keine zusätzlichen DSLs (nur ARC-DSL) → Phase 4.
- **N5** Kein Counter-Example-Guided-Refinement (CEGIS) → Phase 2.
- **N6** Keine Programmsynthese für Code (nur Grids) → Phase 4.
- **N7** Kein gelernter (ML-basierter) Cost-Tuner — nur deterministischer Auto-Tuner auf Benchmark-Daten (siehe §7.6); voller ML-Tuner in Phase 3.

Diese Trennung ist zentral: Phase 1 muss **eigenständig wertvoll** sein und einen klaren Score-Sprung liefern, bevor komplexere Komponenten dazukommen.

---

## 3. Erfolgs-Kriterien (messbar)

| # | Kriterium | Schwelle | Mess-Methode |
|---|-----------|----------|--------------|
| K1 | ARC-AGI-3-Score-Sprung | ≥ +5 Tasks ggü. aktuellem NumPy-Solver | Eval-Suite, identische Hardware |
| K2 | Suchzeit Median (Tiefe ≤ 3) | ≤ 5 s pro Task | `pytest-benchmark` |
| K3 | Suchzeit P95 (Tiefe ≤ 4) | ≤ 30 s Default-Budget; **Eval-Suite mit Soft-Cap 60 s zur ehrlichen Coverage-Messung** | `pytest-benchmark` |
| K4 | Sandbox-Escape-Versuche blockiert | 100 % | Adversarial-Test-Suite (≥ 20 Cases) |
| K5 | False-Positive-Rate (Programm passt auf Demos, falsch auf Test) | ≤ 15 % | Held-Out-Split |
| K6 | Test-Coverage neuer Code | ≥ 90 % | `pytest --cov` |
| K7 | Speicher-Footprint pro Suche | ≤ 1 GB RAM | `tracemalloc` |
| K8 | Cache-Hit-Rate auf wiederholten Tasks | ≥ 80 % | Telemetrie |
| **K9** | **Trace-Vollständigkeit:** Jedes Solved-Programm hat menschenlesbaren Pseudo-Code-Output mit allen Zwischenschritten | **100 %** | **Auto-Test über Eval-Suite** |
| **K10** | **Programm-Replay-Reproduzierbarkeit:** Output bei Re-Execution identisch + < 100 ms | **100 % (identisch); P95 ≤ 100 ms** | **`replay_test.py` über alle Solved-Programme** |

**Neu in v1.1:** K9 und K10 quantifizieren das Differenzierungsmerkmal *Erklärbarkeit*. Ein Programm ohne reproduzierbaren, lesbaren Trace zählt nicht als „solved", auch wenn die Output-Grids stimmen.

**Neu in v1.2 (Honesty-Update K3):** Tiefe-4-Programme mit Higher-Order — typisch Bounding-Box-Extraction oder kombinierte Object-Pipelines — liegen ohne LLM-Prior empirisch im Bereich 10–25 s, vereinzelt auch bis 45 s. Wir trennen daher zwei Budgets:
- **Production-Default:** 30 s P95. Alles, was länger braucht, gilt im Produktionsbetrieb als „nicht gefunden".
- **Eval-Suite Soft-Cap:** 60 s P95. Damit messen wir die *tatsächliche* DSL-Coverage ehrlich, ohne Suchzeit als künstlichen Confounder. Tasks, die zwischen 30 s und 60 s gefunden werden, gehen als „Coverage-positiv, Performance-negativ" in den Benchmark-Bericht — Ziel ist klar: Phase 2 (LLM-Prior) muss diese Klasse unter 30 s drücken.

**Fail-Kriterien (Spec-Rejection):**
- Wenn K1 nicht erreicht → Phase 1 wird verlängert, nicht in Phase 2 übergeleitet.
- Wenn K4 < 100 % → Release blockiert (Security-Hard-Gate).
- Wenn K9 < 100 % → Release blockiert (Trace-Hard-Gate, neu in v1.1).
- Wenn K10 nicht reproduzierbar (Output-Drift bei Replay) → Release blockiert (Determinismus-Hard-Gate, neu in v1.1).

---

## 4. Architektur-Übersicht Phase 1

```
┌─────────────────────────────────────────────────────────────┐
│                    PGE TRINITY (bestehend)                  │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐        │
│  │ Planner  │───▶│  Gatekeeper  │───▶│  Executor   │        │
│  └──────────┘    └──────────────┘    └─────────────┘        │
└──────────┬──────────────┬───────────────────┬───────────────┘
           │              │                   │
           ▼              ▼                   ▼
   ┌───────────────────────────────────────────────────┐
   │       PROGRAM SYNTHESIS CHANNEL (NEU)             │
   │  ┌─────────────────────────────────────────────┐  │
   │  │ 1. TaskSpec-Builder                         │  │
   │  │    (input: ARC-AGI-3 Task → TaskSpec)       │  │
   │  └────────────────────┬────────────────────────┘  │
   │                       ▼                           │
   │  ┌─────────────────────────────────────────────┐  │
   │  │ 2. Cache-Lookup (Tactical Memory)           │  │
   │  │    Key: TaskSpec-Hash + DSL-Version         │  │
   │  └────────────────────┬────────────────────────┘  │
   │                       ▼ (miss)                    │
   │  ┌─────────────────────────────────────────────┐  │
   │  │ 3. Enumerative Search Engine                │  │
   │  │    Bottom-Up bis max_depth                  │  │
   │  │    + Observational Equivalence Pruning      │  │
   │  └────────────────────┬────────────────────────┘  │
   │                       ▼                           │
   │  ┌─────────────────────────────────────────────┐  │
   │  │ 4. Sandbox Executor                         │  │
   │  │    (subprocess, resource-limited)           │  │
   │  └────────────────────┬────────────────────────┘  │
   │                       ▼                           │
   │  ┌─────────────────────────────────────────────┐  │
   │  │ 5. Verifier (5-stage)                       │  │
   │  │    Syntax→Type→Demo→Property→HeldOut        │  │
   │  └────────────────────┬────────────────────────┘  │
   │                       ▼                           │
   │  ┌─────────────────────────────────────────────┐  │
   │  │ 6. Result-Builder + Cache-Write             │  │
   │  └─────────────────────────────────────────────┘  │
   └───────────────────────────────────────────────────┘
                          │
                          ▼
   ┌───────────────────────────────────────────────────┐
   │  Bestehende Komponenten (read-only Integration):  │
   │   • ARC-AGI-3 NumPy-Solver  (Fast-Path Fallback)  │
   │   • State Graph Navigator   (Spec-Annotation)     │
   │   • Knowledge Vault         (DSL-Definition)      │
   │   • Hashline Guard          (Sandbox-Policy)      │
   └───────────────────────────────────────────────────┘
```

**Datenfluss (Happy Path):**
1. Planner erhält ARC-AGI-3-Task → klassifiziert als „synthesizable"
2. Planner erstellt `SynthesisRequest` mit TaskSpec
3. Gatekeeper prüft Capability-Token (`pse:synthesize`)
4. Channel führt Cache-Lookup durch
5. Bei Miss: Enumerative Search startet
6. Hypothesen werden in Sandbox verifiziert
7. Erstes vollständig korrektes Programm gewinnt (Occam-Tiebreaker bei Ties)
8. Result wird in Tactical Memory gespeichert
9. Executor wendet Programm auf Test-Input an
10. Telemetrie wird ans Cognithor-Observability-System gesendet

---

## 5. Verzeichnis- & Datei-Layout

Die Engine liegt unter `cognithor/channels/program_synthesis/`. Alle Pfade sind relativ zum Cognithor-Repo-Root.

```
cognithor/
└── channels/
    └── program_synthesis/
        ├── __init__.py                  # Public API (Channel-Registration)
        ├── README.md                    # Channel-Doku (DE/EN)
        ├── CHANGELOG.md
        │
        ├── core/
        │   ├── __init__.py
        │   ├── types.py                 # TaskSpec, Program, SynthesisResult, Budget
        │   ├── exceptions.py            # PSEError-Hierarchie
        │   └── version.py               # DSL-Version (für Cache-Invalidation)
        │
        ├── dsl/
        │   ├── __init__.py
        │   ├── primitives.py            # Alle ~50 Primitive
        │   ├── types_grid.py            # Grid, Color, Object, Mask
        │   ├── registry.py              # PrimitiveRegistry (Singleton)
        │   ├── signatures.py            # TypeSignature, type-checking
        │   └── catalog.json             # Versionierte DSL-Definition
        │
        ├── search/
        │   ├── __init__.py
        │   ├── enumerative.py           # Bottom-Up-Search (Hauptklasse)
        │   ├── equivalence.py           # ObservationalEquivalencePruner
        │   ├── candidate.py             # ProgramCandidate, Tree-Repräsentation
        │   ├── budget.py                # Budget-Allokator
        │   └── strategies.py            # SearchStrategy-Protocol (für Phase 2)
        │
        ├── verify/
        │   ├── __init__.py
        │   ├── pipeline.py              # Verifier (orchestriert Stages)
        │   ├── stages.py                # SyntaxStage, TypeStage, DemoStage, ...
        │   ├── properties.py            # Property-Tests (Größe, Farben, ...)
        │   └── result.py                # VerificationResult
        │
        ├── sandbox/
        │   ├── __init__.py
        │   ├── executor.py              # SandboxExecutor (subprocess-basiert)
        │   ├── worker.py                # Subprocess-Worker (entry point)
        │   ├── policy.py                # Resource-Limits, AST-Whitelist
        │   └── ipc.py                   # Pickle-basiertes IPC mit Timeout
        │
        ├── integration/
        │   ├── __init__.py
        │   ├── pge_adapter.py           # Bridge zu Planner/Gatekeeper/Executor
        │   ├── tactical_memory.py       # Cache-Layer auf Tactical Memory
        │   ├── capability_tokens.py     # Token-Definitionen
        │   ├── state_graph_bridge.py    # SGN → TaskSpec-Annotationen
        │   └── numpy_solver_bridge.py   # Fallback / Fast-Path
        │
        ├── observability/
        │   ├── __init__.py
        │   ├── metrics.py               # Prometheus-Style Counter/Histogram
        │   ├── tracing.py               # OpenTelemetry-Span-Helpers
        │   └── logger.py                # strukturierter JSON-Logger
        │
        ├── cli/
        │   ├── __init__.py
        │   └── pse_cli.py               # `cognithor pse <task>` CLI
        │
        ├── config/
        │   └── default.yaml             # Default-Konfiguration
        │
        └── data/
            └── arc_agi3_train_subset/   # Symlink zu Eval-Daten

tests/
└── channels/
    └── program_synthesis/
        ├── __init__.py
        ├── conftest.py                  # Fixtures
        ├── unit/
        │   ├── test_types.py
        │   ├── test_dsl_primitives.py   # Pro Primitiv ≥ 3 Tests
        │   ├── test_registry.py
        │   ├── test_signatures.py
        │   ├── test_enumerative.py
        │   ├── test_equivalence.py
        │   ├── test_verifier_stages.py
        │   ├── test_properties.py
        │   ├── test_sandbox_policy.py
        │   └── test_budget.py
        ├── integration/
        │   ├── test_pge_adapter.py
        │   ├── test_tactical_cache.py
        │   ├── test_state_graph_bridge.py
        │   └── test_full_pipeline.py
        ├── security/
        │   ├── test_sandbox_escape.py   # Adversarial Cases
        │   ├── test_resource_limits.py
        │   └── test_capability_tokens.py
        ├── eval/
        │   ├── test_arc_agi3_subset.py  # Großer Benchmark-Test
        │   └── fixtures/
        │       └── known_solutions.json
        └── property/
            └── test_hypothesis_dsl.py   # Hypothesis-basierte Tests

docs/
└── channels/
    └── program_synthesis/
        ├── overview.md
        ├── dsl_reference.md             # Auto-generiert aus catalog.json
        ├── architecture.md
        ├── benchmarks.md
        └── tutorial.md                  # Hello-World für Contributoren
```

**Begründung zur Struktur:**
- **Trennung `core/dsl/search/verify/sandbox`**: jede Schicht ist isoliert testbar und in Phase 2 austauschbar (z. B. neue Search-Strategie, ohne DSL anzufassen).
- **`integration/`**: alle Cognithor-Berührungspunkte an *einem* Ort — wenn sich PGE-Trinity ändert, wissen wir sofort, wo zu patchen ist.
- **`observability/`**: separat, weil die Engine Telemetrie ohne Cognithor-Kern auch standalone (z. B. in Tests) emittieren können muss.
- **`cli/`**: ermöglicht Debugging einzelner Tasks ohne kompletten Cognithor-Boot.

---

## 6. Datenmodell & Typsystem

Alle Typen sind unveränderlich (`@dataclass(frozen=True)` oder `pydantic.BaseModel` mit `model_config = ConfigDict(frozen=True)`), Python 3.12+, vollständig getypt. **Keine** mutable shared state.

### 6.1 Grundtypen (`core/types.py`)

```python
"""
Cognithor Program Synthesis Engine — Core Types.

Alle Datenstrukturen sind frozen/immutable. Mutation ist verboten.
Equality ist strukturell. Hashing wird für Cache-Keys verwendet.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol, TypeAlias
from enum import Enum
import numpy as np
from numpy.typing import NDArray

# Grid-Typ: 2D-Numpy-Array mit Werten 0..9 (ARC-Konvention)
Grid: TypeAlias = NDArray[np.int8]

# Farbe: 0..9, wobei 0 traditionell "Hintergrund" ist
Color: TypeAlias = int

# Beispiel-Paar: (input, output)
Example: TypeAlias = tuple[Grid, Grid]


class TaskDomain(str, Enum):
    ARC_AGI_3 = "arc_agi_3"
    ARC_AGI_2 = "arc_agi_2"
    SYNTHETIC = "synthetic"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Constraint:
    """Eine Constraint, die ein gültiges Programm erfüllen muss."""
    kind: Literal["size_preserving", "color_preserving", "monotonic_size", "custom"]
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskSpec:
    """
    Eindeutige Beschreibung einer Synthese-Aufgabe.

    Wird via stable_hash() zum Cache-Key.
    """
    examples: tuple[Example, ...]
    held_out: tuple[Example, ...] = ()      # für False-Positive-Erkennung
    test_input: Grid | None = None          # eigentliches Test-Grid
    constraints: tuple[Constraint, ...] = ()
    domain: TaskDomain = TaskDomain.ARC_AGI_3
    annotations: dict[str, Any] = field(default_factory=dict)

    def stable_hash(self) -> str:
        """SHA-256 über kanonische Serialisierung. Cache-Key."""
        ...


@dataclass(frozen=True)
class Budget:
    """
    Compute-Budget für eine Synthese-Anfrage. Hard Limits.
    """
    max_depth: int = 4                  # max. Programmtiefe
    max_candidates: int = 50_000        # max. enumerierte Programme
    wall_clock_seconds: float = 30.0    # Hard-Timeout
    max_memory_mb: int = 1024           # Memory-Hard-Limit
    per_candidate_ms: int = 100         # Timeout pro Hypothese-Execution
    cache_lookup: bool = True           # Tactical Memory abfragen?


class SynthesisStatus(str, Enum):
    SUCCESS = "success"               # vollständig korrekt
    PARTIAL = "partial"               # einige Demos korrekt
    NO_SOLUTION = "no_solution"       # nichts in Suchraum
    TIMEOUT = "timeout"
    BUDGET_EXCEEDED = "budget"
    SANDBOX_VIOLATION = "sandbox"
    ERROR = "error"


@dataclass(frozen=True)
class StageResult:
    stage: Literal["syntax", "type", "demo", "property", "held_out"]
    passed: bool
    detail: str = ""
    duration_ms: float = 0.0


@dataclass(frozen=True)
class SynthesisResult:
    status: SynthesisStatus
    program: "Program | None"
    score: float                              # 0..1, Anteil korrekter Demos
    confidence: float                         # 0..1, kalibriert via Held-Out
    cost_seconds: float
    cost_candidates: int
    verifier_trace: tuple[StageResult, ...]
    cache_hit: bool = False
    annotations: dict[str, Any] = field(default_factory=dict)
```

### 6.2 Programm-Repräsentation (`search/candidate.py`)

```python
@dataclass(frozen=True)
class Program:
    """
    Ein Programm ist ein typisierter Baum aus DSL-Primitiven.

    Beispiel: rotate90(recolor(input, 1, 2))
       → Program(
            primitive="rotate90",
            children=(
              Program(primitive="recolor",
                      children=(InputRef(), Const(1), Const(2))),
            )
          )
    """
    primitive: str                          # Name aus DSL-Registry
    children: tuple["ProgramNode", ...]
    output_type: str                        # "Grid", "Color", "Mask", ...

    def depth(self) -> int: ...
    def size(self) -> int: ...              # Anzahl Knoten
    def to_source(self) -> str: ...         # Lesbare Form
    def stable_hash(self) -> str: ...

@dataclass(frozen=True)
class InputRef:
    """Referenz auf das Input-Grid der TaskSpec."""
    output_type: str = "Grid"

@dataclass(frozen=True)
class Const:
    """Konstanter Wert (Color, Int)."""
    value: int | str
    output_type: str

ProgramNode: TypeAlias = Program | InputRef | Const
```

**Designentscheidungen:**
- **`tuple` statt `list`** für `children` → hashbar, frozen-konform.
- **`output_type` als String, nicht `type`** → JSON-serialisierbar für Cache und Logs.
- **`stable_hash`** via SHA-256 über kanonisches String-Form, nicht Python-`hash()` → reproduzierbar prozess-übergreifend.

### 6.3 Exception-Hierarchie (`core/exceptions.py`)

```python
class PSEError(Exception):
    """Basisklasse aller Engine-Fehler."""

class DSLError(PSEError):
    """Fehler im DSL-Bereich (unbekanntes Primitiv, Typ-Mismatch)."""

class TypeMismatchError(DSLError): ...
class UnknownPrimitiveError(DSLError): ...

class SearchError(PSEError): ...
class BudgetExceededError(SearchError): ...
class NoSolutionError(SearchError): ...

class SandboxError(PSEError):
    """Fehler im Sandbox-Bereich. Hard-Fail."""

class SandboxViolationError(SandboxError):
    """Code hat Sandbox-Policy verletzt. Hochsicherheits-Log."""

class SandboxTimeoutError(SandboxError): ...
class SandboxOOMError(SandboxError): ...

class VerificationError(PSEError): ...
```

### 6.4 Predicate-Typsystem (Phase 1.5, neu in v1.1)

Higher-Order-Primitive (`map_objects`, `filter_objects`) brauchen **Lambda-/Predicate-Werte** als Argumente. Da Phase 1 bewusst kein freies Python-Lambda zulässt (Sandbox!), führen wir ein **enumerierbares, geschlossenes Predicate-System** ein.

**Designprinzip:** Ein Predicate ist *kein* Python-Lambda, sondern ein **typisierter, registrierter, enumerationsfähiger DSL-Term**, der wie jedes andere Primitiv von der Search-Engine konstruiert werden kann.

```python
@dataclass(frozen=True)
class Predicate:
    """
    Ein geschlossenes Predicate über einem Domain-Typ.

    Phase 1.5: Nur eine fest definierte Menge von Predicate-Konstruktoren.
    Keine freien Lambdas. Predicate-Bäume werden enumeriert wie normale Programme.
    """
    constructor: str               # z.B. "color_eq", "size_gt", "is_rectangle"
    args: tuple[Any, ...]          # nur primitive Werte (int, str, Color)
    domain: str                    # z.B. "Object", "Mask", "Cell"
    output_type: str = "Bool"      # immer Bool

    def stable_hash(self) -> str: ...
    def to_source(self) -> str: ...

@dataclass(frozen=True)
class Lambda:
    """
    Eine 1-stellige Funktion Object → Object oder Object → T.

    Wie Predicate enumerationsfähig: nur registrierte Constructors,
    keine freien Python-Funktionen.
    """
    body: ProgramNode              # Programm-Baum, der eine bound Variable enthält
    variable_type: str             # Typ der bound Variable
    output_type: str
```

**Erlaubte Predicate-Konstruktoren in Phase 1.5 (geschlossene Menge):**

| Konstruktor | Domain | Argumente | Bedeutung |
|-------------|--------|-----------|-----------|
| `color_eq` | Object | Color | Objekt hat genau diese Farbe |
| `color_in` | Object | tuple[Color, ...] | Objekt hat eine der Farben |
| `size_eq` | Object | Int | Objekt hat genau N Pixel |
| `size_gt` | Object | Int | Objekt hat mehr als N Pixel |
| `size_lt` | Object | Int | Objekt hat weniger als N Pixel |
| `is_rectangle` | Object | — | Objekt ist rechteckig |
| `is_square` | Object | — | Objekt ist quadratisch |
| `is_largest_in` | Object | ObjectSet | Größtes Objekt im Set |
| `is_smallest_in` | Object | ObjectSet | Kleinstes Objekt im Set |
| `touches_border` | Object | — | Berührt Grid-Rand |
| `not` | * | Predicate | Negation |
| `and` | * | Predicate, Predicate | Konjunktion |
| `or` | * | Predicate, Predicate | Disjunktion |

**Begründung der Geschlossenheit:** Jeder freie Lambda-Konstrukt würde die Sandbox-Garantie unterlaufen — der Such-Mechanismus könnte beliebigen Code generieren. Eine geschlossene Predicate-Menge ist enumerierbar, statisch typisierbar, sandbox-sicher und für 80–90 % der ARC-Patterns ausreichend.

**Erweiterung in späteren Phasen:** Phase 2 erweitert um arithmetische Predicate-Constructors (`coordinate_lt`, `position_relative`), Phase 3 öffnet via `pse:dsl:extend`-Capability für Custom-Predicates mit eigenem Audit-Trail.

---

## 7. Die ARC-DSL: ~50 Primitive im Detail

Die DSL ist **orientiert an Hodels arc-dsl** (öffentliches ARC-Solving-Vokabular), aber **typisiert** und **erweitert** um Cognithor-spezifische Primitive (z. B. State-Graph-Hooks).

Jedes Primitiv hat:
- **Name** (Identifier, snake_case)
- **Signatur** (Eingabe-Typen → Ausgabe-Typ)
- **Cost** (für Occam-Prior, niedriger = einfacher)
- **Implementation** (reine Funktion, keine Seiteneffekte)
- **Tests** (≥ 3 pro Primitiv)
- **Beispiel** in der Doku

### 7.1 Typisches Primitiv-Beispiel

```python
@primitive(
    name="rotate90",
    signature=Signature(inputs=("Grid",), output="Grid"),
    cost=1.0,
    description="Rotiert das Grid um 90° im Uhrzeigersinn.",
    examples=[
        ("[[1,2],[3,4]]", "[[3,1],[4,2]]"),
    ],
)
def rotate90(grid: Grid) -> Grid:
    return np.rot90(grid, k=-1).copy()
```

Der `@primitive`-Decorator registriert die Funktion in der `PrimitiveRegistry` und prüft Signatur-Konsistenz beim Modul-Load.

### 7.2 Vollständiger Katalog (Phase-1-Ziel: 50 Primitive)

| # | Name | Signatur | Cost | Kategorie |
|---|------|----------|------|-----------|
| **Geometrisch** ||||
| 1 | `identity` | `Grid → Grid` | 0.1 | Basis |
| 2 | `rotate90` | `Grid → Grid` | 1.0 | Geom |
| 3 | `rotate180` | `Grid → Grid` | 1.0 | Geom |
| 4 | `rotate270` | `Grid → Grid` | 1.0 | Geom |
| 5 | `mirror_horizontal` | `Grid → Grid` | 1.0 | Geom |
| 6 | `mirror_vertical` | `Grid → Grid` | 1.0 | Geom |
| 7 | `transpose` | `Grid → Grid` | 1.0 | Geom |
| 8 | `mirror_diagonal` | `Grid → Grid` | 1.2 | Geom |
| 9 | `mirror_antidiagonal` | `Grid → Grid` | 1.2 | Geom |
| **Farbe** ||||
| 10 | `recolor` | `Grid, Color, Color → Grid` | 1.5 | Farbe |
| 11 | `swap_colors` | `Grid, Color, Color → Grid` | 1.5 | Farbe |
| 12 | `most_common_color` | `Grid → Color` | 1.0 | Farbe |
| 13 | `least_common_color` | `Grid → Color` | 1.0 | Farbe |
| 14 | `color_count` | `Grid → Int` | 1.0 | Farbe |
| 15 | `replace_background` | `Grid, Color → Grid` | 1.5 | Farbe |
| **Größe / Skalierung** ||||
| 16 | `scale_up_2x` | `Grid → Grid` | 2.0 | Skala |
| 17 | `scale_up_3x` | `Grid → Grid` | 2.0 | Skala |
| 18 | `scale_down_2x` | `Grid → Grid` | 2.0 | Skala |
| 19 | `tile_2x` | `Grid → Grid` | 2.0 | Skala |
| 20 | `crop_bbox` | `Grid → Grid` | 1.5 | Skala |
| 21 | `pad_with` | `Grid, Color, Int → Grid` | 1.8 | Skala |
| **Räumlich (Gravity / Shift)** ||||
| 22 | `gravity_down` | `Grid → Grid` | 2.0 | Räuml. |
| 23 | `gravity_up` | `Grid → Grid` | 2.0 | Räuml. |
| 24 | `gravity_left` | `Grid → Grid` | 2.0 | Räuml. |
| 25 | `gravity_right` | `Grid → Grid` | 2.0 | Räuml. |
| 26 | `shift` | `Grid, Int, Int → Grid` | 2.0 | Räuml. |
| 27 | `wrap_shift` | `Grid, Int, Int → Grid` | 2.2 | Räuml. |
| **Objekt-Detektion** ||||
| 28 | `connected_components_4` | `Grid → ObjectSet` | 2.5 | Objekt |
| 29 | `connected_components_8` | `Grid → ObjectSet` | 2.5 | Objekt |
| 30 | `objects_of_color` | `Grid, Color → ObjectSet` | 2.0 | Objekt |
| 31 | `largest_object` | `ObjectSet → Object` | 1.5 | Objekt |
| 32 | `smallest_object` | `ObjectSet → Object` | 1.5 | Objekt |
| 33 | `bounding_box` | `Object → Grid` | 1.5 | Objekt |
| 34 | `object_count` | `ObjectSet → Int` | 1.0 | Objekt |
| 35 | `render_objects` | `ObjectSet, Grid → Grid` | 2.0 | Objekt |
| **Maske / Logik** ||||
| 36 | `mask_eq` | `Grid, Color → Mask` | 1.5 | Mask |
| 37 | `mask_ne` | `Grid, Color → Mask` | 1.5 | Mask |
| 38 | `mask_apply` | `Grid, Mask, Color → Grid` | 2.0 | Mask |
| 39 | `mask_and` | `Mask, Mask → Mask` | 1.5 | Logik |
| 40 | `mask_or` | `Mask, Mask → Mask` | 1.5 | Logik |
| 41 | `mask_xor` | `Mask, Mask → Mask` | 1.5 | Logik |
| 42 | `mask_not` | `Mask → Mask` | 1.2 | Logik |
| **Konstruktion / Komposition** ||||
| 43 | `stack_horizontal` | `Grid, Grid → Grid` | 2.0 | Konstr. |
| 44 | `stack_vertical` | `Grid, Grid → Grid` | 2.0 | Konstr. |
| 45 | `overlay` | `Grid, Grid, Color → Grid` | 2.5 | Konstr. |
| 46 | `frame` | `Grid, Color → Grid` | 1.8 | Konstr. |
| **Konstanten** ||||
| 47 | `const_color_0` ... `const_color_9` | `→ Color` | 0.5 | Const |

> **Hinweis:** Konstanten werden als 10 separate Primitive registriert, sind aber sehr billig.

**Stand: 46 funktionale + 10 Color-Konstanten = 56 Primitive.** Liegt im Zielbereich „~50".

### 7.3 DSL-Versionierung

Jede Änderung am DSL-Katalog ändert die DSL-Version (semver):
- **Major:** Primitiv entfernt oder Signatur geändert (Cache-Bruch).
- **Minor:** Primitiv hinzugefügt (Cache-kompatibel).
- **Patch:** Cost-Anpassung, Doku.

DSL-Version geht in jeden Cache-Key ein → automatische Invalidation.

### 7.4 Typsystem

```python
@dataclass(frozen=True)
class Signature:
    inputs: tuple[str, ...]      # ("Grid", "Color")
    output: str                  # "Grid"

    def matches(self, args: tuple[str, ...]) -> bool:
        return self.inputs == args
```

Erlaubte Typen Phase 1: `Grid`, `Color` (alias `Int` mit Domain 0–9), `Mask`, `Object`, `ObjectSet`, `Int`. **Keine** Generics, **keine** parametrischen Typen — bewusste Vereinfachung. Phase 2 fügt ggf. `List[T]` und Type-Variablen hinzu.

Phase 1.5 ergänzt zwei Typen: `Predicate` und `Lambda` (siehe §6.4). Diese sind **enumerationsfähig, sandbox-sicher** und werden vom Suchalgorithmus wie normale Bausteine behandelt.

### 7.5 Phase 1.5: Higher-Order-Primitive (neu in v1.1)

**Auslöser:** External Review (ChatGPT) hat zu Recht festgestellt, dass die Basis-DSL ohne Higher-Order-Primitive einen niedrigen ARC-Ceiling hat. Drei Primitive bringen unverhältnismäßig viel Power:

| # | Name | Signatur | Cost | Beschreibung |
|---|------|----------|------|-------------|
| H1 | `map_objects` | `ObjectSet, Lambda[Object → Object] → ObjectSet` | 3.0 | Wendet Lambda auf jedes Objekt an |
| H2 | `filter_objects` | `ObjectSet, Predicate[Object] → ObjectSet` | 2.5 | Behält nur Objekte mit `pred(o) == True` |
| H3 | `align_to` | `Object, Object, AlignMode → Object` | 3.0 | Richtet Objekt A relativ zu B aus |
| **H4** | **`sort_objects`** | **`ObjectSet, SortKey → ObjectSet`** | **2.5** | **Sortiert Objekte nach Schlüssel (Größe, Farbe, Position) — neu in v1.2** |
| **H5** | **`branch`** | **`Predicate[Object], Lambda[Object → Object], Lambda[Object → Object] → Lambda[Object → Object]`** | **3.5** | **Conditional: gibt eine Lambda zurück, die je nach Predicate einen von zwei Transforms anwendet — neu in v1.2** |

Wobei `AlignMode` ein enumerierter Typ ist:

```python
class AlignMode(str, Enum):
    CENTER = "center"
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"
```

Und `SortKey` (neu in v1.2) ein enumerierter Typ:

```python
class SortKey(str, Enum):
    SIZE_ASC = "size_asc"             # kleinste zuerst
    SIZE_DESC = "size_desc"           # größte zuerst
    COLOR_ASC = "color_asc"           # niedrigster Color-Wert zuerst
    COLOR_DESC = "color_desc"
    POSITION_ROW = "position_row"     # oben nach unten, links nach rechts
    POSITION_COL = "position_col"     # links nach rechts, oben nach unten
    DISTANCE_FROM_CENTER = "distance_from_center"
```

**Begründung der enumerierten SortKey:** Genauso wie bei Predicates verzichten wir auf freie Lambda-Sortier-Funktionen. Die 7 enumerierten Keys decken alle relevanten ARC-Sortier-Patterns ab (validiert gegen Hodels arc-dsl-Solutions) und bleiben enumerierbar/sandbox-sicher.

**`branch` — Conditional-Logik (neu in v1.2):**

`branch` ist das einzige Higher-Order-Primitiv in Phase 1, das **andere Lambdas konsumiert** (statt Predicate). Es löst genau den Conditional-Gap, den das zweite Review identifiziert hat („wenn Objekt groß → X, sonst → Y").

Beispiel: „Färbe große Objekte rot, kleine bleiben unverändert":

```
map_objects(
    components,
    branch(
        size_gt(5),
        recolor_lambda(2),     # then: rot
        identity_lambda()       # else: unverändert
    )
)
```

**Sicherheits-Eigenschaft von `branch`:** Beide Branches sind **selbst geschlossene Lambda-Konstrukte aus der gleichen, enumerierten Konstruktor-Menge**. `branch` introduziert *keine* neue Ausdrucksmacht in Richtung Turing-Vollständigkeit — nur Fallunterscheidung über schon vorhandenen Bausteinen. Damit bleibt die Sandbox-Garantie erhalten und der Suchraum bleibt enumerierbar (allerdings mit höherem Branching-Faktor — siehe unten).

**Beispielhafte Verwendung in einem synthetisierten Programm:**

```
render_objects(
    map_objects(
        filter_objects(
            connected_components_4(input),
            color_eq(2)
        ),
        recolor_lambda(2, 5)
    ),
    input
)
```

Lesbar: „Finde alle blauen Objekte, färbe sie rot, rendere sie auf das Original-Grid."

**Auswirkung auf Suchraum:**
- Higher-Order-Primitive vergrößern den Branching-Faktor pro Tiefe.
- **`branch` ist der teuerste Posten** — er hat 3 Argumente (Predicate + 2 Lambdas), also kubisches Wachstum gegenüber Sub-Bank-Größen.
- **Mitigation:** Die Predicate-/Lambda-Konstruktoren werden in einer **separaten, kleineren Sub-Bank** geführt. Sie werden nur als Argumente von Higher-Order-Primitiven enumeriert, nicht als eigenständige Top-Level-Programme. `branch` zusätzlich auf max. **Sub-Tiefe 1** für seine Lambda-Argumente begrenzt (kein verschachteltes `branch(branch(...))` in Phase 1). Effektiver Branching-Faktor-Anstieg: ~2.5× bei Tiefe 4, kompensiert durch aggressives Pruning.

**Implementierungs-Notes:**
- `map_objects` und `filter_objects` operieren funktional (immutable ObjectSet → ObjectSet).
- `sort_objects` ist deterministisch stabil (gleicher Tie-Break wie Python `sorted`).
- `align_to` nutzt die Bounding-Box des Referenz-Objekts zur Positionsberechnung.
- `branch` evaluiert das Predicate genau einmal pro Object und delegiert dann an genau einen der beiden Lambdas.
- Alle fünf Primitive sind durch dieselbe Sandbox geschützt wie alle anderen.
- Tests: pro Higher-Order-Primitiv ≥ 8 Tests (komplexer als Basis-Primitive), inkl. Hypothesis-basierte Property-Tests gegen die Predicate-Algebra.
- Für `branch` zusätzlich Property-Tests: `branch(p, f, g) == f` wenn p überall true; `branch(p, f, g) == g` wenn p überall false; `branch(not(p), f, g) == branch(p, g, f)`.

**Stand nach Phase 1.5 (v1.2):** 56 Basis-Primitive + **5 Higher-Order** + 12 Predicate-Konstruktoren + 9 AlignMode-Werte + 7 SortKey-Werte = **89 enumerationsfähige Bausteine**.

### 7.6 Cost-Auto-Tuner (neu in v1.1, Phase 1.5)

**Auslöser:** External Review hat korrekt darauf hingewiesen, dass *Cost-Werte der einzige Lenkhebel der Suche* sind, solange kein LLM-Prior existiert. Manuelle Cost-Werte sind fragil. Wir führen einen **deterministischen, reproduzierbaren Auto-Tuner** ein — explizit **kein** ML-Modell.

**Algorithmus (rein symbolisch, deterministisch):**

```
Input:  catalog mit initialen Costs c0
        benchmark_results: pro Task → set of solving_programs
        learning_rate: ε = 0.05
        rounds: R = 5

Output: catalog mit getunten Costs c*

for round in 1..R:
    for primitive p in catalog:
        success_count  = count of solved programs using p
        failure_weight = sum over failed candidates / total failed
        c*(p) = c(p) * (1 - ε * normalize(success_count))
                       * (1 + ε * failure_weight)
    re-run benchmark with c*
    if no improvement → break
```

**Eigenschaften:**
- Deterministisch (gleicher Input → gleicher Output).
- Reproduzierbar (alle Costs in `catalog.json` versioniert).
- Konservativ (ε = 0.05 verhindert wilde Schwankungen).
- **Kein ML, kein Training, keine Modellgewichte.** Reine symbolische Aktualisierung.

**Implementierungsaufwand:** ~1 Tag (vgl. §21 Woche 6).

**Wann läuft der Tuner?**
- Manuell via `cognithor pse tune --benchmark <subset>`.
- Output ist ein neuer `catalog.json`, der per Pull-Request committed wird.
- **Niemals zur Laufzeit.** Costs sind statisch in einer Release-Version, was Reproduzierbarkeit von Benchmarks garantiert.

**Sicherheits-Aspekt:** Der Tuner bekommt ausschließlich Read-Zugriff auf Benchmark-Ergebnisse, schreibt nur eine neue Catalog-Datei. Capability `pse:dsl:tune` (Admin/dev only).

---

## 8. Enumerative Search-Algorithmus

### 8.1 Strategie: Bottom-Up Enumeration

Wir bauen Programme **von unten** (Tiefe 1) **nach oben** (Tiefe `max_depth`). Pro Tiefe behalten wir alle bisher *nicht-äquivalenten* Programme pro Output-Typ. Diese werden als Bausteine für die nächste Tiefe verwendet.

**Pseudocode:**

```
Input:  TaskSpec spec, Budget budget, Registry R
Output: Optional[Program]

bank: dict[type_str, list[Program]] = {}     # type → Liste von Programmen
seen: dict[type_str, set[fingerprint]] = {}  # für equivalence pruning

# Tiefe 0: Konstanten + InputRef
for prim in R.primitives_with_arity(0):
    p = build_leaf(prim)
    add_if_new(bank, seen, p, spec)

bank["Grid"].append(InputRef())

# Tiefe 1..max_depth
for d in range(1, budget.max_depth + 1):
    new_at_depth = []
    for prim in R.all_primitives():
        for args in cartesian_product_typed(bank, prim.signature.inputs):
            if max(arg.depth for arg in args) != d - 1:
                continue                # nur "frische" Bausteine
            candidate = Program(prim.name, args, prim.signature.output)
            if budget_exhausted():
                return best_so_far
            if check(candidate, spec):  # Demo-Check + Equivalence-Check
                if fully_correct(candidate, spec):
                    return candidate    # Early termination
            new_at_depth.append(candidate)
    merge(bank, new_at_depth)

return best_partial(bank, spec)
```

### 8.2 Wichtige Invarianten

- **I1**: Jeder im `bank` gespeicherte Kandidat hat einen eindeutigen Fingerprint pro Typ → keine Äquivalenz-Doppelung.
- **I2**: `bank[t]` ist nach Cost (aufsteigend) sortiert → Occam-Prior natürlich erzwungen.
- **I3**: Memory-Limit wird pro Tiefe geprüft, nicht nur am Ende.
- **I4**: Bei Erfolg auf einem Demo-Subset (statt allen) wird das Programm als „partial" gespeichert, nicht zurückgegeben — nur „fully correct" terminiert früh.

### 8.3 Klasse: `EnumerativeSearch`

```python
class EnumerativeSearch:
    def __init__(
        self,
        registry: PrimitiveRegistry,
        sandbox: SandboxExecutor,
        equivalence: ObservationalEquivalencePruner,
        verifier: Verifier,
        metrics: MetricsCollector,
    ): ...

    def search(self, spec: TaskSpec, budget: Budget) -> SynthesisResult:
        """
        Synchroner Einstiegspunkt. Throws BudgetExceededError nur bei
        Hard-Failures, ansonsten gibt SynthesisResult mit Status zurück.
        """
        ...

    def _enumerate_depth(self, depth: int, ...) -> Iterator[Program]: ...
    def _check_candidate(self, p: Program, spec: TaskSpec) -> StageResult: ...
```

### 8.4 Komplexität

- Pro Tiefe `d`: O(|R| · k^d), wobei `k` = durchschnittlicher Bank-Eintrag pro Typ.
- **Ohne Pruning** explodiert das schnell. **Mit Observational Equivalence** wächst `k` deutlich langsamer (empirisch ~5-10× kleiner für ARC-DSL bei `d=4`).
- **Hardware-Annahme:** Cognithor läuft auf Alex' RTX-5090-Workstation, aber die Engine ist **CPU-bound** in Phase 1 — keine GPU notwendig. Multi-Core via `concurrent.futures.ProcessPoolExecutor` für Sandbox-Parallelisierung.

---

## 9. Observational Equivalence Pruning

**Idee:** Zwei Programme `p1` und `p2` sind *observational equivalent* bezüglich einer TaskSpec, wenn sie auf **allen Demo-Inputs** dieselben Outputs liefern. Behalte nur eines (das mit niedrigerer Cost).

### 9.1 Fingerprinting

Pro Programm `p` und Input-Liste `[I1, I2, ..., In]`:

```
fingerprint(p) = sha256( bytes(p(I1)) || bytes(p(I2)) || ... || bytes(p(In)) )
```

Bei Exception/Timeout/Sandbox-Violation: spezieller Sentinel-Hash `ERROR_<exception_type>`.

### 9.2 Klasse: `ObservationalEquivalencePruner`

```python
class ObservationalEquivalencePruner:
    def __init__(self, sandbox: SandboxExecutor): ...

    def fingerprint(
        self, program: Program, demo_inputs: tuple[Grid, ...]
    ) -> str | None:
        """None wenn Programm auf >50 % Inputs crashed."""
        ...

    def is_duplicate(
        self, program: Program, fingerprint: str, type_: str
    ) -> bool: ...

    def register(self, program: Program, fingerprint: str, type_: str) -> None: ...
```

### 9.3 Pruning-Effektivität (Erwartung)

| Tiefe | Brutto-Kandidaten | Nach Pruning | Reduktion |
|-------|-------------------|--------------|-----------|
| 1 | ~60 | ~50 | 17 % |
| 2 | ~3 600 | ~800 | 78 % |
| 3 | ~250 000 | ~15 000 | 94 % |
| 4 | ~15 Mio | ~150 000 | 99 % |

(Schätzungen aus Literatur — z. B. „Bottom-Up Synthesis with Observational Equivalence", Albarghouthi et al. — die wir mit unseren Benchmarks empirisch validieren.)

---

## 10. Verifier-Pipeline

Fünf Stages, **streng sequenziell**, früh-Abbruch bei jedem Fehlschlag.

### 10.1 Stages

| # | Name | Prüft | Dauer | Stoppt bei Fail? |
|---|------|-------|-------|-------------------|
| 1 | Syntax | Programmbaum wohlgeformt | < 1 ms | ja |
| 2 | Type | Signaturen rekursiv konsistent | < 1 ms | ja |
| 3 | Demo | Programm auf allen Demo-Inputs → korrekte Outputs | bis 100 ms × n | ja |
| 4 | Property | Optionale Constraints (Größe, Farben) | < 5 ms | ja |
| 5 | Held-Out | Programm auf zurückgehaltenen Demo-Paaren | bis 100 ms × m | nein* |

*Held-Out-Fail markiert ein Programm als „verdächtig" (mögliche Overfit), führt aber nicht zum Verwerfen, sondern zur Reduzierung der `confidence`.

### 10.2 Klasse: `Verifier`

```python
class Verifier:
    def __init__(self, sandbox: SandboxExecutor, properties: PropertySet): ...

    def verify(self, program: Program, spec: TaskSpec) -> VerificationResult:
        results: list[StageResult] = []
        for stage in self.stages:
            r = stage.run(program, spec)
            results.append(r)
            if not r.passed and stage.fail_fast:
                return VerificationResult(
                    passed=False, stages=tuple(results), confidence=0.0
                )
        return self._build_result(results, spec)
```

### 10.3 Property-Tests Phase 1

- `output_grid_nonempty`
- `output_dimensions_match_inputs_or_constant`
- `output_colors_subset_of_input_colors_plus_const`
- `no_nan_no_negative`

---

## 11. Execution-Sandbox

**Pflicht — keine Kompromisse.** Jedes synthetisierte Programm wird in einem subprocess ausgeführt, mit harten Limits.

### 11.1 Architektur

```
Hauptprozess (Cognithor)
        │
        │ pickle(Program, Inputs)
        ▼
   ┌──────────────────────────────┐
   │ Worker-Subprocess            │
   │  • setrlimit RLIMIT_AS, _CPU │
   │  • setrlimit RLIMIT_NOFILE=0 │
   │  • setrlimit RLIMIT_NPROC    │
   │  • alarm(wall_clock)         │
   │  • prctl PR_SET_NO_NEW_PRIVS │
   │  • drop network namespace*   │
   │  • only DSL-AST whitelist    │
   └──────────────────────────────┘
        │
        │ pickle(Result | Exception)
        ▼
Hauptprozess
```

*Network-Namespace-Drop nur unter Linux. Auf Windows (Alex' Workstation) wird **WSL2 als Default-Worker-Pfad** verwendet (siehe §11.6, neu in v1.1). Reines Windows ohne WSL2 wird nur im „Research-Mode" mit expliziter Warnung erlaubt — niemals in Production.

### 11.2 Resource-Limits

| Limit | Default | Konfigurierbar |
|-------|---------|----------------|
| Wall-Clock | 100 ms / Hypothese | ja |
| CPU-Time | 200 ms | ja |
| Memory (Address Space) | 256 MB | ja |
| Open Files | 0 | nein |
| Subprozesse | 0 | nein |
| Stack | 8 MB | nein |

### 11.3 AST-Whitelist

Da wir DSL-Programme als Python-Code generieren (statt nur als Datenstruktur), prüft der Sandbox-Builder **vor jedem Eval** den AST. Erlaubt:

- Funktionsaufrufe nur an registrierte Primitive
- Numpy-Operationen nur über Primitive (kein direkter `np.*`-Zugriff)
- Keine `import`, `exec`, `eval`, `__class__`, `__bases__`, `__subclasses__`
- Keine `open`, `os.*`, `sys.*`, `socket.*`
- Keine f-Strings mit `__` (verhindert Reflection-Tricks)

**Implementierung:** `ast.NodeVisitor`, der bei verbotenen Knoten `SandboxViolationError` raised.

### 11.4 IPC

- **Format:** `pickle` mit `protocol=4` (kompakt, sicher genug *zwischen unseren eigenen Prozessen*).
- **Timeout:** Hauptprozess wartet `wall_clock + 50 ms` Buffer; danach `SIGKILL`.
- **Robustheit:** Bei IPC-Korruption → `SandboxError`, Programm verworfen, **kein** Retry (Programm könnte deterministisch crashen).

> **Sicherheitshinweis:** `pickle` ist nicht sicher gegen *fremde* Eingaben. Hier nur zwischen unseren eigenen Prozessen → akzeptabel. Falls Phase 2 externe Plugins erlaubt → Wechsel zu `msgpack` + Schema-Validation.

### 11.5 Adversarial-Tests (Phase 1 Pflicht)

Mindestens 20 Test-Cases, die alle abgewiesen werden müssen:

1. `eval("...")` direkt
2. `__import__('os').system('rm -rf /')`
3. While-True-Loop (Wall-Clock-Limit)
4. `numpy.zeros((10**9,))` (Memory-Limit)
5. `open('/etc/passwd').read()` (No-Files-Limit)
6. `socket.socket().connect(...)` (Network-Block)
7. Fork-Bomb
8. Stack-Overflow durch Rekursion
9. Pickle-Bomb (Billion-Laughs)
10. ... weitere 11 Cases

**Hard Gate:** 100 % aller Cases müssen blockiert werden, sonst Release-Block (vgl. K4).

### 11.6 Windows-Strategie: WSL2-Worker als Default (neu in v1.1)

**Problemstellung (External Review):** Auf nativem Windows fehlen Linux-Sicherheitsprimitive — kein `setrlimit RLIMIT_AS` mit kernel-enforcement, kein Network-Namespace, kein cgroup. Selbst mit AST-Whitelist bleiben Memory-Pressure und CPU-Exhaustion durch *legitime* DSL-Kombinationen ein Restrisiko.

**Lösung:** Worker-Subprozesse laufen unter Windows **bevorzugt in WSL2**, nicht nativ.

**Architektur:**

```
Cognithor (Hauptprozess, Windows) ─────┐
                                       │
                                       ▼
            ┌──────────────────────────────────────────┐
            │ SandboxRouter detektiert Plattform       │
            │  • Linux native     → subprocess         │
            │  • Windows + WSL2   → wsl.exe Worker     │
            │  • Windows w/o WSL2 → native + WARNING   │
            └──────────────────────────────────────────┘
                                       │
                                       ▼
                              ┌─────────────────────┐
                              │ WSL2 Worker         │
                              │ • Linux kernel      │
                              │ • setrlimit funktt. │
                              │ • netns möglich     │
                              │ • cgroup verfügbar  │
                              └─────────────────────┘
```

**IPC zwischen Cognithor und WSL2-Worker:**
- Pickled Bytes über `wsl.exe -- python -m cognithor.channels.program_synthesis.sandbox.worker`.
- Stdin/Stdout (kein Socket → keine zusätzliche Netzwerk-Angriffsfläche).
- Latency-Overhead WSL2-Boundary: ~3–8 ms, akzeptabel bei 100 ms Per-Candidate-Budget.

**Detection-Logik beim Cognithor-Start:**

```python
def _select_sandbox_strategy() -> SandboxStrategy:
    if sys.platform == "linux":
        return LinuxSubprocessStrategy()
    if sys.platform == "win32":
        if _wsl2_available():
            return WSL2WorkerStrategy()
        log.warning(
            "PSE running on Windows without WSL2. "
            "Reduced isolation. Research-Mode only. "
            "Install WSL2 + Ubuntu for production-grade sandbox."
        )
        return WindowsResearchStrategy()
    return UnsupportedPlatformError(sys.platform)
```

**Research-Mode (Windows ohne WSL2):**
- Beim Start: explizite Konsolen-Warnung + `pse_warning_research_mode_total` Counter.
- Sandbox bleibt aktiv (AST-Whitelist greift weiterhin).
- Aber: max. Wall-Clock auf 10 s begrenzt (kein 30 s), max. Memory auf 256 MB.
- Capability `pse:synthesize:production` ist im Research-Mode **deaktiviert** — nur explorative Calls möglich.

**Konfigurations-Default in `default.yaml`:**

```yaml
sandbox:
  windows_strategy: "wsl2_preferred"   # wsl2_preferred | wsl2_required | research
  wsl2_distro: "Ubuntu"
  research_mode_warning: true
```

**Tests:**
- `test_wsl2_detection.py` — Mock und Real-Test (CI hat WSL2 nicht, aber lokal Pflicht).
- `test_research_mode_limits.py` — Verifiziert reduzierte Limits.
- `test_strategy_selection.py` — Plattform-Matrix.

**Aufwand:** +2 Tage in Woche 4–5 (vgl. aktualisierter Roadmap §21).

---

## 12. Integration in PGE-Trinity

### 12.1 Planner-Seite

Der Planner erkennt eine PSE-Aufgabe an folgenden Signalen:
- Channel-Hint `program_synthesis` explizit
- Task-Typ `arc_grid_transformation` (Auto-Klassifikator)
- Vorhandensein von `examples` mit `input/output`-Paaren in der Anfrage

Auto-Klassifikator (Phase 1: regelbasiert, kein ML):

```python
def is_synthesizable(task: dict) -> bool:
    return (
        "examples" in task
        and len(task["examples"]) >= 2
        and all("input" in ex and "output" in ex for ex in task["examples"])
        and _looks_like_grid(task["examples"][0]["input"])
    )
```

### 12.2 Gatekeeper-Seite

Der Gatekeeper prüft:
1. Capability-Token `pse:synthesize` vorhanden und gültig (Ed25519-Signatur).
2. Budget innerhalb policy-erlaubter Grenzen (z. B. `max_wall_clock ≤ 60s` für nicht-privilegierte Calls).
3. TaskSpec-Größe ≤ 1 MB nach Serialisierung (DoS-Schutz).

Bei Verletzung: `403 GatekeeperRejected` mit strukturiertem Reason.

### 12.3 Executor-Seite

Der Executor wendet das gefundene Programm auf `test_input` an, in derselben Sandbox-Konfiguration. Output wird:
- An den Caller zurückgegeben.
- In Tactical Memory gespeichert (siehe §14).
- Mit `stable_hash(program)` und `stable_hash(spec)` als Audit-Trail in den Knowledge Vault geloggt.

### 12.4 Public API

```python
# cognithor/channels/program_synthesis/__init__.py

from cognithor.core.channels import register_channel
from .integration.pge_adapter import ProgramSynthesisChannel

CHANNEL = ProgramSynthesisChannel()
register_channel("program_synthesis", CHANNEL)

__all__ = ["CHANNEL", "synthesize"]


def synthesize(spec: TaskSpec, budget: Budget | None = None) -> SynthesisResult:
    """Public Convenience-API für direkten Use ohne PGE-Routing."""
    return CHANNEL.synthesize(spec, budget or Budget())
```

---

## 13. Capability-Token & Hashline-Guard-Integration

Cognithor nutzt **Ed25519/HMAC-Capability-Tokens** mit Hashline-Guard. Die PSE definiert vier Capabilities:

| Capability | Zweck | Default-Holder |
|------------|-------|----------------|
| `pse:synthesize` | Suche starten | Planner |
| `pse:synthesize:production` | Volle Wall-Clock-/Memory-Budgets (Linux/WSL2 only) | Planner |
| `pse:execute` | gefundenes Programm ausführen | Executor |
| `pse:cache:read` | Tactical-Cache lesen | Channel selbst |
| `pse:cache:write` | Tactical-Cache schreiben | Channel selbst |
| `pse:dsl:extend` | neues Primitiv registrieren | Admin / nur dev |
| `pse:dsl:tune` | Cost-Auto-Tuner ausführen (neu in v1.1) | Admin / nur dev |

### 13.1 Token-Validation-Flow

```
Caller ──[ Request + Token ]──▶ Gatekeeper
                                    │
                                    ├─▶ Hashline-Guard:
                                    │     • Token-Signatur Ed25519
                                    │     • Replay-Schutz (nonce + ts)
                                    │     • Capability-Match
                                    │     • Scope-Match (z.B. tenant)
                                    │
                                    └─▶ ok → forwards to Channel
                                        nok → 403 + structured log
```

Token-Format ist bestehende Cognithor-Konvention — keine Änderung nötig. **Wichtig:** PSE registriert seine Capabilities beim Start im zentralen Capability-Registry.

### 13.2 Audit-Trail

Jede Synthese-Anfrage erzeugt einen Hashline-Eintrag:

```json
{
  "ts": "2026-04-29T10:13:42.123Z",
  "actor": "planner@cognithor",
  "capability": "pse:synthesize",
  "spec_hash": "sha256:abc...",
  "budget": {"max_depth": 4, "wall_clock_seconds": 30.0},
  "result_status": "success",
  "program_hash": "sha256:def...",
  "duration_ms": 4321,
  "candidates_explored": 8742
}
```

Der Hashline ist append-only, manipulationssicher (Hashkette), und vollständig durchsuchbar.

---

## 14. Tactical-Memory-Integration

### 14.1 Cache-Key

```
cache_key = sha256(
    spec.stable_hash() ||
    dsl_version ||
    budget_class.stable_hash()
)
```

Wobei `budget_class` eine *gröbere* Bucketization ist (z. B. „depth_3_30s" statt exakter Float-Werte) — sonst zu viele Cache-Misses.

### 14.2 Cache-Eintrag

```python
@dataclass(frozen=True)
class CacheEntry:
    spec_hash: str
    dsl_version: str
    program_source: str
    program_hash: str
    score: float
    confidence: float
    cost_seconds: float
    created_at: float           # unix ts
    last_used_at: float
    use_count: int
```

### 14.3 TTL-Strategie

- **Erfolgreiche Lösungen:** TTL = 30 Tage, refresh on hit.
- **Partial-Lösungen:** TTL = 7 Tage.
- **NoSolution-Ergebnisse:** TTL = 1 Tag (DSL kann sich ändern → später vielleicht lösbar).

### 14.4 Konsistenz mit DSL-Updates

Bei DSL-Major-Bump (siehe §7.3) wird der gesamte PSE-Cache invalidiert. Minor-Bumps sind kompatibel — alte Lösungen bleiben gültig (neue Primitive ändern existierende nicht).

---

## 15. State-Graph-Navigator-Brücke

Der bestehende **State Graph Navigator** (SGN) liefert für ARC-AGI-3-Tasks eine annotierte Repräsentation: Knoten = Grid-Zustände, Kanten = vermutete Transformationen.

### 15.1 Was wir nutzen

- **Symmetrie-Hints:** SGN erkennt z. B. „Output ist horizontal gespiegelt" → wir setzen `mirror_horizontal` auf Cost-Faktor 0.5 für diese Task.
- **Größen-Hints:** „Output ist 2× so groß" → `scale_up_2x` priorisieren.
- **Farb-Hints:** „Nur Farbe 2 ändert sich" → Recolor-Primitive priorisieren.

Diese werden als **task-spezifische Cost-Multiplikatoren** in den Budget-Allokator gegeben — die Suche bleibt strukturell identisch, nur die Reihenfolge ändert sich.

### 15.2 API

```python
class StateGraphBridge:
    def annotate(self, spec: TaskSpec, sgn_result: dict) -> TaskSpec:
        """Reichert spec.annotations mit SGN-Hints an."""
        ...

    def cost_multipliers(self, annotations: dict) -> dict[str, float]:
        """Gibt {primitive_name: multiplier} zurück."""
        ...
```

**Wichtig:** Wenn SGN ausfällt oder nicht verfügbar ist, läuft die Suche identisch ohne Hints — keine harte Abhängigkeit. Phase 1 darf **keine** Korrektheit von SGN abhängig machen.

### 15.3 NumPy-Solver-Fallback

Der bestehende NumPy-Grid-Solver bleibt als **Fast-Path**:

```
Anfrage
    │
    ▼
NumPy-Solver (≤ 50 ms) ─── löst? ──▶ ja → return
    │
    │ nein
    ▼
PSE Enumerative Search ─── löst? ──▶ ja → return
    │
    │ nein
    ▼
Partial / NoSolution
```

Damit **regrediert** Phase 1 in keinem Fall ggü. dem aktuellen Stand.

---

## 16. Test-Strategie

### 16.1 Test-Typen & Coverage-Ziele

| Test-Typ | Coverage-Ziel | Anzahl (geschätzt) |
|----------|---------------|--------------------|
| Unit | ≥ 95 % aller Module | ~250 Tests |
| Integration | Pfad-Coverage Channel-API | ~30 Tests |
| Security (Sandbox) | 100 % der Adversarial-Cases | ≥ 20 Tests |
| Property (Hypothesis) | Pro DSL-Primitiv | ~50 Tests |
| Eval (ARC-AGI-3) | ≥ 100 Tasks | 1 großer Test |

**Gesamt:** ~350 neue Tests. Cognithor liegt aktuell bei 11 609+ Tests → wir landen bei ~12 000.

### 16.2 Pflicht-Tests pro DSL-Primitiv

Jedes Primitiv hat **mindestens**:
1. **Happy-Path-Test** mit Beispiel aus Doku.
2. **Edge-Case-Test** (leeres Grid, 1×1, max-Größe 30×30).
3. **Type-Mismatch-Test** (falsche Eingabe → klare Exception).
4. **Determinismus-Test** (zweimal aufrufen → identisch).
5. **Hypothesis-Property-Test** (z. B. `rotate90 ∘ rotate90 ∘ rotate90 ∘ rotate90 == identity`).

### 16.3 Integration-Tests (Pflicht)

- `test_full_pipeline.py`: Echte ARC-AGI-3-Beispiel-Task → SynthesisResult.
- `test_pge_adapter.py`: Round-Trip durch Planner→Gatekeeper→Channel→Executor.
- `test_tactical_cache.py`: Zweite Anfrage hittet Cache.
- `test_state_graph_bridge.py`: Mit/ohne SGN-Annotations identisches Korrekt-Ergebnis.

### 16.4 Security-Tests (Hard-Gate)

`tests/.../security/test_sandbox_escape.py` — pro Adversarial-Case:

```python
@pytest.mark.security
@pytest.mark.parametrize("payload", ADVERSARIAL_PAYLOADS)
def test_sandbox_blocks(payload: str, sandbox: SandboxExecutor):
    with pytest.raises((SandboxViolationError, SandboxTimeoutError, SandboxOOMError)):
        sandbox.execute_raw(payload, timeout_ms=100)
```

CI darf nur grün werden, wenn **alle** security-Tests passen.

### 16.5 Eval-Test (Benchmark)

`tests/.../eval/test_arc_agi3_subset.py`: lädt 100 ARC-AGI-3-Trainings-Tasks, läuft die volle Pipeline, vergleicht mit `known_solutions.json` (von uns gepflegt). Ausgabe: Score, Median-Zeit, P95-Zeit, FP-Rate. **Markiert als `slow`**, läuft nicht in jedem CI-Run, aber nightly.

---

## 17. Telemetrie & Metriken

### 17.1 Counter

- `pse_synthesis_requests_total{status, domain}`
- `pse_sandbox_violations_total{kind}`
- `pse_cache_hits_total`
- `pse_cache_misses_total`
- `pse_dsl_primitive_uses_total{primitive}`

### 17.2 Histogramme

- `pse_synthesis_duration_seconds` (Buckets: 0.1, 0.5, 1, 5, 10, 30, 60)
- `pse_candidates_explored` (Buckets: 100, 1k, 10k, 100k, 1M)
- `pse_program_depth` (Buckets: 1..6)
- `pse_program_size`

### 17.3 Tracing

OpenTelemetry-Spans für jede Anfrage:
- Root: `pse.synthesize`
- Child: `pse.cache.lookup`, `pse.search.enumerate`, `pse.verify.run`, `pse.sandbox.execute`

Spans tragen Attribute: `spec.hash`, `dsl.version`, `budget.depth`, `result.status`.

### 17.4 Strukturiertes Logging

Alle Logs als JSON, Felder: `ts`, `level`, `module`, `event`, `spec_hash`, `program_hash`, `duration_ms`, `extra`. **Kein** Log enthält rohe Grids (Datenschutz/Volume).

---

## 18. Benchmark-Plan (ARC-AGI-3)

### 18.1 Daten

- **Train-Subset:** 100 Tasks, manuell ausgewählt nach Diversität (Geom 30 %, Farbe 30 %, Objekt 25 %, Mixed 15 %).
- **Held-Out-Subset:** 30 weitere Tasks, **niemals** während Entwicklung gesehen → finale Validierung.

### 18.2 Baseline

Aktueller NumPy-Solver (660× Speedup-Solver) auf identischer Hardware. Ergebnisse vor PSE-Implementation einfrieren als `baseline_v0.78.json`.

### 18.3 Metrik

- **Solved@30s:** Anzahl Tasks, die mit `wall_clock=30s` gelöst werden.
- **Solved@5s:** Anzahl mit 5 s.
- **Median-Time-Solved:** Median über alle gelösten Tasks.
- **FP-Rate:** Programme, die alle Demo-Paare bestehen, aber Held-Out failen.

### 18.4 Erfolgs-Schwelle (vgl. K1)

PSE muss `Solved@30s_PSE ≥ Solved@30s_baseline + 5` erreichen. **Ohne** Verschlechterung der `Solved@5s`-Zahl (kein Regress auf Easy-Tasks).

### 18.5 Reproduzierbarkeit

- Fester Random-Seed (für tie-breaking).
- DSL-Version explizit in Output.
- Hardware-Fingerprint im Output.
- `make benchmark` als reproduzierbarer Einstiegspunkt.

---

## 19. Konfiguration & CLI

### 19.1 `default.yaml`

```yaml
pse:
  dsl:
    version: "1.1.0"                       # v1.1: erweitert um Higher-Order
    catalog: "cognithor/channels/program_synthesis/dsl/catalog.json"

  search:
    max_depth: 4
    max_candidates: 50000
    parallel_workers: 4
    early_termination: true
    higher_order_sub_depth: 2              # neu in v1.1: max. Tiefe für Predicates

  budget:
    wall_clock_seconds: 30.0
    per_candidate_ms: 100
    max_memory_mb: 1024

  sandbox:
    enabled: true
    network_block: true
    ast_whitelist_strict: true
    windows_strategy: "wsl2_preferred"     # neu v1.1: wsl2_preferred|wsl2_required|research
    wsl2_distro: "Ubuntu"
    research_mode_warning: true

  cache:
    enabled: true
    ttl_success_days: 30
    ttl_partial_days: 7
    ttl_no_solution_days: 1

  fallback:
    use_numpy_solver_first: true

  tuner:                                   # neu in v1.1
    enabled: false                         # nur manuell via CLI
    learning_rate: 0.05
    rounds: 5
    held_out_protected: true

  trace:                                   # neu in v1.1
    require_for_solved: true               # K9-Hard-Gate
    replay_p95_ms: 100                     # K10-Schwelle

  observability:
    metrics_enabled: true
    tracing_enabled: true
    log_level: "INFO"
```

### 19.2 CLI

```
cognithor pse run <task.json>            # eine Task synthetisieren
cognithor pse benchmark <subset>         # Benchmark-Lauf
cognithor pse cache stats                # Cache-Statistik
cognithor pse cache clear                # Cache leeren
cognithor pse dsl list                   # alle Primitive
cognithor pse dsl describe <name>        # Primitiv-Details
cognithor pse explain <result.json>      # Result als Pseudo-Code (Trace)
cognithor pse replay <result.json>       # Programm erneut ausführen, Output verifizieren
cognithor pse tune --benchmark <subset>  # Cost-Auto-Tuner ausführen (neu in v1.1)
cognithor pse sandbox doctor             # Plattform-Detection, Sandbox-Strategie-Check
```

---

## 20. Risiken & Mitigationen

| # | Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|---|--------|---------------------|--------|------------|
| R1 | Suchraum explodiert (Tiefe 4 zu groß) | mittel | hoch | Aggressives Pruning + Cost-Sortierung; Notfalls Tiefe 3 als Default |
| R2 | Sandbox-Escape unentdeckt | niedrig | extrem | 100 %-Adversarial-Coverage; externes Review; defense in depth |
| R3 | NumPy-Solver liefert besseren Score → PSE „nutzlos" | mittel | mittel | OK — PSE wird trotzdem für Phase-2-Build benötigt; Erfolg = ≥ +5 zusätzliche Tasks; Trace-KPI K9/K10 ist eigenständiger Wert |
| R4 | Pickle-IPC zu langsam | niedrig | niedrig | Wechsel zu shared-memory + msgpack falls > 5 ms Overhead/Call |
| R5 | DSL deckt zu wenig ARC-Operationen ab | **niedrig (war: mittel)** | hoch | **v1.1: Phase 1.5 mit `map_objects`/`filter_objects`/`align_to` jetzt Pflicht (nicht Puffer); Predicate-System geschlossen aber sandbox-sicher** |
| R6 | Test-Coverage-Ziel nicht erreicht | mittel | mittel | Test-First-Discipline; Pre-Commit-Hook mit Coverage-Gate |
| R7 | Windows-Sandbox schwächer als Linux | **niedrig (war: hoch)** | mittel | **v1.1: WSL2-Worker als Default; reines Windows nur Research-Mode mit harter Cap; klare Warnung beim Start** |
| R8 | Tactical Memory wird von anderen Channels verdrängt | niedrig | mittel | Eigene Cache-Partition mit reserviertem Volumen |
| R9 | DSL-Versionierung führt zu Cache-Wipe-Sturm | niedrig | niedrig | Migrations-Tooling (Phase-2-Item) |
| R10 | Über-Engineering verzögert MVP | hoch | hoch | Strenge Non-Goals (§2.2); Wochen-Plan harte Gates (§21) |
| **R11** | **Predicate-System bricht Such-Performance** | **mittel** | **mittel** | **Separate Sub-Bank für Predicates; Sub-Tiefe limitiert auf 2; eigenes Equivalence-Pruning** |
| **R12** | **Auto-Tuner overfitted auf Benchmark** | **mittel** | **niedrig** | **5 Runden Cap, ε = 0.05 konservativ; finale Costs immer manuell reviewed vor Commit; Held-Out-Subset im Tuner ausgeschlossen** |
| **R13** | **Trace-KPI K9 schwer für komplexe Programme automatisch zu validieren** | **mittel** | **mittel** | **Trace ist deterministisch erzeugt aus Programmbaum, also strukturell vollständig per Konstruktion; K9-Test prüft nur Existenz, nicht semantische Qualität** |

---

## 21. Roadmap: Wochenplan 1–7 (v1.1: erweitert)

**Änderungen ggü. v1.0 (External Review übernommen):**
- Roadmap von 6 auf 7 Wochen erweitert.
- Woche 3 (Search-Engine) von 5 auf 8 Tage gestreckt — der algorithmische Kern braucht mehr Zeit.
- Phase 1.5 (Higher-Order-Primitive + Auto-Tuner) als feste Wochen 6–7 statt nur „Puffer".
- WSL2-Worker integriert in Woche 4–5.

### Woche 1: Grundgerüst

- **Tag 1–2:** Repo-Struktur, `core/types.py`, `core/exceptions.py`, `core/version.py`.
- **Tag 3–4:** `dsl/types_grid.py`, `dsl/signatures.py`, `dsl/registry.py`.
- **Tag 5:** Erste 10 Primitive (Geometrisch). Tests grün.

**Gate Woche 1:** `pytest tests/.../unit/test_types.py tests/.../unit/test_dsl_primitives.py -k "rotate or mirror"` grün.

### Woche 2: DSL komplett + Sandbox-Basis

- **Tag 1–3:** Restliche ~46 Basis-Primitive in `dsl/primitives.py`. Pro Primitiv 5 Tests.
- **Tag 4–5:** `sandbox/executor.py`, `sandbox/policy.py`, `sandbox/worker.py`. AST-Whitelist (Linux-Pfad).

**Gate Woche 2:** Alle Basis-DSL-Tests grün; mindestens 10 Adversarial-Sandbox-Tests grün.

### Woche 3: Search-Engine (verlängert: 8 Tage)

- **Tag 1–2:** `search/candidate.py`, `search/budget.py`.
- **Tag 3–5:** `search/enumerative.py` Grundgerüst, Bottom-Up-Enumeration mit Typ-System.
- **Tag 6–7:** `search/equivalence.py` mit Fingerprinting; Performance-Profiling.
- **Tag 8:** Erstes End-to-End auf trivialer Task (z. B. `output = rotate90(input)`); erstes nicht-triviales Beispiel.

**Gate Woche 3:** Mindestens 1 nicht-triviale ARC-Task wird gefunden in < 10 s. Pruning empirisch validiert (Reduktion ≥ 80 % bei Tiefe 3).

### Woche 4: Verifier + Cache + WSL2-Worker

- **Tag 1–2:** `verify/pipeline.py`, `verify/stages.py`, `verify/properties.py`.
- **Tag 3:** `integration/tactical_memory.py`, `integration/capability_tokens.py`.
- **Tag 4–5:** WSL2-Worker-Strategie (`sandbox/strategies/wsl2.py`), Plattform-Detection, Research-Mode-Limits.

**Gate Woche 4:** Verifier vollständig; Cache hittet ≥ 50 % bei Reruns; WSL2-Worker läuft auf Alex' Workstation.

### Woche 5: PGE-Integration + SGN-Bridge + Trace-System

- **Tag 1–2:** `integration/pge_adapter.py`, `integration/state_graph_bridge.py`, `integration/numpy_solver_bridge.py`.
- **Tag 3:** Restliche Adversarial-Sandbox-Tests, alle 100 % grün (Linux + WSL2).
- **Tag 4–5:** **Trace-System** (`Program.to_source`, `replay()`, `trace_test.py`-Suite) — K9, K10 messbar machen.

**Gate Woche 5:** Alle Security-Tests grün; Trace für jedes gelöste Programm vorhanden und replay-fähig in < 100 ms; mindestens 15/30 Train-Tasks gelöst.

### Woche 6: Phase 1.5 — Higher-Order + Predicate-System (v1.2: 1,5 Tage länger)

- **Tag 1:** §6.4 Predicate-/Lambda-Datentypen.
- **Tag 2:** Predicate-Konstruktoren (`color_eq`, `size_gt`, `is_rectangle`, etc.).
- **Tag 3:** `map_objects`, `filter_objects`, `align_to` Implementierung.
- **Tag 4:** Search-Engine-Anpassung für Higher-Order (Sub-Bank für Predicates).
- **Tag 5:** Tests für H1–H3, inkl. Property-Tests.
- **Tag 5,5:** **`sort_objects` mit SortKey-Enum (neu in v1.2).** ~0,5 Tage.
- **Tag 6:** **`branch` mit Conditional-Lambda-Logik (neu in v1.2).** ~1 Tag inkl. Tests und Property-Tests.

**Gate Woche 6:** Alle 5 Higher-Order-Primitive funktionieren; mindestens 4 Tasks gelöst, die in v1.0-DSL **nicht** lösbar waren (mind. 1 davon nutzt `branch` für Conditional-Logik).

### Woche 7: Auto-Tuner + Benchmark + Release

- **Tag 1:** Cost-Auto-Tuner (`cli/pse_cli.py tune`).
- **Tag 2:** Voller Benchmark auf 100 Train-Tasks + 30 Held-Out.
- **Tag 3:** Tuner laufen lassen, neuer `catalog.json`, Re-Benchmark, Vergleichs-Doku.
- **Tag 4:** Doku finalisieren, CLI-Polish, CHANGELOG, README.
- **Tag 5:** Release-Cut v0.80.0, Hashline-Audit-Trail-Verifikation.

**Gate Woche 7 (Release-Gate):** Alle 16 Akzeptanzkriterien (§22) erfüllt.

**Puffer:** 30 % Buffer auf Personenebene eingeplant — bei jedem Wochen-Gate ist eine 1-Tage-Verlängerung erlaubt, ohne den Endtermin zu verschieben (durch Vor-Arbeit aus späteren Wochen kompensierbar).

---

## 22. Akzeptanzkriterien (Definition of Done)

Phase 1 ist *done*, wenn **alle** folgenden Punkte erfüllt sind:

- [ ] **D1** Alle 10 Erfolgs-Kriterien (§3) erfüllt (inkl. neue K9, K10).
- [ ] **D2** ARC-DSL hat ≥ 50 Basis-Primitive plus **5 Higher-Order-Primitive** (Phase 1.5: `map_objects`, `filter_objects`, `align_to`, `sort_objects`, `branch`), jedes mit ≥ 5 Tests (Higher-Order: ≥ 8 Tests; `branch`: ≥ 12 Tests inkl. Algebra-Properties).
- [ ] **D3** Test-Coverage ≥ 90 % auf neuem Code; Gesamt-Coverage Cognithor nicht gesunken.
- [ ] **D4** Alle Security-Tests grün auf Linux **und** WSL2 (100 %-Hard-Gate).
- [ ] **D5** Benchmark gegen Baseline durchgeführt; Ergebnis dokumentiert in `docs/.../benchmarks.md`.
- [ ] **D6** PGE-Trinity-Integration funktioniert end-to-end inkl. Capability-Token-Validation.
- [ ] **D7** Tactical-Memory-Cache wirkt messbar (Hit-Rate ≥ 80 % auf Reruns).
- [ ] **D8** CLI funktioniert: `cognithor pse run`, `benchmark`, `dsl list`, `explain`, `tune`.
- [ ] **D9** README, dsl_reference, architecture, benchmarks, tutorial alle vorhanden und peer-reviewed (durch Tomi oder zweite KI).
- [ ] **D10** Hashline-Audit-Trail funktioniert für jede Anfrage.
- [ ] **D11** Telemetrie-Counter und -Histogramme erscheinen im Cognithor-Observability-Dashboard.
- [ ] **D12** Mindestens 1 erfolgreich gelöster Task wird als „Hello-World"-Beispiel in Tutorial dokumentiert (mit vollständigem Programm-Trace).
- [ ] **D13** Versions-String `pse-1.2.0` korrekt in `__init__.py` und `pyproject.toml` gesetzt.
- [ ] **D14** Apache-2.0-Header auf jeder neuen Datei.
- [ ] **D15** Pre-Commit-Hooks (ruff, mypy --strict, pytest --cov ≥ 90 %) grün.
- [ ] **D16** **Trace-Hard-Gate (neu in v1.1):** Jede in Eval-Suite gelöste Task hat (a) menschenlesbaren Pseudo-Code-Trace und (b) ist replay-bar mit identischem Output in P95 ≤ 100 ms. K9/K10 zu 100 % erfüllt.
- [ ] **D17** **WSL2-Default unter Windows (neu in v1.1):** Auf Alex' Workstation startet PSE per Default mit WSL2-Worker; Research-Mode-Warnung wird emittiert wenn WSL2 nicht verfügbar.
- [ ] **D18** **Auto-Tuner-Pflichtlauf (neu in v1.1):** Final-Catalog wurde mindestens einmal durch Auto-Tuner verbessert ggü. initialen Werten; Diff dokumentiert.

---

## 23. Offene Fragen für Phase 2

Diese Punkte sind **nicht** Teil von Phase 1, sind aber dokumentiert, damit Phase 1 sie nicht ausschließt:

1. **LLM-Prior-Schnittstelle:** Welche Form hat der LLM-Output, der die Suche heuristisch lenkt? Vorschlag: Ranking der Top-K Primitive pro Tiefe.
2. **MCTS oder Best-First?** Phase-2-Entscheidung nach Profiling der Phase-1-Suche.
3. **CEGIS-Integration:** Wo dockt der Counter-Example-Loop an? Vorschlag: zwischen Verifier-Stage 3 und 5.
4. **Cross-Task-Library-Learning (DreamCoder):** Trigger-Bedingung („nach N gelösten Tasks") und Compaction-Kriterium.
5. **DSL-Erweiterungs-Mechanismus:** Sollen neue Primitive zur Laufzeit registrierbar sein? Sicherheits-Implikation: nur mit `pse:dsl:extend`-Capability + Hashline-Eintrag.
6. **GPU-Acceleration:** Lohnt sich GPU für Equivalence-Fingerprinting bei großen Demo-Mengen?
7. **Multi-DSL-Routing:** Wie erkennt der DSL-Selector, dass eine Task **außerhalb** ARC-DSL liegt?
8. **Pattern Completion (neu in v1.2, eigenständiger Phase-2-Workstream):** Das zweite External Review hat Pattern Completion als größten verbleibenden DSL-Gap identifiziert. Reine LLM-Heuristik wird das nicht lösen — es braucht entweder spezielle Pattern-/Periodicity-Detection-Primitive (`detect_period`, `find_repeat_unit`, `complete_symmetry`) oder einen separaten Pattern-Completion-Channel mit eigener Sub-DSL. **Phase-2-Entscheidung:** Eigenständige Sub-DSL bevorzugt, weil Pattern-Tasks strukturell anders sind als Object-Manipulation und der Suchraum sonst explodiert.
9. **Iteration / Fixpunkt-Operator:** Pattern wie „wende X an, bis sich nichts mehr ändert" sind in Phase 1 ausgeschlossen (Halting-Problem in der Sandbox). Phase 2 könnte einen `iterate_until_stable(transform, max_iter=10)` mit hartem Iterations-Limit aufnehmen.

---

## 24. Anhang A: Beispiel-Programme

### A.1 Triviale Rotation

**Task:** `output = rotate90(input)`, 3 Demo-Paare.

**Erwartete Synthese:**
```
rotate90(input)
```
- Tiefe 1, 1 Knoten, gefunden in Tiefe-1-Iteration.
- Cost: 1.0
- Erwartete Suchzeit: < 100 ms.

### A.2 Mittlere Komplexität

**Task:** Output ist horizontal gespiegelt **und** alle blauen Pixel werden zu rot.

**Erwartete Synthese:**
```
mirror_horizontal(recolor(input, 1, 2))
```
- Tiefe 2, 4 Knoten (inkl. Konstanten), gefunden in Tiefe-2-Iteration.
- Cost: 1.0 + 1.5 + 0.5 + 0.5 = 3.5
- Erwartete Suchzeit: < 2 s.

### A.3 Hoch komplex (Grenzfall)

**Task:** Größtes Objekt extrahieren und auf weißem Grund 2× skalieren.

**Erwartete Synthese:**
```
scale_up_2x(bounding_box(largest_object(connected_components_4(input))))
```
- Tiefe 4, 5 Knoten.
- Cost: 2.0 + 1.5 + 1.5 + 2.5 + 0 = 7.5
- Erwartete Suchzeit: 10–25 s.
- **Genau dieser Fall** ist die kritische Klasse, die Phase 1 abdecken soll.

### A.4 Phase 1.5: Higher-Order (neu in v1.1)

**Task:** Alle blauen Objekte (Farbe 1) in rot (Farbe 2) umfärben, alle anderen unverändert lassen, dann auf Original-Grid rendern.

**Erwartete Synthese:**
```
render_objects(
    map_objects(
        filter_objects(
            connected_components_4(input),
            color_eq(1)
        ),
        recolor_lambda(2)
    ),
    input
)
```
- Tiefe 4, ~7 Knoten inkl. Predicates und Lambdas.
- Cost: 2.0 + 3.0 + 2.5 + 2.5 + ... ≈ 12.0
- Erwartete Suchzeit: 15–40 s.
- **Ohne Higher-Order-Primitive:** in v1.0-DSL nicht in Tiefe ≤ 4 lösbar — genau die Klasse, die durch v1.1 dazukommt.

### A.5 Trace-Output (Beispiel zu K9)

Für A.4 erzeugt PSE folgenden Trace:

```
# PSE Solution Trace
# Spec hash:    sha256:9f2c...
# Program hash: sha256:1ab8...
# DSL version:  1.1.0
# Search time:  18.4s, 32874 candidates, depth 4

Step 1: components = connected_components_4(input)
        # → ObjectSet of 7 objects
Step 2: blue_objects = filter_objects(components, color_eq(1))
        # → ObjectSet of 3 objects (filtered from 7)
Step 3: red_objects = map_objects(blue_objects, recolor_lambda(2))
        # → ObjectSet of 3 objects, all color=2
Step 4: result = render_objects(red_objects, input)
        # → Grid 12x10

Replay:  identical output verified in 23ms
```

Dieser Trace ist (a) deterministisch, (b) menschenlesbar, (c) replay-bar — erfüllt K9 und K10.

---

## 25. Anhang B: Glossar

| Begriff | Bedeutung |
|---------|-----------|
| **AGI** | Artificial General Intelligence; Cholletʼs ARC-Benchmark zielt darauf. |
| **ARC-AGI-3** | Aktueller (2026) Chollet-Benchmark mit interaktiven Tasks. |
| **DSL** | Domain-Specific Language; hier: Vokabular für Grid-Transformationen. |
| **Enumerative Search** | Vollständige Aufzählung aller Programme bis zu Tiefe N. |
| **Observational Equivalence** | Zwei Programme verhalten sich auf allen Demo-Inputs gleich. |
| **Occam-Prior** | Bevorzuge das einfachste/kürzeste Programm bei Ties. |
| **PGE-Trinity** | Cognithor-Architektur: Planner → Gatekeeper → Executor. |
| **PSE** | Program Synthesis Engine (das hier spezifizierte Modul). |
| **SGN** | State Graph Navigator, bestehende ARC-AGI-3-Komponente. |
| **Tactical Memory** | Cognithor-Memory-Tier für kurz- bis mittelfristige Caches. |
| **Hashline Guard** | Cognithors Token-Validation- und Audit-Trail-Schicht. |
| **CEGIS** | Counter-Example-Guided Inductive Synthesis (Phase 2). |

---

## 26. Anhang C: Selbstprüfungs-Checkliste

> Diese Checkliste habe ich beim Entwurf der Spec mehrfach durchlaufen. Bei der Prüfung durch ChatGPT bitte gegenprüfen.

**Vollständigkeit der Architektur:**
- [x] Alle Schichten haben klare Verantwortung (§4, §5).
- [x] Jeder Komponente ist eine Datei zugewiesen (§5).
- [x] Public API ist definiert (§12.4).
- [x] Konfiguration ist spezifiziert (§19.1).
- [x] CLI ist spezifiziert (§19.2).

**Technische Konsistenz:**
- [x] Alle Datentypen sind frozen/immutable (§6).
- [x] Hashing-Strategie überall einheitlich (`stable_hash`, SHA-256).
- [x] Cache-Key enthält DSL-Version (§14.1).
- [x] DSL-Versionierung führt zu Cache-Invalidation (§7.3, §14.4).
- [x] Sandbox-Limits sind quantifiziert (§11.2).

**Sicherheit:**
- [x] Adversarial-Tests sind 100 %-Hard-Gate (§11.5, §16.4, §22 D4).
- [x] AST-Whitelist beschrieben (§11.3).
- [x] Capability-Tokens definiert (§13).
- [x] Audit-Trail lückenlos (§13.2).
- [x] Datenschutz: keine rohen Grids in Logs (§17.4).

**Cognithor-Integration:**
- [x] PGE-Trinity klar adressiert (§12).
- [x] Bestehende ARC-AGI-3-Komponenten nicht dupliziert (§15.3).
- [x] Tactical-Memory als Cache, nicht als Re-Implementation (§14).
- [x] Hashline-Guard wiederverwendet, nicht ersetzt (§13).
- [x] Apache-2.0-Konsistenz (§22 D14).

**Mess- & Erfolgsbarkeit:**
- [x] Alle Erfolgs-Kriterien quantifiziert (§3).
- [x] Benchmark-Methode reproduzierbar (§18.5).
- [x] Telemetrie deckt alle Hot-Paths ab (§17).
- [x] Akzeptanzkriterien sind binär entscheidbar (§22).

**Realismus:**
- [x] 4–6-Wochen-Plan hat Puffer (§21).
- [x] Risiken inkl. Mitigationen dokumentiert (§20).
- [x] Non-Goals explizit (§2.2).
- [x] Keine harte Abhängigkeit zu unfertigem Code (§15.1).

**Lesbarkeit für Review:**
- [x] Inhaltsverzeichnis (§0).
- [x] Glossar (§25).
- [x] Beispiele für jeden Anwendungsfall (§24).
- [x] Diagramme (§4).
- [x] Tabellen für Vergleiche (§7.2, §11.2, §20).

**v1.1-spezifische Prüfung (External-Review-Übernahme):**
- [x] Phase 1.5 ist in Roadmap fest verankert, nicht nur Puffer (§21 Woche 6).
- [x] Higher-Order-Primitive haben sandbox-sichere Predicate-Definition (§6.4, §7.5).
- [x] Auto-Tuner ist deterministisch, kein ML (§7.6).
- [x] WSL2-Pfad ist Default unter Windows, Research-Mode hat reduzierte Limits (§11.6).
- [x] Trace-KPIs K9, K10 sind Hard-Gates, nicht Nice-to-have (§3, §22 D16).
- [x] Roadmap auf 7 Wochen erweitert, Woche 3 verlängert (§21).
- [x] Risiken R5, R7 entschärft, R11–R13 für neue Komponenten ergänzt (§20).
- [x] Capability `pse:dsl:tune` neu registriert (§13).
- [x] Trace-Beispiel A.5 zeigt K9/K10 konkret (§24).

**v1.2-spezifische Prüfung (zweites External Review):**
- [x] `sort_objects` als H4 mit enumeriertem `SortKey` (§7.5).
- [x] `branch` als H5 für Conditional-Logik, sandbox-sicher durch geschlossene Lambda-Konstrukte (§7.5).
- [x] K3 ehrlich auf 30 s Default + 60 s Eval-Soft-Cap nachgeschärft (§3).
- [x] Pattern Completion als eigenständiger Phase-2-Workstream dokumentiert (§23 Punkt 8).
- [x] Iteration/Fixpunkt-Operator als Phase-2-Frage hinzugefügt (§23 Punkt 9).
- [x] Roadmap Woche 6 um 1,5 Tage erweitert für H4/H5 (§21).
- [x] D2 und D13 entsprechend aktualisiert (§22).

---

**Ende der Spezifikation v1.2.**

> **Note für ChatGPT-Re-Re-Review (falls gewünscht):** v1.2 ist ein *Patch*, kein Major-Release. Drei der vier Kernpunkte aus dem zweiten Review wurden direkt umgesetzt: `sort_objects`, `branch`, K3-Honesty. Den vierten (Pattern Completion) habe ich bewusst auf Phase 2 verschoben — er ist kein DSL-Patch, sondern braucht eine eigene Sub-DSL.
>
> **Was in v1.2 bewusst nicht eingebaut wurde:**
> - „5 nächste Primitive für +10 Tasks" — das war dein eigener Hinweis auf Scope-Creep, dem ich folge.
> - Clustering / Similarity-Metric — das ist Phase-2-Material (braucht Distanz-Berechnungen, die das Predicate-System sprengen).
> - Sort + Filter als eigene Higher-Order-Variante (`partition_objects`) — `filter` + `sort` reichen, doppelte Primitive vermeiden.
>
> **Phase-1-Ceiling-Schätzung nach v1.2:**
> - DSL-Coverage: ~78–82 % der ARC-AGI-3-Trainings-Tasks (vorher v1.1: 75–80 %).
> - Klare Klassen, die strukturell ungelöst bleiben: Pattern Completion (#1), Symmetry-Reconstruction (#2), Object-Clustering (#3), Multi-Beispiel-Regel-Inferenz (#7), Interaktive Tasks (#10) — siehe zweite Review-Liste.
> - Diese Klassen sind explizite Phase-2-Themen, kein Versagen von Phase 1.
>
> **Strategischer Vermerk für Cognithor-Roadmap:** Mit v1.2-Implementation ist die Außenkommunikation klarer formulierbar — Cognithor PSE liefert *Trace-First-ARC-Solving mit messbarem Score und vollständiger Erklärbarkeit*, was sich substantiell von LLM-only-Ansätzen wie HybridClaw oder II-Agent abhebt. Die Differenzierung wird damit auf Architektur-Ebene überprüfbar, nicht nur auf Marketing-Ebene.
