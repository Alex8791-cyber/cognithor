# Sprint-11 — ARC-AGI-3 Game-Agent Foundation

**Date:** 2026-05-02
**Owner-Direktive:** "sprint 11 go" (nach "erst sprint 10 abschliessen dann auf offizielle aktuelle arc agi 3 umbauen")
**Builds on:** Sprint-10 closed (main HEAD `5d99560a`)

## Was ARC-AGI-3 tatsächlich ist

ARC-AGI-3 ist **kein statischer Grid-Korpus** wie ARC-AGI-1/2 — es ist eine **interaktive Game-Challenge**:

- Repo: `arcprize/ARC-AGI-3-Agents` (MIT)
- Pakete: `arc-agi>=0.9.1` (Environment-Wrapper) und `arcengine` (FrameData / GameAction / GameState)
- Online-API: `https://three.arcprize.org/` (braucht `ARC_API_KEY`)
- Offline-Mode: `arc-agi` Package erlaubt lokale Game-Execution (seit 0.9.3)
- Agent-ABC: `is_done(frames, latest_frame) -> bool` + `choose_action(frames, latest_frame) -> GameAction`
- MAX_ACTIONS = 80 pro Game
- States: `NOT_PLAYED`, `WIN`, `GAME_OVER`, etc.
- Actions: `RESET`, `ACTION1..ACTION7` (manche `is_simple()`, manche `is_complex()` mit `set_data()`)

## Was sich gegenüber ARC-AGI-1 fundamental ändert

| Aspekt | ARC-AGI-1 (unser Sprint-10) | ARC-AGI-3 |
|---|---|---|
| Format | Statisches Input/Output-Mapping | Interaktive Game-Loop |
| Eingang | `examples: list[(input, output)]` Demos | `frames: list[FrameData]` Episode |
| Antwort | Programm das Output produziert | Sequenz von `GameAction` |
| Bewertung | Exakte Grid-Gleichheit | `levels_completed / win_levels` Score |
| Suchraum | Phase-1 Programmenumeration | Game-Tree (depth = bis 80 actions) |
| LLM-Rolle | Prior für Programm-Synthesis | Action-Auswahl pro Frame |

## Architektur-Strategie für Cognithor

**Insight:** Sprint-10's Phase-1-DSL bleibt nutzbar als **Frame-Transformations-Library**. Sie wird nicht das Top-Level mehr — sondern ein Werkzeug, das pro Frame die nächste sinnvolle Action vorschlägt.

```
ARC-AGI-3 FrameData  ─→ FrameBridge ─→ Phase-1 Spec
                                          ↓
                                    EnumerativeSearch
                                    (mit Sprint-10 DSL)
                                          ↓
                                       Programm
                                          ↓
                                  ActionDecoder
                                          ↓
                                     GameAction
```

Plus eine Episode-State-Schicht:
```
Episode-Memory (alle bisherigen Frames + ihre Programme)
               ↓
         ProgressDetector  (sind wir näher am Goal?)
               ↓
         BacktrackPolicy   (RESET wenn stuck?)
```

## Sprint-11 Wave-Plan

**Wave-1 (foundation, this PR):** Skelett ohne Drittanbieter-Abhängigkeit
- Protocol-Klassen die `FrameData` / `GameAction` / `GameState` API spiegeln
- Abstrakter `CognithorPSEAgent` mit `is_done()` + `choose_action()`
- Dummy `RandomActionAgent` als Smoke-Baseline
- Unit-Tests dass Protocol + Action-Selection funktioniert
- README erklärt, wie man unseren Code in den offiziellen Harness plugged

**Wave-2 (bridge):** Frame → DSL Bridge
- `FrameBridge`: FrameData.frame → Cognithor Grid-Tensor (np.int8)
- `ActionDecoder`: DSL-Programm-Output → GameAction (mit Heuristiken für ACTION1..7)
- Tests gegen synthetische Frame-Sequenzen

**Wave-3 (search):** Phase-1-DSL als Frame-Transformer
- `Sprint10DSLAgent`: nutzt EnumerativeSearch um pro Frame eine Transformation zu finden, dann Action ableiten
- Episode-State: track which actions wurden bisher probiert, vermeide Loops
- Tests an offline arc-agi-3 environments (wenn Hardware vorhanden)

**Wave-4 (LLM-driven):** vLLM-qwen3.6:27b im Action-Loop
- `LLMReasoningAgent`: Pro Frame Stage-1 (free-form) + Stage-2 (constrained Action JSON)
- Reused: Sprint-10 Track B vLLM-Backend
- Telemetry: Action-Latency P50/P95

**Wave-5 (validation):** Live ARC-AGI-3 Score
- Cognithor against `https://three.arcprize.org/` API
- Frozen Score-Baselines pro Game-ID
- v0.97.0 Release

## Sprint-11 PR-1 Scope (this PR)

Ziel: Eine **klar definierte API-Surface** auf der nachfolgende Wellen aufbauen, **ohne** sofort die Drittanbieter-Pakete `arc-agi` und `arcengine` als harte Abhängigkeit zu erzwingen.

Konkret:

1. **`src/cognithor/channels/program_synthesis/arc_agi3/protocol.py`** — Protocol-Klassen die exakt das API-Surface von `arcengine.FrameData` / `arcengine.GameAction` / `arcengine.GameState` spiegeln. Cognithor's eigene Code ist gegen diese Protocols typed; eine spätere Wave fügt einen `arcengine`-Adapter hinzu.

2. **`src/cognithor/channels/program_synthesis/arc_agi3/agent.py`** — `CognithorPSEAgent(ABC)` mit denselben zwei Abstract-Methods wie der offizielle ABC (`is_done`, `choose_action`). Plus eine concrete `RandomActionAgent` für Smoke-Tests.

3. **`tests/test_channels/test_program_synthesis/arc_agi3/test_protocol.py`** — Unit-Tests die Protocol-Konformität prüfen.

4. **`tests/test_channels/test_program_synthesis/arc_agi3/test_random_agent.py`** — Smoke-Test: RandomActionAgent läuft eine Mock-Episode bis WIN.

5. **`docs/channels/program_synthesis/arc_agi3.md`** — Channel-Doc, wie man Cognithor's PSE in den offiziellen `ARC-AGI-3-Agents` Harness plugged.

**Nicht in PR-1:** keine arc-agi/arcengine pip-Dependency, kein Live-API-Call, keine echte DSL-Integration (kommt in Wave-2/3).

## Akzeptanz-Kriterien

- [x] Protocol-Klassen klar dokumentiert mit der exakten Field-Liste vom offiziellen ABC
- [x] `CognithorPSEAgent` ABC kann ohne arc-agi installiert importiert werden
- [x] `RandomActionAgent` läuft gegen Mock-Frames bis Episode-Ende
- [x] mypy --strict clean auf den neuen Files
- [x] ruff check + ruff format clean
- [x] keine Regressions in der bestehenden 1464-Test-PSE-Suite

## Production-Ready-Pfad

1. PR-1 (this) — Foundation
2. PR-2 — `FrameBridge` + `ActionDecoder` (Wave-2)
3. PR-3 — `Sprint10DSLAgent` mit Phase-1-Search im Loop (Wave-3)
4. PR-4 — `LLMReasoningAgent` mit qwen3.6:27b (Wave-4)
5. PR-5 — `arcengine`-Adapter + Live-API-Smoke-Tests (Wave-5)
6. PR-6 — Frozen Score-Baselines pro Game-ID (Wave-5)
7. v0.97.0 Release mit Sprint-11-Inhalt

Sprint-11-Gesamtdauer geschätzt: 5-7 Tage Entwickler-Zeit pro Wave (parallelisierbar zwischen DSL- und LLM-Tracks).
