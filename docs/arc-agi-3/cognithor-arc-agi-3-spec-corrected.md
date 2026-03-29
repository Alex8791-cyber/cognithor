# Cognithor × ARC-AGI-3 Integration — Korrigierte Technische Spezifikation

**Version:** 3.0 (korrigiert gegen echte Codebase)
**Datum:** 29. März 2026
**Basis-Spec:** cognithor-arc-agi-3-spec.md v2.1
**Ziel:** Cognithor als ARC-AGI-3-Agent lauffähig machen
**Voraussetzung:** Cognithor v0.66.1, Python 3.12+
**Repo:** github.com/Alex8791-cyber/cognithor | Lizenz: Apache 2.0

---

## Korrekturen gegenüber v2.1

| # | v2.1 (falsch) | v3.0 (korrekt) | Grund |
|---|---------------|-----------------|-------|
| 1 | `cognithor/modules/arc_agi3/` | `src/jarvis/arc/` | Package heißt intern `jarvis` |
| 2 | `from cognithor.core.planner` | `from jarvis.core.planner` | Alle Imports |
| 3 | `cognithor/mcp_tools/arc_agi3_tool.py` | `src/jarvis/mcp/arc_tools.py` | MCP-Konvention |
| 4 | `cognithor/config/arc_agi3_agent.yaml` | Pydantic `ArcConfig` in `config.py` + YAML-Override | Config-Pattern |
| 5 | Tests in `modules/arc_agi3/tests/` | `tests/test_arc/` | Test-Konvention |
| 6 | v0.57.0 | v0.66.1 | Veraltet |
| 7 | "123 MCP Tools" | 107 Tools (97 aktiv + 10 disabled) | Falsche Zahl |
| 8 | `install.sh` anpassen | Existiert nicht — `scripts/bootstrap_windows.py` | Kein install.sh |
| 9 | `start_cognithor.bat` Menü | Kein Menü — neuer `--arc` CLI-Flag | BAT hat kein Menü |
| 10 | Unicode in BAT (✓, ━) | ASCII only (`[OK]`, `=`) | Windows cp1252 |
| 11 | `asyncio.get_event_loop()` | `asyncio.get_running_loop()` | Portabilitätsregel |
| 12 | `python` hardcoded | `sys.executable` | Portabilitätsregel |

---

## 1. Was ARC-AGI-3 verlangt

ARC-AGI-3 ist ein interaktiver Benchmark. Der Agent wird in ein unbekanntes, rundenbasiertes Environment geworfen — ohne Instruktionen, ohne Regeln, ohne Zielangabe. Er muss selbstständig:

- **Explorieren:** Aktionen ausprobieren und beobachten, was sich ändert
- **Modellieren:** Aus Beobachtungen ein Weltmodell ableiten
- **Ziele erkennen:** Ohne Hinweis herausfinden, was "Gewinnen" bedeutet
- **Planen und Ausführen:** Effizient handeln — Scoring: RHAE = (human_steps / agent_steps)²

Jedes Game hat 8-10 Level mit steigender Komplexität.

### ARC-AGI-3 API-Interface

```python
import arc_agi
from arcengine import GameAction, GameState, FrameDataRaw

arc = arc_agi.Arcade()
env = arc.make("ls20")
obs = env.reset()
obs = env.step(GameAction.ACTION1, data={"x": 32, "y": 32})
# obs.state → GameState.PLAYING / WIN / GAME_OVER / NOT_PLAYED
# obs.frame / obs.frame_data → Grid-Daten (Format via Phase 0 zu validieren)
scorecard = arc.get_scorecard()
```

**WICHTIG:** Alle SDK-Annahmen (Grid-Format, Action-Space, Attribute) werden in Phase 0 validiert. Der Code in dieser Spec ist adaptiv geschrieben.

---

## 2. Gap-Analyse: Cognithor vs. ARC-AGI-3

| ARC-AGI-3 Fähigkeit | Cognithor Status | Lücke | Priorität |
|---|---|---|---|
| Exploration | Evolution Engine (idle-time) | Keine Echtzeit-Exploration | KRITISCH |
| Weltmodell | 5-Tier Memory (Langzeit) | Kein In-Session-Lernen | KRITISCH |
| Goal Inference | Ziele kommen vom User | Kein autonomes Goal-Setting | KRITISCH |
| Visuelles Verständnis | Text/API-basiert | Kein Grid/Pixel-Verständnis | KRITISCH |
| Planning | PGE Planner (Prompt→Plan) | Nicht State→Plan | MITTEL |
| Feedback Loop | Executor → Ergebnis | Kein Observe→Act→Learn Zyklus | HOCH |
| Effizienz (RHAE) | Keine Optimierung | Braucht Aktionszähler | MITTEL |

---

## 3. Projektstruktur

```
src/jarvis/
├── arc/                              ← NEU
│   ├── __init__.py
│   ├── __main__.py                   # CLI: python -m jarvis.arc
│   ├── validate_sdk.py               # Phase 0: SDK-Annahmen prüfen
│   ├── adapter.py                    # ARC SDK ↔ Jarvis Brücke
│   ├── episode_memory.py             # In-Session Kurzzeit-Lernen
│   ├── mechanics_model.py            # Cross-Level Regel-Abstraktion
│   ├── goal_inference.py             # Autonomes Ziel-Erkennen
│   ├── explorer.py                   # 3-Phasen Exploration
│   ├── visual_encoder.py             # Grid → Text für LLM
│   ├── agent.py                      # Haupt-Orchestrierung
│   ├── cnn_model.py                  # Optional: CNN Action Predictor
│   ├── swarm.py                      # Parallele Game-Runs
│   ├── audit.py                      # Hash-Chain Audit Trail
│   └── error_handler.py              # Resilience & Recovery
├── mcp/
│   └── arc_tools.py                  # NEU: arc_play, arc_status, arc_replay
├── config.py                         # ERWEITERN: ArcConfig Pydantic-Modell
tests/
├── test_arc/                         ← NEU
│   ├── conftest.py
│   ├── test_episode_memory.py
│   ├── test_goal_inference.py
│   ├── test_explorer.py
│   ├── test_visual_encoder.py
│   ├── test_mechanics_model.py
│   ├── test_audit.py
│   ├── test_error_handler.py
│   └── test_agent.py
pyproject.toml                        # ERWEITERN: [arc] dependency group
```

---

## 4. Module — Detaillierte Spezifikation

### 4.1 Environment Adapter (`src/jarvis/arc/adapter.py`)

Brücke zwischen ARC SDK und Jarvis-Internals. Übersetzt Grid-Observations in Jarvis-kompatible Datenstrukturen.

**Klassen:**
- `ArcObservation` — Dataclass: raw_grid, game_state, step_number, level, grid_diff, changed_pixels, action_history
- `ArcEnvironmentAdapter` — init(game_id), initialize(), act(action, data), reset_level(), get_scorecard()

**Kritische Methode:** `_extract_grid(raw)` — Format hängt vom SDK ab, wird nach Phase 0 angepasst. Fallback: `np.zeros((64, 64, 3))`.

**Import-Abhängigkeiten:** `arc_agi`, `arcengine`, `numpy`

### 4.2 Episode Memory (`src/jarvis/arc/episode_memory.py`)

In-Session Kurzzeit-Gedächtnis. Cognithor's 5-Tier Memory speichert Langzeitwissen — ARC braucht schnelles, volatiles Gedächtnis innerhalb eines Games.

**Klassen:**
- `StateTransition` — Dataclass: state_hash, action, next_state_hash, pixels_changed, resulted_in_win, resulted_in_game_over, level
- `Hypothesis` — Dataclass: description, supporting/contradicting evidence, confidence
- `EpisodeMemory` — record_transition(), hash_grid() (MD5, cached), get_action_effectiveness(), get_unexplored_actions(), is_novel_state(), clear_for_new_level()

**Design-Entscheidung:** MD5 für Grid-Hashing (Geschwindigkeit > Sicherheit, da kein Audit). Max 200.000 Transitions.

**Import-Abhängigkeiten:** `numpy`, `hashlib`, `collections`

### 4.3 Goal Inference Module (`src/jarvis/arc/goal_inference.py`)

Autonomes Erkennen des Spielziels aus Beobachtungen. Analysiert Win/Game-Over Transitions.

**Klassen:**
- `GoalType` — Enum: UNKNOWN, REACH_STATE, CLEAR_BOARD, FILL_PATTERN, NAVIGATE, AVOID, SEQUENCE
- `InferredGoal` — Dataclass: goal_type, description, confidence, evidence
- `GoalInferenceModule` — analyze_win_condition(memory), get_best_goal(), on_level_complete()

**4 Analyse-Strategien:**
1. Win-Transitions → welche Aktionen führen zum Win?
2. Game-Over-Muster → was vermeiden?
3. Pixel-Änderungs-Muster → Board-Clearing?
4. Keine Wins → weiter explorieren

**Import-Abhängigkeiten:** `numpy`

### 4.4 Hypothesis-Driven Explorer (`src/jarvis/arc/explorer.py`)

Ersetzt zufällige Exploration durch systematisches Testen.

**Klassen:**
- `ExplorationPhase` — Enum: DISCOVERY, HYPOTHESIS, EXPLOITATION
- `HypothesisDrivenExplorer` — initialize_discovery(action_space), choose_action(obs, memory, goals)

**3 Phasen:**
1. **Discovery** (max 50 Steps): Alle Aktionen systematisch testen, strategisches Grid-Sampling für complex actions (13 Positionen: Ecken, Mitte, Kreuzpunkte)
2. **Hypothesis** (bis Confidence >0.6): Beste Aktionen bevorzugen (80/20 exploit/explore)
3. **Exploitation**: Bekannte Win-Strategie ausführen

**Import-Abhängigkeiten:** `arcengine.GameAction`, `random`, `numpy`

### 4.5 Visual State Encoder (`src/jarvis/arc/visual_encoder.py`)

Konvertiert 64×64 Grids in LLM-verständliche Textbeschreibungen.

**Klassen:**
- `VisualStateEncoder` — encode_for_llm(grid, diff), encode_compact(grid)

**Encoding-Strategie (nicht das ganze Grid — sprengt Context):**
1. Farbhistogramm (Top 5)
2. Bounding-Box-Erkennung für Regionen (max 5 Farben)
3. Diff-Analyse (Änderungs-Zentrum, Änderungs-Bereich)

**Import-Abhängigkeiten:** `numpy`, `collections.Counter`

### 4.6 Mechanics Model (`src/jarvis/arc/mechanics_model.py`)

Generalisiert Transitions zu wiederverwendbaren Regeln über Level hinweg.

**Klassen:**
- `MechanicType` — Enum: MOVEMENT, TRANSFORMATION, CREATION, DESTRUCTION, TOGGLE, CONDITIONAL, NO_EFFECT, UNKNOWN
- `Mechanic` — Dataclass: mechanic_type, action, description, observed_in_levels, consistency_score
- `MechanicsModel` — analyze_transitions(memory, level), get_reliable_mechanics(), predict_action_effect()

**Design:** EMA (alpha=0.3) für Konsistenz-Score. Min 3 Beobachtungen pro Aktion.

### 4.7 Agent — Haupt-Orchestrierung (`src/jarvis/arc/agent.py`)

Verbindet alle Module. Implementiert den Game Loop.

**Klasse:** `CognithorArcAgent`

**Hybrid-Ansatz:**
- Fast Path: Explorer + Memory + Goals (algorithmisch, >2000 FPS)
- Strategic Path: LLM-Planner alle N Steps (konfigurierbar, default 10)
- Optional: CNN Action Predictor

**Game Loop:**
```
initialize() → discovery_queue aufbauen
while True:
    action = explorer.choose_action()
    if should_consult_llm(): action = llm_planner(state, memory, goals)
    obs = adapter.act(action)
    memory.record_transition()
    if step % 5 == 0: goals.analyze_win_condition()
    if WIN: on_level_complete() → mechanics.analyze() → next level
    if GAME_OVER: reset (max 5x)
    if MAX_STEPS: break
return scorecard
```

### 4.8 CNN Model (Optional) (`src/jarvis/arc/cnn_model.py`)

Online-Learning CNN für Action-Prediction. Trainiert WÄHREND des Spielens.

**Klassen:**
- `ActionPredictor(nn.Module)` — Conv2d Stack → Action Head (7 actions) + Coordinate Head (64×64)
- `OnlineTrainer` — add_experience(), predict(), train_step (batch_size=64)

**Abhängigkeit:** `torch` (nur wenn aktiviert). Kein Hard-Dependency.

### 4.9 Swarm (`src/jarvis/arc/swarm.py`)

Parallele Game-Runs via asyncio + ThreadPool (agent.run() ist sync).

**Klasse:** `ArcSwarmOrchestrator` — run_swarm(game_ids), get_aggregate_score()

**Design:** `asyncio.Semaphore(max_parallel)`, `loop.run_in_executor(None, agent.run)`

### 4.10 Audit Trail (`src/jarvis/arc/audit.py`)

Hash-Chain Audit für jeden Game-Run. Kompatibel mit Cognithor's Hashline Guard.

**Klassen:**
- `ArcAuditEvent` — Dataclass: timestamp, event_type, game_id, level, step, action, ...
- `ArcAuditTrail` — log_event() mit SHA-256 Hash-Chain, export_jsonl(), verify_integrity()

### 4.11 Error Handler (`src/jarvis/arc/error_handler.py`)

Fehler dürfen NIEMALS einen Game-Run abbrechen.

**Komponenten:**
- `retry_on_error` — Decorator mit exponential backoff
- `safe_frame_extract` — Gibt IMMER numpy-Array zurück, probiert frame/frame_data/grid/pixels/data/image
- `GameRunGuard` — Context Manager, fängt alle Exceptions, holt IMMER Scorecard

### 4.12 MCP Tools (`src/jarvis/mcp/arc_tools.py`)

3 neue MCP Tools, registriert via bestehendes `register_builtin_handler`:

- `arc_play` — Game starten (single/benchmark/swarm), returns Scorecard
- `arc_status` — Laufenden Run monitoren
- `arc_replay` — Replay-Link + Zusammenfassung

---

## 5. Config-Integration

**In `src/jarvis/config.py`** — neues Pydantic-Modell:

```python
class ArcConfig(BaseModel):
    enabled: bool = False
    api_key_env: str = "ARC_API_KEY"
    operation_mode: Literal["normal", "competition"] = "normal"
    save_recordings: bool = True
    recording_dir: str = "~/.jarvis/recordings/arc"

    # Explorer
    discovery_max_steps: int = 50
    hypothesis_confidence_threshold: float = 0.6

    # LLM Planner
    llm_enabled: bool = True
    llm_call_interval: int = 10

    # CNN
    cnn_enabled: bool = False
    cnn_device: str = "cuda"

    # Limits
    max_steps_per_level: int = 500
    max_resets_per_level: int = 5
    max_total_steps: int = 5000

    # Memory
    max_transitions: int = 200_000

    # Swarm
    swarm_max_parallel: int = 4
```

Hinzufügen in `JarvisConfig`:
```python
arc: ArcConfig = ArcConfig()
```

---

## 6. pyproject.toml Erweiterung

```toml
[project.optional-dependencies]
arc = [
    "arc-agi>=0.3.0",
    "numpy>=2.0,<3",
]
arc-gpu = [
    "cognithor[arc]",
    "torch>=2.2.0",
]
```

`all` und `full` Groups bleiben unverändert — ARC ist ein separates Feature.

---

## 7. CLI Entry Point

`python -m jarvis.arc --game ls20` oder `cognithor --arc --game ls20`

Alternativ über bestehenden Entry Point: `cognithor --arc` Flag in `__main__.py`.

---

## 8. Wettbewerbs-Constraints

- **Kein Internet** bei Kaggle → LLM-Planner muss deaktivierbar sein (`--no-llm`)
- **Open Source** unter Apache 2.0 (bereits gegeben)
- **Keine Benchmark-spezifischen Harnesses** → Agent muss generell funktionieren
- **Deadlines:** Milestone 1 (30. Juni), Milestone 2 (30. September), Final (2. November)

---

## 9. Phase 0 — SDK-Validierung (VOR ALLEM ANDEREN)

`src/jarvis/arc/validate_sdk.py` prüft alle Annahmen:

| Annahme | Zu prüfen | Wenn falsch → anpassen in |
|---|---|---|
| Grid in `obs.frame` | Attribut-Name und Typ | `adapter.py: _extract_grid()` |
| Grid ist 64×64 | Shape | `adapter.py`, `visual_encoder.py`, `cnn_model.py` |
| `GameAction` hat `is_simple()`/`is_complex()` | Methoden-Existenz | `explorer.py` |
| `obs.state` nutzt `GameState.WIN` / `GAME_OVER` | Enum-Werte | `agent.py`, `episode_memory.py` |
| `obs.levels_completed` existiert | Attribut | `adapter.py` |
| `env.action_space` gibt Liste zurück | Typ | `explorer.py`, `agent.py` |
| `arc.get_scorecard().score` existiert | Attribut | `agent.py` |
| `env.step()` gibt FrameDataRaw zurück | Return-Typ | `adapter.py` |
| `GameAction.RESET` existiert | Enum-Wert | `agent.py`, `explorer.py` |
| `arc.make()` akzeptiert `save_recording` | Parameter | adapter.py |

**Ergebnis wird als `arc_agi3_validation.json` gespeichert.**

---

## 10. Erwartete Ergebnisse

| Metrik | Random | Phase 4 | Phase 7 | Phase 9 |
|---|---|---|---|---|
| RHAE Score | ~0.12% | >1% | >5% | >10% |
| Level 1 Completion | ~50% | >90% | >95% | >98% |
| Durchschnittliche Levels | 1-2 | 3-4 | 5+ | 7+ |

Zum Vergleich: Preview-Gewinner "Stochastic Goose" erreichte 12.58% mit CNN. Frontier LLMs: 0.37%.
