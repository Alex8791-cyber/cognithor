# Cognithor ARC-AGI-3 — Diagnose, Fix & State Graph Navigator

## AN DAS CODING-MODELL: Lies dieses Dokument KOMPLETT bevor du anfängst.

**Aktueller Score:** 0.003 (0.3%) mit Qwen3:32B  
**Ziel-Score:** > 0.03 (3%) nach Abschluss aller Schritte  
**Maximales Potenzial:** 0.05–0.15 (5–15%) mit allen Optimierungen

Dieses Dokument hat drei Teile die STRIKT IN REIHENFOLGE abgearbeitet werden müssen:

1. **DIAGNOSE** — Finde was kaputt ist (Schritte 1–4)
2. **FIX** — Repariere die kaputten Module (Schritt 5)
3. **UPGRADE** — Baue den State Graph Navigator ein (Schritt 6)

Überspringe KEINEN Schritt. Teil 3 ohne Teil 1+2 bringt nichts — der State Graph Navigator arbeitet auf denselben Daten wie die bestehenden Module. Wenn die Daten kaputt sind, ist auch der Graph kaputt.

---

## Hintergrund: Warum 0.003 zu niedrig ist

**RHAE-Scoring:** Score = (human_steps / agent_steps)². Quadratische Bestrafung für jeden verschwendeten Step.

- Frontier-LLMs (Gemini 3.1 Pro): 0.37% — vergleichbar mit unserem Score
- Simpler RL-Agent ("Stochastic Goose", Preview-Gewinner): 12.58% — **34× besser als jedes LLM**
- Der RL-Agent nutzt kein LLM — er kartiert den State-Space als Graph und folgt dem kürzesten Pfad

Das zeigt: Das Problem ist nicht das LLM, sondern dass der Agent zu viele Steps verschwendet. Jeder LLM-Call kostet hunderte Millisekunden. Ein Graph-Lookup kostet < 1ms.

---

# TEIL 1: DIAGNOSE (Schritte 1–4)

## Schritt 1: Baseline messen — Random Agent OHNE Cognithor

Führe dieses Script aus und dokumentiere den **kompletten Output**. Das ist die Baseline. Wenn der Random-Agent ähnlich wie 0.003 scored, liegt das Problem nicht in den Modulen sondern in der Scoring-Methodik.

```python
"""
Diagnostischer ARC-AGI-3 Run — Random Baseline.
Dieses Script NICHT ändern — nur ausführen und Output dokumentieren.
"""
import arc_agi
from arcengine import GameAction, GameState
import time
import json
import random

arc = arc_agi.Arcade()
env = arc.make("ls20")
obs = env.reset()

diagnostics = {
    "frames_received": 0,
    "actions_taken": 0,
    "states_seen": set(),
    "actions_that_changed_frame": 0,
    "actions_with_no_effect": 0,
    "game_overs": 0,
    "wins": 0,
    "resets": 0,
    "action_distribution": {},
    "time_per_step_ms": [],
    "level_reached": 0,
}

prev_frame = None
for step in range(200):
    action = random.choice([a for a in env.action_space if a != GameAction.RESET])
    data = {}
    if action.is_complex():
        data = {"x": random.randint(0, 63), "y": random.randint(0, 63)}
    
    t0 = time.time()
    obs = env.step(action, data=data)
    elapsed = (time.time() - t0) * 1000
    
    diagnostics["time_per_step_ms"].append(elapsed)
    diagnostics["actions_taken"] += 1
    diagnostics["action_distribution"][str(action)] = diagnostics["action_distribution"].get(str(action), 0) + 1
    
    if obs:
        diagnostics["frames_received"] += 1
        current_frame = getattr(obs, 'frame', getattr(obs, 'frame_data', None))
        if current_frame is not None and prev_frame is not None:
            import numpy as np
            try:
                curr = np.array(current_frame)
                prev = np.array(prev_frame)
                if curr.shape == prev.shape:
                    changed = not np.array_equal(curr, prev)
                    if changed:
                        diagnostics["actions_that_changed_frame"] += 1
                    else:
                        diagnostics["actions_with_no_effect"] += 1
            except:
                pass
        prev_frame = current_frame
        
        if obs.state == GameState.WIN:
            diagnostics["wins"] += 1
            diagnostics["level_reached"] += 1
        elif obs.state == GameState.GAME_OVER:
            diagnostics["game_overs"] += 1
            env.step(GameAction.RESET)
            diagnostics["resets"] += 1

diagnostics["states_seen"] = len(diagnostics["states_seen"]) if isinstance(diagnostics["states_seen"], set) else 0
diagnostics["avg_step_time_ms"] = sum(diagnostics["time_per_step_ms"]) / len(diagnostics["time_per_step_ms"])
diagnostics["change_rate"] = diagnostics["actions_that_changed_frame"] / max(diagnostics["actions_taken"], 1)
del diagnostics["time_per_step_ms"]

print(json.dumps(diagnostics, indent=2, default=str))
sc = arc.get_scorecard()
print(f"\nScorecard: {sc}")
```

---

## Schritt 2: Modul-für-Modul Diagnose

Prüfe jedes Modul einzeln. Für jedes Modul: **öffne die Datei, lies den Code, führe den Test aus, dokumentiere das Ergebnis.**

### 2.1 Adapter — Grid-Extraktion (`adapter.py`)

**Das ist der wahrscheinlichste Fehler.** Wenn `_extract_grid()` den falschen Attribut-Namen nutzt, bekommt der gesamte Agent nur Nullen — und ALLE nachfolgenden Module arbeiten auf leeren Daten.

```python
from cognithor.modules.arc_agi3.adapter import ArcEnvironmentAdapter
import numpy as np

adapter = ArcEnvironmentAdapter("ls20")
try:
    obs = adapter.initialize()
    print(f"Observation Typ: {type(obs)}")
    print(f"Grid Shape: {obs.raw_grid.shape}")
    print(f"Grid Dtype: {obs.raw_grid.dtype}")
    print(f"Grid ist nur Nullen: {np.all(obs.raw_grid == 0)}")
    print(f"Game State: {obs.game_state}")
    print(f"Step Number: {obs.step_number}")
    
    if np.all(obs.raw_grid == 0):
        print("\n⚠️ KRITISCH: Grid ist nur Nullen!")
        print("→ _extract_grid() funktioniert nicht.")
        print("→ Alle Module (Encoder, Memory, Explorer, Goals) arbeiten auf leerem Grid.")
        print("→ FIX: Prüfe _extract_grid() und passe den Attribut-Namen an.")
        print("→ Nutze das SDK-Attribut-Discovery unten:")
        
        # SDK-Attribut herausfinden
        import arc_agi
        from arcengine import GameAction
        arc2 = arc_agi.Arcade()
        env2 = arc2.make("ls20")
        raw = env2.reset()
        print(f"\nRohe Observation Attribute:")
        for attr in dir(raw):
            if not attr.startswith('_'):
                val = getattr(raw, attr)
                if not callable(val):
                    print(f"  {attr}: {type(val).__name__} = {str(val)[:200]}")
except Exception as e:
    print(f"⚠️ Adapter-Initialisierung fehlgeschlagen: {e}")
    import traceback
    traceback.print_exc()
```

**Erwartetes Ergebnis:** Grid Shape muss (64, 64, ...) sein und NICHT nur Nullen enthalten.  
**Wenn kaputt:** Das erklärt den gesamten Score von 0.003 — behebe das ZUERST bevor du weitermachst.

### 2.2 Visual Encoder (`visual_encoder.py`)

```python
from cognithor.modules.arc_agi3.visual_encoder import VisualStateEncoder
import numpy as np
import arc_agi
from arcengine import GameAction

encoder = VisualStateEncoder()
arc = arc_agi.Arcade()
env = arc.make("ls20")
obs = env.reset()

# Grid auf ALLEN möglichen Attribut-Namen suchen
grid_attr = None
for attr in ['frame', 'frame_data', 'grid', 'pixels', 'data', 'image']:
    val = getattr(obs, attr, None)
    if val is not None:
        grid_attr = attr
        break

if grid_attr:
    grid = np.array(getattr(obs, grid_attr))
    print(f"Grid Attribut: {grid_attr}, Shape: {grid.shape}, Dtype: {grid.dtype}")
    print(f"Min/Max: {grid.min()} / {grid.max()}, Unique: {len(np.unique(grid))}")
    
    result = encoder.encode_for_llm(grid)
    print(f"\nEncoder Output ({len(result)} Zeichen):\n{result}")
    
    compact = encoder.encode_compact(grid)
    print(f"\nCompact: {compact}")
else:
    print("⚠️ KEIN GRID ATTRIBUT GEFUNDEN!")
    print("Attribute:", [a for a in dir(obs) if not a.startswith('_') and not callable(getattr(obs, a))])
```

**Prüfe:** Ist der Output sinnvoll oder nur Nullen/Müll? Passen die `color_names` zum tatsächlichen Farbformat?

### 2.3 Explorer (`explorer.py`)

```python
from cognithor.modules.arc_agi3.explorer import HypothesisDrivenExplorer, ExplorationPhase
from cognithor.modules.arc_agi3.episode_memory import EpisodeMemory
from cognithor.modules.arc_agi3.goal_inference import GoalInferenceModule
from arcengine import GameAction
from collections import Counter
import numpy as np
import arc_agi

explorer = HypothesisDrivenExplorer()
memory = EpisodeMemory()
goals = GoalInferenceModule()

arc = arc_agi.Arcade()
env = arc.make("ls20")
obs = env.reset()

if hasattr(env, 'action_space'):
    explorer.initialize_discovery(env.action_space)
    print(f"Discovery Queue Länge: {len(explorer.discovery_queue)}")
else:
    print("⚠️ env.action_space existiert nicht!")

dummy_obs = type('O', (), {
    'raw_grid': np.zeros((64, 64, 3), dtype=np.uint8),
    'game_state': 'PLAYING', 'changed_pixels': 0,
    'grid_diff': None, 'level': 0,
})()

action_log, phase_log = [], []
for i in range(50):
    action, data = explorer.choose_action(dummy_obs, memory, goals)
    action_log.append(str(action))
    phase_log.append(explorer.phase.value)

print(f"Aktionsverteilung: {Counter(action_log)}")
print(f"Phasen: {Counter(phase_log)}")
print(f"Phase nach 50 Steps: {explorer.phase.value}")
print(f"Queue übrig: {len(explorer.discovery_queue)}")
```

**Prüfe:** Wechselt die Phase? Werden verschiedene Aktionen getestet? Wenn gleichverteilt → quasi-random.

### 2.4 Episode Memory (`episode_memory.py`)

```python
from cognithor.modules.arc_agi3.episode_memory import EpisodeMemory
import numpy as np

mem = EpisodeMemory()
grid1 = np.random.randint(0, 10, (64, 64, 3), dtype=np.uint8)
grid2 = np.random.randint(0, 10, (64, 64, 3), dtype=np.uint8)

obs1 = type('O', (), {'raw_grid': grid1})()
obs2 = type('O', (), {'raw_grid': grid2, 'changed_pixels': 100, 'game_state': 'PLAYING', 'level': 0})()

for i in range(10):
    mem.record_transition(obs1, f"ACTION{i%6+1}", obs2)

print(f"Transitions: {len(mem.transitions)}")
print(f"Effektivität ACTION1: {mem.get_action_effectiveness('ACTION1')}")
print(f"Unexplored: {mem.get_unexplored_actions(mem.hash_grid(grid1), [f'ACTION{i}' for i in range(1, 7)])}")
print(f"Summary:\n{mem.get_summary_for_llm()}")
```

**Prüfe:** Transitions > 0? Effektivität ≠ 0.5? Summary enthält Daten?

### 2.5 Goal Inference (`goal_inference.py`)

```python
from cognithor.modules.arc_agi3.goal_inference import GoalInferenceModule
from cognithor.modules.arc_agi3.episode_memory import EpisodeMemory, StateTransition

gim = GoalInferenceModule()
memory = EpisodeMemory()

for i in range(90):
    t = StateTransition(state_hash=f"s{i%10}", action=f"ACTION{i%5+1}",
        next_state_hash=f"s{(i+1)%10}", pixels_changed=i*5,
        resulted_in_win=False, resulted_in_game_over=False, level=0)
    memory.transitions.append(t)
    memory.action_effect_map[f"ACTION{i%5+1}"]["total"] += 1
    if i*5 > 0:
        memory.action_effect_map[f"ACTION{i%5+1}"]["caused_change"] += 1

for i in range(5):
    t = StateTransition(state_hash="danger", action="ACTION3",
        next_state_hash="dead", pixels_changed=200,
        resulted_in_win=False, resulted_in_game_over=True, level=0)
    memory.transitions.append(t)
    memory.action_effect_map["ACTION3"]["total"] += 1
    memory.action_effect_map["ACTION3"]["caused_game_over"] += 1

for i in range(5):
    t = StateTransition(state_hash="near_win", action="ACTION2",
        next_state_hash="won", pixels_changed=300,
        resulted_in_win=True, resulted_in_game_over=False, level=0)
    memory.transitions.append(t)
    memory.action_effect_map["ACTION2"]["total"] += 1
    memory.action_effect_map["ACTION2"]["caused_win"] += 1

goals = gim.analyze_win_condition(memory)
for g in goals:
    print(f"  [{g.confidence:.2f}] {g.goal_type.value}: {g.description}")
```

**Prüfe:** Erkennt es ACTION2 → Win und ACTION3 → Gefahr?

### 2.6 Agent-Orchestrierung (`agent.py`)

```python
import logging
logging.basicConfig(level=logging.DEBUG)
from cognithor.modules.arc_agi3.agent import CognithorArcAgent

agent = CognithorArcAgent(game_id="ls20", use_llm_planner=True,
    llm_call_interval=10, max_steps_per_level=100)

original_consult = agent._consult_llm_planner
llm_call_count = 0
def counting_consult(*args, **kwargs):
    global llm_call_count
    llm_call_count += 1
    return original_consult(*args, **kwargs)
agent._consult_llm_planner = counting_consult

scorecard = agent.run()

print(f"\n=== Agent-Diagnose ===")
print(f"Score: {scorecard}")
print(f"Total Steps: {agent.total_steps}")
print(f"Level erreicht: {agent.current_level}")
print(f"LLM-Calls: {llm_call_count}")
print(f"Explorer-Phase: {agent.explorer.phase.value}")
print(f"Transitions in Memory: {len(agent.memory.transitions)}")
print(f"Goals erkannt: {len(agent.goals.current_goals)}")
```

**Prüfe:** LLM-Calls 0 → wird nie aufgerufen. LLM-Calls sehr hoch → zu teuer. Explorer bleibt in DISCOVERY → kein Fortschritt. Transitions 0 → record_transition() wird nicht aufgerufen.

---

## Schritt 3: Top-5 Fehlerquellen (in Reihenfolge prüfen)

| # | Fehlerquelle | Symptom | Fix |
|---|---|---|---|
| 1 | `adapter.py: _extract_grid()` nutzt falschen Attribut-Namen | Grid nur Nullen | Attribut-Name aus SDK-Inspektion anpassen |
| 2 | LLM wird bei JEDEM Step aufgerufen | Steps kosten >100ms statt <1ms | `total_steps % llm_call_interval` prüfen |
| 3 | Explorer Discovery-Queue ist leer | `env.action_space` gibt Unerwartetes zurück | `initialize_discovery()` debuggen |
| 4 | Phase-Transition passiert nie | Explorer bleibt in DISCOVERY (quasi-random) | `_check_phase_transition()` Logik prüfen |
| 5 | GameState.WIN Vergleich scheitert | String "WIN" vs. Enum GameState.WIN | `==` Vergleiche in goal_inference + agent prüfen |

---

## Schritt 4: Ergebnisse dokumentieren

Erstelle `arc_agi3_diagnose_ergebnis.md`:

```markdown
# ARC-AGI-3 Diagnose-Ergebnis

## Baseline (Random Agent ohne Cognithor)
- Score: [hier eintragen]
- Change Rate: [hier eintragen]

## Cognithor Agent (vor Fix)
- Score: 0.003
- LLM-Calls: [hier eintragen]
- Steps: [hier eintragen]

## Modul-Status
| Modul | Status | Problem | Fix |
|---|---|---|---|
| adapter.py | ??? | | |
| visual_encoder.py | ??? | | |
| explorer.py | ??? | | |
| episode_memory.py | ??? | | |
| goal_inference.py | ??? | | |
| agent.py | ??? | | |

## Hauptursache
[Was ist die primäre Ursache?]
```

---

# TEIL 2: FIXES (Schritt 5)

## Schritt 5: Kaputte Module reparieren

Arbeite die Fixes **einzeln** ab — nach jedem Fix einen Run machen und Score messen:

```bash
python -m cognithor.modules.arc_agi3 --game ls20
```

**Fix-Reihenfolge:**
1. `adapter.py` — Grid-Extraktion fixen (wenn kaputt → alles kaputt)
2. `agent.py` — LLM-Call-Frequenz korrigieren
3. `explorer.py` — Discovery-Queue und Phase-Transition
4. `goal_inference.py` — GameState-Vergleiche
5. `visual_encoder.py` — Farbformat-Anpassung

**Score-Protokoll nach jedem Fix:**

```
Fix 1 (adapter): Score = ___
Fix 2 (agent):   Score = ___
Fix 3 (explorer): Score = ___
...
```

**Ziel nach allen Fixes:** Score > 0.01 (1%). Wenn das erreicht ist → weiter zu Teil 3.

---

# TEIL 3: STATE GRAPH NAVIGATOR (Schritt 6)

## Schritt 6: State Graph Navigator implementieren und integrieren

Dieses Modul ist der größte einzelne Hebel für Score-Verbesserung. Es macht den entscheidenden Unterschied zwischen 0.3% (LLM-Ansätze) und 12.58% (RL+Graph-Ansatz des Preview-Gewinners).

### 6.1 Warum der State Graph alles ändert

```
Ohne State Graph (aktuell):
  Frame → Encoder → Explorer → (LLM?) → Aktion → Repeat
  Jeder Step: ~100ms (LLM) | Kein Gedächtnis welche States bereits besucht

Mit State Graph:
  Frame → State Hash → Graph kennt State?
    JA + Win-Pfad bekannt → Folge Pfad (0 LLM-Calls, <1ms pro Step!)
    JA + kein Win-Pfad   → Graph-basierte Exploration (unbesuchte Kanten zuerst)
    NEIN                  → Fallback auf Explorer + LLM
```

**Score-Rechnung:**
- Ohne Graph: ~500 Steps/Level, Mensch ~15 Steps → (15/500)² = 0.09%
- Mit Graph: ~100 Steps Exploration + ~20 Steps Navigation = ~120 total → (15/120)² = 1.56%
- Mit Graph + CNN: ~40 + ~18 = ~58 total → (15/58)² = 6.69%

### 6.2 State Graph Navigator — Vollständiger Code

Erstelle die Datei `cognithor/modules/arc_agi3/state_graph.py`:

```python
"""
State Graph Navigator für ARC-AGI-3
=====================================

Baut einen gerichteten Graphen aller beobachteten Zustandsübergänge auf
und findet kürzeste Pfade zum Win-Zustand.

Kernidee: Statt bei jedem Step das LLM zu fragen, kartieren wir zuerst
das State-Space und navigieren dann effizient.

Inspiriert durch den "Stochastic Goose" Ansatz (12.58% in Preview),
der mit RL + Graph-Search alle Frontier-LLMs um 30× schlug.

Performance-Constraint:
  - Max 200.000 States im Graph (Memory-Limit)
  - Hash-basierte Deduplizierung
  - Graph-Operationen müssen < 1ms sein (2000+ FPS Ziel)
"""

from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Optional
import hashlib
import numpy as np
import heapq


@dataclass
class StateNode:
    """Ein Knoten im State Graph = ein beobachteter Spielzustand."""
    state_hash: str
    visit_count: int = 0
    is_win: bool = False
    is_game_over: bool = False
    level: int = 0
    color_histogram: Optional[tuple] = None
    
    
@dataclass
class StateEdge:
    """Eine Kante im State Graph = eine beobachtete Aktion + Ergebnis."""
    action: str
    action_data: Optional[dict] = None
    pixels_changed: int = 0
    traversal_count: int = 0


class StateGraphNavigator:
    """
    Baut und navigiert den State Graph für ARC-AGI-3 Environments.
    
    Zwei Modi:
    1. EXPLORE: Noch nicht genug kartiert → priorisiere unbesuchte Kanten
    2. NAVIGATE: Win-Pfad bekannt → folge dem kürzesten Pfad
    """

    def __init__(self, max_states: int = 200_000):
        self.nodes: dict[str, StateNode] = {}
        self.edges: dict[str, dict[str, tuple[str, StateEdge]]] = defaultdict(dict)
        self.reverse_edges: dict[str, set[str]] = defaultdict(set)
        self.win_states: set[str] = set()
        self.game_over_states: set[str] = set()
        
        self._cached_win_path: Optional[list[tuple[str, str, dict]]] = None
        self._cache_valid_from: Optional[str] = None
        
        self.max_states = max_states
        self._hash_cache: dict[bytes, str] = {}
        
        self.total_edges = 0
        self.action_patterns_from_previous: dict[str, float] = {}

    # =========================================================
    # Graph-Aufbau
    # =========================================================

    def hash_grid(self, grid: np.ndarray) -> str:
        """Schneller Hash für State-Identifikation."""
        grid_bytes = grid.tobytes()
        if grid_bytes not in self._hash_cache:
            self._hash_cache[grid_bytes] = hashlib.md5(grid_bytes).hexdigest()[:16]
        return self._hash_cache[grid_bytes]

    def add_transition(
        self,
        from_grid: np.ndarray,
        action_str: str,
        action_data: Optional[dict],
        to_grid: np.ndarray,
        pixels_changed: int,
        game_state: str,
        level: int = 0,
    ) -> tuple[str, str]:
        """Beobachtete Transition zum Graphen hinzufügen."""
        from_hash = self.hash_grid(from_grid)
        to_hash = self.hash_grid(to_grid)

        if from_hash not in self.nodes and len(self.nodes) < self.max_states:
            self.nodes[from_hash] = StateNode(
                state_hash=from_hash, level=level,
                color_histogram=self._compute_histogram(from_grid),
            )
        if to_hash not in self.nodes and len(self.nodes) < self.max_states:
            is_win = ("WIN" in str(game_state))
            is_go = ("GAME_OVER" in str(game_state))
            self.nodes[to_hash] = StateNode(
                state_hash=to_hash, is_win=is_win, is_game_over=is_go,
                level=level, color_histogram=self._compute_histogram(to_grid),
            )
            if is_win:
                self.win_states.add(to_hash)
            if is_go:
                self.game_over_states.add(to_hash)

        if from_hash in self.nodes:
            self.nodes[from_hash].visit_count += 1

        edge_key = f"{action_str}_{action_data}" if action_data else action_str
        
        if edge_key not in self.edges[from_hash]:
            self.edges[from_hash][edge_key] = (to_hash, StateEdge(
                action=action_str, action_data=action_data,
                pixels_changed=pixels_changed,
            ))
            self.total_edges += 1
            self.reverse_edges[to_hash].add(from_hash)
            self._cached_win_path = None  # Cache invalidieren
        else:
            _, edge = self.edges[from_hash][edge_key]
            edge.traversal_count += 1

        return from_hash, to_hash

    # =========================================================
    # Pfadsuche
    # =========================================================

    def find_win_path(self, from_hash: str) -> Optional[list[tuple[str, Optional[dict], str]]]:
        """BFS: Kürzesten Pfad von from_hash zu einem Win-State finden.
        Returns: Liste von (action_str, action_data, next_state_hash) oder None."""
        if not self.win_states:
            return None
        if from_hash in self.win_states:
            return []

        if self._cached_win_path is not None and self._cache_valid_from == from_hash:
            return self._cached_win_path

        queue = deque([(from_hash, [])])
        visited = {from_hash}

        while queue:
            current, path = queue.popleft()
            for edge_key, (next_hash, edge) in self.edges.get(current, {}).items():
                if next_hash in visited or next_hash in self.game_over_states:
                    continue
                new_path = path + [(edge.action, edge.action_data, next_hash)]
                if next_hash in self.win_states:
                    self._cached_win_path = new_path
                    self._cache_valid_from = from_hash
                    return new_path
                visited.add(next_hash)
                queue.append((next_hash, new_path))

        return None

    # =========================================================
    # Exploration-Strategie
    # =========================================================

    def get_best_exploration_action(
        self, current_hash: str, available_actions: list[str]
    ) -> Optional[tuple[str, Optional[dict]]]:
        """Wähle die Aktion die am wahrscheinlichsten zu einem NEUEN State führt.
        Prioritäten: 1. Ungetestete Aktionen, 2. Selten besuchte States."""
        current_edges = self.edges.get(current_hash, {})
        
        tested_actions = set()
        for edge_key in current_edges:
            base = edge_key.split("_")[0] if "_" in edge_key else edge_key
            tested_actions.add(base)
        
        untested = [a for a in available_actions if a not in tested_actions and a != "RESET"]
        if untested:
            # Bevorzuge Aktionen die in früheren Leveln gut waren
            if self.action_patterns_from_previous:
                untested.sort(
                    key=lambda a: self.action_patterns_from_previous.get(a, 0),
                    reverse=True,
                )
            return untested[0], None

        candidates = []
        for edge_key, (next_hash, edge) in current_edges.items():
            if next_hash in self.game_over_states:
                continue
            next_node = self.nodes.get(next_hash)
            if next_node:
                visit_score = next_node.visit_count
                outgoing = len(self.edges.get(next_hash, {}))
                exploration_value = 1.0 / (visit_score + 1) + 0.1 * (1.0 / (outgoing + 1))
                candidates.append((exploration_value, edge.action, edge.action_data))

        if candidates:
            candidates.sort(reverse=True)
            _, action, data = candidates[0]
            return action, data

        return None

    def should_navigate(self) -> bool:
        """True wenn ein Win-Pfad bekannt ist."""
        return len(self.win_states) > 0

    def get_exploration_coverage(self) -> dict:
        total_possible = len(self.nodes) * 7
        return {
            "states": len(self.nodes),
            "edges": self.total_edges,
            "win_states": len(self.win_states),
            "game_over_states": len(self.game_over_states),
            "coverage": self.total_edges / max(total_possible, 1),
            "has_win_path": self._cached_win_path is not None,
            "win_path_length": len(self._cached_win_path) if self._cached_win_path else -1,
        }

    # =========================================================
    # Level-Transfer
    # =========================================================

    def extract_action_patterns(self) -> dict[str, float]:
        """Aktionsmuster extrahieren die über Level gelten könnten."""
        patterns = {}
        for from_hash, edges in self.edges.items():
            for edge_key, (to_hash, edge) in edges.items():
                action = edge.action
                if action not in patterns:
                    patterns[action] = {"pos": 0, "neg": 0, "neu": 0}
                if to_hash in self.win_states:
                    patterns[action]["pos"] += 1
                elif to_hash in self.game_over_states:
                    patterns[action]["neg"] += 1
                elif edge.pixels_changed > 0:
                    patterns[action]["pos"] += 0.5
                else:
                    patterns[action]["neu"] += 1

        scores = {}
        for action, c in patterns.items():
            total = c["pos"] + c["neg"] + c["neu"]
            if total > 0:
                scores[action] = (c["pos"] - c["neg"]) / total
        return scores

    def prepare_for_new_level(self):
        """Graph für neues Level: States löschen, Aktionsmuster behalten."""
        self.action_patterns_from_previous = self.extract_action_patterns()
        self.nodes.clear()
        self.edges.clear()
        self.reverse_edges.clear()
        self.win_states.clear()
        self.game_over_states.clear()
        self._cached_win_path = None
        self._cache_valid_from = None
        self._hash_cache.clear()
        self.total_edges = 0

    # =========================================================
    # Hilfsfunktionen
    # =========================================================

    def _compute_histogram(self, grid: np.ndarray) -> tuple:
        flat = grid[:, :, 0].flatten() if grid.ndim == 3 else grid.flatten()
        hist, _ = np.histogram(flat, bins=range(0, 12))
        return tuple(hist)

    def get_summary_for_llm(self) -> str:
        c = self.get_exploration_coverage()
        lines = [f"State Graph: {c['states']} States, {c['edges']} Kanten, "
                 f"{c['win_states']} Wins, {c['game_over_states']} Game-Overs"]
        if c['has_win_path']:
            lines.append(f"Win-Pfad: {c['win_path_length']} Steps")
        else:
            lines.append(f"Kein Win-Pfad. Coverage: {c['coverage']:.1%}")
        return "\n".join(lines)
```

### 6.3 Agent-Integration — `_step()` ersetzen

In `agent.py` folgende Änderungen vornehmen:

**1. Import hinzufügen:**
```python
from .state_graph import StateGraphNavigator
```

**2. In `__init__` hinzufügen:**
```python
self.state_graph = StateGraphNavigator(max_states=200_000)
self._navigation_mode = False
self._current_path: list = []
self._path_index: int = 0
```

**3. `_step()` Methode KOMPLETT ERSETZEN durch:**

```python
def _step(self) -> str:
    """Ein Agent-Step mit State Graph Navigation."""
    if self.adapter.level_step_count >= self.max_steps_per_level:
        return "MAX_STEPS"

    current_hash = self.state_graph.hash_grid(self.current_obs.raw_grid)

    # ===== NAVIGATION: Win-Pfad bekannt → folgen =====
    if self._navigation_mode and self._current_path and self._path_index < len(self._current_path):
        action_str, action_data, expected_next = self._current_path[self._path_index]
        action = self._parse_action(action_str)
        data = action_data or {}
        
        previous_obs = self.current_obs
        self.current_obs = self.adapter.act(action, data)
        self._path_index += 1
        
        # In Memory + Audit + Graph aufzeichnen
        self._record_step(previous_obs, action_str, data)
        
        # Pfad-Validierung: Sind wir noch auf Kurs?
        actual_hash = self.state_graph.hash_grid(self.current_obs.raw_grid)
        if actual_hash != expected_next:
            self._navigation_mode = False
            self._current_path = []
        
        self.total_steps += 1
        return self._check_game_state()

    # Navigation abgeschlossen oder ungültig
    self._navigation_mode = False
    self._current_path = []

    # ===== EXPLORATION: Graph aufbauen =====
    
    # Prüfe ob Win-Pfad jetzt verfügbar
    win_path = self.state_graph.find_win_path(current_hash)
    if win_path:
        self._navigation_mode = True
        self._current_path = win_path
        self._path_index = 0
        return self._step()  # Sofort navigieren

    # Graph-basierte Exploration
    available = [str(a) for a in self.adapter.env.action_space if str(a) != "RESET"]
    graph_action = self.state_graph.get_best_exploration_action(current_hash, available)
    
    if graph_action:
        action_str, action_data = graph_action
        action = self._parse_action(action_str)
        data = action_data or {}
    else:
        # Fallback: Hypothesis Explorer
        action, data = self.explorer.choose_action(self.current_obs, self.memory, self.goals)
        action_str = self._action_to_str(action, data)

    # LLM nur selten und nur wenn kein Win-Pfad bekannt
    if (self.use_llm_planner
            and self.total_steps % self.llm_call_interval == 0
            and self.total_steps > 20
            and not self.state_graph.should_navigate()):
        action, data = self._consult_llm_planner(action, data)
        action_str = self._action_to_str(action, data)

    # Aktion ausführen
    previous_obs = self.current_obs
    self.current_obs = self.adapter.act(action, data)
    
    # In Memory + Audit + Graph aufzeichnen
    self._record_step(previous_obs, action_str, data)

    # Goals selten aktualisieren
    if self.total_steps % 5 == 0:
        self.goals.analyze_win_condition(self.memory)

    self.total_steps += 1
    return self._check_game_state()


def _record_step(self, previous_obs, action_str, data):
    """Transition in Memory, Audit UND State Graph aufzeichnen."""
    full_action = self._action_to_str(
        self._parse_action(action_str), data
    ) if data else action_str
    
    self.memory.record_transition(previous_obs, full_action, self.current_obs)
    self.audit_trail.log_step(
        level=self.current_level, step=self.total_steps,
        action=full_action, game_state=str(self.current_obs.game_state),
        pixels_changed=self.current_obs.changed_pixels,
    )
    self.state_graph.add_transition(
        from_grid=previous_obs.raw_grid, action_str=action_str,
        action_data=data if data else None, to_grid=self.current_obs.raw_grid,
        pixels_changed=self.current_obs.changed_pixels,
        game_state=str(self.current_obs.game_state), level=self.current_level,
    )


def _check_game_state(self) -> str:
    if self.current_obs.game_state == GameState.WIN:
        return "WIN"
    elif self.current_obs.game_state == GameState.GAME_OVER:
        return "GAME_OVER"
    return "CONTINUE"


def _parse_action(self, action_str: str):
    try:
        return GameAction[action_str]
    except (KeyError, AttributeError):
        return GameAction.ACTION1
```

**4. `_on_level_complete()` erweitern — am Anfang hinzufügen:**

```python
# State Graph: Aktionsmuster behalten, States löschen
self.state_graph.prepare_for_new_level()

# Navigation zurücksetzen
self._navigation_mode = False
self._current_path = []
self._path_index = 0
```

**5. LLM-Prompt erweitern — in `_consult_llm_planner()` hinzufügen:**

```python
graph_summary = self.state_graph.get_summary_for_llm()

# Im Prompt ergänzen:
# STATE GRAPH:
# {graph_summary}
```

### 6.4 Tests für State Graph Navigator

Erstelle `cognithor/modules/arc_agi3/tests/test_state_graph.py`:

```python
import pytest
import numpy as np
from cognithor.modules.arc_agi3.state_graph import StateGraphNavigator


class TestStateGraphNavigator:

    def setup_method(self):
        self.graph = StateGraphNavigator(max_states=1000)

    def test_add_transition(self):
        g1 = np.zeros((64, 64, 3), dtype=np.uint8)
        g2 = np.ones((64, 64, 3), dtype=np.uint8)
        h1, h2 = self.graph.add_transition(g1, "ACTION1", None, g2, 100, "PLAYING")
        assert h1 != h2
        assert len(self.graph.nodes) == 2
        assert self.graph.total_edges == 1

    def test_find_win_path_direct(self):
        g1 = np.zeros((64, 64, 3), dtype=np.uint8)
        g2 = np.ones((64, 64, 3), dtype=np.uint8)
        h1, h2 = self.graph.add_transition(g1, "ACTION2", None, g2, 200, "WIN")
        path = self.graph.find_win_path(h1)
        assert path is not None
        assert len(path) == 1
        assert path[0][0] == "ACTION2"

    def test_find_win_path_multi_step(self):
        g1 = np.full((64, 64, 3), 0, dtype=np.uint8)
        g2 = np.full((64, 64, 3), 1, dtype=np.uint8)
        g3 = np.full((64, 64, 3), 2, dtype=np.uint8)
        self.graph.add_transition(g1, "ACTION1", None, g2, 50, "PLAYING")
        self.graph.add_transition(g2, "ACTION3", None, g3, 100, "WIN")
        path = self.graph.find_win_path(self.graph.hash_grid(g1))
        assert path is not None and len(path) == 2

    def test_avoids_game_over_states(self):
        g1 = np.full((64, 64, 3), 0, dtype=np.uint8)
        g_danger = np.full((64, 64, 3), 1, dtype=np.uint8)
        g_safe = np.full((64, 64, 3), 2, dtype=np.uint8)
        g_win = np.full((64, 64, 3), 3, dtype=np.uint8)
        self.graph.add_transition(g1, "ACTION1", None, g_danger, 50, "GAME_OVER")
        self.graph.add_transition(g_danger, "ACTION2", None, g_win, 100, "WIN")
        self.graph.add_transition(g1, "ACTION3", None, g_safe, 30, "PLAYING")
        self.graph.add_transition(g_safe, "ACTION4", None, g_win, 100, "WIN")
        path = self.graph.find_win_path(self.graph.hash_grid(g1))
        assert path is not None and path[0][0] == "ACTION3"  # Sicherer Pfad

    def test_exploration_prioritizes_untested(self):
        g1 = np.zeros((64, 64, 3), dtype=np.uint8)
        g2 = np.ones((64, 64, 3), dtype=np.uint8)
        self.graph.add_transition(g1, "ACTION1", None, g2, 50, "PLAYING")
        result = self.graph.get_best_exploration_action(
            self.graph.hash_grid(g1), ["ACTION1", "ACTION2", "ACTION3"])
        assert result is not None and result[0] in ["ACTION2", "ACTION3"]

    def test_level_transfer(self):
        g1 = np.zeros((64, 64, 3), dtype=np.uint8)
        g_win = np.ones((64, 64, 3), dtype=np.uint8)
        self.graph.add_transition(g1, "ACTION2", None, g_win, 200, "WIN")
        self.graph.prepare_for_new_level()
        assert len(self.graph.nodes) == 0
        assert self.graph.action_patterns_from_previous.get("ACTION2", 0) > 0

    def test_max_states_limit(self):
        graph = StateGraphNavigator(max_states=5)
        for i in range(10):
            g = np.full((64, 64, 3), i, dtype=np.uint8)
            g_next = np.full((64, 64, 3), i+1, dtype=np.uint8)
            graph.add_transition(g, "ACTION1", None, g_next, 10, "PLAYING")
        assert len(graph.nodes) <= 5

    def test_no_win_states(self):
        g1 = np.zeros((64, 64, 3), dtype=np.uint8)
        assert self.graph.find_win_path(self.graph.hash_grid(g1)) is None

    def test_coverage_stats(self):
        g1 = np.zeros((64, 64, 3), dtype=np.uint8)
        g2 = np.ones((64, 64, 3), dtype=np.uint8)
        self.graph.add_transition(g1, "ACTION1", None, g2, 50, "PLAYING")
        cov = self.graph.get_exploration_coverage()
        assert cov["states"] == 2
        assert cov["edges"] == 1
```

### 6.5 Verifikation nach Integration

Nach der State-Graph-Integration, führe diesen Diagnose-Run aus:

```python
from cognithor.modules.arc_agi3.agent import CognithorArcAgent
import logging
logging.basicConfig(level=logging.INFO)

agent = CognithorArcAgent(game_id="ls20", use_llm_planner=True,
    llm_call_interval=20, max_steps_per_level=300)

scorecard = agent.run()

print(f"\n{'='*50}")
print(f"Score: {scorecard}")
print(f"Steps: {agent.total_steps}")
print(f"Level: {agent.current_level}")
print(f"Graph Stats: {agent.state_graph.get_exploration_coverage()}")
print(f"Navigation benutzt: {agent._navigation_mode or agent._path_index > 0}")
print(f"{'='*50}")
```

**Erwartete Verbesserung:**

| Metrik | Vor State Graph | Nach State Graph |
|---|---|---|
| Score | 0.003 | > 0.01 |
| Steps pro Level | ~500 | ~120 |
| LLM-Calls pro Level | ~50 | ~5 |
| Win-States gefunden | ? | > 0 |

Wenn der Score NICHT gestiegen ist, prüfe:
1. Wird `_record_step()` aufgerufen? (Graph muss gefüttert werden)
2. Werden Win-States im Graph registriert? (`graph.win_states` nicht leer?)
3. Findet `find_win_path()` einen Pfad? (BFS-Ergebnis loggen)
4. Wechselt der Agent in Navigation-Mode? (`_navigation_mode` loggen)

---

# ZUSAMMENFASSUNG: Erwartete Score-Entwicklung

| Schritt | Erwarteter Score | Faktor |
|---|---|---|
| Ausgangslage | 0.003 | 1× |
| Nach Modul-Fixes (Teil 2) | > 0.01 | 3× |
| Nach State Graph (Teil 3) | > 0.03 | 10× |
| Zukünftig: + CNN Predictor | > 0.08 | 27× |
