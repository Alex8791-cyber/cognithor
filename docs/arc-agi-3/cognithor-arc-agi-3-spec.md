# Cognithor × ARC-AGI-3 Integration — Technische Spezifikation

**Version:** 2.1  
**Datum:** 29. März 2026  
**Ziel:** Cognithor als ARC-AGI-3-Agent lauffähig machen und gleichzeitig die Architektur um fehlende Fähigkeiten erweitern  
**Voraussetzung:** Cognithor v0.57.0, Python 3.12+, RTX 5090 für lokales RL-Training  
**Repo:** github.com/Alex8791-cyber/cognithor | Lizenz: Apache 2.0  
**Startup:** `start_cognithor.bat` (Windows) / `install.sh` (Linux/Mac)

---

## ⚠️ PHASE 0 — SDK-VALIDIERUNG (VOR ALLEM ANDEREN AUSFÜHREN)

**AN DAS CODING-MODELL:** Diese Spezifikation enthält Annahmen über das ARC-AGI-3 SDK, die NOCH NICHT gegen das echte SDK validiert wurden. Bevor du irgendeinen Code aus dieser Spec implementierst, führe ZUERST die folgenden Validierungsschritte aus. Die Ergebnisse bestimmen, ob der Code in den Modulen 3.1–3.6 angepasst werden muss.

### Phase 0.1: SDK installieren und Objekte inspizieren

```bash
pip install arc-agi --break-system-packages
# oder
uv add arc-agi
```

```python
"""
VALIDIERUNGSSCRIPT — Ausführen und Output dokumentieren.
Jede print-Zeile ist eine Annahme, die bestätigt oder korrigiert werden muss.
"""
import arc_agi
from arcengine import GameAction, GameState
import inspect

# 1. Arcade-Klasse inspizieren
arc = arc_agi.Arcade()
print("=== Arcade Methoden ===")
print([m for m in dir(arc) if not m.startswith('_')])

# 2. Environment erstellen (ls20 ist das öffentliche Demo-Game)
env = arc.make("ls20")
print(f"\n=== Environment Typ: {type(env)} ===")
print(f"Methoden: {[m for m in dir(env) if not m.startswith('_')]}")

# 3. KRITISCH: FrameDataRaw Struktur validieren
obs = env.reset()
print(f"\n=== Observation Typ: {type(obs)} ===")
print(f"Attribute: {[a for a in dir(obs) if not a.startswith('_')]}")

# 4. KRITISCH: Wie sieht das Grid aus?
# Annahme in der Spec: obs.frame ist ein Array, reshape auf 64×64
# Das muss validiert werden!
if hasattr(obs, 'frame'):
    print(f"\nobs.frame Typ: {type(obs.frame)}")
    print(f"obs.frame Länge/Shape: {len(obs.frame) if hasattr(obs.frame, '__len__') else 'N/A'}")
    if hasattr(obs.frame, 'shape'):
        print(f"obs.frame.shape: {obs.frame.shape}")
    # Ersten Pixel inspizieren
    print(f"Erster Wert: {obs.frame[0] if obs.frame else 'LEER'}")
elif hasattr(obs, 'frame_data'):
    print(f"\nobs.frame_data Typ: {type(obs.frame_data)}")
    print(f"obs.frame_data: {str(obs.frame_data)[:500]}")
else:
    print("\n⚠️ WEDER obs.frame NOCH obs.frame_data GEFUNDEN!")
    print(f"Alle Attribute mit Werten:")
    for attr in dir(obs):
        if not attr.startswith('_'):
            val = getattr(obs, attr)
            if not callable(val):
                print(f"  {attr}: {type(val)} = {str(val)[:200]}")

# 5. GameState-Werte validieren
print(f"\n=== GameState Werte ===")
print([s for s in dir(GameState) if not s.startswith('_')])
print(f"obs.state = {obs.state}")

# 6. GameAction-Werte validieren
print(f"\n=== GameAction Werte ===")
all_actions = [a for a in GameAction]
print(f"Alle Actions: {all_actions}")
for a in all_actions:
    print(f"  {a}: is_simple={a.is_simple() if hasattr(a, 'is_simple') else '?'}, "
          f"is_complex={a.is_complex() if hasattr(a, 'is_complex') else '?'}")

# 7. action_space validieren
print(f"\n=== Action Space ===")
if hasattr(env, 'action_space'):
    print(f"env.action_space: {env.action_space}")
else:
    print("⚠️ env.action_space NICHT GEFUNDEN")

# 8. observation_space validieren
print(f"\n=== Observation Space ===")
if hasattr(env, 'observation_space'):
    obs_space = env.observation_space
    print(f"Typ: {type(obs_space)}")
    print(f"Attribute: {[a for a in dir(obs_space) if not a.startswith('_')]}")
else:
    print("⚠️ env.observation_space NICHT GEFUNDEN")

# 9. Step-Return validieren
obs2 = env.step(GameAction.ACTION1)
print(f"\n=== Step Return ===")
print(f"Typ: {type(obs2)}")
print(f"State nach ACTION1: {obs2.state if obs2 else 'None'}")

# 10. Scorecard-Format
sc = arc.get_scorecard()
print(f"\n=== Scorecard ===")
print(f"Typ: {type(sc)}")
if sc:
    print(f"Attribute: {[a for a in dir(sc) if not a.startswith('_')]}")
    if hasattr(sc, 'score'):
        print(f"Score: {sc.score}")

# 11. Recording-Support prüfen
print(f"\n=== Recording Support ===")
print(f"make() akzeptiert save_recording: {'save_recording' in inspect.signature(arc.make).parameters}")
```

### Phase 0.2: Ergebnisse in Anpassungen übersetzen

Nach Ausführung des Validierungsscripts, prüfe folgende Annahmen und passe den Code in den Modulen 3.1–3.6 an:

| Annahme in der Spec | Zu prüfen | Wenn falsch → anpassen in |
|---|---|---|
| Grid liegt in `obs.frame` | Attribut-Name und Typ | `adapter.py: _extract_grid()` |
| Grid ist 64×64×3 RGB | Shape und Datentyp | `adapter.py`, `visual_encoder.py`, `cnn_model.py` |
| `GameAction.ACTION6` ist die einzige complex action | `is_complex()` für alle Actions | `explorer.py: initialize_discovery()` |
| `obs.state` nutzt `GameState.WIN` / `GameState.GAME_OVER` | Enum-Werte | `agent.py: _step()`, `episode_memory.py` |
| `obs.levels_completed` existiert | Attribut vorhanden | `adapter.py: _process_frame()` |
| `env.action_space` gibt Liste zurück | Typ und Inhalt | `explorer.py`, `agent.py` |
| `arc.get_scorecard().score` existiert | Attribut-Struktur | `agent.py: run()` |
| `env.step()` gibt `FrameDataRaw` zurück | Return-Typ | `adapter.py: act()` |
| `GameAction.RESET` existiert und resettet das Level | Verhalten testen | `agent.py`, `explorer.py` |
| `arc.make()` akzeptiert `save_recording=True` | Parameter prüfen | Recording-Integration |
| ARC-Farben sind 0–9 Index | Tatsächliches Farbformat | `visual_encoder.py: color_names` |

**WICHTIG:** Führe KEINE Module-Implementierung durch, bevor diese Tabelle ausgefüllt ist. Jede falsche Annahme zieht sich durch den gesamten Code.

### Phase 0.3: FrameData vs. FrameDataRaw Unterscheidung

Die ARC-AGI-3-Agents Repo nutzt `FrameData` (ein Wrapper), das offizielle SDK nutzt `FrameDataRaw`. Diese haben möglicherweise unterschiedliche Felder:

```python
# Prüfe BEIDE Klassen
from arcengine import FrameDataRaw
try:
    from agents.structs import FrameData  # Nur im ARC-AGI-3-Agents Repo
    print("FrameData Felder:", FrameData.__fields__ if hasattr(FrameData, '__fields__') else dir(FrameData))
except ImportError:
    print("FrameData nicht verfügbar (ARC-AGI-3-Agents Repo nicht geklont)")

print("FrameDataRaw Felder:", [a for a in dir(FrameDataRaw) if not a.startswith('_')])
```

Der `CognithorAgent` Wrapper (Abschnitt 5) nutzt `FrameData` vom Agents-Repo. Der `CognithorArcAgent` (Abschnitt 3.6) nutzt `FrameDataRaw` direkt. Stelle sicher, dass beide korrekt gemappt sind.

---

## 1. Ausgangslage: Was ARC-AGI-3 verlangt

ARC-AGI-3 ist ein interaktiver Benchmark. Der Agent wird in ein unbekanntes, rundenbasiertes Environment (64×64 Farbgrid) geworfen — ohne Instruktionen, ohne Regeln, ohne Zielangabe. Er muss selbstständig:

- **Explorieren:** Aktionen ausprobieren und beobachten, was sich ändert
- **Modellieren:** Aus Beobachtungen ein Weltmodell ableiten (welche Aktion hat welchen Effekt?)
- **Ziele erkennen:** Ohne Hinweis herausfinden, was "Gewinnen" bedeutet
- **Planen und Ausführen:** Effizient handeln — Scoring basiert auf RHAE = (human_steps / agent_steps)², d.h. doppelt so viele Schritte wie ein Mensch → nur 25% Score

Jedes Game hat 8–10 Level mit steigender Komplexität. Neue Mechaniken kommen pro Level dazu.

### ARC-AGI-3 API-Interface (das muss Cognithor bedienen)

```python
# Kernabhängigkeiten
# pip install arc-agi
import arc_agi
from arcengine import GameAction, GameState, FrameDataRaw

# Initialisierung
arc = arc_agi.Arcade()
env = arc.make("ls20")  # Environment laden

# Game Loop
obs = env.reset()       # Initialen Frame holen
action = GameAction.ACTION1  # ACTION1-ACTION6 (simple) + ACTION6 mit x,y (complex)
obs = env.step(action, data={"x": 32, "y": 32})  # Step ausführen

# Observation auswerten
obs.state       # GameState.PLAYING / WIN / GAME_OVER / NOT_PLAYED
obs.frame_data  # 64×64 Grid als Pixel-Array

# Scorecard
scorecard = arc.get_scorecard()
```

**Aktionsraum:** RESET, ACTION1–ACTION6 (simple), ACTION6 mit x,y Koordinaten (0–63) als complex action  
**Observation:** 64×64 Farbgrid pro Frame + GameState  
**Framerate:** >2.000 FPS ohne Rendering möglich

---

## 2. Gap-Analyse: Cognithor-Architektur vs. ARC-AGI-3-Anforderungen

| ARC-AGI-3 Fähigkeit | Cognithor Status | Lücke | Priorität |
|---|---|---|---|
| **Exploration** | Autonomous Evolution Engine existiert, aber für idle-time learning | Keine gezielte, hypothesengetriebene Exploration in Echtzeit | KRITISCH |
| **Weltmodell** | 5-Tier Memory speichert Langzeitwissen | Kein episodisches Arbeitsgedächtnis für In-Session-Lernen | KRITISCH |
| **Goal Inference** | Ziele kommen vom User (Prompt-basiert) | Kein autonomes Goal-Setting aus Beobachtungen | KRITISCH |
| **Planning** | Planner in PGE Trinity | Planner arbeitet Prompt-to-Plan, nicht State-to-Plan | MITTEL |
| **Visuelles Verständnis** | MCP Tools verarbeiten Text/API | Kein Grid/Pixel-Verständnis, kein visueller Encoder | KRITISCH |
| **Feedback Loop** | Executor liefert Ergebnis, Gatekeeper prüft | Kein geschlossener Observe→Act→Learn-Zyklus | HOCH |
| **Effizienz (RHAE)** | Keine Optimierung auf Aktionseffizienz | Braucht Aktionszähler + Effizienz-Bewertung | MITTEL |

---

## 3. Neue Module — Detaillierte Spezifikation

### 3.1 ARC-AGI-3 Environment Adapter

**Zweck:** Brücke zwischen ARC-AGI-3 SDK und Cognithor-Internals. Übersetzt Grid-Observations in Cognithor-kompatible Datenstrukturen und Cognithor-Entscheidungen zurück in GameActions.

**Datei:** `cognithor/modules/arc_agi3/adapter.py`

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import arc_agi
from arcengine import GameAction, GameState, FrameDataRaw


@dataclass
class ArcObservation:
    """Cognithor-interne Repräsentation eines ARC-AGI-3 Frames."""
    raw_grid: np.ndarray          # 64×64×3 RGB oder Farbindex-Array
    game_state: GameState
    step_number: int
    level: int
    levels_completed: int
    grid_diff: Optional[np.ndarray] = None   # Differenz zum vorherigen Frame
    changed_pixels: int = 0
    action_history: list = field(default_factory=list)


class ArcEnvironmentAdapter:
    """Adapter zwischen ARC-AGI-3 SDK und Cognithor Agent Loop."""

    def __init__(self, game_id: str, api_key: Optional[str] = None):
        self.game_id = game_id
        self.arcade = arc_agi.Arcade()
        self.env = None
        self.current_obs: Optional[ArcObservation] = None
        self.previous_grid: Optional[np.ndarray] = None
        self.step_count = 0
        self.level_step_count = 0
        self.total_resets = 0

    def initialize(self) -> ArcObservation:
        """Environment starten und ersten Frame holen."""
        self.env = self.arcade.make(self.game_id)
        if self.env is None:
            raise RuntimeError(f"Game {self.game_id} konnte nicht geladen werden")
        raw = self.env.reset()
        return self._process_frame(raw, action=None)

    def act(self, action: GameAction, data: dict = None) -> ArcObservation:
        """Aktion ausführen und neuen Zustand zurückgeben."""
        raw = self.env.step(action, data=data or {})
        self.step_count += 1
        self.level_step_count += 1
        return self._process_frame(raw, action)

    def reset_level(self) -> ArcObservation:
        """Level zurücksetzen (bei GAME_OVER)."""
        raw = self.env.step(GameAction.RESET)
        self.total_resets += 1
        self.level_step_count = 0
        return self._process_frame(raw, action=GameAction.RESET)

    def _process_frame(self, raw: FrameDataRaw, action) -> ArcObservation:
        """Frame in Cognithor-Observation umwandeln + Diff berechnen."""
        grid = self._extract_grid(raw)
        diff = None
        changed = 0

        if self.previous_grid is not None:
            diff = grid != self.previous_grid
            changed = int(np.sum(diff))

        obs = ArcObservation(
            raw_grid=grid,
            game_state=raw.state if raw else GameState.NOT_PLAYED,
            step_number=self.step_count,
            level=raw.levels_completed if raw else 0,
            levels_completed=raw.levels_completed if raw else 0,
            grid_diff=diff,
            changed_pixels=changed,
        )

        if action is not None:
            obs.action_history = (
                self.current_obs.action_history + [(action, changed)]
                if self.current_obs else [(action, changed)]
            )

        self.previous_grid = grid.copy()
        self.current_obs = obs
        return obs

    def _extract_grid(self, raw: FrameDataRaw) -> np.ndarray:
        """Grid aus FrameDataRaw extrahieren. Format: 64×64 Array."""
        # Exakte Extraktion hängt vom FrameDataRaw-Format ab
        # frame_data enthält das Pixel-Array
        if raw and hasattr(raw, 'frame') and raw.frame is not None:
            return np.array(raw.frame, dtype=np.uint8).reshape(64, 64, -1)
        return np.zeros((64, 64, 3), dtype=np.uint8)

    def get_scorecard(self):
        return self.arcade.get_scorecard()
```

---

### 3.2 Episodisches Arbeitsgedächtnis (Episode Memory)

**Zweck:** Cognithor's 5-Tier Memory speichert Wissen über Sessions hinweg. ARC-AGI-3 braucht ein *kurzlebiges* Gedächtnis, das innerhalb eines Games lernt: welche Aktion hatte welchen Effekt, welche Zustände wurden besucht, welche Hypothesen wurden getestet.

**Datei:** `cognithor/modules/arc_agi3/episode_memory.py`

```python
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib
import numpy as np


@dataclass
class StateTransition:
    """Eine beobachtete Zustandsänderung."""
    state_hash: str               # Hash des Grids VOR der Aktion
    action: str                   # z.B. "ACTION1", "ACTION6_32_15"
    next_state_hash: str          # Hash des Grids NACH der Aktion
    pixels_changed: int           # Anzahl geänderter Pixel
    resulted_in_win: bool = False
    resulted_in_game_over: bool = False
    level: int = 0


@dataclass
class Hypothesis:
    """Eine Hypothese über die Spielmechanik."""
    description: str              # z.B. "ACTION1 bewegt Objekt nach rechts"
    supporting_evidence: int = 0  # Wie oft bestätigt
    contradicting_evidence: int = 0
    confidence: float = 0.0       # supporting / (supporting + contradicting)
    tested_at_steps: list = field(default_factory=list)


class EpisodeMemory:
    """Kurzzeit-Gedächtnis für eine ARC-AGI-3 Game-Session."""

    def __init__(self, max_transitions: int = 200_000):
        self.transitions: list[StateTransition] = []
        self.state_visit_count: dict[str, int] = defaultdict(int)
        self.action_effect_map: dict[str, dict] = defaultdict(
            lambda: {"total": 0, "caused_change": 0, "caused_win": 0, "caused_game_over": 0}
        )
        self.hypotheses: list[Hypothesis] = []
        self.visited_states: set[str] = set()
        self.max_transitions = max_transitions
        self._state_hash_cache: dict[bytes, str] = {}

    def hash_grid(self, grid: np.ndarray) -> str:
        """Schneller Hash eines 64×64 Grids für Zustandsvergleich."""
        grid_bytes = grid.tobytes()
        if grid_bytes not in self._state_hash_cache:
            self._state_hash_cache[grid_bytes] = hashlib.md5(grid_bytes).hexdigest()[:16]
        return self._state_hash_cache[grid_bytes]

    def record_transition(self, obs_before, action_str: str, obs_after) -> StateTransition:
        """Zustandsübergang aufzeichnen und Effekt-Map aktualisieren."""
        s_hash = self.hash_grid(obs_before.raw_grid)
        ns_hash = self.hash_grid(obs_after.raw_grid)

        t = StateTransition(
            state_hash=s_hash,
            action=action_str,
            next_state_hash=ns_hash,
            pixels_changed=obs_after.changed_pixels,
            resulted_in_win=(obs_after.game_state == "WIN"),
            resulted_in_game_over=(obs_after.game_state == "GAME_OVER"),
            level=obs_after.level,
        )

        if len(self.transitions) < self.max_transitions:
            self.transitions.append(t)

        self.state_visit_count[s_hash] += 1
        self.visited_states.add(s_hash)
        self.visited_states.add(ns_hash)

        # Effekt-Map updaten
        effects = self.action_effect_map[action_str]
        effects["total"] += 1
        if t.pixels_changed > 0:
            effects["caused_change"] += 1
        if t.resulted_in_win:
            effects["caused_win"] += 1
        if t.resulted_in_game_over:
            effects["caused_game_over"] += 1

        return t

    def get_action_effectiveness(self, action_str: str) -> float:
        """Wie oft hat diese Aktion eine sichtbare Änderung bewirkt? (0.0–1.0)"""
        e = self.action_effect_map[action_str]
        if e["total"] == 0:
            return 0.5  # Unbekannt → neutral
        return e["caused_change"] / e["total"]

    def get_unexplored_actions(self, current_state_hash: str, all_actions: list[str]) -> list[str]:
        """Welche Aktionen wurden in diesem Zustand noch nicht getestet?"""
        tested = set()
        for t in self.transitions:
            if t.state_hash == current_state_hash:
                tested.add(t.action)
        return [a for a in all_actions if a not in tested]

    def is_novel_state(self, grid: np.ndarray) -> bool:
        """Wurde dieser Zustand schon besucht?"""
        return self.hash_grid(grid) not in self.visited_states

    def add_hypothesis(self, description: str) -> Hypothesis:
        h = Hypothesis(description=description)
        self.hypotheses.append(h)
        return h

    def clear_for_new_level(self):
        """Bei neuem Level: Transitions behalten (cross-level learning),
        aber Level-spezifische State-Visits zurücksetzen."""
        self.state_visit_count.clear()
        # Transitions und action_effect_map bleiben — die Mechaniken
        # gelten oft über Level hinweg

    def get_summary_for_llm(self) -> str:
        """Kompakte Zusammenfassung für LLM-Prompt (Context-Budget beachten)."""
        lines = []
        lines.append(f"Besuchte Zustände: {len(self.visited_states)}")
        lines.append(f"Aufgezeichnete Übergänge: {len(self.transitions)}")
        lines.append("\nAktions-Effektivität:")
        for action, effects in sorted(self.action_effect_map.items()):
            total = effects["total"]
            if total > 0:
                change_rate = effects["caused_change"] / total
                lines.append(
                    f"  {action}: {total}x getestet, "
                    f"{change_rate:.0%} verursachten Änderung, "
                    f"{effects['caused_win']}x Win, "
                    f"{effects['caused_game_over']}x Game Over"
                )
        if self.hypotheses:
            lines.append("\nAktive Hypothesen:")
            for h in self.hypotheses:
                lines.append(f"  [{h.confidence:.0%}] {h.description}")
        return "\n".join(lines)
```

---

### 3.3 Goal Inference Module (GIM)

**Zweck:** Autonomes Erkennen des Spielziels. Sitzt zwischen Gatekeeper und Planner in der PGE Trinity. Analysiert Zustandsübergänge und leitet mögliche Ziele ab.

**Datei:** `cognithor/modules/arc_agi3/goal_inference.py`

```python
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class GoalType(Enum):
    UNKNOWN = "unknown"
    REACH_STATE = "reach_state"         # Bestimmten Zustand erreichen
    CLEAR_BOARD = "clear_board"         # Alle Elemente entfernen
    FILL_PATTERN = "fill_pattern"       # Muster vervollständigen
    NAVIGATE = "navigate"              # Objekt an Zielposition bringen
    AVOID = "avoid"                    # Bestimmte Zustände vermeiden
    SEQUENCE = "sequence"              # Aktionen in richtiger Reihenfolge


@dataclass
class InferredGoal:
    """Ein abgeleitetes Spielziel mit Konfidenz."""
    goal_type: GoalType
    description: str
    confidence: float                  # 0.0–1.0
    evidence: list[str] = field(default_factory=list)
    estimated_steps_remaining: int = -1


class GoalInferenceModule:
    """Leitet aus Beobachtungen und Episoden-Gedächtnis mögliche Ziele ab."""

    def __init__(self):
        self.current_goals: list[InferredGoal] = []
        self.win_states_observed: list[np.ndarray] = []
        self.game_over_states_observed: list[np.ndarray] = []
        self._level_progression_data: list[dict] = []

    def analyze_win_condition(self, episode_memory) -> list[InferredGoal]:
        """Hauptanalyse: Was führt zum Gewinnen?"""
        goals = []

        # Strategie 1: Win-Transitions analysieren
        win_transitions = [t for t in episode_memory.transitions if t.resulted_in_win]
        if win_transitions:
            # Welche Aktionen führten zum Win?
            win_actions = [t.action for t in win_transitions]
            most_common_win_action = max(set(win_actions), key=win_actions.count)
            goals.append(InferredGoal(
                goal_type=GoalType.REACH_STATE,
                description=f"Win-Zustand wird am häufigsten durch {most_common_win_action} erreicht",
                confidence=len(win_transitions) / max(len(episode_memory.transitions), 1),
                evidence=[f"{len(win_transitions)} Win-Übergänge beobachtet"],
            ))

        # Strategie 2: Game-Over-Muster analysieren (was NICHT tun)
        go_transitions = [t for t in episode_memory.transitions if t.resulted_in_game_over]
        if go_transitions:
            dangerous_actions = [t.action for t in go_transitions]
            goals.append(InferredGoal(
                goal_type=GoalType.AVOID,
                description=f"Vermeide häufige Game-Over-Auslöser: {set(dangerous_actions)}",
                confidence=len(go_transitions) / max(len(episode_memory.transitions), 1),
                evidence=[f"{len(go_transitions)} Game-Over-Übergänge beobachtet"],
            ))

        # Strategie 3: Pixel-Veränderungs-Muster
        if episode_memory.transitions:
            avg_change_on_progress = np.mean([
                t.pixels_changed for t in episode_memory.transitions
                if t.pixels_changed > 0 and not t.resulted_in_game_over
            ]) if any(t.pixels_changed > 0 for t in episode_memory.transitions) else 0

            if avg_change_on_progress > 100:
                goals.append(InferredGoal(
                    goal_type=GoalType.CLEAR_BOARD,
                    description="Große Pixel-Änderungen bei Fortschritt → möglicherweise Board-Clearing",
                    confidence=0.3,
                    evidence=[f"Durchschnitt {avg_change_on_progress:.0f} Pixel bei erfolgreichen Aktionen"],
                ))

        # Strategie 4: Wenn keine Wins beobachtet → weiter explorieren
        if not win_transitions:
            goals.append(InferredGoal(
                goal_type=GoalType.UNKNOWN,
                description="Noch kein Win beobachtet — explorative Phase fortsetzen",
                confidence=0.1,
                evidence=["Kein Win-Zustand in Episoden-Gedächtnis"],
            ))

        self.current_goals = sorted(goals, key=lambda g: g.confidence, reverse=True)
        return self.current_goals

    def get_best_goal(self) -> InferredGoal:
        if not self.current_goals:
            return InferredGoal(
                goal_type=GoalType.UNKNOWN,
                description="Keine Ziel-Hypothese verfügbar",
                confidence=0.0,
            )
        return self.current_goals[0]

    def on_level_complete(self, level_data: dict):
        """Daten nach Level-Abschluss speichern für Cross-Level-Lernen."""
        self._level_progression_data.append(level_data)

    def get_summary_for_llm(self) -> str:
        lines = ["Aktuelle Ziel-Hypothesen:"]
        for g in self.current_goals[:3]:  # Max 3 für Context-Budget
            lines.append(f"  [{g.confidence:.0%}] {g.goal_type.value}: {g.description}")
        return "\n".join(lines)
```

---

### 3.4 Hypothesengetriebener Explorer

**Zweck:** Ersetzt zufällige Exploration durch systematisches, hypothesenbasiertes Testen. Nutzt Episode Memory für informierte Entscheidungen.

**Datei:** `cognithor/modules/arc_agi3/explorer.py`

```python
from enum import Enum
import random
import numpy as np
from arcengine import GameAction


class ExplorationPhase(Enum):
    DISCOVERY = "discovery"       # Phase 1: Alle Aktionen systematisch testen
    HYPOTHESIS = "hypothesis"     # Phase 2: Gezielte Hypothesen-Tests
    EXPLOITATION = "exploitation" # Phase 3: Bekanntes Wissen ausnutzen


class HypothesisDrivenExplorer:
    """Systematische Exploration mit Phase-Transitions."""

    def __init__(self):
        self.phase = ExplorationPhase.DISCOVERY
        self.discovery_queue: list[str] = []
        self.action_test_grid: dict[str, set] = {}  # action → set of states where tested
        self._phase_step_count = 0
        self._total_actions_tested = 0

    def initialize_discovery(self, action_space: list[GameAction]):
        """Discovery-Queue aufbauen: Jede Aktion einmal testen."""
        simple_actions = [a for a in action_space if a.is_simple() and a != GameAction.RESET]
        # Für complex actions: Grid-Sampling (nicht alle 4096 Positionen)
        complex_samples = []
        for a in action_space:
            if a.is_complex():
                # Strategisches Sampling: Ecken, Mitte, Kreuzpunkte
                key_positions = [
                    (0, 0), (0, 32), (0, 63),
                    (32, 0), (32, 32), (32, 63),
                    (63, 0), (63, 32), (63, 63),
                    (16, 16), (16, 48), (48, 16), (48, 48),
                ]
                for x, y in key_positions:
                    complex_samples.append((a, {"x": x, "y": y}))

        self.discovery_queue = (
            [(a, {}) for a in simple_actions] + complex_samples
        )

    def choose_action(
        self,
        current_obs,
        episode_memory,
        goal_module,
    ) -> tuple[GameAction, dict]:
        """Nächste Aktion basierend auf Phase und Wissen wählen."""
        self._phase_step_count += 1
        self._check_phase_transition(episode_memory, goal_module)

        if self.phase == ExplorationPhase.DISCOVERY:
            return self._discovery_action(current_obs, episode_memory)
        elif self.phase == ExplorationPhase.HYPOTHESIS:
            return self._hypothesis_action(current_obs, episode_memory, goal_module)
        else:
            return self._exploitation_action(current_obs, episode_memory, goal_module)

    def _discovery_action(self, obs, memory) -> tuple[GameAction, dict]:
        """Phase 1: Systematisch alle Aktionen durchprobieren."""
        # Ungetestete Aktionen in diesem Zustand bevorzugen
        state_hash = memory.hash_grid(obs.raw_grid)
        unexplored = memory.get_unexplored_actions(
            state_hash,
            [str(a) for a, _ in self.discovery_queue]
        )

        if unexplored and self.discovery_queue:
            # Nächste ungetestete Aktion aus Queue
            for i, (action, data) in enumerate(self.discovery_queue):
                if str(action) in unexplored:
                    self.discovery_queue.pop(i)
                    return action, data

        # Queue leer oder alles getestet → nächste Phase
        if self.discovery_queue:
            action, data = self.discovery_queue.pop(0)
            return action, data

        # Fallback: Zufällig
        return self._random_action()

    def _hypothesis_action(self, obs, memory, goals) -> tuple[GameAction, dict]:
        """Phase 2: Hypothesen gezielt testen."""
        best_goal = goals.get_best_goal()

        # Aktionen mit höchster Effektivität bevorzugen
        effectiveness = {}
        for action_str, effects in memory.action_effect_map.items():
            if effects["total"] > 0:
                # Score: Änderungsrate hoch, Game-Over-Rate niedrig
                change_rate = effects["caused_change"] / effects["total"]
                danger_rate = effects["caused_game_over"] / effects["total"]
                effectiveness[action_str] = change_rate * (1 - danger_rate * 2)

        if effectiveness:
            # Top-Aktionen mit etwas Exploration-Noise
            sorted_actions = sorted(effectiveness.items(), key=lambda x: x[1], reverse=True)
            # 80% beste Aktion, 20% zweit/drittbeste
            if random.random() < 0.8 and sorted_actions:
                best_action_str = sorted_actions[0][0]
                return self._parse_action_str(best_action_str)
            elif len(sorted_actions) > 1:
                runner_up = sorted_actions[random.randint(1, min(2, len(sorted_actions)-1))][0]
                return self._parse_action_str(runner_up)

        return self._random_action()

    def _exploitation_action(self, obs, memory, goals) -> tuple[GameAction, dict]:
        """Phase 3: Bekannte Win-Strategie ausführen."""
        win_transitions = [t for t in memory.transitions if t.resulted_in_win]
        if win_transitions:
            # Wiederhole die Aktion, die zum Win geführt hat
            last_win = win_transitions[-1]
            return self._parse_action_str(last_win.action)
        # Kein Win bekannt → zurück zu Hypothesis
        self.phase = ExplorationPhase.HYPOTHESIS
        return self._hypothesis_action(obs, memory, goals)

    def _check_phase_transition(self, memory, goals):
        """Automatischer Phasenwechsel basierend auf Wissensstand."""
        if self.phase == ExplorationPhase.DISCOVERY:
            # → HYPOTHESIS wenn: genug Aktionen getestet ODER Queue leer
            if not self.discovery_queue or self._phase_step_count > 50:
                self.phase = ExplorationPhase.HYPOTHESIS
                self._phase_step_count = 0

        elif self.phase == ExplorationPhase.HYPOTHESIS:
            # → EXPLOITATION wenn: Goal mit >60% Konfidenz gefunden
            best = goals.get_best_goal()
            if best.confidence > 0.6:
                self.phase = ExplorationPhase.EXPLOITATION
                self._phase_step_count = 0

    def _random_action(self) -> tuple[GameAction, dict]:
        actions = [a for a in GameAction if a != GameAction.RESET]
        action = random.choice(actions)
        data = {}
        if action.is_complex():
            data = {"x": random.randint(0, 63), "y": random.randint(0, 63)}
        return action, data

    def _parse_action_str(self, action_str: str) -> tuple[GameAction, dict]:
        """'ACTION6_32_15' → (GameAction.ACTION6, {"x": 32, "y": 15})"""
        parts = action_str.split("_")
        action_name = parts[0] if not parts[0].startswith("GameAction") else parts[0].split(".")[-1]

        try:
            action = GameAction[action_name]
        except KeyError:
            action = GameAction.ACTION1

        data = {}
        if len(parts) == 3 and action.is_complex():
            data = {"x": int(parts[1]), "y": int(parts[2])}
        return action, data
```

---

### 3.5 Visueller State Encoder

**Zweck:** Cognithor arbeitet textbasiert. ARC-AGI-3 liefert 64×64 Grids. Dieses Modul konvertiert visuelle Zustände in kompakte Beschreibungen, die der LLM-basierte Planner verarbeiten kann, und trainiert optional ein CNN-Modell für effizientere Repräsentationen.

**Datei:** `cognithor/modules/arc_agi3/visual_encoder.py`

```python
import numpy as np
from collections import Counter


class VisualStateEncoder:
    """Konvertiert 64×64 Grids in LLM-verständliche Beschreibungen."""

    def __init__(self):
        self.color_names = {
            0: "schwarz", 1: "blau", 2: "rot", 3: "grün",
            4: "gelb", 5: "grau", 6: "magenta", 7: "orange",
            8: "cyan", 9: "braun",
        }

    def encode_for_llm(self, grid: np.ndarray, diff: np.ndarray = None) -> str:
        """Grid in kompakte Textbeschreibung umwandeln.

        Strategie: Nicht das ganze 64×64 Grid beschreiben (4096 Pixel
        sprengen jeden Context), sondern:
        1. Farbverteilung (Histogramm)
        2. Regionen/Cluster erkennen
        3. Änderungen zum Vorgänger hervorheben
        """
        lines = []

        # 1. Farbhistogramm
        if grid.ndim == 3:
            # RGB → vereinfachter Farbindex
            flat = self._rgb_to_index(grid)
        else:
            flat = grid.flatten()

        color_counts = Counter(flat)
        total = len(flat)
        lines.append("Farbverteilung:")
        for color_idx, count in color_counts.most_common(5):
            name = self.color_names.get(color_idx, f"Farbe_{color_idx}")
            pct = count / total * 100
            lines.append(f"  {name}: {pct:.1f}%")

        # 2. Regionen erkennen (vereinfachte Connected-Component-Analyse)
        non_bg = flat != color_counts.most_common(1)[0][0]
        non_bg_count = np.sum(non_bg)
        if non_bg_count > 0 and non_bg_count < total * 0.5:
            grid_2d = flat.reshape(64, 64) if grid.ndim == 2 else self._rgb_to_index(grid)
            regions = self._find_bounding_boxes(grid_2d, background=color_counts.most_common(1)[0][0])
            if regions:
                lines.append(f"\nErkannte Objekte/Regionen: {len(regions)}")
                for i, (color, x1, y1, x2, y2) in enumerate(regions[:5]):
                    name = self.color_names.get(color, f"Farbe_{color}")
                    lines.append(f"  Region {i+1}: {name} bei ({x1},{y1})→({x2},{y2}), "
                               f"Größe {(x2-x1+1)}×{(y2-y1+1)}")

        # 3. Diff zum vorherigen Frame
        if diff is not None and np.any(diff):
            changed_count = np.sum(diff) if diff.dtype == bool else np.sum(diff != 0)
            lines.append(f"\nÄnderungen zum Vorgänger: {changed_count} Pixel")
            # Wo haben sich die Pixel geändert?
            if diff.dtype == bool and diff.ndim >= 2:
                changed_y, changed_x = np.where(diff[:, :, 0] if diff.ndim == 3 else diff)
                if len(changed_x) > 0:
                    lines.append(f"  Änderungs-Zentrum: ({np.mean(changed_x):.0f}, {np.mean(changed_y):.0f})")
                    lines.append(f"  Änderungs-Bereich: x[{np.min(changed_x)}–{np.max(changed_x)}], "
                               f"y[{np.min(changed_y)}–{np.max(changed_y)}]")

        return "\n".join(lines)

    def encode_compact(self, grid: np.ndarray) -> str:
        """Maximale Kompression: nur Farbverteilung + Objektanzahl.
        Für Context-Budget-kritische Situationen."""
        if grid.ndim == 3:
            flat = self._rgb_to_index(grid)
        else:
            flat = grid.flatten()
        color_counts = Counter(flat)
        top3 = color_counts.most_common(3)
        summary = ", ".join(
            f"{self.color_names.get(c, f'F{c}')}:{n}" for c, n in top3
        )
        return f"[{summary}]"

    def _rgb_to_index(self, grid: np.ndarray) -> np.ndarray:
        """RGB-Grid auf ARC-Farbindex mappen (vereinfacht)."""
        # Vereinfachtes Mapping basierend auf dominanter Farbkomponente
        if grid.shape[-1] >= 3:
            return np.argmax(grid[:, :, :3], axis=2).flatten()
        return grid.flatten()

    def _find_bounding_boxes(self, grid_2d, background=0, min_size=4):
        """Einfache Bounding-Box-Erkennung für zusammenhängende Regionen."""
        regions = []
        unique_colors = set(np.unique(grid_2d)) - {background}

        for color in list(unique_colors)[:5]:  # Max 5 Farben analysieren
            mask = grid_2d == color
            ys, xs = np.where(mask)
            if len(xs) >= min_size:
                regions.append((color, int(xs.min()), int(ys.min()),
                              int(xs.max()), int(ys.max())))

        return regions
```

---

### 3.6 Cognithor ARC Agent — Hauptsteuerung

**Zweck:** Orchestriert alle Module und implementiert das ARC-AGI-3 Agent-Interface. Verbindet PGE Trinity mit den neuen ARC-spezifischen Modulen.

**Datei:** `cognithor/modules/arc_agi3/agent.py`

```python
"""
Cognithor ARC-AGI-3 Agent
=========================
Implementiert das ARC-AGI-3-Agents Interface und verbindet:
  - ArcEnvironmentAdapter (SDK-Brücke)
  - EpisodeMemory (Kurzzeit-Lernen)
  - GoalInferenceModule (Autonomes Goal-Setting)
  - HypothesisDrivenExplorer (Systematische Exploration)
  - VisualStateEncoder (Grid → Text für LLM-Planner)
  - Cognithor PGE Trinity (Planner → Gatekeeper → Executor)
"""

import logging
from typing import Optional
from arcengine import GameAction, GameState

# Cognithor-Imports (existierende Module)
# from cognithor.core.planner import Planner
# from cognithor.core.gatekeeper import Gatekeeper
# from cognithor.core.executor import Executor

# Neue ARC-AGI-3-Module
from .episode_memory import EpisodeMemory
from .goal_inference import GoalInferenceModule
from .explorer import HypothesisDrivenExplorer, ExplorationPhase
from .visual_encoder import VisualStateEncoder
from .adapter import ArcEnvironmentAdapter, ArcObservation
from .mechanics_model import MechanicsModel
from .audit import ArcAuditTrail
from .error_handler import GameRunGuard, safe_frame_extract, retry_on_error

logger = logging.getLogger("cognithor.arc_agi3")


class CognithorArcAgent:
    """
    Hauptklasse: Cognithor als ARC-AGI-3 Agent.

    Architektur-Entscheidung: Hybrid-Ansatz
    ----------------------------------------
    - Exploration + Low-Level-Entscheidungen: Algorithmisch (Explorer + Memory)
    - Hypothesenbildung + Strategie: LLM via Cognithor Planner
    - Effizienz-Kritisch: CNN-Modell (optional, für Wettbewerb)

    Diese Architektur folgt der Erkenntnis aus der ARC-AGI-3 Preview:
    Einfache RL/Graph-Search-Ansätze schlagen Frontier-LLMs um 30×,
    weil LLMs pro Aktion zu viele Tokens verbrauchen und zu langsam sind.
    """

    def __init__(
        self,
        game_id: str,
        use_llm_planner: bool = True,
        llm_call_interval: int = 10,    # LLM nur alle N Steps konsultieren
        max_steps_per_level: int = 500,
        max_resets_per_level: int = 5,
    ):
        self.game_id = game_id
        self.use_llm_planner = use_llm_planner
        self.llm_call_interval = llm_call_interval
        self.max_steps_per_level = max_steps_per_level
        self.max_resets_per_level = max_resets_per_level

        # Module initialisieren
        self.adapter = ArcEnvironmentAdapter(game_id)
        self.memory = EpisodeMemory()
        self.goals = GoalInferenceModule()
        self.explorer = HypothesisDrivenExplorer()
        self.encoder = VisualStateEncoder()
        self.mechanics = MechanicsModel()
        self.audit_trail = ArcAuditTrail(game_id, agent_version="cognithor-arc-v1")

        # State
        self.current_obs: Optional[ArcObservation] = None
        self.current_level = 0
        self.level_resets = 0
        self.total_steps = 0

    def run(self) -> dict:
        """Komplettes Game durchspielen. Gibt Scorecard zurück."""
        logger.info(f"Starte Cognithor Agent für Game: {self.game_id}")
        self.audit_trail.log_game_start()

        # Environment initialisieren (mit Error Guard)
        self.current_obs = self.adapter.initialize()
        self.explorer.initialize_discovery(self.adapter.env.action_space)

        # Game Loop
        while True:
            # Schritt ausführen
            result = self._step()

            if result == "WIN":
                logger.info(f"Level {self.current_level} geschafft! "
                          f"Steps: {self.adapter.level_step_count}")
                self._on_level_complete()

            elif result == "GAME_OVER":
                self.level_resets += 1
                if self.level_resets >= self.max_resets_per_level:
                    logger.warning(f"Max Resets erreicht für Level {self.current_level}")
                    break
                self.current_obs = self.adapter.reset_level()
                self.memory.clear_for_new_level()

            elif result == "MAX_STEPS":
                logger.warning(f"Max Steps erreicht für Level {self.current_level}")
                break

            elif result == "DONE":
                break

            self.total_steps += 1

        scorecard = self.adapter.get_scorecard()
        self.audit_trail.log_game_end(
            scorecard.score if scorecard and hasattr(scorecard, 'score') else 0.0
        )
        logger.info(f"Game beendet. Score: {scorecard}")
        return scorecard

    def _step(self) -> str:
        """Ein einzelner Agent-Step."""
        if self.adapter.level_step_count >= self.max_steps_per_level:
            return "MAX_STEPS"

        # 1. Aktion wählen (Explorer entscheidet basierend auf Phase)
        action, data = self.explorer.choose_action(
            self.current_obs, self.memory, self.goals
        )

        # 2. Optional: LLM-Planner für strategische Entscheidungen
        if (self.use_llm_planner
                and self.total_steps % self.llm_call_interval == 0
                and self.total_steps > 0
                and self.explorer.phase != ExplorationPhase.DISCOVERY):
            action, data = self._consult_llm_planner(action, data)

        # 3. Aktion ausführen
        action_str = self._action_to_str(action, data)
        previous_obs = self.current_obs
        self.current_obs = self.adapter.act(action, data)

        # 4. Transition aufzeichnen
        self.memory.record_transition(previous_obs, action_str, self.current_obs)

        # 4b. Audit Trail
        self.audit_trail.log_step(
            level=self.current_level,
            step=self.total_steps,
            action=action_str,
            game_state=str(self.current_obs.game_state),
            pixels_changed=self.current_obs.changed_pixels,
        )

        # 5. Goals aktualisieren (nicht bei jedem Step — zu teuer)
        if self.total_steps % 5 == 0:
            self.goals.analyze_win_condition(self.memory)

        # 6. GameState prüfen
        if self.current_obs.game_state == GameState.WIN:
            return "WIN"
        elif self.current_obs.game_state == GameState.GAME_OVER:
            return "GAME_OVER"

        return "CONTINUE"

    def _consult_llm_planner(self, default_action, default_data):
        """LLM-Planner für strategische Korrekturen nutzen.

        WICHTIG: Nicht bei jedem Step aufrufen!
        LLM-Calls sind 100-1000× langsamer als algorithmische Entscheidungen.
        Nur nutzen für:
        - Hypothesenbildung (Was könnte das Ziel sein?)
        - Strategiewechsel (Explorer steckt fest)
        - Cross-Level-Transfer (Was haben wir in früheren Leveln gelernt?)
        """
        # Prompt zusammenbauen (Context-Budget beachten!)
        state_description = self.encoder.encode_for_llm(
            self.current_obs.raw_grid,
            self.current_obs.grid_diff,
        )
        memory_summary = self.memory.get_summary_for_llm()
        goal_summary = self.goals.get_summary_for_llm()
        mechanics_summary = self.mechanics.get_summary_for_llm()

        prompt = f"""Du bist der Cognithor Planner-Agent in einem ARC-AGI-3 Game.

AKTUELLER ZUSTAND:
{state_description}

BISHERIGES WISSEN:
{memory_summary}

GELERNTE MECHANIKEN:
{mechanics_summary}

ZIEL-HYPOTHESEN:
{goal_summary}

Explorer-Phase: {self.explorer.phase.value}
Level: {self.current_level}, Step: {self.adapter.level_step_count}

AUFGABE: Basierend auf dem bisherigen Wissen:
1. Welche Hypothese über das Spielziel ist am wahrscheinlichsten?
2. Welche Aktion sollte als nächstes getestet werden und warum?
3. Sollte die Explorer-Strategie geändert werden?

Antworte KURZ und STRUKTURIERT."""

        # TODO: Hier Cognithor Planner aufrufen
        # response = self.planner.plan(prompt)
        # Aktion aus Response parsen

        # Fallback: Default-Aktion beibehalten
        return default_action, default_data

    def _on_level_complete(self):
        """Wird nach jedem abgeschlossenen Level aufgerufen."""
        # Mechaniken aus diesem Level abstrahieren
        self.mechanics.analyze_transitions(self.memory, self.current_level)
        self.mechanics.snapshot_level(self.current_level, self.memory)

        self.goals.on_level_complete({
            "level": self.current_level,
            "steps": self.adapter.level_step_count,
            "resets": self.level_resets,
        })
        self.current_level += 1
        self.level_resets = 0
        self.memory.clear_for_new_level()
        # Explorer-Phase zurücksetzen für neues Level
        self.explorer.phase = ExplorationPhase.DISCOVERY
        self.explorer.initialize_discovery(self.adapter.env.action_space)

    @staticmethod
    def _action_to_str(action: GameAction, data: dict) -> str:
        name = action.name if hasattr(action, 'name') else str(action)
        if data and "x" in data and "y" in data:
            return f"{name}_{data['x']}_{data['y']}"
        return name
```

---

## 4. Integration in Cognithor-Projektstruktur (Kurzfassung)

> **Vollständige Struktur mit allen neuen Dateien: Siehe Abschnitt 20.**

```
cognithor/
├── start_cognithor.bat              ← ANPASSEN (neuer ARC-Menüpunkt, siehe 18.3)
├── install.sh                       ← ANPASSEN (ARC-Dependencies, siehe 18.2)
├── modules/
│   ├── arc_agi3/                    ← NEU (14 Dateien + Tests)
│   │   ├── __init__.py
│   │   ├── __main__.py              # CLI Entry Point (siehe 18.4)
│   │   ├── validate_sdk.py          # Phase 0 Validierung (siehe 18.5)
│   │   ├── adapter.py               # ARC SDK ↔ Cognithor Brücke
│   │   ├── episode_memory.py         # Kurzzeit-Lernen pro Game
│   │   ├── mechanics_model.py        # Cross-Level Regel-Abstraktion
│   │   ├── goal_inference.py         # Autonomes Ziel-Erkennen
│   │   ├── explorer.py              # Hypothesengetriebene Exploration
│   │   ├── visual_encoder.py         # Grid → Text Konvertierung
│   │   ├── agent.py                 # Haupt-Orchestrierung
│   │   ├── cnn_model.py             # Optional: CNN für Action-Prediction
│   │   ├── swarm.py                 # Parallele Game-Runs
│   │   ├── audit.py                 # Hashline Guard Integration
│   │   ├── error_handler.py         # Resilience & Error Recovery
│   │   └── tests/                   # 7 Test-Dateien
│   ├── ...
├── mcp_tools/
│   ├── arc_agi3_tool.py             # MCP Tool #124
├── config/
│   ├── arc_agi3_agent.yaml          # Zentrale YAML Config
```

---

## 5. ARC-AGI-3-Agents Repo Integration

Um den Cognithor Agent im offiziellen ARC-AGI-3-Agents Framework zu registrieren:

**Datei:** `ARC-AGI-3-Agents/agents/cognithor_agent.py`

```python
"""Cognithor Agent Wrapper für ARC-AGI-3-Agents Framework."""

from .agent import Agent
from .structs import FrameData, GameAction, GameState

# Cognithor-Module importieren
from cognithor.modules.arc_agi3.episode_memory import EpisodeMemory
from cognithor.modules.arc_agi3.goal_inference import GoalInferenceModule
from cognithor.modules.arc_agi3.explorer import HypothesisDrivenExplorer
from cognithor.modules.arc_agi3.visual_encoder import VisualStateEncoder
import numpy as np


class CognithorAgent(Agent):
    """Cognithor-basierter ARC-AGI-3 Agent."""

    def __init__(self):
        self.memory = EpisodeMemory()
        self.goals = GoalInferenceModule()
        self.explorer = HypothesisDrivenExplorer()
        self.encoder = VisualStateEncoder()
        self._initialized = False
        self._step_count = 0

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        # Erste Initialisierung
        if not self._initialized:
            if hasattr(latest_frame, 'available_actions'):
                self.explorer.initialize_discovery(latest_frame.available_actions)
            self._initialized = True

        # Reset bei Game Over
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self.memory.clear_for_new_level()
            self.explorer.phase = "discovery"
            action = GameAction.RESET
            action.reasoning = "Game Over oder Neustart → Reset"
            return action

        # Grid extrahieren (aus FrameData)
        grid = self._extract_grid(latest_frame)
        obs = type('Obs', (), {
            'raw_grid': grid,
            'game_state': latest_frame.state,
            'changed_pixels': 0,
            'grid_diff': None,
            'level': getattr(latest_frame, 'levels_completed', 0),
        })()

        # Aktion wählen
        action, data = self.explorer.choose_action(obs, self.memory, self.goals)

        # Reasoning setzen
        if action.is_simple():
            action.reasoning = (
                f"Phase: {self.explorer.phase.value}, "
                f"Step: {self._step_count}, "
                f"Ziel: {self.goals.get_best_goal().description[:50]}"
            )
        elif action.is_complex():
            action.set_data(data)
            action.reasoning = {
                "action": action.value,
                "phase": self.explorer.phase.value,
                "reason": f"Ziel: {self.goals.get_best_goal().description[:50]}",
            }

        self._step_count += 1

        # Goal-Analyse alle 5 Steps
        if self._step_count % 5 == 0:
            self.goals.analyze_win_condition(self.memory)

        return action

    def _extract_grid(self, frame: FrameData) -> np.ndarray:
        """Grid aus FrameData extrahieren."""
        if hasattr(frame, 'frame') and frame.frame is not None:
            return np.array(frame.frame, dtype=np.uint8).reshape(64, 64, -1)
        return np.zeros((64, 64, 3), dtype=np.uint8)
```

Registrierung in `agents/__init__.py`:

```python
from .cognithor_agent import CognithorAgent
# Zum AVAILABLE_AGENTS dict hinzufügen
```

Ausführung:

```bash
uv run main.py --agent=cognithoragent --game=ls20
```

---

## 6. Optionale Erweiterung: CNN Action Predictor

Der Top-Score der Preview-Phase (12.58% von "Stochastic Goose") nutzte ein CNN-basiertes Modell, das vorhersagt, welche Aktionen Frame-Änderungen verursachen. Für den Wettbewerb wäre das die wichtigste Erweiterung.

**Datei:** `cognithor/modules/arc_agi3/cnn_model.py`

```python
"""
CNN-basierter Action Predictor.
Trainiert WÄHREND des Spielens (online learning).

Architektur: Leichtgewichtiges CNN
- Input: 64×64×C Grid (C = Farbkanäle)
- Output: 7 Action-Scores (wird eine Aktion eine Änderung bewirken?)
- Zusätzlicher Head: 64×64 Position-Scores für ACTION6

Benötigt: torch (pip install torch --break-system-packages)
Nutzt: RTX 5090 für schnelles Training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque


class ActionPredictor(nn.Module):
    """Vorhersage: Wird eine Aktion den Frame ändern?"""

    def __init__(self, n_colors: int = 10, n_actions: int = 7):
        super().__init__()
        # Encoder: 64×64 → kompakte Repräsentation
        self.conv1 = nn.Conv2d(n_colors, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, stride=2)  # 32×32
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, stride=2) # 16×16
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1, stride=2) # 8×8

        # Action Head: Welche Aktion verursacht Änderung?
        self.action_head = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
            nn.Sigmoid(),
        )

        # Coordinate Head: Wo klicken für ACTION6?
        self.coord_head = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 16×16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 32×32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),    # 64×64
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, n_colors, 64, 64) one-hot encoded
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))

        # Action prediction
        h_flat = h.view(h.size(0), -1)
        action_probs = self.action_head(h_flat)

        # Coordinate prediction
        coord_probs = self.coord_head(h).squeeze(1)  # (batch, 64, 64)

        return action_probs, coord_probs


class OnlineTrainer:
    """Online-Training während des Spielens."""

    def __init__(self, device: str = "cuda", buffer_size: int = 200_000):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = ActionPredictor().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.buffer = deque(maxlen=buffer_size)
        self._seen_hashes = set()
        self.train_interval = 32  # Alle N Samples trainieren
        self.batch_size = 64

    def add_experience(self, grid: np.ndarray, action_idx: int,
                       coord: tuple = None, frame_changed: bool = False):
        """Erfahrung zum Buffer hinzufügen (mit Deduplizierung)."""
        grid_hash = hash(grid.tobytes())
        exp_hash = (grid_hash, action_idx, coord)

        if exp_hash not in self._seen_hashes:
            self._seen_hashes.add(exp_hash)
            self.buffer.append({
                "grid": grid,
                "action": action_idx,
                "coord": coord,
                "changed": frame_changed,
            })

        if len(self.buffer) % self.train_interval == 0 and len(self.buffer) >= self.batch_size:
            self._train_step()

    def predict(self, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vorhersage für aktuellen Grid-Zustand."""
        self.model.eval()
        with torch.no_grad():
            x = self._grid_to_tensor(grid).unsqueeze(0).to(self.device)
            action_probs, coord_probs = self.model(x)
        return action_probs.cpu().numpy()[0], coord_probs.cpu().numpy()[0]

    def _train_step(self):
        """Ein Mini-Batch Trainingsstep."""
        import random
        batch = random.sample(list(self.buffer), min(self.batch_size, len(self.buffer)))

        grids = torch.stack([self._grid_to_tensor(b["grid"]) for b in batch]).to(self.device)
        targets = torch.zeros(len(batch), 7).to(self.device)
        for i, b in enumerate(batch):
            targets[i, b["action"]] = 1.0 if b["changed"] else 0.0

        self.model.train()
        action_probs, _ = self.model(grids)
        loss = F.binary_cross_entropy(action_probs, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _grid_to_tensor(self, grid: np.ndarray) -> torch.Tensor:
        """Grid → One-Hot Tensor (10 Farben × 64 × 64)."""
        if grid.ndim == 3:
            idx = np.argmax(grid[:, :, :3], axis=2)
        else:
            idx = grid.reshape(64, 64)
        one_hot = np.eye(10, dtype=np.float32)[idx]  # 64×64×10
        return torch.from_numpy(one_hot.transpose(2, 0, 1))  # 10×64×64

    def reset_for_new_level(self):
        """Buffer leeren und Modell zurücksetzen für neues Level."""
        self.buffer.clear()
        self._seen_hashes.clear()
        self.model = ActionPredictor().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
```

---

## 7. Implementierungsplan (Kurzfassung)

> **Detaillierter Plan: Siehe Abschnitt 21.**

| Phase | Aufgabe | Abhängigkeiten |
|---|---|---|
| **Phase 0** | SDK-Validierung via `validate_sdk.py` (ZUERST!) | `pip install arc-agi` |
| **Phase 1–3** | Kernmodule (Adapter, Memory, Explorer, Encoder, Goals) | Phase 0 abgeschlossen |
| **Phase 4** | Framework-Integration + YAML Config | Phase 1–3 |
| **Phase 5** | CLI `__main__.py` + `install.sh` + `start_cognithor.bat` Erweiterung | Phase 4 |
| **Phase 6–8** | Audit, MCP Tool, LLM-Planner, CNN Predictor | Phase 5, Optional: RTX 5090 |
| **Phase 9–10** | Swarm, Benchmarking, Community-Release | Alles |

---

## 8. Wettbewerbs-Constraints

Für die offizielle Kaggle-Einreichung gelten:

- **Kein Internet** während der Evaluation → kein API-Aufruf an LLM-Provider möglich
- **Open-Source** unter MIT oder CC0 Pflicht
- **Keine Benchmark-spezifischen Harnesses** erlaubt (muss generell funktionieren)
- **Deadline:** Submissions bis 2. November 2026, Milestones am 30. Juni und 30. September

Das bedeutet: Der LLM-Planner kann im Wettbewerb NICHT extern aufgerufen werden. Zwei Optionen:

1. **Lokales LLM** auf der Kaggle-Maschine (begrenzt durch GPU-Verfügbarkeit)
2. **Rein algorithmischer Agent** (Explorer + CNN + Memory, ohne LLM)

Empfehlung: Phase 1–4 zuerst rein algorithmisch aufbauen, LLM-Integration als optionale Schicht für lokale Experimente.

---

## 9. Erwartete Ergebnisse & Metriken

| Metrik | Random Baseline | Ziel Phase 4 | Ziel Phase 7 | Ziel Phase 9 |
|---|---|---|---|---|
| RHAE Score | ~0.12% | >1% | >5% | >10% |
| Level 1 Completion | ~50% | >90% | >95% | >98% |
| Durchschnittliche Levels | ~1–2 | 3–4 | 5+ | 7+ |
| Steps pro Win (Effizienz) | ~1000+ | <200 | <50 | <25 |
| Test Coverage | — | >80% | >85% | >90% |

Zum Vergleich: Der Preview-Gewinner "Stochastic Goose" (Tufa Labs) erreichte 12.58% mit einem CNN-basierten Ansatz. Frontier LLMs erreichten nur 0.37%. Cognithor mit dem hier beschriebenen Hybrid-Ansatz (algorithmisch + optional LLM + optional CNN) sollte in der 5–15% Range landen.

---

## 10. Community-Impact für Cognithor

Ein funktionierender ARC-AGI-3 Agent wäre:

- **Das erste Open-Source Agent OS mit ARC-AGI-3-Integration** → starkes Differenzierungsmerkmal
- **Messbarer Beweis** für Cognithor's Agentic-Intelligence-Fähigkeiten
- **Reddit/Community-Gold:** "Wir haben Cognithor gegen ARC-AGI-3 antreten lassen — hier sind die Ergebnisse"
- **Wettbewerbsteilnahme** mit realer Chance auf Milestone-Prizes ($75K Open-Source-Pool)
- **Architektur-Showcase** für die PGE Trinity + neue Module

---

## 11. Mechanics Model — Cross-Level-Transfer

Das Episode Memory (Abschnitt 3.2) speichert rohe Transitions. Das reicht nicht für Level-übergreifendes Lernen, weil neue Level neue Mechaniken einführen. Es braucht ein zusätzliches Modul, das generalisierte Regeln ableitet.

**Datei:** `cognithor/modules/arc_agi3/mechanics_model.py`

```python
"""
Mechanics Model: Generalisiert beobachtete Zustandsübergänge
zu wiederverwendbaren Regeln.

Beispiel:
  Level 1: ACTION1 bewegt blaues Objekt nach rechts
  Level 3: ACTION1 bewegt blaues Objekt nach rechts + neues grünes Objekt reagiert
  → Regel: "ACTION1 = horizontale Rechtsbewegung für primäre Objekte"
  → Neue Beobachtung in Level 3 wird als ZUSÄTZLICHE Mechanik erkannt, nicht als Widerspruch

Das ist der entscheidende Unterschied zu reinem Episode Memory:
Mechanics Model abstrahiert, Episode Memory speichert roh.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import numpy as np


class MechanicType(Enum):
    MOVEMENT = "movement"           # Objekt bewegt sich
    TRANSFORMATION = "transformation"  # Objekt ändert Farbe/Form
    CREATION = "creation"           # Neues Objekt entsteht
    DESTRUCTION = "destruction"     # Objekt verschwindet
    TOGGLE = "toggle"              # Zustand wechselt hin und her
    CONDITIONAL = "conditional"     # Effekt hängt von Bedingung ab
    NO_EFFECT = "no_effect"        # Aktion hat keinen sichtbaren Effekt
    UNKNOWN = "unknown"


@dataclass
class Mechanic:
    """Eine generalisierte Spielmechanik."""
    mechanic_type: MechanicType
    action: str                         # Auslösende Aktion
    description: str                    # Menschenlesbare Beschreibung
    observed_in_levels: list[int] = field(default_factory=list)
    observation_count: int = 0
    consistency_score: float = 0.0      # Wie konsistent über Beobachtungen hinweg
    spatial_pattern: Optional[str] = None  # z.B. "rechts", "diagonal", "zentral"
    affected_colors: list[int] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)  # Was muss gelten, damit Mechanik wirkt


class MechanicsModel:
    """Speichert und aktualisiert generalisierte Spielmechaniken."""

    def __init__(self):
        self.mechanics: list[Mechanic] = []
        self.action_to_mechanics: dict[str, list[Mechanic]] = {}
        self._level_snapshots: list[dict] = []

    def analyze_transitions(self, episode_memory, current_level: int):
        """Transitions aus dem Episode Memory zu Mechaniken abstrahieren."""
        for action_str, effects in episode_memory.action_effect_map.items():
            if effects["total"] < 3:
                continue  # Zu wenig Daten

            # Bestehende Mechanik suchen oder neue erstellen
            existing = self._find_mechanic(action_str)

            change_rate = effects["caused_change"] / effects["total"]
            if change_rate < 0.1:
                mtype = MechanicType.NO_EFFECT
            elif change_rate > 0.9:
                mtype = MechanicType.MOVEMENT  # Vermutung, wird durch Spatial-Analyse verfeinert
            else:
                mtype = MechanicType.CONDITIONAL

            if existing:
                existing.observation_count += effects["total"]
                if current_level not in existing.observed_in_levels:
                    existing.observed_in_levels.append(current_level)
                existing.consistency_score = self._calc_consistency(existing, change_rate)
            else:
                m = Mechanic(
                    mechanic_type=mtype,
                    action=action_str,
                    description=f"{action_str}: {change_rate:.0%} Änderungsrate",
                    observed_in_levels=[current_level],
                    observation_count=effects["total"],
                    consistency_score=change_rate,
                )
                self.mechanics.append(m)
                self.action_to_mechanics.setdefault(action_str, []).append(m)

    def analyze_spatial_patterns(self, episode_memory):
        """Räumliche Muster aus Pixel-Diffs ableiten.

        Analysiert, ob Änderungen typischerweise in einer Richtung
        auftreten (→ Bewegung), am gleichen Ort (→ Toggle),
        oder verstreut (→ Transformation).
        """
        for t in episode_memory.transitions:
            if t.pixels_changed == 0:
                continue
            # Hier würde die Grid-Diff-Analyse stattfinden
            # Benötigt Zugriff auf die tatsächlichen Grids, nicht nur Hashes
            # TODO: Episode Memory um Grid-Snapshots erweitern (memory-intensiv)

    def get_reliable_mechanics(self, min_consistency: float = 0.7) -> list[Mechanic]:
        """Nur Mechaniken mit hoher Konsistenz zurückgeben."""
        return [m for m in self.mechanics if m.consistency_score >= min_consistency]

    def get_mechanics_for_action(self, action_str: str) -> list[Mechanic]:
        return self.action_to_mechanics.get(action_str, [])

    def predict_action_effect(self, action_str: str) -> MechanicType:
        """Basierend auf bekannten Mechaniken: Was wird passieren?"""
        mechanics = self.get_mechanics_for_action(action_str)
        if not mechanics:
            return MechanicType.UNKNOWN
        # Höchste Konsistenz gewinnt
        best = max(mechanics, key=lambda m: m.consistency_score)
        return best.mechanic_type

    def snapshot_level(self, level: int, episode_memory):
        """Momentaufnahme des Wissensstands nach einem Level."""
        self._level_snapshots.append({
            "level": level,
            "mechanics_count": len(self.mechanics),
            "reliable_count": len(self.get_reliable_mechanics()),
            "action_map": {
                a: e["caused_change"] / max(e["total"], 1)
                for a, e in episode_memory.action_effect_map.items()
            },
        })

    def get_summary_for_llm(self) -> str:
        lines = ["Bekannte Mechaniken:"]
        for m in self.get_reliable_mechanics()[:5]:
            levels = ",".join(str(l) for l in m.observed_in_levels)
            lines.append(
                f"  [{m.consistency_score:.0%}] {m.action} → {m.mechanic_type.value}: "
                f"{m.description} (Level: {levels})"
            )
        return "\n".join(lines)

    def _find_mechanic(self, action_str: str) -> Optional[Mechanic]:
        mechanics = self.action_to_mechanics.get(action_str, [])
        return mechanics[0] if mechanics else None

    def _calc_consistency(self, mechanic: Mechanic, new_rate: float) -> float:
        """Exponential Moving Average der Konsistenz."""
        alpha = 0.3
        return alpha * new_rate + (1 - alpha) * mechanic.consistency_score
```

Dieses Modul wird in `agent.py` nach jedem Level-Abschluss aufgerufen:

```python
# In CognithorArcAgent._on_level_complete():
self.mechanics_model.analyze_transitions(self.memory, self.current_level)
self.mechanics_model.snapshot_level(self.current_level, self.memory)
```

---

## 12. Error Handling & Resilience

**Datei:** `cognithor/modules/arc_agi3/error_handler.py`

```python
"""
Robustes Error Handling für ARC-AGI-3 Runs.
Fehler dürfen NIEMALS einen kompletten Game-Run abbrechen.
"""

import logging
import time
from functools import wraps
from typing import Callable, Any, Optional

logger = logging.getLogger("cognithor.arc_agi3.errors")


class ArcAgentError(Exception):
    """Basis-Exception für ARC-AGI-3 Agent-Fehler."""
    pass


class FrameExtractionError(ArcAgentError):
    """Grid konnte nicht aus FrameData extrahiert werden."""
    pass


class EnvironmentConnectionError(ArcAgentError):
    """Verbindung zum ARC API fehlgeschlagen."""
    pass


def retry_on_error(
    max_retries: int = 3,
    delay_seconds: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """Decorator: Retry mit exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait = delay_seconds * (backoff_factor ** attempt)
                        logger.warning(
                            f"{func.__name__} fehlgeschlagen (Versuch {attempt+1}/{max_retries+1}): "
                            f"{e}. Retry in {wait:.1f}s..."
                        )
                        time.sleep(wait)
                    else:
                        logger.error(
                            f"{func.__name__} endgültig fehlgeschlagen nach {max_retries+1} Versuchen: {e}"
                        )
            raise last_exception
        return wrapper
    return decorator


def safe_frame_extract(obs, fallback_shape=(64, 64, 3)):
    """Sicher ein Grid aus einer Observation extrahieren.
    Gibt IMMER ein numpy-Array zurück, nie None.
    """
    import numpy as np

    if obs is None:
        logger.warning("Observation ist None — Fallback auf leeres Grid")
        return np.zeros(fallback_shape, dtype=np.uint8)

    # Verschiedene mögliche Attribut-Namen durchprobieren
    for attr in ['frame', 'frame_data', 'grid', 'pixels', 'data', 'image']:
        val = getattr(obs, attr, None)
        if val is not None:
            try:
                arr = np.array(val, dtype=np.uint8)
                if arr.size > 0:
                    # Shape validieren/anpassen
                    if arr.ndim == 1:
                        side = int(np.sqrt(arr.size))
                        if side * side == arr.size:
                            return arr.reshape(side, side)
                        elif arr.size == 64 * 64 * 3:
                            return arr.reshape(64, 64, 3)
                        elif arr.size == 64 * 64:
                            return arr.reshape(64, 64)
                    return arr
            except (ValueError, TypeError) as e:
                logger.debug(f"Attribut '{attr}' konnte nicht konvertiert werden: {e}")
                continue

    logger.warning(f"Kein Grid-Attribut gefunden in {type(obs).__name__}. "
                  f"Vorhandene Attribute: {[a for a in dir(obs) if not a.startswith('_')]}")
    return np.zeros(fallback_shape, dtype=np.uint8)


class GameRunGuard:
    """Context Manager für sichere Game-Runs.

    Fängt alle Fehler, loggt sie, und stellt sicher
    dass die Scorecard immer abgeholt wird.
    """

    def __init__(self, arcade, game_id: str):
        self.arcade = arcade
        self.game_id = game_id
        self.errors: list[dict] = []
        self.env = None

    def __enter__(self):
        try:
            self.env = self.arcade.make(self.game_id)
            if self.env is None:
                raise EnvironmentConnectionError(f"Game '{self.game_id}' konnte nicht erstellt werden")
        except Exception as e:
            self.errors.append({"phase": "init", "error": str(e), "type": type(e).__name__})
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.errors.append({
                "phase": "runtime",
                "error": str(exc_val),
                "type": exc_type.__name__,
            })
            logger.error(f"Game Run für {self.game_id} mit Fehler beendet: {exc_val}")

        # Scorecard IMMER abholen
        try:
            scorecard = self.arcade.get_scorecard()
            if scorecard:
                logger.info(f"Scorecard für {self.game_id}: {scorecard}")
        except Exception as e:
            logger.warning(f"Scorecard konnte nicht abgeholt werden: {e}")

        # Errors zusammenfassen
        if self.errors:
            logger.warning(f"Game Run hatte {len(self.errors)} Fehler: "
                         f"{[e['type'] for e in self.errors]}")

        return True  # Exceptions unterdrücken, nicht propagieren
```

**Anwendung im Agent:**

```python
# In CognithorArcAgent.run():
with GameRunGuard(self.arcade, self.game_id) as guard:
    self.env = guard.env
    # ... Game Loop ...
```

**Integration in Adapter:**

```python
# adapter.py — act() mit Retry dekorieren
@retry_on_error(max_retries=2, exceptions=(ConnectionError, TimeoutError))
def act(self, action, data=None):
    raw = self.env.step(action, data=data or {})
    # ...
```

---

## 13. Hashline Guard Integration

Cognithor nutzt xxHash64 + SHA-256 für Audit-Trails. ARC-AGI-3 Runs müssen dort integriert werden.

**Datei:** `cognithor/modules/arc_agi3/audit.py`

```python
"""
ARC-AGI-3 Audit-Integration für Cognithor Hashline Guard.
Jeder Game-Run wird als auditierbare Event-Kette protokolliert.
"""

import json
import hashlib
import time
from dataclasses import dataclass, asdict
from typing import Optional

# Cognithor Hashline Guard Import
# from cognithor.core.security.hashline_guard import HashlineGuard


@dataclass
class ArcAuditEvent:
    """Ein einzelnes, auditierbares Event in einem ARC-AGI-3 Run."""
    timestamp: float
    event_type: str          # "game_start", "step", "level_complete", "game_end", "error"
    game_id: str
    level: int
    step: int
    action: Optional[str] = None
    game_state: Optional[str] = None
    pixels_changed: Optional[int] = None
    score: Optional[float] = None
    error: Optional[str] = None
    metadata: Optional[dict] = None


class ArcAuditTrail:
    """Audit-Trail für einen kompletten ARC-AGI-3 Game-Run."""

    def __init__(self, game_id: str, agent_version: str = "cognithor-arc-v1"):
        self.game_id = game_id
        self.agent_version = agent_version
        self.events: list[ArcAuditEvent] = []
        self.run_id = hashlib.sha256(
            f"{game_id}:{time.time()}:{agent_version}".encode()
        ).hexdigest()[:16]
        self._previous_hash: Optional[str] = None

    def log_event(self, event: ArcAuditEvent) -> str:
        """Event loggen und Hash-Chain fortsetzen."""
        self.events.append(event)

        # Hash-Chain: Jedes Event enthält den Hash des vorherigen
        event_json = json.dumps(asdict(event), sort_keys=True)
        chain_input = f"{self._previous_hash or 'GENESIS'}:{event_json}"
        event_hash = hashlib.sha256(chain_input.encode()).hexdigest()
        self._previous_hash = event_hash

        return event_hash

    def log_step(self, level: int, step: int, action: str,
                 game_state: str, pixels_changed: int):
        """Convenience-Methode für Step-Events."""
        return self.log_event(ArcAuditEvent(
            timestamp=time.time(),
            event_type="step",
            game_id=self.game_id,
            level=level,
            step=step,
            action=action,
            game_state=game_state,
            pixels_changed=pixels_changed,
        ))

    def log_game_start(self):
        return self.log_event(ArcAuditEvent(
            timestamp=time.time(),
            event_type="game_start",
            game_id=self.game_id,
            level=0,
            step=0,
            metadata={"agent_version": self.agent_version, "run_id": self.run_id},
        ))

    def log_game_end(self, final_score: float):
        return self.log_event(ArcAuditEvent(
            timestamp=time.time(),
            event_type="game_end",
            game_id=self.game_id,
            level=-1,
            step=len(self.events),
            score=final_score,
            metadata={"total_events": len(self.events), "run_id": self.run_id},
        ))

    def export_jsonl(self, filepath: str):
        """Audit-Trail als JSONL exportieren (kompatibel mit ARC Recording-Format)."""
        with open(filepath, 'w') as f:
            for event in self.events:
                f.write(json.dumps(asdict(event)) + "\n")

    def verify_integrity(self) -> bool:
        """Hash-Chain verifizieren."""
        prev_hash = None
        for event in self.events:
            event_json = json.dumps(asdict(event), sort_keys=True)
            chain_input = f"{prev_hash or 'GENESIS'}:{event_json}"
            expected_hash = hashlib.sha256(chain_input.encode()).hexdigest()
            prev_hash = expected_hash
        return prev_hash == self._previous_hash
```

---

## 14. YAML-Konfiguration

**Datei:** `cognithor/config/arc_agi3_agent.yaml`

```yaml
# Cognithor ARC-AGI-3 Agent Konfiguration
# =========================================

agent:
  name: "cognithor-arc-agi3"
  version: "1.0.0"
  description: "Cognithor Agent für ARC-AGI-3 Interactive Reasoning Benchmark"

# ARC-AGI-3 SDK Einstellungen
arc_sdk:
  api_key_env: "ARC_API_KEY"         # Environment Variable für API Key
  operation_mode: "NORMAL"            # NORMAL | ONLINE | OFFLINE | COMPETITION
  save_recordings: true
  recording_dir: "./recordings/arc_agi3"

# Explorer Konfiguration
explorer:
  discovery_phase_max_steps: 50       # Max Steps in Discovery Phase
  hypothesis_confidence_threshold: 0.6  # Ab wann → Exploitation
  exploitation_explore_ratio: 0.2      # 20% Exploration auch in Exploitation
  complex_action_sample_positions:     # Strategische Positionen für complex actions
    - [0, 0]
    - [0, 32]
    - [0, 63]
    - [32, 0]
    - [32, 32]
    - [32, 63]
    - [63, 0]
    - [63, 32]
    - [63, 63]
    - [16, 16]
    - [16, 48]
    - [48, 16]
    - [48, 48]

# Episode Memory Konfiguration
memory:
  max_transitions: 200000
  hash_algorithm: "md5"               # md5 für Geschwindigkeit, sha256 für Audit
  clear_on_new_level: "visits_only"   # "visits_only" | "full" | "none"

# Goal Inference Konfiguration
goal_inference:
  analysis_interval: 5                # Alle N Steps Goals neu bewerten
  max_goals_tracked: 5
  min_transitions_for_analysis: 10

# LLM Planner Konfiguration
llm_planner:
  enabled: true                       # false für rein algorithmischen Mode
  call_interval: 10                   # LLM nur alle N Steps konsultieren
  provider: "anthropic"               # anthropic | openai | ollama | local
  model: "claude-sonnet-4-20250514"   # Für lokale Experimente
  local_model: "qwen2.5-coder:14b"   # Für Kaggle/Offline-Mode
  max_tokens: 500                     # Kurze Antworten für Effizienz
  context_budget: 4000                # Max Tokens für Prompt (Grid + Memory + Goals)

# CNN Action Predictor Konfiguration
cnn_predictor:
  enabled: false                      # true für Wettbewerbs-Mode
  device: "cuda"                      # cuda | cpu
  buffer_size: 200000
  train_interval: 32
  batch_size: 64
  learning_rate: 0.001
  reset_on_new_level: true

# Game Loop Limits
limits:
  max_steps_per_level: 500
  max_resets_per_level: 5
  max_total_steps: 5000
  max_levels: 15

# Audit & Logging
audit:
  enabled: true
  hashline_guard_integration: true
  export_jsonl: true
  export_dir: "./audit/arc_agi3"
  log_level: "INFO"                   # DEBUG für Entwicklung

# Swarm Mode (Abschnitt 16)
swarm:
  enabled: false
  max_parallel_agents: 4
  games: []                           # Leer = alle verfügbaren Games
  result_aggregation: "best"          # "best" | "average" | "weighted"

# Wettbewerbs-spezifische Einstellungen
competition:
  mode: false                         # true = kein Internet, kein externes LLM
  kaggle_gpu: "T4"                    # Erwartete GPU auf Kaggle
  submission_format: "kaggle"
```

**Loader:**

```python
# In agent.py — Config laden
import yaml
from pathlib import Path

def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "arc_agi3_agent.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)
```

---

## 15. MCP Tool Integration

Cognithor hat 123 MCP Tools. Der ARC-AGI-3 Agent braucht mindestens ein eigenes MCP Tool, damit er über alle Cognithor-Channels (Telegram, Discord, Flutter Command Center) steuerbar ist.

**Datei:** `cognithor/mcp_tools/arc_agi3_tool.py`

```python
"""
MCP Tool: arc_agi3_play
========================
Ermöglicht das Starten und Monitoren von ARC-AGI-3 Runs
über alle Cognithor-Channels.

Registrierung in MCP-Registry als Tool #124.
"""

# MCP Tool Schema
ARC_AGI3_PLAY_TOOL = {
    "name": "arc_agi3_play",
    "description": "Startet einen Cognithor ARC-AGI-3 Agent-Run auf einem bestimmten Game. "
                   "Gibt Live-Status und finale Scorecard zurück.",
    "input_schema": {
        "type": "object",
        "properties": {
            "game_id": {
                "type": "string",
                "description": "ARC-AGI-3 Game ID (z.B. 'ls20', 'ft09', 'vc33'). "
                              "Leer = alle verfügbaren Games."
            },
            "mode": {
                "type": "string",
                "enum": ["single", "benchmark", "swarm"],
                "description": "single = ein Game, benchmark = alle Games mit Scorecard, "
                              "swarm = parallel"
            },
            "use_llm": {
                "type": "boolean",
                "description": "LLM-Planner aktivieren (langsamer, aber strategischer)",
                "default": True,
            },
            "use_cnn": {
                "type": "boolean",
                "description": "CNN Action Predictor aktivieren (benötigt GPU)",
                "default": False,
            },
        },
        "required": ["game_id"],
    },
}

ARC_AGI3_STATUS_TOOL = {
    "name": "arc_agi3_status",
    "description": "Gibt den aktuellen Status eines laufenden ARC-AGI-3 Runs zurück.",
    "input_schema": {
        "type": "object",
        "properties": {
            "run_id": {
                "type": "string",
                "description": "Run-ID aus arc_agi3_play"
            },
        },
        "required": ["run_id"],
    },
}

ARC_AGI3_REPLAY_TOOL = {
    "name": "arc_agi3_replay",
    "description": "Gibt den Replay-Link und die Zusammenfassung eines abgeschlossenen Runs zurück.",
    "input_schema": {
        "type": "object",
        "properties": {
            "run_id": {
                "type": "string",
                "description": "Run-ID"
            },
        },
        "required": ["run_id"],
    },
}


async def handle_arc_agi3_play(params: dict) -> dict:
    """MCP Tool Handler für arc_agi3_play."""
    from cognithor.modules.arc_agi3.agent import CognithorArcAgent

    game_id = params.get("game_id", "ls20")
    use_llm = params.get("use_llm", True)
    use_cnn = params.get("use_cnn", False)
    mode = params.get("mode", "single")

    if mode == "single":
        agent = CognithorArcAgent(
            game_id=game_id,
            use_llm_planner=use_llm,
        )
        scorecard = agent.run()
        return {
            "status": "completed",
            "game_id": game_id,
            "scorecard": str(scorecard),
            "run_id": agent.audit_trail.run_id if hasattr(agent, 'audit_trail') else "N/A",
            "total_steps": agent.total_steps,
            "levels_completed": agent.current_level,
        }

    elif mode == "swarm":
        # Swarm Mode — siehe Abschnitt 16
        pass

    return {"status": "error", "message": f"Unbekannter Mode: {mode}"}
```

---

## 16. Swarm Mode

Die ARC-AGI-3 Docs unterstützen Swarm Execution — mehrere Agent-Instanzen parallel auf verschiedene Games. Das passt direkt zu Cognithor's Multi-Agent-System.

**Datei:** `cognithor/modules/arc_agi3/swarm.py`

```python
"""
ARC-AGI-3 Swarm Orchestrator.
Nutzt Cognithor's Multi-Agent-System für parallele Game-Runs.
"""

import asyncio
import logging
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger("cognithor.arc_agi3.swarm")


@dataclass
class SwarmResult:
    """Ergebnis eines Swarm-Runs."""
    game_id: str
    score: float
    levels_completed: int
    total_steps: int
    errors: list[str] = field(default_factory=list)
    run_id: str = ""


class ArcSwarmOrchestrator:
    """Orchestriert parallele ARC-AGI-3 Agent-Runs."""

    def __init__(
        self,
        max_parallel: int = 4,
        config: dict = None,
    ):
        self.max_parallel = max_parallel
        self.config = config or {}
        self.results: list[SwarmResult] = []

    async def run_swarm(self, game_ids: list[str]) -> list[SwarmResult]:
        """Alle Games parallel durchspielen."""
        semaphore = asyncio.Semaphore(self.max_parallel)
        tasks = [self._run_with_semaphore(semaphore, gid) for gid in game_ids]
        self.results = await asyncio.gather(*tasks, return_exceptions=False)
        return self.results

    async def _run_with_semaphore(self, sem, game_id: str) -> SwarmResult:
        async with sem:
            return await self._run_single(game_id)

    async def _run_single(self, game_id: str) -> SwarmResult:
        """Einzelnen Agent-Run in async ausführen."""
        from cognithor.modules.arc_agi3.agent import CognithorArcAgent

        try:
            logger.info(f"Swarm: Starte Agent für {game_id}")
            agent = CognithorArcAgent(
                game_id=game_id,
                use_llm_planner=self.config.get("use_llm", False),
            )
            # run() ist synchron → in Thread-Pool auslagern
            loop = asyncio.get_event_loop()
            scorecard = await loop.run_in_executor(None, agent.run)

            return SwarmResult(
                game_id=game_id,
                score=scorecard.score if hasattr(scorecard, 'score') else 0.0,
                levels_completed=agent.current_level,
                total_steps=agent.total_steps,
                run_id=getattr(agent, 'audit_trail', None) and agent.audit_trail.run_id or "",
            )
        except Exception as e:
            logger.error(f"Swarm: Agent für {game_id} fehlgeschlagen: {e}")
            return SwarmResult(
                game_id=game_id,
                score=0.0,
                levels_completed=0,
                total_steps=0,
                errors=[str(e)],
            )

    def get_aggregate_score(self) -> float:
        """Gesamt-Score über alle Games."""
        scores = [r.score for r in self.results if r.score > 0]
        return sum(scores) / len(scores) if scores else 0.0

    def get_summary(self) -> str:
        lines = [f"Swarm Run: {len(self.results)} Games"]
        lines.append(f"Gesamt-Score: {self.get_aggregate_score():.4f}")
        lines.append("")
        for r in sorted(self.results, key=lambda x: x.score, reverse=True):
            status = "✓" if not r.errors else "✗"
            lines.append(f"  {status} {r.game_id}: Score={r.score:.4f}, "
                        f"Level={r.levels_completed}, Steps={r.total_steps}")
        failed = [r for r in self.results if r.errors]
        if failed:
            lines.append(f"\nFehlgeschlagen: {len(failed)}")
            for r in failed:
                lines.append(f"  {r.game_id}: {r.errors[0][:100]}")
        return "\n".join(lines)
```

---

## 17. Recording & Replay Integration

Das ARC SDK speichert JSONL-Recordings. Diese müssen in Cognithor integriert werden für Debugging, Community-Demos und Wettbewerbs-Einreichung.

```python
# In adapter.py — Recording aktivieren:

class ArcEnvironmentAdapter:
    def __init__(self, game_id: str, save_recording: bool = True, recording_dir: str = "./recordings"):
        # ...
        self.save_recording = save_recording
        self.recording_dir = recording_dir

    def initialize(self) -> ArcObservation:
        self.env = self.arcade.make(
            self.game_id,
            save_recording=self.save_recording,
        )
        # ...
```

Das ARC SDK erstellt automatisch JSONL-Files wenn `save_recording=True`. Der Replay-Link wird am Ende des Runs ausgegeben. Diesen Link in den Audit-Trail und in MCP-Tool-Responses aufnehmen.

---

## 18. Dependency-Management & Startup-Integration

### 18.1 Dependencies

**Datei:** `cognithor/modules/arc_agi3/requirements.txt`

```
# ARC-AGI-3 Core
arc-agi>=0.3.0
arcengine>=0.3.0

# Datenverarbeitung
numpy>=1.26.0

# Konfiguration
pyyaml>=6.0

# Optional: CNN Action Predictor
torch>=2.2.0      # Nur wenn cnn_predictor.enabled = true
torchvision>=0.17.0

# Optional: Swarm Mode
asyncio            # stdlib, kein Install nötig

# Development / Testing
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

**Ergänzung in `cognithor/pyproject.toml`:**

```toml
[project.optional-dependencies]
arc-agi3 = [
    "arc-agi>=0.3.0",
    "arcengine>=0.3.0",
    "numpy>=1.26.0",
    "pyyaml>=6.0",
]
arc-agi3-gpu = [
    "arc-agi>=0.3.0",
    "arcengine>=0.3.0",
    "numpy>=1.26.0",
    "pyyaml>=6.0",
    "torch>=2.2.0",
    "torchvision>=0.17.0",
]
```

### 18.2 install.sh Anpassungen (Linux/Mac)

Cognithor wird über `install.sh` installiert. Dieses Script muss um die ARC-AGI-3 Dependencies erweitert werden.

**AN DAS CODING-MODELL:** Finde `install.sh` im Cognithor-Root-Verzeichnis. Füge folgenden Block NACH der bestehenden pip-Installation der Kern-Dependencies ein. Suche nach dem Abschnitt, wo andere optionale Module installiert werden, und füge dort den ARC-AGI-3 Block ein. Wenn es keinen solchen Abschnitt gibt, füge ihn vor dem Abschluss-Block ein.

**Einzufügender Block in `install.sh`:**

```bash
# ============================================================
# ARC-AGI-3 Integration (Optional)
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ARC-AGI-3 Benchmark Integration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
read -p "ARC-AGI-3 Benchmark-Support installieren? (j/n): " INSTALL_ARC
if [ "$INSTALL_ARC" = "j" ] || [ "$INSTALL_ARC" = "J" ] || [ "$INSTALL_ARC" = "y" ] || [ "$INSTALL_ARC" = "Y" ]; then
    echo "Installiere ARC-AGI-3 Core-Dependencies..."
    pip install arc-agi arcengine --break-system-packages 2>/dev/null || \
    pip install arc-agi arcengine

    # ARC API Key abfragen
    if [ -z "$ARC_API_KEY" ]; then
        echo ""
        echo "Für ARC-AGI-3 wird ein API Key benötigt."
        echo "Registrierung unter: https://three.arcprize.org/user"
        read -p "ARC API Key eingeben (oder Enter zum Überspringen): " ARC_KEY_INPUT
        if [ -n "$ARC_KEY_INPUT" ]; then
            # In .env Datei schreiben
            if [ -f ".env" ]; then
                # Bestehenden Key ersetzen oder neuen hinzufügen
                if grep -q "ARC_API_KEY" .env; then
                    sed -i "s/ARC_API_KEY=.*/ARC_API_KEY=$ARC_KEY_INPUT/" .env
                else
                    echo "ARC_API_KEY=$ARC_KEY_INPUT" >> .env
                fi
            else
                echo "ARC_API_KEY=$ARC_KEY_INPUT" > .env
            fi
            echo "✓ ARC API Key in .env gespeichert"
        else
            echo "⚠ Kein API Key angegeben — anonymer Zugriff wird verwendet (eingeschränkter Game-Zugang)"
        fi
    else
        echo "✓ ARC_API_KEY bereits in Environment gesetzt"
    fi

    # GPU-Support prüfen
    if command -v nvidia-smi &> /dev/null; then
        echo ""
        read -p "GPU erkannt. CNN Action Predictor (PyTorch) installieren? (j/n): " INSTALL_GPU
        if [ "$INSTALL_GPU" = "j" ] || [ "$INSTALL_GPU" = "J" ] || [ "$INSTALL_GPU" = "y" ] || [ "$INSTALL_GPU" = "Y" ]; then
            echo "Installiere PyTorch mit CUDA-Support..."
            pip install torch torchvision --break-system-packages 2>/dev/null || \
            pip install torch torchvision
            echo "✓ PyTorch mit GPU-Support installiert"
        fi
    fi

    # ARC-AGI-3 Agents Repo klonen (optional, für Framework-Integration)
    echo ""
    read -p "ARC-AGI-3 Agents Repo klonen (für Wettbewerbs-Einreichung)? (j/n): " CLONE_AGENTS
    if [ "$CLONE_AGENTS" = "j" ] || [ "$CLONE_AGENTS" = "J" ] || [ "$CLONE_AGENTS" = "y" ] || [ "$CLONE_AGENTS" = "Y" ]; then
        if [ ! -d "ARC-AGI-3-Agents" ]; then
            git clone https://github.com/arcprize/ARC-AGI-3-Agents.git
            echo "✓ ARC-AGI-3 Agents Repo geklont"
        else
            echo "✓ ARC-AGI-3-Agents Verzeichnis existiert bereits"
        fi
    fi

    echo ""
    echo "✓ ARC-AGI-3 Integration abgeschlossen"
    echo "  Starten mit: python -m cognithor.modules.arc_agi3 --game ls20"
    echo "  Oder über Cognithor: start_cognithor.bat / install.sh → dann ARC-Modus wählen"
else
    echo "ARC-AGI-3 übersprungen."
fi
```

**Platzierung:** Dieses Block muss im Installationsablauf von `install.sh` eingefügt werden. Such nach dem Muster, wo andere optionale Module/Features installiert werden (z.B. Voice, Browser Automation, etc.) und füge den ARC-Block dort ein. Falls kein solches Muster existiert, füge ihn NACH der Hauptinstallation aller Kern-Dependencies ein, ABER VOR dem finalen "Installation abgeschlossen"-Output.

### 18.3 start_cognithor.bat Anpassungen (Windows)

Cognithor wird auf Windows über `start_cognithor.bat` gestartet. Dieses Script muss um einen ARC-AGI-3 Startmodus erweitert werden.

**AN DAS CODING-MODELL:** Finde `start_cognithor.bat` im Cognithor-Root-Verzeichnis. Erweitere es um einen ARC-AGI-3 Modus. Hier ist die Logik:

**Einzufügender Block in `start_cognithor.bat`:**

```batch
@REM ============================================================
@REM ARC-AGI-3 Benchmark Modus
@REM ============================================================
@REM Dieser Block gehört in das Hauptmenü von start_cognithor.bat.
@REM Füge eine neue Menüoption hinzu (z.B. Option 7 oder die nächste
@REM freie Nummer) und das zugehörige Label.

@REM --- Menüoption (einfügen wo die anderen Optionen stehen) ---
@REM echo   [X] ARC-AGI-3 Benchmark Agent starten

@REM --- Label (einfügen wo die anderen Labels definiert sind) ---
:arc_agi3
echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo   Cognithor ARC-AGI-3 Agent
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.

@REM Prüfe ob ARC-AGI-3 installiert ist
python -c "import arc_agi" 2>nul
if errorlevel 1 (
    echo [!] ARC-AGI-3 SDK ist nicht installiert.
    echo     Bitte install.sh / pip install arc-agi ausfuehren.
    echo.
    pause
    goto menu
)

@REM Prüfe ARC API Key
if "%ARC_API_KEY%"=="" (
    if exist .env (
        for /f "tokens=1,2 delims==" %%a in (.env) do (
            if "%%a"=="ARC_API_KEY" set ARC_API_KEY=%%b
        )
    )
)

if "%ARC_API_KEY%"=="" (
    echo [!] Kein ARC_API_KEY gefunden.
    echo     Registrierung unter: https://three.arcprize.org/user
    set /p ARC_API_KEY="ARC API Key eingeben (oder Enter fuer anonym): "
)

echo.
echo Verfuegbare Modi:
echo   [1] Einzelnes Game spielen
echo   [2] Alle Games benchmarken
echo   [3] Swarm Mode (parallel)
echo   [4] SDK validieren (Phase 0)
echo.
set /p ARC_MODE="Modus waehlen (1-4): "

if "%ARC_MODE%"=="1" (
    set /p ARC_GAME="Game ID eingeben (z.B. ls20, ft09, vc33): "
    echo Starte Agent fuer %ARC_GAME%...
    python -m cognithor.modules.arc_agi3 --game %ARC_GAME% --config config/arc_agi3_agent.yaml
)
if "%ARC_MODE%"=="2" (
    echo Starte Benchmark ueber alle Games...
    python -m cognithor.modules.arc_agi3 --mode benchmark --config config/arc_agi3_agent.yaml
)
if "%ARC_MODE%"=="3" (
    set /p ARC_PARALLEL="Anzahl paralleler Agents (Standard: 4): "
    if "%ARC_PARALLEL%"=="" set ARC_PARALLEL=4
    echo Starte Swarm mit %ARC_PARALLEL% Agents...
    python -m cognithor.modules.arc_agi3 --mode swarm --parallel %ARC_PARALLEL% --config config/arc_agi3_agent.yaml
)
if "%ARC_MODE%"=="4" (
    echo Fuehre SDK-Validierung aus...
    python -m cognithor.modules.arc_agi3.validate_sdk
)

echo.
pause
goto menu
```

**Platzierung:** Im Hauptmenü von `start_cognithor.bat` gibt es ein Auswahlmenü (typischerweise `echo [1] ...`, `echo [2] ...` etc.). Füge eine neue Nummer für "ARC-AGI-3 Benchmark Agent starten" hinzu. Das zugehörige `:arc_agi3` Label kommt dorthin, wo die anderen Label-Blöcke stehen.

**WICHTIG FÜR DAS CODING-MODELL:** 
- Öffne zuerst `start_cognithor.bat` und identifiziere das existierende Menü-Pattern
- Füge die neue Option konsistent mit dem bestehenden Stil ein
- Beachte die portable Python-Pfade die Cognithor möglicherweise nutzt (nicht globales `python` sondern evtl. `%COGNITHOR_PYTHON%` oder ein lokaler venv-Pfad)
- Prüfe ob `start_cognithor.bat` eine venv aktiviert — falls ja, müssen die ARC-AGI-3 Dependencies IN dieser venv installiert sein

### 18.4 CLI Entry Point (__main__.py)

Damit `python -m cognithor.modules.arc_agi3` funktioniert (wird von beiden Startup-Scripts aufgerufen):

**Datei:** `cognithor/modules/arc_agi3/__main__.py`

```python
"""
CLI Entry Point für den Cognithor ARC-AGI-3 Agent.
Aufruf: python -m cognithor.modules.arc_agi3 [optionen]

Wird von start_cognithor.bat und install.sh aufgerufen.
"""

import argparse
import asyncio
import logging
import sys
import yaml
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Cognithor ARC-AGI-3 Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Einzelnes Game spielen
  python -m cognithor.modules.arc_agi3 --game ls20

  # Alle Games benchmarken
  python -m cognithor.modules.arc_agi3 --mode benchmark

  # Swarm Mode
  python -m cognithor.modules.arc_agi3 --mode swarm --parallel 4

  # Mit eigener Config
  python -m cognithor.modules.arc_agi3 --game ls20 --config config/arc_agi3_agent.yaml

  # Ohne LLM (rein algorithmisch, für Wettbewerb)
  python -m cognithor.modules.arc_agi3 --game ls20 --no-llm

  # Mit CNN Predictor (benötigt GPU)
  python -m cognithor.modules.arc_agi3 --game ls20 --cnn
        """,
    )

    parser.add_argument("--game", type=str, default=None,
                       help="ARC-AGI-3 Game ID (z.B. ls20, ft09, vc33). Leer = alle Games.")
    parser.add_argument("--mode", type=str, choices=["single", "benchmark", "swarm"],
                       default="single", help="Ausführungsmodus")
    parser.add_argument("--config", type=str, default=None,
                       help="Pfad zur YAML-Konfiguration")
    parser.add_argument("--parallel", type=int, default=4,
                       help="Anzahl paralleler Agents im Swarm Mode")
    parser.add_argument("--no-llm", action="store_true",
                       help="LLM-Planner deaktivieren (rein algorithmisch)")
    parser.add_argument("--cnn", action="store_true",
                       help="CNN Action Predictor aktivieren (benötigt GPU)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Debug-Logging aktivieren")
    parser.add_argument("--list-games", action="store_true",
                       help="Verfügbare Games auflisten und beenden")

    args = parser.parse_args()

    # Logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("cognithor.arc_agi3")

    # Config laden
    config = {}
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
            logger.info(f"Config geladen: {config_path}")
        else:
            logger.warning(f"Config nicht gefunden: {config_path} — verwende Defaults")

    # CLI-Argumente überschreiben Config
    if args.no_llm:
        config.setdefault("llm_planner", {})["enabled"] = False
    if args.cnn:
        config.setdefault("cnn_predictor", {})["enabled"] = True

    # Games auflisten
    if args.list_games:
        try:
            import arc_agi
            arcade = arc_agi.Arcade()
            games = arcade.list_environments() if hasattr(arcade, 'list_environments') else []
            if games:
                print(f"\nVerfügbare ARC-AGI-3 Games ({len(games)}):")
                for g in games:
                    print(f"  {g}")
            else:
                print("Keine Games gefunden. Prüfe ARC_API_KEY.")
        except ImportError:
            print("FEHLER: arc-agi nicht installiert. Führe install.sh aus.")
            sys.exit(1)
        sys.exit(0)

    # Abhängigkeiten prüfen
    try:
        import arc_agi
        from arcengine import GameAction, GameState
    except ImportError as e:
        logger.error(f"ARC-AGI-3 SDK nicht installiert: {e}")
        logger.error("Bitte install.sh ausführen und ARC-AGI-3 Support aktivieren,")
        logger.error("oder manuell: pip install arc-agi arcengine")
        sys.exit(1)

    # Ausführen
    if args.mode == "single":
        game_id = args.game or "ls20"
        logger.info(f"Starte Cognithor ARC Agent — Game: {game_id}")

        from .agent import CognithorArcAgent
        agent = CognithorArcAgent(
            game_id=game_id,
            use_llm_planner=config.get("llm_planner", {}).get("enabled", not args.no_llm),
        )
        scorecard = agent.run()

        print(f"\n{'='*50}")
        print(f"  Game: {game_id}")
        print(f"  Score: {scorecard}")
        print(f"  Levels: {agent.current_level}")
        print(f"  Steps: {agent.total_steps}")
        print(f"{'='*50}")

    elif args.mode == "benchmark":
        logger.info("Starte Benchmark über alle Games...")
        from .swarm import ArcSwarmOrchestrator
        import arc_agi

        arcade = arc_agi.Arcade()
        games = arcade.list_environments() if hasattr(arcade, 'list_environments') else ["ls20"]

        orchestrator = ArcSwarmOrchestrator(
            max_parallel=1,  # Benchmark = sequentiell für faire Messung
            config=config,
        )
        results = asyncio.run(orchestrator.run_swarm(games))
        print(f"\n{orchestrator.get_summary()}")

    elif args.mode == "swarm":
        logger.info(f"Starte Swarm mit {args.parallel} parallelen Agents...")
        from .swarm import ArcSwarmOrchestrator
        import arc_agi

        arcade = arc_agi.Arcade()
        games = arcade.list_environments() if hasattr(arcade, 'list_environments') else ["ls20"]

        orchestrator = ArcSwarmOrchestrator(
            max_parallel=args.parallel,
            config=config,
        )
        results = asyncio.run(orchestrator.run_swarm(games))
        print(f"\n{orchestrator.get_summary()}")


if __name__ == "__main__":
    main()
```

### 18.5 SDK-Validierungsscript als eigenes Modul

Damit `python -m cognithor.modules.arc_agi3.validate_sdk` aus `start_cognithor.bat` heraus funktioniert:

**Datei:** `cognithor/modules/arc_agi3/validate_sdk.py`

```python
"""
SDK-Validierung — wird aus start_cognithor.bat (Option 4) aufgerufen
und sollte VOR jeder Implementierung ausgeführt werden.
Entspricht Phase 0 aus der Spezifikation.
"""

import sys
import json
from pathlib import Path


def validate():
    print("=" * 60)
    print("  Cognithor ARC-AGI-3 SDK Validierung (Phase 0)")
    print("=" * 60)

    results = {}
    errors = []

    # 1. Import-Test
    print("\n[1/10] arc-agi Import...")
    try:
        import arc_agi
        from arcengine import GameAction, GameState
        print("  ✓ arc_agi und arcengine importiert")
        results["import"] = "OK"
    except ImportError as e:
        print(f"  ✗ Import fehlgeschlagen: {e}")
        print("    → pip install arc-agi arcengine")
        results["import"] = f"FEHLER: {e}"
        errors.append("Import fehlgeschlagen")
        print("\nAbbruch — SDK nicht installiert.")
        sys.exit(1)

    # 2. Arcade-Klasse
    print("\n[2/10] Arcade-Klasse inspizieren...")
    arc = arc_agi.Arcade()
    methods = [m for m in dir(arc) if not m.startswith('_')]
    print(f"  Methoden: {methods}")
    results["arcade_methods"] = methods

    # 3. Environment erstellen
    print("\n[3/10] Environment 'ls20' erstellen...")
    try:
        env = arc.make("ls20")
        if env is None:
            print("  ✗ arc.make('ls20') gibt None zurück")
            errors.append("Environment-Erstellung gibt None zurück")
            results["env_create"] = "None"
        else:
            env_methods = [m for m in dir(env) if not m.startswith('_')]
            print(f"  ✓ Typ: {type(env).__name__}")
            print(f"  Methoden: {env_methods}")
            results["env_type"] = type(env).__name__
            results["env_methods"] = env_methods
    except Exception as e:
        print(f"  ✗ Fehler: {e}")
        errors.append(f"env erstellen: {e}")
        results["env_create"] = f"FEHLER: {e}"
        env = None

    if env is None:
        print("\nAbbruch — kein Environment verfügbar.")
        _save_results(results, errors)
        sys.exit(1)

    # 4. Reset / erster Frame
    print("\n[4/10] env.reset() — FrameDataRaw inspizieren...")
    try:
        obs = env.reset()
        print(f"  Typ: {type(obs).__name__}")
        attrs = {a: type(getattr(obs, a)).__name__
                 for a in dir(obs) if not a.startswith('_') and not callable(getattr(obs, a))}
        print(f"  Attribute: {attrs}")
        results["obs_type"] = type(obs).__name__
        results["obs_attributes"] = attrs
    except Exception as e:
        print(f"  ✗ Fehler: {e}")
        errors.append(f"reset: {e}")
        obs = None

    # 5. Grid-Format
    print("\n[5/10] Grid-Format identifizieren...")
    if obs:
        for attr_name in ['frame', 'frame_data', 'grid', 'pixels', 'data', 'image']:
            val = getattr(obs, attr_name, None)
            if val is not None:
                try:
                    import numpy as np
                    arr = np.array(val)
                    print(f"  ✓ Gefunden in obs.{attr_name}")
                    print(f"    Typ: {type(val).__name__}")
                    print(f"    Shape: {arr.shape}")
                    print(f"    Dtype: {arr.dtype}")
                    print(f"    Min/Max: {arr.min()} / {arr.max()}")
                    print(f"    Erster Wert: {arr.flat[0]}")
                    results["grid_attribute"] = attr_name
                    results["grid_shape"] = str(arr.shape)
                    results["grid_dtype"] = str(arr.dtype)
                    results["grid_range"] = f"{arr.min()}-{arr.max()}"
                    break
                except Exception as e:
                    print(f"  obs.{attr_name} existiert aber Konvertierung scheitert: {e}")
        else:
            print("  ✗ Kein Grid-Attribut gefunden!")
            print(f"    Alle Nicht-Callable-Attribute: {attrs}")
            errors.append("Kein Grid-Attribut gefunden")
            results["grid_attribute"] = "NICHT_GEFUNDEN"

    # 6. GameState-Werte
    print("\n[6/10] GameState Enum-Werte...")
    states = [s for s in dir(GameState) if not s.startswith('_') and s.isupper()]
    print(f"  Werte: {states}")
    if obs:
        print(f"  Aktueller State: {obs.state}")
    results["game_states"] = states

    # 7. GameAction-Werte
    print("\n[7/10] GameAction Enum-Werte...")
    all_actions = list(GameAction)
    for a in all_actions:
        simple = a.is_simple() if hasattr(a, 'is_simple') else '?'
        complex_ = a.is_complex() if hasattr(a, 'is_complex') else '?'
        print(f"  {a}: simple={simple}, complex={complex_}")
    results["actions"] = [str(a) for a in all_actions]

    # 8. action_space
    print("\n[8/10] env.action_space...")
    if hasattr(env, 'action_space'):
        print(f"  ✓ {env.action_space}")
        results["action_space"] = str(env.action_space)
    else:
        print("  ✗ env.action_space existiert nicht")
        errors.append("action_space nicht vorhanden")
        results["action_space"] = "NICHT_VORHANDEN"

    # 9. Step-Return
    print("\n[9/10] env.step() Return-Format...")
    try:
        obs2 = env.step(GameAction.ACTION1)
        print(f"  Typ: {type(obs2).__name__}")
        print(f"  State: {obs2.state if obs2 else 'None'}")
        if obs2 and hasattr(obs2, 'levels_completed'):
            print(f"  levels_completed: {obs2.levels_completed}")
            results["has_levels_completed"] = True
        else:
            print(f"  ⚠ levels_completed nicht vorhanden")
            results["has_levels_completed"] = False
        results["step_return_type"] = type(obs2).__name__ if obs2 else "None"
    except Exception as e:
        print(f"  ✗ Fehler: {e}")
        errors.append(f"step: {e}")

    # 10. Scorecard
    print("\n[10/10] Scorecard-Format...")
    try:
        sc = arc.get_scorecard()
        if sc:
            print(f"  Typ: {type(sc).__name__}")
            sc_attrs = {a: type(getattr(sc, a)).__name__
                       for a in dir(sc) if not a.startswith('_') and not callable(getattr(sc, a))}
            print(f"  Attribute: {sc_attrs}")
            if hasattr(sc, 'score'):
                print(f"  Score: {sc.score}")
            results["scorecard_type"] = type(sc).__name__
            results["scorecard_attrs"] = sc_attrs
        else:
            print("  Scorecard ist None")
            results["scorecard_type"] = "None"
    except Exception as e:
        print(f"  ✗ Fehler: {e}")

    # Zusammenfassung
    print("\n" + "=" * 60)
    if errors:
        print(f"  ⚠ {len(errors)} PROBLEME GEFUNDEN:")
        for e in errors:
            print(f"    • {e}")
        print("\n  → Passe die Module gemäß Abschnitt 0.2 der Spec an!")
    else:
        print("  ✓ Alle Prüfungen bestanden!")
        print("  → Du kannst mit Phase 1 der Implementierung starten.")
    print("=" * 60)

    _save_results(results, errors)


def _save_results(results, errors):
    """Ergebnisse als JSON speichern für programmatische Auswertung."""
    output = {"results": results, "errors": errors}
    output_path = Path("arc_agi3_validation.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nErgebnisse gespeichert in: {output_path}")
    print("Dieses JSON enthält alle SDK-Details die für die Modul-Anpassung benötigt werden.")


if __name__ == "__main__":
    validate()
```

### 18.6 Hinweise zur venv / portablen Python-Pfaden

**AN DAS CODING-MODELL:** Cognithor nutzt möglicherweise portable Python-Installationen oder ein Virtual Environment. Prüfe folgende Punkte bevor du `install.sh` oder `start_cognithor.bat` änderst:

1. **Prüfe `start_cognithor.bat`:** Wird `python` direkt aufgerufen oder über eine Variable wie `%COGNITHOR_PYTHON%`, `%PYTHON_PATH%` oder einen relativen Pfad wie `.\python\python.exe`? → Verwende denselben Pfad für alle neuen `python`-Aufrufe.

2. **Prüfe `install.sh`:** Wird eine venv aktiviert (`source venv/bin/activate` o.ä.)? → Die ARC-AGI-3 Dependencies müssen INNERHALB dieser venv installiert werden, NACH der Aktivierung.

3. **Prüfe `.env` Handling:** Wie lädt Cognithor die `.env` Datei? Über `python-dotenv`, über ein Shell-Script, oder manuell? → Der `ARC_API_KEY` muss auf dem gleichen Weg verfügbar sein.

4. **Windows-spezifisch:** `start_cognithor.bat` nutzt möglicherweise `chcp 65001` für UTF-8. Die ARC-Menütexte oben nutzen Unicode-Zeichen (━, ✓, ✗) — falls Codepage-Probleme auftreten, ersetze durch ASCII-Alternativen (`=`, `[OK]`, `[FAIL]`).

---

## 19. Tests

Cognithor hat 11.609+ Tests bei 89% Coverage. Die ARC-AGI-3 Module brauchen eigene Tests.

**Datei:** `cognithor/modules/arc_agi3/tests/test_episode_memory.py`

```python
import pytest
import numpy as np
from cognithor.modules.arc_agi3.episode_memory import EpisodeMemory, StateTransition


class TestEpisodeMemory:

    def setup_method(self):
        self.memory = EpisodeMemory(max_transitions=1000)

    def test_hash_grid_deterministic(self):
        """Gleiche Grids müssen gleichen Hash produzieren."""
        grid = np.random.randint(0, 10, (64, 64, 3), dtype=np.uint8)
        h1 = self.memory.hash_grid(grid)
        h2 = self.memory.hash_grid(grid.copy())
        assert h1 == h2

    def test_hash_grid_different_for_different_grids(self):
        """Unterschiedliche Grids müssen unterschiedliche Hashes haben."""
        grid1 = np.zeros((64, 64, 3), dtype=np.uint8)
        grid2 = np.ones((64, 64, 3), dtype=np.uint8)
        assert self.memory.hash_grid(grid1) != self.memory.hash_grid(grid2)

    def test_record_transition(self):
        """Transition korrekt aufzeichnen."""
        obs_before = type('O', (), {'raw_grid': np.zeros((64, 64, 3), dtype=np.uint8)})()
        obs_after = type('O', (), {
            'raw_grid': np.ones((64, 64, 3), dtype=np.uint8),
            'changed_pixels': 4096,
            'game_state': 'PLAYING',
            'level': 0,
        })()
        t = self.memory.record_transition(obs_before, "ACTION1", obs_after)
        assert t.pixels_changed == 4096
        assert t.action == "ACTION1"
        assert len(self.memory.transitions) == 1

    def test_action_effectiveness(self):
        """Effektivität korrekt berechnen."""
        obs = type('O', (), {'raw_grid': np.zeros((64, 64, 3), dtype=np.uint8)})()
        obs_changed = type('O', (), {
            'raw_grid': np.ones((64, 64, 3), dtype=np.uint8),
            'changed_pixels': 100,
            'game_state': 'PLAYING',
            'level': 0,
        })()
        obs_unchanged = type('O', (), {
            'raw_grid': np.zeros((64, 64, 3), dtype=np.uint8),
            'changed_pixels': 0,
            'game_state': 'PLAYING',
            'level': 0,
        })()
        # 3 von 4 Aktionen verursachen Änderung
        for _ in range(3):
            self.memory.record_transition(obs, "ACTION1", obs_changed)
        self.memory.record_transition(obs, "ACTION1", obs_unchanged)

        eff = self.memory.get_action_effectiveness("ACTION1")
        assert eff == 0.75

    def test_unknown_action_effectiveness(self):
        """Unbekannte Aktion gibt 0.5 zurück (neutral)."""
        assert self.memory.get_action_effectiveness("NEVER_USED") == 0.5

    def test_novel_state_detection(self):
        """Neue vs. bekannte Zustände unterscheiden."""
        grid1 = np.zeros((64, 64, 3), dtype=np.uint8)
        grid2 = np.ones((64, 64, 3), dtype=np.uint8)
        assert self.memory.is_novel_state(grid1) == True
        # Hash registrieren
        self.memory.visited_states.add(self.memory.hash_grid(grid1))
        assert self.memory.is_novel_state(grid1) == False
        assert self.memory.is_novel_state(grid2) == True

    def test_max_transitions_limit(self):
        """Buffer-Overflow verhindern."""
        mem = EpisodeMemory(max_transitions=5)
        obs = type('O', (), {
            'raw_grid': np.zeros((64, 64, 3), dtype=np.uint8),
            'changed_pixels': 0,
            'game_state': 'PLAYING',
            'level': 0,
        })()
        for i in range(10):
            mem.record_transition(obs, f"ACTION{i}", obs)
        assert len(mem.transitions) == 5  # Nicht über Limit

    def test_clear_for_new_level(self):
        """Level-Clear behält Transitions, löscht Visit-Counts."""
        obs = type('O', (), {
            'raw_grid': np.zeros((64, 64, 3), dtype=np.uint8),
            'changed_pixels': 0,
            'game_state': 'PLAYING',
            'level': 0,
        })()
        self.memory.record_transition(obs, "ACTION1", obs)
        assert len(self.memory.state_visit_count) > 0
        transitions_before = len(self.memory.transitions)

        self.memory.clear_for_new_level()
        assert len(self.memory.state_visit_count) == 0
        assert len(self.memory.transitions) == transitions_before  # Behalten

    def test_summary_for_llm(self):
        """LLM-Summary muss String sein und Kerninfos enthalten."""
        summary = self.memory.get_summary_for_llm()
        assert isinstance(summary, str)
        assert "Besuchte Zustände" in summary


class TestGoalInference:

    def test_no_wins_returns_unknown(self):
        from cognithor.modules.arc_agi3.goal_inference import GoalInferenceModule, GoalType
        gim = GoalInferenceModule()
        memory = EpisodeMemory()
        goals = gim.analyze_win_condition(memory)
        assert any(g.goal_type == GoalType.UNKNOWN for g in goals)

    def test_win_transitions_generate_goal(self):
        from cognithor.modules.arc_agi3.goal_inference import GoalInferenceModule, GoalType
        gim = GoalInferenceModule()
        memory = EpisodeMemory()
        # Simuliere Win-Transition
        t = StateTransition(
            state_hash="abc", action="ACTION3", next_state_hash="def",
            pixels_changed=50, resulted_in_win=True, level=0,
        )
        memory.transitions.append(t)
        goals = gim.analyze_win_condition(memory)
        assert any(g.goal_type == GoalType.REACH_STATE for g in goals)
        assert "ACTION3" in goals[0].description


class TestExplorer:

    def test_initial_phase_is_discovery(self):
        from cognithor.modules.arc_agi3.explorer import HypothesisDrivenExplorer, ExplorationPhase
        explorer = HypothesisDrivenExplorer()
        assert explorer.phase == ExplorationPhase.DISCOVERY

    def test_phase_transition_after_max_steps(self):
        from cognithor.modules.arc_agi3.explorer import HypothesisDrivenExplorer, ExplorationPhase
        from cognithor.modules.arc_agi3.goal_inference import GoalInferenceModule
        explorer = HypothesisDrivenExplorer()
        explorer._phase_step_count = 51
        explorer._check_phase_transition(EpisodeMemory(), GoalInferenceModule())
        assert explorer.phase == ExplorationPhase.HYPOTHESIS


class TestVisualEncoder:

    def test_encode_returns_string(self):
        from cognithor.modules.arc_agi3.visual_encoder import VisualStateEncoder
        encoder = VisualStateEncoder()
        grid = np.random.randint(0, 10, (64, 64), dtype=np.uint8)
        result = encoder.encode_for_llm(grid)
        assert isinstance(result, str)
        assert "Farbverteilung" in result

    def test_encode_compact(self):
        from cognithor.modules.arc_agi3.visual_encoder import VisualStateEncoder
        encoder = VisualStateEncoder()
        grid = np.zeros((64, 64), dtype=np.uint8)
        result = encoder.encode_compact(grid)
        assert result.startswith("[")
        assert result.endswith("]")


class TestMechanicsModel:

    def test_no_data_returns_empty(self):
        from cognithor.modules.arc_agi3.mechanics_model import MechanicsModel
        mm = MechanicsModel()
        assert mm.get_reliable_mechanics() == []

    def test_predict_unknown_action(self):
        from cognithor.modules.arc_agi3.mechanics_model import MechanicsModel, MechanicType
        mm = MechanicsModel()
        assert mm.predict_action_effect("NEVER_SEEN") == MechanicType.UNKNOWN


class TestAuditTrail:

    def test_hash_chain_integrity(self):
        from cognithor.modules.arc_agi3.audit import ArcAuditTrail
        trail = ArcAuditTrail("ls20")
        trail.log_game_start()
        trail.log_step(0, 1, "ACTION1", "PLAYING", 50)
        trail.log_step(0, 2, "ACTION3", "WIN", 100)
        trail.log_game_end(0.5)
        assert trail.verify_integrity() == True

    def test_export_jsonl(self, tmp_path):
        from cognithor.modules.arc_agi3.audit import ArcAuditTrail
        trail = ArcAuditTrail("ls20")
        trail.log_game_start()
        filepath = str(tmp_path / "test_audit.jsonl")
        trail.export_jsonl(filepath)
        with open(filepath) as f:
            lines = f.readlines()
        assert len(lines) == 1


class TestErrorHandling:

    def test_safe_frame_extract_none(self):
        from cognithor.modules.arc_agi3.error_handler import safe_frame_extract
        result = safe_frame_extract(None)
        assert result.shape == (64, 64, 3)
        assert np.all(result == 0)

    def test_safe_frame_extract_valid(self):
        from cognithor.modules.arc_agi3.error_handler import safe_frame_extract
        obs = type('O', (), {'frame': np.ones((64, 64, 3), dtype=np.uint8)})()
        result = safe_frame_extract(obs)
        assert np.all(result == 1)

    def test_safe_frame_extract_unknown_format(self):
        from cognithor.modules.arc_agi3.error_handler import safe_frame_extract
        obs = type('O', (), {'unknown_field': 42})()
        result = safe_frame_extract(obs)
        assert result.shape == (64, 64, 3)  # Fallback
```

**Test-Ausführung:**

```bash
# Alle ARC-AGI-3 Tests
pytest cognithor/modules/arc_agi3/tests/ -v

# Nur Unit-Tests (ohne SDK-Abhängigkeit)
pytest cognithor/modules/arc_agi3/tests/ -v -m "not integration"

# Mit Coverage
pytest cognithor/modules/arc_agi3/tests/ --cov=cognithor/modules/arc_agi3 --cov-report=term-missing
```

---

## 20. Erweiterte Projektstruktur

```
cognithor/
├── start_cognithor.bat                    ← ANPASSEN (neuer ARC-AGI-3 Menüpunkt, siehe 18.3)
├── install.sh                             ← ANPASSEN (ARC-AGI-3 Installationsblock, siehe 18.2)
├── .env                                   ← ARC_API_KEY hier eintragen
├── modules/
│   ├── arc_agi3/                          ← NEU (komplett)
│   │   ├── __init__.py
│   │   ├── __main__.py                    # CLI Entry Point (siehe 18.4)
│   │   ├── validate_sdk.py                # Phase 0 Validierung (siehe 18.5)
│   │   ├── adapter.py                     # ARC SDK ↔ Cognithor Brücke
│   │   ├── episode_memory.py              # Kurzzeit-Lernen pro Game
│   │   ├── mechanics_model.py             # Cross-Level Regel-Abstraktion
│   │   ├── goal_inference.py              # Autonomes Ziel-Erkennen
│   │   ├── explorer.py                    # Hypothesengetriebene Exploration
│   │   ├── visual_encoder.py              # Grid → Text Konvertierung
│   │   ├── agent.py                       # Haupt-Orchestrierung
│   │   ├── cnn_model.py                   # CNN für Action-Prediction
│   │   ├── swarm.py                       # Parallele Game-Runs
│   │   ├── audit.py                       # Hashline Guard Integration
│   │   ├── error_handler.py               # Resilience & Error Recovery
│   │   ├── requirements.txt               # Dependencies
│   │   └── tests/
│   │       ├── test_episode_memory.py
│   │       ├── test_goal_inference.py
│   │       ├── test_explorer.py
│   │       ├── test_visual_encoder.py
│   │       ├── test_mechanics_model.py
│   │       ├── test_audit.py
│   │       ├── test_error_handler.py
│   │       └── conftest.py
│   ├── ...
├── mcp_tools/
│   ├── arc_agi3_tool.py                   # MCP Tool #124
│   ├── ...
├── config/
│   ├── arc_agi3_agent.yaml                # Zentrale YAML Config
│   ├── ...
├── ARC-AGI-3-Agents/                      ← Optional: geklontes Agents-Repo (über install.sh)
│   ├── agents/
│   │   ├── cognithor_agent.py             # Cognithor Wrapper für Wettbewerb
│   │   └── ...
│   └── ...
```

---

## 21. Aktualisierter Implementierungsplan

| Phase | Zeitrahmen | Aufgabe | Output |
|---|---|---|---|
| **Phase 0** | Tag 1 | SDK installieren, `validate_sdk.py` ausführen (Abschnitt 18.5), Annahmen-Tabelle ausfüllen | `arc_agi3_validation.json` + ausgefüllte Tabelle aus Abschnitt 0.2 |
| **Phase 1** | Woche 1 | Adapter + Episode Memory + Error Handler + Tests | Funktionierender Random-Agent mit Cognithor-Wrapping |
| **Phase 2** | Woche 2 | Visual Encoder + Goal Inference + Mechanics Model | Agent mit Zustandsverständnis |
| **Phase 3** | Woche 2–3 | Hypothesis-Driven Explorer + Agent-Orchestrierung | Agent der systematisch exploriert |
| **Phase 4** | Woche 3 | ARC-AGI-3-Agents Framework Integration + YAML Config | `uv run main.py --agent=cognithoragent` funktioniert |
| **Phase 5** | Woche 3–4 | `__main__.py` CLI + `install.sh` Erweiterung + `start_cognithor.bat` Menüpunkt (Abschnitte 18.2–18.4) | `python -m cognithor.modules.arc_agi3 --game ls20` funktioniert, ARC-Modus im Startmenü verfügbar |
| **Phase 6** | Woche 4 | Audit Trail + Recording Integration + MCP Tool | Reproduzierbare Runs, Channel-Integration |
| **Phase 7** | Woche 4–5 | LLM-Planner Anbindung an PGE Trinity | Hybrid-Agent mit strategischem LLM |
| **Phase 8** | Woche 5–6 | CNN Action Predictor (GPU-abhängig) | Wettbewerbsfähiger Agent |
| **Phase 9** | Woche 6–7 | Swarm Mode + Benchmarking über alle Games | Gesamt-Scorecard |
| **Phase 10** | Woche 7–8 | Tuning, Community-Release, Reddit-Post | v0.58.0 Release mit ARC-AGI-3 Support |

**Startup-Script-Reihenfolge:**
1. `install.sh` → installiert ARC-AGI-3 Dependencies (Phase 5)
2. `start_cognithor.bat` → neuer Menüpunkt "ARC-AGI-3 Agent" (Phase 5)
3. User wählt ARC-Modus → ruft `python -m cognithor.modules.arc_agi3` auf (Phase 5)
4. Oder: User startet Cognithor normal → nutzt MCP Tool `arc_agi3_play` über beliebigen Channel (Phase 6)

**Milestone-Alignment:**
- 30. Juni 2026 (ARC Prize Milestone 1): Phasen 0–7 abgeschlossen, Score > 1%
- 30. September 2026 (ARC Prize Milestone 2): Phasen 8–9 abgeschlossen, Score > 5%
- 2. November 2026 (Submission Deadline): Phase 10 abgeschlossen, optimierter Score
