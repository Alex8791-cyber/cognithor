# ARC-AGI-3 Root Cause Fixes — Design Spec

**Date:** 2026-04-04
**Status:** Approved
**Depends on:** ARC-AGI-3 dual-mode agent (RL + Classic DSL)

## Goal

Fix the explorer (stuck on one action), add telemetry, normalize action space, and increase step budget so the RL agent can actually reach WIN states in ARC-AGI-3 games.

## Problem Summary

Live test on `ls20` showed:
- Agent plays 100 steps, all GameState.NOT_FINISHED, score 0.0
- Explorer stuck on ACTION3 for 80+ steps (no variety)
- Game DOES respond to actions (changed_pixels: 52, 76, 128, 4096)
- GAME_OVER is reachable at step 129 (with action cycling)
- Baseline for Level 1 is 125 steps — agent's max_steps (100) is too low

State detection code (`"WIN" in str(game_state)`) is correct — the agent just never reaches WIN because the explorer doesn't explore.

## 1. Epsilon-Greedy Explorer with Pixel-Reward

Replace the current explorer's action selection (which gets stuck on one action) with an epsilon-greedy approach using `changed_pixels` as reward signal.

```python
class PixelRewardExplorer:
    """Epsilon-greedy action selection using changed_pixels as reward."""

    def __init__(self, epsilon: float = 0.2):
        self.epsilon = epsilon
        self.action_rewards: dict[int, list[float]] = {}

    def select_action(self, available_actions: list) -> Any:
        # Initialize tracking for new actions
        for a in available_actions:
            key = a.value if hasattr(a, "value") else int(a)
            if key not in self.action_rewards:
                self.action_rewards[key] = []

        # Epsilon: random exploration (20% of the time)
        if random.random() < self.epsilon:
            return random.choice(available_actions)

        # Greedy: pick action with highest average reward (last 10)
        # Prefer untested actions (no history = try them first)
        best_action = None
        best_avg = -1.0
        for a in available_actions:
            key = a.value if hasattr(a, "value") else int(a)
            rewards = self.action_rewards[key]
            if not rewards:
                return a  # untested = highest priority
            avg = sum(rewards[-10:]) / len(rewards[-10:])
            if avg > best_avg:
                best_avg = avg
                best_action = a

        return best_action or random.choice(available_actions)

    def record_reward(self, action: Any, changed_pixels: int):
        key = action.value if hasattr(action, "value") else int(action)
        self.action_rewards.setdefault(key, []).append(float(changed_pixels))

    def reset_for_new_level(self):
        """Keep learned rewards across resets (transfer learning)."""
        pass  # intentionally keep rewards
```

Integration in `agent.py`:
- Create `PixelRewardExplorer` in `__init__`
- In `_step()`: use `self.pixel_explorer.select_action(obs.available_actions)` instead of the old explorer for action selection
- After each `adapter.act()`: call `self.pixel_explorer.record_reward(action, new_obs.changed_pixels)`
- The old `HypothesisDrivenExplorer` remains for hypothesis tracking but is not used for action selection

## 2. Max-Steps Increase

Baselines for ls20: `[125, 58, 259, 113, 499, 58, 186, 134, 132]` — Level 5 needs 499 steps alone.

Change defaults:
- `max_steps_per_level`: 500 -> 1000
- `__main__.py` default: 500 -> 1000

## 3. In-Memory Telemetry Tracker

New dataclass in `agent.py`:

```python
@dataclass
class ArcTelemetry:
    """In-memory telemetry for one game run."""

    actions_taken: dict[str, int]  # action_name -> count
    pixels_per_action: dict[str, list[int]]  # action_name -> [changed_pixels]
    states_discovered: int = 0
    stagnation_count: int = 0  # current consecutive 0-pixel steps
    max_stagnation: int = 0
    game_overs: int = 0
    levels_won: int = 0

    def record_step(self, action: str, changed_pixels: int):
        self.actions_taken[action] = self.actions_taken.get(action, 0) + 1
        self.pixels_per_action.setdefault(action, []).append(changed_pixels)
        if changed_pixels == 0:
            self.stagnation_count += 1
            self.max_stagnation = max(self.max_stagnation, self.stagnation_count)
        else:
            self.stagnation_count = 0

    def summary(self) -> str:
        lines = ["=== ARC Telemetry ==="]
        lines.append(f"Actions: {self.actions_taken}")
        for a, pxs in self.pixels_per_action.items():
            avg = sum(pxs) / len(pxs) if pxs else 0
            lines.append(f"  {a}: avg_pixels={avg:.0f}, count={len(pxs)}")
        lines.append(f"States discovered: {self.states_discovered}")
        lines.append(f"Max stagnation: {self.max_stagnation} steps")
        lines.append(f"Game overs: {self.game_overs}")
        lines.append(f"Levels won: {self.levels_won}")
        return "\n".join(lines)
```

Logged at end of `run()`: `log.info("arc_telemetry", summary=self.telemetry.summary())`

## 4. Action-Space Normalization

SDK returns `available_actions: [1, 2, 3, 4]` (raw ints). Normalize to `GameAction` enums in `adapter.py _process_frame()`:

```python
from arcengine.enums import GameAction

normalized = []
for a in raw_actions:
    if isinstance(a, int):
        try:
            normalized.append(GameAction(a))
        except (ValueError, KeyError):
            normalized.append(a)
    else:
        normalized.append(a)
```

This ensures the explorer and CNN model can use `.value` and `.name` consistently.

## 5. Files Changed

| File | Change |
|------|--------|
| `src/jarvis/arc/agent.py` | `ArcTelemetry` dataclass, `PixelRewardExplorer` class, telemetry wiring, max_steps default 1000 |
| `src/jarvis/arc/adapter.py` | Action normalization to GameAction enums |
| `src/jarvis/arc/__main__.py` | max_steps default 1000 |
| `tests/test_arc/test_agent.py` | Tests for PixelRewardExplorer, ArcTelemetry |

## 6. What is NOT Changed (Until Telemetry Provides Data)

- CNN Training — needs valid transitions first
- State-Graph Win-Detection — code works, just needs WIN states to occur
- LLM Planner — later, when explorer baseline is established
- Mechanics Model — later, when explorer explores enough

## 7. Expected Outcome

With epsilon-greedy + 1000 steps:
1. All 4 actions tested (not stuck on one)
2. Actions with more pixel changes preferred
3. Enough steps to reach GAME_OVER (step ~129) and potentially WIN
4. After GAME_OVER: reset, retry with learned action rewards
5. Telemetry shows exactly what's happening for next iteration
