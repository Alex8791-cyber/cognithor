"""ARC-AGI-3 VisionAgent — qwen3-vl guided step-by-step gameplay."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from jarvis.arc.error_handler import safe_frame_extract
from jarvis.arc.game_analyzer import _grid_to_png_b64, _parse_vision_json
from jarvis.utils.logging import get_logger

__all__ = ["VisionAgent"]

log = get_logger(__name__)

_ACTION_NAMES = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "INTERACT", 6: "CLICK", 7: "ACTION7"}
_NAME_TO_ACTION = {"UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4,
                   "INTERACT": 5, "CLICK": 6, "ACTION7": 7,
                   "ACTION1": 1, "ACTION2": 2, "ACTION3": 3, "ACTION4": 4,
                   "ACTION5": 5, "ACTION6": 6}


@dataclass
class VisionResult:
    levels_completed: int = 0
    total_steps: int = 0


class VisionAgent:
    """Step-by-step vision-guided agent using qwen3-vl."""

    def __init__(self, arcade: Any, game_id: str, available_actions: list[int]):
        self._arcade = arcade
        self._game_id = game_id
        self._actions = available_actions
        self._action_str = ", ".join(
            _ACTION_NAMES.get(a, f"ACTION{a}") for a in available_actions
        )

    def solve(self, max_levels: int = 5, timeout_s: float = 300.0) -> VisionResult:
        """Solve game level by level with vision guidance."""
        from arcengine.enums import GameState

        env = self._arcade.make(self._game_id)
        result = VisionResult()
        prev_actions: list[int | tuple] = []

        for level in range(max_levels):
            t0 = time.monotonic()
            obs = env.reset()
            for a in prev_actions:
                if isinstance(a, tuple):
                    obs = env.step(6, data={"x": a[0], "y": a[1]})
                else:
                    obs = env.step(a)

            current_levels = obs.levels_completed
            level_actions: list[int | tuple] = []
            grid = safe_frame_extract(obs)
            strategy = self._get_strategy(grid)
            last_actions: list[str] = []

            for step in range(100):  # max 100 steps per level
                if time.monotonic() - t0 > timeout_s:
                    break

                grid = safe_frame_extract(obs)

                # Ask vision what to do (with history context)
                action, data = self._ask_vision(grid, step, strategy, last_actions)
                if action is None:
                    break

                # Execute
                if action == 6 and data:
                    obs = env.step(6, data=data)
                    level_actions.append((data["x"], data["y"]))
                    last_actions.append(f"CLICK({data['x']},{data['y']})")
                else:
                    obs = env.step(action)
                    level_actions.append(action)
                    last_actions.append(_ACTION_NAMES.get(action, f"A{action}"))

                result.total_steps += 1

                # Check win
                if obs.levels_completed > current_levels:
                    log.info("arc.vision_level_solved",
                             game_id=self._game_id, level=level,
                             steps=len(level_actions),
                             time_s=round(time.monotonic() - t0, 1))
                    prev_actions.extend(level_actions)
                    result.levels_completed += 1
                    break

                if obs.state == GameState.GAME_OVER:
                    log.info("arc.vision_game_over",
                             level=level, step=step)
                    break

                # Get strategy on first step
                if step == 0 and strategy is None:
                    strategy = self._get_strategy(grid)

            if obs.levels_completed <= current_levels:
                break

        return result

    def _get_strategy(self, grid: np.ndarray) -> str | None:
        """Ask vision for high-level strategy (called once per level)."""
        try:
            import ollama

            b64 = _grid_to_png_b64(grid, scale=4)
            resp = ollama.chat(
                model="qwen3-vl:32b",
                messages=[{
                    "role": "user",
                    "content": (
                        f"64x64 pixel game. I can press: {self._action_str}. "
                        "What is the goal of this game? Where should I navigate to? "
                        "Describe the target position briefly."
                    ),
                    "images": [b64],
                }],
                options={"num_predict": 128, "temperature": 0.3, "num_ctx": 4096},
            )
            raw = resp.get("message", {}).get("content", "")
            if "</think>" in raw:
                raw = raw.split("</think>")[-1].strip()
            return raw[:200] if raw else None
        except Exception:
            return None

    def _ask_vision(
        self, grid: np.ndarray, step: int, strategy: str | None,
        last_actions: list[str] | None = None,
    ) -> tuple[int | None, dict | None]:
        """Ask vision for the next action."""
        try:
            import ollama

            b64 = _grid_to_png_b64(grid, scale=4)

            context = ""
            if strategy:
                context = f"Goal: {strategy}\n"
            if last_actions:
                recent = ", ".join(last_actions[-10:])
                context += f"My last moves: {recent}\n"

            resp = ollama.chat(
                model="qwen3-vl:32b",
                messages=[{
                    "role": "user",
                    "content": (
                        f"{context}"
                        f"I can ONLY press: {self._action_str}. "
                        f"What should I press NEXT to reach the goal? Say just the action name."
                    ),
                    "images": [b64],
                }],
                options={"num_predict": 512, "temperature": 0.3, "num_ctx": 8192},
            )
            raw = resp.get("message", {}).get("content", "")
            if "</think>" in raw:
                raw = raw.split("</think>")[-1].strip()

            return self._parse_action(raw)
        except Exception as exc:
            log.debug("arc.vision_agent_error", error=str(exc)[:100])
            return None, None

    def _parse_action(self, raw: str) -> tuple[int | None, dict | None]:
        """Parse vision response into action + data."""
        # Strip markdown bold/italic
        clean = re.sub(r"\*+", "", raw).strip().upper()

        # Check for CLICK with coordinates
        click_match = re.search(r"CLICK\s*\(?(\d+)\s*,\s*(\d+)\)?", clean)
        if click_match:
            x, y = int(click_match.group(1)), int(click_match.group(2))
            return 6, {"x": min(63, x), "y": min(63, y)}

        # Match action names (check available actions first)
        for name, action_id in _NAME_TO_ACTION.items():
            if name in clean and action_id in self._actions:
                log.debug("arc.vision_parsed", raw=raw[:60], action=name)
                return action_id, None

        # If vision says UP/DOWN but only LEFT/RIGHT available (or vice versa),
        # map to the first available keyboard action
        for name, action_id in _NAME_TO_ACTION.items():
            if name in clean:
                # Action not available — try first available keyboard action
                for fallback in [1, 2, 3, 4, 5]:
                    if fallback in self._actions:
                        log.debug("arc.vision_fallback",
                                  wanted=name, using=_ACTION_NAMES.get(fallback))
                        return fallback, None

        # Fallback: try to find any action number
        num_match = re.search(r"ACTION\s*(\d+)", clean)
        if num_match:
            a = int(num_match.group(1))
            if a in self._actions:
                return a, None

        # Last resort: return first available action
        if self._actions:
            log.debug("arc.vision_unparsed", raw=raw[:80])
            return self._actions[0], None

        return None, None
