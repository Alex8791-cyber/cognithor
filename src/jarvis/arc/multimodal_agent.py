"""ARC-AGI-3 Multimodal Agent — vision-based analyze-plan-act loop.

Based on the official ARC-AGI-3 reference implementation.
Uses qwen3-vl:32b to observe game frames, analyze changes,
discover rules, and plan actions. Cumulative memory of
discovered game mechanics.

40-action maximum per game. One LLM call per step (combined
analyze+plan). 128x128 PNG images (2x upscale from 64x64 grid).
"""

from __future__ import annotations

import base64
import io
import json
import re
from typing import Any

import numpy as np
from PIL import Image

from jarvis.utils.logging import get_logger

__all__ = ["MultimodalArcAgent"]

log = get_logger(__name__)

# 16-color palette (official ARC-AGI-3)
PALETTE = [
    (255, 255, 255),
    (0, 0, 0),
    (0, 116, 217),
    (255, 65, 54),
    (46, 204, 64),
    (255, 220, 0),
    (170, 170, 170),
    (255, 133, 27),
    (127, 219, 255),
    (135, 12, 37),
    (240, 18, 190),
    (200, 200, 200),
    (200, 200, 100),
    (100, 50, 150),
    (0, 200, 200),
    (128, 0, 255),
]

ACTION_NAMES = {
    1: "ACTION1 (UP)",
    2: "ACTION2 (DOWN)",
    3: "ACTION3 (LEFT)",
    4: "ACTION4 (RIGHT)",
    5: "ACTION5 (Interact/Undo)",
    6: "ACTION6 (Click x,y)",
}

_SYSTEM_PROMPT = (
    "Du bist ein Agent der ein Puzzle-Spiel loest. "
    "Du siehst 64x64 Pixel-Grids als Bilder. "
    "Dein Ziel: das Level abschliessen in moeglichst wenigen Aktionen. "
    "Lerne aus jeder Aktion was passiert und passe deine Strategie an."
)

MAX_ACTIONS = 40


def grid_to_image(grid: np.ndarray, scale: int = 2) -> Image.Image:
    """Convert 64x64 grid to upscaled PIL Image."""
    if grid.ndim == 3:
        grid = grid[0]
    h, w = grid.shape
    img = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            color = PALETTE[min(int(grid[r, c]), 15)]
            img[r * scale : (r + 1) * scale, c * scale : (c + 1) * scale] = color
    return Image.fromarray(img)


def image_to_b64(img: Image.Image) -> str:
    """PIL Image to base64 PNG string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def image_diff(before: Image.Image, after: Image.Image) -> Image.Image:
    """Highlight changed pixels in red on black background."""
    a = np.array(before)
    b = np.array(after)
    diff_mask = np.any(a != b, axis=-1)
    result = np.zeros_like(a)
    result[diff_mask] = [255, 0, 0]
    return Image.fromarray(result)


def _parse_json(raw: str) -> dict | None:
    """Extract JSON from LLM response. Strips think tags, handles markdown."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # Try direct parse
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    # Markdown code block
    md = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if md:
        try:
            data = json.loads(md.group(1))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    # Balanced brace search
    pos = raw.find("{")
    if pos != -1:
        depth = 0
        for i in range(pos, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
            if depth == 0:
                try:
                    data = json.loads(raw[pos : i + 1])
                    if isinstance(data, dict):
                        return data
                except (json.JSONDecodeError, ValueError):
                    pass
                break

    return None


class MultimodalArcAgent:
    """Vision-based ARC-AGI-3 agent with cumulative rule discovery."""

    def __init__(self, model: str = "qwen3-vl:32b") -> None:
        self.model = model
        self.memory: list[str] = []
        self.rules: list[str] = []
        self.action_history: list[dict] = []
        self.prev_image: Image.Image | None = None
        self.prev_action: str | None = None
        self.prev_expected: str | None = None
        self.total_calls: int = 0

    def _build_memory_prompt(self) -> str:
        """Build cumulative memory of discovered rules and history."""
        lines = ["## Spielwissen"]
        if self.rules:
            lines.append("Entdeckte Regeln:")
            for i, rule in enumerate(self.rules, 1):
                lines.append(f"  {i}. {rule}")
        else:
            lines.append("Noch keine Regeln entdeckt. Exploriere!")

        if self.action_history:
            lines.append(f"\nLetzte {min(10, len(self.action_history))} Aktionen:")
            for entry in self.action_history[-10:]:
                lines.append(f"  {entry['action']}: {entry.get('observation', '?')}")

        return "\n".join(lines)

    def _call_llm(self, prompt: str, images: list[str]) -> dict | None:
        """Send prompt + images to qwen3-vl:32b, return parsed JSON."""
        try:
            import ollama

            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt, "images": images},
            ]

            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={"num_predict": 500, "temperature": 0.3},
            )

            raw = response.get("message", {}).get("content", "")
            self.total_calls += 1
            return _parse_json(raw)

        except Exception as exc:
            log.debug("multimodal_llm_failed", error=str(exc)[:200])
            return None

    def choose_action(
        self,
        current_grid: np.ndarray,
        available_actions: list[int],
    ) -> tuple[int, dict]:
        """Choose next action based on visual analysis.

        Returns (action_id, data_dict).
        """
        current_img = grid_to_image(current_grid)
        current_b64 = image_to_b64(current_img)

        # Build available actions description
        action_desc = ", ".join(ACTION_NAMES.get(a, f"ACTION{a}") for a in available_actions)

        # Build prompt
        images = [current_b64]

        if self.prev_image and self.prev_action:
            # Include before + diff for analysis
            prev_b64 = image_to_b64(self.prev_image)
            diff_img = image_diff(self.prev_image, current_img)
            diff_b64 = image_to_b64(diff_img)
            images = [prev_b64, current_b64, diff_b64]

            prompt = (
                f"{self._build_memory_prompt()}\n\n"
                f"Vorherige Aktion: {self.prev_action}\n"
                f"Erwartet: {self.prev_expected or '?'}\n\n"
                "Bild 1: Vorher. Bild 2: Nachher. Bild 3: Aenderungen (rot).\n\n"
                f"Verfuegbare Aktionen: {action_desc}\n\n"
                "1. Was hat sich geaendert? Neue Regel entdeckt?\n"
                "2. Was soll ich als naechstes tun?\n\n"
                "Antworte als JSON:\n"
                '{"observation": "...", "new_rule": "..." oder null, '
                '"next_action": "ACTION1", "reasoning": "...", '
                '"expected_outcome": "..."}'
            )
        else:
            # First step — no previous action
            prompt = (
                f"{self._build_memory_prompt()}\n\n"
                "Dies ist der Startbildschirm des Spiels.\n\n"
                f"Verfuegbare Aktionen: {action_desc}\n\n"
                "Was siehst du? Was koennte das Ziel sein?\n"
                "Welche Aktion soll ich zuerst probieren?\n\n"
                "Antworte als JSON:\n"
                '{"observation": "...", "next_action": "ACTION1", '
                '"reasoning": "...", "expected_outcome": "..."}'
            )

        # Call LLM
        result = self._call_llm(prompt, images)

        if result:
            # Update memory
            obs = result.get("observation", "")
            new_rule = result.get("new_rule")
            if new_rule and new_rule not in self.rules:
                self.rules.append(new_rule)
                log.info("arc_rule_discovered", rule=new_rule[:80])

            self.action_history.append(
                {
                    "action": result.get("next_action", "?"),
                    "observation": obs[:80],
                    "reasoning": result.get("reasoning", "")[:80],
                }
            )

            # Parse action
            action_str = result.get("next_action", "ACTION1")
            action_id = self._parse_action(action_str, available_actions)

            # Handle click coordinates
            data = {}
            if action_id == 6:
                # Extract x,y from reasoning or action string
                xy_match = re.search(r"(\d+)\s*,\s*(\d+)", action_str)
                if xy_match:
                    data = {"x": int(xy_match.group(1)), "y": int(xy_match.group(2))}
                else:
                    data = {"x": 32, "y": 32}  # center fallback

            self.prev_image = current_img
            self.prev_action = action_str
            self.prev_expected = result.get("expected_outcome", "")

            log.info(
                "arc_multimodal_step",
                action=action_str,
                rules=len(self.rules),
                calls=self.total_calls,
            )

            return action_id, data

        # LLM failed — random fallback
        import random

        action_id = random.choice(available_actions) if available_actions else 1
        self.prev_image = current_img
        return action_id, {}

    @staticmethod
    def _parse_action(action_str: str, available: list[int]) -> int:
        """Parse action string to action ID."""
        # Extract number from "ACTION1", "ACTION2", etc.
        match = re.search(r"ACTION(\d+)", action_str.upper())
        if match:
            action_id = int(match.group(1))
            if action_id in available:
                return action_id
        # Fallback to first available
        return available[0] if available else 1

    def play_game(self, game_id: str) -> dict[str, Any]:
        """Play a complete ARC-AGI-3 game."""
        import arc_agi
        from arcengine.enums import GameState

        arcade = arc_agi.Arcade()
        env = arcade.make(game_id)
        obs = env.reset()

        total_steps = 0
        levels = 0

        for _step in range(MAX_ACTIONS):
            grid = np.array(obs.frame)
            if grid.ndim == 3:
                grid = grid[0]

            available = [
                a.value if hasattr(a, "value") else int(a)
                for a in (obs.available_actions or [1, 2, 3, 4])
            ]

            action_id, data = self.choose_action(grid, available)
            obs = env.step(action_id, data=data or None)
            total_steps += 1

            if obs.levels_completed > levels:
                levels = obs.levels_completed
                log.info("arc_multimodal_level", level=levels, step=total_steps)

            if obs.state == GameState.WIN:
                log.info("arc_multimodal_win", steps=total_steps, levels=levels)
                break

            if obs.state == GameState.GAME_OVER:
                log.info("arc_multimodal_game_over", steps=total_steps)
                obs = env.reset()

        # Scorecard
        score = 0.0
        try:
            sc = arcade.get_scorecard()
            score = float(getattr(sc, "score", 0.0))
        except Exception:
            pass

        return {
            "game_id": game_id,
            "levels_completed": levels,
            "total_steps": total_steps,
            "llm_calls": self.total_calls,
            "rules_discovered": len(self.rules),
            "rules": list(self.rules),
            "score": score,
        }
