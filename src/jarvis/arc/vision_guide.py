"""ARC-AGI-3 Vision Guide — consults qwen3-vl:32b on game frames.

Converts 64x64 color-index grids to 512x512 PNG images, sends them
to the vision model, and returns strategic action recommendations.
"""

from __future__ import annotations

import base64
import io
import json
import re

import numpy as np

from jarvis.utils.logging import get_logger

__all__ = ["ArcVisionGuide", "grid_to_png_b64"]

log = get_logger(__name__)

ARC_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (0, 116, 217),
    2: (255, 65, 54),
    3: (46, 204, 64),
    4: (255, 220, 0),
    5: (170, 170, 170),
    6: (240, 18, 190),
    7: (255, 133, 27),
    8: (127, 219, 255),
    9: (135, 12, 37),
    10: (255, 255, 255),
    11: (200, 200, 100),
    12: (100, 50, 150),
}

_VISION_PROMPT = (
    "Du siehst einen Frame aus einem Puzzle-Spiel.\n"
    "Verfuegbare Aktionen: {action_names}\n\n"
    "{context}"
    "Beschreibe kurz:\n"
    "1. Was siehst du? (Objekte, Farben, Layout)\n"
    "2. Was ist das wahrscheinliche Ziel?\n"
    "3. Welche Aktion empfiehlst du als naechstes?\n\n"
    'Antworte als JSON: {{"goal": "...", "strategy": "...", "next_action": "ACTION1"}}'
)


def grid_to_png_b64(grid: np.ndarray, scale: int = 8) -> str:
    """Convert 64x64 color-index grid to upscaled PNG base64.

    Args:
        grid: 2D array of color indices (0-12).
        scale: Upscale factor (8 = 512x512 output).

    Returns:
        Base64-encoded PNG string.
    """
    from PIL import Image

    if grid.ndim == 3:
        grid = grid[0]

    h, w = grid.shape
    img = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            color = ARC_COLORS.get(int(grid[r, c]), (128, 128, 128))
            img[r * scale : (r + 1) * scale, c * scale : (c + 1) * scale] = color

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _parse_guidance(raw: str) -> dict | None:
    """Parse JSON guidance from vision model response. 3-tier fallback."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # Tier 1: direct parse
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "next_action" in data:
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    # Tier 2: markdown code block
    md = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if md:
        try:
            data = json.loads(md.group(1))
            if isinstance(data, dict) and "next_action" in data:
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    # Tier 3: find JSON object with next_action
    pos = raw.find('"next_action"')
    if pos != -1:
        brace = raw.rfind("{", 0, pos)
        if brace != -1:
            depth = 0
            for i in range(brace, len(raw)):
                if raw[i] == "{":
                    depth += 1
                elif raw[i] == "}":
                    depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(raw[brace : i + 1])
                        if "next_action" in data:
                            return data
                    except (json.JSONDecodeError, ValueError):
                        pass
                    break

    return None


class ArcVisionGuide:
    """Consults qwen3-vl:32b to understand game frames and suggest actions."""

    def __init__(
        self,
        model: str = "qwen3-vl:32b",
        call_interval: int = 50,
        min_pixel_change: int = 100,
    ) -> None:
        self.model = model
        self.call_interval = call_interval
        self.min_pixel_change = min_pixel_change
        self._steps_since_call: int = 0
        self._pixels_since_call: int = 0
        self._last_strategy: dict | None = None
        self._force_next: bool = False  # Wait for call_interval before first call
        self.call_count: int = 0
        self.actions_followed: int = 0

    def should_call(self, changed_pixels: int) -> bool:
        """Check if it's time to consult the vision model."""
        self._steps_since_call += 1
        self._pixels_since_call += changed_pixels

        if self._force_next:
            return True

        return (
            self._steps_since_call >= self.call_interval
            and self._pixels_since_call > self.min_pixel_change
        )

    def force_next_call(self) -> None:
        """Force the next should_call to return True (after GAME_OVER etc.)."""
        self._force_next = True

    def analyze_sync(self, grid: np.ndarray, action_names: list[str]) -> dict | None:
        """Synchronous wrapper for analyze. Returns {goal, strategy, next_action}."""
        try:
            import ollama

            b64 = grid_to_png_b64(grid)

            context = ""
            if self._last_strategy:
                context = (
                    f"Bisherige Strategie: {self._last_strategy.get('strategy', '')}\n"
                    f"Bisheriges Ziel: {self._last_strategy.get('goal', '')}\n\n"
                )

            prompt = _VISION_PROMPT.format(
                action_names=", ".join(action_names),
                context=context,
            )

            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [b64],
                    }
                ],
                options={"num_predict": 2000, "temperature": 0.3, "num_ctx": 8192},
            )

            raw = response.get("message", {}).get("content", "")
            guidance = _parse_guidance(raw)

            if guidance:
                self._last_strategy = guidance
                self.call_count += 1
                log.info(
                    "arc_vision_guidance",
                    goal=guidance.get("goal", "")[:80],
                    action=guidance.get("next_action", ""),
                    calls=self.call_count,
                )

            # Reset counters
            self._steps_since_call = 0
            self._pixels_since_call = 0
            self._force_next = False

            return guidance

        except Exception as exc:
            log.debug("arc_vision_guide_failed", error=str(exc)[:200])
            self._steps_since_call = 0
            self._force_next = False
            return None

    @property
    def current_strategy(self) -> dict | None:
        """Last strategy from vision model (cached between calls)."""
        return self._last_strategy
