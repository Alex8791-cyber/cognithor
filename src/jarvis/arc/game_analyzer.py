"""ARC-AGI-3 GameAnalyzer — sacrifice-level analysis + 2 vision calls to build GameProfile."""

from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from jarvis.utils.logging import get_logger

__all__ = ["GameAnalyzer"]

log = get_logger(__name__)

PALETTE = [
    (255, 255, 255), (0, 0, 0), (0, 116, 217), (255, 65, 54),
    (46, 204, 64), (255, 220, 0), (170, 170, 170), (255, 133, 27),
    (127, 219, 255), (135, 12, 37), (240, 18, 190), (200, 200, 200),
    (200, 200, 100), (100, 50, 150), (0, 200, 200), (128, 0, 255),
]

_ACTION_NAMES = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "Interact", 6: "Click(x,y)"}


def _grid_to_png_b64(grid: np.ndarray, scale: int = 4) -> str:
    """Convert 64x64 colour-index grid to upscaled PNG as base64."""
    from PIL import Image

    if grid.ndim == 3:
        grid = grid[0]
    h, w = grid.shape
    img = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            color = PALETTE[min(int(grid[r, c]), 15)]
            img[r * scale : (r + 1) * scale, c * scale : (c + 1) * scale] = color
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _parse_vision_json(raw: str) -> dict | None:
    """3-tier JSON extraction: direct parse, markdown block, balanced brace."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    md = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if md:
        try:
            data = json.loads(md.group(1))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

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


@dataclass
class SacrificeReport:
    """Results from the sacrifice level exploration."""

    clicks_tested: list[tuple[int, int, str]] = field(default_factory=list)
    movements_tested: dict[int, int] = field(default_factory=dict)
    unique_states_seen: int = 0
    game_over_trigger: str | None = None
    frames: list[np.ndarray] = field(default_factory=list)


class GameAnalyzer:
    """Analyzes ARC-AGI-3 games by sacrificing one level + 2 vision calls."""

    def __init__(self, arcade: Any | None = None):
        self._arcade = arcade

    def _run_sacrifice_level(
        self,
        env: Any,
        initial_grid: np.ndarray,
        available_action_ids: list[int],
    ) -> SacrificeReport:
        """Execute the sacrifice level: test actions systematically."""
        from arcengine.enums import GameState

        from jarvis.arc.error_handler import safe_frame_extract

        report = SacrificeReport()
        report.frames.append(initial_grid.copy())
        seen_states: set[int] = {hash(initial_grid.tobytes())}
        current_grid = initial_grid.copy()

        has_click = 6 in available_action_ids
        has_keyboard = any(a in available_action_ids for a in [1, 2, 3, 4])

        # Phase 1: Test keyboard directions (3 times each)
        if has_keyboard:
            for action_id in [1, 2, 3, 4]:
                if action_id not in available_action_ids:
                    continue
                total_diff = 0
                for _ in range(3):
                    obs = env.step(action_id)
                    new_grid = safe_frame_extract(obs)
                    diff = int(np.sum(new_grid != current_grid))
                    total_diff += diff
                    state_hash = hash(new_grid.tobytes())
                    if state_hash not in seen_states:
                        seen_states.add(state_hash)
                    current_grid = new_grid

                    if hasattr(obs, "state") and obs.state == GameState.GAME_OVER:
                        report.game_over_trigger = f"keyboard_action_{action_id}"
                        report.unique_states_seen = len(seen_states)
                        return report

                report.movements_tested[action_id] = total_diff

        # Phase 2: Test clicks on cluster centers
        if has_click:
            from jarvis.arc.cluster_solver import ClusterSolver

            # Find non-background colours
            unique_colors = [int(c) for c in np.unique(initial_grid) if c != 0]

            for color in unique_colors:
                solver = ClusterSolver(target_color=color, max_skip=0)
                centers = solver.find_clusters(initial_grid)

                for cx, cy in centers:
                    obs = env.step(6, data={"x": cx, "y": cy})
                    new_grid = safe_frame_extract(obs)
                    diff = int(np.sum(new_grid != current_grid))
                    effect = "changed" if diff > 0 else "no_effect"
                    report.clicks_tested.append((cx, cy, effect))

                    state_hash = hash(new_grid.tobytes())
                    if state_hash not in seen_states:
                        seen_states.add(state_hash)
                        report.frames.append(new_grid.copy())
                    current_grid = new_grid

                    if hasattr(obs, "state") and obs.state == GameState.GAME_OVER:
                        report.game_over_trigger = f"click_at_{cx}_{cy}"
                        report.unique_states_seen = len(seen_states)
                        return report

        report.unique_states_seen = len(seen_states)
        return report
