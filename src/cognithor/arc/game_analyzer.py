"""ARC-AGI-3 GameAnalyzer — sacrifice-level analysis + 2 vision calls to build GameProfile."""

from __future__ import annotations

import base64
import contextlib
import io
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from cognithor.arc.game_profile import GameProfile

from datetime import UTC

from cognithor.utils.logging import get_logger

try:
    import ollama
except ImportError:
    ollama = None  # type: ignore[assignment]

__all__ = ["GameAnalyzer"]

log = get_logger(__name__)

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

_ACTION_NAMES = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "Interact", 6: "Click(x,y)"}


def _grid_to_png_b64(grid: np.ndarray[Any, Any], scale: int = 4) -> str:
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


def _parse_vision_json(raw: str) -> dict[str, Any] | None:
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
    toggle_pairs: list[tuple[int, int]] = field(
        default_factory=list
    )  # (source, target) color pairs
    frames: list[np.ndarray[Any, Any]] = field(default_factory=list)


class GameAnalyzer:
    """Analyzes ARC-AGI-3 games by sacrificing one level + 2 vision calls."""

    def __init__(self, arcade: Any | None = None):
        self._arcade = arcade

    def _run_sacrifice_level(
        self,
        env: Any,
        initial_grid: np.ndarray[Any, Any],
        available_action_ids: list[int],
    ) -> SacrificeReport:
        """Execute the sacrifice level: test actions systematically."""
        from arcengine.enums import GameState

        from cognithor.arc.error_handler import safe_frame_extract

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

                    if hasattr(obs, "state") and obs.state == GameState.WIN:
                        report.game_over_trigger = "win_during_sacrifice"
                        report.unique_states_seen = len(seen_states)
                        return report

                report.movements_tested[action_id] = total_diff

        # Phase 2: Test clicks on cluster centers
        if has_click:
            from cognithor.arc.cluster_solver import ClusterSolver

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

                    # Detect toggle pair: what color changed to what
                    if diff > 0:
                        changed_mask = new_grid != current_grid
                        old_vals = current_grid[changed_mask]
                        new_vals = new_grid[changed_mask]
                        if len(old_vals) > 0:
                            # Most common source→target transition
                            from collections import Counter

                            pairs = Counter(zip(old_vals.tolist(), new_vals.tolist(), strict=False))
                            src, tgt = pairs.most_common(1)[0][0]
                            if (src, tgt) not in report.toggle_pairs:
                                report.toggle_pairs.append((src, tgt))

                    state_hash = hash(new_grid.tobytes())
                    if state_hash not in seen_states:
                        seen_states.add(state_hash)
                        report.frames.append(new_grid.copy())
                    current_grid = new_grid

                    if hasattr(obs, "state") and obs.state == GameState.GAME_OVER:
                        report.game_over_trigger = f"click_at_{cx}_{cy}"
                        report.unique_states_seen = len(seen_states)
                        return report

                    if hasattr(obs, "state") and obs.state == GameState.WIN:
                        report.game_over_trigger = "win_during_sacrifice"
                        report.unique_states_seen = len(seen_states)
                        return report

        report.unique_states_seen = len(seen_states)
        return report

    def _vision_call_initial(
        self, grid: np.ndarray[Any, Any], action_ids: list[int]
    ) -> dict[str, Any] | None:
        """Vision call 1: ask what the game is from initial frame."""
        try:
            b64 = _grid_to_png_b64(grid, scale=4)
            action_desc = [f"ACTION{a}={_ACTION_NAMES.get(a, '?')}" for a in action_ids]

            resp = ollama.chat(
                model="qwen3-vl:32b",
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"64x64 pixel puzzle game. "
                            f"Available actions: {', '.join(action_desc)}.\n"
                            "Analyze this game:\n"
                            "1. What type of game is this? (click, keyboard, or mixed)\n"
                            "2. What is the goal?\n"
                            "3. Which colors are interactive?\n"
                            "4. What strategy should I use?\n"
                            'Reply JSON: {"game_type": "click"|"keyboard"|"mixed", '
                            '"target_color": N or null, "strategy": "...", '
                            '"description": "..."}'
                        ),
                        "images": [b64],
                    }
                ],
                options={"num_predict": 8192, "temperature": 0.3, "num_ctx": 8192},
            )

            raw = resp.get("message", {}).get("content", "")
            return _parse_vision_json(raw)
        except Exception as exc:
            log.debug("arc.vision_call_1_failed", error=str(exc)[:200])
            return None

    def _vision_call_final(
        self, grid_before: np.ndarray[Any, Any], grid_after: np.ndarray[Any, Any]
    ) -> dict[str, Any] | None:
        """Vision call 2: compare before/after sacrifice level."""
        try:
            b64_before = _grid_to_png_b64(grid_before, scale=4)
            b64_after = _grid_to_png_b64(grid_after, scale=4)

            # Create diff image: highlight changed pixels in red
            diff_grid = np.where(grid_before != grid_after, 3, grid_before)
            b64_diff = _grid_to_png_b64(diff_grid.astype(np.int8), scale=4)

            resp = ollama.chat(
                model="qwen3-vl:32b",
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Three images of a 64x64 puzzle game:\n"
                            "1. Initial state\n"
                            "2. After testing actions\n"
                            "3. Diff (changes highlighted in red)\n\n"
                            "What changed? What is the win condition?\n"
                            'Reply JSON: {"win_condition": "clear_board"|"reach_state"|'
                            '"navigate"|"unknown", "correction": null or "...", '
                            '"description": "..."}'
                        ),
                        "images": [b64_before, b64_after, b64_diff],
                    }
                ],
                options={"num_predict": 8192, "temperature": 0.3, "num_ctx": 8192},
            )

            raw = resp.get("message", {}).get("content", "")
            return _parse_vision_json(raw)
        except Exception as exc:
            log.debug("arc.vision_call_2_failed", error=str(exc)[:200])
            return None

    def analyze(
        self,
        game_id: str,
        *,
        force: bool = False,
        base_dir: Any | None = None,
    ) -> GameProfile:
        """Analyze a game: load from cache or run sacrifice level + 2 vision calls."""
        from datetime import datetime

        from cognithor.arc.error_handler import safe_frame_extract
        from cognithor.arc.game_profile import GameProfile

        # Cache check
        if not force and GameProfile.exists(game_id, base_dir=base_dir):
            cached = GameProfile.load(game_id, base_dir=base_dir)
            if cached is not None:
                log.info("arc.profile_cache_hit", game_id=game_id)
                return cached

        # Create environment
        env = self._arcade.make(game_id)  # type: ignore[union-attr]
        obs = env.reset()
        initial_grid = safe_frame_extract(obs)

        # Extract available action IDs
        action_ids: list[int] = []
        if hasattr(obs, "available_actions") and obs.available_actions:
            for a in obs.available_actions:
                action_ids.append(a.value if hasattr(a, "value") else int(a))
        if not action_ids:
            action_ids = [1, 2, 3, 4, 5, 6]

        # Determine game type from actions
        has_click = 6 in action_ids
        has_keyboard = any(a in action_ids for a in [1, 2, 3, 4])
        if has_click and has_keyboard:
            game_type = "mixed"
        elif has_click:
            game_type = "click"
        else:
            game_type = "keyboard"

        # Vision call 1
        vision1 = self._vision_call_initial(initial_grid, action_ids)
        if vision1 and "game_type" in vision1:
            game_type = vision1["game_type"]

        # Sacrifice level
        report = self._run_sacrifice_level(env, initial_grid, action_ids)

        # Vision call 2
        final_grid = report.frames[-1] if report.frames else initial_grid
        vision2 = self._vision_call_final(initial_grid, final_grid)

        # Determine win condition
        win_condition = "unknown"
        if vision2 and "win_condition" in vision2:
            win_condition = vision2["win_condition"]

        # Correct game_type if vision2 disagrees
        if vision2 and vision2.get("correction"):
            correction = vision2["correction"]
            if correction in ("click", "keyboard", "mixed"):
                game_type = correction

        # Extract target colors: prefer vision, fallback to sacrifice level toggle detection
        target_colors: list[int] = []
        if vision1 and vision1.get("target_color") is not None:
            with contextlib.suppress(ValueError, TypeError):
                target_colors = [int(vision1["target_color"])]
        if not target_colors and report.toggle_pairs:
            # Use source colors from detected toggle pairs (the color you click on)
            target_colors = list({src for src, _tgt in report.toggle_pairs})

        # Extract click zones from report
        click_zones = [(x, y) for x, y, effect in report.clicks_tested if effect == "changed"]

        # Build movement effects
        movement_effects: dict[int, str] = {}
        for action_id, diff in report.movements_tested.items():
            if diff > 20:
                movement_effects[action_id] = "moves_player"
            elif diff > 0:
                movement_effects[action_id] = "transforms"
            else:
                movement_effects[action_id] = "no_effect"

        profile = GameProfile(
            game_id=game_id,
            game_type=game_type,  # type: ignore[arg-type]
            available_actions=action_ids,
            click_zones=click_zones,
            target_colors=target_colors,
            movement_effects=movement_effects,
            win_condition=win_condition,
            vision_description=vision1.get("description", "") if vision1 else "unavailable",
            vision_strategy=vision1.get("strategy", "") if vision1 else "unavailable",
            strategy_metrics={},
            has_toggles=len(report.toggle_pairs) > 0,
            analyzed_at=datetime.now(UTC).isoformat(),
        )

        profile.save(base_dir=base_dir)
        return profile
