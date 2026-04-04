"""ARC-AGI-3 Hybrid Solver — 1 GPU vision call + fast CPU solver.

Strategy:
1. Take screenshot of initial game frame
2. ONE vision call to qwen3-vl:32b: "What game is this? What should I click/press?"
3. Use the answer to configure the right solver (click targets, keyboard sequence)
4. Execute fast on CPU

This combines GPU intelligence with CPU speed.
"""

from __future__ import annotations

import itertools
import random
import re
import time
from typing import Any

import numpy as np
from scipy import ndimage

from jarvis.utils.logging import get_logger

__all__ = ["HybridSolver", "run_all_games"]

log = get_logger(__name__)


def _ask_vision(grid: np.ndarray, actions: list[int]) -> dict | None:
    """ONE vision call: ask what the game is and how to play."""
    try:
        import base64
        import io
        import json

        from PIL import Image

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

        if grid.ndim == 3:
            grid = grid[0]
        h, w = grid.shape
        scale = 4
        img = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)
        for r in range(h):
            for c in range(w):
                color = PALETTE[min(int(grid[r, c]), 15)]
                img[r * scale : (r + 1) * scale, c * scale : (c + 1) * scale] = color

        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        action_desc = []
        for a in actions:
            names = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "Interact", 6: "Click(x,y)"}
            action_desc.append(f"ACTION{a}={names.get(a, '?')}")

        import ollama

        resp = ollama.chat(
            model="qwen3-vl:32b",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"64x64 pixel puzzle game. Actions: {', '.join(action_desc)}.\n"
                        "1. What type of game is this?\n"
                        "2. For CLICK games: which color should I click? "
                        "Give the color NUMBER (0-15).\n"
                        "3. For KEYBOARD games: what sequence should I try?\n"
                        'Reply JSON: {"game_type": "click" or "keyboard", '
                        '"target_color": N or null, "strategy": "...", '
                        '"first_actions": ["ACTION1","ACTION2",...]}'
                    ),
                    "images": [b64],
                }
            ],
            options={"num_predict": 2000, "temperature": 0.3, "num_ctx": 8192},
        )

        raw = resp.get("message", {}).get("content", "")
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        # Parse JSON
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            pass
        md = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if md:
            try:
                return json.loads(md.group(1))
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
                        return json.loads(raw[pos : i + 1])
                    except (json.JSONDecodeError, ValueError):
                        pass
                    break

        log.debug("vision_parse_failed", raw=raw[:200])
        return None

    except Exception as exc:
        log.debug("vision_call_failed", error=str(exc)[:200])
        return None


class HybridSolver:
    """1 GPU vision call + fast CPU solver per game."""

    def __init__(self, max_skip: int = 6, max_keyboard_steps: int = 300) -> None:
        self.max_skip = max_skip
        self.max_keyboard_steps = max_keyboard_steps

    def solve_game(self, game_id: str, use_vision: bool = True) -> dict[str, Any]:
        """Play one game."""
        import arc_agi

        arcade = arc_agi.Arcade()
        env = arcade.make(game_id)
        obs = env.reset()

        grid = np.array(obs.frame)
        if grid.ndim == 3:
            grid = grid[0]

        actions = [
            a.value if hasattr(a, "value") else int(a) for a in (obs.available_actions or [])
        ]

        t0 = time.monotonic()

        # Step 1: ONE vision call
        guidance = None
        if use_vision:
            guidance = _ask_vision(grid, actions)
            if guidance:
                log.info(
                    "hybrid_vision",
                    game=game_id.split("-")[0],
                    type=guidance.get("game_type"),
                    color=guidance.get("target_color"),
                    strategy=str(guidance.get("strategy", ""))[:80],
                )

        # Step 2: Route to right solver
        has_click = 6 in actions

        if has_click:
            target_color = None
            if guidance and guidance.get("target_color") is not None:
                target_color = int(guidance["target_color"])
            result = self._solve_click(arcade, game_id, grid, target_color)
        else:
            first_actions = None
            if guidance and guidance.get("first_actions"):
                first_actions = guidance["first_actions"]
            result = self._solve_keyboard(arcade, game_id, first_actions)

        result["elapsed_s"] = round(time.monotonic() - t0, 1)
        result["game_id"] = game_id.split("-")[0]
        result["vision_used"] = guidance is not None
        return result

    def _solve_click(
        self,
        arcade: Any,
        game_id: str,
        grid: np.ndarray,
        target_color: int | None = None,
    ) -> dict[str, Any]:
        """Click solver: find clusters of target color, subset search."""
        from arcengine.enums import GameState

        colors, counts = np.unique(grid, return_counts=True)
        bg = colors[np.argmax(counts)]

        if target_color is not None:
            color_order = [target_color]
        else:
            color_order = []
            for c in colors:
                if c == bg:
                    continue
                _, n = ndimage.label(grid == c)
                if 2 <= n <= 30:
                    color_order.append((n, int(c)))
            color_order.sort()
            color_order = [c for _, c in color_order]

        all_solutions: list[list[tuple[int, int]]] = []
        levels = 0

        for _attempt in range(10):
            env = arcade.make(game_id)
            obs = env.reset()
            for sol in all_solutions:
                for cx, cy in sol:
                    obs = env.step(6, data={"x": cx, "y": cy})

            cur_grid = np.array(obs.frame)
            if cur_grid.ndim == 3:
                cur_grid = cur_grid[0]

            solved = False
            for color in color_order:
                labeled, n = ndimage.label(cur_grid == color)
                if n == 0:
                    continue

                centers = []
                for i in range(1, n + 1):
                    ys, xs = np.where(labeled == i)
                    centers.append((int(np.mean(xs)), int(np.mean(ys))))

                for skip in range(min(n + 1, self.max_skip + 1)):
                    if solved:
                        break
                    for combo in itertools.combinations(range(n), skip):
                        click_idx = [i for i in range(n) if i not in combo]
                        clicks = [centers[i] for i in click_idx]

                        env_t = arcade.make(game_id)
                        obs_t = env_t.reset()
                        ok = True
                        for prev in all_solutions:
                            for cx, cy in prev:
                                obs_t = env_t.step(6, data={"x": cx, "y": cy})
                                if obs_t.state == GameState.GAME_OVER:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            continue

                        for cx, cy in clicks:
                            obs_t = env_t.step(6, data={"x": cx, "y": cy})
                            if obs_t.state != GameState.NOT_FINISHED:
                                break

                        if obs_t.levels_completed > levels:
                            all_solutions.append(clicks)
                            levels = obs_t.levels_completed
                            print(
                                f"  L{levels}: color={color} {len(clicks)}/{n} clicks", flush=True
                            )
                            solved = True
                            break
                        if obs_t.state == GameState.GAME_OVER:
                            break
                if solved:
                    break

            if not solved:
                break

        return {
            "levels_completed": levels,
            "total_actions": sum(len(s) for s in all_solutions),
            "method": "click",
        }

    def _solve_keyboard(
        self,
        arcade: Any,
        game_id: str,
        first_actions: list[str] | None = None,
    ) -> dict[str, Any]:
        """Keyboard solver with optional vision-guided first actions."""
        from arcengine.enums import GameState

        env = arcade.make(game_id)
        obs = env.reset()

        actions = [
            a.value if hasattr(a, "value") else int(a)
            for a in (obs.available_actions or [1, 2, 3, 4])
        ]

        name_to_id = {"ACTION1": 1, "ACTION2": 2, "ACTION3": 3, "ACTION4": 4, "ACTION5": 5}

        levels = 0
        total_steps = 0
        prev_grid = np.array(obs.frame)
        if prev_grid.ndim == 3:
            prev_grid = prev_grid[0]

        if first_actions:
            for action_name in first_actions[:20]:
                aid = name_to_id.get(action_name.upper())
                if aid and aid in actions:
                    obs = env.step(aid)
                    total_steps += 1
                    if obs.levels_completed > levels:
                        levels = obs.levels_completed
                    if obs.state == GameState.WIN:
                        return {
                            "levels_completed": levels,
                            "total_actions": total_steps,
                            "method": "keyboard+vision",
                        }
                    if obs.state == GameState.GAME_OVER:
                        obs = env.reset()

        action_reward: dict[int, float] = {a: 1.0 for a in actions}
        for _step in range(self.max_keyboard_steps):
            if random.random() < 0.3:
                action = random.choice(actions)
            else:
                action = max(action_reward, key=lambda a: action_reward[a])

            obs = env.step(action)
            total_steps += 1
            cur = np.array(obs.frame)
            if cur.ndim == 3:
                cur = cur[0]
            changed = int(np.sum(cur != prev_grid))
            prev_grid = cur
            action_reward[action] = action_reward.get(action, 0) * 0.9 + changed * 0.1

            if obs.levels_completed > levels:
                levels = obs.levels_completed
            if obs.state == GameState.WIN:
                break
            if obs.state == GameState.GAME_OVER:
                obs = env.reset()
                prev_grid = np.array(obs.frame)
                if prev_grid.ndim == 3:
                    prev_grid = prev_grid[0]

        return {"levels_completed": levels, "total_actions": total_steps, "method": "keyboard"}


def run_all_games(use_vision: bool = True) -> list[dict]:
    """Run hybrid solver on all games."""
    import arc_agi

    arcade = arc_agi.Arcade()
    envs = arcade.get_environments()

    results = []
    total_levels = 0
    t0 = time.monotonic()

    for e in envs:
        gid = e.game_id
        short = gid.split("-")[0]

        solver = HybridSolver()
        try:
            result = solver.solve_game(gid, use_vision=use_vision)
            total_levels += result["levels_completed"]
            results.append(result)

            status = "WIN" if result["levels_completed"] > 0 else "---"
            vis = "GPU" if result.get("vision_used") else "CPU"
            print(
                f"{short:5s}: {status} levels={result['levels_completed']} "
                f"actions={result['total_actions']} {result['method']} "
                f"[{vis}] {result['elapsed_s']}s",
                flush=True,
            )
        except Exception as exc:
            print(f"{short:5s}: ERROR {str(exc)[:60]}", flush=True)
            results.append({"game_id": short, "levels_completed": 0, "error": str(exc)[:100]})

    elapsed = time.monotonic() - t0
    print(f"\nTOTAL: {total_levels} levels across {len(envs)} games in {elapsed:.0f}s", flush=True)
    return results
