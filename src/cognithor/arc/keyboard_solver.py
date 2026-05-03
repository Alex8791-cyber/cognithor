"""ARC-AGI-3 KeyboardSolver — incremental DFS for keyboard-based games.

Instead of resetting for every search node (BFS pattern), steps forward
with env.step() and only resets when backtracking. ~50x faster than
replay-based BFS for deep mazes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from cognithor.arc.error_handler import safe_frame_extract
from cognithor.utils.logging import get_logger

__all__ = ["KeyboardSolver"]

log = get_logger(__name__)

# Undo map: reverse direction for backtracking without reset
_UNDO = {1: 2, 2: 1, 3: 4, 4: 3}  # UP↔DOWN, LEFT↔RIGHT


@dataclass
class SolveResult:
    """Result from KeyboardSolver."""

    levels_completed: int = 0
    total_steps: int = 0
    score: float = 0.0


class KeyboardSolver:
    """Incremental DFS solver for keyboard-based ARC-AGI-3 games."""

    def __init__(
        self,
        arcade: Any,
        game_id: str,
        keyboard_actions: list[int] | None = None,
        click_positions: list[tuple[int, int]] | None = None,
    ):
        self._arcade = arcade
        self._game_id = game_id
        self._actions = keyboard_actions or [1, 2, 3, 4]
        self._click_positions = click_positions or []

    def solve(
        self,
        max_levels: int = 10,
        timeout_s: float = 300.0,
    ) -> SolveResult:
        """Solve keyboard game level by level with incremental DFS."""

        env = self._arcade.make(self._game_id)
        result = SolveResult()
        prev_solutions: list[list[int]] = []

        for level in range(max_levels):
            t0 = time.monotonic()
            replay_prefix = [a for sol in prev_solutions for a in sol]

            solution = self._solve_level(
                env,
                replay_prefix,
                timeout_s,
            )

            if solution is None:
                break

            # Verify solution actually advances levels
            full_seq = replay_prefix + solution
            obs = env.reset()
            for a in full_seq:
                obs = env.step(a)
            if obs.levels_completed <= level:
                log.info("arc.keyboard_false_positive", level=level)
                break

            # Shorten path: remove redundant steps
            solution = self._shorten_path(env, replay_prefix, solution, level)

            prev_solutions.append(solution)
            result.levels_completed += 1
            result.total_steps += len(solution)
            log.info(
                "arc.keyboard_level_solved",
                game_id=self._game_id,
                level=level,
                steps=len(solution),
                time_s=round(time.monotonic() - t0, 1),
            )

        result.score = float(result.levels_completed)
        return result

    def _solve_level(
        self,
        env: Any,
        replay_prefix: list[int],
        timeout: float,
    ) -> list[int] | None:
        """Try A* pathfinding first, fall back to incremental DFS."""
        # Try A* direct execution
        astar_result = self._astar_solve(env, replay_prefix, min(timeout, 30.0))
        if astar_result is not None:
            return astar_result

        # Fall back to DFS (A* path used as heuristic for action ordering)
        return self._dfs_solve(env, replay_prefix, timeout)

    def _astar_solve(
        self,
        env: Any,
        replay_prefix: list[int],
        timeout: float,
    ) -> list[int] | None:
        """A* pathfinding: detect avatar+goal, find shortest path on grid."""
        import heapq
        import time

        from arcengine.enums import GameState

        t0 = time.monotonic()
        kb_actions = [a for a in self._actions if a in (1, 2, 3, 4)]
        if len(kb_actions) < 2:
            return None

        # Get initial grid and detect avatar via movement
        obs = env.reset()
        for a in replay_prefix:
            obs = env.step(a)
        grid = safe_frame_extract(obs)
        current_levels = obs.levels_completed

        # Detect avatar: pixels that change when we move
        avatar_pixels: set[tuple[int, int]] = set()
        for action in kb_actions[:2]:  # test 2 directions
            obs2 = env.reset()
            for a in replay_prefix:
                obs2 = env.step(a)
            g_before = safe_frame_extract(obs2)
            obs2 = env.step(action)
            g_after = safe_frame_extract(obs2)
            diff = g_before != g_after
            ys, xs = np.where(diff)
            for y, x in zip(ys.tolist(), xs.tolist(), strict=False):
                avatar_pixels.add((x, y))

        if not avatar_pixels:
            return None

        avatar_x = sum(p[0] for p in avatar_pixels) // len(avatar_pixels)
        avatar_y = sum(p[1] for p in avatar_pixels) // len(avatar_pixels)

        # Detect goal: small colored object far from avatar
        best_goal = None
        best_dist = 0
        for c in range(16):
            ys_c, xs_c = np.where(grid == c)
            if 2 <= len(ys_c) <= 50:
                cx, cy = int(np.mean(xs_c)), int(np.mean(ys_c))
                dist = abs(cx - avatar_x) + abs(cy - avatar_y)
                if dist > best_dist:
                    best_dist = dist
                    best_goal = (cx, cy)

        if best_goal is None or best_dist < 5:
            return None

        log.info("arc.astar_start", avatar=(avatar_x, avatar_y), goal=best_goal, dist=best_dist)

        # Build walkable map: downsample to blocks
        block_size = 4
        bh, bw = 64 // block_size, 64 // block_size
        block_grid = np.zeros((bh, bw), dtype=int)
        for by in range(bh):
            for bx in range(bw):
                block = grid[
                    by * block_size : (by + 1) * block_size, bx * block_size : (bx + 1) * block_size
                ]
                colors, counts = np.unique(block, return_counts=True)
                block_grid[by, bx] = colors[np.argmax(counts)]

        # Determine walkable colors (not walls)
        block_grid[0, 0] if block_grid[0, 0] != 0 else 1
        walkable = set(int(c) for c in np.unique(block_grid))

        start_block = (avatar_x // block_size, avatar_y // block_size)
        goal_block = (best_goal[0] // block_size, best_goal[1] // block_size)

        # A* on block grid
        DELTAS = [(0, -1, 1), (0, 1, 2), (-1, 0, 3), (1, 0, 4)]  # UP,DOWN,LEFT,RIGHT
        open_set: list[tuple[int, tuple[int, int], list[int]]] = [(0, start_block, [])]
        closed: set[tuple[int, int]] = set()

        while open_set:
            if time.monotonic() - t0 > timeout:
                break
            cost, pos, path = heapq.heappop(open_set)
            if pos == goal_block:
                # Found path — try executing with different step multipliers
                for mult in [1, 2, 3, 4, 5, 6]:
                    obs = env.reset()
                    for a in replay_prefix:
                        obs = env.step(a)
                    for a in path:
                        for _ in range(mult):
                            obs = env.step(a)
                            if obs.levels_completed > current_levels:
                                log.info(
                                    "arc.astar_solved",
                                    steps=len(path) * mult,
                                    mult=mult,
                                    blocks=len(path),
                                )
                                return [a for a in path for _ in range(mult)]
                            if obs.state == GameState.GAME_OVER:
                                break
                        if obs.state == GameState.GAME_OVER:
                            break
                log.info("arc.astar_path_found_but_no_win", blocks=len(path))
                return None

            if pos in closed:
                continue
            closed.add(pos)

            bx, by = pos
            for dx, dy, action in DELTAS:
                if action not in kb_actions:
                    continue
                nx, ny = bx + dx, by + dy
                if (
                    0 <= nx < bw
                    and 0 <= ny < bh
                    and (nx, ny) not in closed
                    and block_grid[ny, nx] in walkable
                ):
                    dist = abs(nx - goal_block[0]) + abs(ny - goal_block[1])
                    heapq.heappush(open_set, (len(path) + 1 + dist, (nx, ny), [*path, action]))

        return None

    def _dfs_solve(
        self,
        env: Any,
        replay_prefix: list[int],
        timeout: float,
    ) -> list[int] | None:
        """Incremental DFS for one level. Returns action sequence or None."""
        from arcengine.enums import GameState

        t0 = time.monotonic()
        max_states = 50_000
        max_depth = 500

        # Reset and replay to level start
        obs = env.reset()
        for a in replay_prefix:
            obs = env.step(a)

        initial_grid = safe_frame_extract(obs)
        current_levels = obs.levels_completed

        path: list[int] = []
        visited: set[int] = {self._grid_hash(initial_grid)}

        # Encode click positions as negative indices: -1 = click[0], -2 = click[1], etc.
        click_actions = [-(i + 1) for i in range(len(self._click_positions))]
        all_actions = list(self._actions) + click_actions

        def smart_action_order(last_action: int | None) -> list[int]:
            """Order actions: avoid immediate reversal of last action."""
            if last_action is None:
                return list(all_actions)
            reverse = _UNDO.get(last_action)
            ordered = [a for a in all_actions if a != reverse]
            if reverse is not None and reverse in all_actions:
                ordered.insert(0, reverse)
            return ordered

        # Stack of remaining actions to try at each depth
        stack: list[list[int]] = [smart_action_order(None)]

        while stack:
            if time.monotonic() - t0 > timeout:
                break
            if len(visited) > max_states:
                break

            remaining = stack[-1]

            if not remaining:
                # All directions tried at this depth — backtrack
                stack.pop()
                if path:
                    path.pop()
                obs = self._replay_to(env, replay_prefix, path)
                continue

            action = remaining.pop()

            # Depth limit
            if len(path) >= max_depth:
                continue

            # INCREMENTAL step — no reset needed!
            if action < 0:
                # Click at a predefined position
                idx = -(action + 1)
                cx, cy = self._click_positions[idx]
                obs = env.step(6, data={"x": cx, "y": cy})
            else:
                obs = env.step(action)
            actions_taken = [action]

            # Check win
            if obs.levels_completed > current_levels:
                path.extend(actions_taken)
                return path

            # Check game over
            if obs.state == GameState.GAME_OVER:
                self._replay_to(env, replay_prefix, path)
                continue

            grid = safe_frame_extract(obs)
            h = self._grid_hash(grid)

            # Delayed-render fix: some games need 2 steps for grid to update.
            # If this step produced no visible change, repeat the action once.
            if h in visited:
                if action < 0:
                    idx = -(action + 1)
                    cx, cy = self._click_positions[idx]
                    obs = env.step(6, data={"x": cx, "y": cy})
                else:
                    obs = env.step(action)
                actions_taken.append(action)

                if obs.levels_completed > current_levels:
                    path.extend(actions_taken)
                    return path
                if obs.state == GameState.GAME_OVER:
                    self._replay_to(env, replay_prefix, path)
                    continue

                grid = safe_frame_extract(obs)
                h = self._grid_hash(grid)

            if h in visited:
                # Still visited — try undo (keyboard only) or reset
                undo = _UNDO.get(action) if action > 0 else None
                if undo is not None and len(actions_taken) <= 2:
                    for _ in actions_taken:
                        obs = env.step(undo)
                else:
                    self._replay_to(env, replay_prefix, path)
                continue

            # New state — go deeper
            visited.add(h)
            path.extend(actions_taken)
            stack.append(smart_action_order(actions_taken[0]))

            # Clicks and INTERACT are now regular DFS actions (via negative indices)
            # No separate try-click block needed

        log.info(
            "arc.keyboard_dfs_exhausted",
            states=len(visited),
            path_len=len(path),
            time_s=round(time.monotonic() - t0, 1),
        )
        return None

    def _shorten_path(
        self,
        env: Any,
        replay_prefix: list[int],
        solution: list[int],
        target_level: int,
    ) -> list[int]:
        """Remove redundant steps from a DFS solution.

        Iteratively tries removing each step. If the solution still works
        without it, keep the shorter version. O(n^2) but n is small after
        a few passes.
        """
        from arcengine.enums import GameState

        original_len = len(solution)
        improved = True

        while improved:
            improved = False
            i = 0
            while i < len(solution):
                # Try without step i
                candidate = solution[:i] + solution[i + 1 :]
                full_seq = replay_prefix + candidate

                obs = env.reset()
                ok = True
                for a in full_seq:
                    obs = env.step(a)
                    if obs.state == GameState.GAME_OVER:
                        ok = False
                        break

                if ok and obs.levels_completed > target_level:
                    solution = candidate
                    improved = True
                    # Don't increment i — check same index again
                else:
                    i += 1

        if len(solution) < original_len:
            log.info("arc.keyboard_path_shortened", original=original_len, shortened=len(solution))
        return solution

    def _replay_to(self, env: Any, prefix: list[int], path: list[int]) -> Any:
        """Reset env and replay prefix + path. Handles click actions (negative indices)."""
        obs = env.reset()
        for a in prefix:
            obs = env.step(a)
        for a in path:
            if a < 0 and self._click_positions:
                idx = -(a + 1)
                cx, cy = self._click_positions[idx]
                obs = env.step(6, data={"x": cx, "y": cy})
            else:
                obs = env.step(a)
        return obs

    @staticmethod
    def _grid_hash(grid: np.ndarray[Any, Any]) -> int:
        """Hash grid rows 2-62, excluding timer bars."""
        return hash(grid[2:62].tobytes())
