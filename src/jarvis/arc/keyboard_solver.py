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

from jarvis.arc.error_handler import safe_frame_extract
from jarvis.utils.logging import get_logger

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
    ):
        self._arcade = arcade
        self._game_id = game_id
        self._actions = keyboard_actions or [1, 2, 3, 4]

    def solve(
        self,
        max_levels: int = 10,
        timeout_s: float = 300.0,
    ) -> SolveResult:
        """Solve keyboard game level by level with incremental DFS."""
        from arcengine.enums import GameState

        env = self._arcade.make(self._game_id)
        result = SolveResult()
        prev_solutions: list[list[int]] = []

        for level in range(max_levels):
            t0 = time.monotonic()
            replay_prefix = [a for sol in prev_solutions for a in sol]

            solution = self._solve_level(
                env, replay_prefix, timeout_s,
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

        def smart_action_order(last_action: int | None) -> list[int]:
            """Order actions: avoid immediate reversal of last action."""
            if last_action is None:
                return list(self._actions)
            reverse = _UNDO.get(last_action)
            # Put reverse last (popped first = tried last)
            ordered = [a for a in self._actions if a != reverse]
            if reverse in self._actions:
                ordered.insert(0, reverse)  # reverse tried last (.pop() from end)
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
                # Still visited after double-step — try undo or reset
                undo = _UNDO.get(action)
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

            # At each new position, try INTERACT and CLICK
            for try_action in [5, 6, 7]:
                if try_action not in self._actions:
                    continue
                if try_action == 6:
                    # Click at grid center (common target)
                    obs_try = env.step(6, data={"x": 32, "y": 32})
                else:
                    obs_try = env.step(try_action)
                if obs_try.levels_completed > current_levels:
                    path.append(try_action)
                    return path
                # Undo: reset to current path
                self._replay_to(env, replay_prefix, path)

        log.info("arc.keyboard_dfs_exhausted",
                 states=len(visited), path_len=len(path),
                 time_s=round(time.monotonic() - t0, 1))
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
                candidate = solution[:i] + solution[i + 1:]
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
            log.info("arc.keyboard_path_shortened",
                     original=original_len, shortened=len(solution))
        return solution

    @staticmethod
    def _replay_to(env: Any, prefix: list[int], path: list[int]) -> Any:
        """Reset env and replay prefix + path."""
        obs = env.reset()
        for a in prefix:
            obs = env.step(a)
        for a in path:
            obs = env.step(a)
        return obs

    @staticmethod
    def _grid_hash(grid: np.ndarray) -> int:
        """Hash grid rows 2-62, excluding timer bars."""
        return hash(grid[2:62].tobytes())
