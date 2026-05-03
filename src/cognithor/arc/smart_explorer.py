"""ARC-AGI-3 Smart Explorer — systematic state-action graph exploration.

Inspired by the 3rd-place ARC-AGI-3 Preview solution: tracks which actions
have been tested at each state, navigates via shortest path to states with
untested actions, and prioritizes clicks on small salient objects.

Key difference from DFS: DFS goes deep and backtracks. Smart Explorer goes
WIDE — systematically tests every action at every reachable state, navigating
back to frontier states via known shortest paths.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cognithor.arc.error_handler import safe_frame_extract
from cognithor.utils.logging import get_logger

__all__ = ["SmartExplorer"]

log = get_logger(__name__)


@dataclass
class StateNode:
    """A state in the exploration graph."""

    grid_hash: int
    untested: list[int]  # action indices still to try
    tested: dict[int, int | None] = field(default_factory=dict)  # action → next_state_hash or None
    distance_from_start: int = 0
    path_from_start: list[int] = field(default_factory=list)  # action sequence from initial state


@dataclass
class ExploreResult:
    levels_completed: int = 0
    total_steps: int = 0


class SmartExplorer:
    """Systematic exploration: test every action at every state, navigate to frontiers."""

    def __init__(self, arcade: Any, game_id: str, available_actions: list[int]):
        self._arcade = arcade
        self._game_id = game_id
        self._base_actions = [a for a in available_actions if a != 6]
        self._has_click = 6 in available_actions

    def solve(self, max_levels: int = 10, timeout_s: float = 300.0) -> ExploreResult:
        """Solve game level by level."""

        env = self._arcade.make(self._game_id)
        result = ExploreResult()
        replay_prefix: list[int] = []

        for level in range(max_levels):
            t0 = time.monotonic()
            solution = self._explore_level(env, replay_prefix, timeout_s)

            if solution is None:
                break

            # Verify
            full = replay_prefix + solution
            obs = env.reset()
            for a in full:
                if a >= 100:  # encoded click
                    x, y = a // 1000 - 1, a % 1000
                    obs = env.step(6, data={"x": x, "y": y})
                else:
                    obs = env.step(a)
            if obs.levels_completed <= level:
                break

            replay_prefix.extend(solution)
            result.levels_completed += 1
            result.total_steps += len(solution)
            log.info(
                "arc.explorer_level_solved",
                game_id=self._game_id,
                level=level,
                steps=len(solution),
                time_s=round(time.monotonic() - t0, 1),
            )

        return result

    def _explore_level(
        self,
        env: Any,
        replay_prefix: list[int],
        timeout: float,
    ) -> list[int] | None:
        """Explore one level: build state graph, navigate to frontiers."""
        from arcengine.enums import GameState

        t0 = time.monotonic()
        max_states = 50_000

        # Get initial state
        obs = self._replay(env, replay_prefix)
        initial_grid = safe_frame_extract(obs)
        initial_hash = self._hash(initial_grid)
        current_levels = obs.levels_completed

        # Detect click targets via connected components
        click_actions = self._find_click_targets(initial_grid) if self._has_click else []
        all_actions = list(self._base_actions) + click_actions

        # Build state graph
        states: dict[int, StateNode] = {}
        states[initial_hash] = StateNode(
            grid_hash=initial_hash,
            untested=list(all_actions),
            path_from_start=[],
        )

        current_hash = initial_hash

        while time.monotonic() - t0 < timeout and len(states) < max_states:
            node = states[current_hash]

            # Pick untested action at current state
            if node.untested:
                action = node.untested.pop()

                # Execute
                obs = self._replay(env, replay_prefix + node.path_from_start)
                obs = self._step(env, action)

                # Check win
                if obs.levels_completed > current_levels:
                    solution = [*node.path_from_start, action]
                    log.info(
                        "arc.explorer_solved",
                        states=len(states),
                        time_s=round(time.monotonic() - t0, 1),
                    )
                    return solution

                # Check game over
                if obs.state == GameState.GAME_OVER:
                    node.tested[action] = None
                    continue

                # Get new state
                grid = safe_frame_extract(obs)
                new_hash = self._hash(grid)

                node.tested[action] = new_hash

                if new_hash == current_hash:
                    # Action didn't change state — prune it
                    continue

                if new_hash not in states:
                    # Discover new state
                    new_path = [*node.path_from_start, action]

                    # Detect click targets for new state (may differ)
                    new_clicks = self._find_click_targets(grid) if self._has_click else []
                    new_actions = list(self._base_actions) + new_clicks

                    states[new_hash] = StateNode(
                        grid_hash=new_hash,
                        untested=new_actions,
                        distance_from_start=len(new_path),
                        path_from_start=new_path,
                    )

            else:
                # All actions tested at current state — find nearest frontier
                frontier = self._find_nearest_frontier(states, current_hash)
                if frontier is None:
                    break  # fully explored
                current_hash = frontier

        log.info(
            "arc.explorer_exhausted", states=len(states), time_s=round(time.monotonic() - t0, 1)
        )
        return None

    def _find_nearest_frontier(
        self,
        states: dict[int, StateNode],
        current_hash: int,
    ) -> int | None:
        """Find the nearest state that still has untested actions (BFS on graph)."""
        visited = {current_hash}
        queue = deque([current_hash])

        while queue:
            h = queue.popleft()
            node = states.get(h)
            if node is None:
                continue

            # Check if this state has untested actions
            if h != current_hash and node.untested:
                return h

            # Expand via known transitions
            for _action, next_h in node.tested.items():
                if next_h is not None and next_h not in visited and next_h in states:
                    visited.add(next_h)
                    queue.append(next_h)

        return None  # no frontier found

    def _find_click_targets(self, grid: np.ndarray[Any, Any]) -> list[int]:
        """Find click targets via connected components. Prioritize small, salient objects.

        Returns encoded click actions: x*1000 + y + 1000
        (offset to avoid collision with action IDs).
        """
        from scipy import ndimage  # type: ignore[import-untyped]

        targets: list[tuple[int, int, int]] = []  # (size, x, y)

        # Find connected components per non-background color
        bg_color = int(np.argmax(np.bincount(grid.flatten())))

        for color in range(16):
            if color == bg_color:
                continue
            mask = grid == color
            count = int(np.sum(mask))
            if count < 2 or count > 200:
                continue

            labeled, n_components = ndimage.label(mask)
            for comp_id in range(1, n_components + 1):
                ys, xs = np.where(labeled == comp_id)
                size = len(ys)
                if size < 2 or size > 100:
                    continue
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
                targets.append((size, cx, cy))

        # Sort by size (smallest = most likely interactive)
        targets.sort()

        # Encode as action IDs: (x+1)*1000 + y
        # Take top 8 targets max
        click_actions = []
        seen = set()
        for _size, cx, cy in targets[:12]:
            # Deduplicate nearby positions
            key = (cx // 4, cy // 4)
            if key in seen:
                continue
            seen.add(key)
            click_actions.append((cx + 1) * 1000 + cy)
            if len(click_actions) >= 8:
                break

        return click_actions

    def _step(self, env: Any, action: int) -> Any:
        """Execute an action (handles encoded click actions)."""
        if action >= 100:  # encoded click
            x = action // 1000 - 1
            y = action % 1000
            return env.step(6, data={"x": x, "y": y})
        return env.step(action)

    def _replay(self, env: Any, actions: list[int]) -> Any:
        """Reset and replay action sequence."""
        obs = env.reset()
        for a in actions:
            obs = self._step(env, a)
        return obs

    @staticmethod
    def _hash(grid: np.ndarray[Any, Any]) -> int:
        """Hash grid excluding timer bars."""
        return hash(grid[2:62].tobytes())
