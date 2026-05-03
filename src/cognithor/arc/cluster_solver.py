"""ARC-AGI-3 Cluster Solver — solves click-based color-toggle games.

Strategy: Find clusters of a target color, brute-force which subset
to click to complete each level. Works for games where clicking toggles
cell colors and the goal is to reach the right configuration.

This solved ft09 Level 1 (17 clicks) and Level 2 (9 clicks).
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
from scipy import ndimage  # type: ignore[import-untyped]

from cognithor.utils.logging import get_logger

__all__ = ["ClusterSolver"]

log = get_logger(__name__)


class ClusterSolver:
    """Solves click-based ARC games by finding the right cluster subset."""

    def __init__(self, target_color: int = 9, max_skip: int = 6) -> None:
        self.target_color = target_color
        self.max_skip = max_skip

    def find_clusters(self, grid: np.ndarray) -> list[tuple[int, int]]:
        """Find centers of connected components of target_color."""
        if grid.ndim == 3:
            grid = grid[0]

        mask = grid == self.target_color
        labeled, n = ndimage.label(mask)

        centers = []
        for i in range(1, n + 1):
            ys, xs = np.where(labeled == i)
            centers.append((int(np.mean(xs)), int(np.mean(ys))))

        return centers

    def find_solution(
        self,
        env_factory: Any,
        action_id: int,
        prev_solutions: list[list[tuple[int, int]]],
        target_level: int,
    ) -> list[tuple[int, int]] | None:
        """Find the click subset that completes the next level.

        Args:
            env_factory: Callable that creates a fresh environment.
            action_id: The click action ID (e.g. 6).
            prev_solutions: Solutions for previous levels (for replay).
            target_level: The level we're trying to solve (0-indexed).

        Returns:
            List of (x, y) click positions, or None if not found.
        """
        from arcengine.enums import GameState

        # Replay to current level
        env = env_factory()
        obs = env.reset()
        for sol in prev_solutions:
            for cx, cy in sol:
                obs = env.step(action_id, data={"x": cx, "y": cy})

        grid = np.array(obs.frame)
        centers = self.find_clusters(grid)
        n = len(centers)

        if n == 0:
            return None

        current_levels = obs.levels_completed

        for skip in range(min(n + 1, self.max_skip + 1)):
            for skip_combo in itertools.combinations(range(n), skip):
                click_idx = [i for i in range(n) if i not in skip_combo]

                env2 = env_factory()
                obs2 = env2.reset()

                # Replay previous levels
                ok = True
                for sol in prev_solutions:
                    for cx, cy in sol:
                        obs2 = env2.step(action_id, data={"x": cx, "y": cy})
                        if obs2.state == GameState.GAME_OVER:
                            ok = False
                            break
                    if not ok:
                        break

                if not ok:
                    continue

                # Test this combination
                for idx in click_idx:
                    cx, cy = centers[idx]
                    obs2 = env2.step(action_id, data={"x": cx, "y": cy})
                    if obs2.state != GameState.NOT_FINISHED:
                        break

                if obs2.levels_completed > current_levels:
                    solution = [centers[i] for i in click_idx]
                    log.info(
                        "arc_cluster_solved",
                        level=target_level,
                        clicks=len(solution),
                        skipped=skip,
                        clusters=n,
                    )
                    return solution

        return None
