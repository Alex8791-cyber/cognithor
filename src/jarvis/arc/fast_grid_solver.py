"""ARC-AGI-3 Fast Grid Solver — simulates toggles in NumPy, zero SDK calls per combo.

Adapted from user-provided module for arc_agi SDK (not arcadegym).

Usage:
    from jarvis.arc.fast_grid_solver import FastGridSolver
    solver = FastGridSolver(game_id="ft09", verbose=True)
    result = solver.solve_all_levels()
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from jarvis.utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["Cluster", "FastGridSolver", "LevelResult", "find_clusters"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Cluster:
    """A connected pixel block of the same color."""

    color: int
    pixels: list[tuple[int, int]]  # (row, col)

    @property
    def centroid(self) -> tuple[int, int]:
        rows = [p[0] for p in self.pixels]
        cols = [p[1] for p in self.pixels]
        return int(np.mean(rows)), int(np.mean(cols))

    @property
    def size(self) -> int:
        return len(self.pixels)


@dataclass
class LevelResult:
    level: int
    solved: bool
    clicks: list[tuple[int, int]] = field(default_factory=list)
    duration_s: float = 0.0
    combos_tested: int = 0
    notes: str = ""


# ---------------------------------------------------------------------------
# Grid analysis
# ---------------------------------------------------------------------------


def obs_to_grid(obs: Any) -> np.ndarray:
    """Extract color grid from an ARC-AGI observation (FrameDataRaw)."""
    # arc_agi SDK: obs.frame is ndarray shape (1, 64, 64)
    if hasattr(obs, "frame"):
        grid = np.array(obs.frame)
    elif isinstance(obs, np.ndarray):
        grid = obs
    else:
        grid = np.array(obs, dtype=np.int32)

    if grid.ndim == 3:
        grid = grid[0]
    return grid.astype(np.int32)


def find_clusters(grid: np.ndarray, target_color: int) -> list[Cluster]:
    """Find connected regions of target_color (4-connectivity BFS)."""
    rows, cols = np.where(grid == target_color)
    if len(rows) == 0:
        return []

    pixel_set = set(zip(rows.tolist(), cols.tolist(), strict=False))
    visited: set[tuple[int, int]] = set()
    clusters: list[Cluster] = []

    for start in pixel_set:
        if start in visited:
            continue
        component: list[tuple[int, int]] = []
        queue = [start]
        while queue:
            px = queue.pop()
            if px in visited or px not in pixel_set:
                continue
            visited.add(px)
            component.append(px)
            r, c = px
            queue.extend([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)])
        clusters.append(Cluster(color=target_color, pixels=component))

    return clusters


def detect_toggle_pair(
    grid_before: np.ndarray,
    grid_after: np.ndarray,
) -> tuple[int, int] | None:
    """Detect which two colors swap when clicking."""
    diff_mask = grid_before != grid_after
    if not diff_mask.any():
        return None

    before_colors = grid_before[diff_mask]
    after_colors = grid_after[diff_mask]

    pairs: dict[tuple[int, int], int] = {}
    for b, a in zip(before_colors.tolist(), after_colors.tolist(), strict=False):
        key = (int(b), int(a))
        pairs[key] = pairs.get(key, 0) + 1

    if not pairs:
        return None

    return max(pairs, key=lambda k: pairs[k])


# ---------------------------------------------------------------------------
# NumPy simulation (the fast core)
# ---------------------------------------------------------------------------


def simulate_toggle(
    grid: np.ndarray,
    cluster: Cluster,
    source_color: int,
    target_color: int,
) -> np.ndarray:
    """Apply a single toggle to the grid."""
    result = grid.copy()
    for r, c in cluster.pixels:
        if result[r, c] == source_color:
            result[r, c] = target_color
        elif result[r, c] == target_color:
            result[r, c] = source_color
    return result


def simulate_combo(
    grid: np.ndarray,
    clusters: list[Cluster],
    indices: tuple[int, ...],
    source_color: int,
    target_color: int,
) -> np.ndarray:
    """Simulate multiple toggles in sequence."""
    result = grid.copy()
    for idx in indices:
        result = simulate_toggle(result, clusters[idx], source_color, target_color)
    return result


def is_level_complete(grid_after: np.ndarray, source_color: int) -> bool:
    """Check if all source_color pixels have been eliminated."""
    return not (grid_after == source_color).any()


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------


def _nCr(n: int, r: int) -> int:
    if r > n:
        return 0
    if r == 0 or r == n:
        return 1
    r = min(r, n - r)
    result = 1
    for i in range(r):
        result = result * (n - i) // (i + 1)
    return result


class FastGridSolver:
    """Solves ARC-AGI-3 click-toggle games without arcade.make() per combo.

    Flow per level:
      1. Observe grid
      2. Auto-detect toggle pair (which color toggles to which)
      3. Find clusters of source_color
      4. Brute-force all C(n,k) subsets in pure NumPy
      5. Execute winning solution on real env
    """

    def __init__(
        self,
        game_id: str = "ft09",
        max_levels: int = 10,
        verbose: bool = True,
        max_combos_per_level: int = 100_000,
    ) -> None:
        self.game_id = game_id
        self.max_levels = max_levels
        self.verbose = verbose
        self.max_combos_per_level = max_combos_per_level
        self._env: Any = None
        self._arcade: Any = None

    def solve_all_levels(self) -> list[LevelResult]:
        """Solve all levels sequentially."""
        import arc_agi

        self._arcade = arc_agi.Arcade()
        self._env = self._arcade.make(self.game_id)
        obs = self._env.reset()
        results: list[LevelResult] = []

        for level_num in range(self.max_levels):
            self._log(f"\n=== Level {level_num} ===")
            result = self._solve_level(level_num, obs)
            results.append(result)

            if result.solved:
                self._log(
                    f"  SOLVED: {len(result.clicks)} clicks "
                    f"in {result.duration_s:.1f}s ({result.combos_tested} combos)"
                )
                # Get obs for next level
                obs = self._last_obs
                if self._is_game_done():
                    self._log("GAME COMPLETE!")
                    break
            else:
                self._log(f"  FAILED: {result.notes}")
                break

        return results

    def _solve_level(self, level_num: int, obs: Any) -> LevelResult:
        t0 = time.time()
        grid = obs_to_grid(obs)
        self._log(f"Grid: {grid.shape}, colors: {np.unique(grid).tolist()}")

        # 1. Auto-detect toggle pair
        toggle = self._detect_toggle_auto(grid)
        if toggle is None:
            return LevelResult(
                level=level_num,
                solved=False,
                duration_s=time.time() - t0,
                notes="No toggle pair detected",
            )

        source_color, target_color = toggle
        self._log(f"Toggle: {source_color} -> {target_color}")

        # 2. Find clusters
        clusters = find_clusters(grid, source_color)
        self._log(f"Clusters: {len(clusters)} (color={source_color})")

        if not clusters:
            return LevelResult(
                level=level_num,
                solved=False,
                duration_s=time.time() - t0,
                notes=f"No clusters of color {source_color}",
            )

        # 3. Fast subset search in NumPy
        solution, combos_tested = self._fast_subset_search(
            grid, clusters, source_color, target_color
        )

        if solution is None:
            return LevelResult(
                level=level_num,
                solved=False,
                duration_s=time.time() - t0,
                combos_tested=combos_tested,
                notes="No solution found in subset search",
            )

        # 4. Execute on real env
        click_coords = [clusters[i].centroid for i in solution]
        # Centroid returns (row, col) but SDK wants (x=col, y=row)
        self._execute_solution(click_coords)

        return LevelResult(
            level=level_num,
            solved=True,
            clicks=click_coords,
            duration_s=time.time() - t0,
            combos_tested=combos_tested,
        )

    def _fast_subset_search(
        self,
        grid: np.ndarray,
        clusters: list[Cluster],
        source_color: int,
        target_color: int,
    ) -> tuple[tuple[int, ...] | None, int]:
        """Find smallest subset that solves the level. Pure NumPy."""
        n = len(clusters)
        combos_tested = 0

        for k in range(1, n + 1):
            num_combos = _nCr(n, k)
            self._log(f"  k={k}: {num_combos} combinations...")

            for combo in itertools.combinations(range(n), k):
                if combos_tested >= self.max_combos_per_level:
                    self._log(f"  Combo limit reached ({self.max_combos_per_level})")
                    return None, combos_tested

                simulated = simulate_combo(grid, clusters, combo, source_color, target_color)
                combos_tested += 1

                if is_level_complete(simulated, source_color):
                    self._log(f"  FOUND at k={k}, combo={combo} after {combos_tested} tests")
                    return combo, combos_tested

        return None, combos_tested

    def _detect_toggle_auto(self, grid: np.ndarray) -> tuple[int, int] | None:
        """Do a test click on the largest non-bg cluster, observe the toggle."""
        colors, counts = np.unique(grid, return_counts=True)
        bg_color = int(colors[np.argmax(counts)])

        candidates = [
            (int(c), int(n))
            for c, n in zip(colors, counts, strict=False)
            if c != bg_color
        ]
        candidates.sort(key=lambda x: -x[1])

        for color_candidate, _ in candidates[:5]:
            clusters = find_clusters(grid, color_candidate)
            if not clusters:
                continue

            test_cluster = max(clusters, key=lambda c: c.size)
            cr, cc = test_cluster.centroid

            grid_before = obs_to_grid(self._last_obs)

            # Test click (SDK: action=6, data={x=col, y=row})
            self._click(cc, cr)
            grid_after = obs_to_grid(self._last_obs)

            toggle = detect_toggle_pair(grid_before, grid_after)

            # Undo click
            self._click(cc, cr)

            if toggle is not None:
                return toggle

        return None

    # ------------------------------------------------------------------
    # Env helpers (adapted for arc_agi SDK)
    # ------------------------------------------------------------------

    @property
    def _last_obs(self) -> Any:
        """Current observation from env."""
        if hasattr(self._env, "_last_response"):
            return self._env._last_response
        return None

    def _click(self, x: int, y: int) -> Any:
        """Click at (x, y) on the real env. arc_agi SDK format."""
        if self._env is None:
            return None
        return self._env.step(6, data={"x": int(x), "y": int(y)})

    def _execute_solution(self, click_coords: list[tuple[int, int]]) -> None:
        """Execute clicks. Coords are (row, col) from centroid, convert to (x=col, y=row)."""
        for row, col in click_coords:
            self._click(x=col, y=row)

    def _is_game_done(self) -> bool:
        obs = self._last_obs
        if obs is None:
            return False
        from arcengine.enums import GameState

        return obs.state == GameState.WIN

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)
