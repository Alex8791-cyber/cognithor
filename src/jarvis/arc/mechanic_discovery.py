"""ARC-AGI-3 Mechanic Discovery — auto-detects game mechanics for unknown levels.

Adapted from user-provided module for arc_agi SDK.

Usage:
    from jarvis.arc.mechanic_discovery import MechanicDiscovery
    discovery = MechanicDiscovery(env)
    profile = discovery.analyze_level(obs)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

from jarvis.arc.fast_grid_solver import (
    Cluster,
    detect_toggle_pair,
    find_clusters,
    obs_to_grid,
)
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["LevelReplayer", "MechanicDiscovery", "MechanicProfile", "WinCondition"]


# ---------------------------------------------------------------------------
# Win condition types
# ---------------------------------------------------------------------------


class WinCondition(Enum):
    SOURCE_ELIMINATED = auto()
    """All source_color pixels must be gone."""

    TARGET_FILLED = auto()
    """All clusters must be toggled to target_color."""

    SPECIFIC_SUBSET = auto()
    """Only a specific subset must be clicked."""

    UNKNOWN = auto()


# ---------------------------------------------------------------------------
# Mechanic profile
# ---------------------------------------------------------------------------


@dataclass
class MechanicProfile:
    """Complete profile of detected game mechanics for a level."""

    level: int
    source_color: int | None = None
    target_color: int | None = None
    clusters: list[Cluster] = field(default_factory=list)
    win_condition: WinCondition = WinCondition.UNKNOWN
    background_color: int = 0
    all_colors: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    discovery_time_s: float = 0.0

    @property
    def is_solvable(self) -> bool:
        return (
            self.source_color is not None
            and self.target_color is not None
            and len(self.clusters) > 0
            and self.win_condition != WinCondition.UNKNOWN
        )

    def __str__(self) -> str:
        return (
            f"MechanicProfile(level={self.level}, "
            f"toggle={self.source_color}->{self.target_color}, "
            f"clusters={len(self.clusters)}, "
            f"win={self.win_condition.name}, "
            f"solvable={self.is_solvable})"
        )


# ---------------------------------------------------------------------------
# Discovery engine
# ---------------------------------------------------------------------------


class MechanicDiscovery:
    """Analyzes game mechanics of an unknown level via test clicks.

    Performs minimal test clicks (max 2 per candidate, always undo)
    to detect toggle pairs and win conditions.
    """

    def __init__(
        self,
        env: Any,
        verbose: bool = True,
        max_test_clicks: int = 10,
    ) -> None:
        self.env = env
        self.verbose = verbose
        self.max_test_clicks = max_test_clicks
        self._test_clicks_used = 0

    def analyze_level(self, obs: Any, level: int = 0) -> MechanicProfile:
        """Fully analyze the current level."""
        t0 = time.time()
        self._test_clicks_used = 0
        grid = obs_to_grid(obs)
        profile = MechanicProfile(level=level)

        # 1. Color inventory
        profile = self._analyze_colors(grid, profile)
        self._log(f"Colors: {profile.all_colors}, bg: {profile.background_color}")

        # 2. Detect toggle pair
        profile = self._discover_toggle_pair(grid, profile)

        if profile.source_color is None:
            profile.notes.append("No toggle pair found")
            self._log("No toggle detected")
        else:
            self._log(f"Toggle: {profile.source_color} -> {profile.target_color}")

            # 3. Find clusters
            profile.clusters = find_clusters(grid, profile.source_color)
            self._log(f"Clusters: {len(profile.clusters)}")

            # 4. Infer win condition
            profile = self._infer_win_condition(grid, profile)
            self._log(f"Win condition: {profile.win_condition.name}")

        profile.discovery_time_s = time.time() - t0
        return profile

    def _analyze_colors(self, grid: np.ndarray, profile: MechanicProfile) -> MechanicProfile:
        colors, counts = np.unique(grid, return_counts=True)
        profile.all_colors = [int(c) for c in colors]
        profile.background_color = int(colors[np.argmax(counts)])
        return profile

    def _discover_toggle_pair(self, grid: np.ndarray, profile: MechanicProfile) -> MechanicProfile:
        """Try clicking each non-bg color's largest cluster to find toggle."""
        colors, counts = np.unique(grid, return_counts=True)
        bg = profile.background_color
        candidates = [(int(c), int(n)) for c, n in zip(colors, counts, strict=False) if c != bg]
        candidates.sort(key=lambda x: -x[1])

        for color, pixel_count in candidates:
            if self._test_clicks_used >= self.max_test_clicks:
                profile.notes.append("Max test clicks reached")
                break

            clusters = find_clusters(grid, color)
            if not clusters:
                continue

            test_cluster = max(clusters, key=lambda c: c.size)
            cr, cc = test_cluster.centroid

            self._log(f"  Testing color {color} ({pixel_count}px) at ({cc},{cr})")

            # Get current grid
            grid_before = obs_to_grid(self._get_obs())

            # Test click (SDK: x=col, y=row)
            self._click(cc, cr)
            self._test_clicks_used += 1
            grid_after = obs_to_grid(self._get_obs())

            toggle = detect_toggle_pair(grid_before, grid_after)

            # Undo
            self._click(cc, cr)
            self._test_clicks_used += 1

            if toggle is not None:
                profile.source_color, profile.target_color = toggle
                profile.notes.append(f"Toggle found: {toggle[0]}->{toggle[1]}")
                return profile

        return profile

    def _infer_win_condition(self, grid: np.ndarray, profile: MechanicProfile) -> MechanicProfile:
        if profile.source_color is None:
            return profile

        total_non_bg = int(np.sum(grid != profile.background_color))
        source_pixels = int(np.sum(grid == profile.source_color))

        if total_non_bg == 0:
            profile.win_condition = WinCondition.UNKNOWN
            return profile

        source_ratio = source_pixels / total_non_bg

        if source_ratio > 0.4:
            profile.win_condition = WinCondition.SOURCE_ELIMINATED
        elif source_ratio > 0.15:
            profile.win_condition = WinCondition.SPECIFIC_SUBSET
        else:
            profile.win_condition = WinCondition.SOURCE_ELIMINATED

        profile.notes.append(f"source_ratio={source_ratio:.0%}")
        return profile

    # ------------------------------------------------------------------
    # Env helpers (arc_agi SDK)
    # ------------------------------------------------------------------

    def _click(self, x: int, y: int) -> Any:
        """Click at (x, y). arc_agi SDK: env.step(6, data={x, y})."""
        return self.env.step(6, data={"x": int(x), "y": int(y)})

    def _get_obs(self) -> Any:
        """Get current observation without taking a step."""
        if hasattr(self.env, "_last_response"):
            return self.env._last_response
        return None

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[Discovery] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Level replayer
# ---------------------------------------------------------------------------


class LevelReplayer:
    """Replays known solutions to reach a target level."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self._solutions: dict[int, list[tuple[int, int]]] = {}

    def add_solution(self, level: int, clicks: list[tuple[int, int]]) -> None:
        """Add a known solution. Clicks are (x, y) in SDK format."""
        self._solutions[level] = clicks

    def replay_to(self, target_level: int) -> Any:
        """Reset env and replay all levels up to target_level."""
        obs = self.env.reset()
        for level in range(target_level):
            if level not in self._solutions:
                msg = f"No solution for level {level}. Available: {sorted(self._solutions.keys())}"
                raise ValueError(msg)
            for x, y in self._solutions[level]:
                obs = self.env.step(6, data={"x": x, "y": y})
        return obs
