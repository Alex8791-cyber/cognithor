# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-12 — pure-NumPy fast-path planner for click-toggle games.

A planning-only adaptation of :mod:`cognithor.arc.fast_grid_solver`.
The legacy module drove the ``arc_agi`` SDK directly (calling
``env.step()`` between cluster-detection probes); for the new
Sprint-11 game-agent stack we keep only the env-independent core:

* :class:`Cluster` — connected pixel block with centroid/size
* :func:`find_clusters` — 4-connectivity BFS
* :func:`detect_toggle_pair` — pure-NumPy diff between before/after grids
* :func:`simulate_toggle` / :func:`simulate_combo` — apply N toggles to a grid
* :func:`is_level_complete` — "all source-color pixels eliminated"
* :func:`plan_click_solution` — top-level: smallest subset of clusters that
  clears the source colour. Returns the cluster indices to click, or
  ``None`` when no solution found within ``max_combos``.

The agent (e.g. :class:`Sprint10DSLAgent`) calls
:func:`plan_click_solution` as a fast-path before falling through to
the slower DSL search. Cluster index → SDK click coordinates is the
agent's job (translate ``clusters[i].centroid`` to ACTION6 ``(x,y)``).

This module is the cousin of :class:`NumpySolverBridge` (static
ARC-AGI-1 grids) but for **interactive** click-toggle games. There is
no overlap at the call-site.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "Cluster",
    "detect_toggle_pair",
    "find_clusters",
    "is_level_complete",
    "plan_click_solution",
    "simulate_combo",
    "simulate_toggle",
]


@dataclass(frozen=True)
class Cluster:
    """A connected pixel block of the same color."""

    color: int
    pixels: tuple[tuple[int, int], ...]

    @property
    def centroid(self) -> tuple[int, int]:
        rows = [p[0] for p in self.pixels]
        cols = [p[1] for p in self.pixels]
        return int(np.mean(rows)), int(np.mean(cols))

    @property
    def size(self) -> int:
        return len(self.pixels)


def find_clusters(grid: np.ndarray[Any, Any], target_color: int) -> list[Cluster]:
    """Return all 4-connected components of ``target_color`` in ``grid``."""
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
        clusters.append(Cluster(color=target_color, pixels=tuple(component)))

    return clusters


def detect_toggle_pair(
    grid_before: np.ndarray[Any, Any],
    grid_after: np.ndarray[Any, Any],
) -> tuple[int, int] | None:
    """Return ``(source_color, target_color)`` of the dominant swap, or ``None``."""
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


def simulate_toggle(
    grid: np.ndarray[Any, Any],
    cluster: Cluster,
    source_color: int,
    target_color: int,
) -> np.ndarray[Any, Any]:
    """Apply a single toggle to ``grid`` (out-of-place)."""
    result = grid.copy()
    for r, c in cluster.pixels:
        if result[r, c] == source_color:
            result[r, c] = target_color
        elif result[r, c] == target_color:
            result[r, c] = source_color
    return result


def simulate_combo(
    grid: np.ndarray[Any, Any],
    clusters: list[Cluster],
    indices: tuple[int, ...],
    source_color: int,
    target_color: int,
) -> np.ndarray[Any, Any]:
    """Apply toggles for each ``clusters[i]`` (i in ``indices``) sequentially."""
    result = grid.copy()
    for idx in indices:
        result = simulate_toggle(result, clusters[idx], source_color, target_color)
    return result


def is_level_complete(grid_after: np.ndarray[Any, Any], source_color: int) -> bool:
    """Return True iff all ``source_color`` pixels are gone from ``grid_after``."""
    return not (grid_after == source_color).any()


def plan_click_solution(
    grid: np.ndarray[Any, Any],
    source_color: int,
    target_color: int,
    *,
    max_combos: int = 100_000,
) -> tuple[int, ...] | None:
    """Find the smallest subset of ``source_color`` clusters that clears the level.

    Returns a tuple of cluster indices (into ``find_clusters(grid, source_color)``)
    that, when toggled, eliminate every ``source_color`` pixel. Returns ``None``
    when no solution exists within ``max_combos`` simulations.

    The caller is responsible for translating cluster indices to SDK click
    actions (``ACTION6`` with ``(x=col, y=row)`` from each cluster's centroid).
    """
    clusters = find_clusters(grid, source_color)
    if not clusters:
        return None

    n = len(clusters)
    combos_tested = 0

    for k in range(1, n + 1):
        for combo in itertools.combinations(range(n), k):
            if combos_tested >= max_combos:
                return None
            simulated = simulate_combo(grid, clusters, combo, source_color, target_color)
            combos_tested += 1
            if is_level_complete(simulated, source_color):
                return combo

    return None
