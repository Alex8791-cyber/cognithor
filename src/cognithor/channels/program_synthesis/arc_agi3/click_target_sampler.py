# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-12 — salience-based click target sampler.

For ARC-AGI-3 games where the goal is "click the target object" (rather
than the LS20-style "click cluster to toggle colour"), the
:func:`plan_click_solution` fast-path doesn't apply: there's no toggle
pair to invert. Instead the agent has to *guess* a click coordinate
each frame and observe the response.

Picking coordinates by uniform random sampling wastes the frame budget
on the background. The :class:`ClickTargetSampler` instead picks
coordinates ranked by **salience**:

* small connected components of non-background colours (likely "target"
  objects rather than terrain)
* unvisited coordinates (we don't re-click a position we already tried
  in this episode)
* coloured pixels over background (background is always the most
  common colour; we never click pure background)

The sampler maintains a per-grid plan that's recomputed when the grid
changes meaningfully (different colour distribution). Each call to
:meth:`next_click` pops one coordinate from the queue.

This is intentionally pure-NumPy, env-independent, and side-effect-
free — the agent feeds it a grid + observation history, it returns
``(x, y)`` or ``None`` when no salient targets remain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from cognithor.channels.program_synthesis.arc_agi3.fast_grid_planner import (
    find_clusters,
)

if TYPE_CHECKING:
    from cognithor.channels.program_synthesis.arc_agi3.fast_grid_planner import (
        Cluster,
    )

__all__ = ["ClickTargetSampler"]


def _background_color(grid: np.ndarray[Any, Any]) -> int:
    """The most common colour in ``grid`` is treated as background."""
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _rank_clusters(clusters: list[Cluster]) -> list[Cluster]:
    """Rank clusters by salience: smallest first (likely "target" objects).

    Ties broken by row, then column (deterministic).
    """
    return sorted(clusters, key=lambda c: (c.size, c.centroid[0], c.centroid[1]))


class ClickTargetSampler:
    """Salience-ranked click coordinate generator.

    Per-instance state:

    * ``visited`` — set of ``(x, y)`` tuples already emitted this episode.
      Reset across level boundaries via :meth:`reset_for_new_level`.
    * ``queue`` — current grid's salience-ranked click queue. Recomputed
      whenever the grid's colour-distribution signature changes.
    * ``signature`` — last grid's ``(shape, sorted unique colours)``
      tuple. Cheap to compute, stable under translation but invalidates
      when a new colour appears or the shape changes.
    """

    def __init__(self) -> None:
        self._visited: set[tuple[int, int]] = set()
        self._queue: list[tuple[int, int]] = []
        self._signature: tuple[Any, ...] | None = None

    @property
    def visited(self) -> frozenset[tuple[int, int]]:
        return frozenset(self._visited)

    def next_click(self, grid: np.ndarray[Any, Any]) -> tuple[int, int] | None:
        """Return the next salient click coordinate ``(x, y)`` or ``None``.

        Coordinates returned in SDK click format: ``x = col``, ``y = row``.
        Recomputes the queue when the grid's colour-distribution signature
        changes; otherwise keeps popping from the existing queue.
        """
        signature = self._signature_of(grid)
        if signature != self._signature:
            self._queue = self._build_queue(grid)
            self._signature = signature

        while self._queue:
            x, y = self._queue.pop(0)
            if (x, y) in self._visited:
                continue
            self._visited.add((x, y))
            return (x, y)

        return None

    def reset_for_new_level(self) -> None:
        """Clear the visited set + queue on a level transition."""
        self._visited.clear()
        self._queue.clear()
        self._signature = None

    @staticmethod
    def _signature_of(grid: np.ndarray[Any, Any]) -> tuple[Any, ...]:
        return (grid.shape, tuple(sorted(np.unique(grid).tolist())))

    @staticmethod
    def _build_queue(grid: np.ndarray[Any, Any]) -> list[tuple[int, int]]:
        bg = _background_color(grid)
        non_bg_colors = [int(c) for c in np.unique(grid).tolist() if int(c) != bg]
        all_clusters: list[Cluster] = []
        for color in non_bg_colors:
            all_clusters.extend(find_clusters(grid, color))
        ranked = _rank_clusters(all_clusters)
        # SDK convention: x = col, y = row.
        return [(c, r) for r, c in (cl.centroid for cl in ranked)]
