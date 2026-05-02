# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-12 — fast-path planner integration helpers.

The :class:`Sprint10DSLAgent` (Wave-4) drives a least-tried search over
DSL primitives. For click-toggle games (the LS20-style "click pixel,
swap a→b" mechanic) the legacy :func:`plan_click_solution` finds the
winning click sequence in pure NumPy, far faster than the agent can
discover it via single-step DSL search.

This module is the glue:

* :func:`detect_toggle_pair_from_memory` — look at the last two grids
  in :class:`EpisodeMemory` and infer the dominant ``(source, target)``
  swap, or ``None`` when the most recent transition wasn't a clean
  toggle.
* :class:`ClickPlanCache` — runs :func:`plan_click_solution` once per
  ``(state_hash, toggle_pair)`` pair, caches the resulting cluster
  centroids, and pops them one-by-one each subsequent call. The agent
  emits one click per frame; the cache survives across frames so the
  expensive search isn't re-run on every step.

The whole thing is opt-in via :class:`Sprint10DSLAgent`'s
``fast_path_enabled`` flag — disabled by default so existing
behaviour is preserved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cognithor.channels.program_synthesis.arc_agi3.fast_grid_planner import (
    detect_toggle_pair,
    find_clusters,
    plan_click_solution,
)

if TYPE_CHECKING:
    import numpy as np

    from cognithor.channels.program_synthesis.arc_agi3.episode_memory import (
        EpisodeMemory,
    )

__all__ = [
    "ClickPlanCache",
    "detect_toggle_pair_from_memory",
]


def detect_toggle_pair_from_memory(memory: EpisodeMemory) -> tuple[int, int] | None:
    """Infer the most recent ``(source, target)`` toggle from the last two grids.

    Returns ``None`` when the memory has fewer than two snapshots or the
    most recent transition wasn't a clean two-colour swap.
    """
    if len(memory) < 2:
        return None
    window = memory.window(2)
    if len(window) < 2:
        return None
    # window is most-recent first.
    g_after = window[0].grid
    g_before = window[1].grid
    if g_before.shape != g_after.shape:
        return None
    return detect_toggle_pair(g_before, g_after)


class ClickPlanCache:
    """Per-state click-plan cache for the fast-path.

    Keyed by ``(state_hash, source_color, target_color)``. The first call
    runs :func:`plan_click_solution` and stores the resulting list of
    cluster centroids in ``(x=col, y=row)`` form (the SDK's coordinate
    convention for ``ACTION6``). Subsequent calls pop centroids one at
    a time so the agent can emit one click per frame.

    On a cache miss with no available solution the cache stores the
    sentinel "no solution" so :func:`plan_click_solution` is not re-run.
    """

    _NO_SOLUTION: list[tuple[int, int]] = []  # singleton sentinel

    def __init__(self, *, max_combos: int = 100_000) -> None:
        self._max_combos = max_combos
        self._cache: dict[tuple[str, int, int], list[tuple[int, int]]] = {}

    def next_click(
        self,
        *,
        state_hash: str,
        grid: np.ndarray[Any, Any],
        source_color: int,
        target_color: int,
    ) -> tuple[int, int] | None:
        """Return the next ``(x, y)`` click, or ``None`` when no plan exists.

        On the first call for a given ``(state_hash, source, target)`` the
        plan is computed and cached. Each subsequent call pops the next
        click from the queue. Once the queue is exhausted, returns ``None``.
        """
        key = (state_hash, source_color, target_color)
        plan = self._cache.get(key)
        if plan is None:
            plan = self._compute_plan(grid, source_color, target_color)
            self._cache[key] = plan
        if not plan:
            return None
        return plan.pop(0)

    def _compute_plan(
        self,
        grid: np.ndarray[Any, Any],
        source_color: int,
        target_color: int,
    ) -> list[tuple[int, int]]:
        clusters = find_clusters(grid, source_color)
        if not clusters:
            return list(self._NO_SOLUTION)
        indices = plan_click_solution(
            grid,
            source_color,
            target_color,
            max_combos=self._max_combos,
        )
        if indices is None:
            return list(self._NO_SOLUTION)
        # Cluster centroids → SDK click format (x=col, y=row).
        return [(c, r) for r, c in (clusters[i].centroid for i in indices)]

    def clear(self) -> None:
        self._cache.clear()
