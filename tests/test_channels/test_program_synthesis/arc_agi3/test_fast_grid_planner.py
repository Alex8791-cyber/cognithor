# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-12 — fast_grid_planner tests."""

from __future__ import annotations

import numpy as np

from cognithor.channels.program_synthesis.arc_agi3.fast_grid_planner import (
    Cluster,
    detect_toggle_pair,
    find_clusters,
    is_level_complete,
    plan_click_solution,
    simulate_combo,
    simulate_toggle,
)
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)


class TestFindClusters:
    def test_no_pixels_returns_empty(self) -> None:
        grid = np.zeros((4, 4), dtype=np.int32)
        assert find_clusters(grid, target_color=5) == []

    def test_single_cluster(self) -> None:
        grid = np.zeros((4, 4), dtype=np.int32)
        grid[1, 1] = 7
        grid[1, 2] = 7
        grid[2, 1] = 7
        clusters = find_clusters(grid, target_color=7)
        assert len(clusters) == 1
        assert clusters[0].color == 7
        assert clusters[0].size == 3

    def test_two_disconnected_clusters(self) -> None:
        grid = np.zeros((5, 5), dtype=np.int32)
        # Cluster A: top-left
        grid[0, 0] = 3
        grid[0, 1] = 3
        # Cluster B: bottom-right (disconnected)
        grid[4, 4] = 3
        clusters = find_clusters(grid, target_color=3)
        assert len(clusters) == 2
        sizes = sorted(c.size for c in clusters)
        assert sizes == [1, 2]

    def test_diagonal_not_connected(self) -> None:
        # 4-connectivity: diagonals are NOT connected.
        grid = np.zeros((3, 3), dtype=np.int32)
        grid[0, 0] = 9
        grid[1, 1] = 9
        clusters = find_clusters(grid, target_color=9)
        assert len(clusters) == 2

    def test_centroid_computation(self) -> None:
        grid = np.zeros((4, 4), dtype=np.int32)
        grid[1, 1] = 4
        grid[1, 2] = 4
        clusters = find_clusters(grid, target_color=4)
        assert clusters[0].centroid == (1, 1)


class TestDetectTogglePair:
    def test_no_change_returns_none(self) -> None:
        grid = np.array([[1, 2], [3, 4]], dtype=np.int32)
        assert detect_toggle_pair(grid, grid.copy()) is None

    def test_simple_toggle(self) -> None:
        before = np.array([[1, 1], [1, 1]], dtype=np.int32)
        after = np.array([[2, 2], [2, 2]], dtype=np.int32)
        result = detect_toggle_pair(before, after)
        assert result == (1, 2)

    def test_dominant_pair_wins_on_ties(self) -> None:
        # Three 1→2 swaps, one 3→4 swap → (1, 2) dominates.
        before = np.array([[1, 1, 1, 3]], dtype=np.int32)
        after = np.array([[2, 2, 2, 4]], dtype=np.int32)
        result = detect_toggle_pair(before, after)
        assert result == (1, 2)


class TestSimulation:
    def test_simulate_toggle_swaps_pixels(self) -> None:
        grid = np.array([[1, 1], [0, 0]], dtype=np.int32)
        cluster = Cluster(color=1, pixels=((0, 0), (0, 1)))
        result = simulate_toggle(grid, cluster, source_color=1, target_color=2)
        assert result[0, 0] == 2
        assert result[0, 1] == 2
        # Source untouched.
        assert grid[0, 0] == 1

    def test_simulate_combo_chains_toggles(self) -> None:
        grid = np.array([[1, 0], [1, 0]], dtype=np.int32)
        c0 = Cluster(color=1, pixels=((0, 0),))
        c1 = Cluster(color=1, pixels=((1, 0),))
        result = simulate_combo(grid, [c0, c1], (0, 1), source_color=1, target_color=9)
        assert result[0, 0] == 9
        assert result[1, 0] == 9

    def test_is_level_complete(self) -> None:
        grid = np.array([[2, 2], [2, 2]], dtype=np.int32)
        assert is_level_complete(grid, source_color=1) is True
        assert is_level_complete(grid, source_color=2) is False


class TestPlanClickSolution:
    def test_no_clusters_returns_none(self) -> None:
        grid = np.zeros((3, 3), dtype=np.int32)
        assert plan_click_solution(grid, source_color=5, target_color=6) is None

    def test_single_cluster_solution(self) -> None:
        # One blob of source colour → click it once.
        grid = np.zeros((4, 4), dtype=np.int32)
        grid[1, 1] = 1
        grid[1, 2] = 1
        result = plan_click_solution(grid, source_color=1, target_color=2)
        assert result == (0,)

    def test_two_independent_clusters_need_two_clicks(self) -> None:
        grid = np.zeros((5, 5), dtype=np.int32)
        grid[0, 0] = 1
        grid[4, 4] = 1
        result = plan_click_solution(grid, source_color=1, target_color=2)
        assert result is not None
        assert len(result) == 2

    def test_max_combos_caps_search(self) -> None:
        # Many small clusters but a tiny budget → returns None.
        grid = np.zeros((6, 6), dtype=np.int32)
        for i in range(6):
            grid[i, i] = 1
        result = plan_click_solution(grid, source_color=1, target_color=2, max_combos=2)
        # Six singletons → we need all six toggled to win, but C(6,1)=6 alone
        # exceeds the budget of 2 → None.
        assert result is None

    def test_smallest_subset_preferred(self) -> None:
        # Three clusters; toggling any one alone clears that one only,
        # so the only winning subset is all three.
        grid = np.zeros((5, 5), dtype=np.int32)
        grid[0, 0] = 1
        grid[2, 2] = 1
        grid[4, 4] = 1
        result = plan_click_solution(grid, source_color=1, target_color=2)
        assert result is not None
        assert len(result) == 3
