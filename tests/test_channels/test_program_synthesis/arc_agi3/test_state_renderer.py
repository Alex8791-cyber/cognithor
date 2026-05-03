# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-19 — state_renderer tests."""

from __future__ import annotations

import numpy as np
import pytest

from cognithor.channels.program_synthesis.arc_agi3.episode_memory import EpisodeMemory
from cognithor.channels.program_synthesis.arc_agi3.state_renderer import (
    render_cluster_summary,
    render_delta_summary,
    render_state_changes_in_window,
)
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)


def _g(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


# ---------------------------------------------------------------------------
# render_cluster_summary
# ---------------------------------------------------------------------------


class TestRenderClusterSummary:
    def test_empty_grid_returns_marker(self) -> None:
        assert render_cluster_summary(np.zeros((0, 0), dtype=np.int8)) == "(empty grid)"

    def test_only_background_returns_marker(self) -> None:
        grid = np.zeros((5, 5), dtype=np.int8)
        assert render_cluster_summary(grid) == "(no non-background pixels)"

    def test_single_cluster_single_color(self) -> None:
        # 3-cell L-shape of color 4 in a 3x3 grid of zeros
        grid = _g([[4, 4, 0], [0, 4, 0], [0, 0, 0]])
        out = render_cluster_summary(grid)
        assert "color 4" in out
        assert "1 cluster," in out
        assert "3 cells" in out
        assert "biggest 3@" in out

    def test_two_disconnected_clusters_same_color(self) -> None:
        # color 7 has two disconnected pixels
        grid = _g([[7, 0, 7], [0, 0, 0], [0, 0, 0]])
        out = render_cluster_summary(grid)
        assert "color 7: 2 clusters, 2 cells" in out

    def test_orders_by_total_cells_desc(self) -> None:
        # color 4 has 4 cells, color 7 has 2 cells → color 4 first
        grid = _g([[4, 4, 0, 7], [4, 4, 0, 7]])
        out = render_cluster_summary(grid)
        lines = out.splitlines()
        assert "color 4" in lines[0]
        assert "color 7" in lines[1]

    def test_truncates_to_max_lines(self) -> None:
        # Build a grid with 10 distinct colors
        grid = _g([[i for i in range(1, 11)]])
        out = render_cluster_summary(grid, max_lines=3)
        assert len(out.splitlines()) == 3

    def test_custom_background_skips_color(self) -> None:
        # If 8 is also background, color 4 is the only thing left
        grid = _g([[8, 8, 4], [8, 8, 4]])
        out = render_cluster_summary(grid, background_colors=(0, 8))
        assert "color 8" not in out
        assert "color 4" in out


# ---------------------------------------------------------------------------
# render_delta_summary
# ---------------------------------------------------------------------------


class TestRenderDeltaSummary:
    def test_no_change_returns_marker(self) -> None:
        a = _g([[1, 2], [3, 4]])
        assert render_delta_summary(a, a) == "(no change)"

    def test_shape_mismatch_returns_marker(self) -> None:
        a = _g([[1, 2]])
        b = _g([[1, 2], [3, 4]])
        assert render_delta_summary(a, b) == "(grids have different shapes)"

    def test_added_cells_one_color(self) -> None:
        before = _g([[0, 0], [0, 0]])
        after = _g([[4, 4], [0, 0]])
        out = render_delta_summary(before, after)
        assert "color 4: +2 cells" in out
        assert "(2 added, 0 lost)" in out

    def test_lost_cells_one_color(self) -> None:
        before = _g([[7, 7], [7, 0]])
        after = _g([[0, 0], [0, 0]])
        out = render_delta_summary(before, after)
        assert "color 7: -3 cells" in out

    def test_balanced_color_swap_within_one_color(self) -> None:
        # color 4 moved one cell — same total, just different position
        before = _g([[4, 0], [0, 0]])
        after = _g([[0, 4], [0, 0]])
        out = render_delta_summary(before, after)
        assert "color 4: balanced" in out
        assert "(1 added, 1 lost)" in out

    def test_color_swap_between_two(self) -> None:
        before = _g([[4, 4], [7, 7]])
        after = _g([[7, 7], [4, 4]])
        out = render_delta_summary(before, after)
        assert "color 4" in out
        assert "color 7" in out
        # both balanced (each lost 2 + gained 2)
        assert out.count("balanced") == 2

    def test_background_only_change_filtered(self) -> None:
        # change is only in background color → message says so
        before = _g([[0, 0], [0, 0]])
        after = _g([[0, 0], [0, 0]])
        # Identical → no-change branch
        assert render_delta_summary(before, after) == "(no change)"


# ---------------------------------------------------------------------------
# render_state_changes_in_window
# ---------------------------------------------------------------------------


class TestRenderStateChangesInWindow:
    def test_empty_memory(self) -> None:
        m = EpisodeMemory()
        assert render_state_changes_in_window(m) == "(no actions yet)"

    def test_rejects_invalid_max_steps(self) -> None:
        with pytest.raises(ValueError, match="max_steps must be >= 1"):
            render_state_changes_in_window(EpisodeMemory(), max_steps=0)

    def test_single_step_no_pair_yet(self) -> None:
        # Only one entry → no transitions to render
        m = EpisodeMemory()
        m.append(grid=_g([[1]]), action_name="ACTION1", levels_completed=0)
        out = render_state_changes_in_window(m)
        assert out == "(no completed transitions yet)"

    def test_renders_recent_first(self) -> None:
        m = EpisodeMemory()
        m.append(grid=_g([[0, 0]]), action_name="ACTION1", levels_completed=0)
        m.append(grid=_g([[4, 0]]), action_name="ACTION3", levels_completed=0)
        m.append(grid=_g([[4, 4]]), action_name="ACTION3", levels_completed=0)
        out = render_state_changes_in_window(m, max_steps=5)
        lines = out.splitlines()
        # Most recent first: step -1 then step -2
        assert lines[0].startswith("step -1 (ACTION3):")
        assert lines[1].startswith("step -2 (ACTION3):")
        # The first transition added 1 cell of color 4
        assert "+1 cells" in lines[1]
        # Second transition added another cell of color 4
        assert "+1 cells" in lines[0]

    def test_caps_at_max_steps(self) -> None:
        m = EpisodeMemory()
        for i in range(10):
            grid = _g([[(i % 4) + 1]])
            m.append(grid=grid, action_name=f"ACTION{i % 4}", levels_completed=0)
        out = render_state_changes_in_window(m, max_steps=3)
        assert len(out.splitlines()) == 3
