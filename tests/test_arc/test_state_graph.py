"""Tests for StateGraphNavigator and associated dataclasses."""

from __future__ import annotations

import numpy as np

from jarvis.arc.state_graph import StateGraphNavigator


def _grid(val: int) -> np.ndarray:
    return np.full((64, 64), val, dtype=np.int8)


class TestAddTransition:
    def test_basic(self):
        g = StateGraphNavigator(max_states=1000)
        h1, h2 = g.add_transition(_grid(0), "ACTION1", None, _grid(1), 100, "NOT_FINISHED")
        assert h1 != h2
        assert len(g.nodes) == 2
        assert g.total_edges == 1

    def test_win_detection(self):
        g = StateGraphNavigator()
        g.add_transition(_grid(0), "A1", None, _grid(1), 50, "GameState.WIN")
        assert len(g.win_states) == 1

    def test_game_over_detection(self):
        g = StateGraphNavigator()
        g.add_transition(_grid(0), "A1", None, _grid(1), 50, "GameState.GAME_OVER")
        assert len(g.game_over_states) == 1

    def test_duplicate_edge_increments_count(self):
        g = StateGraphNavigator()
        g.add_transition(_grid(0), "A1", None, _grid(1), 50, "NOT_FINISHED")
        g.add_transition(_grid(0), "A1", None, _grid(1), 50, "NOT_FINISHED")
        assert g.total_edges == 1  # same edge, just incremented

    def test_max_states_respected(self):
        g = StateGraphNavigator(max_states=3)
        for i in range(10):
            g.add_transition(_grid(i), "A1", None, _grid(i + 1), 10, "NOT_FINISHED")
        assert len(g.nodes) <= 3


class TestFindWinPath:
    def test_direct_win(self):
        g = StateGraphNavigator()
        h1, _ = g.add_transition(_grid(0), "A2", None, _grid(1), 200, "WIN")
        path = g.find_win_path(h1)
        assert path is not None and len(path) == 1
        assert path[0][0] == "A2"

    def test_multi_step(self):
        g = StateGraphNavigator()
        g.add_transition(_grid(0), "A1", None, _grid(1), 50, "NOT_FINISHED")
        g.add_transition(_grid(1), "A3", None, _grid(2), 100, "WIN")
        path = g.find_win_path(g.hash_grid(_grid(0)))
        assert path is not None and len(path) == 2

    def test_avoids_game_over(self):
        g = StateGraphNavigator()
        g.add_transition(_grid(0), "A1", None, _grid(1), 50, "GAME_OVER")
        g.add_transition(_grid(1), "A2", None, _grid(3), 100, "WIN")  # through game_over
        g.add_transition(_grid(0), "A3", None, _grid(2), 30, "NOT_FINISHED")
        g.add_transition(_grid(2), "A4", None, _grid(3), 100, "WIN")  # safe path
        path = g.find_win_path(g.hash_grid(_grid(0)))
        assert path is not None and path[0][0] == "A3"

    def test_no_win_states(self):
        g = StateGraphNavigator()
        assert g.find_win_path("nonexistent") is None

    def test_already_at_win(self):
        g = StateGraphNavigator()
        g.add_transition(_grid(0), "A1", None, _grid(1), 50, "WIN")
        win_hash = g.hash_grid(_grid(1))
        path = g.find_win_path(win_hash)
        assert path == []


class TestExploration:
    def test_prioritizes_untested(self):
        g = StateGraphNavigator()
        g.add_transition(_grid(0), "ACTION1", None, _grid(1), 50, "NOT_FINISHED")
        h0 = g.hash_grid(_grid(0))
        result = g.get_best_exploration_action(h0, ["ACTION1", "ACTION2", "ACTION3"])
        assert result is not None and result[0] in ["ACTION2", "ACTION3"]

    def test_returns_none_for_unknown_state(self):
        g = StateGraphNavigator()
        result = g.get_best_exploration_action("unknown", ["A1"])
        assert result is not None  # untested action from unknown state


class TestLevelTransfer:
    def test_patterns_preserved(self):
        g = StateGraphNavigator()
        g.add_transition(_grid(0), "ACTION2", None, _grid(1), 200, "WIN")
        g.prepare_for_new_level()
        assert len(g.nodes) == 0
        assert g.action_patterns_from_previous.get("ACTION2", 0) > 0

    def test_graph_cleared(self):
        g = StateGraphNavigator()
        g.add_transition(_grid(0), "A1", None, _grid(1), 50, "NOT_FINISHED")
        g.prepare_for_new_level()
        assert g.total_edges == 0
        assert len(g.win_states) == 0


class TestCoverage:
    def test_stats(self):
        g = StateGraphNavigator()
        g.add_transition(_grid(0), "A1", None, _grid(1), 50, "NOT_FINISHED")
        c = g.get_exploration_coverage()
        assert c["states"] == 2
        assert c["edges"] == 1
        assert c["win_states"] == 0


class TestSummary:
    def test_returns_string(self):
        g = StateGraphNavigator()
        s = g.get_summary_for_llm()
        assert isinstance(s, str)

    def test_includes_win_info(self):
        g = StateGraphNavigator()
        g.add_transition(_grid(0), "A1", None, _grid(1), 50, "WIN")
        s = g.get_summary_for_llm()
        assert "Win" in s or "win" in s or "1" in s
