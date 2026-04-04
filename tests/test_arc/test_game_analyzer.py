"""Tests for GameAnalyzer — opferlevel + vision analysis."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jarvis.arc.game_analyzer import (
    GameAnalyzer,
    SacrificeReport,
    _grid_to_png_b64,
    _parse_vision_json,
)


class TestVisionHelpers:
    def test_grid_to_png_b64_produces_base64(self):
        grid = np.zeros((64, 64), dtype=np.int8)
        grid[10:20, 10:20] = 3  # red block
        b64 = _grid_to_png_b64(grid, scale=4)
        assert isinstance(b64, str)
        assert len(b64) > 100
        # Should be valid base64
        import base64
        raw = base64.b64decode(b64)
        assert raw[:4] == b"\x89PNG"

    def test_grid_to_png_b64_handles_3d(self):
        grid = np.zeros((1, 64, 64), dtype=np.int8)
        b64 = _grid_to_png_b64(grid, scale=2)
        assert isinstance(b64, str)

    def test_parse_vision_json_direct(self):
        raw = '{"game_type": "click", "target_color": 3}'
        result = _parse_vision_json(raw)
        assert result["game_type"] == "click"

    def test_parse_vision_json_markdown(self):
        raw = 'Some text\n```json\n{"game_type": "keyboard"}\n```\nMore text'
        result = _parse_vision_json(raw)
        assert result["game_type"] == "keyboard"

    def test_parse_vision_json_with_think_tags(self):
        raw = '<think>reasoning here</think>\n{"game_type": "mixed"}'
        result = _parse_vision_json(raw)
        assert result["game_type"] == "mixed"

    def test_parse_vision_json_balanced_brace(self):
        raw = 'The answer is {"game_type": "click", "nested": {"a": 1}} and more'
        result = _parse_vision_json(raw)
        assert result["game_type"] == "click"

    def test_parse_vision_json_unparseable(self):
        assert _parse_vision_json("no json here at all") is None


class TestSacrificeReport:
    def test_defaults(self):
        r = SacrificeReport()
        assert r.clicks_tested == []
        assert r.movements_tested == {}
        assert r.unique_states_seen == 0
        assert r.game_over_trigger is None
        assert r.frames == []


from jarvis.arc.error_handler import safe_frame_extract


def _make_mock_obs(grid=None, state="NOT_FINISHED", levels=0, actions=None):
    """Create a mock ARC SDK observation."""
    if grid is None:
        grid = np.zeros((1, 64, 64), dtype=np.int8)
    obs = MagicMock()
    obs.frame = grid
    obs.state = state
    obs.levels_completed = levels
    obs.available_actions = actions or []
    obs.win_levels = 0
    return obs


def _make_mock_game_state(name):
    state = MagicMock()
    state.name = name
    state.__eq__ = lambda self, other: getattr(other, "name", other) == name
    return state


class TestSacrificeLevel:
    def test_run_sacrifice_keyboard_only(self):
        """Keyboard-only game: tests directions 1-4, no clicks."""
        initial_grid = np.zeros((64, 64), dtype=np.int8)
        initial_grid[30:34, 30:34] = 2  # blue block

        moved_grid = np.zeros((64, 64), dtype=np.int8)
        moved_grid[31:35, 30:34] = 2  # shifted down

        not_finished = _make_mock_game_state("NOT_FINISHED")

        mock_env = MagicMock()
        call_count = [0]

        def mock_step(action, data=None):
            call_count[0] += 1
            obs = _make_mock_obs(
                grid=np.expand_dims(moved_grid if call_count[0] % 2 else initial_grid, 0),
                state=not_finished,
                actions=[MagicMock(value=a) for a in [1, 2, 3, 4, 5]],
            )
            return obs

        mock_env.step = mock_step

        analyzer = GameAnalyzer.__new__(GameAnalyzer)
        report = analyzer._run_sacrifice_level(
            mock_env, initial_grid, available_action_ids=[1, 2, 3, 4, 5]
        )

        assert isinstance(report, SacrificeReport)
        # Should have tested all 4 directions
        assert len(report.movements_tested) == 4
        assert report.unique_states_seen >= 1

    def test_run_sacrifice_click_game(self):
        """Click game: finds clusters and tests clicks."""
        initial_grid = np.zeros((64, 64), dtype=np.int8)
        # Two distinct clusters of colour 3
        initial_grid[10:15, 10:15] = 3
        initial_grid[40:45, 40:45] = 3

        toggled_grid = initial_grid.copy()
        toggled_grid[10:15, 10:15] = 5  # toggled to colour 5

        not_finished = _make_mock_game_state("NOT_FINISHED")

        mock_env = MagicMock()
        click_count = [0]

        def mock_step(action, data=None):
            click_count[0] += 1
            obs = _make_mock_obs(
                grid=np.expand_dims(toggled_grid, 0),
                state=not_finished,
                actions=[MagicMock(value=a) for a in [5, 6]],
            )
            return obs

        mock_env.step = mock_step

        analyzer = GameAnalyzer.__new__(GameAnalyzer)
        report = analyzer._run_sacrifice_level(
            mock_env, initial_grid, available_action_ids=[5, 6]
        )

        assert isinstance(report, SacrificeReport)
        assert len(report.clicks_tested) > 0
        assert report.unique_states_seen >= 1

    def test_run_sacrifice_game_over(self):
        """GAME_OVER during sacrifice is handled gracefully."""
        initial_grid = np.zeros((64, 64), dtype=np.int8)
        initial_grid[20:30, 20:30] = 3

        game_over = _make_mock_game_state("GAME_OVER")

        mock_env = MagicMock()
        mock_env.step.return_value = _make_mock_obs(
            grid=np.expand_dims(initial_grid, 0),
            state=game_over,
            actions=[MagicMock(value=6)],
        )

        analyzer = GameAnalyzer.__new__(GameAnalyzer)
        report = analyzer._run_sacrifice_level(
            mock_env, initial_grid, available_action_ids=[5, 6]
        )

        assert report.game_over_trigger is not None
