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


class TestVisionCalls:
    def test_vision_call_1_returns_dict(self):
        analyzer = GameAnalyzer.__new__(GameAnalyzer)
        grid = np.zeros((64, 64), dtype=np.int8)
        grid[10:20, 10:20] = 3

        mock_resp = {
            "message": {
                "content": json.dumps({
                    "game_type": "click",
                    "target_color": 3,
                    "strategy": "Click red clusters",
                    "description": "Grid with red blocks",
                })
            }
        }

        with patch("jarvis.arc.game_analyzer.ollama") as mock_ollama:
            mock_ollama.chat.return_value = mock_resp
            result = analyzer._vision_call_initial(grid, [5, 6])

        assert result is not None
        assert result["game_type"] == "click"

    def test_vision_call_1_ollama_error_returns_none(self):
        analyzer = GameAnalyzer.__new__(GameAnalyzer)
        grid = np.zeros((64, 64), dtype=np.int8)

        with patch("jarvis.arc.game_analyzer.ollama") as mock_ollama:
            mock_ollama.chat.side_effect = ConnectionError("Ollama offline")
            result = analyzer._vision_call_initial(grid, [1, 2, 3, 4])

        assert result is None

    def test_vision_call_2_with_diff(self):
        analyzer = GameAnalyzer.__new__(GameAnalyzer)
        grid_before = np.zeros((64, 64), dtype=np.int8)
        grid_after = np.zeros((64, 64), dtype=np.int8)
        grid_after[10:20, 10:20] = 5

        mock_resp = {
            "message": {
                "content": json.dumps({
                    "win_condition": "clear_board",
                    "correction": None,
                    "description": "Clusters toggled from red to yellow",
                })
            }
        }

        with patch("jarvis.arc.game_analyzer.ollama") as mock_ollama:
            mock_ollama.chat.return_value = mock_resp
            result = analyzer._vision_call_final(grid_before, grid_after)

        assert result is not None
        assert result["win_condition"] == "clear_board"


class TestAnalyze:
    def test_analyze_uses_cache(self, tmp_path):
        from jarvis.arc.game_profile import GameProfile

        # Pre-save a profile
        p = GameProfile(
            game_id="cached_game",
            game_type="click",
            available_actions=[6],
            click_zones=[(10, 10)],
            target_colors=[3],
            movement_effects={},
            win_condition="clear_board",
            vision_description="cached",
            vision_strategy="cached",
            strategy_metrics={},
            analyzed_at="2026-01-01",
        )
        p.save(base_dir=tmp_path)

        analyzer = GameAnalyzer(arcade=None)
        result = analyzer.analyze("cached_game", base_dir=tmp_path)

        assert result.game_id == "cached_game"
        assert result.vision_description == "cached"

    def test_analyze_force_ignores_cache(self, tmp_path):
        from jarvis.arc.game_profile import GameProfile

        p = GameProfile(
            game_id="force_test",
            game_type="click",
            available_actions=[6],
            click_zones=[],
            target_colors=[],
            movement_effects={},
            win_condition="unknown",
            vision_description="old",
            vision_strategy="old",
            strategy_metrics={},
            analyzed_at="2026-01-01",
        )
        p.save(base_dir=tmp_path)

        initial_grid = np.zeros((64, 64), dtype=np.int8)
        initial_grid[10:15, 10:15] = 3
        not_finished = _make_mock_game_state("NOT_FINISHED")
        game_over = _make_mock_game_state("GAME_OVER")

        mock_env = MagicMock()
        step_count = [0]

        def mock_step(action, data=None):
            step_count[0] += 1
            # After a few steps, return GAME_OVER to end sacrifice
            state = game_over if step_count[0] > 3 else not_finished
            return _make_mock_obs(
                grid=np.expand_dims(initial_grid, 0),
                state=state,
                actions=[MagicMock(value=6)],
            )

        mock_env.step = mock_step
        mock_env.reset.return_value = _make_mock_obs(
            grid=np.expand_dims(initial_grid, 0),
            state=not_finished,
            actions=[MagicMock(value=a) for a in [5, 6]],
        )

        mock_arcade = MagicMock()
        mock_arcade.make.return_value = mock_env

        vision_resp = {
            "message": {
                "content": json.dumps({
                    "game_type": "click",
                    "target_color": 3,
                    "strategy": "new strategy",
                    "description": "new description",
                })
            }
        }

        analyzer = GameAnalyzer(arcade=mock_arcade)
        with patch("jarvis.arc.game_analyzer.ollama") as mock_ollama:
            mock_ollama.chat.return_value = vision_resp
            result = analyzer.analyze("force_test", force=True, base_dir=tmp_path)

        assert result.vision_description != "old"


class TestCLIIntegration:
    def test_build_parser_accepts_analyzer_mode(self):
        from jarvis.arc.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["--mode", "analyzer", "--game", "ft09"])
        assert args.mode == "analyzer"
        assert args.game == "ft09"

    def test_build_parser_accepts_reanalyze_flag(self):
        from jarvis.arc.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["--mode", "analyzer", "--reanalyze"])
        assert args.reanalyze is True

    def test_analyzer_mode_requires_game_or_all(self):
        """analyzer mode should work without --game (runs all games)."""
        from jarvis.arc.__main__ import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["--mode", "analyzer"])
        assert args.mode == "analyzer"
        assert args.game == ""
