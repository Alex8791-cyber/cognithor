"""Tests for PerGameSolver -- budget-based strategy execution."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jarvis.arc.game_profile import GameProfile, StrategyMetrics
from jarvis.arc.per_game_solver import (
    BudgetSlot,
    PerGameSolver,
    SolveResult,
    StrategyOutcome,
)


def _make_profile(game_type="click", metrics=None, has_toggles=False) -> GameProfile:
    return GameProfile(
        game_id="test_game",
        game_type=game_type,
        available_actions=[5, 6] if game_type == "click" else [1, 2, 3, 4, 5],
        click_zones=[(10, 10), (30, 30)] if game_type == "click" else [],
        target_colors=[3] if game_type == "click" else [],
        movement_effects={1: "moves_player", 2: "moves_player"} if game_type != "click" else {},
        win_condition="clear_board",
        vision_description="test",
        vision_strategy="test",
        strategy_metrics=metrics or {},
        analyzed_at="2026-04-04",
        has_toggles=has_toggles,
    )


class TestBudgetAllocation:
    def test_default_click_allocation(self):
        profile = _make_profile("click")
        # Default has_toggles=False → sequence_click first
        solver = PerGameSolver(profile, arcade=MagicMock())
        slots = solver._allocate_budget(level_num=0)

        assert len(slots) == 3
        assert slots[0].strategy == "sequence_click"
        assert slots[0].max_actions == 120  # 60% of 200
        assert slots[1].strategy == "cluster_click"
        assert slots[1].max_actions == 60   # 30% of 200
        assert slots[2].strategy == "targeted_click"
        assert slots[2].max_actions == 20   # 10% of 200

    def test_default_keyboard_allocation(self):
        profile = _make_profile("keyboard")
        solver = PerGameSolver(profile, arcade=MagicMock())
        slots = solver._allocate_budget(level_num=0)

        assert len(slots) == 3
        assert slots[0].strategy == "keyboard_explore"
        assert slots[0].max_actions == 100  # 50% of 200

    def test_default_mixed_allocation(self):
        profile = _make_profile("mixed")
        solver = PerGameSolver(profile, arcade=MagicMock())
        slots = solver._allocate_budget(level_num=0)

        assert slots[0].strategy == "hybrid"
        assert slots[0].max_actions == 50  # 50% of 100

    def test_ranked_allocation_overrides_defaults(self):
        metrics = {
            "keyboard_explore": StrategyMetrics(attempts=10, wins=8),
            "cluster_click": StrategyMetrics(attempts=10, wins=2),
            "hybrid": StrategyMetrics(attempts=5, wins=1),
        }
        profile = _make_profile("click", metrics=metrics)
        solver = PerGameSolver(profile, arcade=MagicMock())
        slots = solver._allocate_budget(level_num=0)

        # keyboard_explore has highest win_rate -> gets 50%
        assert slots[0].strategy == "keyboard_explore"
        assert slots[1].strategy == "cluster_click"
        assert slots[2].strategy == "hybrid"


class TestStagnationDetection:
    def test_no_stagnation_with_changes(self):
        solver = PerGameSolver(_make_profile(), arcade=MagicMock())
        grids = [np.random.randint(0, 10, (64, 64)) for _ in range(5)]
        assert solver._detect_stagnation(grids) is False

    def test_stagnation_with_identical_frames(self):
        solver = PerGameSolver(_make_profile(), arcade=MagicMock())
        same = np.zeros((64, 64), dtype=np.int8)
        grids = [same.copy() for _ in range(5)]
        assert solver._detect_stagnation(grids) is True

    def test_stagnation_with_tiny_changes(self):
        solver = PerGameSolver(_make_profile(), arcade=MagicMock())
        base = np.zeros((64, 64), dtype=np.int8)
        grids = []
        for i in range(5):
            g = base.copy()
            g[0, i] = 1  # only 1 pixel changes per frame
            grids.append(g)
        # Max diff between consecutive = 2 pixels (one removed, one added)
        # Under threshold of 10 -> stagnation
        assert solver._detect_stagnation(grids) is True

    def test_no_stagnation_with_short_history(self):
        solver = PerGameSolver(_make_profile(), arcade=MagicMock())
        same = np.zeros((64, 64), dtype=np.int8)
        assert solver._detect_stagnation([same, same]) is False  # < 5 frames


class TestSolveResult:
    def test_defaults(self):
        r = SolveResult(game_id="test", levels_completed=0, total_steps=0, strategy_log=[], score=0.0)
        assert r.game_id == "test"


def _make_mock_game_state(name):
    state = MagicMock()
    state.name = name
    state.__eq__ = lambda self, other: getattr(other, "name", other) == name
    return state


def _make_mock_obs(grid=None, state_name="NOT_FINISHED", levels=0, actions=None):
    if grid is None:
        grid = np.zeros((1, 64, 64), dtype=np.int8)
    obs = MagicMock()
    obs.frame = grid
    obs.state = _make_mock_game_state(state_name)
    obs.levels_completed = levels
    obs.available_actions = actions or []
    obs.win_levels = 0
    return obs


class TestStrategyExecution:
    def test_execute_targeted_click_win(self):
        """targeted_click strategy clicks on known zones and wins."""
        profile = _make_profile("click")

        mock_env = MagicMock()
        step_count = [0]

        def mock_step(action, data=None):
            step_count[0] += 1
            if step_count[0] >= 2:
                return _make_mock_obs(state_name="WIN", levels=1)
            return _make_mock_obs()

        mock_env.step = mock_step

        solver = PerGameSolver(profile, arcade=MagicMock())
        outcome = solver._execute_strategy(mock_env, "targeted_click", max_actions=10)

        assert isinstance(outcome, StrategyOutcome)
        assert outcome.won is True
        assert outcome.steps > 0

    def test_execute_keyboard_explore(self):
        """keyboard_explore strategy runs actions without error."""
        profile = _make_profile("keyboard")
        mock_env = MagicMock()
        call_count = [0]

        def varied_step(action, data=None):
            call_count[0] += 1
            grid = np.full((1, 64, 64), call_count[0] % 256, dtype=np.int8)
            return _make_mock_obs(grid=grid)

        mock_env.step = varied_step

        solver = PerGameSolver(profile, arcade=MagicMock())
        outcome = solver._execute_strategy(mock_env, "keyboard_explore", max_actions=20)

        assert isinstance(outcome, StrategyOutcome)
        assert outcome.steps == 20  # used full budget

    def test_execute_stops_on_game_over(self):
        """Strategy stops when GAME_OVER is received."""
        profile = _make_profile("click")
        mock_env = MagicMock()
        mock_env.step.return_value = _make_mock_obs(state_name="GAME_OVER")

        solver = PerGameSolver(profile, arcade=MagicMock())
        outcome = solver._execute_strategy(mock_env, "targeted_click", max_actions=10)

        assert outcome.won is False
        assert outcome.game_over is True

    def test_execute_stops_on_stagnation(self):
        """Strategy switches on stagnation (identical frames)."""
        profile = _make_profile("keyboard")
        same_grid = np.zeros((1, 64, 64), dtype=np.int8)
        mock_env = MagicMock()
        mock_env.step.return_value = _make_mock_obs(grid=same_grid)

        solver = PerGameSolver(profile, arcade=MagicMock())
        outcome = solver._execute_strategy(mock_env, "keyboard_explore", max_actions=50)

        # Should stop early due to stagnation (after ~5 identical frames)
        assert outcome.steps < 50
        assert outcome.stagnated is True


class TestSolve:
    def test_solve_single_level_win(self):
        """solve() wins a single level and returns SolveResult."""
        profile = _make_profile("click")

        mock_env = MagicMock()
        step_count = [0]

        def mock_step(action, data=None):
            step_count[0] += 1
            if step_count[0] >= 2:
                return _make_mock_obs(state_name="WIN", levels=1)
            return _make_mock_obs()

        mock_env.step = mock_step
        mock_env.reset.return_value = _make_mock_obs()

        mock_arcade = MagicMock()
        mock_arcade.make.return_value = mock_env

        solver = PerGameSolver(profile, arcade=mock_arcade)
        result = solver.solve(max_levels=1)

        assert isinstance(result, SolveResult)
        assert result.levels_completed >= 1
        assert result.total_steps > 0
        assert len(result.strategy_log) >= 1

    def test_solve_skips_failed_level(self):
        """solve() moves to next level after all strategies fail."""
        profile = _make_profile("click")

        mock_env = MagicMock()
        # Always return same grid -> stagnation -> all strategies fail
        same_grid = np.zeros((1, 64, 64), dtype=np.int8)
        mock_env.step.return_value = _make_mock_obs(grid=same_grid)
        mock_env.reset.return_value = _make_mock_obs(grid=same_grid)

        mock_arcade = MagicMock()
        mock_arcade.make.return_value = mock_env

        solver = PerGameSolver(profile, arcade=mock_arcade)
        result = solver.solve(max_levels=2)

        assert result.levels_completed == 0

    def test_solve_updates_profile_metrics(self, tmp_path):
        """solve() updates strategy metrics in the profile."""
        profile = _make_profile("click")

        mock_env = MagicMock()
        step_count = [0]

        def mock_step(action, data=None):
            step_count[0] += 1
            if step_count[0] >= 2:
                return _make_mock_obs(state_name="WIN", levels=1)
            return _make_mock_obs()

        mock_env.step = mock_step
        mock_env.reset.return_value = _make_mock_obs()

        mock_arcade = MagicMock()
        mock_arcade.make.return_value = mock_env

        solver = PerGameSolver(profile, arcade=mock_arcade)
        solver.solve(max_levels=1, base_dir=tmp_path)

        # Profile should have been updated
        assert profile.total_runs == 1
        assert len(profile.strategy_metrics) > 0

    def test_solve_respects_timeout(self):
        """solve() respects the timeout per game."""
        profile = _make_profile("keyboard")

        mock_env = MagicMock()
        # Return varied grids so stagnation doesn't trigger
        call_count = [0]
        def varied_step(action, data=None):
            call_count[0] += 1
            grid = np.full((1, 64, 64), call_count[0] % 16, dtype=np.int8)
            return _make_mock_obs(grid=grid)

        mock_env.step = varied_step
        mock_env.reset.return_value = _make_mock_obs()

        mock_arcade = MagicMock()
        mock_arcade.make.return_value = mock_env

        solver = PerGameSolver(profile, arcade=mock_arcade)
        # With a tiny timeout, should return quickly
        result = solver.solve(max_levels=10, timeout_s=0.1)

        assert isinstance(result, SolveResult)


class TestClusterClickStrategy:
    def test_cluster_click_uses_find_solution(self):
        """cluster_click delegates to ClusterSolver.find_solution() and solves levels."""
        from arcengine.enums import GameState

        profile = _make_profile("click")
        profile.target_colors = [3]

        # Initial grid with 3 clusters of color 3
        grid = np.zeros((64, 64), dtype=np.int8)
        grid[10:15, 10:15] = 3
        grid[30:35, 30:35] = 3
        grid[50:55, 50:55] = 3

        make_count = [0]

        def mock_make(game_id=None):
            make_count[0] += 1
            env = MagicMock()
            click_count = [0]

            def env_step(action, data=None):
                click_count[0] += 1
                obs = MagicMock()
                obs.frame = np.expand_dims(grid, 0)
                # After clicking all 3 clusters, levels_completed increments
                if click_count[0] >= 3:
                    obs.levels_completed = 1
                    obs.state = GameState.NOT_FINISHED
                else:
                    obs.levels_completed = 0
                    obs.state = GameState.NOT_FINISHED
                return obs

            env.step = env_step
            obs0 = MagicMock()
            obs0.frame = np.expand_dims(grid, 0)
            obs0.levels_completed = 0
            obs0.state = GameState.NOT_FINISHED
            env.reset.return_value = obs0
            return env

        mock_arcade = MagicMock()
        mock_arcade.make = mock_make

        solver = PerGameSolver(profile, arcade=mock_arcade)
        outcome = solver._execute_cluster_click(
            initial_grid=grid, target_color=3, max_actions=20
        )

        assert outcome.won is True
        assert outcome.levels_solved >= 1
        assert make_count[0] > 0

    def test_cluster_click_no_target_color_returns_empty(self):
        """cluster_click with no target color returns no-win outcome."""
        profile = _make_profile("click")
        profile.target_colors = []

        solver = PerGameSolver(profile, arcade=MagicMock())
        grid = np.zeros((64, 64), dtype=np.int8)
        outcome = solver._execute_cluster_click(grid, target_color=None, max_actions=10)

        assert outcome.won is False
        assert outcome.steps == 0


class TestEffectivePositionScanner:
    def test_finds_effective_positions(self):
        """Scan should find positions where clicks change the puzzle grid."""
        from arcengine.enums import GameState

        grid_initial = np.zeros((64, 64), dtype=np.int8)
        grid_initial[0, :] = 7  # orange bar at row 0

        grid_changed = grid_initial.copy()
        grid_changed[10:20, 10:20] = 5  # big change at certain region

        def mock_step(action, data=None):
            x = data.get("x", 0) if data else 0
            y = data.get("y", 0) if data else 0
            # Only clicks near (10,10) cause a puzzle change
            if 8 <= x <= 22 and 8 <= y <= 22:
                grid_out = grid_changed.copy()
                grid_out[0, 63] = 4  # bar change
            else:
                grid_out = grid_initial.copy()
                grid_out[0, 63] = 4  # bar-only change
            obs = MagicMock()
            obs.frame = np.expand_dims(grid_out, 0)
            obs.state = GameState.NOT_FINISHED
            obs.levels_completed = 0
            return obs

        mock_env = MagicMock()
        mock_env.step = mock_step
        obs0 = MagicMock()
        obs0.frame = np.expand_dims(grid_initial, 0)
        obs0.state = GameState.NOT_FINISHED
        obs0.levels_completed = 0
        mock_env.reset.return_value = obs0

        solver = PerGameSolver(_make_profile("click"), arcade=MagicMock())
        positions = solver._scan_effective_positions(mock_env, replay_sequence=[])

        assert len(positions) > 0
        # All found positions should be in the effective region
        for x, y in positions:
            assert 6 <= x <= 24 and 6 <= y <= 24, f"Unexpected position ({x},{y})"

    def test_returns_empty_when_no_effective(self):
        """Scan returns empty list when no clicks cause puzzle changes."""
        from arcengine.enums import GameState

        grid = np.zeros((64, 64), dtype=np.int8)
        grid[0, :] = 7

        def mock_step(action, data=None):
            grid_out = grid.copy()
            grid_out[0, 63] = 4  # bar-only change
            obs = MagicMock()
            obs.frame = np.expand_dims(grid_out, 0)
            obs.state = GameState.NOT_FINISHED
            obs.levels_completed = 0
            return obs

        mock_env = MagicMock()
        mock_env.step = mock_step
        obs0 = MagicMock()
        obs0.frame = np.expand_dims(grid, 0)
        obs0.state = GameState.NOT_FINISHED
        obs0.levels_completed = 0
        mock_env.reset.return_value = obs0

        solver = PerGameSolver(_make_profile("click"), arcade=MagicMock())
        positions = solver._scan_effective_positions(mock_env, replay_sequence=[])

        assert positions == []

    def test_separates_different_valve_regions(self):
        """Two distant valve regions should produce separate groups."""
        from arcengine.enums import GameState

        grid = np.zeros((64, 64), dtype=np.int8)
        grid[0, :] = 7

        def mock_step(action, data=None):
            x = data.get("x", 0) if data else 0
            y = data.get("y", 0) if data else 0
            grid_out = grid.copy()
            grid_out[0, 63] = 4
            # Two separate valve regions with different diffs
            if 8 <= x <= 12 and 8 <= y <= 12:
                grid_out[20:30, 20:30] = 3  # 100 px change
            elif 40 <= x <= 44 and 40 <= y <= 44:
                grid_out[50:55, 50:55] = 5  # 25 px change
            obs = MagicMock()
            obs.frame = np.expand_dims(grid_out, 0)
            obs.state = GameState.NOT_FINISHED
            obs.levels_completed = 0
            return obs

        mock_env = MagicMock()
        mock_env.step = mock_step
        obs0 = MagicMock()
        obs0.frame = np.expand_dims(grid, 0)
        obs0.state = GameState.NOT_FINISHED
        obs0.levels_completed = 0
        mock_env.reset.return_value = obs0

        solver = PerGameSolver(_make_profile("click"), arcade=MagicMock())
        positions = solver._scan_effective_positions(mock_env, replay_sequence=[])

        # Two distant regions → at least 2 groups, at most 6
        assert 2 <= len(positions) <= 6
        # Positions from region 1 and region 2 should both be present
        region1 = [p for p in positions if 6 <= p[0] <= 14 and 6 <= p[1] <= 14]
        region2 = [p for p in positions if 38 <= p[0] <= 46 and 38 <= p[1] <= 46]
        assert len(region1) >= 1
        assert len(region2) >= 1

    def test_max_six_groups(self):
        """Scanner should return at most 6 groups."""
        from arcengine.enums import GameState

        grid = np.zeros((64, 64), dtype=np.int8)

        def mock_step(action, data=None):
            x = data.get("x", 0) if data else 0
            grid_out = grid.copy()
            # Every 8-pixel column is a different "valve" with a unique diff
            col_group = x // 8
            diff_size = (col_group + 1) * 10
            grid_out[10 : 10 + diff_size, 0] = col_group + 1
            obs = MagicMock()
            obs.frame = np.expand_dims(grid_out, 0)
            obs.state = GameState.NOT_FINISHED
            obs.levels_completed = 0
            return obs

        mock_env = MagicMock()
        mock_env.step = mock_step
        obs0 = MagicMock()
        obs0.frame = np.expand_dims(grid, 0)
        obs0.state = GameState.NOT_FINISHED
        obs0.levels_completed = 0
        mock_env.reset.return_value = obs0

        solver = PerGameSolver(_make_profile("click"), arcade=MagicMock())
        positions = solver._scan_effective_positions(mock_env, replay_sequence=[])

        assert len(positions) <= 6


class TestSequenceClickStrategy:
    def test_finds_winning_sequence(self):
        """BFS should find a click sequence that solves the level."""
        from arcengine.enums import GameState

        grid = np.zeros((64, 64), dtype=np.int8)
        grid[0, :] = 7  # bar

        click_history = []

        def mock_step(action, data=None):
            x = data.get("x", 0) if data else 0
            y = data.get("y", 0) if data else 0
            click_history.append((x, y))
            grid_out = grid.copy()
            grid_out[0, 63] = 4  # bar change

            # Count effective clicks (at valve position 10,10)
            effective = sum(1 for cx, cy in click_history if 8 <= cx <= 12 and 8 <= cy <= 12)

            obs = MagicMock()
            if effective >= 3:
                obs.levels_completed = 1
            else:
                obs.levels_completed = 0
                # Show some grid change for effective clicks
                if 8 <= x <= 12 and 8 <= y <= 12:
                    grid_out[20:30, 20:30] = effective

            obs.frame = np.expand_dims(grid_out, 0)
            obs.state = GameState.NOT_FINISHED
            return obs

        mock_env = MagicMock()

        def mock_reset():
            click_history.clear()
            obs = MagicMock()
            obs.frame = np.expand_dims(grid, 0)
            obs.state = GameState.NOT_FINISHED
            obs.levels_completed = 0
            return obs

        mock_env.reset = mock_reset
        mock_env.step = mock_step

        mock_arcade = MagicMock()
        mock_arcade.make.return_value = mock_env

        profile = _make_profile("click")
        solver = PerGameSolver(profile, arcade=mock_arcade)
        outcome = solver._execute_sequence_click(max_actions=200)

        assert outcome.won is True
        assert outcome.levels_solved >= 1

    def test_returns_false_when_no_effective_positions(self):
        """Should return won=False when scan finds no effective clicks."""
        from arcengine.enums import GameState

        grid = np.zeros((64, 64), dtype=np.int8)

        def mock_step(action, data=None):
            obs = MagicMock()
            obs.frame = np.expand_dims(grid, 0)
            obs.state = GameState.NOT_FINISHED
            obs.levels_completed = 0
            return obs

        mock_env = MagicMock()
        mock_env.reset.return_value = MagicMock(
            frame=np.expand_dims(grid, 0),
            state=GameState.NOT_FINISHED,
            levels_completed=0,
        )
        mock_env.step = mock_step

        mock_arcade = MagicMock()
        mock_arcade.make.return_value = mock_env

        solver = PerGameSolver(_make_profile("click"), arcade=mock_arcade)
        outcome = solver._execute_sequence_click(max_actions=200)

        assert outcome.won is False
        assert outcome.steps == 0
