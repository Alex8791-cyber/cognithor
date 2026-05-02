# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-12 — GameProfile persistence tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cognithor.channels.program_synthesis.arc_agi3.game_profile import (
    GameProfile,
    StrategyMetrics,
)
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)

if TYPE_CHECKING:
    from pathlib import Path


def _make_profile(game_id: str = "ls20") -> GameProfile:
    return GameProfile(
        game_id=game_id,
        game_type="keyboard",
        available_actions=[0, 1, 2, 3, 4],
        click_zones=[],
        target_colors=[10, 11],
        movement_effects={1: "up", 2: "down", 3: "left", 4: "right"},
        win_condition="reach_door",
        vision_description="LockSmith level",
        vision_strategy="rotate key, navigate to door",
        strategy_metrics={},
    )


class TestStrategyMetrics:
    def test_win_rate_zero_attempts(self) -> None:
        m = StrategyMetrics()
        assert m.win_rate == 0.0

    def test_win_rate_basic(self) -> None:
        m = StrategyMetrics(attempts=10, wins=4)
        assert m.win_rate == 0.4

    def test_default_zero_fields(self) -> None:
        m = StrategyMetrics()
        assert m.attempts == 0
        assert m.wins == 0


class TestGameProfileBasics:
    def test_construct(self) -> None:
        p = _make_profile()
        assert p.game_id == "ls20"
        assert p.game_type == "keyboard"
        assert p.movement_effects[1] == "up"

    def test_to_dict_then_from_dict(self) -> None:
        p = _make_profile()
        p.update_run(score=3)
        p.update_metrics("hybrid", won=True, levels_solved=2, steps=20, budget_ratio=0.5)
        round_tripped = GameProfile.from_dict(p.to_dict())
        assert round_tripped.game_id == p.game_id
        assert round_tripped.total_runs == 1
        assert round_tripped.best_score == 3
        assert "hybrid" in round_tripped.strategy_metrics
        assert round_tripped.strategy_metrics["hybrid"].wins == 1


class TestPersistence:
    def test_save_and_load(self, tmp_path: Path) -> None:
        p = _make_profile()
        p.update_run(score=5)
        p.save(base_dir=tmp_path)

        # Profile file should exist.
        assert (tmp_path / "game_profiles" / "ls20.json").exists()

        loaded = GameProfile.load("ls20", base_dir=tmp_path)
        assert loaded is not None
        assert loaded.game_id == "ls20"
        assert loaded.best_score == 5

    def test_load_missing_returns_none(self, tmp_path: Path) -> None:
        loaded = GameProfile.load("nonexistent", base_dir=tmp_path)
        assert loaded is None

    def test_exists(self, tmp_path: Path) -> None:
        assert GameProfile.exists("ls20", base_dir=tmp_path) is False
        p = _make_profile()
        p.save(base_dir=tmp_path)
        assert GameProfile.exists("ls20", base_dir=tmp_path) is True


class TestMetricsUpdate:
    def test_update_run_increments(self) -> None:
        p = _make_profile()
        p.update_run(score=2)
        p.update_run(score=5)
        p.update_run(score=3)
        assert p.total_runs == 3
        assert p.best_score == 5

    def test_update_metrics_creates_entry(self) -> None:
        p = _make_profile()
        p.update_metrics("cluster_click", won=True, levels_solved=2, steps=15, budget_ratio=0.3)
        assert "cluster_click" in p.strategy_metrics
        assert p.strategy_metrics["cluster_click"].attempts == 1
        assert p.strategy_metrics["cluster_click"].wins == 1
        assert p.strategy_metrics["cluster_click"].avg_steps_to_win == 15.0

    def test_avg_steps_running_average(self) -> None:
        p = _make_profile()
        p.update_metrics("a", won=True, levels_solved=1, steps=10, budget_ratio=0.5)
        p.update_metrics("a", won=True, levels_solved=1, steps=20, budget_ratio=0.5)
        # (10 + 20) / 2 = 15
        assert abs(p.strategy_metrics["a"].avg_steps_to_win - 15.0) < 1e-9

    def test_ranked_strategies_orders_by_win_rate(self) -> None:
        p = _make_profile()
        p.update_metrics("low_winner", won=False, levels_solved=0, steps=10, budget_ratio=0.5)
        p.update_metrics("low_winner", won=False, levels_solved=0, steps=10, budget_ratio=0.5)
        p.update_metrics("high_winner", won=True, levels_solved=2, steps=20, budget_ratio=0.5)
        ranked = p.ranked_strategies()
        # high_winner has 1.0 win rate, low_winner has 0.0
        assert ranked[0] == "high_winner"


class TestDefaultStrategies:
    def test_click_with_toggles(self) -> None:
        p = _make_profile()
        p.game_type = "click"
        p.has_toggles = True
        defaults = p.default_strategies()
        assert defaults[0][0] == "cluster_click"

    def test_click_without_toggles(self) -> None:
        p = _make_profile()
        p.game_type = "click"
        p.has_toggles = False
        defaults = p.default_strategies()
        assert defaults[0][0] == "sequence_click"

    def test_keyboard_game(self) -> None:
        p = _make_profile()
        p.game_type = "keyboard"
        defaults = p.default_strategies()
        assert defaults[0][0] == "keyboard_explore"
