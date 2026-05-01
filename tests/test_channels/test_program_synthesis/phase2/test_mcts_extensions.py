# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""MCTS extensions tests (Sprint-2 Track E — parallel + restart + diversity)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.phase2.config import Phase2Config
from cognithor.channels.program_synthesis.phase2.datatypes import MCTSNode
from cognithor.channels.program_synthesis.phase2.mcts_controller import (
    MCTSActionCandidate,
)
from cognithor.channels.program_synthesis.phase2.mcts_extensions import (
    ParallelMCTSDriver,
    ParallelMCTSResult,
    RestartController,
    apply_diversity_bonus,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


def _root() -> MCTSNode:
    return MCTSNode(primitive="<root>")


def _supplier_with(actions: dict[str, list[tuple[str, float]]]):
    def supply(node: MCTSNode) -> Iterable[MCTSActionCandidate]:
        return [
            MCTSActionCandidate(primitive=p, prior=pr) for p, pr in actions.get(node.primitive, [])
        ]

    return supply


# ---------------------------------------------------------------------------
# Parallel driver
# ---------------------------------------------------------------------------


class TestParallelMCTSDriver:
    @pytest.mark.asyncio
    async def test_runs_n_workers(self) -> None:
        actions = {"<root>": [("a", 1.0)]}
        cfg = Phase2Config(mcts_parallelism_workers=4, mcts_max_iterations=3)
        driver = ParallelMCTSDriver(
            action_supplier=_supplier_with(actions),
            value_estimator=lambda _n, _p: 0.5,
            config=cfg,
        )
        result = await driver.run(_root, wall_clock_budget_seconds=10.0)
        assert isinstance(result, ParallelMCTSResult)
        assert result.workers_used == 4
        assert len(result.per_worker) == 4

    @pytest.mark.asyncio
    async def test_picks_highest_value_worker(self) -> None:
        actions = {"<root>": [("a", 1.0)]}
        # Vary value per call so different workers see different rewards.
        # We use a counter that increments per call — workers run in
        # parallel so the highest-value worker wins.
        counter = {"i": 0}

        def estimator(_n: MCTSNode, _p: tuple[str, ...]) -> float:
            counter["i"] += 1
            # Cycle: 0.1, 0.5, 0.9, 0.3 — best is 0.9.
            return [0.1, 0.5, 0.9, 0.3][counter["i"] % 4]

        cfg = Phase2Config(mcts_parallelism_workers=4, mcts_max_iterations=1)
        driver = ParallelMCTSDriver(
            action_supplier=_supplier_with(actions),
            value_estimator=estimator,
            config=cfg,
        )
        result = await driver.run(_root, wall_clock_budget_seconds=10.0)
        # Best value across all workers.
        assert result.best.best_value == max(r.best_value for r in result.per_worker)

    @pytest.mark.asyncio
    async def test_default_workers_is_one(self) -> None:
        actions = {"<root>": [("a", 1.0)]}
        cfg = Phase2Config()  # default workers = 1
        driver = ParallelMCTSDriver(
            action_supplier=_supplier_with(actions),
            value_estimator=lambda _n, _p: 0.5,
            config=cfg,
        )
        result = await driver.run(_root, wall_clock_budget_seconds=10.0, max_iterations=1)
        assert result.workers_used == 1
        assert len(result.per_worker) == 1

    @pytest.mark.asyncio
    async def test_root_factory_freshness(self) -> None:
        # Each worker must get a fresh root — otherwise tree state
        # would leak across workers and break correctness.
        seen_ids: list[int] = []

        def factory() -> MCTSNode:
            r = _root()
            seen_ids.append(id(r))
            return r

        cfg = Phase2Config(mcts_parallelism_workers=3, mcts_max_iterations=1)
        driver = ParallelMCTSDriver(
            action_supplier=_supplier_with({"<root>": [("a", 1.0)]}),
            value_estimator=lambda _n, _p: 0.5,
            config=cfg,
        )
        await driver.run(factory, wall_clock_budget_seconds=10.0)
        assert len(seen_ids) == 3
        assert len(set(seen_ids)) == 3  # three distinct objects


# ---------------------------------------------------------------------------
# Restart controller
# ---------------------------------------------------------------------------


class TestRestartController:
    @pytest.mark.asyncio
    async def test_no_restart_when_score_threshold_zero(self) -> None:
        # mcts_restart_score_threshold=0.0 means "score is never below
        # threshold" (since values are >= 0) → restart never fires
        # regardless of how much budget was used. Avoids platform clock-
        # resolution flake (Windows 16ms ticks make elapsed_seconds=0.0).
        actions = {"<root>": [("a", 1.0)]}
        cfg = Phase2Config(
            mcts_restart_budget_fraction=0.3,
            mcts_restart_score_threshold=0.0,
            mcts_max_iterations=1,
        )
        rc = RestartController(
            action_supplier=_supplier_with(actions),
            value_estimator=lambda _n, _p: 0.1,
            config=cfg,
        )
        await rc.run(_root(), wall_clock_budget_seconds=10.0)
        assert rc.restarts_triggered == 0

    @pytest.mark.asyncio
    async def test_restart_fires_on_early_low_score_plateau(self) -> None:
        # Simulate: search finishes very fast (below 30% of budget)
        # AND best_value is below 0.5 → next run should bump c_puct.
        actions = {"<root>": [("a", 1.0)]}
        cfg = Phase2Config(
            mcts_restart_budget_fraction=0.3,
            mcts_restart_score_threshold=0.5,
            mcts_restart_c_puct_multiplier=1.5,
            mcts_c_puct=2.0,
            mcts_max_iterations=1,
        )
        rc = RestartController(
            action_supplier=_supplier_with(actions),
            value_estimator=lambda _n, _p: 0.1,  # below 0.5 threshold
            config=cfg,
        )
        result = await rc.run(_root(), wall_clock_budget_seconds=10.0)
        # After 1 iteration on a 10 s budget, elapsed << 30% → restart fires.
        assert result.elapsed_seconds < 3.0
        assert rc.restarts_triggered == 1

    @pytest.mark.asyncio
    async def test_subsequent_run_uses_bumped_c_puct(self) -> None:
        actions = {"<root>": [("a", 1.0)]}
        cfg = Phase2Config(
            mcts_restart_budget_fraction=0.3,
            mcts_restart_score_threshold=0.5,
            mcts_restart_c_puct_multiplier=1.5,
            mcts_c_puct=2.0,
            mcts_max_iterations=1,
        )
        rc = RestartController(
            action_supplier=_supplier_with(actions),
            value_estimator=lambda _n, _p: 0.1,
            config=cfg,
        )
        await rc.run(_root(), wall_clock_budget_seconds=10.0)
        assert rc.last_c_puct == 2.0
        # Second run uses the bumped c_puct.
        await rc.run(_root(), wall_clock_budget_seconds=10.0)
        assert rc.last_c_puct == 3.0  # 2.0 × 1.5

    @pytest.mark.asyncio
    async def test_no_restart_when_score_high(self) -> None:
        # Even with early termination, high score → no restart.
        actions = {"<root>": [("a", 1.0)]}
        cfg = Phase2Config(
            mcts_restart_budget_fraction=0.3,
            mcts_restart_score_threshold=0.5,
            mcts_max_iterations=1,
        )
        rc = RestartController(
            action_supplier=_supplier_with(actions),
            value_estimator=lambda _n, _p: 0.9,  # well above 0.5
            config=cfg,
        )
        await rc.run(_root(), wall_clock_budget_seconds=10.0)
        assert rc.restarts_triggered == 0


# ---------------------------------------------------------------------------
# Diversity bonus
# ---------------------------------------------------------------------------


class TestDiversityBonus:
    def test_no_visited_paths_returns_unchanged(self) -> None:
        cands = [
            MCTSActionCandidate(primitive="rotate90", prior=0.5),
            MCTSActionCandidate(primitive="recolor", prior=0.5),
        ]
        adjusted = apply_diversity_bonus(cands, [])
        assert adjusted == tuple(cands)

    def test_dampens_priors_for_similar_primitives(self) -> None:
        cands = [
            MCTSActionCandidate(primitive="rotate90", prior=0.8),
            MCTSActionCandidate(primitive="completely_different_long_name", prior=0.8),
        ]
        # Already visited rotate180 — rotate90 is very similar (low edit dist),
        # the long-name primitive is far apart.
        visited = [("rotate180",)]
        cfg = Phase2Config(mcts_diversity_bonus_lambda=0.5)
        adjusted = apply_diversity_bonus(cands, visited, config=cfg)
        # rotate90 dampened more than the long-name one.
        rotate90_adj = next(c for c in adjusted if c.primitive == "rotate90")
        long_adj = next(c for c in adjusted if c.primitive.startswith("completely"))
        assert rotate90_adj.prior < long_adj.prior

    def test_lambda_zero_disables(self) -> None:
        cands = [MCTSActionCandidate(primitive="rotate90", prior=0.5)]
        cfg = Phase2Config(mcts_diversity_bonus_lambda=0.0)
        adjusted = apply_diversity_bonus(cands, [("rotate180",)], config=cfg)
        assert adjusted[0].prior == 0.5

    def test_identical_primitive_max_penalty(self) -> None:
        cands = [MCTSActionCandidate(primitive="rotate90", prior=1.0)]
        cfg = Phase2Config(mcts_diversity_bonus_lambda=1.0)
        # Identical token in visited paths → similarity 1.0 → full penalty.
        adjusted = apply_diversity_bonus(cands, [("rotate90",)], config=cfg)
        assert adjusted[0].prior == 0.0

    def test_priors_clamped_to_zero(self) -> None:
        cands = [MCTSActionCandidate(primitive="rotate90", prior=0.5)]
        cfg = Phase2Config(mcts_diversity_bonus_lambda=1.0)
        adjusted = apply_diversity_bonus(cands, [("rotate90",)], config=cfg)
        # Penalty 0.5 × 1.0 = 0.5; new prior 0.5 × (1 - 0.5) = 0.25. Wait —
        # actually penalty should fully zero an exact match. Re-check.
        # Similarity = 1 - (edit_dist / max_len) = 1 - 0 = 1.0.
        # penalty = 1.0 · 1.0 = 1.0. prior · (1 - 1) = 0.
        assert adjusted[0].prior == 0.0


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_invalid_workers_count_raises(self) -> None:
        with pytest.raises(ValueError, match="mcts_parallelism_workers"):
            Phase2Config(mcts_parallelism_workers=0)

    def test_invalid_restart_budget_fraction_raises(self) -> None:
        with pytest.raises(ValueError, match="mcts_restart_budget_fraction"):
            Phase2Config(mcts_restart_budget_fraction=1.5)

    def test_invalid_restart_multiplier_raises(self) -> None:
        with pytest.raises(ValueError, match="mcts_restart_c_puct_multiplier"):
            Phase2Config(mcts_restart_c_puct_multiplier=1.0)

    def test_invalid_diversity_lambda_raises(self) -> None:
        with pytest.raises(ValueError, match="mcts_diversity_bonus_lambda"):
            Phase2Config(mcts_diversity_bonus_lambda=-0.1)
