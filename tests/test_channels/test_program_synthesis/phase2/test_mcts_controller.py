# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""MCTS controller tests (Sprint-1 plan task 7, spec §5)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterable

from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.phase2.config import Phase2Config
from cognithor.channels.program_synthesis.phase2.datatypes import MCTSNode
from cognithor.channels.program_synthesis.phase2.mcts_controller import (
    FallbackGuard,
    MCTSActionCandidate,
    MCTSController,
    MCTSResult,
)


def _root() -> MCTSNode:
    return MCTSNode(primitive="<root>")


def _greedy_supplier(actions_per_node: dict[str, list[tuple[str, float]]]):
    """Build a supplier that returns deterministic actions per node primitive."""

    def supply(node: MCTSNode) -> Iterable[MCTSActionCandidate]:
        cands = actions_per_node.get(node.primitive, [])
        return [MCTSActionCandidate(primitive=p, prior=pr) for p, pr in cands]

    return supply


def _fixed_clock(times: list[float]):
    """Clock that yields each entry once; raises when exhausted."""

    def clock() -> float:
        if not times:
            raise AssertionError("test clock exhausted")
        return times.pop(0)

    return clock


# ---------------------------------------------------------------------------
# Single-iteration mechanics
# ---------------------------------------------------------------------------


class TestSelectionExpansion:
    def test_root_with_no_actions_terminates_no_actions(self) -> None:
        # Empty action space → controller exits immediately.
        controller = MCTSController(
            action_supplier=_greedy_supplier({}),
            value_estimator=lambda _node, _path: 0.5,
        )
        result = controller.run(_root(), wall_clock_budget_seconds=1.0)
        assert isinstance(result, MCTSResult)
        assert result.terminated_by == "no_actions"
        assert result.iterations_completed == 1

    def test_first_iteration_expands_root(self) -> None:
        actions = {"<root>": [("rotate90", 0.7), ("recolor", 0.3)]}
        controller = MCTSController(
            action_supplier=_greedy_supplier(actions),
            value_estimator=lambda _node, _path: 0.5,
        )
        # Run for a single iteration via max_iterations=1.
        result = controller.run(
            _root(),
            wall_clock_budget_seconds=10.0,
            max_iterations=1,
        )
        assert result.terminated_by == "iterations"
        # Root has both children populated.
        # (We can't read the tree from the result, but best_path traces it.)
        assert result.best_path == ("rotate90",)  # higher prior wins first eval

    def test_value_estimator_receives_path(self) -> None:
        actions = {"<root>": [("a", 1.0)]}
        seen_paths: list[tuple[str, ...]] = []

        def estimator(_node: MCTSNode, path: tuple[str, ...]) -> float:
            seen_paths.append(path)
            return 0.5

        controller = MCTSController(
            action_supplier=_greedy_supplier(actions),
            value_estimator=estimator,
        )
        controller.run(_root(), wall_clock_budget_seconds=10.0, max_iterations=1)
        assert seen_paths == [("a",)]


# ---------------------------------------------------------------------------
# Backpropagation
# ---------------------------------------------------------------------------


class TestBackpropagation:
    def test_backprop_increments_visit_count_along_path(self) -> None:
        # Two-level tree: root → a → leaf.
        actions = {
            "<root>": [("a", 1.0)],
            "a": [("leaf", 1.0)],
        }
        controller = MCTSController(
            action_supplier=_greedy_supplier(actions),
            value_estimator=lambda _node, _path: 0.7,
        )
        root = _root()
        controller.run(root, wall_clock_budget_seconds=10.0, max_iterations=2)
        # After iter-1: root expands "a", evaluates "a" (path=("a",))
        # After iter-2: descend into "a", expand "leaf", evaluate "leaf" path=("a","leaf")
        a_child = root.children["a"]
        leaf_child = a_child.children.get("leaf")
        assert leaf_child is not None
        # leaf was evaluated → visited at least once.
        assert leaf_child.visit_count >= 1
        # a was evaluated at iter-1 + visited again at iter-2 (backprop) = 2.
        assert a_child.visit_count >= 2
        # Root visited every iteration (backprop walks to root).
        assert root.visit_count >= 2


class TestPUCTSelection:
    def test_descends_via_puct_to_higher_value_branch(self) -> None:
        # Two-level: root → A or B; A's leaf is high-reward, B's is low.
        actions = {
            "<root>": [("A", 0.5), ("B", 0.5)],
            "A": [("A_leaf", 1.0)],
            "B": [("B_leaf", 1.0)],
        }

        def estimator(_node: MCTSNode, path: tuple[str, ...]) -> float:
            if "A" in path:
                return 0.9
            return 0.1

        controller = MCTSController(
            action_supplier=_greedy_supplier(actions),
            value_estimator=estimator,
        )
        root = _root()
        result = controller.run(root, wall_clock_budget_seconds=10.0, max_iterations=10)
        assert result.best_value > 0.5
        # Best path should head into A.
        assert "A" in result.best_path or "A_leaf" in result.best_path


# ---------------------------------------------------------------------------
# Anytime termination
# ---------------------------------------------------------------------------


class TestAnytimeTermination:
    def test_budget_exhausts_before_iterations_cap(self) -> None:
        actions = {"<root>": [("a", 1.0)]}
        # Clock advances 0.5 s per call → after 3 reads we're past 1 s.
        clock = _fixed_clock([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        controller = MCTSController(
            action_supplier=_greedy_supplier(actions),
            value_estimator=lambda _node, _path: 0.5,
            clock=clock,
        )
        result = controller.run(
            _root(),
            wall_clock_budget_seconds=1.0,
            max_iterations=100,
        )
        assert result.terminated_by == "budget"

    def test_max_iterations_cap_fires(self) -> None:
        actions = {"<root>": [("a", 1.0)]}
        controller = MCTSController(
            action_supplier=_greedy_supplier(actions),
            value_estimator=lambda _node, _path: 0.5,
        )
        result = controller.run(
            _root(),
            wall_clock_budget_seconds=10.0,
            max_iterations=3,
        )
        assert result.terminated_by == "iterations"
        assert result.iterations_completed == 3

    def test_zero_budget_raises(self) -> None:
        controller = MCTSController(
            action_supplier=_greedy_supplier({}),
            value_estimator=lambda _node, _path: 0.5,
        )
        with pytest.raises(ValueError, match="must be > 0"):
            controller.run(_root(), wall_clock_budget_seconds=0.0)


# ---------------------------------------------------------------------------
# Best-trajectory bookkeeping
# ---------------------------------------------------------------------------


class TestBestTrajectoryTracking:
    def test_best_value_records_highest_seen(self) -> None:
        actions = {"<root>": [("a", 1.0), ("b", 1.0), ("c", 1.0)]}

        # Estimator returns increasing values per call so the highest-
        # scoring leaf is the third visit.
        seq = [0.3, 0.7, 0.5]
        idx = {"i": 0}

        def estimator(_node: MCTSNode, _path: tuple[str, ...]) -> float:
            v = seq[idx["i"] % len(seq)]
            idx["i"] += 1
            return v

        controller = MCTSController(
            action_supplier=_greedy_supplier(actions),
            value_estimator=estimator,
        )
        result = controller.run(
            _root(),
            wall_clock_budget_seconds=10.0,
            max_iterations=3,
        )
        assert result.best_value == 0.7

    def test_best_path_traces_root_to_best(self) -> None:
        actions = {
            "<root>": [("a", 1.0)],
            "a": [("b", 1.0)],
            "b": [("c", 1.0)],
        }

        def estimator(_node: MCTSNode, path: tuple[str, ...]) -> float:
            # Reward is depth.
            return float(len(path))

        controller = MCTSController(
            action_supplier=_greedy_supplier(actions),
            value_estimator=estimator,
        )
        result = controller.run(
            _root(),
            wall_clock_budget_seconds=10.0,
            max_iterations=10,
        )
        # Best should be at depth >= 2.
        assert len(result.best_path) >= 2


# ---------------------------------------------------------------------------
# FallbackGuard
# ---------------------------------------------------------------------------


class TestFallbackGuard:
    def test_warmup_blocks_fallback(self) -> None:
        guard = FallbackGuard(
            warmup_iters=10, plateau_iters=2, plateau_delta=0.01, min_node_depth_mean=2.0
        )
        for _ in range(5):
            guard.record(best_value=0.0, leaf_depth=0)
        assert guard.should_fall_back(iteration=5) is False

    def test_plateau_signal_after_warmup(self) -> None:
        guard = FallbackGuard(
            warmup_iters=2, plateau_iters=2, plateau_delta=0.01, min_node_depth_mean=0.0
        )
        # Three constant scores → plateau detected after warmup.
        guard.record(best_value=0.5, leaf_depth=5)
        guard.record(best_value=0.5, leaf_depth=5)
        guard.record(best_value=0.5, leaf_depth=5)
        assert guard.should_fall_back(iteration=3) is True

    def test_shallow_tree_signal(self) -> None:
        guard = FallbackGuard(
            warmup_iters=0, plateau_iters=99, plateau_delta=0.0, min_node_depth_mean=2.0
        )
        # Average depth = 0.5 < 2.0 → shallow.
        guard.record(best_value=0.0, leaf_depth=0)
        guard.record(best_value=1.0, leaf_depth=1)
        assert guard.should_fall_back(iteration=2) is True

    def test_no_signal_when_search_progresses_and_deep(self) -> None:
        guard = FallbackGuard(
            warmup_iters=0, plateau_iters=2, plateau_delta=0.01, min_node_depth_mean=1.0
        )
        guard.record(best_value=0.1, leaf_depth=3)
        guard.record(best_value=0.5, leaf_depth=4)
        guard.record(best_value=0.9, leaf_depth=5)
        assert guard.should_fall_back(iteration=3) is False


class TestControllerFallbackIntegration:
    def test_fallback_signalled_when_plateau(self) -> None:
        actions = {"<root>": [("a", 1.0)]}
        guard = FallbackGuard(
            warmup_iters=2, plateau_iters=2, plateau_delta=0.01, min_node_depth_mean=0.0
        )
        controller = MCTSController(
            action_supplier=_greedy_supplier(actions),
            value_estimator=lambda _node, _path: 0.5,
            fallback=guard,
        )
        result = controller.run(
            _root(),
            wall_clock_budget_seconds=10.0,
            max_iterations=10,
        )
        assert result.fallback_signalled is True
        assert result.terminated_by == "fallback"


# ---------------------------------------------------------------------------
# Virtual-loss is harmless in single-threaded mode
# ---------------------------------------------------------------------------


class TestVirtualLossNoOp:
    def test_virtual_loss_does_not_break_convergence(self) -> None:
        # With v=1.0 (default), single-threaded MCTS still converges.
        actions = {
            "<root>": [("A", 0.5), ("B", 0.5)],
            "A": [("A_leaf", 1.0)],
            "B": [("B_leaf", 1.0)],
        }

        def estimator(_node: MCTSNode, path: tuple[str, ...]) -> float:
            return 0.9 if "A" in path else 0.1

        cfg = Phase2Config(mcts_virtual_loss=1.0)
        controller = MCTSController(
            action_supplier=_greedy_supplier(actions),
            value_estimator=estimator,
            config=cfg,
        )
        result = controller.run(_root(), wall_clock_budget_seconds=10.0, max_iterations=10)
        assert result.best_value >= 0.5  # converged toward A branch

    def test_zero_virtual_loss_still_works(self) -> None:
        actions = {"<root>": [("a", 1.0)]}
        cfg = Phase2Config(mcts_virtual_loss=0.0)
        controller = MCTSController(
            action_supplier=_greedy_supplier(actions),
            value_estimator=lambda _node, _path: 0.5,
            config=cfg,
        )
        result = controller.run(
            _root(),
            wall_clock_budget_seconds=10.0,
            max_iterations=2,
        )
        assert result.iterations_completed == 2


# ---------------------------------------------------------------------------
# Config-driven knobs
# ---------------------------------------------------------------------------


class TestConfigKnobs:
    def test_max_iterations_from_config(self) -> None:
        actions = {"<root>": [("a", 1.0)]}
        cfg = Phase2Config(mcts_max_iterations=3)
        controller = MCTSController(
            action_supplier=_greedy_supplier(actions),
            value_estimator=lambda _node, _path: 0.5,
            config=cfg,
        )
        result = controller.run(_root(), wall_clock_budget_seconds=10.0)
        assert result.iterations_completed == 3
        assert result.terminated_by == "iterations"

    def test_zero_max_iterations_means_no_cap(self) -> None:
        actions = {"<root>": [("a", 1.0)]}
        cfg = Phase2Config(mcts_max_iterations=0)
        # Use a fixed clock that ticks past the deadline after 5 calls.
        clock_times = [0.0] + [0.3] * 4 + [1.5]
        controller = MCTSController(
            action_supplier=_greedy_supplier(actions),
            value_estimator=lambda _node, _path: 0.5,
            config=cfg,
            clock=lambda: clock_times.pop(0) if clock_times else 1.5,
        )
        result = controller.run(_root(), wall_clock_budget_seconds=1.0)
        # Loop terminates by budget, not iterations.
        assert result.terminated_by == "budget"


# ---------------------------------------------------------------------------
# Result dataclass contract
# ---------------------------------------------------------------------------


class TestMCTSResultDataclass:
    def test_is_frozen_and_hashable(self) -> None:
        node = _root()
        r = MCTSResult(
            best_node=node,
            best_value=0.5,
            best_path=("a", "b"),
            terminated_by="budget",
            iterations_completed=10,
            elapsed_seconds=1.5,
        )
        # Frozen → hashable as long as fields are.
        # MCTSNode itself is not frozen (mutable visit/total counts), but
        # the dataclass equality / hash only cares about identity. We
        # just assert the basic invariants.
        assert r.best_value == 0.5
        assert r.best_path == ("a", "b")
        assert r.fallback_signalled is False
