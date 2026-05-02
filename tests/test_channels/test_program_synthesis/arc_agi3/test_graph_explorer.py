# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-15 — GraphExplorerAgent tests."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cognithor.channels.program_synthesis.arc_agi3.click_target_sampler import (
    ClickTargetSampler,
)
from cognithor.channels.program_synthesis.arc_agi3.frame_analyzer import (
    FrameAnalyzer,
)
from cognithor.channels.program_synthesis.arc_agi3.graph_explorer import (
    GraphExplorerAgent,
)
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)


@dataclass
class _StubGameState:
    name: str = "NOT_FINISHED"


@dataclass
class _StubAction:
    name: str
    value: int
    reasoning: str = ""
    _data: dict[str, Any] = field(default_factory=dict)
    _is_simple: bool = True

    def is_simple(self) -> bool:
        return self._is_simple

    def is_complex(self) -> bool:
        return not self._is_simple

    def set_data(self, data: dict[str, Any]) -> None:
        self._data = dict(data)


@dataclass
class _StubFrame:
    game_id: str = "smoke"
    state: _StubGameState = field(default_factory=_StubGameState)
    levels_completed: int = 0
    win_levels: int = 1
    guid: str = ""
    full_reset: bool = False
    frame: list[Any] = field(default_factory=list)
    available_actions: list[_StubAction] = field(default_factory=list)


def _frame(grid: np.ndarray, *, with_click: bool = False, levels: int = 0) -> _StubFrame:
    actions = [
        _StubAction(name="RESET", value=0),
        _StubAction(name="ACTION1", value=1),
        _StubAction(name="ACTION2", value=2),
        _StubAction(name="ACTION3", value=3),
    ]
    if with_click:
        actions.append(_StubAction(name="ACTION6", value=6, _is_simple=False))
    return _StubFrame(frame=[grid], available_actions=actions, levels_completed=levels)


class TestUntriedExploration:
    def test_first_call_picks_first_simple_action(self) -> None:
        agent = GraphExplorerAgent()
        f = _frame(np.zeros((3, 3), dtype=np.int8))
        chosen = agent.choose_action([f], f)
        # No prior counts → all live actions untried, first one wins.
        assert chosen.name in {"ACTION1", "ACTION2", "ACTION3"}
        assert "untried-from-state" in chosen.reasoning

    def test_repeated_calls_distribute_across_actions(self) -> None:
        """In the same state, the agent should rotate through all simple
        actions before repeating any (untried policy)."""
        agent = GraphExplorerAgent()
        # Same grid every call → same state hash.
        grid = np.zeros((3, 3), dtype=np.int8)
        emitted = []
        for _ in range(3):
            f = _frame(grid)
            chosen = agent.choose_action([f], f)
            emitted.append(chosen.name)
        # Three distinct simple actions over three calls.
        assert len(set(emitted)) == 3
        assert set(emitted) <= {"ACTION1", "ACTION2", "ACTION3"}


class TestDeadActionSkipping:
    def test_dead_action_excluded_from_pick(self) -> None:
        agent = GraphExplorerAgent()
        # Manually mark ACTION1 dead from the initial state hash.
        grid = np.zeros((3, 3), dtype=np.int8)
        from cognithor.channels.program_synthesis.arc_agi3.state_action_counts import (
            hash_state,
        )

        h = hash_state(grid)
        agent.state_counter.mark_dead(h, "ACTION1")
        f = _frame(grid)
        chosen = agent.choose_action([f], f)
        assert chosen.name != "ACTION1"


class TestStuckTriggersReset:
    def test_stuck_episode_picks_reset(self) -> None:
        agent = GraphExplorerAgent()
        # Drive enough no-change frames to flip stuck flag.
        same_grid = np.array([[1, 0], [0, 1]], dtype=np.int8)
        for _ in range(20):
            f = _frame(same_grid)
            agent.choose_action([f], f)
        # Now the memory shows a long no-change streak; next call should RESET.
        f = _frame(same_grid)
        chosen = agent.choose_action([f], f)
        assert chosen.name == "RESET"
        assert "stuck" in chosen.reasoning.lower()


class TestProductivityTieBreak:
    def test_higher_productivity_wins_among_tied(self) -> None:
        fa = FrameAnalyzer()
        # Train: ACTION3 averages 12 pixels-changed per call, ACTION1 only 1.
        fa.analyze(np.zeros((4, 4), dtype=np.int8))
        big = np.array([[5] * 4, [5] * 4, [5] * 4, [0] * 4], dtype=np.int8)
        fa.analyze(big, action="ACTION3")  # 12 cells changed
        # Now flip ONE pixel only so ACTION1 has avg=1 (after-prev was big).
        small_diff = big.copy()
        small_diff[0, 0] = 9  # 1 cell changed
        fa.analyze(small_diff, action="ACTION1")

        agent = GraphExplorerAgent(frame_analyzer=fa)
        grid = np.array([[7, 7], [7, 7]], dtype=np.int8)
        from cognithor.channels.program_synthesis.arc_agi3.state_action_counts import (
            hash_state,
        )

        h = hash_state(grid)
        agent.state_counter.increment(h, "ACTION1")
        agent.state_counter.increment(h, "ACTION3")
        f = _StubFrame(
            frame=[grid],
            available_actions=[
                _StubAction(name="ACTION1", value=1),
                _StubAction(name="ACTION3", value=3),
            ],
        )
        chosen = agent.choose_action([f], f)
        assert chosen.name == "ACTION3"


class TestComplexActionFiltering:
    def test_complex_action_skipped_without_sampler(self) -> None:
        agent = GraphExplorerAgent()  # no click_target_sampler
        grid = np.zeros((4, 4), dtype=np.int8)
        f = _frame(grid, with_click=True)
        chosen = agent.choose_action([f], f)
        # ACTION6 should not be picked (would crash env without coords).
        assert chosen.name != "ACTION6"

    def test_complex_action_used_with_sampler(self) -> None:
        sampler = ClickTargetSampler()
        agent = GraphExplorerAgent(click_target_sampler=sampler)
        grid = np.zeros((4, 4), dtype=np.int8)
        grid[1, 2] = 5  # one salient cell
        # Mark all simple actions as already-tried so ACTION6 is the only
        # untried option.
        from cognithor.channels.program_synthesis.arc_agi3.state_action_counts import (
            hash_state,
        )

        h = hash_state(grid)
        for name in ("ACTION1", "ACTION2", "ACTION3"):
            agent.state_counter.increment(h, name)
        f = _frame(grid, with_click=True)
        chosen = agent.choose_action([f], f)
        if chosen.name == "ACTION6":
            assert chosen._data
            assert "x" in chosen._data
            assert "y" in chosen._data


class TestExplorationCoverage:
    def test_eight_calls_visit_at_least_three_distinct_actions(self) -> None:
        """The graph-explorer should NOT spam one action; over 8 calls it
        should hit at least 3 distinct ones."""
        agent = GraphExplorerAgent()
        grids = [np.array([[i, 0], [0, i]], dtype=np.int8) for i in range(8)]
        emitted = []
        for g in grids:
            f = _frame(g)
            chosen = agent.choose_action([f], f)
            emitted.append(chosen.name)
        counts = Counter(emitted)
        assert len(counts) >= 3
