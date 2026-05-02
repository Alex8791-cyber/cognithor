# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-12 PR-5+11 — FrameAnalyzer wiring + level-transition resets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cognithor.channels.program_synthesis.arc_agi3.dsl_agent import Sprint10DSLAgent
from cognithor.channels.program_synthesis.arc_agi3.frame_analyzer import (
    FrameAnalyzer,
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


def _frame(grid: np.ndarray, levels_completed: int = 0) -> _StubFrame:
    actions = [
        _StubAction(name="RESET", value=0),
        _StubAction(name="ACTION1", value=1),
        _StubAction(name="ACTION2", value=2),
    ]
    return _StubFrame(
        frame=[grid],
        available_actions=actions,
        levels_completed=levels_completed,
    )


class TestFrameAnalyzerWiring:
    def test_analyzer_optional(self) -> None:
        agent = Sprint10DSLAgent()  # no analyzer
        assert agent.frame_analyzer is None

    def test_analyzer_receives_frames(self) -> None:
        fa = FrameAnalyzer()
        agent = Sprint10DSLAgent(frame_analyzer=fa)
        # Frame 1: grid full of 1s.
        f1 = _frame(np.array([[1, 1], [1, 1]], dtype=np.int8))
        agent.choose_action([f1], f1)
        # Frame 2: those flipped to 2 → analyzer sees the change tagged
        # with the action chosen on frame 1.
        f2 = _frame(np.array([[2, 2], [2, 2]], dtype=np.int8))
        agent.choose_action([f2], f2)
        # Analyzer recorded one observation tagged with f1's chosen action.
        summary = fa.get_action_summary()
        assert summary  # non-empty
        # The recorded action-name should be one the agent actually picked.
        recorded_actions = set(summary.keys())
        assert recorded_actions <= {"RESET", "ACTION1", "ACTION2"}


class TestLevelTransitionReset:
    def test_no_transition_no_reset(self) -> None:
        agent = Sprint10DSLAgent()
        for _ in range(3):
            f = _frame(np.array([[1, 0], [0, 1]], dtype=np.int8), levels_completed=0)
            agent.choose_action([f], f)
        # 3 calls, all same level → 2 memory entries (first call doesn't
        # record because pending_action_name is None then).
        assert len(agent.memory) == 2

    def test_level_increment_clears_memory(self) -> None:
        agent = Sprint10DSLAgent()
        # Build up some level-0 memory.
        for _ in range(3):
            f = _frame(np.array([[1, 0], [0, 1]], dtype=np.int8), levels_completed=0)
            agent.choose_action([f], f)
        assert len(agent.memory) == 2

        # Level 1 — same shape, but levels_completed went up. Memory cleared.
        f = _frame(np.array([[2, 2], [2, 2]], dtype=np.int8), levels_completed=1)
        agent.choose_action([f], f)
        # After the transition the memory was cleared, then this call
        # didn't append (pending_action_name was reset to None). So len == 0.
        assert len(agent.memory) == 0

    def test_level_increment_clears_state_counter(self) -> None:
        agent = Sprint10DSLAgent()
        for _ in range(3):
            f = _frame(np.array([[1, 0]], dtype=np.int8), levels_completed=0)
            agent.choose_action([f], f)
        # Counter has at least one entry.
        first_state_hash = next(iter(agent.state_counter._counts.keys()))
        assert agent.state_counter.count(first_state_hash, "ACTION1") >= 0

        # Level transition — counter cleared.
        f = _frame(np.array([[1, 0]], dtype=np.int8), levels_completed=1)
        agent.choose_action([f], f)
        # The counter dict was cleared, then the new state's increment
        # was added. So only the new state has any count.
        assert len(agent.state_counter._counts) == 1

    def test_frame_analyzer_keeps_action_effects_across_levels(self) -> None:
        fa = FrameAnalyzer()
        agent = Sprint10DSLAgent(frame_analyzer=fa)

        # Level 0: train an action effect.
        f1 = _frame(np.array([[1, 1], [1, 1]], dtype=np.int8), levels_completed=0)
        agent.choose_action([f1], f1)
        f2 = _frame(np.array([[2, 2], [2, 2]], dtype=np.int8), levels_completed=0)
        agent.choose_action([f2], f2)
        before_summary = fa.get_action_summary()
        assert before_summary

        # Level 1 — analyzer reset_for_new_level was called BUT
        # action_effects were preserved.
        f3 = _frame(np.array([[3, 3]], dtype=np.int8), levels_completed=1)
        agent.choose_action([f3], f3)
        after_summary = fa.get_action_summary()
        # Effects from level 0 still present.
        assert set(after_summary.keys()) >= set(before_summary.keys())
