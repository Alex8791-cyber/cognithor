# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-11 Wave-4 — Sprint10DSLAgent + DSLActionDecoder tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from cognithor.channels.program_synthesis.arc_agi3.dsl_action_decoder import (
    DSLActionDecoder,
)
from cognithor.channels.program_synthesis.arc_agi3.dsl_agent import Sprint10DSLAgent
from cognithor.channels.program_synthesis.arc_agi3.episode_memory import (
    EpisodeMemory,
    StuckDetector,
)
from cognithor.channels.program_synthesis.arc_agi3.frame_bridge import FrameBridge
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
    game_id: str = "ls20"
    state: _StubGameState = field(default_factory=_StubGameState)
    levels_completed: int = 0
    win_levels: int = 1
    guid: str = ""
    full_reset: bool = False
    frame: list[Any] = field(default_factory=list)
    available_actions: list[_StubAction] = field(default_factory=list)


def _frame(grid: np.ndarray, **kwargs: Any) -> _StubFrame:
    actions = kwargs.pop(
        "available_actions",
        [
            _StubAction(name="RESET", value=0),
            _StubAction(name="ACTION1", value=1),
            _StubAction(name="ACTION2", value=2),
            _StubAction(name="ACTION3", value=3),
        ],
    )
    return _StubFrame(frame=[grid], available_actions=actions, **kwargs)


def _g(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int_)


# ---------------------------------------------------------------------------
# DSLActionDecoder
# ---------------------------------------------------------------------------


class TestDSLActionDecoder:
    def test_picks_least_tried_non_reset(self) -> None:
        memory = EpisodeMemory()
        # ACTION1 used 2x, ACTION2 used 1x. With ACTION3 also
        # available and never used (count=0), ACTION3 wins as
        # least-tried. Restrict to ACTION1+ACTION2 to test the
        # tie-break-on-min behaviour cleanly.
        np_grid = np.array([[1]], dtype=np.int8)
        memory.append(grid=np_grid, action_name="ACTION1", levels_completed=0)
        memory.append(grid=np_grid, action_name="ACTION1", levels_completed=0)
        memory.append(grid=np_grid, action_name="ACTION2", levels_completed=0)
        decoder = DSLActionDecoder(memory=memory)
        frame = _frame(
            _g([[1]]),
            available_actions=[
                _StubAction(name="RESET", value=0),
                _StubAction(name="ACTION1", value=1),
                _StubAction(name="ACTION2", value=2),
            ],
        )
        chosen = decoder.decode([frame], frame)
        # Without ACTION3 in the pool, the least-used non-RESET is ACTION2 (count 1).
        assert chosen.name == "ACTION2"

    def test_prefers_never_tried_action(self) -> None:
        """When some actions have count > 0 and one has count 0,
        the never-tried one wins."""
        memory = EpisodeMemory()
        np_grid = np.array([[1]], dtype=np.int8)
        memory.append(grid=np_grid, action_name="ACTION1", levels_completed=0)
        decoder = DSLActionDecoder(memory=memory)
        frame = _frame(
            _g([[1]]),
            available_actions=[
                _StubAction(name="ACTION1", value=1),
                _StubAction(name="ACTION3", value=3),
            ],
        )
        chosen = decoder.decode([frame], frame)
        assert chosen.name == "ACTION3"

    def test_resets_on_stuck(self) -> None:
        memory = EpisodeMemory()
        # 6 identical no-progress steps → stuck (default threshold 5).
        np_grid = np.array([[1]], dtype=np.int8)
        for _ in range(6):
            memory.append(grid=np_grid, action_name="ACTION1", levels_completed=0)
        decoder = DSLActionDecoder(memory=memory)
        frame = _frame(_g([[1]]))
        chosen = decoder.decode([frame], frame)
        assert chosen.name == "RESET"
        assert "stuck" in chosen.reasoning.lower()

    def test_does_not_reset_when_reset_unavailable(self) -> None:
        memory = EpisodeMemory()
        np_grid = np.array([[1]], dtype=np.int8)
        for _ in range(6):
            memory.append(grid=np_grid, action_name="ACTION1", levels_completed=0)
        decoder = DSLActionDecoder(memory=memory)
        # No RESET in available actions; must fall back to least-tried.
        frame = _frame(
            _g([[1]]),
            available_actions=[
                _StubAction(name="ACTION1", value=1),
                _StubAction(name="ACTION2", value=2),
            ],
        )
        chosen = decoder.decode([frame], frame)
        assert chosen.name == "ACTION2"

    def test_picks_first_when_only_reset_available(self) -> None:
        decoder = DSLActionDecoder(memory=EpisodeMemory())
        frame = _frame(_g([[1]]), available_actions=[_StubAction(name="RESET", value=0)])
        chosen = decoder.decode([frame], frame)
        assert chosen.name == "RESET"
        assert "only RESET" in chosen.reasoning

    def test_custom_stuck_threshold(self) -> None:
        memory = EpisodeMemory()
        # Threshold 2 → stuck after 3 no-progress steps.
        np_grid = np.array([[1]], dtype=np.int8)
        for _ in range(3):
            memory.append(grid=np_grid, action_name="ACTION1", levels_completed=0)
        decoder = DSLActionDecoder(memory=memory, stuck_detector=StuckDetector(threshold=2))
        frame = _frame(_g([[1]]))
        chosen = decoder.decode([frame], frame)
        assert chosen.name == "RESET"


# ---------------------------------------------------------------------------
# Sprint10DSLAgent — end-to-end loop
# ---------------------------------------------------------------------------


class TestSprint10DSLAgent:
    def test_runs_full_episode(self) -> None:
        agent = Sprint10DSLAgent()
        frames: list[_StubFrame] = []
        for i in range(10):
            state_name = "WIN" if i >= 5 else "NOT_FINISHED"
            frame = _frame(_g([[i % 9]]), state=_StubGameState(name=state_name))
            frames.append(frame)
            if agent.is_done(frames, frame):
                break
            chosen = agent.choose_action(frames, frame)
            assert chosen.name in {"RESET", "ACTION1", "ACTION2", "ACTION3"}
        assert frames[-1].state.name == "WIN"

    def test_memory_grows_with_actions(self) -> None:
        agent = Sprint10DSLAgent()
        for i in range(5):
            frame = _frame(_g([[i % 9]]))
            agent.choose_action([frame], frame)
        # The first call has no previous action to record; subsequent
        # 4 calls each push one step. So memory length is 4.
        assert len(agent.memory) == 4

    def test_reset_after_stuck(self) -> None:
        # Use a tight threshold so stuck triggers fast.
        memory = EpisodeMemory()
        agent = Sprint10DSLAgent(memory=memory, stuck_detector=StuckDetector(threshold=2))
        # Feed 5 identical frames so the memory builds up "stuck".
        for _ in range(5):
            frame = _frame(_g([[1]]))
            agent.choose_action([frame], frame)
        # The 5th call's chosen action should be RESET because the
        # memory now shows 4 identical steps (threshold 2 +
        # comparison window).
        last_step = agent.memory.last
        assert last_step is not None
        # The last recorded action is from the 4th choose_action call,
        # because the 5th call records the previous step before deciding.
        # The 5th call's RESET will be recorded on the (hypothetical) 6th call.
        # So we just verify the agent eventually picks RESET.
        # Run one more round and check that's RESET.
        frame = _frame(_g([[1]]))
        chosen = agent.choose_action([frame], frame)
        assert chosen.name == "RESET"

    def test_is_done_on_win(self) -> None:
        agent = Sprint10DSLAgent()
        frame = _frame(_g([[1]]), state=_StubGameState(name="WIN"))
        assert agent.is_done([frame], frame) is True

    def test_is_done_on_game_over(self) -> None:
        agent = Sprint10DSLAgent()
        frame = _frame(_g([[1]]), state=_StubGameState(name="GAME_OVER"))
        assert agent.is_done([frame], frame) is True

    def test_is_done_returns_false_otherwise(self) -> None:
        agent = Sprint10DSLAgent()
        frame = _frame(_g([[1]]), state=_StubGameState(name="NOT_FINISHED"))
        assert agent.is_done([frame], frame) is False

    def test_palette_clamp_through_bridge(self) -> None:
        """Frames with values 10..15 get clamped to 9 by the default
        SATURATE policy. Verify the agent doesn't error out on a wide-
        palette frame.
        """
        agent = Sprint10DSLAgent()
        frame = _frame(_g([[10, 11, 15]]))
        chosen = agent.choose_action([frame], frame)
        assert chosen is not None  # didn't raise

    def test_custom_bridge(self) -> None:
        from cognithor.channels.program_synthesis.arc_agi3.frame_bridge import ClampPolicy

        agent = Sprint10DSLAgent(bridge=FrameBridge(clamp_policy=ClampPolicy.MODULO))
        frame = _frame(_g([[10]]))  # 10 % 10 == 0 under MODULO
        agent.choose_action([frame], frame)
        # Just verify it ran and recorded the step.
        # First call doesn't record, so no assertion on memory yet —
        # next round will.
        frame2 = _frame(_g([[10]]))
        agent.choose_action([frame2], frame2)
        last = agent.memory.last
        assert last is not None
        # Under MODULO, both grids became [[0]].
        assert last.grid.tolist() == [[0]]

    def test_strict_clamp_propagates_error(self) -> None:
        from cognithor.channels.program_synthesis.arc_agi3.frame_bridge import ClampPolicy

        agent = Sprint10DSLAgent(bridge=FrameBridge(clamp_policy=ClampPolicy.STRICT))
        frame = _frame(_g([[10]]))
        with pytest.raises(ValueError, match="STRICT"):
            agent.choose_action([frame], frame)
