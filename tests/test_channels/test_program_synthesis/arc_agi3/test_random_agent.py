# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-11 Wave-1 — RandomActionAgent smoke test."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from cognithor.channels.program_synthesis.arc_agi3.agent import (
    CognithorPSEAgent,
    RandomActionAgent,
)
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)


@dataclass
class _StubGameState:
    name: str


@dataclass
class _StubGameAction:
    name: str
    value: int
    reasoning: str = ""
    _data: dict[str, Any] = field(default_factory=dict)

    def is_simple(self) -> bool:
        return True

    def is_complex(self) -> bool:
        return False

    def set_data(self, data: dict[str, Any]) -> None:
        self._data = dict(data)


@dataclass
class _StubFrameData:
    game_id: str
    state: _StubGameState
    levels_completed: int = 0
    win_levels: int = 1
    guid: str = ""
    full_reset: bool = False
    frame: list[Any] = field(default_factory=list)
    available_actions: list[_StubGameAction] = field(default_factory=list)


def _make_actions() -> list[_StubGameAction]:
    return [
        _StubGameAction(name="RESET", value=0),
        _StubGameAction(name="ACTION1", value=1),
        _StubGameAction(name="ACTION2", value=2),
    ]


class TestRandomActionAgent:
    def test_inherits_from_cognithor_pse_agent(self) -> None:
        agent = RandomActionAgent(seed=42)
        assert isinstance(agent, CognithorPSEAgent)

    def test_is_done_returns_true_on_win(self) -> None:
        agent = RandomActionAgent(seed=42)
        frame = _StubFrameData(
            game_id="ls20",
            state=_StubGameState(name="WIN"),
            available_actions=_make_actions(),
        )
        assert agent.is_done([frame], frame) is True

    def test_is_done_returns_true_on_game_over(self) -> None:
        agent = RandomActionAgent(seed=42)
        frame = _StubFrameData(
            game_id="ls20",
            state=_StubGameState(name="GAME_OVER"),
            available_actions=_make_actions(),
        )
        assert agent.is_done([frame], frame) is True

    def test_is_done_returns_false_on_in_progress(self) -> None:
        agent = RandomActionAgent(seed=42)
        frame = _StubFrameData(
            game_id="ls20",
            state=_StubGameState(name="NOT_FINISHED"),
            available_actions=_make_actions(),
        )
        assert agent.is_done([frame], frame) is False

    def test_choose_action_returns_one_of_available(self) -> None:
        agent = RandomActionAgent(seed=42)
        actions = _make_actions()
        frame = _StubFrameData(
            game_id="ls20",
            state=_StubGameState(name="NOT_FINISHED"),
            available_actions=actions,
        )
        choice = agent.choose_action([frame], frame)
        assert choice in actions

    def test_choose_action_sets_reasoning(self) -> None:
        agent = RandomActionAgent(seed=42)
        actions = _make_actions()
        frame = _StubFrameData(
            game_id="ls20",
            state=_StubGameState(name="NOT_FINISHED"),
            available_actions=actions,
        )
        choice = agent.choose_action([frame], frame)
        assert "RandomActionAgent" in choice.reasoning

    def test_choose_action_is_deterministic_given_seed(self) -> None:
        a1 = RandomActionAgent(seed=42)
        a2 = RandomActionAgent(seed=42)
        actions1 = _make_actions()
        actions2 = _make_actions()
        f1 = _StubFrameData(
            game_id="ls20",
            state=_StubGameState(name="NOT_FINISHED"),
            available_actions=actions1,
        )
        f2 = _StubFrameData(
            game_id="ls20",
            state=_StubGameState(name="NOT_FINISHED"),
            available_actions=actions2,
        )
        # Same seed → same action index. Compare by name to avoid
        # comparing across two _make_actions() lists.
        assert a1.choose_action([f1], f1).name == a2.choose_action([f2], f2).name

    def test_choose_action_raises_on_empty_available_actions(self) -> None:
        agent = RandomActionAgent(seed=42)
        frame = _StubFrameData(
            game_id="ls20",
            state=_StubGameState(name="NOT_FINISHED"),
            available_actions=[],
        )
        with pytest.raises(RuntimeError, match="no available_actions"):
            agent.choose_action([frame], frame)

    def test_runs_full_episode_until_win(self) -> None:
        """Smoke-loop: agent picks actions; we mock the harness by
        flipping the state to WIN after 5 actions. Mirrors the upstream
        ``Agent.main()`` loop without the network glue.
        """
        agent = RandomActionAgent(seed=42)
        frames: list[_StubFrameData] = []
        for i in range(10):
            state_name = "WIN" if i >= 5 else "NOT_FINISHED"
            frame = _StubFrameData(
                game_id="ls20",
                state=_StubGameState(name=state_name),
                levels_completed=i,
                available_actions=_make_actions(),
            )
            frames.append(frame)
            if agent.is_done(frames, frame):
                break
            _ = agent.choose_action(frames, frame)
        # Agent must terminate at frame index 5 (state=WIN).
        assert len(frames) == 6
        assert frames[-1].state.name == "WIN"
