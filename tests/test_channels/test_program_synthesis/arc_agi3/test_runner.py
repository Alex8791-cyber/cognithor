# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-12 — EpisodeRunner tests (no real arc_agi import).

Drives the runner against a stubbed env that mirrors the
``arc_agi.Arcade()`` shape the runner expects. Validates the loop
control + the EpisodeResult population.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import numpy as np

from cognithor.channels.program_synthesis.arc_agi3.dsl_agent import Sprint10DSLAgent
from cognithor.channels.program_synthesis.arc_agi3.runner import (
    EpisodeResult,
    EpisodeRunner,
    run_episode,
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


def _frame(grid: np.ndarray, state_name: str = "NOT_FINISHED", levels: int = 0) -> _StubFrame:
    actions = [
        _StubAction(name="RESET", value=0),
        _StubAction(name="ACTION1", value=1),
        _StubAction(name="ACTION2", value=2),
    ]
    return _StubFrame(
        frame=[grid],
        available_actions=actions,
        state=_StubGameState(name=state_name),
        levels_completed=levels,
    )


class _StubEnv:
    """Stub env that emits a fixed sequence of frames."""

    def __init__(self, frames: list[_StubFrame]) -> None:
        self._frames = frames
        self._idx = 0

    def reset(self) -> _StubFrame:
        self._idx = 0
        return self._frames[0]

    def step(self, *args: Any) -> _StubFrame:
        self._idx = min(self._idx + 1, len(self._frames) - 1)
        return self._frames[self._idx]


class _StubArcade:
    def __init__(self, env: _StubEnv) -> None:
        self._env = env

    def make(self, game_id: str) -> _StubEnv:
        return self._env


def _install_arc_agi_stub(env: _StubEnv) -> None:
    """Inject a dummy ``arc_agi`` module into sys.modules so the runner's
    lazy import succeeds against our stub Arcade."""
    mod = types.ModuleType("arc_agi")
    mod.Arcade = lambda: _StubArcade(env)  # type: ignore[attr-defined]
    sys.modules["arc_agi"] = mod


class TestEpisodeRunnerSuccess:
    def test_runs_to_win(self) -> None:
        # 4 frames, last one is WIN.
        frames = [
            _frame(np.zeros((2, 2), dtype=np.int8)),
            _frame(np.array([[1, 0], [0, 0]], dtype=np.int8)),
            _frame(np.array([[1, 1], [0, 0]], dtype=np.int8)),
            _frame(np.ones((2, 2), dtype=np.int8), state_name="WIN", levels=1),
        ]
        env = _StubEnv(frames)
        _install_arc_agi_stub(env)

        agent = Sprint10DSLAgent()
        result = run_episode(agent=agent, game_id="smoke", max_steps=10)

        assert isinstance(result, EpisodeResult)
        assert result.game_id == "smoke"
        assert result.won is True
        assert result.final_state == "WIN"
        assert result.score == 1.0

    def test_timeout_returns_partial_result(self) -> None:
        # Frames never reach WIN. Runner stops at max_steps.
        frames = [_frame(np.zeros((2, 2), dtype=np.int8)) for _ in range(20)]
        env = _StubEnv(frames)
        _install_arc_agi_stub(env)

        agent = Sprint10DSLAgent()
        result = run_episode(agent=agent, game_id="smoke", max_steps=5)
        assert result.won is False
        assert result.final_state == "NOT_FINISHED"
        assert result.total_steps == 5

    def test_finalize_episode_called(self) -> None:
        frames = [
            _frame(np.zeros((2, 2), dtype=np.int8)),
            _frame(np.zeros((2, 2), dtype=np.int8), state_name="WIN", levels=1),
        ]
        env = _StubEnv(frames)
        _install_arc_agi_stub(env)

        # Use an agent with a recording fake finalize_episode.
        called: list[dict[str, Any]] = []

        class _RecordingAgent(Sprint10DSLAgent):
            def finalize_episode(  # type: ignore[override]
                self, *, score: int, won: bool, levels_solved: int, budget_ratio: float = 0.0
            ) -> None:
                called.append(
                    {
                        "score": score,
                        "won": won,
                        "levels_solved": levels_solved,
                        "budget_ratio": budget_ratio,
                    }
                )

        agent = _RecordingAgent()
        run_episode(agent=agent, game_id="smoke", max_steps=10)
        assert called
        assert called[0]["won"] is True
        assert called[0]["levels_solved"] == 1


class TestEpisodeRunnerErrorPaths:
    def test_missing_arc_agi_returns_error(self) -> None:
        # Ensure arc_agi is NOT in sys.modules.
        sys.modules.pop("arc_agi", None)
        agent = Sprint10DSLAgent()
        runner = EpisodeRunner(agent=agent, game_id="smoke", max_steps=10)

        # Make ``import arc_agi`` raise even if pip installed it.
        with patch.dict(sys.modules, {"arc_agi": None}):
            result = runner.run()
        assert result.final_state == "ERROR"
        assert result.error is not None
        assert "arc_agi" in result.error

    def test_arcade_make_returns_none(self) -> None:
        mod = types.ModuleType("arc_agi")

        class _BadArcade:
            def make(self, game_id: str) -> None:
                return None

        mod.Arcade = lambda: _BadArcade()  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"arc_agi": mod}):
            agent = Sprint10DSLAgent()
            result = run_episode(agent=agent, game_id="bogus")
        assert result.final_state == "ERROR"
        assert "returned None" in (result.error or "")


class TestActionPayload:
    def test_simple_action_returns_action_and_no_data(self) -> None:
        action = _StubAction(name="ACTION1", value=1)
        step_action, step_data = EpisodeRunner._action_payload(action)
        assert step_action is action
        assert step_data is None

    def test_complex_action_returns_action_and_data(self) -> None:
        action = _StubAction(name="ACTION6", value=6, _is_simple=False)
        action.set_data({"x": 5, "y": 7})
        step_action, step_data = EpisodeRunner._action_payload(action)
        assert step_action is action
        assert step_data == {"x": 5, "y": 7}
