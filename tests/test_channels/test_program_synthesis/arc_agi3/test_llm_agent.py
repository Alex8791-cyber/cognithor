# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-11 Wave-5 — LLMReasoningAgent + LLMActionDecoder tests.

The LLM-call adapter (:func:`build_vllm_choice_fn`) is hardware-gated
on a running vLLM server with qwen3.6:27b loaded — that's out of
scope for the unit suite. These tests use deterministic stub
``choice_fn`` callables to validate the wiring + fallback behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from cognithor.channels.program_synthesis.arc_agi3.episode_memory import (
    EpisodeMemory,
)
from cognithor.channels.program_synthesis.arc_agi3.frame_bridge import FrameBridge
from cognithor.channels.program_synthesis.arc_agi3.llm_action_decoder import (
    FrameContext,
    LLMActionDecoder,
    render_grid,
    summarise_history,
)
from cognithor.channels.program_synthesis.arc_agi3.llm_agent import (
    LLMReasoningAgent,
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
# Helpers — render_grid + summarise_history
# ---------------------------------------------------------------------------


class TestRenderGrid:
    def test_renders_2d_grid_as_lines(self) -> None:
        out = render_grid(np.array([[1, 2], [3, 4]], dtype=np.int8))
        assert out == "1 2\n3 4"

    def test_renders_single_row(self) -> None:
        out = render_grid(np.array([[0, 1, 2]], dtype=np.int8))
        assert out == "0 1 2"

    def test_rejects_1d_input(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            render_grid(np.array([1, 2, 3], dtype=np.int8))


class TestSummariseHistory:
    def test_empty_memory(self) -> None:
        assert summarise_history(EpisodeMemory()) == "(no actions yet)"

    def test_one_step(self) -> None:
        m = EpisodeMemory()
        m.append(grid=np.array([[1]], dtype=np.int8), action_name="ACTION1", levels_completed=0)
        out = summarise_history(m)
        assert "step -1: ACTION1" in out

    def test_level_marker(self) -> None:
        m = EpisodeMemory()
        m.append(grid=np.array([[1]], dtype=np.int8), action_name="ACTION1", levels_completed=2)
        out = summarise_history(m)
        assert "level=2" in out

    def test_max_steps_bounds(self) -> None:
        m = EpisodeMemory()
        for i in range(20):
            m.append(
                grid=np.array([[i % 9]], dtype=np.int8),
                action_name=f"A{i}",
                levels_completed=0,
            )
        out = summarise_history(m, max_steps=3)
        # Only 3 most-recent steps in the summary.
        assert out.count("step ") == 3

    def test_max_steps_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_steps must be >= 1"):
            summarise_history(EpisodeMemory(), max_steps=0)


# ---------------------------------------------------------------------------
# LLMActionDecoder
# ---------------------------------------------------------------------------


class TestLLMActionDecoder:
    def test_picks_action_from_stub(self) -> None:
        bridge = FrameBridge()
        memory = EpisodeMemory()
        # Stub LLM that always picks ACTION2.
        decoder = LLMActionDecoder(
            bridge=bridge,
            memory=memory,
            choice_fn=lambda ctx: ("ACTION2", "stub picks ACTION2"),
        )
        frame = _frame(_g([[1]]))
        chosen = decoder.decode([frame], frame)
        assert chosen.name == "ACTION2"
        assert chosen.reasoning == "stub picks ACTION2"

    def test_falls_back_when_choice_fn_raises(self) -> None:
        bridge = FrameBridge()
        memory = EpisodeMemory()

        def _raising(ctx: FrameContext) -> tuple[str, str]:
            raise RuntimeError("simulated LLM failure")

        decoder = LLMActionDecoder(bridge=bridge, memory=memory, choice_fn=_raising)
        frame = _frame(_g([[1]]))
        chosen = decoder.decode([frame], frame)
        # Falls back to DSLActionDecoder, which prefers least-tried
        # non-RESET action — ACTION1 (count 0, first in list).
        assert chosen.name in {"ACTION1", "ACTION2", "ACTION3"}
        assert "fallback" in chosen.reasoning.lower()

    def test_falls_back_when_choice_fn_returns_unknown_action(self) -> None:
        bridge = FrameBridge()
        memory = EpisodeMemory()
        decoder = LLMActionDecoder(
            bridge=bridge,
            memory=memory,
            choice_fn=lambda ctx: ("ACTION_BOGUS", "made up"),
        )
        frame = _frame(_g([[1]]))
        chosen = decoder.decode([frame], frame)
        assert chosen.name in {"ACTION1", "ACTION2", "ACTION3"}
        assert "fallback" in chosen.reasoning.lower()
        assert "ACTION_BOGUS" in chosen.reasoning

    def test_passes_full_context_to_stub(self) -> None:
        bridge = FrameBridge()
        memory = EpisodeMemory()
        captured: list[FrameContext] = []

        def _capturing(ctx: FrameContext) -> tuple[str, str]:
            captured.append(ctx)
            return "ACTION1", "ok"

        decoder = LLMActionDecoder(bridge=bridge, memory=memory, choice_fn=_capturing)
        frame = _frame(_g([[3, 4], [5, 6]]), levels_completed=1, win_levels=3)
        decoder.decode([frame], frame)
        assert len(captured) == 1
        ctx = captured[0]
        assert ctx.grid.tolist() == [[3, 4], [5, 6]]
        assert "RESET" in ctx.available_action_names
        assert ctx.levels_completed == 1
        assert ctx.win_levels == 3

    def test_anti_loop_overrides_dead_action(self) -> None:
        # Sprint-16: when the state-counter has marked an action dead
        # at the current state, the LLM's pick is overridden to a
        # least-tried allowed alternative — even if the stub LLM
        # insists on the dead action.
        from cognithor.channels.program_synthesis.arc_agi3.llm_action_decoder import (
            hash_state,
        )
        from cognithor.channels.program_synthesis.arc_agi3.state_action_counts import (
            StateActionCounter,
        )

        bridge = FrameBridge()
        memory = EpisodeMemory()
        counter = StateActionCounter()
        grid = _g([[7]])
        # Bridge converts to int8 and the counter keys off that.
        state_hash = hash_state(bridge.extract_grid(_frame(grid)))
        counter.mark_dead(state_hash, "ACTION6")

        captured: list[FrameContext] = []

        def _stuck_llm(ctx: FrameContext) -> tuple[str, str]:
            captured.append(ctx)
            # LLM ignores the constraint and picks the dead action anyway.
            return "ACTION6", "stub picks the dead action"

        decoder = LLMActionDecoder(
            bridge=bridge,
            memory=memory,
            choice_fn=_stuck_llm,
            state_counter=counter,
        )
        frame = _frame(
            grid,
            available_actions=[
                _StubAction(name="ACTION1", value=1),
                _StubAction(name="ACTION6", value=6),
            ],
        )
        chosen = decoder.decode([frame], frame)

        # Override fired; chosen action is NOT ACTION6.
        assert chosen.name == "ACTION1"
        assert "anti-loop override" in chosen.reasoning
        # Prompt context surfaced the constraint to the LLM.
        ctx = captured[0]
        assert "ACTION6" in ctx.forbidden_action_names
        assert "DEAD" in ctx.state_action_summary

    def test_anti_loop_repeat_threshold_filters(self) -> None:
        # Sprint-16: even without an explicit mark_dead, picking the
        # same action 3+ times at the same state forbids it (repeat-
        # saturation). Mirrors the ACTION6×40 deterministic-loop trap.
        from cognithor.channels.program_synthesis.arc_agi3.llm_action_decoder import (
            hash_state,
        )
        from cognithor.channels.program_synthesis.arc_agi3.state_action_counts import (
            StateActionCounter,
        )

        bridge = FrameBridge()
        memory = EpisodeMemory()
        counter = StateActionCounter()
        grid = _g([[3]])
        state_hash = hash_state(bridge.extract_grid(_frame(grid)))
        for _ in range(3):
            counter.increment(state_hash, "ACTION6")

        decoder = LLMActionDecoder(
            bridge=bridge,
            memory=memory,
            choice_fn=lambda ctx: ("ACTION6", "stuck"),
            state_counter=counter,
        )
        frame = _frame(
            grid,
            available_actions=[
                _StubAction(name="ACTION1", value=1),
                _StubAction(name="ACTION6", value=6),
            ],
        )
        chosen = decoder.decode([frame], frame)
        assert chosen.name != "ACTION6"
        assert "anti-loop override" in chosen.reasoning

    def test_anti_loop_passes_through_when_counter_quiet(self) -> None:
        # When nothing is dead/saturated, behaviour is identical to
        # the pre-Sprint-16 code path.
        from cognithor.channels.program_synthesis.arc_agi3.state_action_counts import (
            StateActionCounter,
        )

        decoder = LLMActionDecoder(
            bridge=FrameBridge(),
            memory=EpisodeMemory(),
            choice_fn=lambda ctx: ("ACTION2", "ok"),
            state_counter=StateActionCounter(),
        )
        frame = _frame(_g([[1]]))
        chosen = decoder.decode([frame], frame)
        assert chosen.name == "ACTION2"
        assert "override" not in chosen.reasoning

    def test_streak_detector_overrides_state_agnostic_loop(self) -> None:
        # Sprint-16 Hebel 2: catches the click-game loop that Hebel 1
        # silently passes — each pick changes the state hash so per-
        # state counts never reach the threshold, but the streak
        # detector flags ACTION6 dominating the recent window with
        # frozen level progress.
        from cognithor.channels.program_synthesis.arc_agi3.episode_memory import (
            ActionStreakDetector,
        )

        memory = EpisodeMemory()
        # Simulate the run #11 trace: ACTION6 picked 4 of last 5 with
        # different grids each time, level frozen at 0.
        for i, action in enumerate(("ACTION1", "ACTION6", "ACTION6", "ACTION6", "ACTION6")):
            memory.append(grid=_g([[i]]), action_name=action, levels_completed=0)

        captured: list[FrameContext] = []

        def _stuck_llm(ctx: FrameContext) -> tuple[str, str]:
            captured.append(ctx)
            # LLM still wants ACTION6; Hebel 2 must override.
            return "ACTION6", "stub picks the streak-stuck action"

        decoder = LLMActionDecoder(
            bridge=FrameBridge(),
            memory=memory,
            choice_fn=_stuck_llm,
            action_streak_detector=ActionStreakDetector(),
        )
        frame = _frame(
            _g([[7]]),
            available_actions=[
                _StubAction(name="ACTION1", value=1),
                _StubAction(name="ACTION6", value=6),
            ],
        )
        chosen = decoder.decode([frame], frame)

        assert chosen.name == "ACTION1"
        assert "anti-loop override" in chosen.reasoning
        ctx = captured[0]
        assert "ACTION6" in ctx.forbidden_action_names
        assert "STREAK-STUCK" in ctx.state_action_summary

    def test_llm_agent_auto_wires_streak_detector(self) -> None:
        # Sprint-16: the agent's __init__ should hand the decoder a
        # default ActionStreakDetector so callers don't have to opt
        # in. Smoke-check by triggering the loop signature on the
        # public LLMReasoningAgent.
        agent = LLMReasoningAgent(choice_fn=lambda ctx: ("ACTION6", "stuck"))
        # Pre-fill the agent's memory with the loop signature.
        for i in range(5):
            agent.memory.append(
                grid=_g([[i]]),
                action_name="ACTION6",
                levels_completed=0,
            )
        frame = _frame(
            _g([[9]]),
            available_actions=[
                _StubAction(name="ACTION1", value=1),
                _StubAction(name="ACTION6", value=6),
            ],
        )
        chosen = agent.choose_action([frame], frame)
        # Override fired through the auto-wired detector.
        assert chosen.name == "ACTION1"


# ---------------------------------------------------------------------------
# LLMReasoningAgent — full episode loop
# ---------------------------------------------------------------------------


class TestLLMReasoningAgent:
    def test_runs_with_stub_choice_fn(self) -> None:
        # Stub picks ACTION1 always, episode ends at WIN.
        agent = LLMReasoningAgent(choice_fn=lambda ctx: ("ACTION1", "stub"))
        frames: list[_StubFrame] = []
        for i in range(8):
            state_name = "WIN" if i >= 4 else "NOT_FINISHED"
            frame = _frame(_g([[i % 9]]), state=_StubGameState(name=state_name))
            frames.append(frame)
            if agent.is_done(frames, frame):
                break
            agent.choose_action(frames, frame)
        assert frames[-1].state.name == "WIN"

    def test_inherits_episode_memory(self) -> None:
        agent = LLMReasoningAgent(choice_fn=lambda ctx: ("ACTION1", "stub"))
        for _ in range(3):
            frame = _frame(_g([[1]]))
            agent.choose_action([frame], frame)
        # 3 calls — first records nothing, next 2 record one each → length 2.
        assert len(agent.memory) == 2

    def test_falls_back_to_dsl_when_llm_fails(self) -> None:
        def _always_fails(ctx: FrameContext) -> tuple[str, str]:
            raise RuntimeError("simulated network error")

        agent = LLMReasoningAgent(choice_fn=_always_fails)
        frame = _frame(_g([[1]]))
        chosen = agent.choose_action([frame], frame)
        # Fell through to DSLActionDecoder (no exception).
        assert chosen.name in {"ACTION1", "ACTION2", "ACTION3"}

    def test_custom_history_steps(self) -> None:
        captured: list[FrameContext] = []

        def _cap(ctx: FrameContext) -> tuple[str, str]:
            captured.append(ctx)
            return "ACTION1", "stub"

        agent = LLMReasoningAgent(choice_fn=_cap, history_steps=2)
        # Run 5 frames; the 5th call's prompt should contain at most
        # 2 history entries (the bound).
        for _ in range(5):
            frame = _frame(_g([[1]]))
            agent.choose_action([frame], frame)
        last_ctx = captured[-1]
        # history_summary uses ", " separator + "step -X: A1" pattern
        # Up to 2 steps shown.
        assert last_ctx.history_summary.count("step ") <= 2

    def test_is_done_inherits_win_game_over_policy(self) -> None:
        agent = LLMReasoningAgent(choice_fn=lambda ctx: ("ACTION1", "stub"))
        for state in ("WIN", "GAME_OVER"):
            frame = _frame(_g([[1]]), state=_StubGameState(name=state))
            assert agent.is_done([frame], frame) is True
        frame = _frame(_g([[1]]), state=_StubGameState(name="NOT_FINISHED"))
        assert agent.is_done([frame], frame) is False
