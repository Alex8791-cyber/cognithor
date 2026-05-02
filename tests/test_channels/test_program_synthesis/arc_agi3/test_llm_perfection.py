# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-13 PR-1 — LLM prompt perfection tests.

Verifies the three Sprint-13 LLM-prompt enrichments:
1. ``GoalInferer`` output appears in the user prompt as ``Goal hypothesis: ...``.
2. Per-game prompt fragments (game_prompts.GAME_PROMPTS) appear in the
   system prompt for known game IDs.
3. ``LLMReasoningAgent`` defaults to a ``GoalInferer`` when none is passed
   so every LLM call gets the goal-hypothesis line.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cognithor.channels.program_synthesis.arc_agi3.episode_memory import (
    EpisodeMemory,
)
from cognithor.channels.program_synthesis.arc_agi3.frame_analyzer import (
    FrameAnalyzer,
)
from cognithor.channels.program_synthesis.arc_agi3.frame_bridge import FrameBridge
from cognithor.channels.program_synthesis.arc_agi3.goal_inferer import GoalInferer
from cognithor.channels.program_synthesis.arc_agi3.llm_action_decoder import (
    FrameContext,
    LLMActionDecoder,
)
from cognithor.channels.program_synthesis.arc_agi3.llm_agent import (
    LLMReasoningAgent,
    _build_system_prompt,
    _build_user_prompt,
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


def _frame(grid: np.ndarray, game_id: str = "smoke") -> _StubFrame:
    actions = [
        _StubAction(name="RESET", value=0),
        _StubAction(name="ACTION1", value=1),
        _StubAction(name="ACTION2", value=2),
    ]
    return _StubFrame(frame=[grid], available_actions=actions, game_id=game_id)


class TestUserPromptGoalSummary:
    def test_goal_summary_renders_in_prompt(self) -> None:
        ctx = FrameContext(
            grid=np.zeros((2, 2), dtype=np.int8),
            available_action_names=["ACTION1"],
            history_summary="(empty)",
            levels_completed=0,
            win_levels=1,
            goal_summary="REACH_STATE: 1 level-up via ACTION3",
        )
        prompt = _build_user_prompt(ctx)
        assert "Goal hypothesis:" in prompt
        assert "REACH_STATE" in prompt

    def test_no_goal_summary_skips_line(self) -> None:
        ctx = FrameContext(
            grid=np.zeros((2, 2), dtype=np.int8),
            available_action_names=["ACTION1"],
            history_summary="(empty)",
            levels_completed=0,
            win_levels=1,
        )
        prompt = _build_user_prompt(ctx)
        assert "Goal hypothesis:" not in prompt


class TestSystemPromptPerGame:
    def test_known_game_id_includes_rules(self) -> None:
        ctx = FrameContext(
            grid=np.zeros((2, 2), dtype=np.int8),
            available_action_names=["ACTION1", "ACTION3", "ACTION6"],
            history_summary="",
            levels_completed=0,
            win_levels=1,
            game_id="ls20",  # known game with Locksmith rules
        )
        sys_prompt = _build_system_prompt(ctx)
        # The Locksmith rules from game_prompts should mention something
        # game-specific (door, key, rotate, etc.).
        assert (
            "key" in sys_prompt.lower()
            or "door" in sys_prompt.lower()
            or "rotate" in sys_prompt.lower()
        )

    def test_unknown_game_id_uses_generic_prompt(self) -> None:
        ctx = FrameContext(
            grid=np.zeros((2, 2), dtype=np.int8),
            available_action_names=["ACTION1"],
            history_summary="",
            levels_completed=0,
            win_levels=1,
            game_id="totally_unknown_game_id",
        )
        sys_prompt = _build_system_prompt(ctx)
        # Falls back to generic prompt — no per-game fragment.
        assert "JSON" in sys_prompt
        assert "key" not in sys_prompt.lower()


class TestLLMActionDecoderGoalWiring:
    def test_decoder_passes_goal_summary_to_choice_fn(self) -> None:
        captured: list[FrameContext] = []

        def _capture(ctx: FrameContext) -> tuple[str, str]:
            captured.append(ctx)
            return "ACTION1", "stub"

        memory = EpisodeMemory()
        # Seed memory with a level-up event so the inferer emits REACH_STATE.
        memory.append(
            grid=np.zeros((2, 2), dtype=np.int8),
            action_name="ACTION1",
            levels_completed=0,
        )
        memory.append(
            grid=np.zeros((2, 2), dtype=np.int8),
            action_name="ACTION3",
            levels_completed=1,
        )

        decoder = LLMActionDecoder(
            bridge=FrameBridge(),
            memory=memory,
            choice_fn=_capture,
            goal_inferer=GoalInferer(),
        )
        f = _frame(np.zeros((2, 2), dtype=np.int8))
        decoder.decode([f], f)
        ctx = captured[0]
        assert ctx.goal_summary
        assert "REACH_STATE" in ctx.goal_summary

    def test_no_inferer_means_empty_summary(self) -> None:
        captured: list[FrameContext] = []

        def _capture(ctx: FrameContext) -> tuple[str, str]:
            captured.append(ctx)
            return "ACTION1", "stub"

        decoder = LLMActionDecoder(
            bridge=FrameBridge(),
            memory=EpisodeMemory(),
            choice_fn=_capture,
        )
        f = _frame(np.zeros((2, 2), dtype=np.int8))
        decoder.decode([f], f)
        assert captured[0].goal_summary == ""


class TestLLMReasoningAgentDefaultGoalInferer:
    def test_agent_creates_default_goal_inferer(self) -> None:
        captured: list[FrameContext] = []

        def _capture(ctx: FrameContext) -> tuple[str, str]:
            captured.append(ctx)
            return "ACTION1", "stub"

        agent = LLMReasoningAgent(
            choice_fn=_capture,
            frame_analyzer=FrameAnalyzer(),
        )
        f = _frame(np.zeros((2, 2), dtype=np.int8), game_id="smoke")
        agent.choose_action([f], f)
        # Default GoalInferer was created → goal_summary populated.
        ctx = captured[0]
        assert ctx.goal_summary  # non-empty
        # No memory yet → "no observations" placeholder.
        assert "no observations" in ctx.goal_summary.lower()

    def test_agent_passes_game_id_through(self) -> None:
        captured: list[FrameContext] = []

        def _capture(ctx: FrameContext) -> tuple[str, str]:
            captured.append(ctx)
            return "ACTION1", "stub"

        agent = LLMReasoningAgent(choice_fn=_capture)
        f = _frame(np.zeros((2, 2), dtype=np.int8), game_id="ls20")
        agent.choose_action([f], f)
        assert captured[0].game_id == "ls20"
