# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-14 PR-1 — PlanningLLMActionDecoder tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from cognithor.channels.program_synthesis.arc_agi3.episode_memory import EpisodeMemory
from cognithor.channels.program_synthesis.arc_agi3.frame_bridge import FrameBridge
from cognithor.channels.program_synthesis.arc_agi3.planning_agent import (
    PlanningLLMReasoningAgent,
    parse_plan_response,
)
from cognithor.channels.program_synthesis.arc_agi3.planning_decoder import (
    PlanningLLMActionDecoder,
    PlanStep,
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


def _frame(grid: np.ndarray, levels: int = 0) -> _StubFrame:
    actions = [
        _StubAction(name="RESET", value=0),
        _StubAction(name="ACTION1", value=1),
        _StubAction(name="ACTION2", value=2),
        _StubAction(name="ACTION6", value=6, _is_simple=False),
    ]
    return _StubFrame(frame=[grid], available_actions=actions, levels_completed=levels)


class TestParsePlanResponse:
    def test_parses_simple_plan(self) -> None:
        text = (
            '{"reasoning": "explore", "plan": ['
            '{"action": "ACTION1", "data": null, "reasoning": "up"},'
            '{"action": "ACTION2", "data": null, "reasoning": "down"}'
            "]}"
        )
        steps, top = parse_plan_response(text)
        assert top == "explore"
        assert len(steps) == 2
        assert steps[0].action_name == "ACTION1"
        assert steps[1].action_name == "ACTION2"

    def test_parses_click_data(self) -> None:
        text = (
            '{"reasoning": "click target", "plan": ['
            '{"action": "ACTION6", "data": {"x": 5, "y": 7}, "reasoning": "hit it"}'
            "]}"
        )
        steps, _ = parse_plan_response(text)
        assert steps[0].data == {"x": 5, "y": 7}

    def test_strips_qwen_thinking_block(self) -> None:
        text = (
            "<think>some reasoning</think>"
            '{"reasoning": "go", "plan": [{"action": "ACTION1", "data": null}]}'
        )
        steps, _ = parse_plan_response(text)
        assert steps[0].action_name == "ACTION1"

    def test_empty_plan_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_plan_response('{"reasoning": "done", "plan": []}')

    def test_missing_json_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_plan_response("no json here")


class TestPlanningDecoder:
    def test_consults_llm_when_queue_empty(self) -> None:
        calls: list[int] = []

        def _planner(ctx: Any) -> tuple[list[PlanStep], str]:
            calls.append(1)
            return [PlanStep("ACTION1"), PlanStep("ACTION2")], "two-step"

        decoder = PlanningLLMActionDecoder(
            bridge=FrameBridge(),
            memory=EpisodeMemory(),
            choice_fn=_planner,
        )
        f = _frame(np.zeros((2, 2), dtype=np.int8))
        # First call → consults LLM.
        chosen, _ = decoder.pick_action([f], f, f.available_actions)
        assert chosen.name == "ACTION1"
        assert len(calls) == 1
        # Second call → uses cached plan.
        chosen, _ = decoder.pick_action([f], f, f.available_actions)
        assert chosen.name == "ACTION2"
        assert len(calls) == 1
        # Third call → queue empty → re-consults.
        chosen, _ = decoder.pick_action([f], f, f.available_actions)
        assert len(calls) == 2

    def test_invalidates_plan_on_level_change(self) -> None:
        calls: list[int] = []

        def _planner(ctx: Any) -> tuple[list[PlanStep], str]:
            calls.append(1)
            return [PlanStep("ACTION1"), PlanStep("ACTION2"), PlanStep("ACTION1")], ""

        decoder = PlanningLLMActionDecoder(
            bridge=FrameBridge(),
            memory=EpisodeMemory(),
            choice_fn=_planner,
        )
        f0 = _frame(np.zeros((2, 2), dtype=np.int8), levels=0)
        decoder.pick_action([f0], f0, f0.available_actions)
        decoder.pick_action([f0], f0, f0.available_actions)
        assert decoder.plan_remaining == 1
        # Level transition → plan invalidated, re-consults on next call.
        f1 = _frame(np.zeros((2, 2), dtype=np.int8), levels=1)
        decoder.pick_action([f1], f1, f1.available_actions)
        assert len(calls) == 2

    def test_falls_back_when_planner_raises(self) -> None:
        def _planner(ctx: Any) -> tuple[list[PlanStep], str]:
            raise RuntimeError("LLM hiccup")

        decoder = PlanningLLMActionDecoder(
            bridge=FrameBridge(),
            memory=EpisodeMemory(),
            choice_fn=_planner,
        )
        f = _frame(np.zeros((2, 2), dtype=np.int8))
        chosen, reason = decoder.pick_action([f], f, f.available_actions)
        # DSL fallback returned a simple action (it filters out ACTION6).
        assert chosen.name in {"ACTION1", "ACTION2", "RESET"}

    def test_horizon_caps_plan(self) -> None:
        def _planner(ctx: Any) -> tuple[list[PlanStep], str]:
            return [PlanStep(f"ACTION{i % 2 + 1}") for i in range(20)], ""

        decoder = PlanningLLMActionDecoder(
            bridge=FrameBridge(),
            memory=EpisodeMemory(),
            choice_fn=_planner,
            plan_horizon=3,
        )
        f = _frame(np.zeros((2, 2), dtype=np.int8))
        decoder.pick_action([f], f, f.available_actions)
        assert decoder.plan_remaining == 2  # 3 total minus 1 popped

    def test_replan_now_invalidates_queue(self) -> None:
        calls: list[int] = []

        def _planner(ctx: Any) -> tuple[list[PlanStep], str]:
            calls.append(1)
            return [PlanStep("ACTION1"), PlanStep("ACTION2")], ""

        decoder = PlanningLLMActionDecoder(
            bridge=FrameBridge(),
            memory=EpisodeMemory(),
            choice_fn=_planner,
        )
        f = _frame(np.zeros((2, 2), dtype=np.int8))
        decoder.pick_action([f], f, f.available_actions)
        decoder.replan_now()
        decoder.pick_action([f], f, f.available_actions)
        assert len(calls) == 2

    def test_unknown_planned_action_invalidates_plan(self) -> None:
        def _planner(ctx: Any) -> tuple[list[PlanStep], str]:
            return [PlanStep("ACTION_BOGUS"), PlanStep("ACTION1")], ""

        decoder = PlanningLLMActionDecoder(
            bridge=FrameBridge(),
            memory=EpisodeMemory(),
            choice_fn=_planner,
        )
        f = _frame(np.zeros((2, 2), dtype=np.int8))
        chosen, reason = decoder.pick_action([f], f, f.available_actions)
        # Plan named an unavailable action → fall back, full plan invalidated.
        assert "fallback" in reason.lower() or chosen.name in {"ACTION1", "ACTION2"}
        assert decoder.plan_remaining == 0


class TestPlanningAgent:
    def test_exposes_plan_remaining(self) -> None:
        def _planner(ctx: Any) -> tuple[list[PlanStep], str]:
            return [PlanStep("ACTION1"), PlanStep("ACTION2"), PlanStep("ACTION1")], ""

        agent = PlanningLLMReasoningAgent(choice_fn=_planner)
        f = _frame(np.zeros((2, 2), dtype=np.int8))
        agent.choose_action([f], f)
        assert agent.plan_remaining == 2
        agent.choose_action([f], f)
        assert agent.plan_remaining == 1

    def test_click_data_propagates_to_action(self) -> None:
        def _planner(ctx: Any) -> tuple[list[PlanStep], str]:
            return [PlanStep("ACTION6", data={"x": 9, "y": 11})], ""

        agent = PlanningLLMReasoningAgent(choice_fn=_planner)
        f = _frame(np.zeros((2, 2), dtype=np.int8))
        chosen = agent.choose_action([f], f)
        assert chosen.name == "ACTION6"
        # set_data was called.
        assert chosen._data == {"x": 9, "y": 11}  # noqa: SLF001
