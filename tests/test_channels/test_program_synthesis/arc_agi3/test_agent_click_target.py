# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-12 PR-13 — ClickTargetSampler wiring in Sprint10DSLAgent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cognithor.channels.program_synthesis.arc_agi3.click_target_sampler import (
    ClickTargetSampler,
)
from cognithor.channels.program_synthesis.arc_agi3.dsl_agent import Sprint10DSLAgent
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


def _frame(grid: np.ndarray, *, with_click: bool = True, levels: int = 0) -> _StubFrame:
    actions = [
        _StubAction(name="RESET", value=0),
        _StubAction(name="ACTION1", value=1),
        _StubAction(name="ACTION2", value=2),
    ]
    if with_click:
        actions.append(_StubAction(name="ACTION6", value=6, _is_simple=False))
    return _StubFrame(frame=[grid], available_actions=actions, levels_completed=levels)


class TestClickTargetSamplerWiring:
    def test_sampler_optional(self) -> None:
        agent = Sprint10DSLAgent()
        # No sampler wired → no crash, no click emitted.
        f = _frame(np.zeros((3, 3), dtype=np.int8))
        chosen = agent.choose_action([f], f)
        assert chosen.name in {"RESET", "ACTION1", "ACTION2", "ACTION6"}

    def test_sampler_emits_action6_when_target_present(self) -> None:
        sampler = ClickTargetSampler()
        agent = Sprint10DSLAgent(click_target_sampler=sampler)
        # Grid has one salient target at (1, 2) → sampler will return (2, 1).
        grid = np.zeros((4, 4), dtype=np.int8)
        grid[1, 2] = 5
        f = _frame(grid)
        chosen = agent.choose_action([f], f)
        assert chosen.name == "ACTION6"
        assert "click-target" in chosen.reasoning

    def test_sampler_skipped_when_no_action6(self) -> None:
        sampler = ClickTargetSampler()
        agent = Sprint10DSLAgent(click_target_sampler=sampler)
        # No ACTION6 in available list → fall through to DSL decoder.
        grid = np.zeros((4, 4), dtype=np.int8)
        grid[1, 2] = 5
        f = _frame(grid, with_click=False)
        chosen = agent.choose_action([f], f)
        assert chosen.name in {"RESET", "ACTION1", "ACTION2"}

    def test_sampler_skipped_when_pure_background(self) -> None:
        sampler = ClickTargetSampler()
        agent = Sprint10DSLAgent(click_target_sampler=sampler)
        # All-zero grid → no targets → sampler returns None → DSL decoder.
        f = _frame(np.zeros((3, 3), dtype=np.int8))
        chosen = agent.choose_action([f], f)
        assert chosen.name != "ACTION6"

    def test_fast_path_takes_priority_over_sampler(self) -> None:
        """Toggle fast-path fires first; sampler only kicks in on miss.

        The toggle detector needs 2 grids in memory, so we need at least
        3 ``choose_action`` calls before fast-path can fire (the agent
        appends pending_action_name's grid only on call N+1).
        """
        sampler = ClickTargetSampler()
        agent = Sprint10DSLAgent(
            fast_path_enabled=True,
            click_target_sampler=sampler,
        )

        # Frame 1: seed the agent (memory still empty after this call).
        g1 = np.zeros((5, 5), dtype=np.int8)
        agent.choose_action([_frame(g1)], _frame(g1))

        # Frame 2: 6 cells of 1 (memory now has 1 entry).
        g2 = np.zeros((5, 5), dtype=np.int8)
        g2[0:2, 0:3] = 1
        agent.choose_action([_frame(g2)], _frame(g2))

        # Frame 3: those 6 flipped to 2 + a small fresh 1-cluster the
        # planner can target. Memory now has 2 entries → toggle pair
        # (1, 2) detected → fast-path fires.
        g3 = np.zeros((5, 5), dtype=np.int8)
        g3[0:2, 0:3] = 2
        g3[2, 3:5] = 1
        chosen = agent.choose_action([_frame(g3)], _frame(g3))
        assert chosen.name == "ACTION6"
        assert "fast-path" in chosen.reasoning  # NOT click-target

    def test_sampler_resets_on_level_transition(self) -> None:
        sampler = ClickTargetSampler()
        agent = Sprint10DSLAgent(click_target_sampler=sampler)
        # Level 0: emit one click.
        g0 = np.zeros((3, 3), dtype=np.int8)
        g0[1, 1] = 5
        agent.choose_action([_frame(g0, levels=0)], _frame(g0, levels=0))
        assert (1, 1) in sampler.visited

        # Level 1: visited set must be cleared.
        g1 = np.zeros((3, 3), dtype=np.int8)
        g1[2, 2] = 5
        agent.choose_action([_frame(g1, levels=1)], _frame(g1, levels=1))
        # After the level-1 click, only the level-1 coordinate is visited.
        assert sampler.visited == frozenset({(2, 2)})
