# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-12 PR-8 — fast_path glue + Sprint10DSLAgent fast-path wiring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cognithor.channels.program_synthesis.arc_agi3.dsl_agent import Sprint10DSLAgent
from cognithor.channels.program_synthesis.arc_agi3.episode_memory import (
    EpisodeMemory,
)
from cognithor.channels.program_synthesis.arc_agi3.fast_path import (
    ClickPlanCache,
    detect_toggle_pair_from_memory,
)
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)

# ---------------------------------------------------------------------------
# Stub harness types — same shape as test_llm_agent.py + test_dsl_agent.py
# ---------------------------------------------------------------------------


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


def _frame(grid: np.ndarray, *, with_click: bool = True) -> _StubFrame:
    actions = [
        _StubAction(name="RESET", value=0),
        _StubAction(name="ACTION1", value=1),
        _StubAction(name="ACTION2", value=2),
    ]
    if with_click:
        actions.append(_StubAction(name="ACTION6", value=6, _is_simple=False))
    return _StubFrame(frame=[grid], available_actions=actions)


# ---------------------------------------------------------------------------
# detect_toggle_pair_from_memory
# ---------------------------------------------------------------------------


class TestDetectTogglePairFromMemory:
    def test_empty_memory_returns_none(self) -> None:
        assert detect_toggle_pair_from_memory(EpisodeMemory()) is None

    def test_single_step_returns_none(self) -> None:
        m = EpisodeMemory()
        m.append(grid=np.zeros((2, 2), dtype=np.int8), action_name="A", levels_completed=0)
        assert detect_toggle_pair_from_memory(m) is None

    def test_clean_swap_detected(self) -> None:
        m = EpisodeMemory()
        m.append(
            grid=np.array([[1, 1], [1, 1]], dtype=np.int8),
            action_name="ACTION6",
            levels_completed=0,
        )
        m.append(
            grid=np.array([[2, 2], [2, 2]], dtype=np.int8),
            action_name="ACTION6",
            levels_completed=0,
        )
        result = detect_toggle_pair_from_memory(m)
        assert result == (1, 2)

    def test_no_change_returns_none(self) -> None:
        m = EpisodeMemory()
        g = np.array([[1, 0], [0, 1]], dtype=np.int8)
        m.append(grid=g, action_name="A", levels_completed=0)
        m.append(grid=g.copy(), action_name="A", levels_completed=0)
        assert detect_toggle_pair_from_memory(m) is None


# ---------------------------------------------------------------------------
# ClickPlanCache
# ---------------------------------------------------------------------------


class TestClickPlanCache:
    def test_first_call_computes_plan(self) -> None:
        cache = ClickPlanCache()
        grid = np.zeros((4, 4), dtype=np.int8)
        grid[1, 1] = 1
        grid[1, 2] = 1
        click = cache.next_click(state_hash="hash1", grid=grid, source_color=1, target_color=2)
        assert click is not None
        # SDK convention: (x=col, y=row) → centroid (1, 1) → x=1, y=1.
        assert click == (1, 1)

    def test_second_call_pops_next(self) -> None:
        cache = ClickPlanCache()
        grid = np.zeros((5, 5), dtype=np.int8)
        grid[0, 0] = 1
        grid[4, 4] = 1
        first = cache.next_click(state_hash="hash2", grid=grid, source_color=1, target_color=2)
        second = cache.next_click(state_hash="hash2", grid=grid, source_color=1, target_color=2)
        third = cache.next_click(state_hash="hash2", grid=grid, source_color=1, target_color=2)
        assert first is not None
        assert second is not None
        # Two clusters → two clicks → third call returns None.
        assert third is None
        assert first != second

    def test_no_solution_returns_none(self) -> None:
        cache = ClickPlanCache()
        grid = np.zeros((3, 3), dtype=np.int8)  # No source pixels.
        click = cache.next_click(state_hash="hash3", grid=grid, source_color=5, target_color=6)
        assert click is None

    def test_cache_does_not_recompute(self) -> None:
        cache = ClickPlanCache()
        grid = np.zeros((3, 3), dtype=np.int8)
        grid[1, 1] = 1
        cache.next_click(state_hash="hash4", grid=grid, source_color=1, target_color=2)
        # Same state hash → cached plan, doesn't matter that grid changed.
        empty_grid = np.zeros((3, 3), dtype=np.int8)
        # Cache hit: would compute None on the empty grid, but cache holds
        # the original plan (already popped to []) so returns None.
        click = cache.next_click(
            state_hash="hash4", grid=empty_grid, source_color=1, target_color=2
        )
        # Plan was 1 click long, already popped → None.
        assert click is None


# ---------------------------------------------------------------------------
# Sprint10DSLAgent fast-path integration
# ---------------------------------------------------------------------------


class TestAgentFastPath:
    def test_disabled_by_default(self) -> None:
        agent = Sprint10DSLAgent()
        assert agent._click_cache is None

    def test_enabled_creates_cache(self) -> None:
        agent = Sprint10DSLAgent(fast_path_enabled=True)
        assert agent._click_cache is not None

    def test_fast_path_skipped_when_no_action6(self) -> None:
        agent = Sprint10DSLAgent(fast_path_enabled=True)
        grid = np.zeros((4, 4), dtype=np.int8)
        grid[1, 1] = 1
        # First frame — ACTION6 missing.
        frame = _frame(grid, with_click=False)
        chosen = agent.choose_action([frame], frame)
        # No click action available → fall through to DSL decoder.
        assert chosen.name in {"ACTION1", "ACTION2"}

    def test_fast_path_skipped_when_no_toggle_pair_observed(self) -> None:
        # On the very first frame the memory is empty → no toggle pair
        # → fast-path skipped → DSL decoder picks something simple.
        agent = Sprint10DSLAgent(fast_path_enabled=True)
        grid = np.zeros((4, 4), dtype=np.int8)
        grid[1, 1] = 1
        frame = _frame(grid)
        chosen = agent.choose_action([frame], frame)
        # First call: empty memory → fast-path bails → DSL decoder.
        assert chosen.name != "ACTION6" or "fast-path" not in chosen.reasoning

    def test_fast_path_fires_after_toggle_observed(self) -> None:
        """End-to-end: feed three same-shape grids. Frames 2→3 show a
        dominant 1→2 swap with a fresh 1-cluster appearing; the
        fast-path should detect the toggle and emit ACTION6."""
        agent = Sprint10DSLAgent(fast_path_enabled=True)

        # Frame 1: any non-trivial start — needed to seed memory.
        g1 = np.zeros((5, 5), dtype=np.int8)
        agent.choose_action([_frame(g1)], _frame(g1))

        # Frame 2: 6 cells of colour 1 (block in top-left).
        g2 = np.zeros((5, 5), dtype=np.int8)
        g2[0:2, 0:3] = 1
        agent.choose_action([_frame(g2)], _frame(g2))

        # Frame 3: those 6 cells flipped to 2 (dominant swap = 1→2),
        # plus a small fresh 1-cluster the planner can target.
        g3 = np.zeros((5, 5), dtype=np.int8)
        g3[0:2, 0:3] = 2
        g3[2, 3:5] = 1
        chosen = agent.choose_action([_frame(g3)], _frame(g3))
        assert chosen.name == "ACTION6"
        assert "fast-path" in chosen.reasoning
