# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-11 Wave-3 — EpisodeMemory + ChangeDetector + StuckDetector tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from cognithor.channels.program_synthesis.arc_agi3.episode_memory import (
    ActionStreakDetector,
    ChangeDetector,
    EpisodeMemory,
    StuckDetector,
    count_actions,
)
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)


@dataclass
class _StubGameState:
    name: str = "NOT_FINISHED"


@dataclass
class _StubFrame:
    game_id: str = "ls20"
    state: _StubGameState = field(default_factory=_StubGameState)
    levels_completed: int = 0
    win_levels: int = 1
    guid: str = ""
    full_reset: bool = False
    frame: list[Any] = field(default_factory=list)
    available_actions: list[Any] = field(default_factory=list)


def _g(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


# ---------------------------------------------------------------------------
# EpisodeMemory
# ---------------------------------------------------------------------------


class TestEpisodeMemory:
    def test_default_capacity_is_sixteen(self) -> None:
        m = EpisodeMemory()
        assert m.capacity == 16
        assert len(m) == 0
        assert m.last is None

    def test_append_increments_length(self) -> None:
        m = EpisodeMemory()
        m.append(grid=_g([[1]]), action_name="ACTION1", levels_completed=0)
        assert len(m) == 1
        assert m.last is not None
        assert m.last.action_name == "ACTION1"

    def test_capacity_evicts_oldest(self) -> None:
        m = EpisodeMemory(capacity=3)
        for i in range(5):
            m.append(grid=_g([[i]]), action_name=f"A{i}", levels_completed=0)
        assert len(m) == 3
        assert m.last is not None
        # Last 3 are A2, A3, A4 (A0, A1 evicted).
        assert [s.action_name for s in m.window(3)] == ["A4", "A3", "A2"]

    def test_window_zero_returns_empty(self) -> None:
        m = EpisodeMemory()
        m.append(grid=_g([[1]]), action_name="ACTION1", levels_completed=0)
        assert m.window(0) == []

    def test_window_negative_raises(self) -> None:
        m = EpisodeMemory()
        with pytest.raises(ValueError, match="n must be >= 0"):
            m.window(-1)

    def test_window_larger_than_buffer_returns_all(self) -> None:
        m = EpisodeMemory()
        m.append(grid=_g([[1]]), action_name="A", levels_completed=0)
        m.append(grid=_g([[2]]), action_name="B", levels_completed=0)
        assert len(m.window(10)) == 2

    def test_clear_resets(self) -> None:
        m = EpisodeMemory()
        m.append(grid=_g([[1]]), action_name="A", levels_completed=0)
        m.clear()
        assert len(m) == 0
        assert m.last is None

    def test_capacity_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="capacity must be >= 1"):
            EpisodeMemory(capacity=0)

    def test_step_grid_is_isolated_copy(self) -> None:
        m = EpisodeMemory()
        original = _g([[1, 2], [3, 4]])
        m.append(grid=original, action_name="A", levels_completed=0)
        original[0, 0] = 99
        # Memory's snapshot should not see the post-append mutation.
        assert m.last is not None
        assert m.last.grid[0, 0] == 1


# ---------------------------------------------------------------------------
# ChangeDetector
# ---------------------------------------------------------------------------


class TestChangeDetector:
    def test_no_change(self) -> None:
        prev = _g([[1, 2], [3, 4]])
        curr = _g([[1, 2], [3, 4]])
        diff = ChangeDetector.diff(
            prev, curr, previous_levels=0, current_frame=_StubFrame(levels_completed=0)
        )
        assert diff.cells_changed == 0
        assert diff.shape_changed is False
        assert diff.levels_advanced is False
        assert diff.full_reset is False
        assert diff.is_change is False

    def test_one_cell_changed(self) -> None:
        prev = _g([[1, 2], [3, 4]])
        curr = _g([[1, 2], [3, 9]])
        diff = ChangeDetector.diff(
            prev, curr, previous_levels=0, current_frame=_StubFrame(levels_completed=0)
        )
        assert diff.cells_changed == 1
        assert diff.is_change is True

    def test_shape_change(self) -> None:
        prev = _g([[1, 2], [3, 4]])
        curr = _g([[5]])
        diff = ChangeDetector.diff(
            prev, curr, previous_levels=0, current_frame=_StubFrame(levels_completed=0)
        )
        assert diff.shape_changed is True
        assert diff.cells_changed == 5  # 4 + 1
        assert diff.is_change is True

    def test_level_advance_flagged(self) -> None:
        prev = _g([[1]])
        curr = _g([[1]])  # grid unchanged but levels_completed went up
        diff = ChangeDetector.diff(
            prev, curr, previous_levels=0, current_frame=_StubFrame(levels_completed=1)
        )
        assert diff.cells_changed == 0
        assert diff.levels_advanced is True
        assert diff.is_change is True

    def test_full_reset_flagged(self) -> None:
        prev = _g([[1]])
        curr = _g([[1]])
        diff = ChangeDetector.diff(
            prev,
            curr,
            previous_levels=0,
            current_frame=_StubFrame(levels_completed=0, full_reset=True),
        )
        assert diff.full_reset is True
        assert diff.is_change is True


# ---------------------------------------------------------------------------
# StuckDetector
# ---------------------------------------------------------------------------


class TestStuckDetector:
    def test_default_threshold_five(self) -> None:
        d = StuckDetector()
        assert d.threshold == 5

    def test_threshold_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="threshold must be >= 1"):
            StuckDetector(threshold=0)

    def test_empty_memory_not_stuck(self) -> None:
        d = StuckDetector(threshold=3)
        assert d.is_stuck(EpisodeMemory()) is False

    def test_too_few_steps_not_stuck(self) -> None:
        d = StuckDetector(threshold=3)
        m = EpisodeMemory()
        m.append(grid=_g([[1]]), action_name="A", levels_completed=0)
        # Need threshold + 1 = 4 steps before a verdict is possible.
        assert d.is_stuck(m) is False

    def test_repeated_no_change_marks_stuck(self) -> None:
        d = StuckDetector(threshold=3)
        m = EpisodeMemory()
        for _ in range(5):
            m.append(grid=_g([[1]]), action_name="A", levels_completed=0)
        assert d.is_stuck(m) is True

    def test_level_advance_breaks_stuck(self) -> None:
        d = StuckDetector(threshold=3)
        m = EpisodeMemory()
        for i in range(5):
            # Level advances at step 2 → not stuck.
            levels = 1 if i >= 2 else 0
            m.append(grid=_g([[1]]), action_name="A", levels_completed=levels)
        assert d.is_stuck(m) is False

    def test_grid_change_breaks_stuck(self) -> None:
        d = StuckDetector(threshold=3)
        m = EpisodeMemory()
        m.append(grid=_g([[1]]), action_name="A", levels_completed=0)
        m.append(grid=_g([[1]]), action_name="A", levels_completed=0)
        m.append(grid=_g([[2]]), action_name="A", levels_completed=0)  # change!
        m.append(grid=_g([[2]]), action_name="A", levels_completed=0)
        m.append(grid=_g([[2]]), action_name="A", levels_completed=0)
        # window of last 4 has a grid change → not stuck.
        assert d.is_stuck(m) is False


# ---------------------------------------------------------------------------
# count_actions helper
# ---------------------------------------------------------------------------


class TestCountActions:
    def test_empty_memory(self) -> None:
        assert count_actions(EpisodeMemory()) == {}

    def test_counts_across_steps(self) -> None:
        m = EpisodeMemory()
        m.append(grid=_g([[1]]), action_name="ACTION1", levels_completed=0)
        m.append(grid=_g([[1]]), action_name="ACTION1", levels_completed=0)
        m.append(grid=_g([[1]]), action_name="ACTION2", levels_completed=0)
        assert count_actions(m) == {"ACTION1": 2, "ACTION2": 1}

    def test_returns_fresh_dict(self) -> None:
        m = EpisodeMemory()
        m.append(grid=_g([[1]]), action_name="A", levels_completed=0)
        c1 = count_actions(m)
        c2 = count_actions(m)
        c1["X"] = 999
        assert "X" not in c2


# ---------------------------------------------------------------------------
# ActionStreakDetector — Sprint-16 Hebel 2
# ---------------------------------------------------------------------------


class TestActionStreakDetector:
    def test_rejects_invalid_window(self) -> None:
        with pytest.raises(ValueError, match="window must be >= 2"):
            ActionStreakDetector(window=1)

    def test_rejects_threshold_above_window(self) -> None:
        with pytest.raises(ValueError, match=r"threshold must be in \[2, window=5\]"):
            ActionStreakDetector(window=5, threshold=6)

    def test_returns_none_when_memory_too_short(self) -> None:
        d = ActionStreakDetector(window=5, threshold=4)
        m = EpisodeMemory()
        for _ in range(4):  # one less than window
            m.append(grid=_g([[1]]), action_name="ACTION6", levels_completed=0)
        assert d.dominant_stuck_action(m) is None

    def test_flags_dominant_action_no_progress(self) -> None:
        # ACTION6 picked 4 of last 5 steps, level frozen → flagged.
        d = ActionStreakDetector(window=5, threshold=4)
        m = EpisodeMemory()
        for action in ("ACTION1", "ACTION6", "ACTION6", "ACTION6", "ACTION6"):
            m.append(grid=_g([[1]]), action_name=action, levels_completed=0)
        assert d.dominant_stuck_action(m) == "ACTION6"

    def test_does_not_flag_when_level_advanced(self) -> None:
        # Same action 4×/5 BUT level went up — that's a winning streak,
        # not a stuck loop.
        d = ActionStreakDetector(window=5, threshold=4)
        m = EpisodeMemory()
        m.append(grid=_g([[1]]), action_name="ACTION1", levels_completed=0)
        for _ in range(4):
            m.append(grid=_g([[1]]), action_name="ACTION6", levels_completed=1)
        assert d.dominant_stuck_action(m) is None

    def test_does_not_flag_diverse_window(self) -> None:
        d = ActionStreakDetector(window=5, threshold=4)
        m = EpisodeMemory()
        for action in ("A", "B", "C", "D", "E"):
            m.append(grid=_g([[1]]), action_name=action, levels_completed=0)
        assert d.dominant_stuck_action(m) is None

    def test_only_considers_recent_window(self) -> None:
        # Earlier streak shouldn't matter — only the last `window`.
        d = ActionStreakDetector(window=5, threshold=4)
        m = EpisodeMemory()
        for _ in range(10):  # noisy older history
            m.append(grid=_g([[1]]), action_name="ACTION6", levels_completed=0)
        for action in ("A", "B", "C", "D", "E"):  # diverse recent
            m.append(grid=_g([[1]]), action_name=action, levels_completed=0)
        assert d.dominant_stuck_action(m) is None

    def test_phase_a_signature_caught(self) -> None:
        # Reproduce Phase-A run #11's signature: 39× ACTION6, level=0.
        # Should always flag ACTION6 once the window fills.
        d = ActionStreakDetector(window=5, threshold=4)
        m = EpisodeMemory(capacity=64)
        for _ in range(39):
            m.append(grid=_g([[1]]), action_name="ACTION6", levels_completed=0)
        assert d.dominant_stuck_action(m) == "ACTION6"
