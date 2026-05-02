# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-13 PR-1 — GoalInferer tests."""

from __future__ import annotations

import numpy as np

from cognithor.channels.program_synthesis.arc_agi3.episode_memory import (
    EpisodeMemory,
)
from cognithor.channels.program_synthesis.arc_agi3.frame_analyzer import (
    FrameAnalyzer,
)
from cognithor.channels.program_synthesis.arc_agi3.goal_inferer import (
    GoalInferer,
    infer_goal_summary,
)
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)


def _g(grid: list[list[int]]) -> np.ndarray:
    return np.array(grid, dtype=np.int8)


class TestEmptyMemory:
    def test_returns_placeholder(self) -> None:
        out = GoalInferer().infer(EpisodeMemory())
        assert "no observations" in out.lower()

    def test_convenience_wrapper_works(self) -> None:
        out = infer_goal_summary(EpisodeMemory())
        assert "no observations" in out.lower()


class TestLevelProgressionHypothesis:
    def test_detects_level_up_event(self) -> None:
        m = EpisodeMemory()
        m.append(grid=_g([[0]]), action_name="ACTION1", levels_completed=0)
        m.append(grid=_g([[1]]), action_name="ACTION3", levels_completed=1)
        out = GoalInferer().infer(m)
        assert "REACH_STATE" in out
        assert "ACTION3" in out

    def test_picks_most_common_winning_action(self) -> None:
        m = EpisodeMemory()
        # Two distinct ACTION3-driven level ups.
        m.append(grid=_g([[0]]), action_name="X", levels_completed=0)
        m.append(grid=_g([[0]]), action_name="ACTION3", levels_completed=1)
        m.append(grid=_g([[0]]), action_name="Y", levels_completed=1)
        m.append(grid=_g([[0]]), action_name="ACTION3", levels_completed=2)
        out = GoalInferer().infer(m)
        assert "REACH_STATE" in out
        assert "ACTION3" in out


class TestStuckHypothesis:
    def test_detects_identical_frames(self) -> None:
        m = EpisodeMemory()
        same = _g([[1, 0], [0, 1]])
        for _ in range(8):
            m.append(grid=same.copy(), action_name="ACTION1", levels_completed=0)
        out = GoalInferer().infer(m)
        assert "STUCK" in out


class TestNoProgressHypothesis:
    def test_full_window_without_levels_emits_signal(self) -> None:
        m = EpisodeMemory()
        for i in range(8):
            m.append(
                grid=_g([[i % 4]]),
                action_name=f"ACT{i}",
                levels_completed=0,
            )
        out = GoalInferer().infer(m)
        # No level ups in a full window → NO_PROGRESS hint.
        assert "NO_PROGRESS" in out


class TestFrameAnalyzerHypothesis:
    def test_broad_effect_signal(self) -> None:
        # Need ≥2 grids with ≥50 changed pixels.
        fa = FrameAnalyzer()
        big_a = np.zeros((10, 10), dtype=np.int8)
        big_b = np.ones((10, 10), dtype=np.int8) * 3
        big_c = np.ones((10, 10), dtype=np.int8) * 7
        fa.analyze(big_a)
        fa.analyze(big_b, action="WIPE")  # 100 px changed
        fa.analyze(big_c, action="WIPE")  # 100 px changed
        m = EpisodeMemory()
        m.append(grid=big_a, action_name="WIPE", levels_completed=0)
        out = GoalInferer().infer(m, fa)
        assert "CLEAR_BOARD" in out or "BROAD_EFFECT" in out

    def test_fine_control_signal(self) -> None:
        fa = FrameAnalyzer()
        a = np.zeros((4, 4), dtype=np.int8)
        b = a.copy()
        b[0, 0] = 5
        c = a.copy()
        c[0, 1] = 5
        fa.analyze(a)
        fa.analyze(b, action="MOVE")
        fa.analyze(c, action="MOVE")
        m = EpisodeMemory()
        m.append(grid=a, action_name="MOVE", levels_completed=0)
        out = GoalInferer().infer(m, fa)
        assert "FINE_CONTROL" in out or "MOVE" in out


class TestRecentWindowParameter:
    def test_validates_minimum_window(self) -> None:
        try:
            GoalInferer(recent_window=1)
        except ValueError:
            return
        raise AssertionError("expected ValueError for recent_window=1")

    def test_window_caps_evidence(self) -> None:
        m = EpisodeMemory()
        # 20 steps with no level progression.
        for i in range(20):
            m.append(grid=_g([[i % 4]]), action_name="X", levels_completed=0)
        # Tight window => still NO_PROGRESS but only counts last 4.
        out = GoalInferer(recent_window=4).infer(m)
        assert "NO_PROGRESS" in out
        # Number stated should be the recent_window size, not 20.
        assert "4 actions" in out or "4 step" in out
