# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-12 — FrameAnalyzer tests."""

from __future__ import annotations

import numpy as np

from cognithor.channels.program_synthesis.arc_agi3.frame_analyzer import (
    FrameAnalyzer,
    MovementInfo,
)
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)


def _grid(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int32)


class TestAnalyzeBasics:
    def test_first_frame_returns_none(self) -> None:
        fa = FrameAnalyzer()
        assert fa.analyze(_grid([[0, 0], [0, 0]])) is None
        assert fa.frame_count == 1

    def test_no_change_returns_none(self) -> None:
        fa = FrameAnalyzer()
        g = _grid([[1, 0], [0, 1]])
        fa.analyze(g)
        assert fa.analyze(g) is None

    def test_movement_detected(self) -> None:
        fa = FrameAnalyzer()
        fa.analyze(_grid([[1, 0], [0, 0]]))
        movement = fa.analyze(_grid([[0, 1], [0, 0]]), action="ACTION3")
        assert movement is not None
        assert movement.action == "ACTION3"
        assert movement.pixels_changed == 2  # Two cells changed.


class TestActionEffects:
    def test_action_effect_recorded(self) -> None:
        fa = FrameAnalyzer()
        fa.analyze(_grid([[1, 0], [0, 0]]))
        fa.analyze(_grid([[0, 0], [1, 0]]), action="DOWN")
        summary = fa.get_action_summary()
        assert "DOWN" in summary
        assert summary["DOWN"]["count"] == 1.0

    def test_summary_averages_over_multiple_calls(self) -> None:
        fa = FrameAnalyzer()
        # Three same-shape frames with two DOWN-tagged transitions.
        fa.analyze(_grid([[1, 0, 0], [0, 0, 0], [0, 0, 0]]))
        fa.analyze(_grid([[0, 0, 0], [1, 0, 0], [0, 0, 0]]), action="DOWN")
        fa.analyze(_grid([[0, 0, 0], [0, 0, 0], [1, 0, 0]]), action="DOWN")
        summary = fa.get_action_summary()
        assert summary["DOWN"]["count"] == 2.0

    def test_unspecified_action_marked_unknown(self) -> None:
        fa = FrameAnalyzer()
        fa.analyze(_grid([[1, 0], [0, 0]]))
        movement = fa.analyze(_grid([[0, 1], [0, 0]]))
        assert movement is not None
        assert movement.action == "unknown"
        # Unspecified action is NOT recorded in action_effects.
        assert fa.get_action_summary() == {}


class TestSuggestAction:
    def test_no_history_no_actions(self) -> None:
        fa = FrameAnalyzer()
        assert fa.suggest_action([]) is None

    def test_untested_action_wins(self) -> None:
        class _A:
            def __init__(self, name: str) -> None:
                self.name = name

        fa = FrameAnalyzer()
        # Train one action.
        fa.analyze(_grid([[1, 0]]))
        fa.analyze(_grid([[0, 1]]), action="ACTION_TESTED")
        actions = [_A("ACTION_TESTED"), _A("ACTION_NEW")]
        suggested = fa.suggest_action(actions)
        # ACTION_NEW has zero history → highest priority.
        assert suggested.name == "ACTION_NEW"

    def test_least_used_wins_among_tested(self) -> None:
        class _A:
            def __init__(self, name: str) -> None:
                self.name = name

        fa = FrameAnalyzer()
        fa.analyze(_grid([[1, 0]]))
        # Use ACTION_A twice, ACTION_B once.
        fa.analyze(_grid([[0, 1]]), action="ACTION_A")
        fa.analyze(_grid([[1, 0]]), action="ACTION_A")
        fa.analyze(_grid([[0, 1]]), action="ACTION_B")
        actions = [_A("ACTION_A"), _A("ACTION_B")]
        suggested = fa.suggest_action(actions)
        assert suggested.name == "ACTION_B"


class TestResetForNewLevel:
    def test_clears_position_keeps_effects(self) -> None:
        fa = FrameAnalyzer()
        fa.analyze(_grid([[1, 0]]))
        fa.analyze(_grid([[0, 1]]), action="DOWN")
        assert fa.get_action_summary()  # non-empty

        fa.reset_for_new_level()
        # Effects survive.
        assert "DOWN" in fa.get_action_summary()
        # Position tracking is cleared.
        assert fa.frame_count == 0
        assert fa.visited_positions == frozenset()


class TestMovementInfoDataclass:
    def test_construct_minimal(self) -> None:
        m = MovementInfo(
            action="ACTION1",
            pixels_changed=4,
            min_row=0,
            max_row=2,
            min_col=0,
            max_col=2,
        )
        assert m.action == "ACTION1"
        assert m.direction_row == 0
        assert m.direction_col == 0
