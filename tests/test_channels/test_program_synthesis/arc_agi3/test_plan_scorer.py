# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-19 Hebel L — plan_scorer tests."""

from __future__ import annotations

import numpy as np
import pytest

from cognithor.channels.program_synthesis.arc_agi3.episode_memory import EpisodeMemory
from cognithor.channels.program_synthesis.arc_agi3.plan_scorer import (
    pick_best_plan,
    score_plan,
)
from cognithor.channels.program_synthesis.arc_agi3.planning_decoder import PlanStep
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)


def _g(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


class TestScorePlanComponents:
    def test_empty_plan_zero(self) -> None:
        assert score_plan([]) == 0.0

    def test_pure_repetition_low_score(self) -> None:
        plan = [PlanStep("ACTION6", reasoning="x") for _ in range(5)]
        s = score_plan(plan, available_action_names=("ACTION6",))
        # diversity 0.2, all ACTION6 click without coords → targetedness 0
        assert s < 0.5

    def test_diverse_plan_high_score(self) -> None:
        plan = [
            PlanStep("ACTION3", reasoning="explore"),
            PlanStep("ACTION4", reasoning="explore"),
            PlanStep("ACTION7", reasoning="explore"),
            PlanStep("ACTION6", data={"x": 12, "y": 8}, reasoning="targeted click"),
            PlanStep("ACTION3", reasoning="follow-up"),
        ]
        s = score_plan(
            plan,
            available_action_names=("ACTION3", "ACTION4", "ACTION6", "ACTION7"),
        )
        assert s > 0.7

    def test_targetedness_penalises_naked_click(self) -> None:
        with_target = [PlanStep("ACTION6", data={"x": 10, "y": 5}, reasoning="r")]
        without = [PlanStep("ACTION6", reasoning="r")]
        s_with = score_plan(with_target, available_action_names=("ACTION6",))
        s_without = score_plan(without, available_action_names=("ACTION6",))
        assert s_with > s_without

    def test_invalid_action_penalised(self) -> None:
        plan = [
            PlanStep("ACTION3", reasoning="r"),
            PlanStep("ACTION_BOGUS", reasoning="r"),
        ]
        s = score_plan(plan, available_action_names=("ACTION3", "ACTION6"))
        # validity = 1/2 = 0.5; the rest of the score is fine
        assert s < 0.85  # capped because validity component drags it down

    def test_anti_repetition_kicks_in_on_dominant_recent(self) -> None:
        # Memory dominated by ACTION6 (5 of last 5)
        m = EpisodeMemory()
        for _ in range(5):
            m.append(grid=_g([[1]]), action_name="ACTION6", levels_completed=0)
        # Plan starting with same action → score 0 from the anti-rep component
        plan_same = [PlanStep("ACTION6", reasoning="r")]
        plan_diff = [PlanStep("ACTION3", reasoning="r")]
        s_same = score_plan(
            plan_same,
            memory=m,
            available_action_names=("ACTION3", "ACTION6"),
        )
        s_diff = score_plan(
            plan_diff,
            memory=m,
            available_action_names=("ACTION3", "ACTION6"),
        )
        assert s_diff > s_same

    def test_pix_delta_safety_penalises_repeat_after_high_delta(self) -> None:
        """Hebel N: if the LAST action produced pixΔ>500 AND the plan's
        first action repeats it, the plan is multiplicatively penalised
        (0.5×). A plan that picks a different first-action wins.
        """
        # Build memory: one tame ACTION3 step, then one ACTION7 that
        # flipped the entire 64×64 grid (pixΔ ~= 4096).
        m = EpisodeMemory()
        before = np.zeros((64, 64), dtype=np.int8)
        m.append(grid=before, action_name="ACTION3", levels_completed=0)
        after = np.full((64, 64), 4, dtype=np.int8)
        m.append(grid=after, action_name="ACTION7", levels_completed=0)

        # Plan A repeats the destructive action; Plan B picks something else.
        plan_repeat = [
            PlanStep("ACTION7", data={"x": 10, "y": 10}, reasoning="r"),
            PlanStep("ACTION3", reasoning="r"),
        ]
        plan_pivot = [
            PlanStep("ACTION3", reasoning="r"),
            PlanStep("ACTION7", data={"x": 10, "y": 10}, reasoning="r"),
        ]
        actions = ("ACTION3", "ACTION7")
        s_repeat = score_plan(plan_repeat, memory=m, available_action_names=actions)
        s_pivot = score_plan(plan_pivot, memory=m, available_action_names=actions)
        # Pivot must beat repeat. Same component composition modulo
        # the pix-delta gate (0.5× for repeat, 1.0× for pivot).
        assert s_pivot > s_repeat
        # Pivot must be roughly twice the repeat (since the only delta
        # is the multiplicative gate).
        assert s_repeat == pytest.approx(s_pivot * 0.5, rel=1e-6)

    def test_pix_delta_safety_no_penalty_when_last_delta_low(self) -> None:
        """Hebel N: if the last action produced a SMALL pixΔ, the gate
        is inactive — repeat is fine.
        """
        m = EpisodeMemory()
        m.append(grid=_g([[0]]), action_name="ACTION3", levels_completed=0)
        # Same grid → pixΔ=0, well below the 500-cell threshold.
        m.append(grid=_g([[0]]), action_name="ACTION7", levels_completed=0)

        plan_repeat = [PlanStep("ACTION7", data={"x": 1, "y": 1}, reasoning="r")]
        plan_pivot = [PlanStep("ACTION3", reasoning="r")]
        actions = ("ACTION3", "ACTION7")
        s_repeat = score_plan(plan_repeat, memory=m, available_action_names=actions)
        s_pivot = score_plan(plan_pivot, memory=m, available_action_names=actions)
        # Both should be in the same ballpark; the repeat is targeted +
        # different colour so it can even win on pure quality.
        assert s_repeat >= s_pivot * 0.95

    def test_pix_delta_safety_two_consecutive_high_penalises_any_first_action(self) -> None:
        """Hebel N broadened (Run #27 finding): when the LAST TWO
        recorded actions BOTH produced pixΔ>500 (destructive-escalation
        regime), ANY plan-first-action gets the 0.5× penalty —
        regardless of whether it matches the last action. This catches
        the case the LLM K=3 candidates rotate ACTION3/4/6/7 to dodge
        the single-action gate while still escalating.
        """
        m = EpisodeMemory()
        zero = np.zeros((64, 64), dtype=np.int8)
        big_a = np.full((64, 64), 4, dtype=np.int8)
        big_b = np.full((64, 64), 7, dtype=np.int8)
        # Two consecutive high-pixΔ transitions (~4096 cells each).
        m.append(grid=zero, action_name="ACTION1", levels_completed=0)
        m.append(grid=big_a, action_name="ACTION3", levels_completed=0)
        m.append(grid=big_b, action_name="ACTION6", levels_completed=0)

        # Plan first action is COMPLETELY DIFFERENT from the last
        # action (ACTION6) — so the single-action repeat trigger does
        # NOT fire. The two-consecutive trigger MUST still apply.
        plan_other = [
            PlanStep("ACTION7", data={"x": 5, "y": 5}, reasoning="r"),
            PlanStep("ACTION3", reasoning="r"),
        ]
        actions = ("ACTION3", "ACTION6", "ACTION7")
        s_other = score_plan(plan_other, memory=m, available_action_names=actions)
        # Compare against the same plan scored without memory (gate
        # inactive). The gate is the only multiplicative delta.
        s_baseline = score_plan(plan_other, available_action_names=actions)
        assert s_other == pytest.approx(s_baseline * 0.5, rel=1e-6)


class TestResetBonus:
    """Sprint-19 Hebel R — additive +0.30 RESET-bonus when the agent is
    both stalled (≥15 steps at current level) and the last action was
    destructive (pixΔ>500). Score clamped to [0, 1].
    """

    def test_reset_plan_gets_bonus_when_stalled_and_destructive(self) -> None:
        m = EpisodeMemory()
        # 16 prior frames at level 0 → "stalled" (≥15)
        for _ in range(15):
            m.append(
                grid=np.zeros((64, 64), dtype=np.int8),
                action_name="ACTION3",
                levels_completed=0,
            )
        # The 16th frame is the destructive one — pixΔ=600 from
        # this transition (10 rows × 60 cols = 600 cells flipped).
        destructive = np.zeros((64, 64), dtype=np.int8)
        destructive[:10, :60] = 4
        m.append(grid=destructive, action_name="ACTION6", levels_completed=0)

        actions = ("RESET", "ACTION3", "ACTION6")
        plan_reset = [PlanStep("RESET", reasoning="restart")]
        plan_other = [PlanStep("ACTION3", reasoning="explore")]
        s_reset = score_plan(plan_reset, memory=m, available_action_names=actions)
        s_other = score_plan(plan_other, memory=m, available_action_names=actions)
        # The two plans have similar quality components, so the +0.30
        # RESET-bonus must put plan_reset clearly ahead.
        assert s_reset > s_other
        assert s_reset - s_other >= 0.20

    def test_no_bonus_when_reset_not_in_available(self) -> None:
        m = EpisodeMemory()
        for _ in range(15):
            m.append(
                grid=np.zeros((64, 64), dtype=np.int8),
                action_name="ACTION3",
                levels_completed=0,
            )
        destructive = np.zeros((64, 64), dtype=np.int8)
        destructive[:10, :60] = 4
        m.append(grid=destructive, action_name="ACTION6", levels_completed=0)

        # RESET NOT in available_action_names → bonus never fires
        # even if the plan claims to start with RESET.
        actions = ("ACTION3", "ACTION6")
        plan_reset = [PlanStep("RESET", reasoning="r")]
        s_reset = score_plan(plan_reset, memory=m, available_action_names=actions)
        # validity=0 (RESET not in actions) → score=0 regardless of bonus.
        assert s_reset == 0.0

    def test_no_bonus_when_not_stalled(self) -> None:
        m = EpisodeMemory()
        # Only 5 prior frames at level 0 → below the 15-step threshold.
        for _ in range(5):
            m.append(
                grid=np.zeros((64, 64), dtype=np.int8),
                action_name="ACTION3",
                levels_completed=0,
            )
        destructive = np.zeros((64, 64), dtype=np.int8)
        destructive[:10, :60] = 4
        m.append(grid=destructive, action_name="ACTION6", levels_completed=0)

        actions = ("RESET", "ACTION3", "ACTION6")
        plan_reset = [PlanStep("RESET", reasoning="r")]
        s_reset_with_memory = score_plan(plan_reset, memory=m, available_action_names=actions)
        # Same plan but no memory at all (so the bonus path can't fire).
        s_reset_no_memory = score_plan(plan_reset, available_action_names=actions)
        # Bonus should NOT fire; the two scores match.
        assert s_reset_with_memory == pytest.approx(s_reset_no_memory, rel=1e-6)

    def test_no_bonus_when_last_pix_delta_low(self) -> None:
        m = EpisodeMemory()
        for _ in range(20):
            m.append(
                grid=np.zeros((64, 64), dtype=np.int8),
                action_name="ACTION3",
                levels_completed=0,
            )
        # Same grid → pixΔ=0 (below the 500 threshold).
        m.append(
            grid=np.zeros((64, 64), dtype=np.int8),
            action_name="ACTION6",
            levels_completed=0,
        )

        actions = ("RESET", "ACTION3", "ACTION6")
        plan_reset = [PlanStep("RESET", reasoning="r")]
        s_with_memory = score_plan(plan_reset, memory=m, available_action_names=actions)
        s_no_memory = score_plan(plan_reset, available_action_names=actions)
        assert s_with_memory == pytest.approx(s_no_memory, rel=1e-6)


class TestPickBestPlan:
    def test_empty_list_returns_empty(self) -> None:
        assert pick_best_plan([]) == ([], "")

    def test_picks_highest_scoring(self) -> None:
        weak = ([PlanStep("ACTION6") for _ in range(5)], "r1")
        strong = (
            [
                PlanStep("ACTION3", reasoning="explore"),
                PlanStep("ACTION4", reasoning="explore"),
                PlanStep("ACTION6", data={"x": 5, "y": 5}, reasoning="target"),
            ],
            "r2",
        )
        best, reasoning = pick_best_plan(
            [weak, strong],
            available_action_names=("ACTION3", "ACTION4", "ACTION6"),
        )
        assert best == strong[0]
        assert reasoning == "r2"

    def test_ties_break_by_first_appearance(self) -> None:
        # Two identical plans → first one wins
        p = [PlanStep("ACTION3", reasoning="r")]
        a = (p, "first")
        b = (p, "second")
        best, reasoning = pick_best_plan(
            [a, b],
            available_action_names=("ACTION3",),
        )
        assert reasoning == "first"
