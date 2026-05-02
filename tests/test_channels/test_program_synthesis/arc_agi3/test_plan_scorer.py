# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-19 Hebel L — plan_scorer tests."""

from __future__ import annotations

import numpy as np

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
