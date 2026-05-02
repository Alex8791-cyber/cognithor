# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-13 — compact goal inferer for the new EpisodeMemory shape.

The legacy :mod:`cognithor.arc.goal_inference` module operated on the
old ``transitions`` memory model with explicit ``resulted_in_win``
flags; the new :class:`EpisodeMemory` only stores
``(grid, action_name, levels_completed)``.

This module implements a slimmer, evidence-driven inferer that reads
the new memory + an optional :class:`FrameAnalyzer` and produces a
short text the LLM can quote into its decision. The goal is **not**
to enumerate every possible objective — it's to give the LLM a
running hypothesis it can confirm or override:

* ``"Last 3 steps progressed levels via ACTION3"`` — REACH_STATE
  hint.
* ``"No level progress in 12 steps; recent ACTION6 pattern caused
  most pixel changes"`` — explore-this-action hint.
* ``"Stuck for 8 frames after ACTION1; consider RESET"`` — stuck hint.

The output is a single string. Empty inputs produce a placeholder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cognithor.channels.program_synthesis.arc_agi3.episode_memory import (
        EpisodeMemory,
    )
    from cognithor.channels.program_synthesis.arc_agi3.frame_analyzer import (
        FrameAnalyzer,
    )

__all__ = ["GoalInferer", "infer_goal_summary"]


class GoalInferer:
    """Evidence-driven goal-hint generator for the LLM prompt.

    Stateless across calls — the inferer reads the memory + analyzer
    fresh on every :meth:`infer` invocation. Designed to run on every
    ``choose_action`` (cost: a few list comprehensions + numpy diffs).
    """

    def __init__(self, *, recent_window: int = 8) -> None:
        if recent_window < 2:
            raise ValueError(f"recent_window must be >= 2, got {recent_window}")
        self._recent_window = recent_window

    def infer(
        self,
        memory: EpisodeMemory,
        frame_analyzer: FrameAnalyzer | None = None,
    ) -> str:
        """Return a one-paragraph LLM-readable goal summary."""
        steps = memory.window(self._recent_window)
        if not steps:
            return "(no observations yet — explore broadly)"

        hypotheses: list[str] = []

        # Hypothesis 1 — level progression in the recent window.
        # ``window`` returns most-recent first; level_jumps[i] is True if
        # going FROM ``steps[i+1]`` to ``steps[i]`` increased levels.
        level_jumps = [
            steps[i].action_name
            for i in range(len(steps) - 1)
            if steps[i].levels_completed > steps[i + 1].levels_completed
        ]
        if level_jumps:
            most_common = max(set(level_jumps), key=level_jumps.count)
            hypotheses.append(
                f"REACH_STATE: {len(level_jumps)} level-up event(s) in last "
                f"{len(steps)} steps; most-effective action: {most_common}"
            )

        # Hypothesis 2 — pixel-change pattern via FrameAnalyzer.
        if frame_analyzer is not None:
            summary = frame_analyzer.get_action_summary()
            if summary:
                # Pick the action with the highest avg_pixels — likely the
                # "broad-effect" action (clears board / moves sprite).
                top = max(summary.items(), key=lambda kv: kv[1]["avg_pixels"])
                avg_px = top[1]["avg_pixels"]
                if avg_px > 50:
                    hypotheses.append(
                        f"CLEAR_BOARD/BROAD_EFFECT: {top[0]} averages "
                        f"{avg_px:.0f} pixel changes per step"
                    )
                elif avg_px > 0:
                    hypotheses.append(
                        f"FINE_CONTROL: {top[0]} averages "
                        f"{avg_px:.1f} pixel changes — likely movement"
                    )

        # Hypothesis 3 — stuck signal.
        if len(steps) >= self._recent_window:
            last_grids = [s.grid for s in steps]
            shapes = {g.shape for g in last_grids}
            if len(shapes) == 1:
                # All same shape — quantify identity vs movement.
                base = last_grids[0]
                identical = sum(1 for g in last_grids[1:] if np.array_equal(g, base))
                if identical >= len(last_grids) - 1:
                    hypotheses.append(
                        f"STUCK: last {len(last_grids)} frames identical; "
                        "consider RESET or a different action family"
                    )

        # Hypothesis 4 — no progress despite many actions.
        if not level_jumps and len(steps) >= self._recent_window:
            hypotheses.append(
                f"NO_PROGRESS: {len(steps)} actions without a level-up; "
                "try an under-explored action"
            )

        if not hypotheses:
            return "(no strong signal yet)"
        return "; ".join(hypotheses)


# Convenience standalone function for one-shot use.


def infer_goal_summary(
    memory: EpisodeMemory,
    frame_analyzer: FrameAnalyzer | None = None,
    *,
    recent_window: int = 8,
) -> str:
    """Convenience wrapper: ``GoalInferer(recent_window=...).infer(...)``."""
    return GoalInferer(recent_window=recent_window).infer(memory, frame_analyzer)
