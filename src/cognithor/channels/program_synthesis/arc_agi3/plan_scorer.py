# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-19 Hebel L — deterministic plan-quality scorer.

Single-LLM-call planning is greedy: the model emits whatever sequence
its temperature=0 greedy decode produces. We have no way to tell
whether it's a *good* plan vs a *plausible-looking but losing* plan.

This module gives a **post-hoc deterministic score** that any plan
candidate can be evaluated against, using nothing but the
``EpisodeMemory`` + ``FrameAnalyzer`` (already wired in our agents).
The score is heuristic — not a real game simulator — but it's enough
to penalise the obvious failure modes we've seen:

* PURE-REPETITION plans (same action 5×): low score
* ACTION6-with-no-coords plans: low score (Sprint-17 finding)
* DEAD/saturated-action plans: low score (Sprint-16 finding)
* High pixΔ-monotonic-growth plans: medium-low (Sprint-18 finding —
  these usually end in GAME_OVER)
* Diverse + uses TARGETED ACTION6 (with x/y data) + has reasoning: high

Designed to be cheap (microseconds) so a generator that produces
K=3-5 candidate plans can score them all and pick the best, without
extra LLM calls. The downstream agent gets the highest-scoring plan
to execute.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cognithor.channels.program_synthesis.arc_agi3.episode_memory import (
        EpisodeMemory,
    )
    from cognithor.channels.program_synthesis.arc_agi3.planning_decoder import (
        PlanStep,
    )


def score_plan(
    plan: list[PlanStep],
    *,
    memory: EpisodeMemory | None = None,
    available_action_names: tuple[str, ...] = (),
) -> float:
    """Heuristic score in ``[0.0, 1.0]``; higher is better.

    Components (all clamped to [0, 1] then averaged):

    1. **Diversity** — distinct actions in the plan / total plan length.
       A plan picking 3 different actions in 5 steps scores 0.6.
    2. **Validity** — plan steps in available_action_names.
       Penalises plans naming actions the game doesn't expose.
    3. **Targetedness of clicks** — fraction of ACTION6/ACTION7 entries
       in the plan that have ``data.x`` / ``data.y`` set. Click without
       coords is the Sprint-17 anti-pattern.
    4. **Anti-repetition vs recent memory** — if the plan would be the
       3rd consecutive call to repeat the same action that just
       dominated the last 5 memory entries, score 0; otherwise 1.
    5. **Reasoning quality proxy** — fraction of plan steps that have
       any ``reasoning`` text. Coarse but catches "naked" plans.

    Empty plans score 0.
    """
    if not plan:
        return 0.0
    n = len(plan)

    # 1. Diversity
    distinct = len({s.action_name for s in plan})
    diversity = min(1.0, distinct / max(1, n))

    # 2. Validity (skip if available_action_names not provided)
    if available_action_names:
        valid_count = sum(1 for s in plan if s.action_name in available_action_names)
        validity = valid_count / n
    else:
        validity = 1.0

    # 3. Targetedness of clicks
    click_steps = [s for s in plan if s.action_name in ("ACTION6", "ACTION7")]
    if click_steps:
        targeted = sum(
            1
            for s in click_steps
            if s.data is not None
            and "x" in s.data
            and "y" in s.data
            and (int(s.data["x"]) != 0 or int(s.data["y"]) != 0)
        )
        targetedness = targeted / len(click_steps)
    else:
        targetedness = 1.0  # no clicks = nothing to penalise

    # 4. Anti-repetition vs recent memory
    anti_repetition = 1.0
    if memory is not None and len(memory) >= 5:
        recent = memory.window(5)
        recent_actions = [s.action_name for s in recent]
        # Check if a single action dominates the recent window AND is the
        # plan's first step.
        from collections import Counter

        c = Counter(recent_actions)
        if c:
            most_recent_action, recent_count = c.most_common(1)[0]
            if recent_count >= 4 and plan[0].action_name == most_recent_action:
                anti_repetition = 0.0

    # 5. Reasoning quality proxy
    with_reasoning = sum(1 for s in plan if s.reasoning.strip())
    reasoning = with_reasoning / n

    # Sprint-19 Hebel T — exploration bonus for first-time-action.
    # Run #28's 64-step bp35 episode used ACTION3/4/6/7 ~equally but
    # never RESET. Plans whose first action has NEVER been recorded in
    # this episode's memory get a +0.20 post-hoc additive bonus, so
    # candidate plans that explore an unused action class can beat
    # equally-quality plans that re-use already-tried actions on a
    # thin diversity margin. ``available_action_names`` validates that
    # the action is actually offered (don't reward made-up actions).
    exploration_bonus = 0.0
    first_action = plan[0].action_name
    if (
        memory is not None
        and len(memory) > 0
        and (not available_action_names or first_action in available_action_names)
    ):
        try:
            seen_actions = {s.action_name for s in memory.window(80)}
            if first_action not in seen_actions:
                exploration_bonus = 0.20
        except Exception:
            pass

    # 6. Sprint-19 Hebel N — pixΔ-safety gate.
    #
    # Run #26c finding (motivation): the LLM queued plans whose first
    # action repeated the destructive last action (pixΔ>500), then
    # GAME_OVER. Initial Hebel N caught that single-action repeat case.
    #
    # Run #27 finding (broadening): with K=3 sampling + Hebel L active
    # the LLM stopped repeating the *same* action — but it now rotates
    # between ACTION3/4/6/7 each producing pixΔ ~525-636, ending in
    # the same destructive cascade (528 → 525 → 525 → 525 → 636 →
    # GAME_OVER at step 48). The single-action gate didn't fire
    # because the action keeps changing.
    #
    # Broadened gate: TWO consecutive prior steps with pixΔ>500 means
    # the agent is in the destructive-escalation regime regardless of
    # action class — ANY plan-first-action gets the multiplicative
    # 0.5× penalty. The prior single-action repeat case still applies
    # as a separate trigger (relevant when only the last single step
    # had pixΔ>500).
    pix_delta_safety = 1.0
    if memory is not None and len(memory) >= 2:
        try:
            import numpy as _np

            recent = memory.window(3)
            last_after, last_before = recent[0], recent[1]
            last_pix_delta = (
                int(_np.sum(last_before.grid != last_after.grid))
                if last_before.grid.shape == last_after.grid.shape
                else 0
            )
            second_pix_delta = 0
            if len(recent) >= 3:
                second_after, second_before = recent[1], recent[2]
                if second_before.grid.shape == second_after.grid.shape:
                    second_pix_delta = int(_np.sum(second_before.grid != second_after.grid))

            two_consecutive_high = last_pix_delta > 500 and second_pix_delta > 500
            single_action_repeat = (
                last_pix_delta > 500 and plan[0].action_name == last_after.action_name
            )
            if two_consecutive_high or single_action_repeat:
                pix_delta_safety = 0.5
        except Exception:
            pass

    # Validity + anti_repetition + pix_delta_safety are multiplicative
    # gates (plan with invalid actions, stuck-action repetition, or
    # destructive-action escalation is fundamentally broken).
    # Diversity, targetedness, reasoning are additive quality
    # components averaged.
    additive_avg = (diversity + targetedness + reasoning) / 3.0
    base = additive_avg * validity * anti_repetition * pix_delta_safety
    # Sprint-19 Hebel T: post-hoc additive exploration boost; clamp
    # to [0, 1] so the bonus never inflates beyond a legal score.
    return min(1.0, base + exploration_bonus)


def pick_best_plan(
    candidates: list[tuple[list[PlanStep], str]],
    *,
    memory: EpisodeMemory | None = None,
    available_action_names: tuple[str, ...] = (),
) -> tuple[list[PlanStep], str]:
    """Return the highest-scoring plan from a list of ``(plan, reasoning)``.

    Empty input → empty plan with empty reasoning. Ties broken by first
    appearance (so a temperature-0 baseline candidate wins ties).
    """
    if not candidates:
        return ([], "")
    best_idx = 0
    best_score = -1.0
    for i, (plan, _reasoning) in enumerate(candidates):
        s = score_plan(
            plan,
            memory=memory,
            available_action_names=available_action_names,
        )
        if s > best_score:
            best_score = s
            best_idx = i
    return candidates[best_idx]


__all__ = ["pick_best_plan", "score_plan"]
