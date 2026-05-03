# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-14 PR-1 — Multi-step planning LLM decoder.

The Sprint-12/13 :class:`LLMActionDecoder` calls the LLM *once per
step*. With Qwen3.6-27B NVFP4 at ~30-40s per inference and an
80-step ARC-AGI-3 episode, a single game costs ~40-50 min of pure
inference time and the LLM never gets to reason about *sequences* of
actions — every decision is one-shot.

:class:`PlanningLLMActionDecoder` flips the loop: the LLM is asked
for **the next N actions as a plan**. The agent executes the plan
one action at a time, only consulting the LLM again when:

* the plan queue is empty;
* the level just changed (level transition invalidates the plan);
* the agent is stuck (StuckDetector flagged ≥ threshold no-op
  frames — likely the plan was wrong);
* the user / agent explicitly invalidates the plan via
  :meth:`replan_now`.

Cost model: with ``plan_horizon=5`` the LLM-call rate drops by ~5×
on smooth runs (≈9 LLM calls / 80 steps instead of ~80). Quality
upside: the LLM can quote action *sequences* like "ACTION3 to move
up, then ACTION6 at (5, 5) to interact" instead of single shots —
which actually matches the ARC-AGI-3 game semantics.

The plan format is a JSON array, each entry a single-action dict::

    [
      {"action": "ACTION3", "data": null,                  "reasoning": "..."},
      {"action": "ACTION6", "data": {"x": 12, "y": 18},    "reasoning": "..."},
      ...
    ]

Falls back to the wrapped :class:`ActionDecoder` (default
:class:`DSLActionDecoder`) on parse failure / unknown action /
LLM-side exception, exactly like the single-step decoder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cognithor.channels.program_synthesis.arc_agi3.action_decoder import ActionDecoder
from cognithor.channels.program_synthesis.arc_agi3.dsl_action_decoder import (
    DSLActionDecoder,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from cognithor.channels.program_synthesis.arc_agi3.episode_memory import (
        EpisodeMemory,
    )
    from cognithor.channels.program_synthesis.arc_agi3.frame_analyzer import (
        FrameAnalyzer,
    )
    from cognithor.channels.program_synthesis.arc_agi3.frame_bridge import FrameBridge
    from cognithor.channels.program_synthesis.arc_agi3.goal_inferer import GoalInferer
    from cognithor.channels.program_synthesis.arc_agi3.llm_action_decoder import (
        FrameContext,
    )
    from cognithor.channels.program_synthesis.arc_agi3.protocol import (
        FrameDataProtocol,
        GameActionProtocol,
    )

__all__ = [
    "PlanStep",
    "PlanningChoiceFn",
    "PlanningLLMActionDecoder",
]


@dataclass(frozen=True)
class PlanStep:
    """One action in a plan: name + optional click data + free-text reason."""

    action_name: str
    data: dict[str, int] | None = None  # {"x": col, "y": row} for ACTION6/7
    reasoning: str = ""


# A planning choice-fn returns ``(plan, top_level_reasoning)``. The plan
# is a list of :class:`PlanStep`; the top_level_reasoning is logged once
# on plan acceptance for audit/debug.
PlanningChoiceFn = "Callable[[FrameContext], tuple[list[PlanStep], str]]"


@dataclass
class _PlanState:
    """Internal cursor: the queue + index + the prompt context the
    plan was generated against."""

    plan: list[PlanStep] = field(default_factory=list)
    cursor: int = 0
    levels_when_planned: int = 0


class PlanningLLMActionDecoder(ActionDecoder):
    """Multi-step LLM planner.

    Construction mirrors :class:`LLMActionDecoder` but takes a
    :data:`PlanningChoiceFn` that returns a plan instead of a single
    action.
    """

    def __init__(
        self,
        *,
        bridge: FrameBridge,
        memory: EpisodeMemory,
        choice_fn: Callable[[FrameContext], tuple[list[PlanStep], str]],
        fallback: ActionDecoder | None = None,
        history_steps: int = 8,
        plan_horizon: int = 5,
        frame_analyzer: FrameAnalyzer | None = None,
        goal_inferer: GoalInferer | None = None,
        state_counter: Any = None,
        action_streak_detector: Any = None,
    ) -> None:
        if plan_horizon < 1:
            raise ValueError(f"plan_horizon must be >= 1, got {plan_horizon}")
        self._bridge = bridge
        self._memory = memory
        self._choice_fn = choice_fn
        self._fallback = fallback if fallback is not None else DSLActionDecoder(memory=memory)
        self._history_steps = history_steps
        self._plan_horizon = plan_horizon
        self._frame_analyzer = frame_analyzer
        self._goal_inferer = goal_inferer
        # Sprint-19 Hebel C: mirror the LLMActionDecoder's anti-loop
        # signals into the planning decoder. Without these, a planning
        # LLM that emits "ACTION6 ACTION6 ACTION6 ACTION6 ACTION6" as
        # a 5-step plan executes ALL 5 sequentially (post Sprint-16 the
        # single-step decoder caught this; planning decoder didn't).
        self._state_counter = state_counter
        self._action_streak_detector = action_streak_detector
        self._state = _PlanState()

    @property
    def plan_remaining(self) -> int:
        """How many plan steps are still queued (test/debug accessor)."""
        return max(0, len(self._state.plan) - self._state.cursor)

    def _wire_complex_action_data(
        self,
        action: Any,
        frames: list[FrameDataProtocol],
        latest_frame: FrameDataProtocol,
    ) -> None:
        """No-op override: data already set by ``pick_action`` from the plan.

        The base :class:`ActionDecoder._wire_complex_action_data` clobbers
        whatever was set with ``(0, 0)`` by default. The planner-driven
        path already has correct ``(x, y)`` from the LLM's plan, so we
        skip this step entirely.
        """
        del action, frames, latest_frame

    def replan_now(self) -> None:
        """Invalidate the current plan; the next ``pick_action`` call
        will trigger a fresh LLM consultation."""
        self._state = _PlanState()

    def pick_action(
        self,
        frames: list[FrameDataProtocol],
        latest_frame: FrameDataProtocol,
        available_actions: list[GameActionProtocol],
    ) -> tuple[GameActionProtocol, str]:
        # Invalidate plan on level transition.
        if self._state.plan and latest_frame.levels_completed != self._state.levels_when_planned:
            self._state = _PlanState()

        # Re-plan when the queue is empty.
        if self.plan_remaining == 0:
            new_plan = self._consult_llm(frames, latest_frame, available_actions)
            if new_plan is None:
                # LLM failed → single-step fallback.
                return self._fallback.pick_action(frames, latest_frame, available_actions)
            self._state = _PlanState(
                plan=new_plan,
                cursor=0,
                levels_when_planned=latest_frame.levels_completed,
            )

        # Pop the next plan step.
        step = self._state.plan[self._state.cursor]
        self._state.cursor += 1

        # Sprint-19 Hebel C: anti-loop check for the chosen plan-step.
        # If the same action has dominated the recent window without
        # level progress, the plan is reinforcing a stuck pattern —
        # skip this step + invalidate rest of plan + force fallback
        # so the LLM is consulted again next call.
        if self._action_streak_detector is not None:
            stuck = self._action_streak_detector.dominant_stuck_action(self._memory)
            if (
                stuck is not None
                and stuck == step.action_name
                and any(a.name != stuck for a in available_actions)
            ):
                self._state = _PlanState()
                fallback_action, fb_reasoning = self._fallback.pick_action(
                    frames, latest_frame, available_actions
                )
                return fallback_action, (
                    f"PlanningLLMActionDecoder anti-loop override: plan named "
                    f"{step.action_name!r} but it dominated the recent window "
                    f"without level progress; falling back to {fallback_action.name!r}"
                )

        # Match the planned action name against the per-frame whitelist.
        # If the LLM hallucinated an unavailable action, fall back to DSL
        # for THIS step but keep the rest of the plan (it might recover).
        for action in available_actions:
            if action.name == step.action_name:
                if step.data and hasattr(action, "set_data"):
                    action.set_data({"x": int(step.data["x"]), "y": int(step.data["y"])})
                reason = (
                    f"plan-step {self._state.cursor}/{len(self._state.plan)}: "
                    f"{step.action_name}" + (f" → {step.reasoning}" if step.reasoning else "")
                )
                return action, reason

        # Plan named an unknown action → invalidate the rest, fall back.
        self._state = _PlanState()
        fallback_action, fb_reasoning = self._fallback.pick_action(
            frames, latest_frame, available_actions
        )
        return fallback_action, (
            f"PlanningLLMActionDecoder fallback (plan named "
            f"{step.action_name!r} which is not available): {fb_reasoning}"
        )

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _consult_llm(
        self,
        frames: list[FrameDataProtocol],
        latest_frame: FrameDataProtocol,
        available_actions: list[GameActionProtocol],
    ) -> list[PlanStep] | None:
        # Lazy-import to keep this module importable without llm_action_decoder.
        from cognithor.channels.program_synthesis.arc_agi3.llm_action_decoder import (
            FrameContext,
            summarise_action_effects,
            summarise_history,
        )

        try:
            grid = self._bridge.extract_grid(latest_frame)
        except Exception:
            return None
        action_effects_summary = (
            summarise_action_effects(self._frame_analyzer)
            if self._frame_analyzer is not None
            else ""
        )
        goal_summary = (
            self._goal_inferer.infer(self._memory, self._frame_analyzer)
            if self._goal_inferer is not None
            else ""
        )
        # Construct FrameContext, omitting Sprint-13 fields if not present
        # in this build (goal_summary + game_id were added in #307).
        ctx_kwargs: dict[str, Any] = {
            "grid": grid,
            "available_action_names": [a.name for a in available_actions],
            "history_summary": summarise_history(self._memory, self._history_steps),
            "levels_completed": latest_frame.levels_completed,
            "win_levels": latest_frame.win_levels,
            "action_effects_summary": action_effects_summary,
        }
        from dataclasses import fields as _dc_fields

        existing = {f.name for f in _dc_fields(FrameContext)}
        if "goal_summary" in existing:
            ctx_kwargs["goal_summary"] = goal_summary
        if "game_id" in existing:
            ctx_kwargs["game_id"] = getattr(latest_frame, "game_id", "")
        # Sprint-19 Hebel B + D: populate prev-frame + structured-text
        # fields if FrameContext schema supports them. The vision
        # planning factory reads these to build a 2-image multimodal
        # prompt + cluster/delta annotation.
        if "prev_grid" in existing and len(self._memory) > 0:
            try:
                prev_step = self._memory.window(1)[0]
                ctx_kwargs["prev_grid"] = prev_step.grid
                ctx_kwargs["prev_action_name"] = prev_step.action_name
            except Exception:
                pass
        if "cluster_summary" in existing:
            try:
                from cognithor.channels.program_synthesis.arc_agi3.state_renderer import (
                    render_cluster_summary,
                    render_state_changes_in_window,
                )

                ctx_kwargs["cluster_summary"] = render_cluster_summary(grid)
                ctx_kwargs["delta_window_summary"] = render_state_changes_in_window(
                    self._memory, max_steps=5
                )
            except Exception:
                pass
        if "steps_at_current_level" in existing:
            # Sprint-19 Hebel O: count how many consecutive recent memory
            # entries share the current ``levels_completed``. The vision
            # prompt template reads this to decide whether to inject a
            # stalled-progress warning. Walk memory.window() from most
            # recent backwards while the level matches.
            try:
                current_level = latest_frame.levels_completed
                steps_at = 0
                # Use a generous window — bp35 has 9 levels with episodes
                # capped at 80 steps, so a stall of >50 steps is the
                # whole episode. Walking 80 entries is microsecond-cheap
                # so we don't bother capping.
                for step_entry in self._memory.window(80):
                    if step_entry.levels_completed == current_level:
                        steps_at += 1
                    else:
                        break
                ctx_kwargs["steps_at_current_level"] = steps_at
            except Exception:
                pass
        ctx = FrameContext(**ctx_kwargs)

        try:
            plan, _top_reasoning = self._choice_fn(ctx)
        except Exception:
            # Silent fallback hides genuine bugs in the LLM choice-fn —
            # set ``COGNITHOR_PSE_DEBUG_LLM=1`` to surface the actual
            # exception (full traceback to stderr) before returning the
            # ``None`` that triggers the DSL fallback. Off by default to
            # preserve the contract: production runs must keep playing.
            import os as _os

            if _os.environ.get("COGNITHOR_PSE_DEBUG_LLM"):
                import sys as _sys
                import traceback as _tb

                print(
                    "[pse-debug] choice_fn raised; falling back to DSL:",
                    file=_sys.stderr,
                )
                _tb.print_exc(file=_sys.stderr)
            return None
        if not plan:
            return None
        # Trim to horizon — the LLM may over-deliver.
        return list(plan[: self._plan_horizon])
