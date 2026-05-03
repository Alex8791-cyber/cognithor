# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-11 Wave-5 — LLM-driven action decoder.

The Wave-5 :class:`LLMActionDecoder` replaces Wave-4's heuristic
"least-tried action" policy with an LLM call: per frame, format a
prompt that describes the current grid + recent history + available
actions, ask the LLM which action to take, and return the
LLM-suggested action.

The actual LLM call is injected as a ``prompt_to_choice`` callable.
This keeps the decoder synchronous (mandatory — the upstream
``Agent.choose_action`` is sync) AND testable (tests pass a stub
callable that returns deterministic choices). A production factory
:func:`build_vllm_choice_fn` wraps a Sprint-10 :class:`VLLMBackend`
into the right shape, doing the async-in-sync via :func:`asyncio.run`.

Without a running vLLM server, the production factory fails fast on
the first call; the decoder then falls back to its Wave-4
:class:`DSLActionDecoder` policy if a fallback is configured. This
mirrors Sprint-10 Track B's hardware-gated wiring.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cognithor.channels.program_synthesis.arc_agi3.action_decoder import ActionDecoder
from cognithor.channels.program_synthesis.arc_agi3.dsl_action_decoder import (
    DSLActionDecoder,
)
from cognithor.channels.program_synthesis.arc_agi3.state_action_counts import hash_state

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from cognithor.channels.program_synthesis.arc_agi3.episode_memory import (
        ActionStreakDetector,
        EpisodeMemory,
    )
    from cognithor.channels.program_synthesis.arc_agi3.frame_analyzer import (
        FrameAnalyzer,
    )
    from cognithor.channels.program_synthesis.arc_agi3.frame_bridge import FrameBridge
    from cognithor.channels.program_synthesis.arc_agi3.goal_inferer import GoalInferer
    from cognithor.channels.program_synthesis.arc_agi3.protocol import (
        FrameDataProtocol,
        GameActionProtocol,
    )
    from cognithor.channels.program_synthesis.arc_agi3.state_action_counts import (
        StateActionCounter,
    )

    _Grid = NDArray[np.int8]


# Sprint-16 anti-loop: how many times the SAME (state, action) combo
# may be picked before the decoder treats it as dead even without an
# explicit no-op observation. Conservative — `mark_dead` from the
# parent agent's no-op detector usually fires faster, this is a
# fallback for states where the action *does* change pixels but not
# in a way that helps progress (e.g. a click that toggles a useless
# light on and off).
_REPEAT_THRESHOLD = 3


@dataclass(frozen=True)
class FrameContext:
    """Bundle of inputs the LLM-prompt-builder reads.

    The decoder constructs this once per :meth:`pick_action` call,
    then hands it to the injected ``prompt_to_choice`` callable. The
    callable returns ``(action_name, reasoning)``; the decoder
    matches the name against ``available_actions`` and returns the
    matching action.

    ``action_effects_summary`` is an optional one-line description of
    what each action has historically done (Sprint-12 PR-12 — fed by
    a wired :class:`FrameAnalyzer`). Empty string when no analyser is
    wired or no observations have been recorded yet.
    """

    grid: _Grid
    available_action_names: list[str]
    history_summary: str
    levels_completed: int
    win_levels: int
    action_effects_summary: str = ""
    # Sprint-13 PR-1: goal hypothesis from the new GoalInferer (reads
    # the EpisodeMemory + FrameAnalyzer). Empty when no inferer wired.
    goal_summary: str = ""
    # Sprint-13 PR-1: game_id lets per-game prompt fragments
    # (game_prompts.GAME_PROMPTS) be looked up by the prompt builder.
    game_id: str = ""
    # Sprint-16 anti-loop: per-(current state, action) counts with the
    # actions the LLM is forbidden to pick at this state because they
    # are dead (no-op observed) or repeat-saturated. Empty string when
    # no state-counter is wired so legacy prompt builders stay
    # byte-identical.
    state_action_summary: str = ""
    forbidden_action_names: tuple[str, ...] = ()
    # Sprint-19 Hebel B (vision delta): the previous frame's grid +
    # the action that led from prev → current. Lets the vision-prompt
    # builder send a side-by-side "before/after" image pair so the LLM
    # can visually diff. ``None`` when no prior frame exists yet.
    prev_grid: _Grid | None = None
    prev_action_name: str = ""
    # Sprint-19 Hebel D (structured text): pre-rendered cluster
    # decomposition + recent-window delta-summary. Optional — empty
    # strings disable the corresponding prompt sections so legacy
    # prompts stay byte-identical.
    cluster_summary: str = ""
    delta_window_summary: str = ""
    # Sprint-19 Hebel O (stalled-progress signal): how many consecutive
    # recent memory entries have the same ``levels_completed`` as the
    # current frame. Computed by the decoder from memory; lets the
    # prompt builder inject a "you've been stuck for N steps" warning
    # when the agent is failing for too long. ``0`` means "no stall
    # observed yet" so legacy prompt builders stay byte-identical.
    steps_at_current_level: int = 0
    # Sprint-19 Hebel S (per-action pixΔ histogram): one line per
    # action seen in the episode showing avg / max pixΔ + count, with
    # DANGER / CAUTION suffixes matching Hebel M's risk vocabulary.
    # Lets the LLM reason about per-action risk (max) and impact
    # (avg) at a glance. Empty string disables the corresponding
    # prompt section.
    action_pixel_history: str = ""


# A callable that takes a :class:`FrameContext` and returns
# ``(chosen_action_name, reasoning_text)``. The action name MUST be
# one of ``ctx.available_action_names``; the decoder validates this.
ChoiceFn = Callable[[FrameContext], tuple[str, str]]


def render_grid(grid: _Grid) -> str:
    """Render an ``int8`` grid as a multi-line ASCII string for the LLM.

    Single-digit values mean each row is a contiguous ``012345``-style
    string. The renderer pads with spaces between columns for readability,
    which makes the column structure visible in the LLM's chat-window
    rendering.
    """
    if grid.ndim != 2:
        raise ValueError(f"render_grid: expected 2-D grid, got {grid.ndim}-D")
    return "\n".join(" ".join(str(int(c)) for c in row) for row in grid)


def summarise_action_effects(analyzer: FrameAnalyzer, *, max_actions: int = 6) -> str:
    """Compact one-line summary of FrameAnalyzer's per-action effect model.

    Format: ``"ACTION1: row+1, col+0 (n=4); ACTION3: row+0, col-1 (n=2); ..."``.
    Empty observations yield ``"(no action effects observed yet)"``.

    The Sprint-12 :class:`Sprint10DSLAgent` feeds a wired :class:`FrameAnalyzer`
    each frame; this helper formats its accumulated knowledge into a string the
    LLM can read in the next prompt. Knowing "ACTION3 has historically moved
    your sprite down" is critical for keyboard-controlled games where the
    upstream API doesn't announce what each ``ACTIONx`` does.
    """
    summary = analyzer.get_action_summary()
    if not summary:
        return "(no action effects observed yet)"
    items = sorted(summary.items(), key=lambda kv: -kv[1]["count"])[:max_actions]
    parts: list[str] = []
    for name, stats in items:
        row = stats["avg_direction_row"]
        col = stats["avg_direction_col"]
        n = int(stats["count"])
        parts.append(f"{name}: row{row:+.0f}, col{col:+.0f} (n={n})")
    return "; ".join(parts)


def summarise_history(memory: EpisodeMemory, max_steps: int = 8) -> str:
    """Compact text description of the last ``max_steps`` actions.

    Format: ``"step -3: ACTION1 (Δ23), step -2: RESET (level up, Δ0), ..."``
    The ``Δ<N>`` field is the pixel-change count *after* that action,
    computed on-the-fly by diffing consecutive grids in the window.
    Sprint-17 addition: lets the LLM see which actions actually moved
    the game state (large Δ) vs which were no-ops (Δ=0 or Δ=1, only
    the cursor moved). Critical for click-target games where ACTION6
    with arbitrary coords looks the same as a meaningful action without
    this signal.

    Used in the Stage-1 prompt so the LLM knows what's been tried.
    Empty history yields ``"(no actions yet)"``.
    """
    if max_steps < 1:
        raise ValueError(f"summarise_history: max_steps must be >= 1, got {max_steps}")
    window = memory.window(max_steps)
    if not window:
        return "(no actions yet)"
    parts: list[str] = []
    for i, step in enumerate(window):
        # ``window`` is most-recent first; show step -1 for the
        # most recent, step -2 for the prior, etc.
        idx = -(i + 1)
        # pixels_changed = diff to the *next-older* grid in the window.
        # Last entry has no older neighbour to diff against → omit.
        delta_marker = ""
        if i + 1 < len(window):
            prev = window[i + 1]
            if prev.grid.shape == step.grid.shape:
                import numpy as _np

                delta = int(_np.sum(prev.grid != step.grid))
                delta_marker = f", Δ{delta}"
        levels_marker = f", level={step.levels_completed}" if step.levels_completed > 0 else ""
        parts.append(f"step {idx}: {step.action_name}{levels_marker}{delta_marker}")
    return "; ".join(parts)


class LLMActionDecoder(ActionDecoder):
    """Stateful decoder that delegates the choice to an LLM.

    Construct with:

    * ``bridge`` — :class:`FrameBridge` to convert the live frame to
      the Cognithor int8 grid that gets rendered for the prompt
    * ``memory`` — :class:`EpisodeMemory` for the history-summary
    * ``choice_fn`` — synchronous callable from
      :class:`FrameContext` to ``(action_name, reasoning)``. Production
      uses :func:`build_vllm_choice_fn`; tests pass deterministic stubs.
    * ``fallback`` (optional) — another :class:`ActionDecoder` used
      when ``choice_fn`` raises or returns an unknown action.
      Default: a fresh :class:`DSLActionDecoder` over the same memory.
    * ``history_steps`` — how many recent steps to summarise in the
      prompt. Default 8.
    """

    def __init__(
        self,
        *,
        bridge: FrameBridge,
        memory: EpisodeMemory,
        choice_fn: ChoiceFn,
        fallback: ActionDecoder | None = None,
        history_steps: int = 8,
        frame_analyzer: FrameAnalyzer | None = None,
        goal_inferer: GoalInferer | None = None,
        state_counter: StateActionCounter | None = None,
        action_streak_detector: ActionStreakDetector | None = None,
    ) -> None:
        self._bridge = bridge
        self._memory = memory
        self._choice_fn = choice_fn
        self._fallback = fallback if fallback is not None else DSLActionDecoder(memory=memory)
        self._history_steps = history_steps
        # Sprint-12 PR-12: optional FrameAnalyzer for per-action movement
        # signatures fed into the LLM prompt. Default None preserves baseline.
        self._frame_analyzer = frame_analyzer
        # Sprint-13 PR-1: optional GoalInferer for evidence-driven
        # goal-hypothesis text in the LLM prompt.
        self._goal_inferer = goal_inferer
        # Sprint-16 Hebel 1: per-(state, action) counter so the decoder
        # can filter dead/repeat-saturated combos out of the LLM's
        # choice set AND override the LLM if it ignores the constraint.
        # Doesn't catch click-game loops where each pick changes the
        # state hash — that's what Hebel 2 covers below.
        self._state_counter = state_counter
        # Sprint-16 Hebel 2: state-agnostic action-streak detector.
        # Looks at recent memory and forbids any action that dominated
        # the last ``window`` picks while ``levels_completed`` stayed
        # flat. This catches click-game ACTION6×40 loops where Hebel 1
        # silently passes (each click changes the cursor pixel → new
        # state hash → counter resets).
        self._action_streak_detector = action_streak_detector

    def pick_action(
        self,
        frames: list[FrameDataProtocol],
        latest_frame: FrameDataProtocol,
        available_actions: list[GameActionProtocol],
    ) -> tuple[GameActionProtocol, str]:
        # Build the prompt context.
        try:
            grid = self._bridge.extract_grid(latest_frame)
        except Exception as exc:  # pragma: no cover — bridge errors propagate
            # BUG-PSE-007 fix: prior code had a malformed
            # ``+ (...)[0:0] or fallback(...)`` chain that called the
            # fallback twice. Single clean fallback call now.
            del exc
            return self._fallback.pick_action(frames, latest_frame, available_actions)

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

        # Sprint-16 anti-loop: ask the state-counter which actions are
        # already dead (mark_dead from no-op observations) or
        # repeat-saturated at the current state. ``forbidden`` is fed
        # back to the LLM via the prompt + used to override the LLM's
        # choice if it picks one anyway.
        forbidden, state_action_summary = self._compute_forbidden(grid, available_actions)
        # If everything is forbidden the agent is genuinely stuck —
        # return all actions to the LLM rather than crashing on an
        # empty choice set; the prompt then shows every action as
        # "all-tried" so the LLM can pick the least-bad one.
        if len(forbidden) == len(available_actions):
            forbidden = ()

        # BUG-PSE-001 fix: populate Sprint-19 fields in the single-step
        # decoder too, matching what PlanningLLMActionDecoder does.
        # Without these, vision-mode + cluster/delta annotation are
        # silently disabled when wired through this decoder.
        prev_grid = None
        prev_action_name = ""
        cluster_summary_text = ""
        delta_window_text = ""
        if len(self._memory) > 0:
            try:
                prev_step = self._memory.window(1)[0]
                prev_grid = prev_step.grid
                prev_action_name = prev_step.action_name
            except Exception:
                pass
        try:
            from cognithor.channels.program_synthesis.arc_agi3.state_renderer import (
                render_cluster_summary,
                render_state_changes_in_window,
            )

            cluster_summary_text = render_cluster_summary(grid)
            delta_window_text = render_state_changes_in_window(self._memory, max_steps=5)
        except Exception:
            pass

        ctx = FrameContext(
            grid=grid,
            available_action_names=[a.name for a in available_actions],
            history_summary=summarise_history(self._memory, self._history_steps),
            levels_completed=latest_frame.levels_completed,
            win_levels=latest_frame.win_levels,
            action_effects_summary=action_effects_summary,
            goal_summary=goal_summary,
            game_id=getattr(latest_frame, "game_id", ""),
            state_action_summary=state_action_summary,
            forbidden_action_names=forbidden,
            prev_grid=prev_grid,
            prev_action_name=prev_action_name,
            cluster_summary=cluster_summary_text,
            delta_window_summary=delta_window_text,
        )

        # Call the LLM (or stub). Failures fall back to the DSL decoder.
        try:
            chosen_name, reasoning = self._choice_fn(ctx)
        except Exception as exc:
            fallback_action, fb_reasoning = self._fallback.pick_action(
                frames, latest_frame, available_actions
            )
            return fallback_action, (
                f"LLMActionDecoder fallback ({type(exc).__name__}): {fb_reasoning}"
            )

        # Sprint-16 post-LLM override: if the LLM picked a forbidden
        # action despite being told, swap to the least-tried allowed
        # alternative. This is what actually breaks the
        # deterministic-loop trap — the prompt-side warning alone is
        # advisory; the override is what makes the constraint binding.
        if forbidden and chosen_name in forbidden:
            allowed = [a for a in available_actions if a.name not in forbidden]
            if allowed:
                state_hash = hash_state(grid) if self._state_counter is not None else ""
                fallback_pick = min(
                    allowed,
                    key=lambda a: (
                        self._state_counter.count(state_hash, a.name)
                        if self._state_counter is not None
                        else 0
                    ),
                )
                return fallback_pick, (
                    f"LLMActionDecoder anti-loop override: LLM picked {chosen_name!r} "
                    f"but it's forbidden at this state (dead/repeat); "
                    f"fell back to least-tried allowed action {fallback_pick.name!r}"
                )

        # Validate the LLM's name against the whitelist.
        for action in available_actions:
            if action.name == chosen_name:
                return action, reasoning or f"LLMActionDecoder chose {chosen_name}"

        # LLM picked an action not in the whitelist — fall back.
        fallback_action, fb_reasoning = self._fallback.pick_action(
            frames, latest_frame, available_actions
        )
        return fallback_action, (
            f"LLMActionDecoder fallback (LLM returned {chosen_name!r} which is "
            f"not available): {fb_reasoning}"
        )

    def _compute_forbidden(
        self,
        grid: _Grid,
        available_actions: list[GameActionProtocol],
    ) -> tuple[tuple[str, ...], str]:
        """Decide which available action names are forbidden at the
        current state, and produce the LLM-facing summary line.

        Returns ``(forbidden_tuple, summary_string)``.
        ``forbidden_tuple`` is empty when no detector is wired or
        nothing is currently dead/repeat-saturated/streak-stuck.

        Combines two complementary signals:

        * Hebel 1 — :class:`StateActionCounter`: per-(state, action)
          dead/repeat tracking. Catches "ACTION1 always no-ops at
          this exact frame".
        * Hebel 2 — :class:`ActionStreakDetector`: state-agnostic
          recent-window dominance. Catches "ACTION6 picked 4/5 of
          the last steps with no level progress" — the click-game
          loop signature where each pick mutates the state hash so
          Hebel 1 silently passes.
        """
        forbidden_set: set[str] = set()
        rows: list[str] = []

        # Hebel 1 — per-(state, action) signals.
        if self._state_counter is not None:
            state_hash = hash_state(grid)
            dead = self._state_counter.all_dead_actions(state_hash)
            forbidden_set.update(dead)
            for action in available_actions:
                count = self._state_counter.count(state_hash, action.name)
                is_dead = action.name in dead
                if not is_dead and count >= _REPEAT_THRESHOLD:
                    forbidden_set.add(action.name)
                tag = (
                    "DEAD"
                    if is_dead
                    else ("REPEAT-SATURATED" if count >= _REPEAT_THRESHOLD else f"{count}×")
                )
                rows.append(f"{action.name}: {tag}")

        # Hebel 2 — recent-window streak with no level progress.
        # Read-only; the detector inspects ``self._memory``.
        if self._action_streak_detector is not None:
            stuck_action = self._action_streak_detector.dominant_stuck_action(self._memory)
            if stuck_action is not None and any(a.name == stuck_action for a in available_actions):
                forbidden_set.add(stuck_action)
                rows.append(
                    f"{stuck_action}: STREAK-STUCK "
                    f"({self._action_streak_detector.threshold}+/"
                    f"{self._action_streak_detector.window} recent picks, "
                    "no level progress)"
                )

        if not forbidden_set and not rows:
            return (), ""
        # Stable order — use the available_actions list order so the
        # summary matches what the prompt enumerates.
        forbidden = tuple(a.name for a in available_actions if a.name in forbidden_set)
        summary = "; ".join(rows)
        return forbidden, summary


__all__ = [
    "ChoiceFn",
    "FrameContext",
    "LLMActionDecoder",
    "render_grid",
    "summarise_action_effects",
    "summarise_history",
]
