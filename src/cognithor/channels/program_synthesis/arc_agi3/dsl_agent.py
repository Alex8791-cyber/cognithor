# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-11 Wave-4 — Sprint10DSLAgent: stateful DSL-aware ARC-AGI-3 agent.

The Wave-4 agent wires Wave-1's :class:`CognithorPSEAgent` ABC,
Wave-2's :class:`FrameBridge` + :class:`ActionDecoder`, and Wave-3's
:class:`EpisodeMemory` + :class:`StuckDetector` into a complete
runnable agent. It's the first concrete agent that uses the
Sprint-10 DSL surface (via :class:`FrameBridge`) AND maintains
episode state for multi-frame reasoning.

The agent's ``choose_action`` flow:

1. Bridge the current ``FrameData`` to a Cognithor int8 grid.
2. Append the previous step's ``(grid, action_name, levels_completed)``
   to the memory — but only after the second frame, since step 1's
   "previous action" is meaningless before the agent has acted once.
3. Delegate to the :class:`DSLActionDecoder`, which scores actions
   against the memory state.
4. Track the just-chosen action's name so step 2 can record it on
   the next frame.

The agent stops on WIN. Subclasses can override :meth:`is_done` for
GAME_OVER policies (e.g. retry loops).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from cognithor.channels.program_synthesis.arc_agi3.agent import CognithorPSEAgent
from cognithor.channels.program_synthesis.arc_agi3.dsl_action_decoder import (
    DSLActionDecoder,
)
from cognithor.channels.program_synthesis.arc_agi3.episode_memory import (
    EpisodeMemory,
    StuckDetector,
)
from cognithor.channels.program_synthesis.arc_agi3.fast_path import (
    ClickPlanCache,
    detect_toggle_pair_from_memory,
)
from cognithor.channels.program_synthesis.arc_agi3.frame_bridge import FrameBridge
from cognithor.channels.program_synthesis.arc_agi3.state_action_counts import (
    StateActionCounter,
    hash_state,
)
from cognithor.channels.program_synthesis.arc_agi3.state_graph import (
    StateGraphNavigator,
)

if TYPE_CHECKING:
    from cognithor.channels.program_synthesis.arc_agi3.action_decoder import (
        ActionDecoder,
    )
    from cognithor.channels.program_synthesis.arc_agi3.audit import ArcAuditTrail
    from cognithor.channels.program_synthesis.arc_agi3.click_target_sampler import (
        ClickTargetSampler,
    )
    from cognithor.channels.program_synthesis.arc_agi3.frame_analyzer import (
        FrameAnalyzer,
    )
    from cognithor.channels.program_synthesis.arc_agi3.game_profile import GameProfile
    from cognithor.channels.program_synthesis.arc_agi3.llm_telemetry import LLMTelemetry
    from cognithor.channels.program_synthesis.arc_agi3.mtp_stats import MTPStats
    from cognithor.channels.program_synthesis.arc_agi3.protocol import (
        FrameDataProtocol,
        GameActionProtocol,
    )


class Sprint10DSLAgent(CognithorPSEAgent):
    """Stateful agent that uses the Sprint-10 DSL (via FrameBridge)
    and maintains an :class:`EpisodeMemory` to drive a least-tried-
    non-RESET action policy with stuck-detection.

    Sprint-11 Wave-4 baseline. The Wave-5 LLM agent subclasses this
    and replaces the decoder with an LLM-prior-weighted scorer; the
    memory + stuck plumbing stays the same.
    """

    def __init__(
        self,
        *,
        bridge: FrameBridge | None = None,
        memory: EpisodeMemory | None = None,
        stuck_detector: StuckDetector | None = None,
        state_counter: StateActionCounter | None = None,
        state_graph: StateGraphNavigator | None = None,
        audit_trail: ArcAuditTrail | None = None,
        game_profile: GameProfile | None = None,
        strategy_name: str = "sprint10_dsl",
        frame_analyzer: FrameAnalyzer | None = None,
        fast_path_enabled: bool = False,
        click_target_sampler: ClickTargetSampler | None = None,
        telemetry: LLMTelemetry | None = None,
        mtp_stats: MTPStats | None = None,
    ) -> None:
        self._bridge = bridge if bridge is not None else FrameBridge()
        self._memory = memory if memory is not None else EpisodeMemory()
        self._stuck = stuck_detector if stuck_detector is not None else StuckDetector()
        # Sprint-12: state-keyed action counts + state graph (lifted from
        # cognithor.arc). Both opt-in — pass None to keep Wave-4 behaviour.
        self._state_counter = state_counter if state_counter is not None else StateActionCounter()
        self._state_graph = state_graph if state_graph is not None else StateGraphNavigator()
        self._decoder: ActionDecoder = DSLActionDecoder(
            memory=self._memory,
            stuck_detector=self._stuck,
            state_counter=self._state_counter,
        )
        # Tracks the most-recently-chosen action's name so we can
        # record it in the memory when the next frame arrives.
        # ``None`` until the first ``choose_action`` call.
        self._pending_action_name: str | None = None
        self._pending_levels: int | None = None
        # Cache the previous-frame grid so we can detect no-op transitions
        # and feed the StateGraphNavigator with full (from, action, to) tuples.
        self._prev_grid: np.ndarray[Any, Any] | None = None
        self._prev_state_hash: str | None = None
        # Sprint-12 PR-6+7: optional cross-episode persistence hooks.
        # Both default to None (no-op); pass instances to enable.
        self._audit_trail = audit_trail
        self._game_profile = game_profile
        self._strategy_name = strategy_name
        self._audit_started = False
        self._step_count = 0
        # Sprint-12 PR-11: track levels_completed so we can detect a
        # level-transition and reset per-level state (memory, state-keyed
        # counts, state-graph) without losing the cross-level audit trail
        # or game-profile aggregates.
        self._last_levels_seen: int | None = None
        # Sprint-12 PR-5: per-action movement-signature tracker. When a
        # FrameAnalyzer is wired in, choose_action feeds it each
        # observation so it can build a model of "ACTION3 → moves down"
        # over time. Action effects survive level boundaries; positions
        # don't.
        self._frame_analyzer = frame_analyzer
        # Sprint-12 PR-8: pure-NumPy click-toggle fast-path. Disabled by
        # default; enable via constructor flag to short-circuit the DSL
        # search when the most recent transition reveals a toggle pair.
        self._fast_path_enabled = fast_path_enabled
        self._click_cache: ClickPlanCache | None = ClickPlanCache() if fast_path_enabled else None
        # Sprint-12 PR-13: salience-ranked click-target sampler. When
        # ACTION6 is available and the toggle fast-path can't fire (no
        # toggle pair observed), this picks a click coordinate from the
        # smallest non-background object instead of a uniform-random one.
        # Reset on level transition so visited-set doesn't leak across
        # levels.
        self._click_target_sampler = click_target_sampler
        # Sprint-15: optional LLM-call + MTP-stats aggregators. When wired,
        # the audit trail picks the most-recent record/snapshot for each
        # step and rides those telemetry numbers on the same hash chain
        # (no separate join required at bench-comparison time).
        self._telemetry = telemetry
        self._mtp_stats = mtp_stats

    @property
    def memory(self) -> EpisodeMemory:
        """Read-only access for tests + downstream Wave-5 wiring."""
        return self._memory

    @property
    def state_counter(self) -> StateActionCounter:
        """Read-only access to the Blind-Squirrel-style state-action counter."""
        return self._state_counter

    @property
    def state_graph(self) -> StateGraphNavigator:
        """Read-only access to the BFS-replay state graph."""
        return self._state_graph

    def is_done(
        self,
        frames: list[FrameDataProtocol],
        latest_frame: FrameDataProtocol,
    ) -> bool:
        return latest_frame.state.name in {"WIN", "GAME_OVER"}

    def choose_action(
        self,
        frames: list[FrameDataProtocol],
        latest_frame: FrameDataProtocol,
    ) -> GameActionProtocol:
        # Step 0 — start the audit trail on the very first call.
        if self._audit_trail is not None and not self._audit_started:
            self._audit_trail.log_game_start()
            self._audit_started = True

        # Step 1 — bridge the current frame to a Cognithor grid.
        # Errors (out-of-range, wrong shape) propagate; the upstream
        # harness logs and retries one frame later if this happens.
        current_grid = self._bridge.extract_grid(latest_frame)
        current_hash = hash_state(current_grid)

        # Step 1a — feed the FrameAnalyzer (if wired) so the per-action
        # movement-signature model stays current. The action tag is the
        # PREVIOUS action (the one whose result we are now observing),
        # which matches FrameAnalyzer's semantics.
        if self._frame_analyzer is not None:
            self._frame_analyzer.analyze(current_grid, action=self._pending_action_name)

        # Step 1b — detect a level transition. When ``levels_completed``
        # increased since the last call, reset the per-level state
        # (EpisodeMemory + StateActionCounter + StateGraph). The
        # FrameAnalyzer's reset_for_new_level() preserves learned action
        # effects but clears position tracking. Audit trail + GameProfile
        # are deliberately NOT reset — they aggregate across levels.
        if (
            self._last_levels_seen is not None
            and latest_frame.levels_completed > self._last_levels_seen
        ):
            # Sprint-20 Hebel V: persist the trajectory that led to the
            # level-up BEFORE clearing memory. Future episodes on this
            # game can then prompt-inject the action sequence as a
            # few-shot demonstration. ``_win_demo_store`` is set by
            # PlanningLLMReasoningAgent; missing on the heuristic agent
            # → no-op via getattr default.
            store = getattr(self, "_win_demo_store", None)
            if store is not None:
                import contextlib

                with contextlib.suppress(Exception):
                    store.record_level_up(
                        game_id=getattr(latest_frame, "game_id", "") or "unknown",
                        from_level=self._last_levels_seen,
                        to_level=latest_frame.levels_completed,
                        memory=self._memory,
                    )
            self._memory.clear()
            self._state_counter.clear()
            self._state_graph = type(self._state_graph)()  # fresh graph
            if self._frame_analyzer is not None:
                self._frame_analyzer.reset_for_new_level()
            if self._click_target_sampler is not None:
                self._click_target_sampler.reset_for_new_level()
            self._pending_action_name = None
            self._prev_grid = None
            self._prev_state_hash = None

        # Step 2 — record the previous step in the memory if we
        # already chose an action. The grid we pair with the
        # previous action is the *current* grid (i.e. the frame
        # that resulted from that action).
        # Sprint-12: also feeds the state-graph + flags no-op transitions.
        if self._pending_action_name is not None:
            self._memory.append(
                grid=current_grid,
                action_name=self._pending_action_name,
                levels_completed=latest_frame.levels_completed,
            )
            # State-graph: record the (from, action, to) edge. Skip when
            # the grid shape changed (e.g. across a level boundary) — the
            # graph only models intra-level transitions.
            if (
                self._prev_grid is not None
                and self._prev_state_hash is not None
                and self._prev_grid.shape == current_grid.shape
            ):
                pixels_changed = int(np.sum(self._prev_grid != current_grid))
                self._state_graph.add_transition(
                    from_grid=self._prev_grid,
                    action_str=self._pending_action_name,
                    action_data=None,
                    to_grid=current_grid,
                    pixels_changed=pixels_changed,
                    game_state=latest_frame.state.name,
                    level=latest_frame.levels_completed,
                )
                # No-op detection: if the grid didn't change, mark this
                # (state, action) edge dead so we don't repeat it.
                if self._prev_state_hash == current_hash:
                    self._state_counter.mark_dead(self._prev_state_hash, self._pending_action_name)

        # Step 3a — try the toggle-fast-path planner if enabled. The
        # planner only fires when ACTION6 is available AND we've seen a
        # clean toggle pair in the last two frames AND the planner finds
        # a winning click sequence. On a miss it returns None.
        chosen: GameActionProtocol | None = None
        if self._click_cache is not None:
            chosen = self._try_fast_path(latest_frame, current_grid, current_hash)

        # Step 3b — try the salience-based click sampler. Fires only when
        # the toggle fast-path didn't, ACTION6 is available, and the
        # sampler has a non-empty queue. Useful for "click target"
        # games where there's no toggle pair to detect.
        if chosen is None and self._click_target_sampler is not None:
            chosen = self._try_click_target(latest_frame, current_grid)

        # Step 3c — delegate to the DSL decoder when no click sampler
        # fired.
        if chosen is None:
            chosen = self._decoder.decode(frames, latest_frame)

        # Step 4 — bookkeeping: increment the (current_state, chosen) count
        # so the next decode skips it if alternatives exist.
        self._state_counter.increment(current_hash, chosen.name)

        # Step 5 — remember choice + grid for the next call.
        self._pending_action_name = chosen.name
        self._pending_levels = latest_frame.levels_completed
        self._prev_grid = current_grid
        self._prev_state_hash = current_hash
        self._last_levels_seen = latest_frame.levels_completed

        # Step 6 — log the step into the audit trail when one is wired.
        if self._audit_trail is not None:
            pixels_changed = 0
            if self._prev_grid is not None and len(self._memory) > 1:
                # Compare against the prior memory entry's grid for the
                # delta (we just appended current_grid as the "result of
                # the previous action", so its predecessor is the old
                # state).
                prior_steps = self._memory.window(2)
                if len(prior_steps) >= 2 and prior_steps[1].grid.shape == current_grid.shape:
                    pixels_changed = int(np.sum(prior_steps[1].grid != current_grid))
            telemetry_kwargs = self._latest_telemetry_kwargs()
            self._audit_trail.log_step(
                level=latest_frame.levels_completed,
                step=self._step_count,
                action=chosen.name,
                game_state=latest_frame.state.name,
                pixels_changed=pixels_changed,
                **telemetry_kwargs,
            )
            self._step_count += 1

        return chosen

    def _latest_telemetry_kwargs(self) -> dict[str, Any]:
        """Snapshot the most-recent LLM-call + MTP record as audit kwargs.

        Returns the new Sprint-15 ``llm_*`` and ``mtp_*`` fields when
        the corresponding aggregator has at least one entry; otherwise
        empty (so legacy callers stay byte-identical).

        Each call is read-only; the aggregator stays untouched so a
        subsequent step that doesn't trigger a new LLM call won't
        re-log stale telemetry — the per-call write happens inside the
        choice-fn factories, so a step without a fresh record sees
        ``len(records)`` unchanged and the helper returns the previous
        record (intentionally; the LLM call drove the prior decision
        chain too).
        """
        kwargs: dict[str, Any] = {}
        if self._telemetry is not None and self._telemetry.records:
            rec = self._telemetry.records[-1]
            kwargs["llm_input_tokens"] = rec.input_tokens
            kwargs["llm_output_tokens"] = rec.output_tokens
            kwargs["llm_think_tokens"] = rec.think_tokens
            kwargs["llm_finish_reason"] = rec.finish_reason
            kwargs["llm_wall_clock_s"] = rec.wall_clock_s
            kwargs["llm_ttft_s"] = rec.ttft_s
            # Sprint-19 Hebel P: forward the model's top-level
            # reasoning when the telemetry-wrapped choice-fn captured
            # it. ``None`` is the legacy default and stays JSONL-clean.
            if getattr(rec, "reasoning", None) is not None:
                kwargs["llm_reasoning"] = rec.reasoning
        if self._mtp_stats is not None and self._mtp_stats.snapshots:
            snap = self._mtp_stats.snapshots[-1]
            kwargs["mtp_drafts_proposed"] = snap.drafts_proposed
            kwargs["mtp_drafts_accepted"] = snap.drafts_accepted
            kwargs["mtp_acceptance_rate"] = snap.acceptance_rate
        return kwargs

    @property
    def frame_analyzer(self) -> FrameAnalyzer | None:
        """Read-only access to the wired FrameAnalyzer (or None)."""
        return self._frame_analyzer

    def finalize_episode(
        self,
        *,
        score: int,
        won: bool,
        levels_solved: int,
        budget_ratio: float = 0.0,
    ) -> None:
        """Close out the episode: log game_end + roll up the GameProfile.

        Idempotent across multiple calls; subsequent calls after the
        first are no-ops.
        """
        if self._audit_trail is not None and self._audit_started:
            self._audit_trail.log_game_end(final_score=float(score))
            # Mark closed so a second finalize_episode() is silent.
            self._audit_started = False
        if self._game_profile is not None:
            self._game_profile.update_run(score=score)
            self._game_profile.update_metrics(
                self._strategy_name,
                won=won,
                levels_solved=levels_solved,
                steps=self._step_count,
                budget_ratio=budget_ratio,
            )

    def _try_fast_path(
        self,
        latest_frame: FrameDataProtocol,
        current_grid: np.ndarray[Any, Any],
        current_hash: str,
    ) -> GameActionProtocol | None:
        """Click-toggle fast-path. Returns ACTION6 with click coordinates
        when a winning plan exists, ``None`` otherwise.

        Side-effect-free on miss; the cache is keyed by ``(state_hash,
        source, target)`` so repeated calls on the same state don't
        re-run the NumPy search.
        """
        if self._click_cache is None:
            return None
        # ACTION6 must be available for clicks.
        click_action: GameActionProtocol | None = None
        for action in latest_frame.available_actions:
            if action.name == "ACTION6":
                click_action = action
                break
        if click_action is None:
            return None
        # Need a toggle pair signal from the last two grids.
        toggle = detect_toggle_pair_from_memory(self._memory)
        if toggle is None:
            return None
        source_color, target_color = toggle
        click_xy = self._click_cache.next_click(
            state_hash=current_hash,
            grid=current_grid,
            source_color=source_color,
            target_color=target_color,
        )
        if click_xy is None:
            return None
        x, y = click_xy
        click_action.set_data({"x": int(x), "y": int(y)})
        click_action.reasoning = (
            f"Sprint10DSLAgent fast-path: plan_click_solution "
            f"({source_color}->{target_color}) → click ({x},{y})"
        )
        return click_action

    def _try_click_target(
        self,
        latest_frame: FrameDataProtocol,
        current_grid: np.ndarray[Any, Any],
    ) -> GameActionProtocol | None:
        """Salience-based click target sampling.

        For "click target" games (no toggle pair, just "click the right
        object"). Returns ACTION6 with the next salient ``(x, y)`` from
        :class:`ClickTargetSampler`, or ``None`` when ACTION6 isn't
        available or the sampler queue is empty.
        """
        if self._click_target_sampler is None:
            return None
        click_action: GameActionProtocol | None = None
        for action in latest_frame.available_actions:
            if action.name == "ACTION6":
                click_action = action
                break
        if click_action is None:
            return None
        click_xy = self._click_target_sampler.next_click(current_grid)
        if click_xy is None:
            return None
        x, y = click_xy
        click_action.set_data({"x": int(x), "y": int(y)})
        click_action.reasoning = f"Sprint10DSLAgent click-target sampler → click ({x},{y})"
        return click_action


__all__ = ["Sprint10DSLAgent"]
