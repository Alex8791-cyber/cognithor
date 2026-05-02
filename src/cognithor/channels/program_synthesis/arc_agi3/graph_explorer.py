# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-15 — training-free State-Graph Exploration agent.

Mirrors the 3rd-place ARC-AGI-3 entry (`arc-agi-3-just-explore`) and
the Blind Squirrel state-graph approach: maintain an explicit graph
of visited states, and at every step pick the (current_state,
action) pair that has been tried the *fewest* times — biased toward
actions known to actually change the frame.

Why this beats pure-LLM agents in our setup:

* **0 LLM cost per step.** Action selection is a few hash lookups +
  a Counter min().
* **Provably exhaustive.** With ``max_steps=80`` the agent visits at
  least N distinct (state, action) pairs before any repetition,
  where N is bounded by the state space's branching factor.
* **Carries learning across runs.** The :class:`GameProfile` records
  which (state-hash, action) edges proved productive, so a re-run on
  the same game starts with a head-start.

Action selection priority (per :meth:`choose_action`):

1. **Reset on stuck.** If :class:`StuckDetector` flagged the episode
   as stuck AND ``RESET`` is available, pick it.
2. **Skip dead actions.** Actions that the
   :class:`StateActionCounter` has marked dead from the current state
   are removed from the candidate set entirely.
3. **Untried-from-state actions first.** Among the remaining live
   actions at the current state, pick the one with ``count == 0`` if
   any exists; this drives exploration toward the unknown frontier.
4. **Productive action bonus.** Among already-tried live actions,
   prefer the one whose past observations on this game showed the
   highest ``avg_pixels`` change (via the wired
   :class:`FrameAnalyzer`). Falls back to least-tried if no analyzer.
5. **Click-action coords.** When ``ACTION6`` / ``ACTION7`` is the
   chosen complex action, the wired :class:`ClickTargetSampler`
   provides ``(x, y)``. Without a sampler the agent skips complex
   actions (the env would crash on missing data).

The agent inherits :class:`Sprint10DSLAgent`'s memory + state-graph
+ audit + game-profile plumbing unchanged. The only difference is
the action-selection policy: it never asks the LLM. Use this when
you want a strong heuristic baseline OR as a sub-agent inside an
orchestrator-style stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from cognithor.channels.program_synthesis.arc_agi3.dsl_agent import Sprint10DSLAgent
from cognithor.channels.program_synthesis.arc_agi3.state_action_counts import hash_state

if TYPE_CHECKING:
    from cognithor.channels.program_synthesis.arc_agi3.protocol import (
        FrameDataProtocol,
        GameActionProtocol,
    )


__all__ = ["GraphExplorerAgent"]


class GraphExplorerAgent(Sprint10DSLAgent):
    """Training-free state-graph exploration agent (no LLM).

    Same constructor as :class:`Sprint10DSLAgent` (same memory +
    state-graph + audit + profile + frame-analyzer + click-sampler
    plumbing); the difference is the override of
    :meth:`_select_action_from_state`, the policy that picks the
    next action given the current state hash + available actions.

    Use when you want a fast deterministic baseline OR as the action
    layer inside an orchestrator stack.
    """

    def choose_action(
        self,
        frames: list[FrameDataProtocol],
        latest_frame: FrameDataProtocol,
    ) -> GameActionProtocol:
        # Run the parent's preamble (frame bridge, memory append,
        # state-graph wiring, level-transition reset, frame-analyzer
        # feed) so all the bookkeeping side-effects fire identically.
        # Then OVERRIDE the action-selection by replacing the decoder.
        # The simplest path: temporarily swap the decoder, call super,
        # restore. Cleaner: build a thin local "decode" inline. Below
        # is the inline version — avoids the swap-restore dance.
        current_grid = self._bridge.extract_grid(latest_frame)
        current_hash = hash_state(current_grid)

        # Sprint-12 PR-5 — feed the FrameAnalyzer.
        if self._frame_analyzer is not None:
            self._frame_analyzer.analyze(current_grid, action=self._pending_action_name)

        # Level-transition reset (mirror parent semantics).
        if (
            self._last_levels_seen is not None
            and latest_frame.levels_completed > self._last_levels_seen
        ):
            self._memory.clear()
            self._state_counter.clear()
            self._state_graph = type(self._state_graph)()
            if self._frame_analyzer is not None:
                self._frame_analyzer.reset_for_new_level()
            if self._click_target_sampler is not None:
                self._click_target_sampler.reset_for_new_level()
            self._pending_action_name = None
            self._prev_grid = None
            self._prev_state_hash = None

        # Memory + state-graph append for the previous step.
        if self._pending_action_name is not None:
            self._memory.append(
                grid=current_grid,
                action_name=self._pending_action_name,
                levels_completed=latest_frame.levels_completed,
            )
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
                if self._prev_state_hash == current_hash:
                    self._state_counter.mark_dead(self._prev_state_hash, self._pending_action_name)

        # Step 0 — start the audit trail on the very first call.
        if self._audit_trail is not None and not self._audit_started:
            self._audit_trail.log_game_start()
            self._audit_started = True

        # ====== POLICY: pick next action via graph exploration ======
        chosen = self._select_via_graph(latest_frame, current_grid, current_hash)

        # Step 4 — bookkeeping.
        self._state_counter.increment(current_hash, chosen.name)
        self._pending_action_name = chosen.name
        self._pending_levels = latest_frame.levels_completed
        self._prev_grid = current_grid
        self._prev_state_hash = current_hash
        self._last_levels_seen = latest_frame.levels_completed

        # Audit log.
        if self._audit_trail is not None:
            pixels_changed = 0
            if self._prev_grid is not None and len(self._memory) > 1:
                prior = self._memory.window(2)
                if len(prior) >= 2 and prior[1].grid.shape == current_grid.shape:
                    pixels_changed = int(np.sum(prior[1].grid != current_grid))
            self._audit_trail.log_step(
                level=latest_frame.levels_completed,
                step=self._step_count,
                action=chosen.name,
                game_state=latest_frame.state.name,
                pixels_changed=pixels_changed,
            )
            self._step_count += 1

        return chosen

    # ------------------------------------------------------------------
    # action selection
    # ------------------------------------------------------------------

    def _select_via_graph(
        self,
        latest_frame: FrameDataProtocol,
        current_grid: np.ndarray[Any, Any],
        current_hash: str,
    ) -> GameActionProtocol:
        """Pick the next action using the priority described in the
        module docstring."""
        available = list(latest_frame.available_actions)
        if not available:
            raise RuntimeError(
                f"GraphExplorerAgent: latest frame for {latest_frame.game_id!r} "
                "has no available_actions"
            )

        # 1. Reset on stuck.
        if self._stuck.is_stuck(self._memory):
            for a in available:
                if a.name == "RESET":
                    a.reasoning = (
                        f"GraphExplorerAgent: stuck for ≥{self._stuck.threshold} "
                        "steps; RESET to escape the loop"
                    )
                    return a

        # 2. Skip dead actions for this state.
        dead = self._state_counter.all_dead_actions(current_hash)
        live = [a for a in available if a.name != "RESET" and a.name not in dead]
        if not live:
            # Only RESET (or all dead) — pick RESET if available, else first.
            for a in available:
                if a.name == "RESET":
                    a.reasoning = "GraphExplorerAgent: all non-RESET dead from this state"
                    return a
            available[0].reasoning = "GraphExplorerAgent: all actions exhausted; first-fallback"
            return available[0]

        # Filter complex actions (ACTION6/ACTION7) when no click sampler is
        # wired — emitting them with default (0, 0) crashes env.step.
        if self._click_target_sampler is None and self._click_cache is None:
            simple_live = [a for a in live if not getattr(a, "is_complex", lambda: False)()]
            if simple_live:
                live = simple_live

        # 3. Prefer untried-from-state actions (count == 0). Within that
        # set tie-break by *global* episode-count (least-used overall) so
        # the agent rotates across action families instead of spamming
        # ACTION1 in every fresh state.
        untried = [a for a in live if self._state_counter.count(current_hash, a.name) == 0]
        if untried:
            from cognithor.channels.program_synthesis.arc_agi3.episode_memory import (
                count_actions,
            )

            global_counts = count_actions(self._memory)
            min_global = min(global_counts.get(a.name, 0) for a in untried)
            globally_least = [a for a in untried if global_counts.get(a.name, 0) == min_global]
            chosen = self._tie_break_by_productivity(globally_least)
            chosen = self._maybe_attach_click_data(chosen, current_grid)
            chosen.reasoning = (
                f"GraphExplorerAgent: untried-from-state ({chosen.name}, "
                f"{len(dead)} known-dead skipped, {len(untried)} untried, "
                f"global-count={min_global})"
            )
            return chosen

        # 4. All live actions tried — pick least-tried, ties by productivity.
        min_count = min(self._state_counter.count(current_hash, a.name) for a in live)
        candidates = [
            a for a in live if self._state_counter.count(current_hash, a.name) == min_count
        ]
        chosen = self._tie_break_by_productivity(candidates)
        chosen = self._maybe_attach_click_data(chosen, current_grid)
        chosen.reasoning = (
            f"GraphExplorerAgent: least-tried-from-state "
            f"({chosen.name} picked {min_count}× from this state, "
            f"{len(dead)} known-dead skipped)"
        )
        return chosen

    def _tie_break_by_productivity(
        self, candidates: list[GameActionProtocol]
    ) -> GameActionProtocol:
        """When several actions have the same count, prefer the one with
        the highest historical ``avg_pixels`` change (productivity).

        Without a wired :class:`FrameAnalyzer` falls back to the first
        action — the order in ``available_actions`` is the upstream
        SDK's preference, which is a stable enough tie-break.
        """
        if self._frame_analyzer is None or len(candidates) == 1:
            return candidates[0]
        summary = self._frame_analyzer.get_action_summary()
        if not summary:
            return candidates[0]

        def _productivity(a: GameActionProtocol) -> float:
            stats = summary.get(a.name)
            if stats is None:
                return -1.0  # never observed → treat as low productivity for tie-break
            return float(stats.get("avg_pixels", 0.0))

        return max(candidates, key=_productivity)

    def _maybe_attach_click_data(
        self, action: GameActionProtocol, current_grid: np.ndarray[Any, Any]
    ) -> GameActionProtocol:
        """Attach ``(x, y)`` to a complex action via the wired sampler.

        For ACTION6/ACTION7 the env requires ``data={"x": ..., "y": ...}``.
        We pull from the click_target_sampler (preferred) or, failing
        that, the click_cache (toggle fast-path). If neither is wired
        the caller already filtered out complex actions.
        """
        is_complex = getattr(action, "is_complex", lambda: False)()
        if not is_complex:
            return action
        if self._click_target_sampler is not None:
            xy = self._click_target_sampler.next_click(current_grid)
            if xy is not None:
                x, y = xy
                action.set_data({"x": int(x), "y": int(y)})
                return action
        # Defensive fallback — pin to centre of grid so the env doesn't
        # crash on missing data. Better than (0, 0) which always lands
        # on the same corner.
        h, w = current_grid.shape
        action.set_data({"x": w // 2, "y": h // 2})
        return action
