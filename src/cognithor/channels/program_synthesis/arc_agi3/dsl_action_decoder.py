# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-11 Wave-4 — DSL-aware action decoder.

The Wave-4 :class:`DSLActionDecoder` is the first decoder that uses
the Sprint-10 DSL + the Wave-3 episode memory to score the
available actions. The policy is intentionally simple — it's the
strong baseline against which Wave-5's LLM-driven decoder will be
measured.

Policy (in order of priority):

1. **Reset on stuck**: if the :class:`StuckDetector` flags the
   episode as stuck AND ``RESET`` is available, pick it. Cheap escape
   from a dead loop.
2. **Least-tried among non-RESET actions**: pick the action that has
   the lowest historical pick count in this episode, preferring the
   first such action when ties occur. Penalises over-explored
   actions and prevents the agent from spending the entire 80-action
   budget on a single move.
3. **Fall back to first available** if step 2 produces no candidate
   (e.g. only RESET is available).

Wave-5 will subclass this with an LLM-prior-weighted score; the
core "explore the unexplored" policy stays the same.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cognithor.channels.program_synthesis.arc_agi3.action_decoder import ActionDecoder
from cognithor.channels.program_synthesis.arc_agi3.episode_memory import (
    EpisodeMemory,
    StuckDetector,
    count_actions,
)
from cognithor.channels.program_synthesis.arc_agi3.state_action_counts import (
    StateActionCounter,
    hash_state,
)

if TYPE_CHECKING:
    from cognithor.channels.program_synthesis.arc_agi3.protocol import (
        FrameDataProtocol,
        GameActionProtocol,
    )


class DSLActionDecoder(ActionDecoder):
    """Stateful decoder over an :class:`EpisodeMemory`.

    The decoder reads from the memory but never mutates it — the
    agent is responsible for appending new steps after each call.
    """

    def __init__(
        self,
        *,
        memory: EpisodeMemory,
        stuck_detector: StuckDetector | None = None,
        state_counter: StateActionCounter | None = None,
    ) -> None:
        self._memory = memory
        self._stuck = stuck_detector if stuck_detector is not None else StuckDetector()
        # Sprint-12: Blind-Squirrel-style state-keyed counts. When None,
        # falls back to global episode counting (Wave-4 behaviour).
        self._state_counter = state_counter

    @property
    def memory(self) -> EpisodeMemory:
        return self._memory

    @property
    def state_counter(self) -> StateActionCounter | None:
        """Read-only access for the agent's pre/post-step bookkeeping."""
        return self._state_counter

    def pick_action(
        self,
        frames: list[FrameDataProtocol],
        latest_frame: FrameDataProtocol,
        available_actions: list[GameActionProtocol],
    ) -> tuple[GameActionProtocol, str]:
        # Step 1 — reset on stuck if RESET is available.
        if self._stuck.is_stuck(self._memory):
            for action in available_actions:
                if action.name == "RESET":
                    return action, (
                        f"DSLActionDecoder: stuck for ≥{self._stuck.threshold} steps; "
                        "RESET to escape the loop"
                    )

        # Filter out complex actions (ACTION6, ACTION7) — they need
        # ``set_data({"x": .., "y": ..})`` coords, which the DSL decoder
        # does NOT produce. The fast-path / click-target sampler / LLM
        # are the only paths that should ever emit complex actions. If
        # we let one slip through, env.step crashes with KeyError 'x'.
        simple_actions = [
            a for a in available_actions if not getattr(a, "is_complex", lambda: False)()
        ]

        # Step 2 — Sprint-12 state-keyed prioritisation when available.
        # Picks the action that has been tried fewest times **from the
        # current state**, skipping any action proven dead from that
        # state (no-op transition). This mirrors the Blind-Squirrel
        # state-graph exploration policy.
        last_step = self._memory.last
        if self._state_counter is not None and last_step is not None:
            current_hash = hash_state(last_step.grid)
            dead_here = self._state_counter.all_dead_actions(current_hash)
            live = [a for a in simple_actions if a.name != "RESET" and a.name not in dead_here]
            if live:
                best = min(
                    live,
                    key=lambda a: self._state_counter.count(current_hash, a.name),  # type: ignore[union-attr]
                )
                best_count = self._state_counter.count(current_hash, best.name)
                return best, (
                    f"DSLActionDecoder: least-tried-from-state "
                    f"({best.name} picked {best_count}× from this state, "
                    f"{len(dead_here)} known-dead skipped)"
                )

        # Step 3 — fallback (Wave-4): least-tried globally.
        counts = count_actions(self._memory)
        candidates = [a for a in simple_actions if a.name != "RESET"]
        if candidates:
            best = min(candidates, key=lambda a: counts.get(a.name, 0))
            best_count = counts.get(best.name, 0)
            return best, (
                f"DSLActionDecoder: least-tried non-RESET ({best.name} picked {best_count}× so far)"
            )

        # Step 4 — only RESET (or only complex actions) available.
        # Prefer a simple action when one exists (RESET); otherwise the
        # first available is the only choice.
        if simple_actions:
            return simple_actions[0], (
                "DSLActionDecoder: only RESET available among simple actions, picking it"
            )
        return available_actions[0], (
            "DSLActionDecoder: no simple actions available, falling back to first option"
        )


__all__ = ["DSLActionDecoder"]
