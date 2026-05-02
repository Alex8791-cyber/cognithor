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
    ) -> None:
        self._memory = memory
        self._stuck = stuck_detector if stuck_detector is not None else StuckDetector()

    @property
    def memory(self) -> EpisodeMemory:
        return self._memory

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

        # Step 2 — least-tried non-RESET action.
        counts = count_actions(self._memory)
        candidates = [a for a in available_actions if a.name != "RESET"]
        if candidates:
            # Pick the candidate with the smallest historical count.
            # ``min`` is stable, so the first candidate with the
            # minimum count wins on ties.
            best = min(candidates, key=lambda a: counts.get(a.name, 0))
            best_count = counts.get(best.name, 0)
            return best, (
                f"DSLActionDecoder: least-tried non-RESET ({best.name} picked {best_count}× so far)"
            )

        # Step 3 — fall back to first available (likely RESET only).
        return available_actions[0], ("DSLActionDecoder: only RESET available, picking it")


__all__ = ["DSLActionDecoder"]
