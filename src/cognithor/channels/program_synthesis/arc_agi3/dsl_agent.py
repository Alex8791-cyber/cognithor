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

from typing import TYPE_CHECKING

import numpy as np

from cognithor.channels.program_synthesis.arc_agi3.agent import CognithorPSEAgent
from cognithor.channels.program_synthesis.arc_agi3.dsl_action_decoder import (
    DSLActionDecoder,
)
from cognithor.channels.program_synthesis.arc_agi3.episode_memory import (
    EpisodeMemory,
    StuckDetector,
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
        self._prev_grid: np.ndarray | None = None
        self._prev_state_hash: str | None = None

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
        # Step 1 — bridge the current frame to a Cognithor grid.
        # Errors (out-of-range, wrong shape) propagate; the upstream
        # harness logs and retries one frame later if this happens.
        current_grid = self._bridge.extract_grid(latest_frame)
        current_hash = hash_state(current_grid)

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
            # State-graph: record the (from, action, to) edge.
            if self._prev_grid is not None and self._prev_state_hash is not None:
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

        # Step 3 — delegate to the DSL decoder.
        chosen = self._decoder.decode(frames, latest_frame)

        # Step 4 — bookkeeping: increment the (current_state, chosen) count
        # so the next decode skips it if alternatives exist.
        self._state_counter.increment(current_hash, chosen.name)

        # Step 5 — remember choice + grid for the next call.
        self._pending_action_name = chosen.name
        self._pending_levels = latest_frame.levels_completed
        self._prev_grid = current_grid
        self._prev_state_hash = current_hash
        return chosen


__all__ = ["Sprint10DSLAgent"]
