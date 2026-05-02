# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-12 — state-keyed action counts (Blind-Squirrel-style).

The Wave-4 :class:`DSLActionDecoder` counts each action **globally** in
the episode. That over-penalises useful actions: if ACTION1 worked in
state-A but failed in state-B, the global count grows the same.

Blind Squirrel (2nd place public ARC-AGI-3 leaderboard, 6.71 %, 13/25
games) keeps a per-(state, action) count: an action is "least-tried"
relative to the *current state*, not over the whole episode. This
mirrors the agent's actual exploration position. Plus: when an action
is **observed** to be a no-op from a state (the resulting frame equals
the source frame), the (state, action) pair is **dead** — never pick it
from that state again.

This module is small + pure: it only owns the counter / dead-edge
bookkeeping. The decoder integrates it on top of
:class:`EpisodeMemory` history.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    _Grid = NDArray[np.int8]


def hash_state(grid: _Grid) -> str:
    """Stable 16-hex-char hash of a grid for keyed lookups.

    Collisions are negligible at this length for the ~80-action episodes
    we run, but the key is small enough to cheap-store as a dict key
    everywhere. Mirrors the hashing in
    :class:`StateGraphNavigator._hash_grid`.
    """
    return hashlib.sha1(grid.tobytes()).hexdigest()[:16]


class StateActionCounter:
    """Per-(state-hash, action-name) tracking.

    Two channels:

    * ``count(state, action)`` — number of times we've **picked** this
      action from this state in the current episode
    * ``is_dead(state, action)`` — True if the action is known to be a
      no-op from this state (frame after = frame before)
    """

    def __init__(self) -> None:
        self._counts: dict[tuple[str, str], int] = defaultdict(int)
        self._dead: set[tuple[str, str]] = set()

    def count(self, state_hash: str, action_name: str) -> int:
        return self._counts[(state_hash, action_name)]

    def increment(self, state_hash: str, action_name: str) -> None:
        self._counts[(state_hash, action_name)] += 1

    def mark_dead(self, state_hash: str, action_name: str) -> None:
        self._dead.add((state_hash, action_name))

    def is_dead(self, state_hash: str, action_name: str) -> bool:
        return (state_hash, action_name) in self._dead

    def all_dead_actions(self, state_hash: str) -> set[str]:
        """Return all action names known dead from this state."""
        return {a for (s, a) in self._dead if s == state_hash}

    def clear(self) -> None:
        self._counts.clear()
        self._dead.clear()


__all__ = ["StateActionCounter", "hash_state"]
