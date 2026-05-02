# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-11 Wave-3 — Episode-state tracking for ARC-AGI-3 agents.

ARC-AGI-3 agents see one frame at a time. To reason about progress
("are we still moving?", "is this an unproductive loop?", "should we
reset?") they need a small structured memory of the recent past.

This module ships three pure components, each composable on its own:

- :class:`EpisodeMemory` — a ring-buffer of ``(frame, action)``
  tuples. Bounded so it never grows unboundedly even when an agent
  hits the upstream MAX_ACTIONS=80.
- :class:`ChangeDetector` — compares two frames and returns a
  structured similarity report (cells changed, level-up, full-reset).
- :class:`StuckDetector` — flags when the agent has been stuck for
  ``threshold`` frames (no grid change AND no level progress).

Wave-4's :class:`Sprint10DSLAgent` wires all three into a stateful
action policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from cognithor.channels.program_synthesis.arc_agi3.protocol import (
        FrameDataProtocol,
    )


_Grid = NDArray[np.int8]


@dataclass(frozen=True)
class EpisodeStep:
    """A single ``(grid, action_name, levels_completed)`` snapshot.

    Frozen so the memory can be safely shared between the agent and
    the change/stuck detectors without cloning. The grid is a copy of
    the bridge output; the action name is the upstream
    ``GameAction.name`` string, not the enum (keeps the memory
    arcengine-import-clean).
    """

    grid: _Grid
    action_name: str
    levels_completed: int


class EpisodeMemory:
    """Bounded ring-buffer of recent steps.

    Default capacity 16 is safe — enough to detect stuck loops at
    typical game depths, small enough that comparison ops stay cheap.
    The buffer drops the oldest step on overflow; ``last`` and
    ``window`` always reflect the most-recent state.
    """

    def __init__(self, capacity: int = 16) -> None:
        if capacity < 1:
            raise ValueError(f"EpisodeMemory: capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._buffer: list[EpisodeStep] = []

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._buffer)

    def append(
        self,
        *,
        grid: _Grid,
        action_name: str,
        levels_completed: int,
    ) -> None:
        """Add a new step and evict the oldest if at capacity."""
        step = EpisodeStep(
            grid=grid.copy(),
            action_name=action_name,
            levels_completed=levels_completed,
        )
        self._buffer.append(step)
        if len(self._buffer) > self._capacity:
            self._buffer = self._buffer[-self._capacity :]

    @property
    def last(self) -> EpisodeStep | None:
        """The most-recent step, or None if the memory is empty."""
        return self._buffer[-1] if self._buffer else None

    def window(self, n: int) -> list[EpisodeStep]:
        """Return the last *n* steps (most recent first), bounded by len()."""
        if n < 0:
            raise ValueError(f"EpisodeMemory.window: n must be >= 0, got {n}")
        if n == 0:
            return []
        return list(reversed(self._buffer[-n:]))

    def clear(self) -> None:
        self._buffer = []


@dataclass(frozen=True)
class FrameChange:
    """Structured diff between two consecutive frames.

    - ``cells_changed`` — count of grid cells that differ
    - ``shape_changed`` — True if the two grids have different shape
    - ``levels_advanced`` — True if ``levels_completed`` increased
    - ``full_reset`` — mirror of :attr:`FrameDataProtocol.full_reset`
      (e.g. after a successful RESET action)
    """

    cells_changed: int
    shape_changed: bool
    levels_advanced: bool
    full_reset: bool

    @property
    def is_change(self) -> bool:
        """Any meaningful change occurred."""
        return (
            self.cells_changed > 0 or self.shape_changed or self.levels_advanced or self.full_reset
        )


class ChangeDetector:
    """Diff two grids + their wrapping FrameData.

    Pure: no internal state. Use case: feed the previous and current
    frame, get a :class:`FrameChange` describing what (if anything)
    moved.
    """

    @staticmethod
    def diff(
        previous_grid: _Grid,
        current_grid: _Grid,
        *,
        previous_levels: int,
        current_frame: FrameDataProtocol,
    ) -> FrameChange:
        if previous_grid.shape != current_grid.shape:
            return FrameChange(
                cells_changed=int(previous_grid.size + current_grid.size),
                shape_changed=True,
                levels_advanced=current_frame.levels_completed > previous_levels,
                full_reset=current_frame.full_reset,
            )
        diff = np.not_equal(previous_grid, current_grid)
        return FrameChange(
            cells_changed=int(diff.sum()),
            shape_changed=False,
            levels_advanced=current_frame.levels_completed > previous_levels,
            full_reset=current_frame.full_reset,
        )


class StuckDetector:
    """Flag when the agent has been stuck for ``threshold`` frames.

    "Stuck" = the last ``threshold`` consecutive steps showed
    no cell change AND no level progress. The detector is read-only
    against an :class:`EpisodeMemory`; it doesn't mutate state.

    Wave-4's policy uses this to decide when to RESET — staying stuck
    longer than ``threshold`` wastes the upstream MAX_ACTIONS budget.
    """

    DEFAULT_THRESHOLD: int = 5

    def __init__(self, threshold: int = DEFAULT_THRESHOLD) -> None:
        if threshold < 1:
            raise ValueError(f"StuckDetector: threshold must be >= 1, got {threshold}")
        self._threshold = threshold

    @property
    def threshold(self) -> int:
        return self._threshold

    def is_stuck(self, memory: EpisodeMemory) -> bool:
        """Return True if the last ``threshold`` steps showed no change."""
        if len(memory) < self._threshold + 1:
            return False
        window = memory.window(self._threshold + 1)
        # ``window`` is most-recent-first; pair adjacent steps to
        # detect change in either direction.
        for i in range(len(window) - 1):
            current = window[i]
            previous = window[i + 1]
            if current.levels_completed > previous.levels_completed:
                return False
            if current.grid.shape != previous.grid.shape:
                return False
            if not np.array_equal(current.grid, previous.grid):
                return False
        return True


@dataclass(frozen=True)
class _ActionFrequency:
    """Helper for Wave-4: counts how often each action has been picked."""

    counts: dict[str, int] = field(default_factory=dict)


def count_actions(memory: EpisodeMemory) -> dict[str, int]:
    """Return per-action-name pick counts over the current memory.

    Convenience helper for Wave-4 policies that want to penalise
    over-used actions ("explored already") without re-implementing
    the counting loop. Returns a fresh dict on every call so callers
    can mutate freely.
    """
    counts: dict[str, int] = {}
    for step in memory.window(len(memory)):
        counts[step.action_name] = counts.get(step.action_name, 0) + 1
    return counts


class ActionStreakDetector:
    """Sprint-16 Hebel 2 — level-progress-aware action-streak detection.

    The Hebel-1 ``StateActionCounter`` keys on ``(state_hash, action)``
    so it never trips when each pick *also* changes the state (e.g.
    a cursor-click in a click-target game shifts the cursor pixel,
    making the next state hash distinct, resetting the count). Phase-A
    run #11 confirmed this empirically: ACTION6 picked 39 / 40 with
    Hebel 1 active because the click moved the cursor pixel each time
    and the per-state count never reached the threshold.

    This detector instead tracks the most-recent action *names* in
    memory order and asks: did one action dominate the last ``window``
    picks **without any level progress in that window**? If yes, that
    action is "stuck" and the decoder should forbid it.

    The level-progress guard is what keeps this from over-firing on a
    legitimate strategy that picks the same action 5 times in a row
    *while levels are advancing*. Stuck = same action + flat level
    count.
    """

    DEFAULT_WINDOW: int = 5
    DEFAULT_THRESHOLD: int = 4

    def __init__(
        self,
        *,
        window: int = DEFAULT_WINDOW,
        threshold: int = DEFAULT_THRESHOLD,
    ) -> None:
        if window < 2:
            raise ValueError(f"ActionStreakDetector: window must be >= 2, got {window}")
        if threshold < 2 or threshold > window:
            raise ValueError(
                f"ActionStreakDetector: threshold must be in [2, window={window}], got {threshold}"
            )
        self._window = window
        self._threshold = threshold

    @property
    def window(self) -> int:
        return self._window

    @property
    def threshold(self) -> int:
        return self._threshold

    def dominant_stuck_action(self, memory: EpisodeMemory) -> str | None:
        """Return the action name that dominates a stuck window, or ``None``.

        "Dominates" = picked at least ``threshold`` times in the last
        ``window`` steps. "Stuck" = ``levels_completed`` did not
        increase across that window.

        Returns ``None`` when:
        * memory has fewer than ``window`` steps yet,
        * no single action reached the threshold,
        * level progressed during the window (so the streak is fine).
        """
        if len(memory) < self._window:
            return None
        recent = memory.window(self._window)  # most-recent first
        # Level progress check first — cheap, common short-circuit.
        first_level = recent[-1].levels_completed
        last_level = recent[0].levels_completed
        if last_level > first_level:
            return None
        # Action dominance.
        counts: dict[str, int] = {}
        for step in recent:
            counts[step.action_name] = counts.get(step.action_name, 0) + 1
        for name, count in counts.items():
            if count >= self._threshold:
                return name
        return None


__all__ = [
    "ActionStreakDetector",
    "ChangeDetector",
    "EpisodeMemory",
    "EpisodeStep",
    "FrameChange",
    "StuckDetector",
    "count_actions",
]
