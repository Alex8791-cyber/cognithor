# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-12 — FrameAnalyzer (lifted from cognithor.arc.frame_analyzer).

Learns per-action movement signatures from successive frames so the
agent can build a model of what each ACTIONx button actually does.
This is critical for keyboard-controlled games where ARC-AGI-3 does
not announce which action is "up", "down", "left", "right" — the
agent has to discover that mapping from observed pixel motion.

Usage::

    fa = FrameAnalyzer()
    fa.analyze(grid_t0, action=None)        # bootstrap
    fa.analyze(grid_t1, action="ACTION3")   # records "ACTION3 → down"
    summary = fa.get_action_summary()
    # → {"ACTION3": {"avg_pixels": 2, "avg_direction_row": +1, "count": 1, ...}}

The Wave-3 :class:`EpisodeMemory`/:class:`ChangeDetector` already
counts pixel-changes; FrameAnalyzer is the structured cousin that
also remembers direction + region, so an LLM-driven decoder can be
told "ACTION3 has historically moved your sprite downward" instead of
just "ACTION3 changed 2 pixels".

Stateless across game-switches; carries learned action effects across
levels of the same game (the operator calls
:meth:`reset_for_new_level` on level transitions).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["FrameAnalyzer", "MovementInfo"]


@dataclass
class MovementInfo:
    """One observed movement: which action caused which pixel-region delta."""

    action: str
    pixels_changed: int
    min_row: int
    max_row: int
    min_col: int
    max_col: int
    direction_row: int = 0  # negative = up, positive = down
    direction_col: int = 0  # negative = left, positive = right


class FrameAnalyzer:
    """Tracks objects + learns per-action movement signatures across frames."""

    def __init__(self) -> None:
        self._prev_grid: np.ndarray[Any, Any] | None = None
        self._prev_movement: MovementInfo | None = None
        self._action_effects: dict[str, list[MovementInfo]] = {}
        self._static_mask: np.ndarray[Any, Any] | None = None
        self._visited_positions: set[tuple[int, int]] = set()
        self._frame_count: int = 0

    def analyze(
        self,
        grid: np.ndarray[Any, Any],
        action: str | None = None,
    ) -> MovementInfo | None:
        """Analyze ``grid`` against the previous frame, optionally tagging the
        observed change with ``action``. Returns the :class:`MovementInfo` if
        a change was detected, ``None`` if the grid was identical to the
        previous one (or this is the first frame).
        """
        if grid.ndim == 3:
            grid = grid[0]

        self._frame_count += 1
        movement: MovementInfo | None = None

        if self._prev_grid is not None:
            diff = grid != self._prev_grid
            n_changed = int(np.sum(diff))

            if n_changed > 0:
                rows = np.where(diff.any(axis=1))[0]
                cols = np.where(diff.any(axis=0))[0]

                center_row = int(np.mean(rows))
                center_col = int(np.mean(cols))

                prev_center_row = 0
                prev_center_col = 0
                if self._prev_movement is not None:
                    prev_center_row = (
                        self._prev_movement.min_row + self._prev_movement.max_row
                    ) // 2
                    prev_center_col = (
                        self._prev_movement.min_col + self._prev_movement.max_col
                    ) // 2

                movement = MovementInfo(
                    action=action or "unknown",
                    pixels_changed=n_changed,
                    min_row=int(rows[0]),
                    max_row=int(rows[-1]),
                    min_col=int(cols[0]),
                    max_col=int(cols[-1]),
                    direction_row=(
                        center_row - prev_center_row if self._prev_movement is not None else 0
                    ),
                    direction_col=(
                        center_col - prev_center_col if self._prev_movement is not None else 0
                    ),
                )

                if action:
                    self._action_effects.setdefault(action, []).append(movement)

                self._visited_positions.add((center_row, center_col))
                self._prev_movement = movement

            if self._static_mask is None:
                self._static_mask = ~diff
            else:
                self._static_mask &= ~diff

        self._prev_grid = grid.copy()
        return movement

    def get_action_summary(self) -> dict[str, dict[str, float]]:
        """Return ``{action: {avg_pixels, avg_direction_row, avg_direction_col, count}}``."""
        summary: dict[str, dict[str, float]] = {}
        for action, movements in self._action_effects.items():
            if not movements:
                continue
            n = len(movements)
            summary[action] = {
                "avg_pixels": sum(m.pixels_changed for m in movements) / n,
                "avg_direction_row": sum(m.direction_row for m in movements) / n,
                "avg_direction_col": sum(m.direction_col for m in movements) / n,
                "count": float(n),
            }
        return summary

    def suggest_action(self, available_actions: list[Any]) -> Any | None:
        """Round-robin exploration with a bias toward directional actions.

        * Untested actions get top priority.
        * Among tested ones, prefer the least-used (balanced exploration).
        * Tie-break by stronger directional signal (high abs row+col delta).
        """
        if not available_actions:
            return None

        ranked: list[tuple[Any, float, int]] = []
        for a in available_actions:
            key = a.name if hasattr(a, "name") else str(a)
            effects = self._action_effects.get(key, [])
            if not effects:
                # Untested → highest priority.
                return a

            recent = effects[-10:]
            avg_dir = (
                sum(abs(m.direction_row) for m in recent)
                + sum(abs(m.direction_col) for m in recent)
            ) / len(recent)
            ranked.append((a, avg_dir, len(effects)))

        ranked.sort(key=lambda t: (t[2], -t[1]))
        return ranked[0][0] if ranked else None

    def reset_for_new_level(self) -> None:
        """Drop position tracking but **keep** learned action effects.

        Useful on level boundaries: the new level's geometry is fresh, but
        what each action does (up/down/left/right) is unchanged.
        """
        self._prev_grid = None
        self._prev_movement = None
        self._static_mask = None
        self._visited_positions.clear()
        self._frame_count = 0

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def visited_positions(self) -> frozenset[tuple[int, int]]:
        return frozenset(self._visited_positions)
