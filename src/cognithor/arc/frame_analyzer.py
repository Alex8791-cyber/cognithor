"""ARC-AGI-3 Frame Analyzer — tracks objects and movement patterns in game frames.

Analyzes 64x64 color-indexed grids to detect:
- Moving regions (objects that change position between frames)
- Static regions (background/terrain)
- Action effects (which actions cause which movements)
- Progress signals (moving toward unexplored areas)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cognithor.utils.logging import get_logger

__all__ = ["FrameAnalyzer", "MovementInfo"]

log = get_logger(__name__)


@dataclass
class MovementInfo:
    """Describes a movement detected between two frames."""

    action: str
    pixels_changed: int
    min_row: int
    max_row: int
    min_col: int
    max_col: int
    direction_row: int = 0  # negative = up, positive = down
    direction_col: int = 0  # negative = left, positive = right


class FrameAnalyzer:
    """Tracks objects and learns action effects from game frames."""

    def __init__(self) -> None:
        self._prev_grid: np.ndarray | None = None  # type: ignore[type-arg]
        self._prev_movement: MovementInfo | None = None
        self._action_effects: dict[str, list[MovementInfo]] = {}
        self._static_mask: np.ndarray | None = None  # type: ignore[type-arg]
        self._visited_positions: set[tuple[int, int]] = set()
        self._frame_count: int = 0

    def analyze(self, grid: np.ndarray[Any, Any], action: str | None = None) -> MovementInfo | None:
        """Analyze a new frame, optionally with the action that produced it.

        Returns MovementInfo if movement was detected, None otherwise.
        """
        if grid.ndim == 3:
            grid = grid[0]  # Remove batch dimension

        self._frame_count += 1
        movement = None

        if self._prev_grid is not None:
            diff = grid != self._prev_grid
            n_changed = int(np.sum(diff))

            if n_changed > 0:
                rows = np.where(diff.any(axis=1))[0]
                cols = np.where(diff.any(axis=0))[0]

                # Track direction by comparing centers of changed regions
                center_row = int(np.mean(rows))
                center_col = int(np.mean(cols))

                prev_center_row = 0
                prev_center_col = 0
                if self._prev_movement:
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
                    direction_row=center_row - prev_center_row if self._prev_movement else 0,
                    direction_col=center_col - prev_center_col if self._prev_movement else 0,
                )

                # Record action effect
                if action:
                    self._action_effects.setdefault(action, []).append(movement)

                # Track visited positions
                self._visited_positions.add((center_row, center_col))

                self._prev_movement = movement

            # Build static mask (pixels that never change)
            if self._static_mask is None:
                self._static_mask = ~diff
            else:
                self._static_mask &= ~diff

        self._prev_grid = grid.copy()
        return movement

    def get_action_summary(self) -> dict[str, dict[str, float]]:
        """Summarize what each action does on average."""
        summary = {}
        for action, movements in self._action_effects.items():
            if not movements:
                continue
            avg_pixels = sum(m.pixels_changed for m in movements) / len(movements)
            avg_dir_row = sum(m.direction_row for m in movements) / len(movements)
            avg_dir_col = sum(m.direction_col for m in movements) / len(movements)
            summary[action] = {
                "avg_pixels": avg_pixels,
                "avg_direction_row": avg_dir_row,
                "avg_direction_col": avg_dir_col,
                "count": len(movements),
            }
        return summary

    def suggest_action(self, available_actions: list[Any]) -> Any | None:
        """Suggest the most promising action based on learned effects.

        Strategy: cycle through actions that cause meaningful directional
        movement, not just pixel count. Prefer actions that haven't been
        used recently (round-robin with directional bias).
        """
        if not self._action_effects or not available_actions:
            return None

        # Find actions that cause real directional movement (not just noise)
        directional = []
        for a in available_actions:
            key = a.name if hasattr(a, "name") else str(a)
            effects = self._action_effects.get(key, [])
            if not effects:
                return a  # Untested = highest priority

            recent = effects[-10:]
            avg_dir = abs(sum(m.direction_row for m in recent)) + abs(
                sum(m.direction_col for m in recent)
            )
            avg_dir /= len(recent)  # type: ignore[assignment]
            directional.append((a, avg_dir, len(effects)))

        # Sort by least-used first (balanced exploration), then by direction
        directional.sort(key=lambda x: (x[2], -x[1]))
        return directional[0][0] if directional else None

    def reset_for_new_level(self) -> None:
        """Keep action effects but clear position tracking."""
        self._prev_grid = None
        self._prev_movement = None
        self._static_mask = None
        self._visited_positions.clear()
        self._frame_count = 0
        # Keep _action_effects for transfer learning
