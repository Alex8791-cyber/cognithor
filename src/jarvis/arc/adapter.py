"""ARC-AGI-3 environment adapter: wraps the SDK into a clean ArcObservation API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from jarvis.arc.error_handler import EnvironmentConnectionError, safe_frame_extract
from jarvis.utils.logging import get_logger

__all__ = [
    "ArcObservation",
    "ArcEnvironmentAdapter",
]

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# ArcObservation dataclass
# ---------------------------------------------------------------------------


@dataclass
class ArcObservation:
    """Normalised, step-level snapshot of one ARC game frame.

    Attributes:
        raw_grid: 64x64 int8 colour-index grid extracted from the SDK frame.
        game_state: GameState enum value (or string representation).
        step_number: Cumulative step counter across the whole episode.
        level: Current level index (derived from levels_completed).
        levels_completed: Number of levels fully solved so far.
        grid_diff: Boolean mask of pixels that changed vs. the previous frame.
        changed_pixels: Count of pixels that differ from the previous frame.
        action_history: Ordered list of actions taken to reach this observation.
        available_actions: Actions the SDK reports as legal in this state.
        win_levels: Number of win-counted levels per the SDK scorecard field.
    """

    raw_grid: np.ndarray
    game_state: Any
    step_number: int
    level: int
    levels_completed: int
    grid_diff: np.ndarray | None = None
    changed_pixels: int = 0
    action_history: list[Any] = field(default_factory=list)
    available_actions: list[Any] = field(default_factory=list)
    win_levels: int = 0


# ---------------------------------------------------------------------------
# ArcEnvironmentAdapter
# ---------------------------------------------------------------------------


class ArcEnvironmentAdapter:
    """Thin adapter around the ``arc_agi`` SDK for one game session.

    The SDK is imported lazily inside :meth:`initialize` so that the module is
    importable even when ``arc_agi`` is not installed.

    Args:
        game_id: The environment identifier passed to ``arcade.make()``.
    """

    def __init__(self, game_id: str) -> None:
        self.game_id = game_id
        self.arcade: Any = None
        self.env: Any = None
        self.current_obs: ArcObservation | None = None
        self.previous_grid: np.ndarray | None = None
        self.step_count: int = 0
        self.level_step_count: int = 0
        self.total_resets: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def initialize(self) -> ArcObservation:
        """Create the arcade, make the environment, and return the first frame.

        Returns:
            The initial :class:`ArcObservation` after ``env.reset()``.

        Raises:
            EnvironmentConnectionError: If the SDK is unavailable or
                ``arcade.make()`` returns ``None``.
        """
        try:
            import arc_agi  # type: ignore[import]
        except ImportError as exc:
            raise EnvironmentConnectionError(
                "arc_agi SDK is not installed. Install it to use ArcEnvironmentAdapter."
            ) from exc

        self.arcade = arc_agi.Arcade()
        self.env = self.arcade.make(self.game_id)

        if self.env is None:
            raise EnvironmentConnectionError(
                f"arcade.make('{self.game_id}') returned None — check that the game_id is valid."
            )

        log.info("arc.adapter.initialized", game_id=self.game_id)
        raw = self.env.reset()
        return self._process_frame(raw, action=None)

    def act(self, action: Any, data: dict[str, Any] | None = None) -> ArcObservation:
        """Send *action* to the environment and return the resulting observation.

        Args:
            action: A ``GameAction`` enum value.
            data: Optional payload forwarded to ``env.step()``
                (e.g. ``{"x": 32, "y": 32}``).

        Returns:
            The new :class:`ArcObservation` after the step.
        """
        raw = self.env.step(action, data=data or {})
        self.step_count += 1
        self.level_step_count += 1
        log.debug(
            "arc.adapter.act",
            action=str(action),
            step=self.step_count,
            level_step=self.level_step_count,
        )
        return self._process_frame(raw, action=action)

    def reset_level(self) -> ArcObservation:
        """Send a RESET action and return the observation for the fresh level.

        Returns:
            The :class:`ArcObservation` from the post-reset frame.
        """
        try:
            from arc_agi import GameAction  # type: ignore[import]
        except ImportError as exc:
            raise EnvironmentConnectionError(
                "arc_agi SDK is not installed — cannot import GameAction for RESET."
            ) from exc

        raw = self.env.step(GameAction.RESET)
        self.total_resets += 1
        self.level_step_count = 0
        log.info("arc.adapter.reset_level", total_resets=self.total_resets)
        return self._process_frame(raw, action="RESET")

    def get_scorecard(self) -> Any:
        """Return the arcade scorecard (``EnvironmentScorecard`` with ``.score``).

        Returns:
            The object returned by ``self.arcade.get_scorecard()``.
        """
        return self.arcade.get_scorecard()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_frame(self, raw: Any, action: Any) -> ArcObservation:
        """Convert a raw SDK frame into a rich :class:`ArcObservation`.

        Computes the pixel-level diff against the previous frame, accumulates
        action history, and extracts all metadata fields from ``raw``.

        Args:
            raw: ``FrameDataRaw`` object (or duck-typed stand-in) from the SDK.
            action: The action that produced this frame (``None`` on reset).

        Returns:
            Populated :class:`ArcObservation`.
        """
        grid = self._extract_grid(raw)

        # Pixel diff against the previous frame
        grid_diff: np.ndarray | None = None
        changed_pixels: int = 0
        if self.previous_grid is not None:
            grid_diff = grid != self.previous_grid
            changed_pixels = int(np.sum(grid_diff))

        # Extract SDK metadata with safe fallbacks
        game_state = getattr(raw, "state", None)
        levels_completed: int = int(getattr(raw, "levels_completed", 0) or 0)
        win_levels: int = int(getattr(raw, "win_levels", 0) or 0)

        raw_actions = getattr(raw, "available_actions", None)
        available_actions: list[Any] = list(raw_actions) if raw_actions is not None else []

        # Build action history from the previous obs (if any)
        prev_history: list[Any] = []
        if self.current_obs is not None:
            prev_history = list(self.current_obs.action_history)
        if action is not None:
            prev_history.append(action)

        obs = ArcObservation(
            raw_grid=grid,
            game_state=game_state,
            step_number=self.step_count,
            level=levels_completed,
            levels_completed=levels_completed,
            grid_diff=grid_diff,
            changed_pixels=changed_pixels,
            action_history=prev_history,
            available_actions=available_actions,
            win_levels=win_levels,
        )

        # Update adapter state
        self.previous_grid = grid.copy()
        self.current_obs = obs

        log.debug(
            "arc.adapter.frame_processed",
            step=self.step_count,
            changed_pixels=changed_pixels,
            game_state=str(game_state),
            levels_completed=levels_completed,
        )
        return obs

    def _extract_grid(self, raw: Any) -> np.ndarray:
        """Delegate frame extraction to :func:`safe_frame_extract`.

        Args:
            raw: Raw observation object from the SDK.

        Returns:
            2-D ``(64, 64)`` int8 numpy array.
        """
        return safe_frame_extract(raw)
