"""ARC-AGI-3 goal inference: derives high-level objectives from episode transitions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis.arc.episode_memory import EpisodeMemory

__all__ = [
    "GoalInferenceModule",
    "GoalType",
    "InferredGoal",
]

# ---------------------------------------------------------------------------
# Enums & dataclasses
# ---------------------------------------------------------------------------

_WIN_PIXEL_THRESHOLD = 100  # avg pixels_changed that suggests a clear-board goal


class GoalType(Enum):
    """High-level categories of ARC-AGI-3 puzzle objectives."""

    UNKNOWN = "UNKNOWN"
    REACH_STATE = "REACH_STATE"
    CLEAR_BOARD = "CLEAR_BOARD"
    FILL_PATTERN = "FILL_PATTERN"
    NAVIGATE = "NAVIGATE"
    AVOID = "AVOID"
    SEQUENCE = "SEQUENCE"


@dataclass
class InferredGoal:
    """A single inferred goal with associated confidence and evidence."""

    goal_type: GoalType
    description: str
    confidence: float  # 0.0 – 1.0
    evidence: list[str] = field(default_factory=list)
    estimated_steps_remaining: int = -1


# ---------------------------------------------------------------------------
# GoalInferenceModule
# ---------------------------------------------------------------------------


class GoalInferenceModule:
    """Derives the puzzle's implicit goal from observed episode transitions.

    After calling :meth:`analyze_win_condition` the inferred goals are stored in
    :attr:`current_goals` sorted by confidence (highest first).  The module is
    intentionally stateless between analyses — each call fully replaces the
    stored goals.

    Args:
        None – all state is initialised as empty collections.
    """

    def __init__(self) -> None:
        self.current_goals: list[InferredGoal] = []
        self.win_states_observed: list[str] = []
        self.game_over_states_observed: list[str] = []
        self._level_progression_data: list[dict] = []

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze_win_condition(self, episode_memory: EpisodeMemory) -> list[InferredGoal]:
        """Inspect *episode_memory* transitions and infer the current goal.

        Four inference strategies are applied in order:

        1. **Win transitions** – if any transition led to a WIN, emit a
           ``REACH_STATE`` goal naming the most common winning action.
        2. **Game-over patterns** – if any transition led to GAME_OVER, emit an
           ``AVOID`` goal naming the most common dangerous action.
        3. **Pixel change patterns** – if the average ``pixels_changed`` across
           all transitions exceeds :data:`_WIN_PIXEL_THRESHOLD`, emit a
           ``CLEAR_BOARD`` goal.
        4. **Fallback** – if no wins have been observed yet, emit an ``UNKNOWN``
           goal with low confidence.

        The resulting list is sorted by confidence (descending) and stored in
        :attr:`current_goals`.

        Args:
            episode_memory: The current episode's :class:`~jarvis.arc.episode_memory.EpisodeMemory`.

        Returns:
            The sorted list of :class:`InferredGoal` objects (same object as
            :attr:`current_goals`).
        """
        goals: list[InferredGoal] = []
        transitions = episode_memory.transitions

        win_transitions = [t for t in transitions if t.resulted_in_win]
        game_over_transitions = [t for t in transitions if t.resulted_in_game_over]

        # -- Strategy 1: win transitions → REACH_STATE ----------------------
        if win_transitions:
            self.win_states_observed.extend(t.next_state_hash for t in win_transitions)
            win_action_counts: Counter[str] = Counter(t.action for t in win_transitions)
            most_common_win_action, win_count = win_action_counts.most_common(1)[0]
            confidence = min(0.5 + 0.1 * win_count, 0.95)
            goals.append(
                InferredGoal(
                    goal_type=GoalType.REACH_STATE,
                    description=f"Reach win state via {most_common_win_action}",
                    confidence=confidence,
                    evidence=[
                        f"{win_count} winning transition(s) observed",
                        f"Most effective action: {most_common_win_action}",
                    ],
                )
            )

        # -- Strategy 2: game-over patterns → AVOID -------------------------
        if game_over_transitions:
            self.game_over_states_observed.extend(t.next_state_hash for t in game_over_transitions)
            go_action_counts: Counter[str] = Counter(t.action for t in game_over_transitions)
            most_common_go_action, go_count = go_action_counts.most_common(1)[0]
            confidence = min(0.4 + 0.1 * go_count, 0.85)
            goals.append(
                InferredGoal(
                    goal_type=GoalType.AVOID,
                    description=f"Avoid game-over triggered by {most_common_go_action}",
                    confidence=confidence,
                    evidence=[
                        f"{go_count} game-over transition(s) observed",
                        f"Most dangerous action: {most_common_go_action}",
                    ],
                )
            )

        # -- Strategy 3: pixel change patterns → CLEAR_BOARD ----------------
        if transitions:
            avg_pixels = sum(t.pixels_changed for t in transitions) / len(transitions)
            if avg_pixels > _WIN_PIXEL_THRESHOLD:
                goals.append(
                    InferredGoal(
                        goal_type=GoalType.CLEAR_BOARD,
                        description=(
                            f"Clear the board (avg {avg_pixels:.0f} pixels changed per step)"
                        ),
                        confidence=0.6,
                        evidence=[
                            f"Average pixels changed: {avg_pixels:.1f}",
                            f"Threshold: {_WIN_PIXEL_THRESHOLD}",
                        ],
                    )
                )

        # -- Strategy 4: no wins observed → UNKNOWN -------------------------
        if not win_transitions:
            goals.append(
                InferredGoal(
                    goal_type=GoalType.UNKNOWN,
                    description="Goal unknown — no winning transitions observed yet",
                    confidence=0.1,
                    evidence=["No winning transitions in episode memory"],
                )
            )

        # Sort by confidence descending and persist
        goals.sort(key=lambda g: g.confidence, reverse=True)
        self.current_goals = goals
        return goals

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_best_goal(self) -> InferredGoal:
        """Return the goal with the highest confidence.

        If :attr:`current_goals` is empty (no analysis run yet), returns a
        synthetic ``UNKNOWN`` goal with ``confidence=0.0``.
        """
        if not self.current_goals:
            return InferredGoal(
                goal_type=GoalType.UNKNOWN,
                description="No goals analysed yet",
                confidence=0.0,
            )
        return max(self.current_goals, key=lambda g: g.confidence)

    # ------------------------------------------------------------------
    # Level lifecycle
    # ------------------------------------------------------------------

    def on_level_complete(self, level_data: dict) -> None:
        """Record *level_data* for cross-level progression analysis.

        Args:
            level_data: Arbitrary dict describing the completed level
                (e.g. ``{"level": 1, "steps": 42}``).
        """
        self._level_progression_data.append(level_data)

    # ------------------------------------------------------------------
    # LLM summary
    # ------------------------------------------------------------------

    def get_summary_for_llm(self) -> str:
        """Return a compact text summary of the top-3 current goals for LLM injection."""
        if not self.current_goals:
            return "No goals inferred yet."

        lines: list[str] = ["Inferred goals (top 3):"]
        for i, goal in enumerate(self.current_goals[:3], start=1):
            lines.append(
                f"  {i}. [{goal.goal_type.value}] {goal.description}"
                f" (confidence: {goal.confidence:.0%})"
            )
            for ev in goal.evidence[:2]:
                lines.append(f"     - {ev}")

        if self._level_progression_data:
            lines.append(f"Levels completed: {len(self._level_progression_data)}")

        return "\n".join(lines)
