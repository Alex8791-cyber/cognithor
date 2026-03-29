"""ARC-AGI-3 mechanics model: cross-level rule abstraction from episode transitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis.arc.episode_memory import EpisodeMemory

__all__ = [
    "Mechanic",
    "MechanicType",
    "MechanicsModel",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_OBSERVATIONS = 3  # minimum total observations before classifying
_HIGH_CHANGE_THRESHOLD = 0.9  # >90% change rate → MOVEMENT
_LOW_CHANGE_THRESHOLD = 0.1  # <10% change rate → NO_EFFECT
_EMA_ALPHA = 0.3  # exponential moving average weight for consistency updates


# ---------------------------------------------------------------------------
# Enums & dataclasses
# ---------------------------------------------------------------------------


class MechanicType(Enum):
    """Taxonomy of observable ARC-AGI-3 game mechanics."""

    MOVEMENT = "MOVEMENT"
    TRANSFORMATION = "TRANSFORMATION"
    CREATION = "CREATION"
    DESTRUCTION = "DESTRUCTION"
    TOGGLE = "TOGGLE"
    CONDITIONAL = "CONDITIONAL"
    NO_EFFECT = "NO_EFFECT"
    UNKNOWN = "UNKNOWN"


@dataclass
class Mechanic:
    """A generalised, reusable mechanic inferred from raw action-effect observations.

    Attributes:
        mechanic_type: The broad category this mechanic belongs to.
        action: The action string that triggers this mechanic.
        description: Human-readable description of what this mechanic does.
        observed_in_levels: List of level indices where this mechanic was observed.
        observation_count: Cumulative number of times this mechanic was updated.
        consistency_score: EMA-smoothed confidence that the mechanic behaves
            consistently (0.0–1.0).
        spatial_pattern: Optional string describing a spatial pattern (e.g. "diagonal").
        affected_colors: Grid colour indices involved in this mechanic.
        preconditions: Natural-language preconditions that must hold for the mechanic
            to trigger.
    """

    mechanic_type: MechanicType
    action: str
    description: str
    observed_in_levels: list[int] = field(default_factory=list)
    observation_count: int = 0
    consistency_score: float = 0.0
    spatial_pattern: str | None = None
    affected_colors: list[int] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# MechanicsModel
# ---------------------------------------------------------------------------


class MechanicsModel:
    """Cross-level rule abstraction layer.

    Generalises raw episode transitions into reusable :class:`Mechanic` objects.
    Mechanics are updated incrementally via an exponential moving average so that
    later observations can reinforce or erode earlier beliefs.

    Typical usage::

        model = MechanicsModel()
        model.analyze_transitions(episode_memory, current_level=0)
        reliable = model.get_reliable_mechanics(min_consistency=0.7)
        effect = model.predict_action_effect("ACTION1")
    """

    def __init__(self) -> None:
        self.mechanics: list[Mechanic] = []
        self.action_to_mechanics: dict[str, list[Mechanic]] = {}
        self._level_snapshots: list[dict] = []

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze_transitions(self, episode_memory: EpisodeMemory, current_level: int) -> None:
        """Inspect *episode_memory* and update mechanics for *current_level*.

        For each action in ``episode_memory.action_effect_map`` with at least
        :data:`_MIN_OBSERVATIONS` total observations the method:

        1. Classifies the action as :attr:`~MechanicType.NO_EFFECT`,
           :attr:`~MechanicType.MOVEMENT`, or :attr:`~MechanicType.CONDITIONAL`
           based on the change-rate thresholds.
        2. Looks for an existing :class:`Mechanic` for the same action and type
           (creating one if none exists).
        3. Updates ``observation_count``, ``observed_in_levels``, and
           ``consistency_score`` (EMA with ``alpha=0.3``).

        Args:
            episode_memory: The episode's accumulated statistics.
            current_level: Index of the level currently being played.
        """
        for action, stats in episode_memory.action_effect_map.items():
            total = stats["total"]
            if total < _MIN_OBSERVATIONS:
                continue

            caused_change = stats["caused_change"]
            change_rate = caused_change / total

            # Classify based on change rate
            if change_rate < _LOW_CHANGE_THRESHOLD:
                mechanic_type = MechanicType.NO_EFFECT
                description = (
                    f"Action {action!r} rarely causes any state change ({change_rate:.0%})"
                )
            elif change_rate >= _HIGH_CHANGE_THRESHOLD:
                mechanic_type = MechanicType.MOVEMENT
                description = (
                    f"Action {action!r} consistently moves/changes state ({change_rate:.0%})"
                )
            else:
                mechanic_type = MechanicType.CONDITIONAL
                description = f"Action {action!r} conditionally causes changes ({change_rate:.0%})"

            # Find or create mechanic
            mechanic = self._find_mechanic(action, mechanic_type)
            if mechanic is None:
                mechanic = Mechanic(
                    mechanic_type=mechanic_type,
                    action=action,
                    description=description,
                )
                self.mechanics.append(mechanic)
                self.action_to_mechanics.setdefault(action, []).append(mechanic)

            # Update tracking fields
            mechanic.observation_count += total
            if current_level not in mechanic.observed_in_levels:
                mechanic.observed_in_levels.append(current_level)

            # EMA consistency update: use change_rate as the signal for
            # MOVEMENT/CONDITIONAL; for NO_EFFECT invert so high = consistent
            if mechanic_type == MechanicType.NO_EFFECT:
                new_signal = 1.0 - change_rate
            else:
                new_signal = change_rate

            if mechanic.consistency_score == 0.0 and mechanic.observation_count == total:
                # First update — seed with raw signal
                mechanic.consistency_score = new_signal
            else:
                mechanic.consistency_score = (
                    _EMA_ALPHA * new_signal + (1.0 - _EMA_ALPHA) * mechanic.consistency_score
                )

    def _find_mechanic(self, action: str, mechanic_type: MechanicType) -> Mechanic | None:
        """Return the first existing :class:`Mechanic` matching *action* and *mechanic_type*."""
        for m in self.action_to_mechanics.get(action, []):
            if m.mechanic_type == mechanic_type:
                return m
        return None

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_reliable_mechanics(self, min_consistency: float = 0.7) -> list[Mechanic]:
        """Return mechanics whose ``consistency_score`` meets *min_consistency*.

        Results are sorted by consistency score descending.

        Args:
            min_consistency: Lower bound (inclusive) for ``consistency_score``.

        Returns:
            Filtered and sorted list of :class:`Mechanic` objects.
        """
        return sorted(
            (m for m in self.mechanics if m.consistency_score >= min_consistency),
            key=lambda m: m.consistency_score,
            reverse=True,
        )

    def get_mechanics_for_action(self, action_str: str) -> list[Mechanic]:
        """Return all mechanics associated with *action_str*.

        Args:
            action_str: The action key to look up.

        Returns:
            List of :class:`Mechanic` objects (empty list if action is unknown).
        """
        return self.action_to_mechanics.get(action_str, [])

    def predict_action_effect(self, action_str: str) -> MechanicType:
        """Predict the most likely effect of *action_str*.

        Selects the :class:`Mechanic` with the highest ``consistency_score`` among
        all mechanics associated with the action.

        Args:
            action_str: The action to predict.

        Returns:
            The :class:`MechanicType` of the best-scoring mechanic, or
            :attr:`~MechanicType.UNKNOWN` if the action has not been observed.
        """
        candidates = self.action_to_mechanics.get(action_str, [])
        if not candidates:
            return MechanicType.UNKNOWN
        best = max(candidates, key=lambda m: m.consistency_score)
        return best.mechanic_type

    # ------------------------------------------------------------------
    # Level snapshots
    # ------------------------------------------------------------------

    def snapshot_level(self, level: int, episode_memory: EpisodeMemory) -> None:
        """Save a snapshot of *episode_memory* statistics for *level*.

        Snapshots are appended to :attr:`_level_snapshots` and can be used for
        cross-level trend analysis.

        Args:
            level: The level index to record.
            episode_memory: The episode memory at the time of snapshotting.
        """
        snapshot: dict = {
            "level": level,
            "total_transitions": len(episode_memory.transitions),
            "visited_states": len(episode_memory.visited_states),
            "action_count": len(episode_memory.action_effect_map),
            "mechanics_known": len(self.mechanics),
        }
        self._level_snapshots.append(snapshot)

    # ------------------------------------------------------------------
    # LLM summary
    # ------------------------------------------------------------------

    def get_summary_for_llm(self) -> str:
        """Return a compact, human-readable mechanics summary for LLM injection.

        Lists up to five of the most reliable mechanics sorted by consistency.

        Returns:
            Multi-line string suitable for embedding in an LLM prompt.
        """
        if not self.mechanics:
            return "No mechanics learned yet."

        reliable = self.get_reliable_mechanics(min_consistency=0.0)[:5]
        lines: list[str] = [f"Known mechanics ({len(self.mechanics)} total, top 5 shown):"]
        for i, m in enumerate(reliable, start=1):
            lines.append(
                f"  {i}. [{m.mechanic_type.value}] {m.action}: {m.description}"
                f" (consistency: {m.consistency_score:.0%},"
                f" observed {m.observation_count}x in levels {m.observed_in_levels})"
            )

        if self._level_snapshots:
            lines.append(f"Level snapshots: {len(self._level_snapshots)}")

        return "\n".join(lines)
