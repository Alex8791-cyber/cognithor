"""ARC-AGI-3 in-session episode memory: volatile short-term memory for a single game run."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "EpisodeMemory",
    "Hypothesis",
    "StateTransition",
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class StateTransition:
    """Records a single state→action→state transition within a game run."""

    state_hash: str
    action: str
    next_state_hash: str
    pixels_changed: int
    resulted_in_win: bool = False
    resulted_in_game_over: bool = False
    level: int = 0


@dataclass
class Hypothesis:
    """A testable hypothesis about game mechanics, tracked with evidence counters."""

    description: str
    supporting_evidence: int = 0
    contradicting_evidence: int = 0
    confidence: float = 0.0
    tested_at_steps: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper — action effect entry factory
# ---------------------------------------------------------------------------


def _new_effect_entry() -> dict[str, int]:
    return {"total": 0, "caused_change": 0, "caused_win": 0, "caused_game_over": 0}


# ---------------------------------------------------------------------------
# EpisodeMemory
# ---------------------------------------------------------------------------


class EpisodeMemory:
    """Volatile, fast, in-session short-term memory for a single ARC-AGI-3 game.

    Lives only during one game run; never persisted to disk.  Tracks state
    transitions, action effectiveness, visited states, and hypotheses about
    the puzzle mechanics.

    Args:
        max_transitions: Hard cap on the number of stored transitions to
            prevent unbounded memory growth in very long runs.
    """

    def __init__(self, max_transitions: int = 200_000) -> None:
        self.max_transitions = max_transitions
        self.transitions: list[StateTransition] = []
        self.state_visit_count: defaultdict[str, int] = defaultdict(int)
        self.action_effect_map: defaultdict[str, dict[str, int]] = defaultdict(_new_effect_entry)
        self.hypotheses: list[Hypothesis] = []
        self.visited_states: set[str] = set()
        self._state_hash_cache: dict[tuple, str] = {}
        self._state_action_index: defaultdict[str, set[str]] = defaultdict(set)
        # Index of next-state hashes per action for fast novelty computation
        self._action_next_states: defaultdict[str, set[str]] = defaultdict(set)

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------

    def hash_grid(self, grid: np.ndarray) -> str:
        """Return a 16-character hex MD5 hash of *grid*, cached by content.

        The cache key includes shape and dtype so that arrays with identical
        raw bytes but different shapes or dtypes do not collide.
        """
        raw = grid.tobytes()
        key = (grid.shape, grid.dtype.str, raw)
        cached = self._state_hash_cache.get(key)
        if cached is not None:
            return cached
        shape_prefix = np.array(grid.shape, dtype=np.int64).tobytes()
        digest = hashlib.md5(shape_prefix + raw, usedforsecurity=False).hexdigest()[:16]
        self._state_hash_cache[key] = digest
        return digest

    # ------------------------------------------------------------------
    # Transition recording
    # ------------------------------------------------------------------

    @staticmethod
    def _is_win(obs: Any) -> bool:
        """Defensively check whether *obs* represents a WIN game state."""
        gs = getattr(obs, "game_state", None)
        if gs is None:
            return False
        # String-based observation (mock tests) or enum whose repr includes WIN
        gs_str = str(gs)
        return gs_str in ("WIN", "GameState.WIN")

    @staticmethod
    def _is_game_over(obs: Any) -> bool:
        """Defensively check whether *obs* represents a GAME_OVER game state."""
        gs = getattr(obs, "game_state", None)
        if gs is None:
            return False
        gs_str = str(gs)
        return gs_str in ("GAME_OVER", "GameState.GAME_OVER")

    def record_transition(
        self,
        obs_before: Any,
        action_str: str,
        obs_after: Any,
    ) -> StateTransition:
        """Record a state→action→state transition.

        Hashes both observations' grids, updates action effect counters, and
        registers both states in *visited_states*.  Does nothing beyond
        returning the transition once *max_transitions* is reached.

        Args:
            obs_before: Observation before the action was taken. Must have
                ``raw_grid`` attribute (np.ndarray).
            action_str: The action taken (e.g. ``"ACTION1"``).
            obs_after: Observation after the action. Must have ``raw_grid``
                and optionally ``game_state``, ``changed_pixels``, ``level``.

        Returns:
            The recorded :class:`StateTransition`.
        """
        before_hash = self.hash_grid(obs_before.raw_grid)
        after_hash = self.hash_grid(obs_after.raw_grid)

        won = self._is_win(obs_after)
        game_over = self._is_game_over(obs_after)
        pixels_changed = getattr(obs_after, "changed_pixels", 0) or 0
        level = getattr(obs_after, "level", 0) or 0

        transition = StateTransition(
            state_hash=before_hash,
            action=action_str,
            next_state_hash=after_hash,
            pixels_changed=pixels_changed,
            resulted_in_win=won,
            resulted_in_game_over=game_over,
            level=level,
        )

        # Update action effect statistics
        entry = self.action_effect_map[action_str]
        entry["total"] += 1
        if pixels_changed > 0:
            entry["caused_change"] += 1
        if won:
            entry["caused_win"] += 1
        if game_over:
            entry["caused_game_over"] += 1

        # Track visit counts
        self.state_visit_count[before_hash] += 1
        self.visited_states.add(before_hash)
        self.visited_states.add(after_hash)

        # Update secondary index for O(1) unexplored-action lookup
        self._state_action_index[before_hash].add(action_str)

        # Update next-state index for novelty computation
        self._action_next_states[action_str].add(after_hash)

        # Respect cap
        if len(self.transitions) < self.max_transitions:
            self.transitions.append(transition)

        return transition

    # ------------------------------------------------------------------
    # Action effectiveness
    # ------------------------------------------------------------------

    def get_action_effectiveness(self, action_str: str) -> float:
        """Return the fraction of times *action_str* caused a state change.

        Returns ``0.5`` (neutral prior) for actions that have never been tried.
        """
        entry = self.action_effect_map.get(action_str)
        if entry is None or entry["total"] == 0:
            return 0.5
        return entry["caused_change"] / entry["total"]

    def get_action_novelty(self, action_str: str) -> float:
        """Return how often *action_str* produces a previously unseen next state.

        Uses the ``_action_next_states`` index for O(1) unique-state count and
        ``action_effect_map`` for the total use count.

        Returns ``0.5`` (neutral prior) for actions that have never been tried.
        """
        entry = self.action_effect_map.get(action_str)
        total = entry["total"] if entry is not None else 0
        if total == 0:
            return 0.5
        unique_next = len(self._action_next_states.get(action_str, set()))
        return unique_next / total

    # ------------------------------------------------------------------
    # Exploration helpers
    # ------------------------------------------------------------------

    def get_unexplored_actions(
        self,
        current_state_hash: str,
        all_actions: list[str],
    ) -> list[str]:
        """Return actions from *all_actions* not yet tried from *current_state_hash*."""
        tried = self._state_action_index.get(current_state_hash, set())
        return [a for a in all_actions if a not in tried]

    def is_novel_state(self, grid: np.ndarray) -> bool:
        """Return ``True`` if *grid* has not been seen in this episode."""
        return self.hash_grid(grid) not in self.visited_states

    # ------------------------------------------------------------------
    # Hypotheses
    # ------------------------------------------------------------------

    def add_hypothesis(self, description: str) -> Hypothesis:
        """Create and store a new hypothesis about puzzle mechanics."""
        h = Hypothesis(description=description)
        self.hypotheses.append(h)
        return h

    # ------------------------------------------------------------------
    # Level management
    # ------------------------------------------------------------------

    def clear_for_new_level(self) -> None:
        """Reset per-level state while preserving cross-level learning.

        Clears:
        - ``state_visit_count`` — per-level visit counters
        - ``visited_states`` — so novel-state detection resets

        Keeps:
        - ``transitions`` — full history for replay / debugging
        - ``action_effect_map`` — accumulated effectiveness data
        - ``hypotheses`` — carry forward between levels
        """
        self.state_visit_count.clear()
        self.visited_states.clear()
        self._state_hash_cache.clear()

    # ------------------------------------------------------------------
    # LLM summary
    # ------------------------------------------------------------------

    def get_summary_for_llm(self) -> str:
        """Return a compact, human-readable summary for injection into LLM context."""
        lines: list[str] = [
            f"Besuchte Zustaende: {len(self.visited_states)}",
            f"Transitions: {len(self.transitions)}",
            f"Hypothesen: {len(self.hypotheses)}",
        ]

        # Top-5 actions by effectiveness
        if self.action_effect_map:
            lines.append("Aktions-Effektivitaet (top 5):")
            sorted_actions = sorted(
                self.action_effect_map.items(),
                key=lambda kv: kv[1]["caused_change"] / max(kv[1]["total"], 1),
                reverse=True,
            )
            for action, stats in sorted_actions[:5]:
                eff = stats["caused_change"] / max(stats["total"], 1)
                lines.append(
                    f"  {action}: {eff:.0%} wirksam"
                    f" ({stats['total']} Versuche,"
                    f" {stats['caused_win']} Gewinne,"
                    f" {stats['caused_game_over']} Verluste)"
                )

        # Active hypotheses (all, capped at 10)
        if self.hypotheses:
            lines.append("Aktive Hypothesen:")
            for h in self.hypotheses[:10]:
                lines.append(
                    f"  [{h.confidence:.0%}] {h.description}"
                    f" (+{h.supporting_evidence} / -{h.contradicting_evidence})"
                )

        return "\n".join(lines)
