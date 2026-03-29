"""ARC-AGI-3 hypothesis-driven explorer: phase-based action selection strategy."""

from __future__ import annotations

import random
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jarvis.arc.episode_memory import EpisodeMemory
    from jarvis.arc.goal_inference import GoalInferenceModule

__all__ = [
    "ExplorationPhase",
    "HypothesisDrivenExplorer",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DISCOVERY_MAX_STEPS = 50  # switch to HYPOTHESIS after this many discovery steps
_EXPLOITATION_CONFIDENCE_THRESHOLD = 0.6  # min goal confidence for EXPLOITATION
_HYPOTHESIS_BEST_RATIO = 0.8  # 80% chance to pick best action in hypothesis phase

# Strategic positions for complex-action sampling (13 points)
_STRATEGIC_POSITIONS: list[tuple[int, int]] = [
    (0, 0),
    (0, 32),
    (0, 63),
    (32, 0),
    (32, 32),
    (32, 63),
    (63, 0),
    (63, 32),
    (63, 63),
    (16, 16),
    (16, 48),
    (48, 16),
    (48, 48),
]


# ---------------------------------------------------------------------------
# ExplorationPhase enum
# ---------------------------------------------------------------------------


class ExplorationPhase(Enum):
    """Ordered phases of the hypothesis-driven exploration strategy."""

    DISCOVERY = "DISCOVERY"
    HYPOTHESIS = "HYPOTHESIS"
    EXPLOITATION = "EXPLOITATION"


# ---------------------------------------------------------------------------
# HypothesisDrivenExplorer
# ---------------------------------------------------------------------------


class HypothesisDrivenExplorer:
    """Phase-based action explorer for ARC-AGI-3 puzzles.

    Exploration proceeds through three phases:

    * **DISCOVERY** – systematically exercise every available action (including
      complex-action samples at strategic grid positions) to build an initial
      picture of the game mechanics.
    * **HYPOTHESIS** – use accumulated action-effectiveness scores to make
      informed choices, preferring the most effective action 80 % of the time.
    * **EXPLOITATION** – replay action sequences that previously led to a win;
      falls back to hypothesis-style selection when no win history exists.

    The explorer is intentionally decoupled from any concrete ``GameAction``
    type: it treats action objects as opaque, checking for ``is_simple()`` /
    ``is_complex()`` methods defensively and referring to them only by their
    ``.name`` attribute for string-based lookups.
    """

    def __init__(self) -> None:
        self.phase: ExplorationPhase = ExplorationPhase.DISCOVERY
        self.discovery_queue: list[tuple[Any, dict]] = []
        self.action_test_grid: dict[str, list[dict]] = {}
        self._phase_step_count: int = 0
        self._total_actions_tested: int = 0
        # Populated by initialize_discovery(); used for random fallback
        self._available_actions: list[Any] = []
        self._simple_actions: list[Any] = []
        self._complex_actions: list[Any] = []
        # Cycle / stuck detection
        self._recent_state_hashes: list[str] = []
        self._stuck_counter: int = 0
        self._max_recent_states: int = 20

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize_discovery(self, action_space: list[Any]) -> None:
        """Build the discovery queue from the *dynamic* action space.

        Args:
            action_space: List of ``GameAction``-like objects OR raw ints supplied
                by the environment.  The SDK returns ``available_actions`` as ints
                (e.g. ``[1, 2, 3, 4]``); ``env.action_space`` may return
                ``GameAction`` enums.  Both formats are handled.
        """
        # Normalise: convert raw ints to GameAction enums if possible
        normalised = self._normalise_actions(action_space)
        self._available_actions = normalised
        simple: list[Any] = []
        complex_: list[Any] = []

        for action in normalised:
            # Defensive attribute check — not all mock / future SDK versions
            # may expose is_simple / is_complex.
            if hasattr(action, "is_simple") and action.is_simple():
                # Filter out RESET
                if getattr(action, "name", None) != "RESET":
                    simple.append(action)
            elif hasattr(action, "is_complex") and action.is_complex():
                complex_.append(action)
            elif isinstance(action, int):
                # Raw int without GameAction conversion — treat as simple, skip 0 (RESET)
                if action != 0:
                    simple.append(action)

        self._simple_actions = simple
        self._complex_actions = complex_

        # Build queue: simple actions (no data) + complex samples at strategic positions
        queue: list[tuple[Any, dict]] = []
        for action in simple:
            queue.append((action, {}))

        for action in complex_:
            for x, y in _STRATEGIC_POSITIONS:
                queue.append((action, {"x": x, "y": y}))

        self.discovery_queue = queue

        # Initialise action_test_grid entries
        for action in normalised:
            name = getattr(action, "name", str(action))
            if name not in self.action_test_grid:
                self.action_test_grid[name] = []

    @staticmethod
    def _normalise_actions(action_space: list[Any]) -> list[Any]:
        """Convert raw int actions to GameAction enums if possible.

        The ARC SDK returns ``obs.available_actions`` as plain ints (e.g.
        ``[1, 2, 3, 4]``) while ``env.action_space`` may return GameAction
        enums.  This method ensures we always work with enum objects.
        """
        if not action_space:
            return []
        # Already GameAction-like (has .name attribute)?
        if hasattr(action_space[0], "name"):
            return list(action_space)
        # Raw ints — try to convert via arcengine.GameAction
        try:
            from arcengine import GameAction

            return [GameAction(v) for v in action_space]
        except (ImportError, ValueError):
            # Can't convert — return as-is
            return list(action_space)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def choose_action(
        self,
        current_obs: Any,
        episode_memory: EpisodeMemory,
        goal_module: GoalInferenceModule,
    ) -> tuple[Any, dict]:
        """Select the next action according to the current exploration phase.

        Increments :attr:`_phase_step_count` and :attr:`_total_actions_tested`
        on every call, then delegates to the phase-specific method before
        checking for a phase transition.

        Args:
            current_obs: Current environment observation.
            episode_memory: In-session episode memory.
            goal_module: Goal inference module (may have ``current_goals``).

        Returns:
            A ``(action, data)`` tuple where *data* is ``{}`` for simple
            actions and ``{"x": int, "y": int}`` for complex ones.
        """
        self._phase_step_count += 1
        self._total_actions_tested += 1

        # ------------------------------------------------------------------
        # Cycle / stuck detection — must run before phase delegation so that
        # ALL phases benefit from the anti-stuck mechanism.
        # ------------------------------------------------------------------
        if hasattr(current_obs, "raw_grid") and current_obs.raw_grid is not None:
            current_hash = episode_memory.hash_grid(current_obs.raw_grid)
            repeat_count = self._recent_state_hashes.count(current_hash)
            if repeat_count >= 3:
                # Stuck in a cycle — break out with the least-used action
                self._stuck_counter += 1
                result = self._anti_stuck_action(current_obs, episode_memory)
            else:
                result = self._phase_action(current_obs, episode_memory, goal_module)
            self._recent_state_hashes.append(current_hash)
            if len(self._recent_state_hashes) > self._max_recent_states:
                self._recent_state_hashes.pop(0)
        else:
            result = self._phase_action(current_obs, episode_memory, goal_module)

        self._check_phase_transition(episode_memory, goal_module)
        return result

    def _phase_action(
        self,
        current_obs: Any,
        episode_memory: EpisodeMemory,
        goal_module: GoalInferenceModule,
    ) -> tuple[Any, dict]:
        """Delegate to the current phase's action-selection method."""
        if self.phase == ExplorationPhase.DISCOVERY:
            return self._discovery_action(current_obs, episode_memory)
        if self.phase == ExplorationPhase.HYPOTHESIS:
            return self._hypothesis_action(current_obs, episode_memory, goal_module)
        return self._exploitation_action(current_obs, episode_memory, goal_module)

    # ------------------------------------------------------------------
    # Phase-specific action selection
    # ------------------------------------------------------------------

    def _discovery_action(self, obs: Any, memory: EpisodeMemory) -> tuple[Any, dict]:
        """Pop from the discovery queue, preferring unexplored actions in the current state."""
        if not self.discovery_queue:
            return self._random_action()

        # Try to find an action not yet tried from the current state
        if hasattr(obs, "raw_grid") and obs.raw_grid is not None:
            current_hash = memory.hash_grid(obs.raw_grid)
            tried = memory._state_action_index.get(current_hash, set())
            for i, (action, data) in enumerate(self.discovery_queue):
                name = getattr(action, "name", str(action))
                action_key = name if not data else f"{name}_{data.get('x', '')}_{data.get('y', '')}"
                if action_key not in tried:
                    return self.discovery_queue.pop(i)

        # Default: pop from front
        return self.discovery_queue.pop(0)

    def _hypothesis_action(
        self, obs: Any, memory: EpisodeMemory, goals: GoalInferenceModule
    ) -> tuple[Any, dict]:
        """Choose action using a composite novelty+effectiveness score.

        Scoring formula:
            score = (change_rate * 0.3) + (novelty * 0.5) - (danger * 0.8)

        where:
        * ``change_rate`` — fraction of uses that caused any pixel change
        * ``novelty``     — fraction of uses that produced a distinct next state
        * ``danger``      — fraction of uses that caused game-over
        """
        if not self._available_actions:
            return self._random_action()

        scored: list[tuple[float, Any, dict]] = []

        for action in self._simple_actions:
            name = getattr(action, "name", str(action))
            effects = memory.action_effect_map.get(name, {})
            total = effects.get("total", 0) if isinstance(effects, dict) else 0
            change_rate = effects.get("caused_change", 0) / total if total > 0 else 0.5
            novelty = memory.get_action_novelty(name)
            danger = effects.get("caused_game_over", 0) / total if total > 0 else 0.0
            score = (change_rate * 0.3) + (novelty * 0.5) - (danger * 0.8)
            scored.append((score, action, {}))

        # Also score complex actions with the middle strategic position
        for action in self._complex_actions:
            name = getattr(action, "name", str(action))
            mid_x, mid_y = 32, 32
            action_key = f"{name}_{mid_x}_{mid_y}"
            effects = memory.action_effect_map.get(action_key, {})
            total = effects.get("total", 0) if isinstance(effects, dict) else 0
            change_rate = effects.get("caused_change", 0) / total if total > 0 else 0.5
            novelty = memory.get_action_novelty(action_key)
            danger = effects.get("caused_game_over", 0) / total if total > 0 else 0.0
            score = (change_rate * 0.3) + (novelty * 0.5) - (danger * 0.8)
            scored.append((score, action, {"x": mid_x, "y": mid_y}))

        if not scored:
            return self._random_action()

        scored.sort(key=lambda t: t[0], reverse=True)

        if len(scored) == 1 or random.random() < _HYPOTHESIS_BEST_RATIO:
            _, action, data = scored[0]
        else:
            # Runner-up
            _, action, data = scored[1]

        return action, data

    def _exploitation_action(
        self, obs: Any, memory: EpisodeMemory, goals: GoalInferenceModule
    ) -> tuple[Any, dict]:
        """Replay winning action sequences; falls back to hypothesis when none exist."""
        # Look for transitions that resulted in a win
        win_transitions = [t for t in memory.transitions if t.resulted_in_win]
        if win_transitions:
            # Pick the most recent winning action
            latest_win = win_transitions[-1]
            action_str = latest_win.action
            return self._parse_action_str(action_str)

        # No wins recorded — fall back to hypothesis-style selection
        return self._hypothesis_action(obs, memory, goals)

    # ------------------------------------------------------------------
    # Phase transitions
    # ------------------------------------------------------------------

    def _check_phase_transition(self, memory: EpisodeMemory, goals: GoalInferenceModule) -> None:
        """Evaluate and apply phase transitions where conditions are met."""
        if self.phase == ExplorationPhase.DISCOVERY:
            empty_queue = len(self.discovery_queue) == 0
            exceeded_steps = self._phase_step_count > _DISCOVERY_MAX_STEPS
            if empty_queue or exceeded_steps:
                self.phase = ExplorationPhase.HYPOTHESIS
                self._phase_step_count = 0

        elif self.phase == ExplorationPhase.HYPOTHESIS:
            best_confidence = 0.0
            for goal in getattr(goals, "current_goals", []):
                c = getattr(goal, "confidence", 0.0)
                if c > best_confidence:
                    best_confidence = c
            if best_confidence > _EXPLOITATION_CONFIDENCE_THRESHOLD:
                self.phase = ExplorationPhase.EXPLOITATION
                self._phase_step_count = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _anti_stuck_action(self, obs: Any, memory: EpisodeMemory) -> tuple[Any, dict]:
        """When stuck in a cycle, pick the least-tried action to break out.

        Iterates over ``_simple_actions`` and selects the one with the fewest
        recorded uses.  Falls back to :meth:`_random_action` when no simple
        actions are available.
        """
        action_counts: dict[Any, int] = {}
        for action in self._simple_actions:
            name = getattr(action, "name", str(action))
            effects = memory.action_effect_map.get(name, {})
            action_counts[action] = effects.get("total", 0) if isinstance(effects, dict) else 0

        if action_counts:
            least_used = min(action_counts, key=action_counts.get)
            return least_used, {}
        return self._random_action()

    def _random_action(self) -> tuple[Any, dict]:
        """Return a uniformly random action from the stored available actions.

        Falls back to the first available action when the list is empty
        (should not happen in normal usage but guards against edge cases in
        tests).
        """
        if not self._available_actions:
            raise RuntimeError("No available actions — call initialize_discovery() first.")
        action = random.choice(self._available_actions)
        if hasattr(action, "is_complex") and action.is_complex():
            x = random.randint(0, 63)
            y = random.randint(0, 63)
            return action, {"x": x, "y": y}
        return action, {}

    def _parse_action_str(self, action_str: str) -> tuple[Any, dict]:
        """Parse a string such as ``"ACTION6_32_15"`` back to ``(action, data)``.

        Performs a defensive lookup against the stored available actions by
        matching the ``.name`` attribute.  Falls back to :meth:`_random_action`
        if the action name is not found.

        Args:
            action_str: Encoded action string.  Simple actions use their bare
                name (e.g. ``"ACTION1"``); complex ones append ``_x_y``
                coordinates (e.g. ``"ACTION6_32_15"``).

        Returns:
            ``(action, data)`` tuple.
        """
        parts = action_str.split("_")
        # Heuristic: if the last two parts are digits the string encodes coords
        data: dict[str, int] = {}
        if len(parts) >= 3:
            try:
                y = int(parts[-1])
                x = int(parts[-2])
                action_name = "_".join(parts[:-2])
                data = {"x": x, "y": y}
            except ValueError:
                action_name = action_str
        else:
            action_name = action_str

        # Look up against stored actions
        for action in self._available_actions:
            if getattr(action, "name", None) == action_name:
                return action, data

        # Name not found — fall back to random
        return self._random_action()
