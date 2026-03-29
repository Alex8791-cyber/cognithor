"""CognithorArcAgent — main orchestration for ARC-AGI-3 game sessions."""

from __future__ import annotations

from typing import Any

from jarvis.arc.adapter import ArcEnvironmentAdapter, ArcObservation
from jarvis.arc.audit import ArcAuditTrail
from jarvis.arc.episode_memory import EpisodeMemory
from jarvis.arc.explorer import ExplorationPhase, HypothesisDrivenExplorer
from jarvis.arc.goal_inference import GoalInferenceModule
from jarvis.arc.mechanics_model import MechanicsModel
from jarvis.arc.visual_encoder import VisualStateEncoder
from jarvis.utils.logging import get_logger

__all__ = ["CognithorArcAgent"]

log = get_logger(__name__)

# Number of steps between goal re-analysis
_GOAL_REANALYSIS_INTERVAL = 5


class CognithorArcAgent:
    """Hybrid ARC-AGI-3 Agent.

    Fast Path: Explorer + Memory (algorithmic, >2000 FPS)
    Strategic Path: LLM Planner every N steps (optional)

    Args:
        game_id: The ARC-AGI-3 environment identifier.
        use_llm_planner: Whether to consult the LLM planner periodically.
        llm_call_interval: Number of steps between LLM planner consultations.
        max_steps_per_level: Maximum steps before a level is abandoned.
        max_resets_per_level: Maximum game-over resets before giving up on a level.
    """

    def __init__(
        self,
        game_id: str,
        use_llm_planner: bool = True,
        llm_call_interval: int = 10,
        max_steps_per_level: int = 500,
        max_resets_per_level: int = 5,
    ) -> None:
        self.game_id = game_id
        self.use_llm_planner = use_llm_planner
        self.llm_call_interval = llm_call_interval
        self.max_steps_per_level = max_steps_per_level
        self.max_resets_per_level = max_resets_per_level

        # Initialise all subsystem modules
        self.adapter = ArcEnvironmentAdapter(game_id)
        self.memory = EpisodeMemory()
        self.goals = GoalInferenceModule()
        self.explorer = HypothesisDrivenExplorer()
        self.encoder = VisualStateEncoder()
        self.mechanics = MechanicsModel()
        self.audit_trail = ArcAuditTrail(game_id)

        # Runtime state
        self.current_obs: ArcObservation | None = None
        self.current_level: int = 0
        self.level_resets: int = 0
        self.total_steps: int = 0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run the full agent loop until the game ends.

        Steps through levels, handling WIN (advance) and GAME_OVER (reset) until
        the game is finished or max steps are exhausted.

        Returns:
            A scorecard dict with keys ``game_id``, ``levels_completed``,
            ``total_steps``, ``total_resets``, and ``score``.
        """
        log.info("arc.agent.run.start", game_id=self.game_id)
        self.audit_trail.log_game_start()

        # Phase 0: initialise the environment
        self.current_obs = self.adapter.initialize()
        self.explorer.initialize_discovery(self.adapter.env.action_space)

        # Main agent loop
        while True:
            result = self._step()

            if result == "WIN":
                self._on_level_complete()
                # Check whether the whole game is over after the level transition
                state_str = str(self.current_obs.game_state) if self.current_obs else ""
                if "GAME_OVER" in state_str or "WIN" in state_str:
                    # The scorecard from the SDK will tell us if we truly finished
                    break

            elif result == "GAME_OVER":
                if self.level_resets >= self.max_resets_per_level:
                    log.warning(
                        "arc.agent.max_resets_reached",
                        level=self.current_level,
                        resets=self.level_resets,
                    )
                    break
                # Reset the level
                self.current_obs = self.adapter.reset_level()
                self.level_resets += 1
                self.memory.clear_for_new_level()
                log.info(
                    "arc.agent.level_reset",
                    level=self.current_level,
                    resets=self.level_resets,
                )

            elif result == "DONE":
                break

        # Retrieve final scorecard
        try:
            scorecard = self.adapter.get_scorecard()
            final_score = float(getattr(scorecard, "score", 0.0))
        except Exception:
            final_score = 0.0
            scorecard = None

        self.audit_trail.log_game_end(final_score)

        result_dict: dict[str, Any] = {
            "game_id": self.game_id,
            "levels_completed": self.current_level,
            "total_steps": self.total_steps,
            "total_resets": self.adapter.total_resets,
            "score": final_score,
        }
        log.info("arc.agent.run.done", **result_dict)
        return result_dict

    # ------------------------------------------------------------------
    # Single step
    # ------------------------------------------------------------------

    def _step(self) -> str:
        """Execute one agent step.

        Returns:
            ``"WIN"`` if the level was completed, ``"GAME_OVER"`` if a reset is
            required, ``"DONE"`` if the max-steps budget is exhausted, or
            ``"CONTINUE"`` otherwise.
        """
        # Guard: max steps per level
        if self.adapter.level_step_count >= self.max_steps_per_level:
            log.warning(
                "arc.agent.max_steps_per_level",
                level=self.current_level,
                steps=self.adapter.level_step_count,
            )
            return "DONE"

        # Choose action via explorer
        action, data = self.explorer.choose_action(
            self.current_obs,
            self.memory,
            self.goals,
        )

        # Optional LLM planner consultation
        if self.use_llm_planner and (self.total_steps % self.llm_call_interval == 0):
            action, data = self._consult_llm_planner(action, data)

        # Execute action
        action_str = self._action_to_str(action, data)
        previous_obs = self.current_obs
        self.current_obs = self.adapter.act(action, data)
        self.total_steps += 1

        # Record transition in episode memory
        self.memory.record_transition(previous_obs, action_str, self.current_obs)

        # Audit step
        self.audit_trail.log_step(
            level=self.current_level,
            step=self.total_steps,
            action=action_str,
            game_state=str(self.current_obs.game_state),
            pixels_changed=self.current_obs.changed_pixels,
        )

        # Periodic goal re-analysis
        if self.total_steps % _GOAL_REANALYSIS_INTERVAL == 0:
            self.goals.analyze_win_condition(self.memory)

        # Evaluate terminal game state
        state_str = str(self.current_obs.game_state)
        if "WIN" in state_str:
            return "WIN"
        elif "GAME_OVER" in state_str:
            return "GAME_OVER"
        return "CONTINUE"

    # ------------------------------------------------------------------
    # Level completion
    # ------------------------------------------------------------------

    def _on_level_complete(self) -> None:
        """Handle post-WIN level bookkeeping and prepare for the next level."""
        log.info(
            "arc.agent.level_complete",
            level=self.current_level,
            steps=self.adapter.level_step_count,
            resets=self.level_resets,
        )

        # Distil knowledge from this level's episode
        self.mechanics.analyze_transitions(self.memory, self.current_level)
        self.mechanics.snapshot_level(self.current_level, self.memory)
        self.goals.on_level_complete(
            {
                "level": self.current_level,
                "steps": self.adapter.level_step_count,
                "resets": self.level_resets,
            }
        )

        # Advance level counters
        self.current_level += 1
        self.level_resets = 0

        # Reset per-level state
        self.memory.clear_for_new_level()

        # Restart discovery phase for the new level
        self.explorer.phase = ExplorationPhase.DISCOVERY
        available_actions = (
            self.current_obs.available_actions
            if self.current_obs and self.current_obs.available_actions
            else self.adapter.env.action_space
        )
        self.explorer.initialize_discovery(available_actions)

    # ------------------------------------------------------------------
    # LLM planner stub
    # ------------------------------------------------------------------

    def _consult_llm_planner(
        self,
        default_action: Any,
        default_data: dict[str, Any],
    ) -> tuple[Any, dict[str, Any]]:
        """Stub for LLM planner integration (real PGE wiring is a later task).

        Logs that the planner would be called and returns the defaults unchanged.

        Args:
            default_action: The action pre-selected by the explorer.
            default_data: The data dict pre-selected by the explorer.

        Returns:
            The unchanged ``(default_action, default_data)`` tuple.
        """
        log.debug(
            "arc.agent.llm_planner.stub",
            step=self.total_steps,
            action=self._action_to_str(default_action, default_data),
            note="LLM planner not yet wired — returning explorer default",
        )
        return default_action, default_data

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _action_to_str(action: Any, data: dict[str, Any]) -> str:
        """Encode an action + data payload as a canonical string.

        Simple actions (no coordinates) become ``"ACTION1"``.
        Complex actions with x/y coordinates become ``"ACTION6_32_15"``.

        Args:
            action: A ``GameAction``-like object with a ``.name`` attribute, or
                any object whose ``str()`` representation is usable.
            data: Optional payload dict; ``{"x": int, "y": int}`` appended as
                underscore-separated suffixes when both keys are present.

        Returns:
            A canonical string representation of the action.
        """
        name: str = getattr(action, "name", None) or str(action)
        if data and "x" in data and "y" in data:
            return f"{name}_{data['x']}_{data['y']}"
        return name
