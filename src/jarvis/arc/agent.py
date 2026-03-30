"""CognithorArcAgent — main orchestration for ARC-AGI-3 game sessions."""

from __future__ import annotations

from typing import Any

from jarvis.arc.adapter import ArcEnvironmentAdapter, ArcObservation
from jarvis.arc.audit import ArcAuditTrail
from jarvis.arc.episode_memory import EpisodeMemory
from jarvis.arc.explorer import ExplorationPhase, HypothesisDrivenExplorer
from jarvis.arc.goal_inference import GoalInferenceModule
from jarvis.arc.mechanics_model import MechanicsModel
from jarvis.arc.state_graph import StateGraphNavigator
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
        use_llm_planner: bool = False,
        llm_call_interval: int = 30,
        max_steps_per_level: int = 500,
        max_resets_per_level: int = 20,
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
        self.state_graph = StateGraphNavigator(max_states=200_000)
        self._navigation_mode = False
        self._current_path: list = []
        self._path_index: int = 0

        # CNN Action Predictor (online learning during gameplay)
        self._cnn_trainer: Any = None
        try:
            from jarvis.arc.cnn_model import OnlineTrainer, _TORCH_AVAILABLE

            if _TORCH_AVAILABLE:
                self._cnn_trainer = OnlineTrainer(device="cuda")
                log.info("arc.agent.cnn_initialized", device=str(self._cnn_trainer._device))
        except Exception:
            log.debug("arc.agent.cnn_not_available", exc_info=True)

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
        """One agent step with State Graph Navigation."""
        if self.adapter.level_step_count >= self.max_steps_per_level:
            return "DONE"

        current_hash = self.state_graph.hash_grid(self.current_obs.raw_grid)

        # === NAVIGATION MODE: Win path known → follow it ===
        if (
            self._navigation_mode
            and self._current_path
            and self._path_index < len(self._current_path)
        ):
            action_str, action_data, expected_next = self._current_path[self._path_index]
            action = self._resolve_action(action_str)
            data = action_data or {}

            previous_obs = self.current_obs
            self.current_obs = self.adapter.act(action, data)
            self._path_index += 1
            self.total_steps += 1

            self._record_step(previous_obs, action_str, data)

            # Validate: are we still on the expected path?
            actual_hash = self.state_graph.hash_grid(self.current_obs.raw_grid)
            if actual_hash != expected_next:
                log.info("arc.agent.path_diverged", step=self.total_steps)
                self._navigation_mode = False
                self._current_path = []

            return self._check_game_state()

        # Navigation finished or invalid
        self._navigation_mode = False
        self._current_path = []

        # === Check if a win path is now available ===
        win_path = self.state_graph.find_win_path(current_hash)
        if win_path:
            log.info("arc.agent.win_path_found", length=len(win_path), step=self.total_steps)
            self._navigation_mode = True
            self._current_path = win_path
            self._path_index = 0
            return self._step()  # Immediately start navigating

        # === EXPLORATION MODE: Build the graph ===
        available_actions = [
            a for a in self.explorer._available_actions if getattr(a, "name", "") != "RESET"
        ]
        available_names = [getattr(a, "name", str(a)) for a in available_actions]

        # Decision priority: CNN prediction → Graph exploration → Explorer fallback
        action_str: str | None = None
        action: Any = None
        data: dict[str, Any] = {}

        # 1. CNN-guided action selection (after enough training data)
        if self._cnn_trainer is not None and self.adapter.level_step_count > 20:
            try:
                action_probs, coord_probs = self._cnn_trainer.predict(self.current_obs.raw_grid)
                # Mask unavailable actions: only score available ones
                import numpy as np

                masked = np.full_like(action_probs, -1.0)
                for a in available_actions:
                    idx = getattr(a, "value", 0)
                    if 0 <= idx < len(action_probs):
                        masked[idx] = action_probs[idx]

                best_idx = int(np.argmax(masked))
                # Only use CNN if confidence is meaningful (> 0.3)
                if masked[best_idx] > 0.3:
                    action = self._resolve_action_by_value(best_idx)
                    if action is not None:
                        action_str = getattr(action, "name", str(action))
                        # For complex actions, use coord_probs heat-map
                        if hasattr(action, "is_complex") and action.is_complex():
                            best_pos = np.unravel_index(np.argmax(coord_probs), (64, 64))
                            data = {"x": int(best_pos[1]), "y": int(best_pos[0])}
            except Exception:
                log.debug("arc.agent.cnn_predict_failed", exc_info=True)

        # 2. Graph-guided exploration (untested actions first)
        if action_str is None:
            graph_action = self.state_graph.get_best_exploration_action(
                current_hash, available_names
            )
            if graph_action:
                action_str, action_data = graph_action
                action = self._resolve_action(action_str)
                data = action_data or {}
                if hasattr(action, "is_complex") and action.is_complex() and "x" not in data:
                    import random

                    data = {"x": random.randint(0, 63), "y": random.randint(0, 63)}

        # 3. Explorer fallback
        if action_str is None:
            action, data = self.explorer.choose_action(self.current_obs, self.memory, self.goals)
            action_str = self._action_to_str(action, data)

        # LLM only rarely (disabled by default)
        if (
            self.use_llm_planner
            and self.adapter.level_step_count > 50
            and self.total_steps % self.llm_call_interval == 0
            and not self.state_graph.should_navigate()
        ):
            action, data = self._consult_llm_planner(action, data)
            action_str = self._action_to_str(action, data)

        # Execute action
        previous_obs = self.current_obs
        self.current_obs = self.adapter.act(action, data)
        self.total_steps += 1

        # Record in memory + audit + graph
        self._record_step(previous_obs, action_str, data)

        # Feed CNN trainer with experience (online learning)
        if self._cnn_trainer is not None and action is not None:
            try:
                action_idx = getattr(action, "value", 0)
                frame_changed = self.current_obs.changed_pixels > 0
                coord = (data.get("y"), data.get("x")) if data.get("x") is not None else None
                self._cnn_trainer.add_experience(
                    previous_obs.raw_grid, action_idx, coord, frame_changed
                )
            except Exception:
                log.debug("arc.agent.cnn_train_failed", exc_info=True)

        # Periodic goal analysis
        if self.total_steps % _GOAL_REANALYSIS_INTERVAL == 0:
            self.goals.analyze_win_condition(self.memory)

        return self._check_game_state()

    # ------------------------------------------------------------------
    # Step helpers
    # ------------------------------------------------------------------

    def _record_step(self, previous_obs: Any, action_str: str, data: dict[str, Any]) -> None:
        """Record transition in memory, audit trail, AND state graph."""
        full_action = (
            self._action_to_str(self._resolve_action(action_str), data) if data else action_str
        )

        self.memory.record_transition(previous_obs, full_action, self.current_obs)
        self.audit_trail.log_step(
            level=self.current_level,
            step=self.total_steps,
            action=full_action,
            game_state=str(self.current_obs.game_state),
            pixels_changed=self.current_obs.changed_pixels,
        )
        self.state_graph.add_transition(
            from_grid=previous_obs.raw_grid,
            action_str=action_str,
            action_data=data if data else None,
            to_grid=self.current_obs.raw_grid,
            pixels_changed=self.current_obs.changed_pixels,
            game_state=str(self.current_obs.game_state),
            level=self.current_level,
        )

    def _check_game_state(self) -> str:
        """Evaluate terminal game state from current observation."""
        state_str = str(self.current_obs.game_state)
        if "WIN" in state_str:
            return "WIN"
        if "GAME_OVER" in state_str:
            return "GAME_OVER"
        return "CONTINUE"

    def _resolve_action(self, action_str: str) -> Any:
        """Convert action name string to a GameAction enum value."""
        for a in self.explorer._available_actions:
            if getattr(a, "name", str(a)) == action_str:
                return a
        # Fallback: try arcengine
        try:
            from arcengine import GameAction

            return GameAction[action_str]
        except (ImportError, KeyError):
            pass
        # Last resort: return first available action
        if self.explorer._available_actions:
            return self.explorer._available_actions[0]
        return action_str

    def _resolve_action_by_value(self, value: int) -> Any:
        """Convert an integer action value to a GameAction enum."""
        for a in self.explorer._available_actions:
            if getattr(a, "value", -1) == value:
                return a
        try:
            from arcengine import GameAction

            return GameAction(value)
        except (ImportError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Level completion
    # ------------------------------------------------------------------

    def _on_level_complete(self) -> None:
        """Handle post-WIN level bookkeeping and prepare for the next level."""
        # State Graph: preserve action patterns, clear state space
        self.state_graph.prepare_for_new_level()
        self._navigation_mode = False
        self._current_path = []
        self._path_index = 0

        # CNN: reset model for new level mechanics
        if self._cnn_trainer is not None:
            self._cnn_trainer.reset_for_new_level()

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
        """Consult the LLM for strategic guidance.

        Builds a compact prompt from the current state, episode memory,
        and goal hypotheses.  Asks the LLM which action to take next.
        Falls back to *default_action* on any failure.

        Args:
            default_action: The action pre-selected by the explorer.
            default_data: The data dict pre-selected by the explorer.

        Returns:
            ``(action, data)`` — either the LLM's recommendation or the default.
        """
        try:
            state_desc = self.encoder.encode_for_llm(
                self.current_obs.raw_grid,
                self.current_obs.grid_diff,
            )
            memory_summary = self.memory.get_summary_for_llm()
            goal_summary = self.goals.get_summary_for_llm()
            mechanics_summary = self.mechanics.get_summary_for_llm()
            graph_summary = self.state_graph.get_summary_for_llm()

            # Build available action names
            action_names = [
                getattr(a, "name", str(a))
                for a in self.explorer._available_actions
                if getattr(a, "name", "") != "RESET"
            ]

            # Per-action effectiveness for the prompt
            action_stats = []
            for aname in action_names:
                eff = self.memory.get_action_effectiveness(aname)
                nov = self.memory.get_action_novelty(aname)
                effects = self.memory.action_effect_map.get(aname, {})
                total = effects.get("total", 0) if isinstance(effects, dict) else 0
                wins = effects.get("caused_win", 0) if isinstance(effects, dict) else 0
                gos = effects.get("caused_game_over", 0) if isinstance(effects, dict) else 0
                action_stats.append(
                    f"  {aname}: {eff:.0%} change, {nov:.0%} novelty, "
                    f"{total} uses, {wins} wins, {gos} game_overs"
                )
            action_block = "\n".join(action_stats)

            prompt = (
                "You are playing an ARC-AGI-3 puzzle game with NO instructions. "
                "You must discover the rules by trying different actions and observing "
                "what changes. Your goal is to reach a WIN state efficiently.\n\n"
                "CRITICAL: Do NOT always pick the same action. Vary your choices to "
                "explore the game mechanics. Pick the action most likely to lead to "
                "NEW, UNEXPLORED states.\n\n"
                f"AVAILABLE ACTIONS + STATS:\n{action_block}\n\n"
                f"GRID STATE:\n{state_desc}\n\n"
                f"STATE GRAPH:\n{graph_summary}\n\n"
                f"MEMORY:\n{memory_summary}\n\n"
                f"GOALS:\n{goal_summary}\n\n"
                f"Level: {self.current_level}, Step: {self.adapter.level_step_count}, "
                f"Resets: {self.level_resets}\n\n"
                "Pick the SINGLE BEST action to try next. Reply with ONLY the action "
                "name. Nothing else."
            )

            # Call Ollama directly via httpx (lightweight, no full PGE overhead)
            import httpx

            ollama_url = "http://localhost:11434/api/generate"
            resp = httpx.post(
                ollama_url,
                json={
                    "model": "qwen3.5:27b-16k",
                    "prompt": prompt + "\n/no_think",
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 2000,
                        "num_ctx": 16384,
                    },
                },
                timeout=120.0,
            )
            if resp.status_code == 200:
                text = resp.json().get("response", "").strip()
                # Parse action from response
                for a in self.explorer._available_actions:
                    name = getattr(a, "name", str(a))
                    if name in text and name != "RESET":
                        log.info(
                            "arc.agent.llm_planner.recommendation",
                            step=self.total_steps,
                            recommended=name,
                            raw=text[:50],
                        )
                        return a, {}
                log.debug(
                    "arc.agent.llm_planner.unparseable",
                    step=self.total_steps,
                    raw=text[:100],
                )
        except Exception as exc:
            log.debug("arc.agent.llm_planner.failed", error=str(exc)[:80])

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
