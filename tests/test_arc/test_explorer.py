"""Tests for HypothesisDrivenExplorer (Task 6)."""

from __future__ import annotations

from unittest.mock import MagicMock
from jarvis.arc.explorer import HypothesisDrivenExplorer, ExplorationPhase
from jarvis.arc.episode_memory import EpisodeMemory
from jarvis.arc.goal_inference import GoalInferenceModule, InferredGoal, GoalType


def _make_mock_action(name, value, simple=True):
    """Create a mock GameAction."""
    action = MagicMock()
    action.name = name
    action.value = value
    action.is_simple = MagicMock(return_value=simple)
    action.is_complex = MagicMock(return_value=not simple)
    action.__str__ = lambda s: f"GameAction.{name}"
    action.__repr__ = lambda s: f"GameAction.{name}"
    action.__eq__ = lambda s, o: hasattr(o, "name") and s.name == o.name
    action.__hash__ = lambda s: hash(name)
    return action


RESET = _make_mock_action("RESET", 0)
ACTION1 = _make_mock_action("ACTION1", 1)
ACTION2 = _make_mock_action("ACTION2", 2)
ACTION3 = _make_mock_action("ACTION3", 3)
ACTION6 = _make_mock_action("ACTION6", 6, simple=False)


class TestInitialization:
    def test_initial_phase_is_discovery(self):
        e = HypothesisDrivenExplorer()
        assert e.phase == ExplorationPhase.DISCOVERY

    def test_initialize_discovery_simple_only(self):
        e = HypothesisDrivenExplorer()
        e.initialize_discovery([RESET, ACTION1, ACTION2, ACTION3])
        # RESET filtered out, 3 simple actions remain
        assert len(e.discovery_queue) == 3
        # All should be (action, {}) tuples
        for action, data in e.discovery_queue:
            assert data == {}

    def test_initialize_discovery_with_complex(self):
        e = HypothesisDrivenExplorer()
        e.initialize_discovery([RESET, ACTION1, ACTION6])
        # 1 simple + 13 complex samples
        assert len(e.discovery_queue) == 14

    def test_stores_available_actions(self):
        e = HypothesisDrivenExplorer()
        e.initialize_discovery([ACTION1, ACTION2])
        assert len(e._available_actions) == 2


class TestPhaseTransitions:
    def test_discovery_to_hypothesis_on_empty_queue(self):
        e = HypothesisDrivenExplorer()
        e.initialize_discovery([ACTION1])
        e.discovery_queue.clear()
        e._check_phase_transition(EpisodeMemory(), GoalInferenceModule())
        assert e.phase == ExplorationPhase.HYPOTHESIS

    def test_discovery_to_hypothesis_on_max_steps(self):
        e = HypothesisDrivenExplorer()
        e.initialize_discovery([ACTION1])
        e._phase_step_count = 51
        e._check_phase_transition(EpisodeMemory(), GoalInferenceModule())
        assert e.phase == ExplorationPhase.HYPOTHESIS

    def test_hypothesis_to_exploitation_on_high_confidence(self):
        e = HypothesisDrivenExplorer()
        e.phase = ExplorationPhase.HYPOTHESIS
        goals = GoalInferenceModule()
        goals.current_goals = [InferredGoal(GoalType.REACH_STATE, "win", 0.8)]
        e._check_phase_transition(EpisodeMemory(), goals)
        assert e.phase == ExplorationPhase.EXPLOITATION

    def test_stays_in_hypothesis_with_low_confidence(self):
        e = HypothesisDrivenExplorer()
        e.phase = ExplorationPhase.HYPOTHESIS
        goals = GoalInferenceModule()
        goals.current_goals = [InferredGoal(GoalType.UNKNOWN, "dunno", 0.3)]
        e._check_phase_transition(EpisodeMemory(), goals)
        assert e.phase == ExplorationPhase.HYPOTHESIS


class TestChooseAction:
    def _make_obs(self):
        import numpy as np

        return type(
            "Obs",
            (),
            {
                "raw_grid": np.zeros((64, 64), dtype="int8"),
                "game_state": "NOT_FINISHED",
                "changed_pixels": 0,
                "level": 0,
            },
        )()

    def test_discovery_returns_from_queue(self):
        e = HypothesisDrivenExplorer()
        e.initialize_discovery([ACTION1, ACTION2])
        obs = self._make_obs()
        mem = EpisodeMemory()
        goals = GoalInferenceModule()
        action, data = e.choose_action(obs, mem, goals)
        # Should return something (not crash)
        assert action is not None

    def test_exploitation_with_no_wins_falls_back(self):
        e = HypothesisDrivenExplorer()
        e.initialize_discovery([ACTION1])
        e.phase = ExplorationPhase.EXPLOITATION
        obs = self._make_obs()
        mem = EpisodeMemory()
        goals = GoalInferenceModule()
        action, data = e.choose_action(obs, mem, goals)
        # Should not crash, falls back to hypothesis/random
        assert action is not None


class TestParseAction:
    def test_simple_action(self):
        e = HypothesisDrivenExplorer()
        e.initialize_discovery([ACTION1, ACTION6])
        action, data = e._parse_action_str("ACTION1")
        assert data == {}

    def test_complex_action_with_coords(self):
        e = HypothesisDrivenExplorer()
        e.initialize_discovery([ACTION1, ACTION6])
        action, data = e._parse_action_str("ACTION6_32_15")
        assert data.get("x") == 32
        assert data.get("y") == 15
