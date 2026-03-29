"""Tests for CognithorArcAgent — orchestration logic, tested with mocks."""

from __future__ import annotations

from unittest.mock import MagicMock

from jarvis.arc.agent import CognithorArcAgent


class TestActionToStr:
    def test_simple_action(self):
        action = MagicMock()
        action.name = "ACTION1"
        result = CognithorArcAgent._action_to_str(action, {})
        assert result == "ACTION1"

    def test_complex_action_with_coords(self):
        action = MagicMock()
        action.name = "ACTION6"
        result = CognithorArcAgent._action_to_str(action, {"x": 32, "y": 15})
        assert result == "ACTION6_32_15"

    def test_action_without_name(self):
        action = "RAW_ACTION"
        result = CognithorArcAgent._action_to_str(action, {})
        assert isinstance(result, str)

    def test_data_missing_y_key_no_suffix(self):
        action = MagicMock()
        action.name = "ACTION2"
        result = CognithorArcAgent._action_to_str(action, {"x": 10})
        assert result == "ACTION2"

    def test_data_missing_x_key_no_suffix(self):
        action = MagicMock()
        action.name = "ACTION3"
        result = CognithorArcAgent._action_to_str(action, {"y": 20})
        assert result == "ACTION3"

    def test_zero_coords(self):
        action = MagicMock()
        action.name = "ACTION9"
        result = CognithorArcAgent._action_to_str(action, {"x": 0, "y": 0})
        assert result == "ACTION9_0_0"


class TestAgentInit:
    def test_creates_all_modules(self):
        agent = CognithorArcAgent("ls20", use_llm_planner=False)
        assert agent.adapter is not None
        assert agent.memory is not None
        assert agent.goals is not None
        assert agent.explorer is not None
        assert agent.encoder is not None
        assert agent.mechanics is not None
        assert agent.audit_trail is not None
        assert agent.state_graph is not None
        assert agent.game_id == "ls20"
        assert agent.current_level == 0
        assert agent.total_steps == 0

    def test_default_params(self):
        agent = CognithorArcAgent("test")
        assert agent.max_steps_per_level == 500
        assert agent.max_resets_per_level == 20
        assert agent.llm_call_interval == 30

    def test_use_llm_planner_false(self):
        agent = CognithorArcAgent("test", use_llm_planner=False)
        assert agent.use_llm_planner is False

    def test_use_llm_planner_true(self):
        agent = CognithorArcAgent("test", use_llm_planner=True)
        assert agent.use_llm_planner is True

    def test_custom_params(self):
        agent = CognithorArcAgent(
            "g1",
            use_llm_planner=False,
            llm_call_interval=20,
            max_steps_per_level=100,
            max_resets_per_level=2,
        )
        assert agent.llm_call_interval == 20
        assert agent.max_steps_per_level == 100
        assert agent.max_resets_per_level == 2

    def test_initial_obs_is_none(self):
        agent = CognithorArcAgent("test")
        assert agent.current_obs is None

    def test_initial_level_resets_zero(self):
        agent = CognithorArcAgent("test")
        assert agent.level_resets == 0

    def test_audit_trail_game_id(self):
        agent = CognithorArcAgent("my_game")
        assert agent.audit_trail.game_id == "my_game"


class TestOnLevelComplete:
    def test_increments_level(self):
        agent = CognithorArcAgent("ls20", use_llm_planner=False)
        # Mock the adapter to have an env with action_space
        agent.adapter.env = MagicMock()
        agent.adapter.env.action_space = [MagicMock(name="A1")]
        agent.adapter.level_step_count = 42
        agent.level_resets = 3

        agent._on_level_complete()
        assert agent.current_level == 1
        assert agent.level_resets == 0

    def test_resets_explorer_phase(self):
        from jarvis.arc.explorer import ExplorationPhase

        agent = CognithorArcAgent("ls20", use_llm_planner=False)
        agent.adapter.env = MagicMock()
        agent.adapter.env.action_space = [MagicMock(name="A1")]
        agent.explorer.phase = ExplorationPhase.EXPLOITATION

        agent._on_level_complete()
        assert agent.explorer.phase == ExplorationPhase.DISCOVERY

    def test_clears_memory_for_new_level(self):
        agent = CognithorArcAgent("ls20", use_llm_planner=False)
        agent.adapter.env = MagicMock()
        agent.adapter.env.action_space = [MagicMock(name="A1")]

        # Put some state into memory so clear_for_new_level has something to clear
        agent.memory.state_visit_count["abc"] = 3
        agent.memory.visited_states.add("abc")

        agent._on_level_complete()

        assert len(agent.memory.state_visit_count) == 0
        assert len(agent.memory.visited_states) == 0

    def test_goals_on_level_complete_called(self):
        agent = CognithorArcAgent("ls20", use_llm_planner=False)
        agent.adapter.env = MagicMock()
        agent.adapter.env.action_space = [MagicMock(name="A1")]
        agent.adapter.level_step_count = 10
        agent.level_resets = 1

        agent._on_level_complete()

        # Goals module should have recorded the level data
        assert len(agent.goals._level_progression_data) == 1
        recorded = agent.goals._level_progression_data[0]
        assert recorded["level"] == 0
        assert recorded["steps"] == 10
        assert recorded["resets"] == 1

    def test_multiple_levels(self):
        agent = CognithorArcAgent("multi", use_llm_planner=False)
        agent.adapter.env = MagicMock()
        agent.adapter.env.action_space = [MagicMock(name="A1")]

        agent._on_level_complete()
        agent._on_level_complete()
        agent._on_level_complete()

        assert agent.current_level == 3
        assert len(agent.goals._level_progression_data) == 3


class TestConsultLlmPlannerStub:
    def test_returns_defaults_unchanged(self):
        agent = CognithorArcAgent("test", use_llm_planner=True)
        action = MagicMock()
        action.name = "ACTION1"
        data = {"x": 5, "y": 10}

        returned_action, returned_data = agent._consult_llm_planner(action, data)

        assert returned_action is action
        assert returned_data is data

    def test_returns_simple_action_unchanged(self):
        agent = CognithorArcAgent("test", use_llm_planner=True)
        action = MagicMock()
        action.name = "MOVE"
        data: dict = {}

        returned_action, returned_data = agent._consult_llm_planner(action, data)
        assert returned_action is action
        assert returned_data is data
