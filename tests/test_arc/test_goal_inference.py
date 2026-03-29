from jarvis.arc.goal_inference import GoalInferenceModule, GoalType, InferredGoal
from jarvis.arc.episode_memory import EpisodeMemory, StateTransition


class TestAnalyzeWinCondition:
    def test_no_data_returns_unknown(self):
        gim = GoalInferenceModule()
        mem = EpisodeMemory()
        goals = gim.analyze_win_condition(mem)
        assert any(g.goal_type == GoalType.UNKNOWN for g in goals)

    def test_win_transitions_produce_reach_state(self):
        gim = GoalInferenceModule()
        mem = EpisodeMemory()
        t = StateTransition(
            state_hash="a",
            action="ACTION3",
            next_state_hash="b",
            pixels_changed=50,
            resulted_in_win=True,
            level=0,
        )
        mem.transitions.append(t)
        goals = gim.analyze_win_condition(mem)
        assert any(g.goal_type == GoalType.REACH_STATE for g in goals)
        reach = [g for g in goals if g.goal_type == GoalType.REACH_STATE][0]
        assert "ACTION3" in reach.description

    def test_game_over_produces_avoid(self):
        gim = GoalInferenceModule()
        mem = EpisodeMemory()
        t = StateTransition(
            state_hash="x",
            action="ACTION2",
            next_state_hash="y",
            pixels_changed=10,
            resulted_in_game_over=True,
            level=0,
        )
        mem.transitions.append(t)
        goals = gim.analyze_win_condition(mem)
        assert any(g.goal_type == GoalType.AVOID for g in goals)

    def test_large_pixel_change_produces_clear_board(self):
        gim = GoalInferenceModule()
        mem = EpisodeMemory()
        for i in range(5):
            t = StateTransition(
                state_hash=f"s{i}",
                action="ACTION1",
                next_state_hash=f"s{i + 1}",
                pixels_changed=200,
                level=0,
            )
            mem.transitions.append(t)
        goals = gim.analyze_win_condition(mem)
        assert any(g.goal_type == GoalType.CLEAR_BOARD for g in goals)

    def test_sorted_by_confidence(self):
        gim = GoalInferenceModule()
        mem = EpisodeMemory()
        # Add both win and game_over
        mem.transitions.append(
            StateTransition(
                state_hash="a",
                action="A1",
                next_state_hash="b",
                pixels_changed=10,
                resulted_in_win=True,
                level=0,
            )
        )
        mem.transitions.append(
            StateTransition(
                state_hash="c",
                action="A2",
                next_state_hash="d",
                pixels_changed=5,
                resulted_in_game_over=True,
                level=0,
            )
        )
        goals = gim.analyze_win_condition(mem)
        confidences = [g.confidence for g in goals]
        assert confidences == sorted(confidences, reverse=True)


class TestGetBestGoal:
    def test_empty_returns_unknown(self):
        gim = GoalInferenceModule()
        best = gim.get_best_goal()
        assert best.goal_type == GoalType.UNKNOWN
        assert best.confidence == 0.0

    def test_returns_highest_confidence(self):
        gim = GoalInferenceModule()
        gim.current_goals = [
            InferredGoal(GoalType.AVOID, "avoid", 0.3),
            InferredGoal(GoalType.REACH_STATE, "reach", 0.8),
        ]
        assert gim.get_best_goal().goal_type == GoalType.REACH_STATE


class TestLevelComplete:
    def test_stores_data(self):
        gim = GoalInferenceModule()
        gim.on_level_complete({"level": 1, "steps": 42})
        assert len(gim._level_progression_data) == 1


class TestSummary:
    def test_returns_string(self):
        gim = GoalInferenceModule()
        s = gim.get_summary_for_llm()
        assert isinstance(s, str)

    def test_includes_goal_info(self):
        gim = GoalInferenceModule()
        gim.current_goals = [
            InferredGoal(GoalType.REACH_STATE, "win via ACTION3", 0.7),
        ]
        s = gim.get_summary_for_llm()
        assert "ACTION3" in s or "reach" in s.lower()
