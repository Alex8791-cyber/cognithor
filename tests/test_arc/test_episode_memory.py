import pytest
import numpy as np
from jarvis.arc.episode_memory import EpisodeMemory, StateTransition, Hypothesis


def _make_obs(grid=None, game_state="NOT_FINISHED", changed_pixels=0, level=0):
    """Helper to create mock observation objects."""
    if grid is None:
        grid = np.zeros((64, 64), dtype=np.int8)
    return type(
        "Obs",
        (),
        {
            "raw_grid": grid,
            "game_state": game_state,
            "changed_pixels": changed_pixels,
            "level": level,
        },
    )()


class TestHashGrid:
    def test_deterministic(self):
        mem = EpisodeMemory()
        grid = np.random.randint(0, 10, (64, 64), dtype=np.int8)
        assert mem.hash_grid(grid) == mem.hash_grid(grid.copy())

    def test_different_grids_differ(self):
        mem = EpisodeMemory()
        g1 = np.zeros((64, 64), dtype=np.int8)
        g2 = np.ones((64, 64), dtype=np.int8)
        assert mem.hash_grid(g1) != mem.hash_grid(g2)

    def test_hash_length(self):
        mem = EpisodeMemory()
        grid = np.zeros((64, 64), dtype=np.int8)
        assert len(mem.hash_grid(grid)) == 16

    def test_cached(self):
        mem = EpisodeMemory()
        grid = np.zeros((64, 64), dtype=np.int8)
        h1 = mem.hash_grid(grid)
        assert len(mem._state_hash_cache) == 1
        h2 = mem.hash_grid(grid)
        assert h1 == h2


class TestRecordTransition:
    def test_basic_recording(self):
        mem = EpisodeMemory()
        before = _make_obs()
        after = _make_obs(
            grid=np.ones((64, 64), dtype=np.int8),
            changed_pixels=4096,
        )
        t = mem.record_transition(before, "ACTION1", after)
        assert t.action == "ACTION1"
        assert t.pixels_changed == 4096
        assert len(mem.transitions) == 1

    def test_win_detection(self):
        mem = EpisodeMemory()
        before = _make_obs()
        after = _make_obs(game_state="WIN", changed_pixels=10)
        t = mem.record_transition(before, "ACTION3", after)
        assert t.resulted_in_win is True
        assert mem.action_effect_map["ACTION3"]["caused_win"] == 1

    def test_game_over_detection(self):
        mem = EpisodeMemory()
        before = _make_obs()
        after = _make_obs(game_state="GAME_OVER")
        t = mem.record_transition(before, "ACTION2", after)
        assert t.resulted_in_game_over is True

    def test_max_transitions_respected(self):
        mem = EpisodeMemory(max_transitions=5)
        obs = _make_obs()
        for i in range(10):
            mem.record_transition(obs, f"ACTION{i}", obs)
        assert len(mem.transitions) == 5

    def test_visited_states_tracked(self):
        mem = EpisodeMemory()
        before = _make_obs()
        after = _make_obs(grid=np.ones((64, 64), dtype=np.int8))
        mem.record_transition(before, "ACTION1", after)
        assert len(mem.visited_states) == 2  # before + after


class TestActionEffectiveness:
    def test_unknown_action(self):
        mem = EpisodeMemory()
        assert mem.get_action_effectiveness("NEVER_USED") == 0.5

    def test_calculated_correctly(self):
        mem = EpisodeMemory()
        obs = _make_obs()
        changed = _make_obs(changed_pixels=100)
        unchanged = _make_obs(changed_pixels=0)
        for _ in range(3):
            mem.record_transition(obs, "A1", changed)
        mem.record_transition(obs, "A1", unchanged)
        assert mem.get_action_effectiveness("A1") == 0.75


class TestExploration:
    def test_unexplored_actions(self):
        mem = EpisodeMemory()
        obs = _make_obs()
        mem.record_transition(obs, "A1", obs)
        state_hash = mem.hash_grid(obs.raw_grid)
        unexplored = mem.get_unexplored_actions(state_hash, ["A1", "A2", "A3"])
        assert "A1" not in unexplored
        assert "A2" in unexplored
        assert "A3" in unexplored

    def test_novel_state(self):
        mem = EpisodeMemory()
        g1 = np.zeros((64, 64), dtype=np.int8)
        g2 = np.ones((64, 64), dtype=np.int8)
        assert mem.is_novel_state(g1) is True
        mem.visited_states.add(mem.hash_grid(g1))
        assert mem.is_novel_state(g1) is False
        assert mem.is_novel_state(g2) is True


class TestHypotheses:
    def test_add_hypothesis(self):
        mem = EpisodeMemory()
        h = mem.add_hypothesis("ACTION1 moves object right")
        assert isinstance(h, Hypothesis)
        assert h.description == "ACTION1 moves object right"
        assert len(mem.hypotheses) == 1


class TestLevelManagement:
    def test_clear_for_new_level(self):
        mem = EpisodeMemory()
        obs = _make_obs()
        mem.record_transition(obs, "A1", obs)
        transitions_before = len(mem.transitions)
        effects_before = dict(mem.action_effect_map)

        mem.clear_for_new_level()
        assert len(mem.state_visit_count) == 0
        assert len(mem.transitions) == transitions_before  # kept
        assert len(mem.action_effect_map) == len(effects_before)  # kept


class TestSummary:
    def test_returns_string(self):
        mem = EpisodeMemory()
        s = mem.get_summary_for_llm()
        assert isinstance(s, str)
        assert "Besuchte" in s or "visited" in s.lower() or "Zust" in s

    def test_includes_action_data(self):
        mem = EpisodeMemory()
        obs = _make_obs()
        mem.record_transition(obs, "ACTION1", obs)
        s = mem.get_summary_for_llm()
        assert "ACTION1" in s
