"""Tests for jarvis.arc.mechanics_model."""

from __future__ import annotations

from jarvis.arc.episode_memory import EpisodeMemory
from jarvis.arc.mechanics_model import MechanicType, MechanicsModel


def _populated_memory():
    """Memory with enough data for analysis."""
    mem = EpisodeMemory()
    # Simulate ACTION1 causing changes 9/10 times
    for i in range(9):
        mem.action_effect_map["ACTION1"]["total"] += 1
        mem.action_effect_map["ACTION1"]["caused_change"] += 1
    mem.action_effect_map["ACTION1"]["total"] += 1  # 1 no-change

    # ACTION2 never causes change
    for _ in range(5):
        mem.action_effect_map["ACTION2"]["total"] += 1

    # ACTION3 causes change 50% of the time
    for _ in range(4):
        mem.action_effect_map["ACTION3"]["total"] += 1
        mem.action_effect_map["ACTION3"]["caused_change"] += 1
    for _ in range(4):
        mem.action_effect_map["ACTION3"]["total"] += 1

    return mem


class TestAnalyzeTransitions:
    def test_high_change_rate_classified_as_movement(self):
        mm = MechanicsModel()
        mem = _populated_memory()
        mm.analyze_transitions(mem, current_level=0)
        m = mm.get_mechanics_for_action("ACTION1")
        assert len(m) >= 1
        assert m[0].mechanic_type == MechanicType.MOVEMENT

    def test_low_change_rate_classified_as_no_effect(self):
        mm = MechanicsModel()
        mem = _populated_memory()
        mm.analyze_transitions(mem, current_level=0)
        m = mm.get_mechanics_for_action("ACTION2")
        assert len(m) >= 1
        assert m[0].mechanic_type == MechanicType.NO_EFFECT

    def test_medium_change_rate_classified_as_conditional(self):
        mm = MechanicsModel()
        mem = _populated_memory()
        mm.analyze_transitions(mem, current_level=0)
        m = mm.get_mechanics_for_action("ACTION3")
        assert len(m) >= 1
        assert m[0].mechanic_type == MechanicType.CONDITIONAL

    def test_skips_low_observation_count(self):
        mm = MechanicsModel()
        mem = EpisodeMemory()
        mem.action_effect_map["RARE"]["total"] = 2
        mem.action_effect_map["RARE"]["caused_change"] = 2
        mm.analyze_transitions(mem, current_level=0)
        assert mm.get_mechanics_for_action("RARE") == []

    def test_updates_existing_mechanic(self):
        mm = MechanicsModel()
        mem = _populated_memory()
        mm.analyze_transitions(mem, current_level=0)
        count1 = mm.get_mechanics_for_action("ACTION1")[0].observation_count
        mm.analyze_transitions(mem, current_level=1)
        count2 = mm.get_mechanics_for_action("ACTION1")[0].observation_count
        assert count2 > count1

    def test_tracks_levels(self):
        mm = MechanicsModel()
        mem = _populated_memory()
        mm.analyze_transitions(mem, current_level=0)
        mm.analyze_transitions(mem, current_level=2)
        m = mm.get_mechanics_for_action("ACTION1")[0]
        assert 0 in m.observed_in_levels
        assert 2 in m.observed_in_levels


class TestReliableMechanics:
    def test_empty_returns_empty(self):
        mm = MechanicsModel()
        assert mm.get_reliable_mechanics() == []

    def test_filters_by_consistency(self):
        mm = MechanicsModel()
        mem = _populated_memory()
        mm.analyze_transitions(mem, current_level=0)
        reliable = mm.get_reliable_mechanics(min_consistency=0.5)
        assert len(reliable) >= 1  # ACTION1 should qualify
        for m in reliable:
            assert m.consistency_score >= 0.5


class TestPrediction:
    def test_unknown_action(self):
        mm = MechanicsModel()
        assert mm.predict_action_effect("NEVER_SEEN") == MechanicType.UNKNOWN

    def test_predicts_known_action(self):
        mm = MechanicsModel()
        mem = _populated_memory()
        mm.analyze_transitions(mem, current_level=0)
        result = mm.predict_action_effect("ACTION1")
        assert result == MechanicType.MOVEMENT


class TestSnapshot:
    def test_saves_snapshot(self):
        mm = MechanicsModel()
        mem = _populated_memory()
        mm.snapshot_level(0, mem)
        assert len(mm._level_snapshots) == 1
        assert mm._level_snapshots[0]["level"] == 0


class TestSummary:
    def test_returns_string(self):
        mm = MechanicsModel()
        s = mm.get_summary_for_llm()
        assert isinstance(s, str)

    def test_includes_mechanics(self):
        mm = MechanicsModel()
        mem = _populated_memory()
        mm.analyze_transitions(mem, current_level=0)
        s = mm.get_summary_for_llm()
        assert "ACTION1" in s or "Mechanik" in s or "mechanic" in s.lower()
