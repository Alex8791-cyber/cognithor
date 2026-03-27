"""Tests fuer Evolution Checkpoint + Resume."""

from __future__ import annotations

import pytest

from jarvis.core.checkpointing import CheckpointStore, PersistentCheckpoint
from jarvis.evolution.resume import EvolutionResumer, ResumeState, EVOLUTION_STEPS


@pytest.fixture()
def store(tmp_path):
    """CheckpointStore mit temp-Verzeichnis."""
    return CheckpointStore(tmp_path / "checkpoints")


@pytest.fixture()
def resumer(store):
    return EvolutionResumer(store)


def _save_evolution_checkpoint(store, cycle_id, step_name, step_index, **extra):
    """Helper: speichert einen Evolution-Checkpoint."""
    state = {
        "cycle_id": cycle_id,
        "step_name": step_name,
        "step_index": step_index,
        "gaps_found": extra.get("gaps_found", 0),
        "research_topic": extra.get("research_topic", ""),
        "research_text": extra.get("research_text", ""),
        "skill_created": extra.get("skill_created", ""),
        "steps_completed": extra.get("steps_completed", []),
    }
    cp = PersistentCheckpoint(
        session_id=f"evolution-{cycle_id}",
        agent_id="evolution-loop",
        state=state,
    )
    store.save(cp)


class TestResumeState:
    def test_no_checkpoint(self, resumer):
        """Kein Checkpoint -> has_checkpoint=False."""
        state = resumer.get_resume_state(999)
        assert state.has_checkpoint is False
        assert state.is_complete is False

    def test_resume_after_scout(self, store, resumer):
        """Checkpoint nach Scout -> next_step=research."""
        _save_evolution_checkpoint(
            store, cycle_id=1, step_name="scout", step_index=0,
            gaps_found=3, steps_completed=["scout"],
        )
        state = resumer.get_resume_state(1)
        assert state.has_checkpoint is True
        assert state.last_step == "scout"
        assert state.next_step == "research"
        assert state.next_step_index == 1
        assert state.gaps_found == 3
        assert not state.is_complete

    def test_resume_after_research(self, store, resumer):
        """Checkpoint nach Research -> next_step=build."""
        _save_evolution_checkpoint(
            store, cycle_id=2, step_name="research", step_index=1,
            research_topic="Python async patterns",
            research_text="Some research...",
            steps_completed=["scout", "research"],
        )
        state = resumer.get_resume_state(2)
        assert state.next_step == "build"
        assert state.research_topic == "Python async patterns"

    def test_resume_after_build(self, store, resumer):
        """Checkpoint nach Build -> next_step=reflect."""
        _save_evolution_checkpoint(
            store, cycle_id=3, step_name="build", step_index=2,
            skill_created="async_patterns",
            steps_completed=["scout", "research", "build"],
        )
        state = resumer.get_resume_state(3)
        assert state.next_step == "reflect"
        assert state.skill_created == "async_patterns"

    def test_complete_after_reflect(self, store, resumer):
        """Checkpoint nach Reflect -> is_complete=True."""
        _save_evolution_checkpoint(
            store, cycle_id=4, step_name="reflect", step_index=3,
            steps_completed=["scout", "research", "build", "reflect"],
        )
        state = resumer.get_resume_state(4)
        assert state.is_complete is True
        assert state.next_step == ""

    def test_to_dict(self, store, resumer):
        """to_dict enthaelt alle wichtigen Felder."""
        _save_evolution_checkpoint(
            store, cycle_id=5, step_name="scout", step_index=0,
        )
        state = resumer.get_resume_state(5)
        d = state.to_dict()
        assert "has_checkpoint" in d
        assert "cycle_id" in d
        assert "next_step" in d
        assert "is_complete" in d


class TestEvolutionResumer:
    def test_get_latest_cycle_id_empty(self, resumer):
        """Keine Checkpoints -> None."""
        assert resumer.get_latest_cycle_id() is None

    def test_get_latest_cycle_id(self, store, resumer):
        """Mehrere Cycles -> hoechste ID."""
        _save_evolution_checkpoint(store, 1, "scout", 0)
        _save_evolution_checkpoint(store, 5, "research", 1)
        _save_evolution_checkpoint(store, 3, "build", 2)
        assert resumer.get_latest_cycle_id() == 5

    def test_list_cycles(self, store, resumer):
        """list_cycles gibt sortierte Cycle-IDs."""
        _save_evolution_checkpoint(store, 3, "scout", 0)
        _save_evolution_checkpoint(store, 1, "scout", 0)
        _save_evolution_checkpoint(store, 7, "scout", 0)
        assert resumer.list_cycles() == [1, 3, 7]

    def test_clear_cycle(self, store, resumer):
        """clear_cycle entfernt alle Checkpoints eines Cycles."""
        _save_evolution_checkpoint(store, 10, "scout", 0)
        _save_evolution_checkpoint(store, 10, "research", 1)
        count = resumer.clear_cycle(10)
        assert count == 2
        assert resumer.get_resume_state(10).has_checkpoint is False

    def test_clear_nonexistent(self, resumer):
        """clear_cycle fuer unbekannten Cycle -> 0."""
        assert resumer.clear_cycle(999) == 0


class TestEvolutionSteps:
    def test_step_order(self):
        """EVOLUTION_STEPS hat die richtige Reihenfolge."""
        assert EVOLUTION_STEPS == ["scout", "research", "build", "reflect"]
