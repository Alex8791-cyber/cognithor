"""Evolution cycle resume — load checkpoint and continue from where we stopped."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.core.checkpointing import CheckpointStore

log = get_logger(__name__)

__all__ = ["EvolutionResumer", "ResumeState"]

# Step order in an evolution cycle
EVOLUTION_STEPS = ["scout", "research", "build", "reflect"]


@dataclass
class ResumeState:
    """Describes what needs to be done to resume a cycle."""

    has_checkpoint: bool = False
    cycle_id: int = 0
    last_step: str = ""
    last_step_index: int = -1
    next_step: str = ""
    next_step_index: int = 0
    is_complete: bool = False
    # Data carried over from checkpoint
    gaps_found: int = 0
    research_topic: str = ""
    research_text: str = ""
    skill_created: str = ""
    steps_completed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_checkpoint": self.has_checkpoint,
            "cycle_id": self.cycle_id,
            "last_step": self.last_step,
            "next_step": self.next_step,
            "is_complete": self.is_complete,
            "gaps_found": self.gaps_found,
            "research_topic": self.research_topic,
            "skill_created": self.skill_created,
            "steps_completed": self.steps_completed,
        }


class EvolutionResumer:
    """Manages checkpoint-based resume for evolution cycles.

    Usage:
        resumer = EvolutionResumer(checkpoint_store)
        state = resumer.get_resume_state(cycle_id)
        if state.has_checkpoint and not state.is_complete:
            # Resume from state.next_step
    """

    def __init__(self, checkpoint_store: CheckpointStore) -> None:
        self._store = checkpoint_store

    def get_resume_state(self, cycle_id: int) -> ResumeState:
        """Check if a cycle has a checkpoint and determine resume point."""
        session_id = f"evolution-{cycle_id}"
        cp = self._store.get_latest(session_id)
        if cp is None:
            return ResumeState()

        state_data = cp.state
        last_step = state_data.get("step_name", "")
        last_step_index = state_data.get("step_index", -1)

        # Determine next step
        is_complete = last_step == "reflect"
        next_step_index = last_step_index + 1
        next_step = ""
        if not is_complete and next_step_index < len(EVOLUTION_STEPS):
            next_step = EVOLUTION_STEPS[next_step_index]

        return ResumeState(
            has_checkpoint=True,
            cycle_id=cycle_id,
            last_step=last_step,
            last_step_index=last_step_index,
            next_step=next_step,
            next_step_index=next_step_index,
            is_complete=is_complete,
            gaps_found=state_data.get("gaps_found", 0),
            research_topic=state_data.get("research_topic", ""),
            research_text=state_data.get("research_text", ""),
            skill_created=state_data.get("skill_created", ""),
            steps_completed=state_data.get("steps_completed", []),
        )

    def get_latest_cycle_id(self) -> int | None:
        """Find the most recent evolution cycle ID from checkpoints."""
        checkpoint_dir = self._store._dir
        if not checkpoint_dir.exists():
            return None
        cycle_ids = []
        for d in checkpoint_dir.iterdir():
            if d.is_dir() and d.name.startswith("evolution-"):
                try:
                    cycle_ids.append(int(d.name.split("-", 1)[1]))
                except (ValueError, IndexError):
                    continue
        return max(cycle_ids) if cycle_ids else None

    def clear_cycle(self, cycle_id: int) -> int:
        """Clear all checkpoints for a cycle."""
        return self._store.clear_session(f"evolution-{cycle_id}")

    def list_cycles(self) -> list[int]:
        """List all cycle IDs that have checkpoints."""
        checkpoint_dir = self._store._dir
        if not checkpoint_dir.exists():
            return []
        cycle_ids = []
        for d in checkpoint_dir.iterdir():
            if d.is_dir() and d.name.startswith("evolution-"):
                try:
                    cycle_ids.append(int(d.name.split("-", 1)[1]))
                except (ValueError, IndexError):
                    continue
        return sorted(cycle_ids)
