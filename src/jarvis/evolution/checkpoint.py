"""Evolution cycle checkpointing — step-level state persistence."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvolutionCheckpoint:
    """State of an evolution cycle at a specific step."""

    cycle_id: int = 0
    step_name: str = ""  # "scout" | "research" | "build" | "reflect" | "complete"
    step_index: int = 0  # 0-based index within STEPS
    timestamp: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    # Accumulated data from completed steps
    gaps_found: int = 0
    research_topic: str = ""
    research_text: str = ""
    skill_created: str = ""
    steps_completed: list[str] = field(default_factory=list)
    # Delta: only what changed since last checkpoint
    delta: dict[str, Any] = field(default_factory=dict)

    # Ordered step names for the evolution cycle
    STEPS: list[str] = field(
        default_factory=lambda: ["scout", "research", "build", "reflect"],
        repr=False,
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "step_name": self.step_name,
            "step_index": self.step_index,
            "timestamp": self.timestamp,
            "gaps_found": self.gaps_found,
            "research_topic": self.research_topic,
            "research_text": self.research_text[:500],  # Truncate for storage
            "skill_created": self.skill_created,
            "steps_completed": self.steps_completed,
            "delta": self.delta,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvolutionCheckpoint:
        return cls(
            cycle_id=d.get("cycle_id", 0),
            step_name=d.get("step_name", ""),
            step_index=d.get("step_index", 0),
            timestamp=d.get("timestamp", ""),
            gaps_found=d.get("gaps_found", 0),
            research_topic=d.get("research_topic", ""),
            research_text=d.get("research_text", ""),
            skill_created=d.get("skill_created", ""),
            steps_completed=d.get("steps_completed", []),
            delta=d.get("delta", {}),
        )

    @classmethod
    def from_json(cls, data: str) -> EvolutionCheckpoint:
        return cls.from_dict(json.loads(data))

    @property
    def next_step_index(self) -> int:
        """Index of the next step to execute."""
        return self.step_index + 1

    @property
    def is_complete(self) -> bool:
        """True if all steps have been completed."""
        return self.step_index >= len(self.STEPS) - 1 and self.step_name == "reflect"
