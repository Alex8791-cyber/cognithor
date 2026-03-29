"""ARC-AGI-3 Audit Trail with SHA-256 hash chain for tamper detection."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Any

__all__ = ["ArcAuditEvent", "ArcAuditTrail"]


@dataclass
class ArcAuditEvent:
    """A single auditable event in an ARC game session."""

    timestamp: float
    event_type: str  # "game_start", "step", "level_complete", "game_end", "error"
    game_id: str
    level: int
    step: int
    action: str | None = None
    game_state: str | None = None
    pixels_changed: int | None = None
    score: float | None = None
    error: str | None = None
    metadata: dict[str, Any] | None = None


def _event_to_json(event: ArcAuditEvent) -> str:
    """Serialize an event to a canonical JSON string (sorted keys)."""
    return json.dumps(asdict(event), sort_keys=True, ensure_ascii=False)


class ArcAuditTrail:
    """Append-only audit trail with a SHA-256 hash chain for tamper detection."""

    def __init__(self, game_id: str, agent_version: str = "cognithor-arc-v1") -> None:
        self.game_id = game_id
        self.agent_version = agent_version
        self.events: list[ArcAuditEvent] = []
        self._hashes: list[str] = []
        self._previous_hash: str | None = None

        # run_id: first 16 chars of SHA-256(game_id:timestamp:version)
        seed = f"{game_id}:{time.time()}:{agent_version}"
        self.run_id = hashlib.sha256(seed.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Core append method
    # ------------------------------------------------------------------

    def log_event(self, event: ArcAuditEvent) -> str:
        """Append *event* to the trail, compute its chain hash, and return it."""
        event_json = _event_to_json(event)
        prev = self._previous_hash if self._previous_hash is not None else "GENESIS"
        chain_input = f"{prev}:{event_json}"
        new_hash = hashlib.sha256(chain_input.encode()).hexdigest()

        self.events.append(event)
        self._hashes.append(new_hash)
        self._previous_hash = new_hash
        return new_hash

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def log_game_start(self) -> str:
        """Log a game_start event and return its chain hash."""
        event = ArcAuditEvent(
            timestamp=time.time(),
            event_type="game_start",
            game_id=self.game_id,
            level=0,
            step=0,
            metadata={"agent_version": self.agent_version, "run_id": self.run_id},
        )
        return self.log_event(event)

    def log_game_end(self, final_score: float) -> str:
        """Log a game_end event with the final score and return its chain hash."""
        event = ArcAuditEvent(
            timestamp=time.time(),
            event_type="game_end",
            game_id=self.game_id,
            level=0,
            step=0,
            score=final_score,
        )
        return self.log_event(event)

    def log_step(
        self,
        level: int,
        step: int,
        action: str,
        game_state: str,
        pixels_changed: int,
    ) -> str:
        """Log a single agent step and return its chain hash."""
        event = ArcAuditEvent(
            timestamp=time.time(),
            event_type="step",
            game_id=self.game_id,
            level=level,
            step=step,
            action=action,
            game_state=game_state,
            pixels_changed=pixels_changed,
        )
        return self.log_event(event)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_jsonl(self, filepath: str) -> None:
        """Write all events as JSONL (one JSON object per line)."""
        with open(filepath, "w", encoding="utf-8") as fh:
            for event in self.events:
                fh.write(_event_to_json(event) + "\n")

    # ------------------------------------------------------------------
    # Integrity verification
    # ------------------------------------------------------------------

    def verify_integrity(self) -> bool:
        """Replay the hash chain from scratch and confirm it matches stored hashes."""
        if not self.events:
            return True

        prev = "GENESIS"
        for event, stored_hash in zip(self.events, self._hashes, strict=False):
            event_json = _event_to_json(event)
            chain_input = f"{prev}:{event_json}"
            expected = hashlib.sha256(chain_input.encode()).hexdigest()
            if expected != stored_hash:
                return False
            prev = stored_hash

        return True
