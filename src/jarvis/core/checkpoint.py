"""Checkpoint-Manager: Snapshots und Rollback fuer den Agent-Kernel."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from jarvis.models import Checkpoint, KernelState, ToolResult
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class CheckpointManager:
    """Verwaltet Checkpoints fuer Kernel-Rollback. In-Memory pro Session."""

    MAX_CHECKPOINTS_PER_SESSION = 100

    def __init__(self) -> None:
        self._checkpoints: dict[str, list[Checkpoint]] = {}  # session_id -> checkpoints

    def create_checkpoint(
        self,
        session_id: str,
        kernel_state: KernelState,
        working_memory_snapshot: dict[str, Any] | None = None,
        completed_nodes: list[str] | None = None,
        tool_results: list[ToolResult] | None = None,
    ) -> Checkpoint:
        """Erstellt einen neuen Checkpoint."""
        cp = Checkpoint(
            session_id=session_id,
            kernel_state=kernel_state,
            working_memory_snapshot=working_memory_snapshot or {},
            completed_nodes=completed_nodes or [],
            tool_results=tool_results or [],
        )
        cps = self._checkpoints.setdefault(session_id, [])
        cps.append(cp)
        # Evict oldest checkpoints if over limit
        if len(cps) > self.MAX_CHECKPOINTS_PER_SESSION:
            self._checkpoints[session_id] = cps[-self.MAX_CHECKPOINTS_PER_SESSION:]
        log.debug(
            "checkpoint_created",
            session=session_id[:8],
            state=kernel_state.value,
            checkpoint_id=cp.id[:8],
        )
        return cp

    def restore_checkpoint(self, checkpoint_id: str) -> Checkpoint | None:
        """Holt einen spezifischen Checkpoint."""
        for checkpoints in self._checkpoints.values():
            for cp in checkpoints:
                if cp.id == checkpoint_id:
                    log.info("checkpoint_restored", checkpoint_id=checkpoint_id[:8])
                    return cp
        return None

    def list_checkpoints(self, session_id: str) -> list[Checkpoint]:
        """Alle Checkpoints einer Session (chronologisch)."""
        return list(self._checkpoints.get(session_id, []))

    def get_latest(self, session_id: str) -> Checkpoint | None:
        """Neuester Checkpoint einer Session."""
        checkpoints = self._checkpoints.get(session_id, [])
        return checkpoints[-1] if checkpoints else None

    def clear_session(self, session_id: str) -> None:
        """Loescht alle Checkpoints einer Session."""
        self._checkpoints.pop(session_id, None)

    @property
    def total_checkpoints(self) -> int:
        return sum(len(cps) for cps in self._checkpoints.values())
