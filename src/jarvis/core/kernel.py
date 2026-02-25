"""Agent-Kernel: State-Machine-basierte Execution mit Checkpoint/Rollback."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from jarvis.core.checkpoint import CheckpointManager
from jarvis.core.plan_graph import PlanGraph
from jarvis.models import (
    Checkpoint,
    KernelState,
    PlanNode,
    ToolResult,
)
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.core.executor import Executor
    from jarvis.core.gatekeeper import Gatekeeper
    from jarvis.core.planner import Planner
    from jarvis.models import PlannedAction, SessionContext

log = get_logger(__name__)


# Erlaubte State-Uebergaenge
_VALID_TRANSITIONS: dict[KernelState, set[KernelState]] = {
    KernelState.IDLE: {KernelState.ROUTING, KernelState.PLANNING, KernelState.ERROR},
    KernelState.ROUTING: {KernelState.PLANNING, KernelState.ERROR},
    KernelState.PLANNING: {KernelState.GATING, KernelState.DONE, KernelState.ERROR},
    KernelState.GATING: {KernelState.EXECUTING, KernelState.DONE, KernelState.ERROR},
    KernelState.EXECUTING: {KernelState.REFLECTING, KernelState.PLANNING, KernelState.ERROR},
    KernelState.REFLECTING: {KernelState.DONE, KernelState.PLANNING, KernelState.ERROR},
    KernelState.DONE: {KernelState.IDLE},
    KernelState.ERROR: {KernelState.IDLE, KernelState.PLANNING},
}


class KernelError(Exception):
    """Fehler im Agent-Kernel."""


class InvalidTransitionError(KernelError):
    """Ungueltiger State-Uebergang."""


class AgentKernel:
    """State-Machine-basierter Agent-Kernel mit Checkpoint/Rollback.

    Wraps den bestehenden PGE-Zyklus in eine formale State Machine.
    Feature-Flag: use_kernel=True in Config aktiviert den Kernel-Modus.
    """

    def __init__(
        self,
        session_id: str = "",
        checkpoint_manager: CheckpointManager | None = None,
    ) -> None:
        self._session_id = session_id
        self._state = KernelState.IDLE
        self._checkpoint_mgr = checkpoint_manager or CheckpointManager()
        self._completed_nodes: list[str] = []
        self._tool_results: list[ToolResult] = []
        self._error: str | None = None

    @property
    def state(self) -> KernelState:
        return self._state

    @property
    def completed_nodes(self) -> list[str]:
        return list(self._completed_nodes)

    @property
    def tool_results(self) -> list[ToolResult]:
        return list(self._tool_results)

    @property
    def error(self) -> str | None:
        return self._error

    def transition(self, new_state: KernelState) -> None:
        """Validiert und fuehrt einen State-Uebergang durch.

        Raises:
            InvalidTransitionError: Bei ungueltigem Uebergang.
        """
        allowed = _VALID_TRANSITIONS.get(self._state, set())
        if new_state not in allowed:
            raise InvalidTransitionError(
                f"Uebergang von {self._state.value} nach {new_state.value} nicht erlaubt. "
                f"Erlaubt: {[s.value for s in allowed]}"
            )
        old = self._state
        self._state = new_state
        log.debug("kernel_transition", old=old.value, new=new_state.value)

    def checkpoint(self, working_memory_snapshot: dict[str, Any] | None = None) -> Checkpoint:
        """Erstellt einen Checkpoint des aktuellen Zustands."""
        return self._checkpoint_mgr.create_checkpoint(
            session_id=self._session_id,
            kernel_state=self._state,
            working_memory_snapshot=working_memory_snapshot or {},
            completed_nodes=list(self._completed_nodes),
            tool_results=list(self._tool_results),
        )

    def rollback(self, checkpoint_id: str) -> bool:
        """Setzt den Kernel-Zustand auf einen Checkpoint zurueck.

        Returns:
            True bei Erfolg, False wenn Checkpoint nicht gefunden.
        """
        cp = self._checkpoint_mgr.restore_checkpoint(checkpoint_id)
        if cp is None:
            log.warning("rollback_failed", checkpoint_id=checkpoint_id[:8])
            return False

        self._state = cp.kernel_state
        self._completed_nodes = list(cp.completed_nodes)
        self._tool_results = list(cp.tool_results)
        self._error = None
        log.info(
            "kernel_rollback",
            checkpoint_id=checkpoint_id[:8],
            state=self._state.value,
        )
        return True

    async def execute_plan(
        self,
        plan_graph: PlanGraph,
        execute_fn: Any = None,
    ) -> list[ToolResult]:
        """Fuehrt einen Plan-Graph topologisch aus.

        Args:
            plan_graph: Der auszufuehrende Plan-Graph.
            execute_fn: Async Callable(tool_name, params) -> ToolResult.

        Returns:
            Liste aller ToolResults.
        """
        self.transition(KernelState.EXECUTING)
        results: list[ToolResult] = []

        try:
            order = plan_graph.topological_order()
        except Exception as e:
            self._error = str(e)
            self.transition(KernelState.ERROR)
            return results

        completed: set[str] = set()

        for node_id in order:
            node = plan_graph.get_node(node_id)
            if node is None:
                continue

            # Checkpoint vor jedem Node
            self.checkpoint()

            if execute_fn:
                try:
                    result = await execute_fn(node.tool, node.params)
                except Exception as e:
                    result = ToolResult(
                        tool_name=node.tool,
                        content=str(e),
                        is_error=True,
                        error_type=type(e).__name__,
                    )
            else:
                # Dry-run mode
                result = ToolResult(
                    tool_name=node.tool,
                    content=f"Dry-run: {node.tool}({node.params})",
                )

            results.append(result)
            self._tool_results.append(result)

            if result.success:
                completed.add(node_id)
                self._completed_nodes.append(node_id)

        return results

    def reset(self) -> None:
        """Setzt den Kernel in den IDLE-Zustand zurueck."""
        self._state = KernelState.IDLE
        self._completed_nodes.clear()
        self._tool_results.clear()
        self._error = None
