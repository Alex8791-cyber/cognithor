"""Graph Engine — Kern-Ausführungslogik für Graph Orchestrator v18.

Führt einen GraphDefinition aus:
  - Traversiert Nodes entlang Edges
  - Conditional Routing basierend auf Router-Output
  - Parallele Ausführung von Branches
  - Loop-Erkennung mit max_iterations
  - Checkpoint-Erstellung vor/nach Nodes
  - HITL-Pause/Resume
  - Timeout und Retry pro Node
  - Vollständiger Audit-Trail (ExecutionRecord)

Execution-Flow:
  1. Start mit initial GraphState
  2. Entry-Node ausführen
  3. Outgoing Edges evaluieren:
     a. DIRECT → nächsten Node ausführen
     b. CONDITIONAL → Router-Entscheidung folgen
  4. HITL-Node → Pausieren, Checkpoint erstellen
  5. END erreicht → Execution abgeschlossen
"""

from __future__ import annotations

import asyncio
import copy
import time
from typing import Any, AsyncIterator

from jarvis.graph.state import StateManager
from jarvis.graph.types import (
    Checkpoint,
    Edge,
    EdgeType,
    END,
    ExecutionRecord,
    ExecutionStatus,
    GraphDefinition,
    GraphState,
    Node,
    NodeResult,
    NodeStatus,
    NodeType,
    START,
)
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# Default-Limits
MAX_ITERATIONS = 50
MAX_NODES_PER_EXECUTION = 200


class GraphEngine:
    """Führt Graph-Definitionen aus.

    Usage:
        engine = GraphEngine()
        graph = GraphBuilder("my_flow").add_node(...).add_edge(...).build()
        state = GraphState(messages=[], step=0)
        result = await engine.run(graph, state)
    """

    def __init__(self, state_manager: StateManager | None = None,
                 max_iterations: int = MAX_ITERATIONS,
                 max_nodes: int = MAX_NODES_PER_EXECUTION) -> None:
        self._state_mgr = state_manager or StateManager()
        self._max_iterations = max_iterations
        self._max_nodes = max_nodes
        self._running_executions: dict[str, ExecutionRecord] = {}
        self._execution_count = 0
        self._total_nodes_executed = 0

    # ── Main Execution ───────────────────────────────────────────

    async def run(self, graph: GraphDefinition, initial_state: GraphState,
                  *, execution_id: str = "") -> ExecutionRecord:
        """Führt einen Graphen vollständig aus.

        Args:
            graph: Graph-Definition
            initial_state: Anfangszustand
            execution_id: Optional, für Resume

        Returns:
            ExecutionRecord mit allen Ergebnissen
        """
        # Validierung
        errors = graph.validate()
        if errors:
            record = ExecutionRecord(
                graph_name=graph.name,
                status=ExecutionStatus.FAILED,
                error=f"Validation errors: {'; '.join(errors)}",
            )
            return record

        # Execution-Record
        record = self._state_mgr.create_execution(graph.name, initial_state)
        if execution_id:
            record.execution_id = execution_id
        self._running_executions[record.execution_id] = record
        self._execution_count += 1

        state = initial_state.copy()
        current_node = graph.entry_point
        visit_count: dict[str, int] = {}
        nodes_executed = 0

        try:
            while current_node and current_node != END:
                # Loop-Protection
                visit_count[current_node] = visit_count.get(current_node, 0) + 1
                if visit_count[current_node] > self._max_iterations:
                    record.status = ExecutionStatus.FAILED
                    record.error = f"Max iterations ({self._max_iterations}) exceeded at node '{current_node}'"
                    break

                # Global node limit
                nodes_executed += 1
                if nodes_executed > self._max_nodes:
                    record.status = ExecutionStatus.FAILED
                    record.error = f"Max nodes ({self._max_nodes}) exceeded"
                    break

                node = graph.get_node(current_node)
                if node is None:
                    record.status = ExecutionStatus.FAILED
                    record.error = f"Node '{current_node}' not found"
                    break

                # Checkpoint before
                if node.checkpoint_before:
                    self._state_mgr.create_checkpoint(
                        record.execution_id, graph.name,
                        current_node, state, record.node_results,
                    )

                # HITL: Pausieren
                if node.node_type == NodeType.HITL:
                    cp = self._state_mgr.create_checkpoint(
                        record.execution_id, graph.name,
                        current_node, state, record.node_results,
                        status=ExecutionStatus.PAUSED,
                    )
                    record.status = ExecutionStatus.PAUSED
                    record.checkpoints.append(cp.checkpoint_id)

                    # Handler trotzdem ausführen (kann State modifizieren)
                    if node.handler:
                        result = await self._execute_node(node, state)
                        record.node_results.append(result)
                        if result.status == NodeStatus.COMPLETED and result.state_after:
                            state = result.state_after
                    break  # Pause

                # PARALLEL Node: execute all outgoing targets concurrently
                if node.node_type == NodeType.PARALLEL:
                    outgoing = graph.get_outgoing_edges(current_node)
                    parallel_targets = [e.target for e in outgoing if e.target != END]
                    parallel_nodes = [graph.get_node(t) for t in parallel_targets]
                    parallel_nodes = [n for n in parallel_nodes if n is not None]
                    if parallel_nodes:
                        parallel_results = await self._execute_parallel(parallel_nodes, state)
                        for pr in parallel_results:
                            record.node_results.append(pr)
                            self._total_nodes_executed += 1
                            if pr.status == NodeStatus.FAILED:
                                record.status = ExecutionStatus.FAILED
                                record.error = f"Parallel node '{pr.node_name}' failed: {pr.error}"
                                break
                            if pr.state_after:
                                state = pr.state_after
                        if record.status == ExecutionStatus.FAILED:
                            break
                        # Continue from the last parallel target's outgoing edges
                        if parallel_targets:
                            last_target = parallel_targets[-1]
                            last_result = parallel_results[-1] if parallel_results else result
                            current_node = self._resolve_next_node(
                                graph, last_target, last_result, state
                            )
                        else:
                            current_node = END
                    else:
                        current_node = END
                    continue

                # Node ausführen
                result = await self._execute_node(node, state)
                record.node_results.append(result)
                self._total_nodes_executed += 1

                if result.status == NodeStatus.FAILED:
                    record.status = ExecutionStatus.FAILED
                    record.error = f"Node '{current_node}' failed: {result.error}"
                    break

                # State updaten
                if result.state_after:
                    state = result.state_after

                # Checkpoint after
                if node.checkpoint_after:
                    self._state_mgr.create_checkpoint(
                        record.execution_id, graph.name,
                        current_node, state, record.node_results,
                    )

                # Nächsten Node bestimmen
                current_node = self._resolve_next_node(
                    graph, current_node, result, state
                )

            # Abschluss
            if record.status == ExecutionStatus.RUNNING:
                record.status = ExecutionStatus.COMPLETED

        except asyncio.CancelledError:
            record.status = ExecutionStatus.CANCELED
        except Exception as exc:
            record.status = ExecutionStatus.FAILED
            record.error = str(exc)
            log.error("graph_execution_error", graph=graph.name, error=str(exc))

        record.final_state = state.to_dict()
        record.completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        record.total_duration_ms = sum(r.duration_ms for r in record.node_results)
        self._running_executions.pop(record.execution_id, None)
        self._state_mgr.update_execution(record)
        return record

    # ── Streaming Execution ──────────────────────────────────────

    async def run_stream(self, graph: GraphDefinition,
                         initial_state: GraphState) -> AsyncIterator[NodeResult]:
        """Führt Graph aus und yielded NodeResults in Echtzeit."""
        errors = graph.validate()
        if errors:
            yield NodeResult(
                node_name="__validation__", status=NodeStatus.FAILED,
                error=f"Validation: {'; '.join(errors)}",
            )
            return

        record = self._state_mgr.create_execution(graph.name, initial_state)
        state = initial_state.copy()
        current_node = graph.entry_point
        visit_count: dict[str, int] = {}
        nodes_executed = 0

        while current_node and current_node != END:
            visit_count[current_node] = visit_count.get(current_node, 0) + 1
            if visit_count[current_node] > self._max_iterations:
                yield NodeResult(node_name=current_node, status=NodeStatus.FAILED,
                                 error="Max iterations exceeded")
                return

            nodes_executed += 1
            if nodes_executed > self._max_nodes:
                yield NodeResult(node_name=current_node, status=NodeStatus.FAILED,
                                 error="Max nodes exceeded")
                return

            node = graph.get_node(current_node)
            if node is None:
                yield NodeResult(node_name=current_node, status=NodeStatus.FAILED,
                                 error="Node not found")
                return

            if node.node_type == NodeType.HITL:
                self._state_mgr.create_checkpoint(
                    record.execution_id, graph.name,
                    current_node, state, record.node_results,
                    status=ExecutionStatus.PAUSED,
                )
                if node.handler:
                    result = await self._execute_node(node, state)
                    if result.state_after:
                        state = result.state_after
                    yield result
                yield NodeResult(node_name=current_node, status=NodeStatus.COMPLETED,
                                 router_decision="__paused__")
                return

            result = await self._execute_node(node, state)
            record.node_results.append(result)
            yield result

            if result.status == NodeStatus.FAILED:
                return

            if result.state_after:
                state = result.state_after

            current_node = self._resolve_next_node(graph, current_node, result, state)

        record.status = ExecutionStatus.COMPLETED
        record.final_state = state.to_dict()
        self._state_mgr.update_execution(record)

    # ── Resume from Checkpoint ───────────────────────────────────

    async def resume(self, graph: GraphDefinition,
                     checkpoint_id: str = "",
                     execution_id: str = "",
                     resume_input: dict[str, Any] | None = None) -> ExecutionRecord:
        """Setzt eine pausierte Execution fort.

        Args:
            graph: Gleiche Graph-Definition
            checkpoint_id: Spezifischer Checkpoint oder ""
            execution_id: Execution-ID (nutzt letzten Checkpoint)
            resume_input: Optionaler Input der in den State gemergt wird
        """
        # Checkpoint laden
        if checkpoint_id:
            state, current_node = self._state_mgr.restore_state(checkpoint_id)
        elif execution_id:
            state, current_node = self._state_mgr.restore_from_latest(execution_id)
        else:
            return ExecutionRecord(
                graph_name=graph.name,
                status=ExecutionStatus.FAILED,
                error="checkpoint_id or execution_id required",
            )

        if state is None:
            return ExecutionRecord(
                graph_name=graph.name,
                status=ExecutionStatus.FAILED,
                error="Checkpoint not found",
            )

        # Resume-Input einmergen
        if resume_input:
            state.update(resume_input)

        # Nächsten Node nach HITL bestimmen
        node = graph.get_node(current_node)
        if node and node.node_type == NodeType.HITL:
            next_node = self._resolve_next_node(
                graph, current_node,
                NodeResult(node_name=current_node, status=NodeStatus.COMPLETED),
                state,
            )
            if next_node:
                current_node = next_node

        # Neuen mini-Graph ab current_node ausführen
        # Kopie erstellen um shared GraphDefinition nicht zu mutieren
        local_graph = copy.copy(graph)
        local_graph.entry_point = current_node

        # Run without validation (partial graph)
        record = self._state_mgr.create_execution(local_graph.name, state)
        if execution_id:
            record.execution_id = execution_id
        self._running_executions[record.execution_id] = record
        self._execution_count += 1

        visit_count: dict[str, int] = {}
        nodes_executed = 0

        try:
            while current_node and current_node != END:
                visit_count[current_node] = visit_count.get(current_node, 0) + 1
                if visit_count[current_node] > self._max_iterations:
                    record.status = ExecutionStatus.FAILED
                    record.error = f"Max iterations exceeded at '{current_node}'"
                    break

                nodes_executed += 1
                if nodes_executed > self._max_nodes:
                    record.status = ExecutionStatus.FAILED
                    record.error = "Max nodes exceeded"
                    break

                node = local_graph.get_node(current_node)
                if node is None:
                    record.status = ExecutionStatus.FAILED
                    record.error = f"Node '{current_node}' not found"
                    break

                if node.node_type == NodeType.HITL:
                    cp = self._state_mgr.create_checkpoint(
                        record.execution_id, local_graph.name,
                        current_node, state, record.node_results,
                        status=ExecutionStatus.PAUSED,
                    )
                    record.status = ExecutionStatus.PAUSED
                    record.checkpoints.append(cp.checkpoint_id)
                    if node.handler:
                        result = await self._execute_node(node, state)
                        record.node_results.append(result)
                        if result.status == NodeStatus.COMPLETED and result.state_after:
                            state = result.state_after
                    break

                result = await self._execute_node(node, state)
                record.node_results.append(result)
                self._total_nodes_executed += 1

                if result.status == NodeStatus.FAILED:
                    record.status = ExecutionStatus.FAILED
                    record.error = f"Node '{current_node}' failed: {result.error}"
                    break

                if result.state_after:
                    state = result.state_after

                current_node = self._resolve_next_node(local_graph, current_node, result, state)

            if record.status == ExecutionStatus.RUNNING:
                record.status = ExecutionStatus.COMPLETED

        except Exception as exc:
            record.status = ExecutionStatus.FAILED
            record.error = str(exc)

        record.final_state = state.to_dict()
        record.completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        record.total_duration_ms = sum(r.duration_ms for r in record.node_results)
        self._running_executions.pop(record.execution_id, None)
        self._state_mgr.update_execution(record)
        return record

    # ── Parallel Execution ───────────────────────────────────────

    async def _execute_parallel(self, nodes: list[Node],
                                 state: GraphState) -> list[NodeResult]:
        """Führt mehrere Nodes parallel aus."""
        tasks = [self._execute_node(node, state.copy()) for node in nodes]
        return list(await asyncio.gather(*tasks))

    # ── Node Execution ───────────────────────────────────────────

    async def _execute_node(self, node: Node, state: GraphState) -> NodeResult:
        """Führt einen einzelnen Node aus (mit Retry + Timeout)."""
        start = time.monotonic()
        last_error = ""

        for attempt in range(node.retry_count + 1):
            try:
                if node.node_type == NodeType.PASSTHROUGH:
                    return NodeResult(
                        node_name=node.name, status=NodeStatus.COMPLETED,
                        state_after=state,
                        duration_ms=int((time.monotonic() - start) * 1000),
                    )

                if node.handler is None:
                    return NodeResult(
                        node_name=node.name, status=NodeStatus.COMPLETED,
                        state_after=state,
                        duration_ms=int((time.monotonic() - start) * 1000),
                    )

                # Timeout
                result_state = await asyncio.wait_for(
                    node.handler(state),
                    timeout=node.timeout_seconds,
                )

                # Router: Ergebnis ist ein String (Edge-Condition)
                if node.node_type == NodeType.ROUTER:
                    if isinstance(result_state, str):
                        return NodeResult(
                            node_name=node.name, status=NodeStatus.COMPLETED,
                            state_after=state, router_decision=result_state,
                            duration_ms=int((time.monotonic() - start) * 1000),
                            retry_attempts=attempt,
                        )
                    elif isinstance(result_state, GraphState):
                        # Router kann auch State modifizieren und Decision im State hinterlegen
                        decision = result_state.get("__router_decision__", "")
                        return NodeResult(
                            node_name=node.name, status=NodeStatus.COMPLETED,
                            state_after=result_state, router_decision=decision,
                            duration_ms=int((time.monotonic() - start) * 1000),
                            retry_attempts=attempt,
                        )

                # Normal: Ergebnis ist GraphState
                if isinstance(result_state, GraphState):
                    return NodeResult(
                        node_name=node.name, status=NodeStatus.COMPLETED,
                        state_after=result_state,
                        duration_ms=int((time.monotonic() - start) * 1000),
                        retry_attempts=attempt,
                    )
                elif isinstance(result_state, dict):
                    new_state = state.copy()
                    new_state.update(result_state)
                    return NodeResult(
                        node_name=node.name, status=NodeStatus.COMPLETED,
                        state_after=new_state,
                        duration_ms=int((time.monotonic() - start) * 1000),
                        retry_attempts=attempt,
                    )
                else:
                    return NodeResult(
                        node_name=node.name, status=NodeStatus.COMPLETED,
                        state_after=state,
                        duration_ms=int((time.monotonic() - start) * 1000),
                        retry_attempts=attempt,
                    )

            except asyncio.TimeoutError:
                last_error = f"Timeout after {node.timeout_seconds}s"
            except Exception as exc:
                last_error = str(exc)

            # Retry delay
            if attempt < node.retry_count:
                await asyncio.sleep(node.retry_delay_seconds)

        return NodeResult(
            node_name=node.name, status=NodeStatus.FAILED,
            error=last_error,
            duration_ms=int((time.monotonic() - start) * 1000),
            retry_attempts=node.retry_count,
        )

    # ── Edge Resolution ──────────────────────────────────────────

    def _resolve_next_node(self, graph: GraphDefinition, current: str,
                           result: NodeResult, state: GraphState) -> str:
        """Bestimmt den nächsten Node basierend auf Edges."""
        outgoing = graph.get_outgoing_edges(current)
        if not outgoing:
            return END

        # Sortiert nach Priorität (höher zuerst)
        outgoing.sort(key=lambda e: e.priority, reverse=True)

        # Router-Decision
        if result.router_decision:
            for edge in outgoing:
                if edge.edge_type == EdgeType.CONDITIONAL:
                    if edge.condition == result.router_decision:
                        return edge.target
            # Fallback: Default-Edge (ohne Condition)
            for edge in outgoing:
                if edge.edge_type == EdgeType.DIRECT:
                    return edge.target
            # Kein Match → erster CONDITIONAL als Fallback
            for edge in outgoing:
                if edge.condition == "__default__" or edge.condition == "*":
                    return edge.target
            return END

        # Direct edges
        direct = [e for e in outgoing if e.edge_type == EdgeType.DIRECT]
        if direct:
            return direct[0].target

        return END

    # ── Execution Management ─────────────────────────────────────

    async def cancel(self, execution_id: str) -> bool:
        """Bricht eine laufende Execution ab."""
        record = self._state_mgr.get_execution(execution_id)
        if record and record.status in (ExecutionStatus.RUNNING, ExecutionStatus.PAUSED):
            record.status = ExecutionStatus.CANCELED
            record.completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self._state_mgr.update_execution(record)
            return True
        return False

    def get_execution(self, execution_id: str) -> ExecutionRecord | None:
        return self._state_mgr.get_execution(execution_id)

    def list_executions(self, **kwargs: Any) -> list[ExecutionRecord]:
        return self._state_mgr.list_executions(**kwargs)

    # ── Stats ────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        return {
            "total_executions": self._execution_count,
            "total_nodes_executed": self._total_nodes_executed,
            "running": len(self._running_executions),
            "max_iterations": self._max_iterations,
            "max_nodes": self._max_nodes,
            "state_manager": self._state_mgr.stats(),
        }
