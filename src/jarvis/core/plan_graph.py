"""Plan-Graph (DAG) fuer deterministische Execution."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from jarvis.models import ActionPlan, PlanNode, RiskLevel
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class CycleError(Exception):
    """Zyklus im Plan-Graph erkannt."""


class PlanGraph:
    """DAG aus PlanNodes mit topologischer Sortierung."""

    def __init__(self) -> None:
        self._nodes: dict[str, PlanNode] = {}
        self._edges: dict[str, set[str]] = defaultdict(set)  # node_id -> set of dependent node_ids
        self._reverse_edges: dict[str, set[str]] = defaultdict(set)  # node_id -> set of dependency node_ids

    def add_node(self, node: PlanNode) -> None:
        """Fuegt einen Knoten zum Graph hinzu."""
        self._nodes[node.id] = node
        for dep_id in node.depends_on:
            self.add_edge(dep_id, node.id)

    def add_edge(self, from_id: str, to_id: str) -> None:
        """Fuegt eine Kante hinzu: from_id muss vor to_id fertig sein."""
        self._edges[from_id].add(to_id)
        self._reverse_edges[to_id].add(from_id)

    def topological_order(self) -> list[str]:
        """Gibt die Knoten in topologischer Reihenfolge zurueck.

        Raises:
            CycleError: Wenn der Graph Zyklen enthaelt.
        """
        in_degree: dict[str, int] = {nid: 0 for nid in self._nodes}
        for nid in self._nodes:
            for dep in self._reverse_edges.get(nid, set()):
                if dep in self._nodes:
                    in_degree[nid] = in_degree.get(nid, 0) + 1

        queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
        result: list[str] = []

        while queue:
            nid = queue.popleft()
            result.append(nid)
            for dependent in self._edges.get(nid, set()):
                if dependent in in_degree:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        if len(result) != len(self._nodes):
            raise CycleError(
                f"Zyklus erkannt: {len(result)} von {len(self._nodes)} Knoten erreichbar"
            )

        return result

    def get_ready_nodes(self, completed_ids: set[str]) -> list[str]:
        """Gibt Knoten zurueck deren Dependencies erfuellt sind."""
        ready: list[str] = []
        for nid, node in self._nodes.items():
            if nid in completed_ids:
                continue
            deps = self._reverse_edges.get(nid, set())
            # Only consider deps that are actual nodes in the graph
            relevant_deps = deps & set(self._nodes.keys())
            if relevant_deps.issubset(completed_ids):
                ready.append(nid)
        return ready

    def validate(self) -> list[str]:
        """Validiert den Graph. Gibt Liste von Problemen zurueck (leer = OK)."""
        problems: list[str] = []

        # Zyklus-Check
        try:
            self.topological_order()
        except CycleError as e:
            problems.append(str(e))

        # Erreichbarkeits-Check: Referenzierte Dependencies muessen existieren
        for nid, node in self._nodes.items():
            for dep_id in node.depends_on:
                if dep_id not in self._nodes:
                    problems.append(f"Node {nid} referenziert unbekannte Dependency {dep_id}")

        return problems

    def get_node(self, node_id: str) -> PlanNode | None:
        """Holt einen Knoten nach ID."""
        return self._nodes.get(node_id)

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def nodes(self) -> list[PlanNode]:
        return list(self._nodes.values())

    @classmethod
    def from_action_plan(cls, plan: ActionPlan) -> PlanGraph:
        """Factory: Erstellt PlanGraph aus bestehendem ActionPlan."""
        graph = cls()
        # Map step-index to node-id
        index_to_id: dict[int, str] = {}

        for i, step in enumerate(plan.steps):
            node = PlanNode(
                tool=step.tool,
                params=step.params,
                depends_on=[],
                estimated_risk=step.risk_estimate,
            )
            index_to_id[i] = node.id
            graph._nodes[node.id] = node

        # Resolve dependencies (step indices -> node ids)
        for i, step in enumerate(plan.steps):
            node_id = index_to_id[i]
            node = graph._nodes[node_id]
            dep_ids = []
            for dep_idx in step.depends_on:
                if dep_idx in index_to_id:
                    dep_ids.append(index_to_id[dep_idx])
                    graph.add_edge(index_to_id[dep_idx], node_id)
            # Update node with resolved dependencies
            graph._nodes[node_id] = PlanNode(
                id=node.id,
                tool=node.tool,
                params=node.params,
                depends_on=dep_ids,
                estimated_risk=node.estimated_risk,
            )

        return graph
