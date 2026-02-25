"""Graph Orchestrator Types -- v18.

DAG-basierte Workflow-Engine nach LangGraph-Vorbild.
Unterstützt Conditional Edges, Parallel Branches, Loops,
Checkpoints und Human-in-the-Loop.

Kern-Konzepte:
  - GraphState:  Typisierter Zustand der durch den Graphen fließt
  - Node:        Verarbeitungseinheit (Funktion, LLM, Tool, HITL, Router)
  - Edge:        Verbindung zwischen Nodes (direkt oder konditional)
  - Checkpoint:  Serialisierbarer Snapshot für Pause/Resume
  - Graph:       Container für Nodes + Edges mit Validierung
"""

from __future__ import annotations

import copy
import json
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable


# ── Constants ────────────────────────────────────────────────────

START = "__start__"
END = "__end__"
GRAPH_VERSION = "1.0"


# ── Enums ────────────────────────────────────────────────────────

class NodeType(str, Enum):
    """Typ eines Graph-Knotens."""
    FUNCTION = "function"       # Sync/Async Python-Funktion
    LLM = "llm"                 # LLM-Aufruf
    TOOL = "tool"               # MCP-Tool-Aufruf
    ROUTER = "router"           # Conditional Branching
    HITL = "hitl"               # Human-in-the-Loop (Pause/Resume)
    PARALLEL = "parallel"       # Parallele Ausführung
    SUBGRAPH = "subgraph"       # Verschachtelter Graph
    CHECKPOINT = "checkpoint"   # Expliziter Checkpoint
    PASSTHROUGH = "passthrough" # State durchreichen (No-Op)


class EdgeType(str, Enum):
    """Typ einer Graph-Kante."""
    DIRECT = "direct"           # Immer folgen
    CONDITIONAL = "conditional" # Basierend auf Router-Ergebnis


class ExecutionStatus(str, Enum):
    """Status einer Graph-Ausführung."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"          # HITL oder expliziter Pause
    WAITING = "waiting"        # Wartet auf externe Eingabe
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    TIMED_OUT = "timed_out"


class NodeStatus(str, Enum):
    """Status eines einzelnen Knotens."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# ── Node Handler Type ────────────────────────────────────────────

# Ein Node-Handler nimmt GraphState und gibt (modifizierten) GraphState zurück.
# Für Router: gibt den Namen der nächsten Edge zurück.
NodeHandler = Callable[["GraphState"], Awaitable["GraphState"]]
RouterHandler = Callable[["GraphState"], Awaitable[str]]


# ── GraphState ───────────────────────────────────────────────────

class GraphState:
    """Typisierter Zustand der durch den Graphen fließt.

    Verhält sich wie ein dict mit Attribut-Zugriff.
    Jede Node-Funktion bekommt den State, modifiziert ihn und gibt ihn zurück.

    Beispiel:
        state = GraphState(messages=[], step=0)
        state["messages"].append("Hello")
        state.step = 1
    """

    def __init__(self, **kwargs: Any) -> None:
        self._data: dict[str, Any] = dict(kwargs)
        self._metadata: dict[str, Any] = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "version": GRAPH_VERSION,
        }

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"GraphState has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self._data[name] = value

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def keys(self) -> Any:
        return self._data.keys()

    def values(self) -> Any:
        return self._data.values()

    def items(self) -> Any:
        return self._data.items()

    def update(self, other: dict[str, Any] | GraphState) -> None:
        if isinstance(other, GraphState):
            self._data.update(other._data)
        else:
            self._data.update(other)

    def copy(self) -> GraphState:
        new = GraphState()
        new._data = copy.deepcopy(self._data)
        new._metadata = copy.deepcopy(self._metadata)
        return new

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def to_json(self) -> str:
        return json.dumps(self._data, default=str, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphState:
        return cls(**data)

    def __repr__(self) -> str:
        keys = ", ".join(self._data.keys())
        return f"GraphState({keys})"


# ── Node ─────────────────────────────────────────────────────────

@dataclass
class Node:
    """Ein Knoten im Execution-Graph."""
    name: str
    node_type: NodeType = NodeType.FUNCTION
    handler: NodeHandler | RouterHandler | None = None
    description: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    retry_delay_seconds: float = 1.0
    timeout_seconds: float = 300.0
    checkpoint_before: bool = False
    checkpoint_after: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.node_type.value,
            "description": self.description,
            "retry_count": self.retry_count,
            "timeout_seconds": self.timeout_seconds,
            "checkpoint_before": self.checkpoint_before,
            "checkpoint_after": self.checkpoint_after,
        }


# ── Edge ─────────────────────────────────────────────────────────

@dataclass
class Edge:
    """Eine Kante zwischen zwei Nodes."""
    source: str
    target: str
    edge_type: EdgeType = EdgeType.DIRECT
    condition: str = ""     # Für CONDITIONAL: Wert der mit Router-Output verglichen wird
    priority: int = 0       # Höhere Priorität wird zuerst geprüft
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "source": self.source, "target": self.target,
            "type": self.edge_type.value,
        }
        if self.condition:
            d["condition"] = self.condition
        return d


# ── Node Execution Result ────────────────────────────────────────

@dataclass
class NodeResult:
    """Ergebnis der Ausführung eines einzelnen Knotens."""
    node_name: str
    status: NodeStatus
    state_after: GraphState | None = None
    router_decision: str = ""
    error: str = ""
    duration_ms: int = 0
    retry_attempts: int = 0
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "node": self.node_name, "status": self.status.value,
            "duration_ms": self.duration_ms, "timestamp": self.timestamp,
        }
        if self.error:
            d["error"] = self.error
        if self.router_decision:
            d["router_decision"] = self.router_decision
        if self.retry_attempts:
            d["retries"] = self.retry_attempts
        return d


# ── Checkpoint ───────────────────────────────────────────────────

@dataclass
class Checkpoint:
    """Serialisierbarer Snapshot einer Graph-Ausführung."""
    checkpoint_id: str = ""
    execution_id: str = ""
    graph_name: str = ""
    current_node: str = ""
    state: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    status: ExecutionStatus = ExecutionStatus.PAUSED
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.checkpoint_id:
            self.checkpoint_id = uuid.uuid4().hex[:12]
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "execution_id": self.execution_id,
            "graph_name": self.graph_name,
            "current_node": self.current_node,
            "state": self.state,
            "history": self.history,
            "status": self.status.value,
            "created_at": self.created_at,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Checkpoint:
        return cls(
            checkpoint_id=data.get("checkpoint_id", ""),
            execution_id=data.get("execution_id", ""),
            graph_name=data.get("graph_name", ""),
            current_node=data.get("current_node", ""),
            state=data.get("state", {}),
            history=data.get("history", []),
            status=ExecutionStatus(data.get("status", "paused")),
            created_at=data.get("created_at", ""),
        )

    @classmethod
    def from_json(cls, raw: str) -> Checkpoint:
        return cls.from_dict(json.loads(raw))


# ── Execution Record ────────────────────────────────────────────

@dataclass
class ExecutionRecord:
    """Vollständiger Record einer Graph-Ausführung."""
    execution_id: str = ""
    graph_name: str = ""
    status: ExecutionStatus = ExecutionStatus.PENDING
    initial_state: dict[str, Any] = field(default_factory=dict)
    final_state: dict[str, Any] = field(default_factory=dict)
    node_results: list[NodeResult] = field(default_factory=list)
    checkpoints: list[str] = field(default_factory=list)  # checkpoint_ids
    started_at: str = ""
    completed_at: str = ""
    total_duration_ms: int = 0
    error: str = ""

    def __post_init__(self) -> None:
        if not self.execution_id:
            self.execution_id = uuid.uuid4().hex[:12]

    @property
    def node_count(self) -> int:
        return len(self.node_results)

    @property
    def success_rate(self) -> float:
        if not self.node_results:
            return 0.0
        ok = sum(1 for r in self.node_results if r.status == NodeStatus.COMPLETED)
        return ok / len(self.node_results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "graph_name": self.graph_name,
            "status": self.status.value,
            "nodes_executed": self.node_count,
            "success_rate": round(self.success_rate, 3),
            "total_duration_ms": self.total_duration_ms,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }


# ── Graph Definition ─────────────────────────────────────────────

class GraphDefinition:
    """Container für Nodes und Edges mit Validierung.

    Wird vom GraphBuilder erzeugt und vom GraphEngine ausgeführt.
    """

    def __init__(self, name: str = "", description: str = "") -> None:
        self.name = name or f"graph_{uuid.uuid4().hex[:8]}"
        self.description = description
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.entry_point: str = ""
        self.metadata: dict[str, Any] = {
            "version": GRAPH_VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def add_node(self, node: Node) -> None:
        self.nodes[node.name] = node

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)

    def get_node(self, name: str) -> Node | None:
        return self.nodes.get(name)

    def get_outgoing_edges(self, node_name: str) -> list[Edge]:
        return [e for e in self.edges if e.source == node_name]

    def get_incoming_edges(self, node_name: str) -> list[Edge]:
        return [e for e in self.edges if e.target == node_name]

    def get_successors(self, node_name: str) -> list[str]:
        return [e.target for e in self.get_outgoing_edges(node_name)]

    def get_predecessors(self, node_name: str) -> list[str]:
        return [e.source for e in self.get_incoming_edges(node_name)]

    # ── Validation ───────────────────────────────────────────────

    def validate(self) -> list[str]:
        """Validiert den Graphen und gibt Fehler zurück."""
        errors: list[str] = []

        if not self.nodes:
            errors.append("Graph has no nodes")
            return errors

        # Entry point
        if not self.entry_point:
            errors.append("No entry point defined")
        elif self.entry_point not in self.nodes:
            errors.append(f"Entry point '{self.entry_point}' not found in nodes")

        # Edge-Referenzen prüfen
        for edge in self.edges:
            if edge.source != START and edge.source not in self.nodes:
                errors.append(f"Edge source '{edge.source}' not found")
            if edge.target != END and edge.target not in self.nodes:
                errors.append(f"Edge target '{edge.target}' not found")

        # Erreichbarkeit: Jeder Node muss von entry erreichbar sein
        if self.entry_point and self.entry_point in self.nodes:
            reachable = self._find_reachable(self.entry_point)
            for name in self.nodes:
                if name not in reachable and name != self.entry_point:
                    errors.append(f"Node '{name}' not reachable from entry point")

        # Router-Nodes müssen conditional edges haben
        for name, node in self.nodes.items():
            if node.node_type == NodeType.ROUTER:
                cond_edges = [e for e in self.get_outgoing_edges(name)
                              if e.edge_type == EdgeType.CONDITIONAL]
                if not cond_edges:
                    errors.append(f"Router node '{name}' has no conditional edges")

        return errors

    def _find_reachable(self, start: str) -> set[str]:
        """BFS um erreichbare Nodes zu finden."""
        visited: set[str] = set()
        queue = [start]
        while queue:
            current = queue.pop(0)
            if current in visited or current == END:
                continue
            visited.add(current)
            for successor in self.get_successors(current):
                if successor not in visited:
                    queue.append(successor)
        return visited

    def detect_cycles(self) -> list[list[str]]:
        """Erkennt Zyklen im Graphen (erlaubt für Loops, aber gezählt)."""
        cycles: list[list[str]] = []
        visited: set[str] = set()
        rec_stack: set[str] = set()
        path: list[str] = []

        def dfs(node: str) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for successor in self.get_successors(node):
                if successor == END:
                    continue
                if successor not in visited:
                    dfs(successor)
                elif successor in rec_stack:
                    idx = path.index(successor)
                    cycles.append(path[idx:] + [successor])

            path.pop()
            rec_stack.discard(node)

        for name in self.nodes:
            if name not in visited:
                dfs(name)

        return cycles

    def topological_sort(self) -> list[str] | None:
        """Topologische Sortierung (None wenn Zyklen vorhanden)."""
        in_degree: dict[str, int] = {name: 0 for name in self.nodes}
        for edge in self.edges:
            if edge.target in in_degree:
                in_degree[edge.target] += 1

        queue = [n for n, d in in_degree.items() if d == 0]
        result: list[str] = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for successor in self.get_successors(node):
                if successor in in_degree:
                    in_degree[successor] -= 1
                    if in_degree[successor] == 0:
                        queue.append(successor)

        if len(result) != len(self.nodes):
            return None  # Zyklen vorhanden
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "entry_point": self.entry_point,
            "nodes": {n: nd.to_dict() for n, nd in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "cycles": len(self.detect_cycles()),
        }

    def to_mermaid(self) -> str:
        """Generiert Mermaid-Diagramm des Graphen."""
        lines = ["graph TD"]
        for name, node in self.nodes.items():
            shape = {
                NodeType.ROUTER: f"{{{{{name}}}}}",
                NodeType.HITL: f"[/{name}/]",
                NodeType.PARALLEL: f"[[{name}]]",
                NodeType.SUBGRAPH: f"[({name})]",
            }.get(node.node_type, f"[{name}]")
            lines.append(f"    {name}{shape}")

        for edge in self.edges:
            src = edge.source if edge.source != START else "START((Start))"
            tgt = edge.target if edge.target != END else "END((End))"
            if edge.condition:
                lines.append(f"    {src} -->|{edge.condition}| {tgt}")
            else:
                lines.append(f"    {src} --> {tgt}")

        return "\n".join(lines)
