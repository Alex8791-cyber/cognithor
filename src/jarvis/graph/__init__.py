"""Jarvis Graph Orchestrator v18 -- DAG-basierte Workflow-Engine.

LangGraph-inspirierte State-Graph-Engine mit:
  - Conditional Edges + Router-Nodes
  - Parallel Branches
  - Loop-Support mit Cycle-Protection
  - Checkpoint/Resume (HITL)
  - Fluent GraphBuilder API
  - Built-in Node-Handlers
  - Mermaid-Diagramm-Export

Usage:
    from jarvis.graph import GraphBuilder, GraphEngine, GraphState, END, NodeType

    graph = (
        GraphBuilder("my_flow")
        .add_node("step1", my_handler)
        .add_node("step2", my_handler2)
        .add_edge("step1", "step2")
        .add_edge("step2", END)
        .build()
    )

    engine = GraphEngine()
    result = await engine.run(graph, GraphState(data="input"))
"""

from jarvis.graph.types import (
    GRAPH_VERSION,
    START,
    END,
    NodeType,
    EdgeType,
    ExecutionStatus,
    NodeStatus,
    GraphState,
    Node,
    Edge,
    NodeResult,
    Checkpoint,
    ExecutionRecord,
    GraphDefinition,
)
from jarvis.graph.state import StateManager
from jarvis.graph.engine import GraphEngine
from jarvis.graph.builder import (
    GraphBuilder,
    linear_graph,
    branch_graph,
    loop_graph,
)
from jarvis.graph.nodes import (
    llm_node,
    tool_node,
    transform_node,
    condition_node,
    threshold_router,
    key_router,
    delay_node,
    log_node,
    accumulate_node,
    gate_node,
    counter_node,
    set_value_node,
    merge_node,
)

__all__ = [
    # Constants
    "GRAPH_VERSION", "START", "END",
    # Enums
    "NodeType", "EdgeType", "ExecutionStatus", "NodeStatus",
    # Core Types
    "GraphState", "Node", "Edge", "NodeResult",
    "Checkpoint", "ExecutionRecord", "GraphDefinition",
    # Engine & State
    "StateManager", "GraphEngine",
    # Builder
    "GraphBuilder", "linear_graph", "branch_graph", "loop_graph",
    # Built-in Nodes
    "llm_node", "tool_node", "transform_node", "condition_node",
    "threshold_router", "key_router", "delay_node", "log_node",
    "accumulate_node", "gate_node", "counter_node", "set_value_node",
    "merge_node",
]
