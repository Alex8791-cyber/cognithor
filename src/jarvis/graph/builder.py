"""Graph Builder -- Fluent API zum Erstellen von Graph-Definitionen (v18).

Ermöglicht deklarativen Graphenbau:

    graph = (
        GraphBuilder("customer_support")
        .add_node("classify", classify_intent, node_type=NodeType.ROUTER)
        .add_node("faq", handle_faq)
        .add_node("ticket", create_ticket)
        .add_node("human", human_review, node_type=NodeType.HITL)
        .set_entry("classify")
        .add_edge("classify", "faq", condition="faq")
        .add_edge("classify", "ticket", condition="ticket")
        .add_edge("classify", "human", condition="complex")
        .add_edge("faq", END)
        .add_edge("ticket", "human")
        .add_edge("human", END)
        .build()
    )

Alternativ: Kompakt-Syntax:

    graph = (
        GraphBuilder("pipeline")
        .chain("fetch", "process", "validate", "store")
        .build()
    )
"""

from __future__ import annotations

from typing import Any, Callable, Awaitable

from jarvis.graph.types import (
    Edge,
    EdgeType,
    END,
    GraphDefinition,
    GraphState,
    Node,
    NodeType,
    START,
)


class GraphBuilder:
    """Fluent Builder für GraphDefinitions."""

    def __init__(self, name: str = "", description: str = "") -> None:
        self._graph = GraphDefinition(name=name, description=description)
        self._built = False

    # ── Node Methods ─────────────────────────────────────────────

    def add_node(
        self,
        name: str,
        handler: Callable | None = None,
        *,
        node_type: NodeType = NodeType.FUNCTION,
        description: str = "",
        retry_count: int = 0,
        retry_delay: float = 1.0,
        timeout: float = 300.0,
        checkpoint_before: bool = False,
        checkpoint_after: bool = False,
        config: dict[str, Any] | None = None,
    ) -> GraphBuilder:
        """Fügt einen Node hinzu."""
        node = Node(
            name=name,
            node_type=node_type,
            handler=handler,
            description=description,
            retry_count=retry_count,
            retry_delay_seconds=retry_delay,
            timeout_seconds=timeout,
            checkpoint_before=checkpoint_before,
            checkpoint_after=checkpoint_after,
            config=config or {},
        )
        self._graph.add_node(node)
        return self

    def add_router(
        self,
        name: str,
        handler: Callable,
        *,
        description: str = "",
        timeout: float = 60.0,
    ) -> GraphBuilder:
        """Fügt einen Router-Node hinzu (Shortcut)."""
        return self.add_node(
            name, handler,
            node_type=NodeType.ROUTER,
            description=description or f"Router: {name}",
            timeout=timeout,
        )

    def add_hitl(
        self,
        name: str,
        handler: Callable | None = None,
        *,
        description: str = "",
    ) -> GraphBuilder:
        """Fügt einen Human-in-the-Loop-Node hinzu."""
        return self.add_node(
            name, handler,
            node_type=NodeType.HITL,
            description=description or f"HITL: {name}",
            checkpoint_before=True,
        )

    def add_passthrough(self, name: str) -> GraphBuilder:
        """Fügt einen No-Op-Node hinzu (für Merge-Punkte)."""
        return self.add_node(name, node_type=NodeType.PASSTHROUGH)

    # ── Edge Methods ─────────────────────────────────────────────

    def add_edge(
        self,
        source: str,
        target: str,
        *,
        condition: str = "",
        priority: int = 0,
    ) -> GraphBuilder:
        """Fügt eine Kante hinzu."""
        edge_type = EdgeType.CONDITIONAL if condition else EdgeType.DIRECT
        edge = Edge(
            source=source, target=target,
            edge_type=edge_type, condition=condition,
            priority=priority,
        )
        self._graph.add_edge(edge)
        return self

    def add_conditional_edges(
        self,
        source: str,
        mapping: dict[str, str],
        *,
        default: str = "",
    ) -> GraphBuilder:
        """Fügt mehrere konditionale Kanten auf einmal hinzu.

        Args:
            source: Router-Node
            mapping: {condition_value: target_node}
            default: Fallback-Target
        """
        for condition, target in mapping.items():
            self.add_edge(source, target, condition=condition)
        if default:
            self.add_edge(source, default, condition="__default__")
        return self

    # ── Convenience Methods ──────────────────────────────────────

    def chain(self, *node_names: str) -> GraphBuilder:
        """Verkettet Nodes linear (A → B → C → ...).

        Nodes müssen vorher mit add_node() hinzugefügt worden sein,
        oder werden als Passthrough-Nodes erstellt.
        """
        for name in node_names:
            if name not in self._graph.nodes and name != END:
                self.add_passthrough(name)

        for i in range(len(node_names) - 1):
            self.add_edge(node_names[i], node_names[i + 1])

        if not self._graph.entry_point and node_names:
            self._graph.entry_point = node_names[0]

        return self

    def set_entry(self, node_name: str) -> GraphBuilder:
        """Setzt den Entry-Point des Graphen."""
        self._graph.entry_point = node_name
        return self

    def set_metadata(self, key: str, value: Any) -> GraphBuilder:
        """Setzt Metadaten am Graphen."""
        self._graph.metadata[key] = value
        return self

    # ── Build ────────────────────────────────────────────────────

    def build(self) -> GraphDefinition:
        """Erstellt und validiert die GraphDefinition.

        Raises:
            ValueError: Wenn der Graph ungültig ist
        """
        if self._built:
            raise ValueError("GraphBuilder already built -- create a new builder")

        # Auto-Entry: Erster hinzugefügter Node
        if not self._graph.entry_point and self._graph.nodes:
            self._graph.entry_point = next(iter(self._graph.nodes))

        errors = self._graph.validate()
        if errors:
            raise ValueError(f"Invalid graph: {'; '.join(errors)}")

        self._built = True
        return self._graph

    def build_unchecked(self) -> GraphDefinition:
        """Erstellt GraphDefinition ohne Validierung (für Tests)."""
        if not self._graph.entry_point and self._graph.nodes:
            self._graph.entry_point = next(iter(self._graph.nodes))
        self._built = True
        return self._graph

    # ── Inspection ───────────────────────────────────────────────

    @property
    def node_count(self) -> int:
        return len(self._graph.nodes)

    @property
    def edge_count(self) -> int:
        return len(self._graph.edges)


# ── Prebuilt Graph Templates ────────────────────────────────────

def linear_graph(name: str,
                 steps: list[tuple[str, Callable]]) -> GraphDefinition:
    """Erstellt einen linearen Graphen (A → B → C → END).

    Args:
        name: Graph-Name
        steps: Liste von (node_name, handler) Paaren
    """
    builder = GraphBuilder(name)
    for node_name, handler in steps:
        builder.add_node(node_name, handler)

    names = [n for n, _ in steps]
    builder.chain(*names, END)
    return builder.build()


def branch_graph(
    name: str,
    router_name: str,
    router_handler: Callable,
    branches: dict[str, Callable],
    *,
    merge_node: str = "",
    merge_handler: Callable | None = None,
) -> GraphDefinition:
    """Erstellt einen Branching-Graphen (Router → Branches → Optional Merge → END).

    Args:
        name: Graph-Name
        router_name: Name des Router-Nodes
        router_handler: Router-Handler (gibt Branch-Name zurück)
        branches: {branch_name: handler}
        merge_node: Optionaler Merge-Point
        merge_handler: Handler für Merge-Node
    """
    builder = GraphBuilder(name)
    builder.add_router(router_name, router_handler)

    for branch_name, handler in branches.items():
        builder.add_node(branch_name, handler)
        builder.add_edge(router_name, branch_name, condition=branch_name)

        if merge_node:
            builder.add_edge(branch_name, merge_node)
        else:
            builder.add_edge(branch_name, END)

    if merge_node:
        builder.add_node(merge_node, merge_handler)
        builder.add_edge(merge_node, END)

    builder.set_entry(router_name)
    return builder.build()


def loop_graph(
    name: str,
    body_name: str,
    body_handler: Callable,
    condition_name: str,
    condition_handler: Callable,
    *,
    continue_condition: str = "continue",
    exit_condition: str = "exit",
) -> GraphDefinition:
    """Erstellt einen Loop-Graphen (Body → Condition → Body oder END).

    Args:
        name: Graph-Name
        body_name: Loop-Body-Node
        body_handler: Body-Handler
        condition_name: Condition-Check-Node (Router)
        condition_handler: Router der 'continue' oder 'exit' zurückgibt
    """
    builder = GraphBuilder(name)
    builder.add_node(body_name, body_handler)
    builder.add_router(condition_name, condition_handler)
    builder.add_edge(body_name, condition_name)
    builder.add_edge(condition_name, body_name, condition=continue_condition)
    builder.add_edge(condition_name, END, condition=exit_condition)
    builder.set_entry(body_name)
    return builder.build()
