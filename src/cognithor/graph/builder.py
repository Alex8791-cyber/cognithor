"""Graph Builder -- Fluent API for creating graph definitions (v18).

Enables declarative graph construction:

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

Alternative: compact syntax:

    graph = (
        GraphBuilder("pipeline")
        .chain("fetch", "process", "validate", "store")
        .build()
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cognithor.graph.types import (
    END,
    Edge,
    EdgeType,
    GraphDefinition,
    Node,
    NodeType,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class GraphBuilder:
    """Fluent builder for GraphDefinitions."""

    def __init__(self, name: str = "", description: str = "") -> None:
        self._graph = GraphDefinition(name=name, description=description)
        self._built = False

    # ── Node Methods ─────────────────────────────────────────────

    def add_node(
        self,
        name: str,
        handler: Callable | None = None,  # type: ignore[type-arg]
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
        """Adds a node."""
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
        handler: Callable[..., Any],
        *,
        description: str = "",
        timeout: float = 60.0,
    ) -> GraphBuilder:
        """Adds a router node (shortcut)."""
        return self.add_node(
            name,
            handler,
            node_type=NodeType.ROUTER,
            description=description or f"Router: {name}",
            timeout=timeout,
        )

    def add_hitl(
        self,
        name: str,
        handler: Callable | None = None,  # type: ignore[type-arg]
        *,
        description: str = "",
    ) -> GraphBuilder:
        """Adds a human-in-the-loop node."""
        return self.add_node(
            name,
            handler,
            node_type=NodeType.HITL,
            description=description or f"HITL: {name}",
            checkpoint_before=True,
        )

    def add_passthrough(self, name: str) -> GraphBuilder:
        """Adds a no-op node (for merge points)."""
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
        """Adds an edge."""
        edge_type = EdgeType.CONDITIONAL if condition else EdgeType.DIRECT
        edge = Edge(
            source=source,
            target=target,
            edge_type=edge_type,
            condition=condition,
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
        """Adds multiple conditional edges at once.

        Args:
            source: Router node
            mapping: {condition_value: target_node}
            default: Fallback target
        """
        for condition, target in mapping.items():
            self.add_edge(source, target, condition=condition)
        if default:
            self.add_edge(source, default, condition="__default__")
        return self

    # ── Convenience Methods ──────────────────────────────────────

    def chain(self, *node_names: str) -> GraphBuilder:
        """Chains nodes linearly (A -> B -> C -> ...).

        Nodes must have been added with add_node() beforehand,
        or will be created as passthrough nodes.
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
        """Sets the entry point of the graph."""
        self._graph.entry_point = node_name
        return self

    def set_metadata(self, key: str, value: Any) -> GraphBuilder:
        """Sets metadata on the graph."""
        self._graph.metadata[key] = value
        return self

    # ── Build ────────────────────────────────────────────────────

    def build(self) -> GraphDefinition:
        """Creates and validates the GraphDefinition.

        Raises:
            ValueError: If the graph is invalid
        """
        if self._built:
            raise ValueError("GraphBuilder already built -- create a new builder")

        # Auto-entry: first added node
        if not self._graph.entry_point and self._graph.nodes:
            self._graph.entry_point = next(iter(self._graph.nodes))

        errors = self._graph.validate()
        if errors:
            raise ValueError(f"Invalid graph: {'; '.join(errors)}")

        self._built = True
        return self._graph

    def build_unchecked(self) -> GraphDefinition:
        """Creates GraphDefinition without validation (for tests)."""
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


# ── Prebuilt graph templates ────────────────────────────────────


def linear_graph(name: str, steps: list[tuple[str, Callable[..., Any]]]) -> GraphDefinition:
    """Creates a linear graph (A -> B -> C -> END).

    Args:
        name: Graph name
        steps: List of (node_name, handler) pairs
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
    router_handler: Callable[..., Any],
    branches: dict[str, Callable[..., Any]],
    *,
    merge_node: str = "",
    merge_handler: Callable | None = None,  # type: ignore[type-arg]
) -> GraphDefinition:
    """Creates a branching graph (Router -> Branches -> Optional Merge -> END).

    Args:
        name: Graph name
        router_name: Name of the router node
        router_handler: Router handler (returns branch name)
        branches: {branch_name: handler}
        merge_node: Optional merge point
        merge_handler: Handler for merge node
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
    body_handler: Callable[..., Any],
    condition_name: str,
    condition_handler: Callable[..., Any],
    *,
    continue_condition: str = "continue",
    exit_condition: str = "exit",
) -> GraphDefinition:
    """Creates a loop graph (Body -> Condition -> Body or END).

    Args:
        name: Graph name
        body_name: Loop body node
        body_handler: Body handler
        condition_name: Condition check node (router)
        condition_handler: Router that returns 'continue' or 'exit'
    """
    builder = GraphBuilder(name)
    builder.add_node(body_name, body_handler)
    builder.add_router(condition_name, condition_handler)
    builder.add_edge(body_name, condition_name)
    builder.add_edge(condition_name, body_name, condition=continue_condition)
    builder.add_edge(condition_name, END, condition=exit_condition)
    builder.set_entry(body_name)
    return builder.build()
