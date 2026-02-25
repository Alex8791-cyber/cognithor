"""Tests: Graph Orchestrator v18.

Tests für alle v18-Module: Types, StateManager, GraphEngine,
GraphBuilder, Built-in Nodes, Integration.
Kein externes LLM, kein Netzwerk — alles lokal testbar.
"""

import asyncio
import json
import tempfile
import pytest
from pathlib import Path
from typing import Any

from jarvis.graph.types import (
    START, END, GRAPH_VERSION,
    NodeType, EdgeType, ExecutionStatus, NodeStatus,
    GraphState, Node, Edge, NodeResult,
    Checkpoint, ExecutionRecord, GraphDefinition,
)
from jarvis.graph.state import StateManager
from jarvis.graph.engine import GraphEngine
from jarvis.graph.builder import (
    GraphBuilder, linear_graph, branch_graph, loop_graph,
)
from jarvis.graph.nodes import (
    llm_node, tool_node, transform_node, condition_node,
    threshold_router, key_router, delay_node, log_node,
    accumulate_node, gate_node, counter_node, set_value_node,
    merge_node,
)


# ============================================================================
# Helper Handlers
# ============================================================================

async def increment_handler(state: GraphState) -> GraphState:
    state["counter"] = state.get("counter", 0) + 1
    return state

async def double_handler(state: GraphState) -> GraphState:
    state["value"] = state.get("value", 0) * 2
    return state

async def append_handler(state: GraphState) -> GraphState:
    msgs = state.get("messages", [])
    msgs.append(f"step_{len(msgs)}")
    state["messages"] = msgs
    return state

async def failing_handler(state: GraphState) -> GraphState:
    raise ValueError("Intentional failure")

async def simple_router(state: GraphState) -> str:
    value = state.get("route", "default")
    return str(value)

async def loop_router(state: GraphState) -> str:
    if state.get("counter", 0) >= state.get("max_count", 3):
        return "exit"
    return "continue"


# ============================================================================
# GraphState Tests
# ============================================================================


class TestGraphState:
    def test_basic(self):
        state = GraphState(name="test", value=42)
        assert state.name == "test"
        assert state["value"] == 42

    def test_setattr(self):
        state = GraphState()
        state.x = 10
        assert state.x == 10
        assert state["x"] == 10

    def test_setitem(self):
        state = GraphState()
        state["y"] = 20
        assert state.y == 20

    def test_contains(self):
        state = GraphState(key="val")
        assert "key" in state
        assert "missing" not in state

    def test_get_default(self):
        state = GraphState()
        assert state.get("missing", 42) == 42

    def test_keys_values_items(self):
        state = GraphState(a=1, b=2)
        assert set(state.keys()) == {"a", "b"}
        assert set(state.values()) == {1, 2}
        assert len(list(state.items())) == 2

    def test_update_dict(self):
        state = GraphState(a=1)
        state.update({"b": 2, "c": 3})
        assert state.b == 2
        assert state.c == 3

    def test_update_state(self):
        s1 = GraphState(a=1)
        s2 = GraphState(b=2)
        s1.update(s2)
        assert s1.b == 2

    def test_copy(self):
        state = GraphState(items=[1, 2, 3])
        copied = state.copy()
        copied["items"].append(4)
        assert len(state["items"]) == 3  # Original unverändert

    def test_to_dict(self):
        state = GraphState(x=1, y="hello")
        d = state.to_dict()
        assert d == {"x": 1, "y": "hello"}

    def test_to_json(self):
        state = GraphState(x=1)
        j = state.to_json()
        assert json.loads(j) == {"x": 1}

    def test_from_dict(self):
        state = GraphState.from_dict({"a": 1, "b": [1, 2]})
        assert state.a == 1
        assert state.b == [1, 2]

    def test_repr(self):
        state = GraphState(x=1, y=2)
        assert "GraphState" in repr(state)

    def test_missing_attr(self):
        state = GraphState()
        with pytest.raises(AttributeError):
            _ = state.nonexistent


# ============================================================================
# Node, Edge, NodeResult Tests
# ============================================================================


class TestNode:
    def test_basic(self):
        n = Node(name="process", node_type=NodeType.FUNCTION)
        assert n.name == "process"
        assert n.retry_count == 0

    def test_to_dict(self):
        n = Node(name="router", node_type=NodeType.ROUTER, timeout_seconds=60)
        d = n.to_dict()
        assert d["type"] == "router"
        assert d["timeout_seconds"] == 60

    def test_all_node_types(self):
        for nt in NodeType:
            n = Node(name=f"test_{nt.value}", node_type=nt)
            assert n.node_type == nt


class TestEdge:
    def test_direct(self):
        e = Edge(source="a", target="b")
        assert e.edge_type == EdgeType.DIRECT
        d = e.to_dict()
        assert d["source"] == "a"

    def test_conditional(self):
        e = Edge(source="router", target="branch_a",
                 edge_type=EdgeType.CONDITIONAL, condition="yes")
        assert e.condition == "yes"
        d = e.to_dict()
        assert d["condition"] == "yes"


class TestNodeResult:
    def test_success(self):
        r = NodeResult(node_name="step1", status=NodeStatus.COMPLETED, duration_ms=50)
        d = r.to_dict()
        assert d["status"] == "completed"
        assert d["duration_ms"] == 50

    def test_failure(self):
        r = NodeResult(node_name="step1", status=NodeStatus.FAILED, error="oops")
        d = r.to_dict()
        assert d["error"] == "oops"

    def test_router_decision(self):
        r = NodeResult(node_name="router", status=NodeStatus.COMPLETED,
                       router_decision="branch_a")
        d = r.to_dict()
        assert d["router_decision"] == "branch_a"


# ============================================================================
# Checkpoint Tests
# ============================================================================


class TestCheckpoint:
    def test_basic(self):
        cp = Checkpoint(execution_id="exec-1", graph_name="test",
                        current_node="step2", state={"x": 1})
        assert cp.checkpoint_id
        assert cp.created_at

    def test_serialization(self):
        cp = Checkpoint(execution_id="e1", graph_name="g", state={"a": 1})
        j = cp.to_json()
        restored = Checkpoint.from_json(j)
        assert restored.execution_id == "e1"
        assert restored.state["a"] == 1

    def test_roundtrip(self):
        cp = Checkpoint(execution_id="e2", graph_name="g2",
                        current_node="n1", state={"x": [1, 2, 3]},
                        history=[{"node": "n0", "status": "completed"}])
        d = cp.to_dict()
        restored = Checkpoint.from_dict(d)
        assert restored.state["x"] == [1, 2, 3]
        assert len(restored.history) == 1


class TestExecutionRecord:
    def test_basic(self):
        r = ExecutionRecord(graph_name="test")
        assert r.execution_id
        assert r.node_count == 0
        assert r.success_rate == 0.0

    def test_success_rate(self):
        r = ExecutionRecord(graph_name="test", node_results=[
            NodeResult(node_name="a", status=NodeStatus.COMPLETED),
            NodeResult(node_name="b", status=NodeStatus.COMPLETED),
            NodeResult(node_name="c", status=NodeStatus.FAILED),
        ])
        assert r.success_rate == pytest.approx(2/3)

    def test_to_dict(self):
        r = ExecutionRecord(graph_name="test", status=ExecutionStatus.COMPLETED)
        d = r.to_dict()
        assert d["graph_name"] == "test"
        assert d["status"] == "completed"


# ============================================================================
# GraphDefinition Tests
# ============================================================================


class TestGraphDefinition:
    def test_basic(self):
        g = GraphDefinition(name="test")
        g.add_node(Node(name="a"))
        g.add_node(Node(name="b"))
        g.add_edge(Edge(source="a", target="b"))
        g.entry_point = "a"
        assert len(g.nodes) == 2
        assert len(g.edges) == 1

    def test_successors_predecessors(self):
        g = GraphDefinition()
        g.add_node(Node(name="a"))
        g.add_node(Node(name="b"))
        g.add_node(Node(name="c"))
        g.add_edge(Edge(source="a", target="b"))
        g.add_edge(Edge(source="a", target="c"))
        assert set(g.get_successors("a")) == {"b", "c"}
        assert g.get_predecessors("b") == ["a"]

    def test_validation_no_nodes(self):
        g = GraphDefinition()
        errors = g.validate()
        assert len(errors) > 0

    def test_validation_no_entry(self):
        g = GraphDefinition()
        g.add_node(Node(name="a"))
        errors = g.validate()
        assert any("entry" in e.lower() for e in errors)

    def test_validation_ok(self):
        g = GraphDefinition()
        g.add_node(Node(name="a"))
        g.entry_point = "a"
        errors = g.validate()
        assert len(errors) == 0

    def test_validation_router_without_conditional(self):
        g = GraphDefinition()
        g.add_node(Node(name="r", node_type=NodeType.ROUTER))
        g.add_edge(Edge(source="r", target=END))  # DIRECT, not CONDITIONAL
        g.entry_point = "r"
        errors = g.validate()
        assert any("conditional" in e.lower() for e in errors)

    def test_detect_cycles(self):
        g = GraphDefinition()
        g.add_node(Node(name="a"))
        g.add_node(Node(name="b"))
        g.add_edge(Edge(source="a", target="b"))
        g.add_edge(Edge(source="b", target="a"))
        g.entry_point = "a"
        cycles = g.detect_cycles()
        assert len(cycles) >= 1

    def test_no_cycles(self):
        g = GraphDefinition()
        g.add_node(Node(name="a"))
        g.add_node(Node(name="b"))
        g.add_edge(Edge(source="a", target="b"))
        g.entry_point = "a"
        assert g.detect_cycles() == []

    def test_topological_sort(self):
        g = GraphDefinition()
        g.add_node(Node(name="a"))
        g.add_node(Node(name="b"))
        g.add_node(Node(name="c"))
        g.add_edge(Edge(source="a", target="b"))
        g.add_edge(Edge(source="b", target="c"))
        g.entry_point = "a"
        order = g.topological_sort()
        assert order == ["a", "b", "c"]

    def test_topological_sort_with_cycles(self):
        g = GraphDefinition()
        g.add_node(Node(name="a"))
        g.add_node(Node(name="b"))
        g.add_edge(Edge(source="a", target="b"))
        g.add_edge(Edge(source="b", target="a"))
        g.entry_point = "a"
        assert g.topological_sort() is None

    def test_to_dict(self):
        g = GraphDefinition(name="test_graph")
        g.add_node(Node(name="a"))
        g.entry_point = "a"
        d = g.to_dict()
        assert d["name"] == "test_graph"
        assert d["node_count"] == 1

    def test_to_mermaid(self):
        g = GraphDefinition()
        g.add_node(Node(name="start_node"))
        g.add_node(Node(name="router", node_type=NodeType.ROUTER))
        g.add_edge(Edge(source="start_node", target="router"))
        g.entry_point = "start_node"
        mermaid = g.to_mermaid()
        assert "graph TD" in mermaid
        assert "start_node" in mermaid

    def test_unreachable_node(self):
        g = GraphDefinition()
        g.add_node(Node(name="a"))
        g.add_node(Node(name="b"))
        g.add_node(Node(name="isolated"))
        g.add_edge(Edge(source="a", target="b"))
        g.entry_point = "a"
        errors = g.validate()
        assert any("isolated" in e for e in errors)


# ============================================================================
# StateManager Tests
# ============================================================================


class TestStateManager:
    @pytest.fixture
    def tmpdir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_create_checkpoint(self, tmpdir):
        mgr = StateManager(storage_dir=tmpdir)
        state = GraphState(x=1, y=2)
        cp = mgr.create_checkpoint("exec-1", "graph-1", "node-a", state)
        assert cp.checkpoint_id
        assert cp.state == {"x": 1, "y": 2}

    def test_get_checkpoint(self, tmpdir):
        mgr = StateManager(storage_dir=tmpdir)
        state = GraphState(data="test")
        cp = mgr.create_checkpoint("e1", "g1", "n1", state)
        loaded = mgr.get_checkpoint(cp.checkpoint_id)
        assert loaded is not None
        assert loaded.state["data"] == "test"

    def test_get_latest_checkpoint(self, tmpdir):
        mgr = StateManager(storage_dir=tmpdir)
        state1 = GraphState(step=1)
        state2 = GraphState(step=2)
        mgr.create_checkpoint("e1", "g1", "n1", state1)
        cp2 = mgr.create_checkpoint("e1", "g1", "n2", state2)
        latest = mgr.get_latest_checkpoint("e1")
        assert latest is not None
        assert latest.checkpoint_id == cp2.checkpoint_id

    def test_restore_state(self, tmpdir):
        mgr = StateManager(storage_dir=tmpdir)
        state = GraphState(counter=5, messages=["hello"])
        cp = mgr.create_checkpoint("e1", "g1", "step3", state)
        restored, node = mgr.restore_state(cp.checkpoint_id)
        assert restored is not None
        assert restored.counter == 5
        assert node == "step3"

    def test_restore_nonexistent(self, tmpdir):
        mgr = StateManager(storage_dir=tmpdir)
        state, node = mgr.restore_state("nonexistent")
        assert state is None
        assert node == ""

    def test_save_to_disk(self, tmpdir):
        mgr = StateManager(storage_dir=tmpdir)
        state = GraphState(x=42)
        cp = mgr.create_checkpoint("e1", "g1", "n1", state)
        assert mgr.save_checkpoint_to_disk(cp.checkpoint_id)

        # Fresh manager
        mgr2 = StateManager(storage_dir=tmpdir)
        loaded = mgr2.get_checkpoint(cp.checkpoint_id)
        assert loaded is not None
        assert loaded.state["x"] == 42

    def test_delete_checkpoint(self, tmpdir):
        mgr = StateManager(storage_dir=tmpdir)
        cp = mgr.create_checkpoint("e1", "g1", "n1", GraphState())
        mgr.save_checkpoint_to_disk(cp.checkpoint_id)
        assert mgr.delete_checkpoint(cp.checkpoint_id)
        assert mgr.get_checkpoint(cp.checkpoint_id) is None

    def test_execution_record(self, tmpdir):
        mgr = StateManager(storage_dir=tmpdir)
        state = GraphState(x=1)
        record = mgr.create_execution("test_graph", state)
        assert record.execution_id
        assert record.status == ExecutionStatus.RUNNING

        loaded = mgr.get_execution(record.execution_id)
        assert loaded is not None

    def test_list_executions(self, tmpdir):
        mgr = StateManager(storage_dir=tmpdir)
        mgr.create_execution("g1", GraphState())
        mgr.create_execution("g2", GraphState())
        records = mgr.list_executions()
        assert len(records) == 2

    def test_stats(self, tmpdir):
        mgr = StateManager(storage_dir=tmpdir)
        stats = mgr.stats()
        assert stats["checkpoints"] == 0
        assert stats["executions"] == 0


# ============================================================================
# GraphBuilder Tests
# ============================================================================


class TestGraphBuilder:
    def test_basic_build(self):
        graph = (
            GraphBuilder("test")
            .add_node("a", increment_handler)
            .add_node("b", double_handler)
            .add_edge("a", "b")
            .add_edge("b", END)
            .set_entry("a")
            .build()
        )
        assert graph.name == "test"
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 2

    def test_auto_entry(self):
        graph = (
            GraphBuilder("auto")
            .add_node("first", increment_handler)
            .add_edge("first", END)
            .build()
        )
        assert graph.entry_point == "first"

    def test_chain(self):
        graph = (
            GraphBuilder("chained")
            .add_node("a", increment_handler)
            .add_node("b", double_handler)
            .add_node("c", append_handler)
            .chain("a", "b", "c", END)
            .build()
        )
        assert len(graph.edges) == 3
        assert graph.entry_point == "a"

    def test_add_router(self):
        graph = (
            GraphBuilder("routed")
            .add_router("classify", simple_router)
            .add_node("a", increment_handler)
            .add_node("b", double_handler)
            .add_conditional_edges("classify", {"a": "a", "b": "b"})
            .add_edge("a", END)
            .add_edge("b", END)
            .set_entry("classify")
            .build()
        )
        router = graph.get_node("classify")
        assert router.node_type == NodeType.ROUTER

    def test_add_hitl(self):
        graph = (
            GraphBuilder("hitl_test")
            .add_node("process", increment_handler)
            .add_hitl("review")
            .add_node("finalize", increment_handler)
            .chain("process", "review", "finalize", END)
            .build()
        )
        hitl = graph.get_node("review")
        assert hitl.node_type == NodeType.HITL
        assert hitl.checkpoint_before

    def test_add_conditional_edges(self):
        graph = (
            GraphBuilder("cond")
            .add_router("r", simple_router)
            .add_node("x", increment_handler)
            .add_node("y", increment_handler)
            .add_conditional_edges("r", {"option_x": "x", "option_y": "y"}, default=END)
            .add_edge("x", END)
            .add_edge("y", END)
            .set_entry("r")
            .build()
        )
        cond_edges = [e for e in graph.edges if e.edge_type == EdgeType.CONDITIONAL]
        assert len(cond_edges) == 3  # 2 + default

    def test_validation_error(self):
        with pytest.raises(ValueError, match="Invalid graph"):
            GraphBuilder("empty").build()

    def test_double_build(self):
        builder = GraphBuilder("x").add_node("a", increment_handler).add_edge("a", END)
        builder.build()
        with pytest.raises(ValueError, match="already built"):
            builder.build()

    def test_build_unchecked(self):
        graph = GraphBuilder("unchecked").build_unchecked()
        assert graph is not None

    def test_metadata(self):
        graph = (
            GraphBuilder("meta")
            .add_node("a", increment_handler)
            .add_edge("a", END)
            .set_metadata("author", "test")
            .build()
        )
        assert graph.metadata["author"] == "test"


# ============================================================================
# Template Builders Tests
# ============================================================================


class TestTemplateBuilders:
    def test_linear_graph(self):
        graph = linear_graph("pipeline", [
            ("fetch", increment_handler),
            ("process", double_handler),
            ("store", append_handler),
        ])
        assert len(graph.nodes) == 3
        assert graph.entry_point == "fetch"

    def test_branch_graph(self):
        graph = branch_graph(
            "branching",
            router_name="classify",
            router_handler=simple_router,
            branches={
                "fast": increment_handler,
                "slow": double_handler,
            },
        )
        assert len(graph.nodes) == 3  # router + 2 branches

    def test_branch_graph_with_merge(self):
        graph = branch_graph(
            "merge_test",
            router_name="split",
            router_handler=simple_router,
            branches={
                "a": increment_handler,
                "b": double_handler,
            },
            merge_node="combine",
            merge_handler=append_handler,
        )
        assert "combine" in graph.nodes

    def test_loop_graph(self):
        graph = loop_graph(
            "retry_loop",
            body_name="process",
            body_handler=counter_node("counter"),
            condition_name="check",
            condition_handler=loop_router,
        )
        cycles = graph.detect_cycles()
        assert len(cycles) >= 1


# ============================================================================
# GraphEngine Tests
# ============================================================================


class TestGraphEngine:
    @pytest.mark.asyncio
    async def test_simple_linear(self):
        graph = linear_graph("simple", [
            ("a", increment_handler),
            ("b", increment_handler),
        ])
        engine = GraphEngine()
        result = await engine.run(graph, GraphState(counter=0))
        assert result.status == ExecutionStatus.COMPLETED
        assert result.final_state["counter"] == 2
        assert result.node_count == 2

    @pytest.mark.asyncio
    async def test_conditional_routing(self):
        graph = (
            GraphBuilder("routing")
            .add_router("router", simple_router)
            .add_node("path_a", set_value_node(result="went_a"))
            .add_node("path_b", set_value_node(result="went_b"))
            .add_conditional_edges("router", {"a": "path_a", "b": "path_b"})
            .add_edge("path_a", END)
            .add_edge("path_b", END)
            .set_entry("router")
            .build()
        )
        engine = GraphEngine()

        result_a = await engine.run(graph, GraphState(route="a"))
        assert result_a.final_state["result"] == "went_a"

        result_b = await engine.run(graph, GraphState(route="b"))
        assert result_b.final_state["result"] == "went_b"

    @pytest.mark.asyncio
    async def test_loop_execution(self):
        graph = loop_graph(
            "counter_loop",
            body_name="increment",
            body_handler=counter_node("counter"),
            condition_name="check",
            condition_handler=loop_router,
        )
        engine = GraphEngine()
        result = await engine.run(graph, GraphState(counter=0, max_count=3))
        assert result.status == ExecutionStatus.COMPLETED
        assert result.final_state["counter"] == 3

    @pytest.mark.asyncio
    async def test_loop_max_iterations(self):
        """Loop der nie terminiert → max_iterations."""
        async def always_continue(state: GraphState) -> str:
            return "continue"

        graph = loop_graph(
            "infinite", body_name="body", body_handler=increment_handler,
            condition_name="check", condition_handler=always_continue,
        )
        engine = GraphEngine(max_iterations=5)
        result = await engine.run(graph, GraphState(counter=0))
        assert result.status == ExecutionStatus.FAILED
        assert "iterations" in result.error.lower()

    @pytest.mark.asyncio
    async def test_node_failure(self):
        graph = linear_graph("failing", [
            ("ok_step", increment_handler),
            ("bad_step", failing_handler),
        ])
        engine = GraphEngine()
        result = await engine.run(graph, GraphState(counter=0))
        assert result.status == ExecutionStatus.FAILED
        assert "bad_step" in result.error

    @pytest.mark.asyncio
    async def test_node_retry(self):
        call_count = 0

        async def flaky_handler(state: GraphState) -> GraphState:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Flaky error")
            state["success"] = True
            return state

        graph = (
            GraphBuilder("retry_test")
            .add_node("flaky", flaky_handler, retry_count=3, retry_delay=0.01)
            .add_edge("flaky", END)
            .build()
        )
        engine = GraphEngine()
        result = await engine.run(graph, GraphState())
        assert result.status == ExecutionStatus.COMPLETED
        assert result.final_state["success"]

    @pytest.mark.asyncio
    async def test_node_timeout(self):
        async def slow_handler(state: GraphState) -> GraphState:
            await asyncio.sleep(10)
            return state

        graph = (
            GraphBuilder("timeout_test")
            .add_node("slow", slow_handler, timeout=0.1)
            .add_edge("slow", END)
            .build()
        )
        engine = GraphEngine()
        result = await engine.run(graph, GraphState())
        assert result.status == ExecutionStatus.FAILED
        assert "timeout" in result.error.lower()

    @pytest.mark.asyncio
    async def test_hitl_pause(self):
        graph = (
            GraphBuilder("hitl_test")
            .add_node("before", increment_handler)
            .add_hitl("review")
            .add_node("after", increment_handler)
            .chain("before", "review", "after", END)
            .build()
        )
        engine = GraphEngine()
        result = await engine.run(graph, GraphState(counter=0))
        assert result.status == ExecutionStatus.PAUSED
        assert result.final_state["counter"] == 1  # before ran

    @pytest.mark.asyncio
    async def test_hitl_resume(self):
        graph = (
            GraphBuilder("hitl_resume")
            .add_node("before", increment_handler)
            .add_hitl("review")
            .add_node("after", increment_handler)
            .chain("before", "review", "after", END)
            .build()
        )
        engine = GraphEngine()

        # Phase 1: Run until pause
        result1 = await engine.run(graph, GraphState(counter=0))
        assert result1.status == ExecutionStatus.PAUSED
        cp_id = result1.checkpoints[0] if result1.checkpoints else ""
        assert cp_id

        # Phase 2: Resume
        result2 = await engine.resume(
            graph, checkpoint_id=cp_id,
            resume_input={"human_approved": True},
        )
        assert result2.status == ExecutionStatus.COMPLETED
        assert result2.final_state["counter"] == 2
        assert result2.final_state["human_approved"]

    @pytest.mark.asyncio
    async def test_checkpoint_before_after(self):
        graph = (
            GraphBuilder("cp_test")
            .add_node("step1", increment_handler, checkpoint_after=True)
            .add_node("step2", increment_handler, checkpoint_before=True)
            .chain("step1", "step2", END)
            .build()
        )
        engine = GraphEngine()
        result = await engine.run(graph, GraphState(counter=0))
        assert result.status == ExecutionStatus.COMPLETED
        cps = engine._state_mgr.list_checkpoints(result.execution_id)
        assert len(cps) >= 2

    @pytest.mark.asyncio
    async def test_passthrough_node(self):
        graph = (
            GraphBuilder("passthrough")
            .add_node("step1", increment_handler)
            .add_passthrough("merge")
            .add_node("step2", increment_handler)
            .chain("step1", "merge", "step2", END)
            .build()
        )
        engine = GraphEngine()
        result = await engine.run(graph, GraphState(counter=0))
        assert result.status == ExecutionStatus.COMPLETED
        assert result.final_state["counter"] == 2

    @pytest.mark.asyncio
    async def test_handler_returns_dict(self):
        async def dict_handler(state: GraphState) -> dict:
            return {"result": "from_dict"}

        graph = (
            GraphBuilder("dict_return")
            .add_node("step", dict_handler)
            .add_edge("step", END)
            .build()
        )
        engine = GraphEngine()
        result = await engine.run(graph, GraphState())
        assert result.final_state["result"] == "from_dict"

    @pytest.mark.asyncio
    async def test_validation_failure(self):
        graph = GraphDefinition(name="invalid")
        engine = GraphEngine()
        result = await engine.run(graph, GraphState())
        assert result.status == ExecutionStatus.FAILED
        assert "validation" in result.error.lower()

    @pytest.mark.asyncio
    async def test_max_nodes_limit(self):
        graph = loop_graph(
            "many_nodes",
            body_name="body", body_handler=increment_handler,
            condition_name="check", condition_handler=loop_router,
        )
        engine = GraphEngine(max_nodes=5)
        result = await engine.run(graph, GraphState(counter=0, max_count=100))
        assert result.status == ExecutionStatus.FAILED
        assert "max nodes" in result.error.lower()

    @pytest.mark.asyncio
    async def test_cancel(self):
        engine = GraphEngine()
        state = GraphState()
        record = engine._state_mgr.create_execution("test", state)
        record.status = ExecutionStatus.RUNNING
        engine._state_mgr.update_execution(record)

        assert await engine.cancel(record.execution_id)
        updated = engine.get_execution(record.execution_id)
        assert updated.status == ExecutionStatus.CANCELED

    @pytest.mark.asyncio
    async def test_stream_execution(self):
        graph = linear_graph("stream_test", [
            ("a", increment_handler),
            ("b", increment_handler),
        ])
        engine = GraphEngine()
        results = []
        async for node_result in engine.run_stream(graph, GraphState(counter=0)):
            results.append(node_result)
        assert len(results) == 2
        assert all(r.status == NodeStatus.COMPLETED for r in results)

    @pytest.mark.asyncio
    async def test_stream_with_failure(self):
        graph = linear_graph("stream_fail", [
            ("ok", increment_handler),
            ("bad", failing_handler),
        ])
        engine = GraphEngine()
        results = []
        async for node_result in engine.run_stream(graph, GraphState(counter=0)):
            results.append(node_result)
        assert results[-1].status == NodeStatus.FAILED

    def test_stats(self):
        engine = GraphEngine()
        stats = engine.stats()
        assert stats["total_executions"] == 0
        assert stats["running"] == 0


# ============================================================================
# Built-in Nodes Tests
# ============================================================================


class TestBuiltinNodes:
    @pytest.mark.asyncio
    async def test_llm_node(self):
        handler = llm_node("Summarize: {text}", output_key="summary")
        state = GraphState(text="Hello world")
        result = await handler(state)
        assert "summary" in result

    @pytest.mark.asyncio
    async def test_llm_node_custom_handler(self):
        async def mock_llm(prompt: str, model: str = "") -> str:
            return f"Response to: {prompt[:50]}"

        handler = llm_node("Q: {question}", output_key="answer", llm_handler=mock_llm)
        state = GraphState(question="What is AI?")
        result = await handler(state)
        assert "Response to:" in result["answer"]

    @pytest.mark.asyncio
    async def test_tool_node(self):
        handler = tool_node("search", params_key="query", result_key="results")
        state = GraphState(query={"q": "test"})
        result = await handler(state)
        assert result["results"]["tool"] == "search"

    @pytest.mark.asyncio
    async def test_transform_node(self):
        def upper(data: dict) -> dict:
            data["text"] = data.get("text", "").upper()
            return data

        handler = transform_node(upper)
        state = GraphState(text="hello")
        result = await handler(state)
        assert result["text"] == "HELLO"

    @pytest.mark.asyncio
    async def test_condition_node(self):
        def check(data: dict) -> str:
            return "yes" if data.get("score", 0) > 50 else "no"

        handler = condition_node(check)
        state = GraphState(score=75)
        decision = await handler(state)
        assert decision == "yes"

    @pytest.mark.asyncio
    async def test_threshold_router(self):
        handler = threshold_router("confidence", 0.8, above="high", below="low")
        assert await handler(GraphState(confidence=0.9)) == "high"
        assert await handler(GraphState(confidence=0.5)) == "low"

    @pytest.mark.asyncio
    async def test_key_router(self):
        handler = key_router("intent", {"greeting": "greet", "question": "qa"}, default="fallback")
        assert await handler(GraphState(intent="greeting")) == "greet"
        assert await handler(GraphState(intent="unknown")) == "fallback"

    @pytest.mark.asyncio
    async def test_delay_node(self):
        handler = delay_node(0.01)
        state = GraphState()
        result = await handler(state)
        assert result is not None

    @pytest.mark.asyncio
    async def test_log_node(self):
        handler = log_node("Test log", log_keys=["counter"])
        state = GraphState(counter=5)
        result = await handler(state)
        assert len(result["__log__"]) == 1
        assert result["__log__"][0]["counter"] == 5

    @pytest.mark.asyncio
    async def test_accumulate_node(self):
        handler = accumulate_node(["result_a", "result_b"], target_key="all_results")
        state = GraphState(result_a="data_a", result_b="data_b")
        result = await handler(state)
        assert len(result["all_results"]) == 2

    @pytest.mark.asyncio
    async def test_gate_node_pass(self):
        handler = gate_node(lambda d: d.get("ready", False))
        state = GraphState(ready=True)
        result = await handler(state)
        assert result is not None

    @pytest.mark.asyncio
    async def test_gate_node_fail(self):
        handler = gate_node(lambda d: d.get("ready", False), error_message="Not ready")
        state = GraphState(ready=False)
        with pytest.raises(ValueError, match="Not ready"):
            await handler(state)

    @pytest.mark.asyncio
    async def test_counter_node(self):
        handler = counter_node("count", increment=5)
        state = GraphState(count=10)
        result = await handler(state)
        assert result["count"] == 15

    @pytest.mark.asyncio
    async def test_set_value_node(self):
        handler = set_value_node(status="done", score=100)
        state = GraphState()
        result = await handler(state)
        assert result["status"] == "done"
        assert result["score"] == 100

    @pytest.mark.asyncio
    async def test_merge_node(self):
        def my_merge(data: dict) -> dict:
            data["merged"] = True
            return data

        handler = merge_node(my_merge)
        state = GraphState(x=1)
        result = await handler(state)
        assert result["merged"]

    @pytest.mark.asyncio
    async def test_merge_node_passthrough(self):
        handler = merge_node()
        state = GraphState(x=1)
        result = await handler(state)
        assert result["x"] == 1


# ============================================================================
# Integration Tests
# ============================================================================


class TestGraphIntegration:
    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """Realistischer Pipeline: Fetch → Process → Validate → Store."""
        async def fetch(state: GraphState) -> GraphState:
            state["data"] = {"items": [1, 2, 3]}
            return state

        async def process(state: GraphState) -> GraphState:
            items = state["data"]["items"]
            state["processed"] = [x * 2 for x in items]
            return state

        async def validate(state: GraphState) -> str:
            if all(x > 0 for x in state["processed"]):
                return "valid"
            return "invalid"

        async def store(state: GraphState) -> GraphState:
            state["stored"] = True
            return state

        async def handle_error(state: GraphState) -> GraphState:
            state["error"] = "Validation failed"
            return state

        graph = (
            GraphBuilder("data_pipeline", description="ETL Pipeline")
            .add_node("fetch", fetch)
            .add_node("process", process)
            .add_router("validate", validate)
            .add_node("store", store)
            .add_node("handle_error", handle_error)
            .add_edge("fetch", "process")
            .add_edge("process", "validate")
            .add_conditional_edges("validate", {
                "valid": "store",
                "invalid": "handle_error",
            })
            .add_edge("store", END)
            .add_edge("handle_error", END)
            .set_entry("fetch")
            .build()
        )

        engine = GraphEngine()
        result = await engine.run(graph, GraphState())
        assert result.status == ExecutionStatus.COMPLETED
        assert result.final_state["stored"]
        assert result.node_count == 4  # fetch, process, validate, store

    @pytest.mark.asyncio
    async def test_customer_support_flow(self):
        """Simulate Customer-Support mit HITL."""
        async def classify(state: GraphState) -> str:
            text = state.get("message", "")
            if "refund" in text.lower():
                return "refund"
            return "general"

        async def auto_reply(state: GraphState) -> GraphState:
            state["reply"] = "Thanks for contacting us!"
            return state

        async def refund_review(state: GraphState) -> GraphState:
            state["needs_review"] = True
            return state

        graph = (
            GraphBuilder("support")
            .add_router("classify", classify)
            .add_node("auto_reply", auto_reply)
            .add_hitl("human_review", refund_review)
            .add_conditional_edges("classify", {
                "general": "auto_reply",
                "refund": "human_review",
            })
            .add_edge("auto_reply", END)
            .add_edge("human_review", END)
            .set_entry("classify")
            .build()
        )

        engine = GraphEngine()

        # General query → auto reply
        r1 = await engine.run(graph, GraphState(message="Hello, how are you?"))
        assert r1.status == ExecutionStatus.COMPLETED
        assert r1.final_state["reply"] == "Thanks for contacting us!"

        # Refund query → HITL pause
        r2 = await engine.run(graph, GraphState(message="I want a refund"))
        assert r2.status == ExecutionStatus.PAUSED
        assert r2.final_state.get("needs_review")

    @pytest.mark.asyncio
    async def test_retry_loop_pattern(self):
        """Pattern: Versuche bis Erfolg oder max Retries."""
        attempt_count = 0

        async def attempt(state: GraphState) -> GraphState:
            nonlocal attempt_count
            attempt_count += 1
            state["attempts"] = attempt_count
            state["success"] = attempt_count >= 3
            return state

        async def check_success(state: GraphState) -> str:
            if state.get("success"):
                return "exit"
            return "continue"

        graph = loop_graph(
            "retry_pattern",
            body_name="attempt", body_handler=attempt,
            condition_name="check", condition_handler=check_success,
        )
        engine = GraphEngine()
        result = await engine.run(graph, GraphState())
        assert result.status == ExecutionStatus.COMPLETED
        assert result.final_state["attempts"] == 3

    @pytest.mark.asyncio
    async def test_mermaid_generation(self):
        """Prüft Mermaid-Export."""
        graph = (
            GraphBuilder("mermaid_test")
            .add_router("router", simple_router)
            .add_node("a", increment_handler)
            .add_node("b", increment_handler)
            .add_conditional_edges("router", {"opt_a": "a", "opt_b": "b"})
            .add_edge("a", END)
            .add_edge("b", END)
            .set_entry("router")
            .build()
        )
        mermaid = graph.to_mermaid()
        assert "graph TD" in mermaid
        assert "router" in mermaid
        assert "opt_a" in mermaid

    @pytest.mark.asyncio
    async def test_complex_graph_with_checkpoints_and_stream(self):
        """Komplexer Graph mit Checkpoints und Streaming."""
        graph = (
            GraphBuilder("complex")
            .add_node("init", set_value_node(step="init"), checkpoint_after=True)
            .add_node("process", counter_node("counter"))
            .add_node("log", log_node("Processing done", log_keys=["counter"]))
            .chain("init", "process", "log", END)
            .build()
        )

        engine = GraphEngine()

        # Streaming
        results = []
        async for nr in engine.run_stream(graph, GraphState(counter=0)):
            results.append(nr)
        assert len(results) == 3
        assert all(r.status == NodeStatus.COMPLETED for r in results)
