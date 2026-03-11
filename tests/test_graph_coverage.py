"""Tests for graph/engine.py, graph/state.py, graph/types.py -- Coverage boost."""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.graph.engine import GraphEngine, MAX_ITERATIONS, MAX_NODES_PER_EXECUTION
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


# ============================================================================
# Helpers
# ============================================================================


def _simple_graph(handler=None, entry="step1", node_type=NodeType.FUNCTION):
    """Create a minimal valid graph: step1 -> END."""
    g = GraphDefinition(name="test_graph")
    g.add_node(Node(name="step1", node_type=node_type, handler=handler))
    g.add_edge(Edge(source="step1", target=END))
    g.entry_point = entry
    return g


def _two_node_graph(h1=None, h2=None):
    """step1 -> step2 -> END."""
    g = GraphDefinition(name="two_step")
    g.add_node(Node(name="step1", handler=h1))
    g.add_node(Node(name="step2", handler=h2))
    g.add_edge(Edge(source="step1", target="step2"))
    g.add_edge(Edge(source="step2", target=END))
    g.entry_point = "step1"
    return g


def _router_graph(router_handler):
    """Router node with two conditional edges: 'a' -> nodeA, 'b' -> nodeB."""
    g = GraphDefinition(name="router_graph")
    g.add_node(Node(name="router", node_type=NodeType.ROUTER, handler=router_handler))
    g.add_node(Node(name="nodeA", handler=None))
    g.add_node(Node(name="nodeB", handler=None))
    g.add_edge(Edge(source="router", target="nodeA", edge_type=EdgeType.CONDITIONAL, condition="a"))
    g.add_edge(Edge(source="router", target="nodeB", edge_type=EdgeType.CONDITIONAL, condition="b"))
    g.add_edge(Edge(source="nodeA", target=END))
    g.add_edge(Edge(source="nodeB", target=END))
    g.entry_point = "router"
    return g


# ============================================================================
# GraphState Tests
# ============================================================================


class TestGraphState:
    def test_init_and_access(self):
        s = GraphState(messages=[], step=0)
        assert s.step == 0
        assert s["messages"] == []

    def test_setattr(self):
        s = GraphState()
        s.foo = "bar"
        assert s["foo"] == "bar"

    def test_setitem(self):
        s = GraphState()
        s["x"] = 42
        assert s.x == 42

    def test_contains(self):
        s = GraphState(a=1)
        assert "a" in s
        assert "b" not in s

    def test_get_default(self):
        s = GraphState()
        assert s.get("missing", 99) == 99

    def test_keys_values_items(self):
        s = GraphState(a=1, b=2)
        assert set(s.keys()) == {"a", "b"}
        assert set(s.values()) == {1, 2}
        assert dict(s.items()) == {"a": 1, "b": 2}

    def test_update_dict(self):
        s = GraphState(x=1)
        s.update({"y": 2})
        assert s["y"] == 2

    def test_update_graphstate(self):
        s1 = GraphState(a=1)
        s2 = GraphState(b=2)
        s1.update(s2)
        assert s1["b"] == 2

    def test_copy(self):
        s = GraphState(data=[1, 2])
        c = s.copy()
        c["data"].append(3)
        assert len(s["data"]) == 2  # Deep copy

    def test_to_dict(self):
        s = GraphState(a=1)
        d = s.to_dict()
        assert d == {"a": 1}

    def test_to_json(self):
        s = GraphState(a=1)
        j = s.to_json()
        assert json.loads(j) == {"a": 1}

    def test_from_dict(self):
        s = GraphState.from_dict({"x": 42})
        assert s["x"] == 42

    def test_repr(self):
        s = GraphState(a=1, b=2)
        r = repr(s)
        assert "GraphState" in r

    def test_attribute_error(self):
        s = GraphState()
        with pytest.raises(AttributeError):
            _ = s.nonexistent


# ============================================================================
# GraphDefinition Tests
# ============================================================================


class TestGraphDefinition:
    def test_validate_no_nodes(self):
        g = GraphDefinition()
        errors = g.validate()
        assert any("no nodes" in e for e in errors)

    def test_validate_no_entry(self):
        g = GraphDefinition()
        g.add_node(Node(name="n1"))
        errors = g.validate()
        assert any("entry point" in e.lower() for e in errors)

    def test_validate_entry_not_found(self):
        g = GraphDefinition()
        g.add_node(Node(name="n1"))
        g.entry_point = "missing"
        errors = g.validate()
        assert any("not found" in e for e in errors)

    def test_validate_edge_source_not_found(self):
        g = GraphDefinition()
        g.add_node(Node(name="n1"))
        g.entry_point = "n1"
        g.add_edge(Edge(source="ghost", target="n1"))
        errors = g.validate()
        assert any("ghost" in e for e in errors)

    def test_validate_unreachable_node(self):
        g = GraphDefinition()
        g.add_node(Node(name="n1"))
        g.add_node(Node(name="island"))
        g.entry_point = "n1"
        g.add_edge(Edge(source="n1", target=END))
        errors = g.validate()
        assert any("island" in e for e in errors)

    def test_validate_router_no_conditional_edges(self):
        g = GraphDefinition()
        g.add_node(Node(name="r", node_type=NodeType.ROUTER))
        g.entry_point = "r"
        g.add_edge(Edge(source="r", target=END))  # DIRECT, not CONDITIONAL
        errors = g.validate()
        assert any("conditional" in e.lower() for e in errors)

    def test_valid_graph(self):
        g = _simple_graph()
        errors = g.validate()
        assert errors == []

    def test_detect_cycles(self):
        g = GraphDefinition()
        g.add_node(Node(name="a"))
        g.add_node(Node(name="b"))
        g.add_edge(Edge(source="a", target="b"))
        g.add_edge(Edge(source="b", target="a"))
        g.entry_point = "a"
        cycles = g.detect_cycles()
        assert len(cycles) > 0

    def test_detect_no_cycles(self):
        g = _simple_graph()
        assert g.detect_cycles() == []

    def test_topological_sort(self):
        g = _two_node_graph()
        result = g.topological_sort()
        assert result is not None
        assert result.index("step1") < result.index("step2")

    def test_topological_sort_cycle(self):
        g = GraphDefinition()
        g.add_node(Node(name="a"))
        g.add_node(Node(name="b"))
        g.add_edge(Edge(source="a", target="b"))
        g.add_edge(Edge(source="b", target="a"))
        g.entry_point = "a"
        assert g.topological_sort() is None

    def test_to_dict(self):
        g = _simple_graph()
        d = g.to_dict()
        assert d["name"] == "test_graph"
        assert d["node_count"] == 1

    def test_to_mermaid(self):
        g = _simple_graph()
        m = g.to_mermaid()
        assert "graph TD" in m
        assert "step1" in m

    def test_get_incoming_edges(self):
        g = _two_node_graph()
        incoming = g.get_incoming_edges("step2")
        assert len(incoming) == 1
        assert incoming[0].source == "step1"

    def test_get_predecessors(self):
        g = _two_node_graph()
        preds = g.get_predecessors("step2")
        assert "step1" in preds

    def test_get_successors(self):
        g = _two_node_graph()
        succs = g.get_successors("step1")
        assert "step2" in succs


# ============================================================================
# GraphEngine -- run() Tests
# ============================================================================


class TestGraphEngineRun:
    @pytest.mark.asyncio
    async def test_run_simple(self):
        async def handler(state):
            state["result"] = "done"
            return state

        g = _simple_graph(handler=handler)
        engine = GraphEngine()
        state = GraphState(result="")
        record = await engine.run(g, state)
        assert record.status == ExecutionStatus.COMPLETED
        assert record.final_state.get("result") == "done"

    @pytest.mark.asyncio
    async def test_run_validation_error(self):
        g = GraphDefinition(name="bad")
        engine = GraphEngine()
        record = await engine.run(g, GraphState())
        assert record.status == ExecutionStatus.FAILED
        assert "Validation" in record.error

    @pytest.mark.asyncio
    async def test_run_missing_node(self):
        g = GraphDefinition(name="missing_node")
        g.add_node(Node(name="step1"))
        g.add_edge(Edge(source="step1", target="ghost"))
        g.entry_point = "step1"
        engine = GraphEngine()
        record = await engine.run(g, GraphState())
        # step1 completes, then tries ghost which is not found
        assert record.status == ExecutionStatus.FAILED
        assert "not found" in record.error

    @pytest.mark.asyncio
    async def test_run_two_nodes(self):
        async def h1(state):
            state["x"] = 1
            return state

        async def h2(state):
            state["y"] = state.get("x", 0) + 1
            return state

        g = _two_node_graph(h1, h2)
        engine = GraphEngine()
        record = await engine.run(g, GraphState())
        assert record.status == ExecutionStatus.COMPLETED
        assert record.final_state.get("y") == 2

    @pytest.mark.asyncio
    async def test_run_node_handler_exception(self):
        async def bad_handler(state):
            raise ValueError("Node crash")

        g = _simple_graph(handler=bad_handler)
        engine = GraphEngine()
        record = await engine.run(g, GraphState())
        assert record.status == ExecutionStatus.FAILED
        assert "Node crash" in record.error or "failed" in record.error.lower()

    @pytest.mark.asyncio
    async def test_run_passthrough_node(self):
        g = _simple_graph(node_type=NodeType.PASSTHROUGH)
        engine = GraphEngine()
        record = await engine.run(g, GraphState(x=1))
        assert record.status == ExecutionStatus.COMPLETED
        assert record.final_state.get("x") == 1

    @pytest.mark.asyncio
    async def test_run_no_handler_node(self):
        """Node without handler completes with unchanged state."""
        g = _simple_graph(handler=None)
        engine = GraphEngine()
        record = await engine.run(g, GraphState(x=1))
        assert record.status == ExecutionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_run_dict_return(self):
        """Handler returning dict should merge into state."""

        async def handler(state):
            return {"new_key": "value"}

        g = _simple_graph(handler=handler)
        engine = GraphEngine()
        record = await engine.run(g, GraphState())
        assert record.status == ExecutionStatus.COMPLETED
        assert record.final_state.get("new_key") == "value"

    @pytest.mark.asyncio
    async def test_run_loop_protection(self):
        """Graph that loops should fail with max iterations error."""
        g = GraphDefinition(name="loop")
        g.add_node(Node(name="n1"))
        g.add_edge(Edge(source="n1", target="n1"))  # self-loop
        g.entry_point = "n1"
        engine = GraphEngine(max_iterations=3)
        record = await engine.run(g, GraphState())
        assert record.status == ExecutionStatus.FAILED
        assert "Max iterations" in record.error

    @pytest.mark.asyncio
    async def test_run_max_nodes_exceeded(self):
        """Exceed max_nodes limit."""
        g = GraphDefinition(name="many")
        g.add_node(Node(name="n1"))
        g.add_edge(Edge(source="n1", target="n1"))  # self-loop
        g.entry_point = "n1"
        engine = GraphEngine(max_iterations=100, max_nodes=2)
        record = await engine.run(g, GraphState())
        assert record.status == ExecutionStatus.FAILED
        assert "Max nodes" in record.error

    @pytest.mark.asyncio
    async def test_run_checkpoint_before_after(self):
        async def handler(state):
            return state

        g = GraphDefinition(name="cp")
        g.add_node(
            Node(name="step1", handler=handler, checkpoint_before=True, checkpoint_after=True)
        )
        g.add_edge(Edge(source="step1", target=END))
        g.entry_point = "step1"
        engine = GraphEngine()
        record = await engine.run(g, GraphState())
        assert record.status == ExecutionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_run_execution_id(self):
        g = _simple_graph()
        engine = GraphEngine()
        record = await engine.run(g, GraphState(), execution_id="custom_id")
        assert record.execution_id == "custom_id"

    @pytest.mark.asyncio
    async def test_run_cancellation(self):
        """CancelledError should result in CANCELED status."""

        async def slow_handler(state):
            await asyncio.sleep(100)
            return state

        g = _simple_graph(handler=slow_handler)
        engine = GraphEngine()

        async def run_and_cancel():
            task = asyncio.create_task(engine.run(g, GraphState()))
            await asyncio.sleep(0.05)
            task.cancel()
            return await task

        record = await run_and_cancel()
        assert record.status == ExecutionStatus.CANCELED


# ============================================================================
# GraphEngine -- Router Tests
# ============================================================================


class TestGraphEngineRouter:
    @pytest.mark.asyncio
    async def test_router_string_decision(self):
        async def router(state):
            return "a"

        g = _router_graph(router)
        engine = GraphEngine()
        record = await engine.run(g, GraphState())
        assert record.status == ExecutionStatus.COMPLETED
        # Should have executed router -> nodeA
        node_names = [r.node_name for r in record.node_results]
        assert "router" in node_names
        assert "nodeA" in node_names

    @pytest.mark.asyncio
    async def test_router_state_decision(self):
        """Router returning GraphState with __router_decision__."""

        async def router(state):
            state["__router_decision__"] = "b"
            return state

        g = _router_graph(router)
        engine = GraphEngine()
        record = await engine.run(g, GraphState())
        assert record.status == ExecutionStatus.COMPLETED
        node_names = [r.node_name for r in record.node_results]
        assert "nodeB" in node_names

    @pytest.mark.asyncio
    async def test_router_no_match_fallback_direct(self):
        """Router decision doesn't match any conditional edge -> fallback to DIRECT."""

        async def router(state):
            return "nonexistent"

        g = GraphDefinition(name="fallback")
        g.add_node(Node(name="router", node_type=NodeType.ROUTER, handler=router))
        g.add_node(Node(name="default_node"))
        g.add_edge(
            Edge(
                source="router",
                target="default_node",
                edge_type=EdgeType.CONDITIONAL,
                condition="a",
            )
        )
        g.add_edge(Edge(source="router", target="default_node", edge_type=EdgeType.DIRECT))
        g.add_edge(Edge(source="default_node", target=END))
        g.entry_point = "router"
        engine = GraphEngine()
        record = await engine.run(g, GraphState())
        assert record.status == ExecutionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_router_default_wildcard(self):
        """Router with __default__ edge."""

        async def router(state):
            return "unknown"

        g = GraphDefinition(name="wildcard")
        g.add_node(Node(name="router", node_type=NodeType.ROUTER, handler=router))
        g.add_node(Node(name="fallback"))
        g.add_edge(
            Edge(
                source="router",
                target="fallback",
                edge_type=EdgeType.CONDITIONAL,
                condition="__default__",
            )
        )
        g.add_edge(Edge(source="fallback", target=END))
        g.entry_point = "router"
        engine = GraphEngine()
        record = await engine.run(g, GraphState())
        assert record.status == ExecutionStatus.COMPLETED


# ============================================================================
# GraphEngine -- HITL Tests
# ============================================================================


class TestGraphEngineHITL:
    @pytest.mark.asyncio
    async def test_hitl_pauses_execution(self):
        g = GraphDefinition(name="hitl_test")
        g.add_node(Node(name="hitl_node", node_type=NodeType.HITL))
        g.add_edge(Edge(source="hitl_node", target=END))
        g.entry_point = "hitl_node"
        engine = GraphEngine()
        record = await engine.run(g, GraphState())
        assert record.status == ExecutionStatus.PAUSED
        assert len(record.checkpoints) > 0

    @pytest.mark.asyncio
    async def test_hitl_with_handler(self):
        async def hitl_handler(state):
            state["human_input"] = "pending"
            return state

        g = GraphDefinition(name="hitl_handler")
        g.add_node(Node(name="hitl_node", node_type=NodeType.HITL, handler=hitl_handler))
        g.add_edge(Edge(source="hitl_node", target=END))
        g.entry_point = "hitl_node"
        engine = GraphEngine()
        record = await engine.run(g, GraphState())
        assert record.status == ExecutionStatus.PAUSED

    @pytest.mark.asyncio
    async def test_hitl_after_normal_node(self):
        async def step1_handler(state):
            state["step1"] = True
            return state

        g = GraphDefinition(name="hitl_after")
        g.add_node(Node(name="step1", handler=step1_handler))
        g.add_node(Node(name="hitl_node", node_type=NodeType.HITL))
        g.add_node(Node(name="step2"))
        g.add_edge(Edge(source="step1", target="hitl_node"))
        g.add_edge(Edge(source="hitl_node", target="step2"))
        g.add_edge(Edge(source="step2", target=END))
        g.entry_point = "step1"
        engine = GraphEngine()
        record = await engine.run(g, GraphState())
        assert record.status == ExecutionStatus.PAUSED
        assert record.final_state.get("step1") is True


# ============================================================================
# GraphEngine -- Parallel Tests
# ============================================================================


class TestGraphEngineParallel:
    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        async def h_a(state):
            state["a"] = True
            return state

        async def h_b(state):
            state["b"] = True
            return state

        g = GraphDefinition(name="parallel_test")
        g.add_node(Node(name="fork", node_type=NodeType.PARALLEL))
        g.add_node(Node(name="branch_a", handler=h_a))
        g.add_node(Node(name="branch_b", handler=h_b))
        g.add_edge(Edge(source="fork", target="branch_a"))
        g.add_edge(Edge(source="fork", target="branch_b"))
        g.add_edge(Edge(source="branch_b", target=END))
        g.entry_point = "fork"
        engine = GraphEngine()
        record = await engine.run(g, GraphState())
        assert record.status == ExecutionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_parallel_empty_targets(self):
        """PARALLEL node with only END edges."""
        g = GraphDefinition(name="parallel_empty")
        g.add_node(Node(name="fork", node_type=NodeType.PARALLEL))
        g.add_edge(Edge(source="fork", target=END))
        g.entry_point = "fork"
        engine = GraphEngine()
        record = await engine.run(g, GraphState())
        assert record.status == ExecutionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_parallel_failure(self):
        async def failing(state):
            raise RuntimeError("parallel fail")

        g = GraphDefinition(name="parallel_fail")
        g.add_node(Node(name="fork", node_type=NodeType.PARALLEL))
        g.add_node(Node(name="bad", handler=failing))
        g.add_edge(Edge(source="fork", target="bad"))
        g.add_edge(Edge(source="bad", target=END))
        g.entry_point = "fork"
        engine = GraphEngine()
        record = await engine.run(g, GraphState())
        assert record.status == ExecutionStatus.FAILED


# ============================================================================
# GraphEngine -- run_stream() Tests
# ============================================================================


class TestGraphEngineRunStream:
    @pytest.mark.asyncio
    async def test_stream_simple(self):
        async def handler(state):
            state["x"] = 1
            return state

        g = _simple_graph(handler=handler)
        engine = GraphEngine()
        results = []
        async for nr in engine.run_stream(g, GraphState()):
            results.append(nr)
        assert len(results) >= 1
        assert results[0].status == NodeStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_stream_validation_error(self):
        g = GraphDefinition(name="bad")
        engine = GraphEngine()
        results = []
        async for nr in engine.run_stream(g, GraphState()):
            results.append(nr)
        assert len(results) == 1
        assert results[0].status == NodeStatus.FAILED
        assert "Validation" in results[0].error

    @pytest.mark.asyncio
    async def test_stream_missing_node(self):
        g = GraphDefinition(name="missing")
        g.add_node(Node(name="step1"))
        g.entry_point = "step1"
        # step1 -> step2, but step2 doesn't exist
        g.add_edge(Edge(source="step1", target="missing_node"))
        engine = GraphEngine()
        results = []
        async for nr in engine.run_stream(g, GraphState()):
            results.append(nr)
        # step1 completes, then missing_node fails
        assert any(r.status == NodeStatus.FAILED for r in results)

    @pytest.mark.asyncio
    async def test_stream_loop_protection(self):
        g = GraphDefinition(name="loop_stream")
        g.add_node(Node(name="n1"))
        g.add_edge(Edge(source="n1", target="n1"))
        g.entry_point = "n1"
        engine = GraphEngine(max_iterations=2)
        results = []
        async for nr in engine.run_stream(g, GraphState()):
            results.append(nr)
        assert any("iterations" in (r.error or "") for r in results)

    @pytest.mark.asyncio
    async def test_stream_max_nodes(self):
        g = GraphDefinition(name="max_stream")
        g.add_node(Node(name="n1"))
        g.add_edge(Edge(source="n1", target="n1"))
        g.entry_point = "n1"
        engine = GraphEngine(max_iterations=100, max_nodes=2)
        results = []
        async for nr in engine.run_stream(g, GraphState()):
            results.append(nr)
        assert any("nodes" in (r.error or "").lower() for r in results)

    @pytest.mark.asyncio
    async def test_stream_hitl(self):
        g = GraphDefinition(name="hitl_stream")
        g.add_node(Node(name="hitl", node_type=NodeType.HITL))
        g.add_edge(Edge(source="hitl", target=END))
        g.entry_point = "hitl"
        engine = GraphEngine()
        results = []
        async for nr in engine.run_stream(g, GraphState()):
            results.append(nr)
        assert any(r.router_decision == "__paused__" for r in results)

    @pytest.mark.asyncio
    async def test_stream_failed_node_stops(self):
        async def fail(state):
            raise RuntimeError("boom")

        g = _simple_graph(handler=fail)
        engine = GraphEngine()
        results = []
        async for nr in engine.run_stream(g, GraphState()):
            results.append(nr)
        assert results[-1].status == NodeStatus.FAILED


# ============================================================================
# GraphEngine -- resume() Tests
# ============================================================================


class TestGraphEngineResume:
    @pytest.mark.asyncio
    async def test_resume_from_checkpoint(self):
        """Pause at HITL, then resume."""

        async def step_after(state):
            state["resumed"] = True
            return state

        g = GraphDefinition(name="resume_test")
        g.add_node(Node(name="hitl", node_type=NodeType.HITL))
        g.add_node(Node(name="after", handler=step_after))
        g.add_edge(Edge(source="hitl", target="after"))
        g.add_edge(Edge(source="after", target=END))
        g.entry_point = "hitl"

        engine = GraphEngine()
        record1 = await engine.run(g, GraphState(x=1))
        assert record1.status == ExecutionStatus.PAUSED
        assert len(record1.checkpoints) > 0

        # Resume with checkpoint_id
        cp_id = record1.checkpoints[0]
        record2 = await engine.resume(g, checkpoint_id=cp_id, resume_input={"human": "yes"})
        assert record2.status == ExecutionStatus.COMPLETED
        assert record2.final_state.get("resumed") is True

    @pytest.mark.asyncio
    async def test_resume_from_execution_id(self):
        g = GraphDefinition(name="resume_exec")
        g.add_node(Node(name="hitl", node_type=NodeType.HITL))
        g.add_node(Node(name="done"))
        g.add_edge(Edge(source="hitl", target="done"))
        g.add_edge(Edge(source="done", target=END))
        g.entry_point = "hitl"

        engine = GraphEngine()
        record1 = await engine.run(g, GraphState())
        assert record1.status == ExecutionStatus.PAUSED

        record2 = await engine.resume(g, execution_id=record1.execution_id)
        assert record2.status == ExecutionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_resume_no_ids(self):
        g = GraphDefinition(name="no_id")
        engine = GraphEngine()
        record = await engine.resume(g)
        assert record.status == ExecutionStatus.FAILED
        assert "required" in record.error

    @pytest.mark.asyncio
    async def test_resume_checkpoint_not_found(self):
        g = GraphDefinition(name="not_found")
        engine = GraphEngine()
        record = await engine.resume(g, checkpoint_id="nonexistent")
        assert record.status == ExecutionStatus.FAILED
        assert "not found" in record.error.lower()

    @pytest.mark.asyncio
    async def test_resume_with_exception(self):
        """Resume with a handler that raises."""

        async def crash(state):
            raise RuntimeError("resume crash")

        g = GraphDefinition(name="resume_crash")
        g.add_node(Node(name="hitl", node_type=NodeType.HITL))
        g.add_node(Node(name="crash_node", handler=crash))
        g.add_edge(Edge(source="hitl", target="crash_node"))
        g.add_edge(Edge(source="crash_node", target=END))
        g.entry_point = "hitl"

        engine = GraphEngine()
        record1 = await engine.run(g, GraphState())
        cp_id = record1.checkpoints[0]
        record2 = await engine.resume(g, checkpoint_id=cp_id)
        assert record2.status == ExecutionStatus.FAILED


# ============================================================================
# GraphEngine -- Node Execution details
# ============================================================================


class TestGraphEngineNodeExecution:
    @pytest.mark.asyncio
    async def test_node_timeout(self):
        async def slow(state):
            await asyncio.sleep(10)
            return state

        g = GraphDefinition(name="timeout")
        g.add_node(Node(name="slow", handler=slow, timeout_seconds=0.05))
        g.add_edge(Edge(source="slow", target=END))
        g.entry_point = "slow"
        engine = GraphEngine()
        record = await engine.run(g, GraphState())
        assert record.status == ExecutionStatus.FAILED
        assert "Timeout" in record.error or "timeout" in record.error.lower()

    @pytest.mark.asyncio
    async def test_node_retry(self):
        call_count = 0

        async def flaky(state):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("flaky")
            state["ok"] = True
            return state

        g = GraphDefinition(name="retry")
        g.add_node(Node(name="flaky", handler=flaky, retry_count=3, retry_delay_seconds=0.01))
        g.add_edge(Edge(source="flaky", target=END))
        g.entry_point = "flaky"
        engine = GraphEngine()
        record = await engine.run(g, GraphState())
        assert record.status == ExecutionStatus.COMPLETED
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_node_returns_non_state(self):
        """Handler returning neither GraphState nor dict."""

        async def handler(state):
            return 42

        g = _simple_graph(handler=handler)
        engine = GraphEngine()
        record = await engine.run(g, GraphState())
        assert record.status == ExecutionStatus.COMPLETED


# ============================================================================
# GraphEngine -- cancel, stats, management
# ============================================================================


class TestGraphEngineManagement:
    @pytest.mark.asyncio
    async def test_cancel_execution(self):
        g = GraphDefinition(name="cancel_test")
        g.add_node(Node(name="hitl", node_type=NodeType.HITL))
        g.add_edge(Edge(source="hitl", target=END))
        g.entry_point = "hitl"
        engine = GraphEngine()
        record = await engine.run(g, GraphState())
        assert record.status == ExecutionStatus.PAUSED

        cancelled = await engine.cancel(record.execution_id)
        assert cancelled is True

    @pytest.mark.asyncio
    async def test_cancel_nonexistent(self):
        engine = GraphEngine()
        assert await engine.cancel("nope") is False

    def test_get_execution(self):
        engine = GraphEngine()
        assert engine.get_execution("nope") is None

    def test_list_executions(self):
        engine = GraphEngine()
        assert engine.list_executions() == []

    def test_stats(self):
        engine = GraphEngine()
        s = engine.stats()
        assert "total_executions" in s
        assert "running" in s
        assert "state_manager" in s


# ============================================================================
# StateManager Tests
# ============================================================================


class TestStateManager:
    def test_create_execution(self):
        sm = StateManager(storage_dir=str(Path(tempfile.gettempdir()) / "test_graph_sm"))
        record = sm.create_execution("my_graph", GraphState(x=1))
        assert record.graph_name == "my_graph"
        assert record.status == ExecutionStatus.RUNNING

    def test_get_execution(self):
        sm = StateManager(storage_dir=str(Path(tempfile.gettempdir()) / "test_graph_sm"))
        record = sm.create_execution("g", GraphState())
        assert sm.get_execution(record.execution_id) is record
        assert sm.get_execution("nope") is None

    def test_update_execution(self):
        sm = StateManager(storage_dir=str(Path(tempfile.gettempdir()) / "test_graph_sm"))
        record = sm.create_execution("g", GraphState())
        record.status = ExecutionStatus.COMPLETED
        sm.update_execution(record)
        assert sm.get_execution(record.execution_id).status == ExecutionStatus.COMPLETED

    def test_list_executions(self):
        sm = StateManager(storage_dir=str(Path(tempfile.gettempdir()) / "test_graph_sm"))
        sm.create_execution("a", GraphState())
        sm.create_execution("b", GraphState())
        assert len(sm.list_executions()) == 2

    def test_list_executions_with_filter(self):
        sm = StateManager(storage_dir=str(Path(tempfile.gettempdir()) / "test_graph_sm"))
        sm.create_execution("a", GraphState())
        r = sm.create_execution("b", GraphState())
        r.status = ExecutionStatus.COMPLETED
        sm.update_execution(r)
        running = sm.list_executions(status=ExecutionStatus.RUNNING)
        assert len(running) == 1

    def test_create_checkpoint(self):
        sm = StateManager(storage_dir=str(Path(tempfile.gettempdir()) / "test_graph_sm"))
        record = sm.create_execution("g", GraphState())
        cp = sm.create_checkpoint(record.execution_id, "g", "n1", GraphState(x=1))
        assert cp.current_node == "n1"
        assert cp.execution_id == record.execution_id

    def test_get_checkpoint(self):
        sm = StateManager(storage_dir=str(Path(tempfile.gettempdir()) / "test_graph_sm"))
        cp = sm.create_checkpoint("exec1", "g", "n1", GraphState())
        assert sm.get_checkpoint(cp.checkpoint_id) is cp
        assert sm.get_checkpoint("nope") is None

    def test_get_latest_checkpoint(self):
        sm = StateManager(storage_dir=str(Path(tempfile.gettempdir()) / "test_graph_sm"))
        sm.create_checkpoint("exec1", "g", "n1", GraphState())
        cp2 = sm.create_checkpoint("exec1", "g", "n2", GraphState())
        latest = sm.get_latest_checkpoint("exec1")
        assert latest is not None

    def test_delete_checkpoint(self):
        sm = StateManager(storage_dir=str(Path(tempfile.gettempdir()) / "test_graph_sm"))
        cp = sm.create_checkpoint("exec1", "g", "n1", GraphState())
        assert sm.get_checkpoint(cp.checkpoint_id) is not None
        sm.delete_checkpoint(cp.checkpoint_id)
        assert cp.checkpoint_id not in sm._checkpoints

    def test_list_checkpoints(self):
        sm = StateManager(storage_dir=str(Path(tempfile.gettempdir()) / "test_graph_sm"))
        sm.create_checkpoint("exec1", "g", "n1", GraphState())
        sm.create_checkpoint("exec2", "g", "n1", GraphState())
        all_cps = sm.list_checkpoints()
        assert len(all_cps) == 2
        filtered = sm.list_checkpoints(execution_id="exec1")
        assert len(filtered) == 1

    def test_restore_state(self):
        sm = StateManager(storage_dir=str(Path(tempfile.gettempdir()) / "test_graph_sm"))
        cp = sm.create_checkpoint("exec1", "g", "n1", GraphState(val=42))
        state, node = sm.restore_state(cp.checkpoint_id)
        assert state is not None
        assert state["val"] == 42
        assert node == "n1"

    def test_restore_state_not_found(self):
        sm = StateManager(storage_dir=str(Path(tempfile.gettempdir()) / "test_graph_sm"))
        state, node = sm.restore_state("nope")
        assert state is None
        assert node == ""

    def test_restore_from_latest(self):
        sm = StateManager(storage_dir=str(Path(tempfile.gettempdir()) / "test_graph_sm"))
        sm.create_checkpoint("exec1", "g", "n1", GraphState(a=1))
        sm.create_checkpoint("exec1", "g", "n2", GraphState(a=2))
        state, node = sm.restore_from_latest("exec1")
        assert state is not None

    def test_restore_from_latest_not_found(self):
        sm = StateManager(storage_dir=str(Path(tempfile.gettempdir()) / "test_graph_sm"))
        state, node = sm.restore_from_latest("nope")
        assert state is None

    def test_save_checkpoint_to_disk(self, tmp_path):
        sm = StateManager(storage_dir=str(tmp_path))
        cp = sm.create_checkpoint("exec1", "g", "n1", GraphState(x=1))
        assert sm.save_checkpoint_to_disk(cp.checkpoint_id) is True
        assert (tmp_path / f"{cp.checkpoint_id}.json").exists()

    def test_save_checkpoint_not_found(self, tmp_path):
        sm = StateManager(storage_dir=str(tmp_path))
        assert sm.save_checkpoint_to_disk("nonexistent") is False

    def test_save_all_checkpoints(self, tmp_path):
        sm = StateManager(storage_dir=str(tmp_path))
        sm.create_checkpoint("exec1", "g", "n1", GraphState())
        sm.create_checkpoint("exec2", "g", "n2", GraphState())
        count = sm.save_all_checkpoints()
        assert count == 2

    def test_load_from_disk(self, tmp_path):
        sm = StateManager(storage_dir=str(tmp_path))
        cp = sm.create_checkpoint("exec1", "g", "n1", GraphState(val=99))
        sm.save_checkpoint_to_disk(cp.checkpoint_id)
        # Clear in-memory cache
        sm._checkpoints.clear()
        loaded = sm.get_checkpoint(cp.checkpoint_id)
        assert loaded is not None
        assert loaded.state.get("val") == 99

    def test_cleanup(self, tmp_path):
        sm = StateManager(storage_dir=str(tmp_path))
        # Create a checkpoint with old timestamp
        cp = sm.create_checkpoint("exec1", "g", "n1", GraphState())
        cp.created_at = "2000-01-01T00:00:00Z"
        sm._checkpoints[cp.checkpoint_id] = cp
        removed = sm.cleanup(max_age_days=1)
        assert removed >= 1

    def test_cleanup_max_limit(self, tmp_path):
        sm = StateManager(storage_dir=str(tmp_path))
        for i in range(5):
            sm.create_checkpoint(f"exec{i}", "g", f"n{i}", GraphState())
        removed = sm.cleanup(max_checkpoints=2)
        assert removed >= 3

    def test_stats(self):
        sm = StateManager(storage_dir=str(Path(tempfile.gettempdir()) / "test_graph_sm"))
        s = sm.stats()
        assert "checkpoints" in s
        assert "executions" in s
        assert "storage_dir" in s


# ============================================================================
# Checkpoint / ExecutionRecord / NodeResult / Edge / Node serialization
# ============================================================================


class TestTypeSerialization:
    def test_checkpoint_to_dict_from_dict(self):
        cp = Checkpoint(
            execution_id="e1", graph_name="g", current_node="n1", state={"a": 1}, history=[]
        )
        d = cp.to_dict()
        cp2 = Checkpoint.from_dict(d)
        assert cp2.execution_id == "e1"

    def test_checkpoint_to_json_from_json(self):
        cp = Checkpoint(state={"x": 1})
        j = cp.to_json()
        cp2 = Checkpoint.from_json(j)
        assert cp2.state == {"x": 1}

    def test_execution_record_properties(self):
        r = ExecutionRecord()
        assert r.node_count == 0
        assert r.success_rate == 0.0
        r.node_results = [
            NodeResult(node_name="n1", status=NodeStatus.COMPLETED),
            NodeResult(node_name="n2", status=NodeStatus.FAILED),
        ]
        assert r.node_count == 2
        assert r.success_rate == 0.5

    def test_execution_record_to_dict(self):
        r = ExecutionRecord(graph_name="test")
        d = r.to_dict()
        assert d["graph_name"] == "test"

    def test_node_result_to_dict(self):
        nr = NodeResult(
            node_name="n1", status=NodeStatus.COMPLETED, router_decision="go", retry_attempts=2
        )
        d = nr.to_dict()
        assert d["router_decision"] == "go"
        assert d["retries"] == 2

    def test_edge_to_dict(self):
        e = Edge(source="a", target="b", condition="yes")
        d = e.to_dict()
        assert d["condition"] == "yes"

    def test_node_to_dict(self):
        n = Node(name="n1", node_type=NodeType.ROUTER)
        d = n.to_dict()
        assert d["type"] == "router"
