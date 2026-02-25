"""Tests fuer AgentKernel."""

import pytest
from jarvis.core.kernel import AgentKernel, InvalidTransitionError
from jarvis.core.plan_graph import PlanGraph
from jarvis.models import KernelState, PlanNode, ToolResult


class TestAgentKernel:
    def setup_method(self):
        self.kernel = AgentKernel(session_id="test-session")

    def test_initial_state(self):
        assert self.kernel.state == KernelState.IDLE

    def test_valid_transition(self):
        self.kernel.transition(KernelState.PLANNING)
        assert self.kernel.state == KernelState.PLANNING

    def test_invalid_transition(self):
        with pytest.raises(InvalidTransitionError):
            self.kernel.transition(KernelState.REFLECTING)

    def test_transition_chain(self):
        self.kernel.transition(KernelState.PLANNING)
        self.kernel.transition(KernelState.GATING)
        self.kernel.transition(KernelState.EXECUTING)
        self.kernel.transition(KernelState.REFLECTING)
        self.kernel.transition(KernelState.DONE)
        assert self.kernel.state == KernelState.DONE

    def test_error_transition(self):
        self.kernel.transition(KernelState.PLANNING)
        self.kernel.transition(KernelState.ERROR)
        assert self.kernel.state == KernelState.ERROR

    def test_recovery_from_error(self):
        self.kernel.transition(KernelState.ERROR)
        self.kernel.transition(KernelState.IDLE)
        assert self.kernel.state == KernelState.IDLE

    def test_checkpoint_and_rollback(self):
        self.kernel.transition(KernelState.PLANNING)
        cp = self.kernel.checkpoint({"test": "data"})

        self.kernel.transition(KernelState.GATING)
        self.kernel.transition(KernelState.EXECUTING)

        success = self.kernel.rollback(cp.id)
        assert success
        assert self.kernel.state == KernelState.PLANNING

    def test_rollback_nonexistent(self):
        assert not self.kernel.rollback("nonexistent")

    @pytest.mark.asyncio
    async def test_execute_plan_dry_run(self):
        graph = PlanGraph()
        graph.add_node(PlanNode(id="n1", tool="read_file", params={"path": "/tmp"}))
        graph.add_node(PlanNode(id="n2", tool="write_file", depends_on=["n1"]))

        self.kernel.transition(KernelState.PLANNING)
        self.kernel.transition(KernelState.GATING)

        results = await self.kernel.execute_plan(graph)
        assert len(results) == 2
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_execute_plan_with_fn(self):
        graph = PlanGraph()
        graph.add_node(PlanNode(id="n1", tool="read_file"))

        async def mock_execute(tool, params):
            return ToolResult(tool_name=tool, content="mock result")

        self.kernel.transition(KernelState.PLANNING)
        self.kernel.transition(KernelState.GATING)

        results = await self.kernel.execute_plan(graph, execute_fn=mock_execute)
        assert len(results) == 1
        assert results[0].content == "mock result"

    @pytest.mark.asyncio
    async def test_execute_plan_with_error(self):
        graph = PlanGraph()
        graph.add_node(PlanNode(id="n1", tool="bad_tool"))

        async def failing_execute(tool, params):
            raise RuntimeError("Tool failed")

        self.kernel.transition(KernelState.PLANNING)
        self.kernel.transition(KernelState.GATING)

        results = await self.kernel.execute_plan(graph, execute_fn=failing_execute)
        assert len(results) == 1
        assert results[0].is_error

    def test_reset(self):
        self.kernel.transition(KernelState.PLANNING)
        self.kernel.reset()
        assert self.kernel.state == KernelState.IDLE
        assert self.kernel.completed_nodes == []
        assert self.kernel.tool_results == []

    def test_properties(self):
        assert self.kernel.completed_nodes == []
        assert self.kernel.tool_results == []
        assert self.kernel.error is None
