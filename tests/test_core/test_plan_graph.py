"""Tests fuer PlanGraph."""

import pytest
from jarvis.core.plan_graph import PlanGraph, CycleError
from jarvis.models import ActionPlan, PlanNode, PlannedAction, RiskLevel


class TestPlanGraph:
    def test_add_node(self):
        graph = PlanGraph()
        node = PlanNode(id="n1", tool="read_file", params={"path": "/tmp"})
        graph.add_node(node)
        assert graph.node_count == 1
        assert graph.get_node("n1") is not None

    def test_topological_order_simple(self):
        graph = PlanGraph()
        n1 = PlanNode(id="n1", tool="read_file")
        n2 = PlanNode(id="n2", tool="write_file", depends_on=["n1"])
        graph.add_node(n1)
        graph.add_node(n2)

        order = graph.topological_order()
        assert order.index("n1") < order.index("n2")

    def test_topological_order_parallel(self):
        graph = PlanGraph()
        graph.add_node(PlanNode(id="n1", tool="a"))
        graph.add_node(PlanNode(id="n2", tool="b"))
        graph.add_node(PlanNode(id="n3", tool="c", depends_on=["n1", "n2"]))

        order = graph.topological_order()
        assert order.index("n1") < order.index("n3")
        assert order.index("n2") < order.index("n3")

    def test_cycle_detection(self):
        graph = PlanGraph()
        graph.add_node(PlanNode(id="n1", tool="a", depends_on=["n2"]))
        graph.add_node(PlanNode(id="n2", tool="b", depends_on=["n1"]))

        with pytest.raises(CycleError):
            graph.topological_order()

    def test_get_ready_nodes(self):
        graph = PlanGraph()
        graph.add_node(PlanNode(id="n1", tool="a"))
        graph.add_node(PlanNode(id="n2", tool="b", depends_on=["n1"]))
        graph.add_node(PlanNode(id="n3", tool="c"))

        ready = graph.get_ready_nodes(set())
        assert "n1" in ready
        assert "n3" in ready
        assert "n2" not in ready

        ready2 = graph.get_ready_nodes({"n1"})
        assert "n2" in ready2

    def test_validate_ok(self):
        graph = PlanGraph()
        graph.add_node(PlanNode(id="n1", tool="a"))
        graph.add_node(PlanNode(id="n2", tool="b", depends_on=["n1"]))
        problems = graph.validate()
        assert len(problems) == 0

    def test_validate_missing_dependency(self):
        graph = PlanGraph()
        graph.add_node(PlanNode(id="n1", tool="a", depends_on=["nonexistent"]))
        problems = graph.validate()
        assert len(problems) > 0

    def test_from_action_plan(self):
        plan = ActionPlan(
            goal="Test",
            steps=[
                PlannedAction(tool="read_file", params={"path": "/tmp"}),
                PlannedAction(tool="write_file", params={"path": "/out"}, depends_on=[0]),
            ],
        )
        graph = PlanGraph.from_action_plan(plan)
        assert graph.node_count == 2
        order = graph.topological_order()
        assert len(order) == 2

    def test_empty_graph(self):
        graph = PlanGraph()
        assert graph.node_count == 0
        assert graph.topological_order() == []
        assert graph.get_ready_nodes(set()) == []
        assert graph.validate() == []

    def test_nodes_property(self):
        graph = PlanGraph()
        graph.add_node(PlanNode(id="n1", tool="a"))
        graph.add_node(PlanNode(id="n2", tool="b"))
        assert len(graph.nodes) == 2
