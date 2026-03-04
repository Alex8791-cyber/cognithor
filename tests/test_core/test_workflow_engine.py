"""Tests for the DAG Workflow Engine.

Covers: schema validation, cycle detection, topological sort, node execution
(tool, LLM, condition, human approval), retry strategies, parallel execution,
dependency propagation, template resolution, condition evaluation,
checkpoint/resume, and full integration workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.core.workflow_engine import WorkflowEngine
from jarvis.core.workflow_schema import (
    NodeResult,
    NodeStatus,
    NodeType,
    RetryStrategy,
    WorkflowDefinition,
    WorkflowNode,
    WorkflowRun,
    WorkflowValidationError,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

@dataclass
class MockToolOutput:
    """Mimics the object returned by ``mcp_client.call_tool()``."""

    content: str = "tool output"
    is_error: bool = False


@dataclass
class MockGateDecision:
    """Mimics a single GateDecision from the Gatekeeper."""

    is_allowed: bool = True
    reason: str = ""


def _tool(
    id: str,
    tool_name: str = "read_file",
    params: dict[str, Any] | None = None,
    depends_on: list[str] | None = None,
    **kwargs: Any,
) -> WorkflowNode:
    return WorkflowNode(
        id=id,
        type=NodeType.TOOL,
        name=id,
        tool_name=tool_name,
        tool_params=params or {},
        depends_on=depends_on or [],
        **kwargs,
    )


def _llm(
    id: str,
    prompt: str = "Summarize",
    depends_on: list[str] | None = None,
    **kwargs: Any,
) -> WorkflowNode:
    return WorkflowNode(
        id=id,
        type=NodeType.LLM,
        name=id,
        prompt=prompt,
        depends_on=depends_on or [],
        **kwargs,
    )


def _condition(
    id: str,
    condition: str = "true",
    on_true: str | None = None,
    on_false: str | None = None,
    depends_on: list[str] | None = None,
) -> WorkflowNode:
    return WorkflowNode(
        id=id,
        type=NodeType.CONDITION,
        name=id,
        condition=condition,
        on_true=on_true,
        on_false=on_false,
        depends_on=depends_on or [],
    )


def _approval(
    id: str,
    message: str = "Approve?",
    depends_on: list[str] | None = None,
) -> WorkflowNode:
    return WorkflowNode(
        id=id,
        type=NodeType.HUMAN_APPROVAL,
        name=id,
        approval_message=message,
        depends_on=depends_on or [],
    )


def _wf(name: str, nodes: list[WorkflowNode], **kwargs: Any) -> WorkflowDefinition:
    return WorkflowDefinition(name=name, nodes=nodes, **kwargs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_mcp() -> AsyncMock:
    mcp = AsyncMock()
    mcp.call_tool = AsyncMock(return_value=MockToolOutput())
    return mcp


@pytest.fixture()
def mock_llm() -> AsyncMock:
    return AsyncMock(return_value="LLM response text")


@pytest.fixture()
def mock_approval_yes() -> AsyncMock:
    return AsyncMock(return_value=True)


@pytest.fixture()
def mock_approval_no() -> AsyncMock:
    return AsyncMock(return_value=False)


@pytest.fixture()
def engine(mock_mcp: AsyncMock, mock_llm: AsyncMock, mock_approval_yes: AsyncMock, tmp_path: Path) -> WorkflowEngine:
    return WorkflowEngine(
        mcp_client=mock_mcp,
        llm_func=mock_llm,
        approval_func=mock_approval_yes,
        checkpoint_dir=tmp_path / "checkpoints",
    )


@pytest.fixture()
def bare_engine() -> WorkflowEngine:
    """Engine without any subsystem dependencies."""
    return WorkflowEngine()


# ===========================================================================
# 1. Schema Tests
# ===========================================================================

class TestWorkflowSchema:
    """Test the Pydantic models for workflow definitions."""

    def test_create_tool_node(self) -> None:
        node = _tool("a", tool_name="web_search", params={"query": "test"})
        assert node.id == "a"
        assert node.type == NodeType.TOOL
        assert node.tool_name == "web_search"
        assert node.tool_params == {"query": "test"}

    def test_create_llm_node(self) -> None:
        node = _llm("b", prompt="Explain ${a.output}")
        assert node.type == NodeType.LLM
        assert node.prompt == "Explain ${a.output}"

    def test_create_condition_node(self) -> None:
        node = _condition("c", condition='${a.status} == "success"', on_true="d", on_false="e")
        assert node.type == NodeType.CONDITION
        assert node.on_true == "d"
        assert node.on_false == "e"

    def test_workflow_definition_node_ids(self) -> None:
        wf = _wf("test", [_tool("a"), _tool("b"), _llm("c")])
        assert wf.node_ids == {"a", "b", "c"}

    def test_workflow_get_node_found(self) -> None:
        wf = _wf("test", [_tool("x")])
        assert wf.get_node("x") is not None
        assert wf.get_node("x").id == "x"  # type: ignore[union-attr]

    def test_workflow_get_node_not_found(self) -> None:
        wf = _wf("test", [_tool("x")])
        assert wf.get_node("missing") is None

    def test_node_result_defaults(self) -> None:
        nr = NodeResult(node_id="a")
        assert nr.status == NodeStatus.PENDING
        assert nr.output == ""
        assert nr.error is None
        assert nr.retry_count == 0

    def test_workflow_run_not_complete_when_empty(self) -> None:
        run = WorkflowRun()
        assert run.is_complete is False
        assert run.is_success is False

    def test_workflow_run_complete_all_success(self) -> None:
        run = WorkflowRun(
            node_results={
                "a": NodeResult(node_id="a", status=NodeStatus.SUCCESS),
                "b": NodeResult(node_id="b", status=NodeStatus.SUCCESS),
            }
        )
        assert run.is_complete is True
        assert run.is_success is True
        assert run.failed_nodes == []

    def test_workflow_run_complete_with_failure(self) -> None:
        run = WorkflowRun(
            node_results={
                "a": NodeResult(node_id="a", status=NodeStatus.SUCCESS),
                "b": NodeResult(node_id="b", status=NodeStatus.FAILURE, error="boom"),
            }
        )
        assert run.is_complete is True
        assert run.is_success is False
        assert run.failed_nodes == ["b"]

    def test_workflow_run_skipped_counts_as_success(self) -> None:
        run = WorkflowRun(
            node_results={
                "a": NodeResult(node_id="a", status=NodeStatus.SUCCESS),
                "b": NodeResult(node_id="b", status=NodeStatus.SKIPPED),
            }
        )
        assert run.is_complete is True
        assert run.is_success is True

    def test_workflow_from_yaml(self) -> None:
        yaml_str = """
name: "YAML Test"
nodes:
  - id: a
    type: tool
    tool_name: read_file
    tool_params:
      path: /test
  - id: b
    type: llm
    prompt: "Summarize"
    depends_on: [a]
"""
        wf = WorkflowDefinition.from_yaml(yaml_str)
        assert wf.name == "YAML Test"
        assert len(wf.nodes) == 2
        assert wf.nodes[0].tool_name == "read_file"
        assert wf.nodes[1].depends_on == ["a"]


# ===========================================================================
# 2. Validation Tests
# ===========================================================================

class TestValidation:
    """Test workflow validation (structural correctness)."""

    def test_valid_workflow_no_errors(self, engine: WorkflowEngine) -> None:
        wf = _wf("ok", [_tool("a"), _tool("b", depends_on=["a"])])
        assert engine.validate(wf) == []

    def test_duplicate_node_ids(self, engine: WorkflowEngine) -> None:
        wf = _wf("dup", [_tool("a"), _tool("a")])
        errors = engine.validate(wf)
        assert any("Duplicate" in e for e in errors)

    def test_unknown_dependency(self, engine: WorkflowEngine) -> None:
        wf = _wf("bad", [_tool("a", depends_on=["ghost"])])
        errors = engine.validate(wf)
        assert any("unknown node 'ghost'" in e for e in errors)

    def test_unknown_condition_on_true(self, engine: WorkflowEngine) -> None:
        wf = _wf("bad", [_condition("c", on_true="ghost")])
        errors = engine.validate(wf)
        assert any("on_true" in e for e in errors)

    def test_unknown_condition_on_false(self, engine: WorkflowEngine) -> None:
        wf = _wf("bad", [_condition("c", on_false="ghost")])
        errors = engine.validate(wf)
        assert any("on_false" in e for e in errors)

    def test_tool_node_without_tool_name(self, engine: WorkflowEngine) -> None:
        node = WorkflowNode(id="t", type=NodeType.TOOL)
        wf = _wf("bad", [node])
        errors = engine.validate(wf)
        assert any("missing tool_name" in e for e in errors)

    def test_llm_node_without_prompt(self, engine: WorkflowEngine) -> None:
        node = WorkflowNode(id="l", type=NodeType.LLM)
        wf = _wf("bad", [node])
        errors = engine.validate(wf)
        assert any("missing prompt" in e for e in errors)

    def test_condition_node_without_condition(self, engine: WorkflowEngine) -> None:
        node = WorkflowNode(id="c", type=NodeType.CONDITION)
        wf = _wf("bad", [node])
        errors = engine.validate(wf)
        assert any("missing condition" in e for e in errors)

    def test_template_self_reference_in_prompt(self, engine: WorkflowEngine) -> None:
        node = WorkflowNode(id="x", type=NodeType.LLM, prompt="Use ${x.output}")
        wf = _wf("bad", [node])
        errors = engine.validate(wf)
        assert any("references itself" in e for e in errors)

    def test_template_self_reference_in_tool_params(self, engine: WorkflowEngine) -> None:
        node = WorkflowNode(
            id="t", type=NodeType.TOOL, tool_name="do_thing",
            tool_params={"data": "${t.output}"},
        )
        wf = _wf("bad", [node])
        errors = engine.validate(wf)
        assert any("references itself" in e for e in errors)

    def test_template_self_reference_in_condition(self, engine: WorkflowEngine) -> None:
        node = WorkflowNode(
            id="c", type=NodeType.CONDITION,
            condition='${c.status} == "success"', on_true="a",
        )
        wf = _wf("bad", [_tool("a"), node])
        errors = engine.validate(wf)
        assert any("references itself" in e for e in errors)

    def test_template_cross_reference_is_ok(self, engine: WorkflowEngine) -> None:
        wf = _wf("ok", [
            _tool("a"),
            _llm("b", prompt="Use ${a.output}", depends_on=["a"]),
        ])
        assert engine.validate(wf) == []

    async def test_execute_rejects_invalid_workflow(self, engine: WorkflowEngine) -> None:
        wf = _wf("bad", [_tool("a", depends_on=["ghost"])])
        with pytest.raises(WorkflowValidationError):
            await engine.execute(wf)


# ===========================================================================
# 3. Cycle Detection
# ===========================================================================

class TestCycleDetection:
    """Test DAG cycle detection."""

    def test_no_cycle(self, engine: WorkflowEngine) -> None:
        wf = _wf("ok", [_tool("a"), _tool("b", depends_on=["a"])])
        assert engine.validate(wf) == []

    def test_simple_cycle(self, engine: WorkflowEngine) -> None:
        wf = _wf("cycle", [
            _tool("a", depends_on=["b"]),
            _tool("b", depends_on=["a"]),
        ])
        errors = engine.validate(wf)
        assert any("Cycle" in e for e in errors)

    def test_three_node_cycle(self, engine: WorkflowEngine) -> None:
        wf = _wf("cycle3", [
            _tool("a", depends_on=["c"]),
            _tool("b", depends_on=["a"]),
            _tool("c", depends_on=["b"]),
        ])
        errors = engine.validate(wf)
        assert any("Cycle" in e for e in errors)

    def test_self_cycle(self, engine: WorkflowEngine) -> None:
        wf = _wf("self", [_tool("a", depends_on=["a"])])
        errors = engine.validate(wf)
        assert any("Cycle" in e for e in errors)


# ===========================================================================
# 4. Topological Sort
# ===========================================================================

class TestTopologicalSort:
    """Test Kahn's algorithm for execution layer ordering."""

    def test_linear_chain(self, engine: WorkflowEngine) -> None:
        wf = _wf("linear", [
            _tool("a"),
            _tool("b", depends_on=["a"]),
            _tool("c", depends_on=["b"]),
        ])
        layers = engine.topological_sort(wf)
        assert layers == [["a"], ["b"], ["c"]]

    def test_parallel_start(self, engine: WorkflowEngine) -> None:
        wf = _wf("par", [_tool("a"), _tool("b"), _tool("c")])
        layers = engine.topological_sort(wf)
        assert len(layers) == 1
        assert set(layers[0]) == {"a", "b", "c"}

    def test_diamond(self, engine: WorkflowEngine) -> None:
        wf = _wf("diamond", [
            _tool("a"),
            _tool("b", depends_on=["a"]),
            _tool("c", depends_on=["a"]),
            _tool("d", depends_on=["b", "c"]),
        ])
        layers = engine.topological_sort(wf)
        assert layers[0] == ["a"]
        assert set(layers[1]) == {"b", "c"}
        assert layers[2] == ["d"]

    def test_cycle_raises(self, engine: WorkflowEngine) -> None:
        wf = _wf("cycle", [
            _tool("a", depends_on=["b"]),
            _tool("b", depends_on=["a"]),
        ])
        with pytest.raises(WorkflowValidationError, match="Cycle"):
            engine.topological_sort(wf)


# ===========================================================================
# 5. Template Resolution
# ===========================================================================

class TestTemplateResolution:
    """Test ${node_id.field} template substitution."""

    def test_resolve_output(self, engine: WorkflowEngine) -> None:
        run = WorkflowRun(
            node_results={"a": NodeResult(node_id="a", output="hello")}
        )
        result = engine._resolve_template("Got: ${a.output}", run)
        assert result == "Got: hello"

    def test_resolve_status(self, engine: WorkflowEngine) -> None:
        run = WorkflowRun(
            node_results={"a": NodeResult(node_id="a", status=NodeStatus.SUCCESS)}
        )
        result = engine._resolve_template("${a.status}", run)
        assert result == "success"

    def test_resolve_multiple(self, engine: WorkflowEngine) -> None:
        run = WorkflowRun(
            node_results={
                "a": NodeResult(node_id="a", output="X"),
                "b": NodeResult(node_id="b", output="Y"),
            }
        )
        result = engine._resolve_template("${a.output} and ${b.output}", run)
        assert result == "X and Y"

    def test_unknown_node_left_unresolved(self, engine: WorkflowEngine) -> None:
        run = WorkflowRun()
        result = engine._resolve_template("${missing.output}", run)
        assert result == "${missing.output}"

    def test_unknown_field_empty(self, engine: WorkflowEngine) -> None:
        run = WorkflowRun(
            node_results={"a": NodeResult(node_id="a")}
        )
        result = engine._resolve_template("${a.nonexistent}", run)
        assert result == ""

    def test_no_templates_unchanged(self, engine: WorkflowEngine) -> None:
        run = WorkflowRun()
        result = engine._resolve_template("plain text", run)
        assert result == "plain text"

    def test_resolve_params(self, engine: WorkflowEngine) -> None:
        run = WorkflowRun(
            node_results={"s": NodeResult(node_id="s", output="Python asyncio")}
        )
        params = {"query": "${s.output}", "limit": 5}
        resolved = engine._resolve_params(params, run)
        assert resolved == {"query": "Python asyncio", "limit": 5}


# ===========================================================================
# 6. Condition Evaluation
# ===========================================================================

class TestConditionEvaluation:
    """Test safe condition expression evaluation."""

    def test_true_literal(self, engine: WorkflowEngine) -> None:
        run = WorkflowRun()
        assert engine._evaluate_condition("true", run) is True

    def test_false_literal(self, engine: WorkflowEngine) -> None:
        run = WorkflowRun()
        assert engine._evaluate_condition("false", run) is False

    def test_equality_true(self, engine: WorkflowEngine) -> None:
        run = WorkflowRun()
        assert engine._evaluate_condition('"hello" == "hello"', run) is True

    def test_equality_false(self, engine: WorkflowEngine) -> None:
        run = WorkflowRun()
        assert engine._evaluate_condition('"a" == "b"', run) is False

    def test_inequality(self, engine: WorkflowEngine) -> None:
        run = WorkflowRun()
        assert engine._evaluate_condition('"a" != "b"', run) is True

    def test_inequality_false(self, engine: WorkflowEngine) -> None:
        run = WorkflowRun()
        assert engine._evaluate_condition('"x" != "x"', run) is False

    def test_contains_true(self, engine: WorkflowEngine) -> None:
        run = WorkflowRun()
        assert engine._evaluate_condition('"hello world" contains "world"', run) is True

    def test_contains_false(self, engine: WorkflowEngine) -> None:
        run = WorkflowRun()
        assert engine._evaluate_condition('"hello" contains "xyz"', run) is False

    def test_template_in_condition(self, engine: WorkflowEngine) -> None:
        run = WorkflowRun(
            node_results={"a": NodeResult(node_id="a", status=NodeStatus.SUCCESS)}
        )
        assert engine._evaluate_condition('${a.status} == "success"', run) is True

    def test_unknown_expression_returns_false(self, engine: WorkflowEngine) -> None:
        run = WorkflowRun()
        assert engine._evaluate_condition("something weird", run) is False


# ===========================================================================
# 7. Tool Node Execution
# ===========================================================================

class TestToolNodeExecution:
    """Test tool node execution via MCP client."""

    async def test_successful_tool(self, engine: WorkflowEngine, mock_mcp: AsyncMock) -> None:
        wf = _wf("t", [_tool("a")])
        run = await engine.execute(wf)
        assert run.is_success
        assert run.node_results["a"].status == NodeStatus.SUCCESS
        assert run.node_results["a"].output == "tool output"
        mock_mcp.call_tool.assert_called_once()

    async def test_tool_with_error(self, engine: WorkflowEngine, mock_mcp: AsyncMock) -> None:
        mock_mcp.call_tool.return_value = MockToolOutput(content="fail", is_error=True)
        wf = _wf("t", [_tool("a")])
        run = await engine.execute(wf)
        assert run.node_results["a"].status == NodeStatus.FAILURE

    async def test_tool_without_mcp_client(self, bare_engine: WorkflowEngine) -> None:
        wf = _wf("t", [_tool("a")])
        run = await bare_engine.execute(wf)
        assert run.node_results["a"].status == NodeStatus.FAILURE
        assert "No MCP client" in (run.node_results["a"].error or "")

    async def test_tool_template_params(self, engine: WorkflowEngine, mock_mcp: AsyncMock) -> None:
        mock_mcp.call_tool.return_value = MockToolOutput(content="data from search")
        wf = _wf("t", [
            _tool("a", tool_name="search", params={"query": "test"}),
            _tool("b", tool_name="save", params={"content": "${a.output}"}, depends_on=["a"]),
        ])
        run = await engine.execute(wf)
        assert run.is_success
        # Second call should have resolved template
        second_call = mock_mcp.call_tool.call_args_list[1]
        assert second_call[0][1]["content"] == "data from search"

    async def test_tool_gatekeeper_blocks(self, mock_mcp: AsyncMock, mock_llm: AsyncMock) -> None:
        gk = MagicMock()
        gk.evaluate.return_value = [MockGateDecision(is_allowed=False, reason="too risky")]
        session = MagicMock()
        engine = WorkflowEngine(mcp_client=mock_mcp, gatekeeper=gk, llm_func=mock_llm)
        wf = _wf("t", [_tool("a")])
        run = await engine.execute(wf, session=session)
        assert run.node_results["a"].status == NodeStatus.FAILURE
        assert "Gatekeeper blocked" in (run.node_results["a"].error or "")
        mock_mcp.call_tool.assert_not_called()

    async def test_tool_gatekeeper_allows(self, mock_mcp: AsyncMock, mock_llm: AsyncMock) -> None:
        gk = MagicMock()
        gk.evaluate.return_value = [MockGateDecision(is_allowed=True)]
        session = MagicMock()
        engine = WorkflowEngine(mcp_client=mock_mcp, gatekeeper=gk, llm_func=mock_llm)
        wf = _wf("t", [_tool("a")])
        run = await engine.execute(wf, session=session)
        assert run.node_results["a"].status == NodeStatus.SUCCESS
        mock_mcp.call_tool.assert_called_once()


# ===========================================================================
# 8. LLM Node Execution
# ===========================================================================

class TestLLMNodeExecution:
    """Test LLM node execution."""

    async def test_successful_llm(self, engine: WorkflowEngine) -> None:
        wf = _wf("l", [_llm("a", prompt="Explain X")])
        run = await engine.execute(wf)
        assert run.is_success
        assert run.node_results["a"].output == "LLM response text"

    async def test_llm_without_func(self, bare_engine: WorkflowEngine) -> None:
        wf = _wf("l", [_llm("a")])
        run = await bare_engine.execute(wf)
        assert run.node_results["a"].status == NodeStatus.FAILURE
        assert "No LLM function" in (run.node_results["a"].error or "")

    async def test_llm_template_prompt(self, engine: WorkflowEngine, mock_mcp: AsyncMock, mock_llm: AsyncMock) -> None:
        mock_mcp.call_tool.return_value = MockToolOutput(content="raw data")
        wf = _wf("l", [
            _tool("search", tool_name="web_search"),
            _llm("summarize", prompt="Summarize: ${search.output}", depends_on=["search"]),
        ])
        run = await engine.execute(wf)
        assert run.is_success
        # LLM should have been called with resolved prompt
        mock_llm.assert_called_once_with("Summarize: raw data")

    async def test_llm_exception_becomes_failure(self, mock_mcp: AsyncMock) -> None:
        llm = AsyncMock(side_effect=RuntimeError("LLM crashed"))
        engine = WorkflowEngine(mcp_client=mock_mcp, llm_func=llm)
        wf = _wf("l", [_llm("a")])
        run = await engine.execute(wf)
        assert run.node_results["a"].status == NodeStatus.FAILURE


# ===========================================================================
# 9. Condition Node Execution
# ===========================================================================

class TestConditionNodeExecution:
    """Test conditional branching."""

    async def test_condition_true_branch(self, engine: WorkflowEngine, mock_mcp: AsyncMock) -> None:
        wf = _wf("cond", [
            _tool("a"),
            _condition("check", condition='${a.status} == "success"', on_true="good", on_false="bad", depends_on=["a"]),
            _tool("good", depends_on=["check"]),
            _tool("bad", depends_on=["check"]),
        ])
        run = await engine.execute(wf)
        assert run.node_results["check"].output == "true"
        assert run.node_results["good"].status == NodeStatus.SUCCESS
        assert run.node_results["bad"].status == NodeStatus.SKIPPED

    async def test_condition_false_branch(self, engine: WorkflowEngine, mock_mcp: AsyncMock) -> None:
        mock_mcp.call_tool.return_value = MockToolOutput(content="fail", is_error=True)
        wf = _wf("cond", [
            _tool("a"),
            _condition("check", condition='${a.status} == "success"', on_true="good", on_false="bad", depends_on=["a"]),
            _tool("good", depends_on=["check"]),
            _tool("bad", depends_on=["check"]),
        ])
        # Node "a" fails → condition evaluates to false because a.status != "success"
        # But "a" failing means "check" depends on "a" which failed → "check" gets SKIPPED
        run = await engine.execute(wf)
        # When a fails, check is skipped (dependency failed), and both good/bad are skipped too
        assert run.node_results["a"].status == NodeStatus.FAILURE
        assert run.node_results["check"].status == NodeStatus.SKIPPED

    async def test_condition_skips_descendants(self, engine: WorkflowEngine, mock_mcp: AsyncMock) -> None:
        wf = _wf("cond", [
            _tool("a"),
            _condition("check", condition="true", on_true="path_a", on_false="path_b", depends_on=["a"]),
            _tool("path_a", depends_on=["check"]),
            _tool("path_b", depends_on=["check"]),
            _tool("after_b", depends_on=["path_b"]),  # descendant of skipped path
        ])
        run = await engine.execute(wf)
        assert run.node_results["path_a"].status == NodeStatus.SUCCESS
        assert run.node_results["path_b"].status == NodeStatus.SKIPPED
        assert run.node_results["after_b"].status == NodeStatus.SKIPPED


# ===========================================================================
# 10. Human Approval Node
# ===========================================================================

class TestHumanApprovalNode:
    """Test human approval gates."""

    async def test_approval_granted(self, engine: WorkflowEngine) -> None:
        wf = _wf("appr", [_approval("ask")])
        run = await engine.execute(wf)
        assert run.node_results["ask"].status == NodeStatus.SUCCESS
        assert run.node_results["ask"].output == "approved"

    async def test_approval_denied(self, mock_mcp: AsyncMock, mock_llm: AsyncMock, mock_approval_no: AsyncMock) -> None:
        engine = WorkflowEngine(
            mcp_client=mock_mcp, llm_func=mock_llm, approval_func=mock_approval_no
        )
        wf = _wf("appr", [_approval("ask")])
        run = await engine.execute(wf)
        assert run.node_results["ask"].status == NodeStatus.FAILURE
        assert run.node_results["ask"].output == "denied"

    async def test_approval_without_callback(self, bare_engine: WorkflowEngine) -> None:
        wf = _wf("appr", [_approval("ask")])
        run = await bare_engine.execute(wf)
        assert run.node_results["ask"].status == NodeStatus.FAILURE
        assert "No approval callback" in (run.node_results["ask"].error or "")


# ===========================================================================
# 11. Retry Strategies
# ===========================================================================

class TestRetryStrategies:
    """Test retry delay calculation and retry execution."""

    def test_exponential_delays(self, engine: WorkflowEngine) -> None:
        assert engine._calculate_retry_delay(RetryStrategy.EXPONENTIAL, 0) == 1.0
        assert engine._calculate_retry_delay(RetryStrategy.EXPONENTIAL, 1) == 2.0
        assert engine._calculate_retry_delay(RetryStrategy.EXPONENTIAL, 2) == 4.0
        assert engine._calculate_retry_delay(RetryStrategy.EXPONENTIAL, 3) == 8.0

    def test_exponential_capped_at_30(self, engine: WorkflowEngine) -> None:
        assert engine._calculate_retry_delay(RetryStrategy.EXPONENTIAL, 10) == 30.0

    def test_linear_delays(self, engine: WorkflowEngine) -> None:
        assert engine._calculate_retry_delay(RetryStrategy.LINEAR, 0) == 1.0
        assert engine._calculate_retry_delay(RetryStrategy.LINEAR, 1) == 2.0
        assert engine._calculate_retry_delay(RetryStrategy.LINEAR, 2) == 3.0

    def test_linear_capped_at_15(self, engine: WorkflowEngine) -> None:
        assert engine._calculate_retry_delay(RetryStrategy.LINEAR, 20) == 15.0

    def test_none_strategy_zero_delay(self, engine: WorkflowEngine) -> None:
        assert engine._calculate_retry_delay(RetryStrategy.NONE, 0) == 0.0
        assert engine._calculate_retry_delay(RetryStrategy.NONE, 5) == 0.0

    async def test_retry_succeeds_on_second_attempt(self, mock_llm: AsyncMock) -> None:
        call_count = 0

        async def flaky_mcp_call(tool_name: str, params: dict) -> MockToolOutput:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("first attempt timeout")
            return MockToolOutput(content="success on retry")

        mcp = AsyncMock()
        mcp.call_tool = flaky_mcp_call
        engine = WorkflowEngine(mcp_client=mcp, llm_func=mock_llm)

        node = _tool("a", retry_strategy=RetryStrategy.NONE, max_retries=2, timeout_seconds=5)
        wf = _wf("retry", [node])
        run = await engine.execute(wf)
        assert run.node_results["a"].status == NodeStatus.SUCCESS
        assert run.node_results["a"].retry_count == 1

    async def test_all_retries_exhausted(self, mock_llm: AsyncMock) -> None:
        mcp = AsyncMock()
        mcp.call_tool = AsyncMock(side_effect=TimeoutError("always fails"))
        engine = WorkflowEngine(mcp_client=mcp, llm_func=mock_llm)

        node = _tool("a", retry_strategy=RetryStrategy.NONE, max_retries=1, timeout_seconds=5)
        wf = _wf("retry", [node])
        run = await engine.execute(wf)
        assert run.node_results["a"].status == NodeStatus.FAILURE
        assert run.node_results["a"].retry_count == 1


# ===========================================================================
# 12. Parallel Execution
# ===========================================================================

class TestParallelExecution:
    """Test concurrent node execution."""

    async def test_independent_nodes_run_parallel(self, engine: WorkflowEngine, mock_mcp: AsyncMock) -> None:
        wf = _wf("par", [_tool("a"), _tool("b"), _tool("c")])
        run = await engine.execute(wf)
        assert run.is_success
        assert mock_mcp.call_tool.call_count == 3

    async def test_diamond_pattern(self, engine: WorkflowEngine, mock_mcp: AsyncMock) -> None:
        wf = _wf("diamond", [
            _tool("start"),
            _tool("left", depends_on=["start"]),
            _tool("right", depends_on=["start"]),
            _tool("end", depends_on=["left", "right"]),
        ])
        run = await engine.execute(wf)
        assert run.is_success
        assert mock_mcp.call_tool.call_count == 4
        # "end" must complete after "left" and "right"
        assert run.node_results["end"].status == NodeStatus.SUCCESS


# ===========================================================================
# 13. Dependency Propagation
# ===========================================================================

class TestDependencyPropagation:
    """Test failure and skip propagation through the graph."""

    async def test_failed_dependency_skips_downstream(self, engine: WorkflowEngine, mock_mcp: AsyncMock) -> None:
        mock_mcp.call_tool.return_value = MockToolOutput(content="error", is_error=True)
        wf = _wf("prop", [
            _tool("a"),
            _tool("b", depends_on=["a"]),
            _tool("c", depends_on=["b"]),
        ])
        run = await engine.execute(wf)
        assert run.node_results["a"].status == NodeStatus.FAILURE
        assert run.node_results["b"].status == NodeStatus.SKIPPED
        assert run.node_results["c"].status == NodeStatus.SKIPPED

    async def test_partial_failure_in_diamond(self, mock_llm: AsyncMock) -> None:
        call_count = 0

        async def selective_mcp(tool_name: str, params: dict) -> MockToolOutput:
            nonlocal call_count
            call_count += 1
            if tool_name == "fail_tool":
                return MockToolOutput(content="error", is_error=True)
            return MockToolOutput(content="ok")

        mcp = AsyncMock()
        mcp.call_tool = selective_mcp
        engine = WorkflowEngine(mcp_client=mcp, llm_func=mock_llm)

        wf = _wf("partial", [
            _tool("start", tool_name="ok_tool"),
            _tool("left", tool_name="ok_tool", depends_on=["start"]),
            _tool("right", tool_name="fail_tool", depends_on=["start"]),
            _tool("end", tool_name="ok_tool", depends_on=["left", "right"]),
        ])
        run = await engine.execute(wf)
        assert run.node_results["start"].status == NodeStatus.SUCCESS
        assert run.node_results["left"].status == NodeStatus.SUCCESS
        assert run.node_results["right"].status == NodeStatus.FAILURE
        # "end" depends on "right" which failed → skipped
        assert run.node_results["end"].status == NodeStatus.SKIPPED


# ===========================================================================
# 14. Checkpoint / Resume
# ===========================================================================

class TestCheckpointResume:
    """Test checkpoint persistence and workflow resumption."""

    async def test_checkpoint_created(self, engine: WorkflowEngine, tmp_path: Path) -> None:
        wf = _wf("cp", [_tool("a")])
        run = await engine.execute(wf)
        cp_files = list((tmp_path / "checkpoints").glob("*.json"))
        assert len(cp_files) >= 1

    async def test_load_checkpoint(self, engine: WorkflowEngine, tmp_path: Path) -> None:
        # Create a checkpoint manually
        run = WorkflowRun(
            id="test_run",
            workflow_id="wf1",
            workflow_name="Test",
            status=NodeStatus.RUNNING,
            node_results={
                "a": NodeResult(node_id="a", status=NodeStatus.SUCCESS, output="done"),
                "b": NodeResult(node_id="b", status=NodeStatus.RUNNING),
            },
        )
        cp_dir = tmp_path / "cp"
        cp_dir.mkdir()
        cp_path = cp_dir / "test_run.json"
        cp_path.write_text(run.model_dump_json(indent=2), "utf-8")

        loaded = engine._load_checkpoint(cp_path)
        assert loaded.id == "test_run"
        assert loaded.node_results["a"].status == NodeStatus.SUCCESS

    async def test_resume_resets_running_to_pending(
        self, mock_mcp: AsyncMock, mock_llm: AsyncMock, tmp_path: Path
    ) -> None:
        # Simulate an interrupted run where node "b" was RUNNING
        run = WorkflowRun(
            id="resume_test",
            workflow_id="wf1",
            workflow_name="Test",
            node_results={
                "a": NodeResult(node_id="a", status=NodeStatus.SUCCESS, output="done"),
                "b": NodeResult(node_id="b", status=NodeStatus.RUNNING),
            },
        )
        cp_dir = tmp_path / "resume_cp"
        cp_dir.mkdir()
        cp_path = cp_dir / "resume_test.json"
        cp_path.write_text(run.model_dump_json(indent=2), "utf-8")

        wf = _wf("test", [
            _tool("a"),
            _tool("b", depends_on=["a"]),
        ], id="wf1")

        engine = WorkflowEngine(
            mcp_client=mock_mcp,
            llm_func=mock_llm,
            checkpoint_dir=tmp_path / "new_cp",
        )
        resumed = await engine.resume(cp_path, wf)

        assert resumed.node_results["a"].status == NodeStatus.SUCCESS
        assert resumed.node_results["b"].status == NodeStatus.SUCCESS
        # MCP should only be called once (for "b", not "a" which was already done)
        mock_mcp.call_tool.assert_called_once()


# ===========================================================================
# 15. Global Timeout
# ===========================================================================

class TestGlobalTimeout:
    """Test workflow-level timeout."""

    async def test_timeout_marks_pending_as_skipped(self, mock_llm: AsyncMock) -> None:
        async def slow_tool(tool_name: str, params: dict) -> MockToolOutput:
            await asyncio.sleep(10)
            return MockToolOutput()

        mcp = AsyncMock()
        mcp.call_tool = slow_tool
        engine = WorkflowEngine(mcp_client=mcp, llm_func=mock_llm)

        wf = _wf("slow", [
            _tool("a"),
            _tool("b", depends_on=["a"]),
        ], global_timeout_seconds=1)

        run = await engine.execute(wf)
        # At least some nodes should be SKIPPED due to timeout
        statuses = {nr.status for nr in run.node_results.values()}
        assert NodeStatus.SKIPPED in statuses or NodeStatus.FAILURE in statuses


# ===========================================================================
# 16. Status Callback
# ===========================================================================

class TestStatusCallback:
    """Test fire-and-forget status callbacks."""

    async def test_status_callback_called(self, mock_mcp: AsyncMock, mock_llm: AsyncMock) -> None:
        callback = AsyncMock()
        engine = WorkflowEngine(
            mcp_client=mock_mcp, llm_func=mock_llm, status_callback=callback
        )
        wf = _wf("cb", [_tool("a")])
        await engine.execute(wf)
        callback.assert_called()

    async def test_status_callback_failure_ignored(self, mock_mcp: AsyncMock, mock_llm: AsyncMock) -> None:
        callback = AsyncMock(side_effect=RuntimeError("callback crashed"))
        engine = WorkflowEngine(
            mcp_client=mock_mcp, llm_func=mock_llm, status_callback=callback
        )
        wf = _wf("cb", [_tool("a")])
        run = await engine.execute(wf)
        # Workflow should still succeed despite callback failure
        assert run.is_success


# ===========================================================================
# 17. Full Integration Workflows
# ===========================================================================

class TestWorkflowIntegration:
    """End-to-end workflow execution tests."""

    async def test_linear_three_step(self, engine: WorkflowEngine) -> None:
        wf = _wf("linear", [
            _tool("search", tool_name="web_search", params={"query": "test"}),
            _llm("analyze", prompt="Analyze: ${search.output}", depends_on=["search"]),
            _tool("save", tool_name="vault_save", params={"content": "${analyze.output}"}, depends_on=["analyze"]),
        ])
        run = await engine.execute(wf)
        assert run.is_success
        assert run.status == NodeStatus.SUCCESS
        assert all(
            run.node_results[nid].status in (NodeStatus.SUCCESS, NodeStatus.SKIPPED)
            for nid in ["search", "analyze", "save"]
        )

    async def test_ten_node_workflow(self, engine: WorkflowEngine) -> None:
        """Acceptance criterion: workflow with ≥10 nodes runs stably."""
        nodes = [_tool(f"step_{i}") for i in range(5)]
        # Add 5 dependent nodes
        for i in range(5, 10):
            nodes.append(_tool(f"step_{i}", depends_on=[f"step_{i - 5}"]))
        wf = _wf("big", nodes)
        run = await engine.execute(wf)
        assert run.is_success
        assert len(run.node_results) == 10
        assert all(
            nr.status == NodeStatus.SUCCESS for nr in run.node_results.values()
        )

    async def test_mixed_node_types(self, engine: WorkflowEngine) -> None:
        wf = _wf("mixed", [
            _tool("fetch", tool_name="web_fetch"),
            _llm("analyze", prompt="Analyze: ${fetch.output}", depends_on=["fetch"]),
            _condition(
                "check",
                condition='${analyze.status} == "success"',
                on_true="save",
                on_false="report_error",
                depends_on=["analyze"],
            ),
            _tool("save", tool_name="vault_save", depends_on=["check"]),
            _tool("report_error", tool_name="write_file", depends_on=["check"]),
            _approval("review", message="Save ${analyze.output}?", depends_on=["save"]),
        ])
        run = await engine.execute(wf)
        assert run.node_results["fetch"].status == NodeStatus.SUCCESS
        assert run.node_results["analyze"].status == NodeStatus.SUCCESS
        assert run.node_results["check"].status == NodeStatus.SUCCESS
        assert run.node_results["save"].status == NodeStatus.SUCCESS
        assert run.node_results["report_error"].status == NodeStatus.SKIPPED
        assert run.node_results["review"].status == NodeStatus.SUCCESS

    async def test_workflow_run_has_timing(self, engine: WorkflowEngine) -> None:
        wf = _wf("timing", [_tool("a")])
        run = await engine.execute(wf)
        assert run.started_at is not None
        assert run.completed_at is not None
        assert run.completed_at >= run.started_at
        assert run.node_results["a"].duration_ms >= 0

    async def test_context_passed_to_run(self, engine: WorkflowEngine) -> None:
        wf = _wf("ctx", [_tool("a")])
        run = await engine.execute(wf, context={"user": "alex", "priority": "high"})
        assert run.context == {"user": "alex", "priority": "high"}


import asyncio  # noqa: E402 — needed for TestGlobalTimeout
