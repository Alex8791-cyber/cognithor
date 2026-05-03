"""Workflow definition models for DAG-based multi-step agent pipelines.

Provides typed schemas for defining workflows as directed acyclic graphs (DAGs)
with support for tool execution, LLM calls, conditional branching, and human
approval gates.

Example workflow (YAML)::

    name: "Research Pipeline"
    nodes:
      - id: search
        type: tool
        tool_name: search_and_read
        tool_params:
          query: "Python asyncio best practices"
      - id: analyze
        type: llm
        prompt: "Analysiere: ${search.output}"
        depends_on: [search]
      - id: check
        type: condition
        condition: '${analyze.status} == "success"'
        on_true: save
        on_false: retry_search
        depends_on: [analyze]
      - id: save
        type: tool
        tool_name: vault_save
        tool_params:
          title: "Asyncio Research"
          content: "${analyze.output}"
        depends_on: [check]
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class NodeType(StrEnum):
    """Type of workflow node."""

    LLM = "llm"
    TOOL = "tool"
    CONDITION = "condition"
    HUMAN_APPROVAL = "human_approval"


class NodeStatus(StrEnum):
    """Execution status of a workflow node or run."""

    PENDING = "pending"
    WAITING = "waiting"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"


class RetryStrategy(StrEnum):
    """Retry behavior for failed nodes."""

    NONE = "none"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"


# ---------------------------------------------------------------------------
# Workflow Definition (immutable)
# ---------------------------------------------------------------------------


class WorkflowNode(BaseModel, frozen=True):
    """A single node in the workflow DAG."""

    id: str
    type: NodeType
    name: str = ""
    description: str = ""

    # Tool node ---------------------------------------------------------------
    tool_name: str | None = None
    tool_params: dict[str, Any] = Field(default_factory=dict)

    # LLM node ----------------------------------------------------------------
    prompt: str | None = None
    model: str | None = None

    # Condition node ----------------------------------------------------------
    condition: str | None = None
    on_true: str | None = None
    on_false: str | None = None

    # Human approval ----------------------------------------------------------
    approval_message: str | None = None

    # Execution config --------------------------------------------------------
    timeout_seconds: int = 60
    retry_strategy: RetryStrategy = RetryStrategy.NONE
    max_retries: int = 0

    # Dependencies (node IDs that must complete before this node runs) --------
    depends_on: list[str] = Field(default_factory=list)


class WorkflowDefinition(BaseModel, frozen=True):
    """Complete workflow definition as a directed acyclic graph."""

    id: str = Field(default_factory=_new_id)
    name: str
    description: str = ""
    version: str = "1.0"
    nodes: list[WorkflowNode]
    max_parallel: int = 5
    global_timeout_seconds: int = 600

    def get_node(self, node_id: str) -> WorkflowNode | None:
        """Look up a node by ID. Returns ``None`` if not found."""
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    @property
    def node_ids(self) -> set[str]:
        """Set of all node IDs in this workflow."""
        return {n.id for n in self.nodes}

    @classmethod
    def from_yaml(cls, yaml_str: str) -> WorkflowDefinition:
        """Parse a workflow from a YAML string."""
        import yaml

        data = yaml.safe_load(yaml_str)
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Runtime State (mutable)
# ---------------------------------------------------------------------------


class NodeResult(BaseModel):
    """Result of executing a single workflow node."""

    node_id: str
    status: NodeStatus = NodeStatus.PENDING
    output: str = ""
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: int = 0
    retry_count: int = 0


class WorkflowRun(BaseModel):
    """Runtime state of a workflow execution."""

    id: str = Field(default_factory=_new_id)
    workflow_id: str = ""
    workflow_name: str = ""
    status: NodeStatus = NodeStatus.PENDING
    node_results: dict[str, NodeResult] = Field(default_factory=dict)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    context: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        """True when every node has reached a terminal status."""
        if not self.node_results:
            return False
        terminal = {NodeStatus.SUCCESS, NodeStatus.FAILURE, NodeStatus.SKIPPED}
        return all(r.status in terminal for r in self.node_results.values())

    @property
    def is_success(self) -> bool:
        """True when all nodes are either SUCCESS or SKIPPED (no failures)."""
        if not self.is_complete:
            return False
        return all(
            r.status in (NodeStatus.SUCCESS, NodeStatus.SKIPPED) for r in self.node_results.values()
        )

    @property
    def failed_nodes(self) -> list[str]:
        """Node IDs that ended with FAILURE status."""
        return [nid for nid, r in self.node_results.items() if r.status == NodeStatus.FAILURE]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class WorkflowValidationError(Exception):
    """Raised when a workflow definition fails validation."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(f"Workflow validation failed: {'; '.join(errors)}")
