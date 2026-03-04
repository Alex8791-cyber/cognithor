"""Agent Delegation Engine — structured task delegation with contracts.

Provides typed task contracts, agent capability discovery, and validated
delegation chains. Integrates with the DAG Workflow Engine as agent-type
nodes and with the Orchestrator for sub-agent execution.

Usage::

    engine = DelegationEngine(agent_router=router, orchestrator=orch)
    result = await engine.delegate(
        task="Recherchiere Python asyncio",
        from_agent="jarvis",
        to_agent="researcher",
        contract=TaskContract(
            input_schema={"query": "str"},
            output_schema={"summary": "str", "sources": "list[str]"},
        ),
    )
    assert result.validated  # Output matches contract

Architecture: §7.3 (Delegation), §9.2 (Multi-Agent-Routing)
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.core.agent_router import AgentProfile, AgentRouter
    from jarvis.core.orchestrator import Orchestrator
    from jarvis.security.audit import AuditTrail

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Task Contract
# ---------------------------------------------------------------------------


class TaskContract(BaseModel, frozen=True):
    """Typed input/output schema for delegation tasks.

    Schemas are dicts mapping field names to type strings.
    Validation checks that returned data contains all required keys.
    """

    input_schema: dict[str, str] = Field(default_factory=dict)
    output_schema: dict[str, str] = Field(default_factory=dict)
    description: str = ""
    timeout_seconds: int = 300
    require_all_output_fields: bool = True


# ---------------------------------------------------------------------------
# Agent Capability
# ---------------------------------------------------------------------------


class AgentCapability(BaseModel, frozen=True):
    """Structured capability description for an agent."""

    name: str
    description: str = ""
    input_types: list[str] = Field(default_factory=list)
    output_types: list[str] = Field(default_factory=list)
    tools_required: list[str] = Field(default_factory=list)
    priority: int = 0


# ---------------------------------------------------------------------------
# Delegation Status
# ---------------------------------------------------------------------------


class DelegationStatus(StrEnum):
    """Status of a delegation request."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    REJECTED = "rejected"
    VALIDATION_FAILED = "validation_failed"


# ---------------------------------------------------------------------------
# Delegation Result
# ---------------------------------------------------------------------------


class DelegationResult(BaseModel):
    """Result of a delegation request with contract validation."""

    delegation_id: str = ""
    from_agent: str = ""
    to_agent: str = ""
    task: str = ""
    status: DelegationStatus = DelegationStatus.PENDING
    output: dict[str, Any] = Field(default_factory=dict)
    raw_response: str = ""
    validated: bool = False
    validation_errors: list[str] = Field(default_factory=list)
    duration_ms: int = 0
    depth: int = 0
    sub_delegations: list[DelegationResult] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent Registry (Capabilities Discovery)
# ---------------------------------------------------------------------------


class AgentRegistry:
    """Registry for agent capabilities discovery.

    Wraps the AgentRouter and enriches it with structured capabilities
    that go beyond simple keyword matching.
    """

    def __init__(self) -> None:
        self._capabilities: dict[str, list[AgentCapability]] = {}

    def register_capabilities(
        self, agent_name: str, capabilities: list[AgentCapability],
    ) -> None:
        """Register capabilities for an agent."""
        self._capabilities[agent_name] = list(capabilities)

    def get_capabilities(self, agent_name: str) -> list[AgentCapability]:
        """Get capabilities for an agent."""
        return list(self._capabilities.get(agent_name, []))

    def find_agents_for_capability(self, capability_name: str) -> list[str]:
        """Find agents that have a specific capability."""
        result = []
        for agent_name, caps in self._capabilities.items():
            if any(c.name == capability_name for c in caps):
                result.append(agent_name)
        return result

    def find_best_agent(
        self,
        *,
        capability: str = "",
        required_tools: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> str | None:
        """Find the best agent for a given capability or tool requirements.

        Returns agent name or None if no match.
        """
        exclude_set = set(exclude or [])
        best_name: str | None = None
        best_priority = -1

        for agent_name, caps in self._capabilities.items():
            if agent_name in exclude_set:
                continue

            for cap in caps:
                match = False
                if capability and cap.name == capability:
                    match = True
                if required_tools:
                    if all(t in cap.tools_required for t in required_tools):
                        match = True

                if match and cap.priority > best_priority:
                    best_priority = cap.priority
                    best_name = agent_name

        return best_name

    @property
    def registered_agents(self) -> list[str]:
        """List of agent names with registered capabilities."""
        return list(self._capabilities.keys())

    def stats(self) -> dict[str, Any]:
        """Registry statistics."""
        total_caps = sum(len(caps) for caps in self._capabilities.values())
        return {
            "agents": len(self._capabilities),
            "total_capabilities": total_caps,
        }


# ---------------------------------------------------------------------------
# Contract Validator
# ---------------------------------------------------------------------------


def validate_output(
    contract: TaskContract,
    output: dict[str, Any],
) -> list[str]:
    """Validate delegation output against a contract's output_schema.

    Returns list of errors (empty = valid).
    """
    errors: list[str] = []
    if not contract.output_schema:
        return errors

    for field_name, field_type in contract.output_schema.items():
        if field_name not in output:
            if contract.require_all_output_fields:
                errors.append(f"Missing required field: '{field_name}'")
            continue

        value = output[field_name]
        # Basic type checking
        expected = field_type.lower().strip()
        if expected == "str" and not isinstance(value, str):
            errors.append(f"Field '{field_name}': expected str, got {type(value).__name__}")
        elif expected == "int" and not isinstance(value, int):
            errors.append(f"Field '{field_name}': expected int, got {type(value).__name__}")
        elif expected == "float" and not isinstance(value, (int, float)):
            errors.append(f"Field '{field_name}': expected float, got {type(value).__name__}")
        elif expected == "bool" and not isinstance(value, bool):
            errors.append(f"Field '{field_name}': expected bool, got {type(value).__name__}")
        elif expected.startswith("list") and not isinstance(value, list):
            errors.append(f"Field '{field_name}': expected list, got {type(value).__name__}")
        elif expected.startswith("dict") and not isinstance(value, dict):
            errors.append(f"Field '{field_name}': expected dict, got {type(value).__name__}")

    return errors


# ---------------------------------------------------------------------------
# Delegation Engine
# ---------------------------------------------------------------------------


class DelegationEngine:
    """Orchestrates agent-to-agent delegation with contract validation.

    Ties together AgentRouter (permissions), Orchestrator (execution),
    AgentRegistry (discovery), and contract validation.
    """

    def __init__(
        self,
        *,
        agent_router: AgentRouter | None = None,
        orchestrator: Orchestrator | None = None,
        registry: AgentRegistry | None = None,
        audit_trail: AuditTrail | None = None,
    ) -> None:
        self._router = agent_router
        self._orchestrator = orchestrator
        self._registry = registry or AgentRegistry()
        self._audit = audit_trail
        self._history: list[DelegationResult] = []

    @property
    def registry(self) -> AgentRegistry:
        """Access the capabilities registry."""
        return self._registry

    async def delegate(
        self,
        task: str,
        *,
        from_agent: str = "jarvis",
        to_agent: str = "",
        contract: TaskContract | None = None,
        context: dict[str, Any] | None = None,
        depth: int = 0,
    ) -> DelegationResult:
        """Delegate a task to another agent with optional contract validation.

        Args:
            task: The task description.
            from_agent: Name of the delegating agent.
            to_agent: Target agent name. If empty, auto-discovered.
            contract: Optional typed input/output contract.
            context: Additional context for the task.
            depth: Current delegation depth (for recursion protection).

        Returns:
            DelegationResult with validated output.
        """
        delegation_id = uuid.uuid4().hex[:12]
        start_time = time.monotonic()
        contract = contract or TaskContract()

        result = DelegationResult(
            delegation_id=delegation_id,
            from_agent=from_agent,
            to_agent=to_agent,
            task=task,
            depth=depth,
        )

        # Auto-discover target agent if not specified
        if not to_agent:
            to_agent = self._auto_discover_agent(task, from_agent)
            if not to_agent:
                result.status = DelegationStatus.REJECTED
                result.validation_errors = ["No suitable agent found for task"]
                self._record(result, start_time)
                return result
            result.to_agent = to_agent

        # Check delegation permission
        if not self._can_delegate(from_agent, to_agent, depth):
            result.status = DelegationStatus.REJECTED
            result.validation_errors = [
                f"Agent '{from_agent}' cannot delegate to '{to_agent}' "
                f"(depth={depth})"
            ]
            self._record(result, start_time)
            return result

        # Execute delegation
        result.status = DelegationStatus.RUNNING
        try:
            timeout = contract.timeout_seconds
            raw_response = await asyncio.wait_for(
                self._execute_delegation(to_agent, task, context or {}),
                timeout=timeout,
            )
            result.raw_response = raw_response
            result.output = self._parse_output(raw_response)
            result.status = DelegationStatus.SUCCESS
        except asyncio.TimeoutError:
            result.status = DelegationStatus.TIMEOUT
            result.validation_errors = [
                f"Delegation timed out after {contract.timeout_seconds}s"
            ]
            self._record(result, start_time)
            return result
        except Exception as exc:
            result.status = DelegationStatus.FAILURE
            result.validation_errors = [f"Execution error: {exc}"]
            self._record(result, start_time)
            return result

        # Validate output against contract
        if contract.output_schema:
            errors = validate_output(contract, result.output)
            if errors:
                result.status = DelegationStatus.VALIDATION_FAILED
                result.validation_errors = errors
                result.validated = False
            else:
                result.validated = True
        else:
            result.validated = True

        self._record(result, start_time)
        return result

    async def delegate_chain(
        self,
        steps: list[dict[str, Any]],
        *,
        from_agent: str = "jarvis",
        initial_context: dict[str, Any] | None = None,
    ) -> list[DelegationResult]:
        """Execute a chain of delegations, passing output forward.

        Each step is a dict: {"to_agent": str, "task": str, "contract": TaskContract}
        The output of each step is merged into context for the next.

        Returns list of DelegationResult for each step.
        """
        results: list[DelegationResult] = []
        context = dict(initial_context or {})

        for i, step in enumerate(steps):
            to_agent = step.get("to_agent", "")
            task = step.get("task", "")
            contract = step.get("contract")

            # Inject previous output into task via template
            for key, val in context.items():
                task = task.replace(f"${{{key}}}", str(val))

            result = await self.delegate(
                task,
                from_agent=from_agent,
                to_agent=to_agent,
                contract=contract,
                context=context,
                depth=i,
            )
            results.append(result)

            if result.status not in (
                DelegationStatus.SUCCESS,
                DelegationStatus.VALIDATION_FAILED,
            ):
                break  # Stop chain on hard failure

            # Merge output into context for next step
            context.update(result.output)
            # Also pass raw response
            context[f"step_{i}_response"] = result.raw_response

        return results

    @property
    def history(self) -> list[DelegationResult]:
        """All delegation results in this session."""
        return list(self._history)

    def stats(self) -> dict[str, Any]:
        """Delegation statistics."""
        total = len(self._history)
        success = sum(1 for r in self._history if r.status == DelegationStatus.SUCCESS)
        failed = sum(1 for r in self._history if r.status == DelegationStatus.FAILURE)
        return {
            "total": total,
            "success": success,
            "failed": failed,
            "timeout": sum(1 for r in self._history if r.status == DelegationStatus.TIMEOUT),
            "rejected": sum(1 for r in self._history if r.status == DelegationStatus.REJECTED),
            "validation_failed": sum(
                1 for r in self._history
                if r.status == DelegationStatus.VALIDATION_FAILED
            ),
            "avg_duration_ms": (
                sum(r.duration_ms for r in self._history) // total if total else 0
            ),
        }

    # -- Internal helpers --------------------------------------------------

    def _can_delegate(
        self, from_agent: str, to_agent: str, depth: int,
    ) -> bool:
        """Check if delegation is permitted."""
        if self._router:
            return self._router.can_delegate(from_agent, to_agent)
        # Without router, only check depth
        return depth < 10

    def _auto_discover_agent(self, task: str, exclude_agent: str) -> str | None:
        """Find best agent for a task via registry or router."""
        # Try registry first (capability-based)
        for agent_name, caps in self._registry._capabilities.items():
            if agent_name == exclude_agent:
                continue
            for cap in caps:
                # Simple keyword matching against capability description
                if any(kw in task.lower() for kw in cap.name.lower().split()):
                    return agent_name

        # Fall back to router keyword matching
        if self._router:
            route = self._router.route(task)
            if route and route.agent_name != exclude_agent:
                return route.agent_name

        return None

    async def _execute_delegation(
        self, to_agent: str, task: str, context: dict[str, Any],
    ) -> str:
        """Execute the actual delegation via orchestrator or router."""
        if self._orchestrator and self._orchestrator._runner:
            from jarvis.models import AgentType, SubAgentConfig

            config = SubAgentConfig(
                task=task,
                agent_type=AgentType.WORKER,
            )
            result = await self._orchestrator._runner(config, to_agent)
            return result.response if result else ""

        # Fallback: return empty (no execution backend available)
        log.warning(
            "delegation_no_backend",
            to_agent=to_agent,
            task=task[:100],
        )
        return ""

    def _parse_output(self, raw_response: str) -> dict[str, Any]:
        """Parse structured output from raw response.

        Tries JSON first, then wraps as {"response": raw}.
        """
        import json

        raw = raw_response.strip()
        if raw.startswith("{"):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass
        return {"response": raw}

    def _record(self, result: DelegationResult, start_time: float) -> None:
        """Record delegation result and audit."""
        result.duration_ms = int((time.monotonic() - start_time) * 1000)
        self._history.append(result)

        if self._audit:
            try:
                self._audit.log(
                    event="agent_delegation",
                    from_agent=result.from_agent,
                    to_agent=result.to_agent,
                    status=result.status,
                    duration_ms=result.duration_ms,
                )
            except Exception:
                pass

        log.info(
            "delegation_completed",
            delegation_id=result.delegation_id,
            from_agent=result.from_agent,
            to_agent=result.to_agent,
            status=result.status,
            duration_ms=result.duration_ms,
            validated=result.validated,
        )
