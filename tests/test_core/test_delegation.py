"""Tests for the Agent Delegation Engine.

Covers task contracts, agent registry, contract validation,
delegation execution, delegation chains, and edge cases.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.core.delegation import (
    AgentCapability,
    AgentRegistry,
    DelegationEngine,
    DelegationResult,
    DelegationStatus,
    TaskContract,
    validate_output,
)


# ============================================================================
# Helpers
# ============================================================================


def _cap(name: str, priority: int = 0, tools: list[str] | None = None) -> AgentCapability:
    return AgentCapability(
        name=name,
        description=f"Can {name}",
        priority=priority,
        tools_required=tools or [],
    )


def _make_runner(response: str = "done") -> AsyncMock:
    """Create a mock orchestrator runner that returns a fixed response."""
    result = MagicMock()
    result.response = response
    runner = AsyncMock(return_value=result)
    return runner


def _make_orchestrator(response: str = "done") -> MagicMock:
    orch = MagicMock()
    runner = _make_runner(response)
    orch._runner = runner
    return orch


def _make_router(can_delegate: bool = True) -> MagicMock:
    router = MagicMock()
    router.can_delegate = MagicMock(return_value=can_delegate)
    route_result = MagicMock()
    route_result.agent = MagicMock()
    route_result.agent.name = "researcher"
    router.route = MagicMock(return_value=route_result)
    return router


# ============================================================================
# TaskContract
# ============================================================================


class TestTaskContract:
    def test_default_contract(self) -> None:
        c = TaskContract()
        assert c.input_schema == {}
        assert c.output_schema == {}
        assert c.timeout_seconds == 300

    def test_custom_contract(self) -> None:
        c = TaskContract(
            input_schema={"query": "str"},
            output_schema={"summary": "str", "sources": "list[str]"},
            description="Research contract",
            timeout_seconds=60,
        )
        assert c.input_schema["query"] == "str"
        assert c.output_schema["summary"] == "str"
        assert c.timeout_seconds == 60

    def test_contract_is_frozen(self) -> None:
        c = TaskContract()
        with pytest.raises(Exception):
            c.timeout_seconds = 999  # type: ignore[misc]


# ============================================================================
# AgentCapability
# ============================================================================


class TestAgentCapability:
    def test_create_capability(self) -> None:
        cap = AgentCapability(
            name="web_search",
            description="Search the web",
            tools_required=["web_search", "search_and_read"],
            priority=5,
        )
        assert cap.name == "web_search"
        assert len(cap.tools_required) == 2

    def test_capability_is_frozen(self) -> None:
        cap = _cap("test")
        with pytest.raises(Exception):
            cap.priority = 99  # type: ignore[misc]


# ============================================================================
# AgentRegistry
# ============================================================================


class TestAgentRegistry:
    def test_register_and_get(self) -> None:
        reg = AgentRegistry()
        reg.register_capabilities("researcher", [_cap("search"), _cap("summarize")])
        assert len(reg.get_capabilities("researcher")) == 2

    def test_get_unknown_agent(self) -> None:
        reg = AgentRegistry()
        assert reg.get_capabilities("unknown") == []

    def test_find_agents_for_capability(self) -> None:
        reg = AgentRegistry()
        reg.register_capabilities("a", [_cap("search")])
        reg.register_capabilities("b", [_cap("search"), _cap("code")])
        reg.register_capabilities("c", [_cap("code")])
        assert sorted(reg.find_agents_for_capability("search")) == ["a", "b"]
        assert sorted(reg.find_agents_for_capability("code")) == ["b", "c"]

    def test_find_agents_for_unknown_capability(self) -> None:
        reg = AgentRegistry()
        reg.register_capabilities("a", [_cap("search")])
        assert reg.find_agents_for_capability("unknown") == []

    def test_find_best_agent_by_capability(self) -> None:
        reg = AgentRegistry()
        reg.register_capabilities("a", [_cap("search", priority=3)])
        reg.register_capabilities("b", [_cap("search", priority=7)])
        assert reg.find_best_agent(capability="search") == "b"

    def test_find_best_agent_with_exclude(self) -> None:
        reg = AgentRegistry()
        reg.register_capabilities("a", [_cap("search", priority=3)])
        reg.register_capabilities("b", [_cap("search", priority=7)])
        assert reg.find_best_agent(capability="search", exclude=["b"]) == "a"

    def test_find_best_agent_by_tools(self) -> None:
        reg = AgentRegistry()
        reg.register_capabilities(
            "coder",
            [
                _cap("coding", priority=5, tools=["exec_command", "write_file"]),
            ],
        )
        reg.register_capabilities(
            "researcher",
            [
                _cap("research", priority=5, tools=["web_search"]),
            ],
        )
        assert reg.find_best_agent(required_tools=["exec_command"]) == "coder"

    def test_find_best_agent_no_match(self) -> None:
        reg = AgentRegistry()
        reg.register_capabilities("a", [_cap("search")])
        assert reg.find_best_agent(capability="unknown") is None

    def test_registered_agents(self) -> None:
        reg = AgentRegistry()
        reg.register_capabilities("a", [_cap("x")])
        reg.register_capabilities("b", [_cap("y")])
        assert sorted(reg.registered_agents) == ["a", "b"]

    def test_stats(self) -> None:
        reg = AgentRegistry()
        reg.register_capabilities("a", [_cap("x"), _cap("y")])
        reg.register_capabilities("b", [_cap("z")])
        s = reg.stats()
        assert s["agents"] == 2
        assert s["total_capabilities"] == 3


# ============================================================================
# Contract Validation
# ============================================================================


class TestContractValidation:
    def test_empty_schema_always_valid(self) -> None:
        c = TaskContract()
        assert validate_output(c, {"anything": 42}) == []

    def test_missing_required_field(self) -> None:
        c = TaskContract(output_schema={"name": "str", "age": "int"})
        errors = validate_output(c, {"name": "Alice"})
        assert len(errors) == 1
        assert "age" in errors[0]

    def test_all_fields_present(self) -> None:
        c = TaskContract(output_schema={"name": "str", "count": "int"})
        errors = validate_output(c, {"name": "test", "count": 5})
        assert errors == []

    def test_wrong_type_str(self) -> None:
        c = TaskContract(output_schema={"name": "str"})
        errors = validate_output(c, {"name": 123})
        assert any("expected str" in e for e in errors)

    def test_wrong_type_int(self) -> None:
        c = TaskContract(output_schema={"count": "int"})
        errors = validate_output(c, {"count": "five"})
        assert any("expected int" in e for e in errors)

    def test_wrong_type_list(self) -> None:
        c = TaskContract(output_schema={"items": "list[str]"})
        errors = validate_output(c, {"items": "not a list"})
        assert any("expected list" in e for e in errors)

    def test_wrong_type_dict(self) -> None:
        c = TaskContract(output_schema={"data": "dict"})
        errors = validate_output(c, {"data": "not a dict"})
        assert any("expected dict" in e for e in errors)

    def test_wrong_type_float(self) -> None:
        c = TaskContract(output_schema={"score": "float"})
        errors = validate_output(c, {"score": "high"})
        assert any("expected float" in e for e in errors)

    def test_int_is_valid_float(self) -> None:
        c = TaskContract(output_schema={"score": "float"})
        errors = validate_output(c, {"score": 5})
        assert errors == []

    def test_wrong_type_bool(self) -> None:
        c = TaskContract(output_schema={"flag": "bool"})
        errors = validate_output(c, {"flag": 1})
        assert any("expected bool" in e for e in errors)

    def test_optional_fields(self) -> None:
        c = TaskContract(
            output_schema={"name": "str", "age": "int"},
            require_all_output_fields=False,
        )
        errors = validate_output(c, {"name": "Alice"})
        assert errors == []

    def test_extra_fields_ignored(self) -> None:
        c = TaskContract(output_schema={"name": "str"})
        errors = validate_output(c, {"name": "Alice", "extra": True})
        assert errors == []


# ============================================================================
# DelegationEngine — Basic Delegation
# ============================================================================


class TestDelegation:
    async def test_successful_delegation(self) -> None:
        orch = _make_orchestrator('{"summary": "Found results"}')
        engine = DelegationEngine(orchestrator=orch)
        result = await engine.delegate(
            "search for python",
            from_agent="jarvis",
            to_agent="researcher",
        )
        assert result.status == DelegationStatus.SUCCESS
        assert result.validated is True
        assert result.to_agent == "researcher"

    async def test_delegation_with_contract_validation(self) -> None:
        orch = _make_orchestrator('{"summary": "Results", "count": 5}')
        contract = TaskContract(
            output_schema={"summary": "str", "count": "int"},
        )
        engine = DelegationEngine(orchestrator=orch)
        result = await engine.delegate(
            "search",
            to_agent="researcher",
            contract=contract,
        )
        assert result.status == DelegationStatus.SUCCESS
        assert result.validated is True
        assert result.output["summary"] == "Results"

    async def test_delegation_validation_fails(self) -> None:
        orch = _make_orchestrator('{"summary": "Results"}')
        contract = TaskContract(
            output_schema={"summary": "str", "missing_field": "int"},
        )
        engine = DelegationEngine(orchestrator=orch)
        result = await engine.delegate(
            "search",
            to_agent="researcher",
            contract=contract,
        )
        assert result.status == DelegationStatus.VALIDATION_FAILED
        assert not result.validated
        assert any("missing_field" in e for e in result.validation_errors)

    async def test_delegation_no_backend(self) -> None:
        engine = DelegationEngine()
        result = await engine.delegate(
            "task",
            to_agent="researcher",
        )
        assert result.status == DelegationStatus.SUCCESS
        assert result.raw_response == ""

    async def test_delegation_permission_denied(self) -> None:
        router = _make_router(can_delegate=False)
        engine = DelegationEngine(agent_router=router)
        result = await engine.delegate(
            "task",
            from_agent="a",
            to_agent="b",
        )
        assert result.status == DelegationStatus.REJECTED
        assert "cannot delegate" in result.validation_errors[0]

    async def test_delegation_timeout(self) -> None:
        async def slow_runner(config, agent):
            await asyncio.sleep(5)
            return MagicMock(response="late")

        orch = MagicMock()
        orch._runner = slow_runner
        contract = TaskContract(timeout_seconds=0)  # Instant timeout
        engine = DelegationEngine(orchestrator=orch)
        result = await engine.delegate(
            "task",
            to_agent="slow_agent",
            contract=contract,
        )
        assert result.status == DelegationStatus.TIMEOUT

    async def test_delegation_execution_error(self) -> None:
        async def failing_runner(config, agent):
            raise RuntimeError("Agent crashed")

        orch = MagicMock()
        orch._runner = failing_runner
        engine = DelegationEngine(orchestrator=orch)
        result = await engine.delegate("task", to_agent="broken")
        assert result.status == DelegationStatus.FAILURE
        assert "crashed" in result.validation_errors[0]

    async def test_delegation_records_history(self) -> None:
        orch = _make_orchestrator("ok")
        engine = DelegationEngine(orchestrator=orch)
        await engine.delegate("a", to_agent="x")
        await engine.delegate("b", to_agent="y")
        assert len(engine.history) == 2

    async def test_delegation_has_duration(self) -> None:
        orch = _make_orchestrator("ok")
        engine = DelegationEngine(orchestrator=orch)
        result = await engine.delegate("task", to_agent="x")
        assert result.duration_ms >= 0


# ============================================================================
# Auto-Discovery
# ============================================================================


class TestAutoDiscovery:
    async def test_auto_discover_via_registry(self) -> None:
        reg = AgentRegistry()
        reg.register_capabilities("researcher", [_cap("search")])
        engine = DelegationEngine(registry=reg, orchestrator=_make_orchestrator("ok"))
        result = await engine.delegate("search for something", from_agent="jarvis")
        assert result.to_agent == "researcher"
        assert result.status == DelegationStatus.SUCCESS

    async def test_auto_discover_via_router(self) -> None:
        router = _make_router()
        engine = DelegationEngine(agent_router=router, orchestrator=_make_orchestrator("ok"))
        result = await engine.delegate("something", from_agent="jarvis")
        assert result.to_agent == "researcher"

    async def test_auto_discover_no_match(self) -> None:
        engine = DelegationEngine()
        result = await engine.delegate("xyzzy", from_agent="jarvis")
        assert result.status == DelegationStatus.REJECTED
        assert "No suitable agent" in result.validation_errors[0]


# ============================================================================
# Delegation Chain
# ============================================================================


class TestDelegationChain:
    async def test_simple_chain(self) -> None:
        orch = _make_orchestrator('{"data": "gathered"}')
        engine = DelegationEngine(orchestrator=orch)
        results = await engine.delegate_chain(
            [
                {"to_agent": "researcher", "task": "gather data"},
                {"to_agent": "analyst", "task": "analyze ${data}"},
            ]
        )
        assert len(results) == 2
        assert all(r.status == DelegationStatus.SUCCESS for r in results)

    async def test_chain_stops_on_failure(self) -> None:
        async def fail_runner(config, agent):
            raise RuntimeError("fail")

        orch = MagicMock()
        orch._runner = fail_runner
        engine = DelegationEngine(orchestrator=orch)
        results = await engine.delegate_chain(
            [
                {"to_agent": "a", "task": "step 1"},
                {"to_agent": "b", "task": "step 2"},
            ]
        )
        assert len(results) == 1
        assert results[0].status == DelegationStatus.FAILURE

    async def test_chain_passes_context(self) -> None:
        orch = _make_orchestrator('{"info": "enriched"}')
        engine = DelegationEngine(orchestrator=orch)
        results = await engine.delegate_chain(
            [
                {"to_agent": "a", "task": "get info"},
                {"to_agent": "b", "task": "use ${info}"},
            ],
            initial_context={"seed": "value"},
        )
        assert len(results) == 2

    async def test_chain_with_contracts(self) -> None:
        orch = _make_orchestrator('{"name": "Alice"}')
        contract = TaskContract(output_schema={"name": "str"})
        engine = DelegationEngine(orchestrator=orch)
        results = await engine.delegate_chain(
            [
                {"to_agent": "a", "task": "find name", "contract": contract},
            ]
        )
        assert len(results) == 1
        assert results[0].validated is True


# ============================================================================
# Stats
# ============================================================================


class TestDelegationStats:
    async def test_stats_empty(self) -> None:
        engine = DelegationEngine()
        s = engine.stats()
        assert s["total"] == 0

    async def test_stats_after_delegations(self) -> None:
        orch = _make_orchestrator("ok")
        engine = DelegationEngine(orchestrator=orch)
        await engine.delegate("a", to_agent="x")
        await engine.delegate("b", to_agent="y")
        s = engine.stats()
        assert s["total"] == 2
        assert s["success"] == 2

    async def test_stats_mixed(self) -> None:
        orch = _make_orchestrator("ok")
        router = _make_router(can_delegate=False)
        engine = DelegationEngine(orchestrator=orch, agent_router=router)
        await engine.delegate("a", from_agent="x", to_agent="y")
        s = engine.stats()
        assert s["rejected"] == 1


# ============================================================================
# Parse Output
# ============================================================================


class TestParseOutput:
    def test_json_response(self) -> None:
        engine = DelegationEngine()
        result = engine._parse_output('{"key": "value"}')
        assert result == {"key": "value"}

    def test_plain_text_response(self) -> None:
        engine = DelegationEngine()
        result = engine._parse_output("Hello world")
        assert result == {"response": "Hello world"}

    def test_invalid_json(self) -> None:
        engine = DelegationEngine()
        result = engine._parse_output("{broken json")
        assert result == {"response": "{broken json"}

    def test_empty_response(self) -> None:
        engine = DelegationEngine()
        result = engine._parse_output("")
        assert result == {"response": ""}
