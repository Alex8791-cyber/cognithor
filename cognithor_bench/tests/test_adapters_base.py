"""Adapter Protocol — runtime-checkable + minimal contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cognithor_bench.adapters.base import Adapter, ScenarioInput, ScenarioResult


def test_adapter_is_runtime_checkable_protocol() -> None:
    class Dummy:
        name = "dummy"

        async def run(self, scenario: ScenarioInput) -> ScenarioResult:
            return ScenarioResult(
                id=scenario.id,
                output="x",
                success=False,
                duration_sec=0.0,
                error=None,
            )

    assert isinstance(Dummy(), Adapter)


def test_scenario_input_required_fields() -> None:
    s = ScenarioInput(id="s1", task="2+2", expected="4", timeout_sec=10, requires=())
    assert s.id == "s1"
    assert s.task == "2+2"
    assert s.expected == "4"


def test_scenario_result_required_fields() -> None:
    r = ScenarioResult(id="s1", output="4", success=True, duration_sec=0.1, error=None)
    assert r.id == "s1"
    assert r.success is True


def test_scenario_input_is_frozen() -> None:
    s = ScenarioInput(id="s1", task="2+2", expected="4", timeout_sec=10, requires=())
    with pytest.raises(ValidationError):
        s.id = "modified"  # type: ignore[misc]
