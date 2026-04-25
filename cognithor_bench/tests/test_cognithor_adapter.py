"""CognithorAdapter — wraps cognithor.crew.Crew for benchmark execution."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognithor_bench.adapters.base import ScenarioInput
from cognithor_bench.adapters.cognithor_adapter import CognithorAdapter


@pytest.mark.asyncio
async def test_cognithor_adapter_name() -> None:
    a = CognithorAdapter(model="ollama/qwen3:8b")
    assert a.name == "cognithor"


@pytest.mark.asyncio
async def test_cognithor_adapter_runs_scenario_success() -> None:
    """A passing scenario produces ScenarioResult(success=True) with output."""
    fake_output = MagicMock()
    fake_output.raw = "4"
    fake_output.tasks_outputs = []

    with patch("cognithor_bench.adapters.cognithor_adapter.Crew") as crew_cls:
        crew = MagicMock()
        crew.kickoff_async = AsyncMock(return_value=fake_output)
        crew_cls.return_value = crew

        a = CognithorAdapter(model="ollama/qwen3:8b")
        scenario = ScenarioInput(id="s1", task="2+2", expected="4", timeout_sec=10, requires=())
        result = await a.run(scenario)

        assert result.id == "s1"
        assert result.output == "4"
        assert result.success is True
        assert result.error is None


@pytest.mark.asyncio
async def test_cognithor_adapter_failure_reports_error() -> None:
    """An exception inside kickoff_async produces ScenarioResult(success=False, error)."""
    with patch("cognithor_bench.adapters.cognithor_adapter.Crew") as crew_cls:
        crew = MagicMock()
        crew.kickoff_async = AsyncMock(side_effect=RuntimeError("boom"))
        crew_cls.return_value = crew

        a = CognithorAdapter(model="ollama/qwen3:8b")
        scenario = ScenarioInput(id="s1", task="x", expected="y", timeout_sec=10, requires=())
        result = await a.run(scenario)

        assert result.success is False
        assert result.error is not None
        assert "boom" in result.error


@pytest.mark.asyncio
async def test_cognithor_adapter_substring_match_is_case_insensitive() -> None:
    """Expected '4' matched against 'The answer is 4.' counts as success."""
    fake_output = MagicMock()
    fake_output.raw = "The answer is 4."
    fake_output.tasks_outputs = []

    with patch("cognithor_bench.adapters.cognithor_adapter.Crew") as crew_cls:
        crew = MagicMock()
        crew.kickoff_async = AsyncMock(return_value=fake_output)
        crew_cls.return_value = crew

        a = CognithorAdapter(model="ollama/qwen3:8b")
        scenario = ScenarioInput(id="s1", task="2+2", expected="4", timeout_sec=10, requires=())
        result = await a.run(scenario)

        assert result.success is True
