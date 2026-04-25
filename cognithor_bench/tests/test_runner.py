"""BenchRunner — core async loop with repetition + sub-sampling."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from cognithor_bench.adapters.base import ScenarioResult
from cognithor_bench.runner import BenchRunner

if TYPE_CHECKING:
    from pathlib import Path


def _scenario_file(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "scen.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


@pytest.mark.asyncio
async def test_runner_executes_each_scenario(tmp_path: Path) -> None:
    rows = [
        {"id": "a", "task": "ta", "expected": "x", "timeout_sec": 10, "requires": []},
        {"id": "b", "task": "tb", "expected": "y", "timeout_sec": 10, "requires": []},
    ]
    path = _scenario_file(tmp_path, rows)

    adapter = MagicMock()
    adapter.name = "test"
    adapter.run = AsyncMock(
        side_effect=[
            ScenarioResult(id="a", output="x", success=True, duration_sec=0.1, error=None),
            ScenarioResult(id="b", output="z", success=False, duration_sec=0.2, error=None),
        ]
    )

    runner = BenchRunner(adapter=adapter)
    results = await runner.run_file(path, repeat=1, subsample=1.0)

    assert len(results) == 2
    assert results[0].id == "a"
    assert results[1].success is False


@pytest.mark.asyncio
async def test_runner_repeats_scenarios(tmp_path: Path) -> None:
    rows = [{"id": "a", "task": "ta", "expected": "x", "timeout_sec": 10, "requires": []}]
    path = _scenario_file(tmp_path, rows)

    adapter = MagicMock()
    adapter.name = "test"
    adapter.run = AsyncMock(
        return_value=ScenarioResult(
            id="a",
            output="x",
            success=True,
            duration_sec=0.05,
            error=None,
        )
    )

    runner = BenchRunner(adapter=adapter)
    results = await runner.run_file(path, repeat=3, subsample=1.0)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_runner_subsample_reduces_count(tmp_path: Path) -> None:
    rows = [
        {"id": str(i), "task": "t", "expected": "x", "timeout_sec": 10, "requires": []}
        for i in range(10)
    ]
    path = _scenario_file(tmp_path, rows)

    adapter = MagicMock()
    adapter.name = "test"
    adapter.run = AsyncMock(
        side_effect=lambda s: ScenarioResult(
            id=s.id,
            output="x",
            success=True,
            duration_sec=0.01,
            error=None,
        )
    )

    runner = BenchRunner(adapter=adapter, seed=42)
    results = await runner.run_file(path, repeat=1, subsample=0.5)
    assert len(results) == 5  # 10 * 0.5


@pytest.mark.asyncio
async def test_runner_writes_results_to_dir(tmp_path: Path) -> None:
    rows = [{"id": "a", "task": "t", "expected": "x", "timeout_sec": 10, "requires": []}]
    path = _scenario_file(tmp_path, rows)

    adapter = MagicMock()
    adapter.name = "test"
    adapter.run = AsyncMock(
        return_value=ScenarioResult(
            id="a",
            output="x",
            success=True,
            duration_sec=0.01,
            error=None,
        )
    )

    out_dir = tmp_path / "results"
    runner = BenchRunner(adapter=adapter)
    await runner.run_file(path, repeat=1, subsample=1.0, output_dir=out_dir)

    files = list(out_dir.glob("*.jsonl"))
    assert len(files) == 1
    body = files[0].read_text(encoding="utf-8")
    assert json.loads(body.splitlines()[0])["id"] == "a"
