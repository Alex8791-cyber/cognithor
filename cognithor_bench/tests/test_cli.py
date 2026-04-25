"""CLI — argparse, run / tabulate subcommands."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognithor_bench.adapters.base import ScenarioResult
from cognithor_bench.cli import main

if TYPE_CHECKING:
    from pathlib import Path


def _scenarios(tmp_path: Path) -> Path:
    p = tmp_path / "scen.jsonl"
    p.write_text(
        json.dumps({"id": "a", "task": "ta", "expected": "x", "timeout_sec": 5, "requires": []})
        + "\n",
        encoding="utf-8",
    )
    return p


def test_cli_help_exits_zero(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "cognithor-bench" in captured.out
    assert "run" in captured.out
    assert "tabulate" in captured.out


def test_cli_run_missing_scenario_exits_nonzero(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["run", str(tmp_path / "doesnotexist.jsonl")])
    assert exc.value.code != 0


def test_cli_run_invokes_adapter(tmp_path: Path) -> None:
    p = _scenarios(tmp_path)
    out = tmp_path / "results"

    with patch("cognithor_bench.cli.CognithorAdapter") as ca:
        instance = MagicMock()
        instance.name = "cognithor"
        instance.run = AsyncMock(
            return_value=ScenarioResult(
                id="a",
                output="x",
                success=True,
                duration_sec=0.01,
                error=None,
            )
        )
        ca.return_value = instance

        rc = main(["run", str(p), "--repeat", "1", "--output-dir", str(out)])
        assert rc == 0
        assert any(out.glob("*.jsonl"))


def test_cli_run_picks_autogen_adapter_when_flag_set(tmp_path: Path) -> None:
    p = _scenarios(tmp_path)
    with (
        patch("cognithor_bench.cli.AutoGenAdapter") as aa,
        patch("cognithor_bench.cli.CognithorAdapter") as ca,
    ):
        instance = MagicMock()
        instance.name = "autogen"
        instance.run = AsyncMock(
            return_value=ScenarioResult(
                id="a",
                output="x",
                success=True,
                duration_sec=0.01,
                error=None,
            )
        )
        aa.return_value = instance

        rc = main(["run", str(p), "--adapter", "autogen"])
        assert rc == 0
        aa.assert_called_once()
        ca.assert_not_called()


def test_cli_tabulate_aggregates_directory(tmp_path: Path, capsys) -> None:
    out = tmp_path / "results"
    out.mkdir()
    (out / "x.jsonl").write_text(
        json.dumps(
            ScenarioResult(
                id="a",
                output="x",
                success=True,
                duration_sec=0.1,
                error=None,
            ).model_dump()
        )
        + "\n",
        encoding="utf-8",
    )
    rc = main(["tabulate", str(out)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "| a |" in captured.out


def test_cli_run_native_is_default_no_docker_invoked(tmp_path: Path) -> None:
    """Spec: --native is default, --docker is opt-in. No docker call without flag."""
    p = _scenarios(tmp_path)
    with (
        patch("cognithor_bench.cli.CognithorAdapter") as ca,
        patch("cognithor_bench.cli._run_under_docker") as docker,
    ):
        instance = MagicMock()
        instance.name = "cognithor"
        instance.run = AsyncMock(
            return_value=ScenarioResult(
                id="a",
                output="x",
                success=True,
                duration_sec=0.01,
                error=None,
            )
        )
        ca.return_value = instance

        rc = main(["run", str(p)])
        assert rc == 0
        docker.assert_not_called()
