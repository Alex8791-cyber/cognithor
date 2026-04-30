# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""``cognithor pse`` CLI tests (spec §19.2)."""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING

import pytest

from cognithor.channels.program_synthesis.cli import main

if TYPE_CHECKING:
    from pathlib import Path


def _capture(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    rc = main(argv, stream=buf)
    return rc, buf.getvalue()


# ---------------------------------------------------------------------------
# pse dsl list / describe
# ---------------------------------------------------------------------------


class TestDslList:
    def test_lists_all_primitives(self) -> None:
        rc, out = _capture(["dsl", "list"])
        assert rc == 0
        # Spot-check a few canonical primitives.
        assert "rotate90" in out
        assert "recolor" in out
        assert "primitives registered" in out

    def test_columns_present(self) -> None:
        rc, out = _capture(["dsl", "list"])
        assert rc == 0
        assert "name" in out
        assert "arity" in out
        assert "cost" in out
        assert "output" in out


class TestDslDescribe:
    def test_known_primitive_describe(self) -> None:
        rc, out = _capture(["dsl", "describe", "rotate90"])
        assert rc == 0
        assert "rotate90" in out
        assert "Grid" in out

    def test_unknown_primitive_returns_2(self) -> None:
        rc, out = _capture(["dsl", "describe", "definitely_not_a_primitive"])
        assert rc == 2
        assert "unknown" in out

    def test_recolor_shows_ternary_signature(self) -> None:
        rc, out = _capture(["dsl", "describe", "recolor"])
        assert rc == 0
        # recolor: (Grid, Color, Color) -> Grid
        assert "Color" in out
        assert "Grid" in out


# ---------------------------------------------------------------------------
# pse sandbox doctor
# ---------------------------------------------------------------------------


class TestSandboxDoctor:
    def test_prints_strategy_info(self) -> None:
        rc, out = _capture(["sandbox", "doctor"])
        assert rc == 0
        assert "strategy" in out
        assert "capabilities" in out
        # SYNTHESIZE is always granted regardless of platform.
        assert "pse:synthesize" in out


# ---------------------------------------------------------------------------
# pse run
# ---------------------------------------------------------------------------


class TestRun:
    def test_solves_rotate90(self, tmp_path: Path) -> None:
        task = {
            "examples": [
                {"input": [[1, 2], [3, 4]], "output": [[3, 1], [4, 2]]},
                {"input": [[5, 6, 7]], "output": [[5], [6], [7]]},
            ],
            "budget": {
                "max_depth": 2,
                "wall_clock_seconds": 10.0,
            },
        }
        path = tmp_path / "task.json"
        path.write_text(json.dumps(task), encoding="utf-8")
        rc, out = _capture(["run", str(path)])
        assert rc == 0
        assert "status       : success" in out
        # The solution should be rotate90.
        assert "rotate90(input)" in out
        # Trace block must include the K9 header.
        assert "PSE Solution Trace" in out

    def test_no_solution_returns_1(self, tmp_path: Path) -> None:
        # Two contradictory demos → no single program solves both.
        task = {
            "examples": [
                {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]},
                {"input": [[5, 6], [7, 8]], "output": [[7, 5], [8, 6]]},
            ],
            "budget": {"max_depth": 2, "wall_clock_seconds": 5.0},
        }
        path = tmp_path / "task.json"
        path.write_text(json.dumps(task), encoding="utf-8")
        rc, out = _capture(["run", str(path)])
        # Run completes (process exit ≠ 0 because not SUCCESS) without crashing.
        assert rc == 1
        # status line is present even on partial / no-solution.
        assert "status       :" in out

    def test_missing_file_returns_2(self) -> None:
        rc, out = _capture(["run", "/nonexistent/task.json"])
        assert rc == 2
        assert "error" in out

    def test_invalid_json_returns_2(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("not valid json {[", encoding="utf-8")
        rc, out = _capture(["run", str(path)])
        assert rc == 2
        assert "error" in out

    def test_payload_without_examples_returns_2(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"budget": {"max_depth": 4}}), encoding="utf-8")
        rc, out = _capture(["run", str(path)])
        assert rc == 2
        assert "synthesis task" in out


# ---------------------------------------------------------------------------
# Argparse front-end
# ---------------------------------------------------------------------------


class TestArgparse:
    def test_no_args_exits_with_error(self) -> None:
        # argparse exits with rc=2 when required subcommand is missing.
        with pytest.raises(SystemExit) as exc:
            _capture([])
        assert exc.value.code == 2

    def test_unknown_subcommand_exits_with_error(self) -> None:
        with pytest.raises(SystemExit) as exc:
            _capture(["does_not_exist"])
        assert exc.value.code == 2

    def test_version_flag_works(self) -> None:
        # --version makes argparse exit cleanly with code 0.
        with pytest.raises(SystemExit) as exc:
            _capture(["--version"])
        assert exc.value.code == 0
