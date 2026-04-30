# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Drift-gate for the Hello-World tutorial (spec D12).

The tutorial in ``docs/channels/program_synthesis/tutorial.md`` quotes
the verbatim CLI output of ``cognithor pse run`` against a fixed task
JSON. If the DSL, the trace formatter, or the search engine change in
a way that perturbs that output, this test fails so the tutorial gets
updated alongside the code.

The test asserts on the *shape* of the output (status, program source,
trace markers, step line, replayable round-trip) rather than every
byte — a bytewise comparison would be too brittle (timestamps, sandbox
strategy, etc. vary by host) but the documented invariants must hold.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from cognithor.channels.program_synthesis.cli import main
from cognithor.channels.program_synthesis.core.types import (
    Budget,
    SynthesisStatus,
    TaskSpec,
)
from cognithor.channels.program_synthesis.integration.pge_adapter import (
    ProgramSynthesisChannel,
    SynthesisRequest,
)
from cognithor.channels.program_synthesis.trace import replay_program

TUTORIAL_PATH = (
    Path(__file__).resolve().parents[4] / "docs" / "channels" / "program_synthesis" / "tutorial.md"
)

HELLO_WORLD_TASK = {
    "examples": [
        {"input": [[1, 2], [3, 4]], "output": [[3, 1], [4, 2]]},
        {"input": [[5, 6], [7, 8]], "output": [[7, 5], [8, 6]]},
        {
            "input": [[1, 1, 2], [3, 4, 5], [6, 7, 8]],
            "output": [[6, 3, 1], [7, 4, 1], [8, 5, 2]],
        },
    ],
    "budget": {"max_depth": 2, "wall_clock_seconds": 30.0},
}


@pytest.fixture()
def task_json_path(tmp_path: Path) -> Path:
    p = tmp_path / "hello_world.json"
    p.write_text(json.dumps(HELLO_WORLD_TASK), encoding="utf-8")
    return p


class TestTutorialFile:
    def test_tutorial_doc_exists(self) -> None:
        assert TUTORIAL_PATH.is_file(), (
            "docs/channels/program_synthesis/tutorial.md is missing — D12 "
            "requires a Hello-World tutorial that walks through one solved "
            "task with a full trace."
        )

    def test_tutorial_doc_quotes_the_run_command(self) -> None:
        body = TUTORIAL_PATH.read_text(encoding="utf-8")
        assert "cognithor pse run" in body
        assert "Step 1:" in body
        assert "Program hash" in body
        assert "rotate90(input)" in body


class TestTutorialReproduces:
    def test_cli_run_reports_success_and_program(self, task_json_path: Path) -> None:
        buf = io.StringIO()
        rc = main(["run", str(task_json_path)], stream=buf)
        out = buf.getvalue()
        assert rc == 0, out
        assert "status       : success" in out
        assert "program: rotate90(input)" in out
        # Trace block invariants.
        assert "# PSE Solution Trace" in out
        assert "# Program hash: sha256:" in out
        assert "# DSL version:" in out
        assert "# Search time:" in out
        assert "Step 1: step1 = rotate90(input)" in out

    def test_replay_round_trips_under_k10_budget(self) -> None:
        spec = TaskSpec(
            examples=tuple(
                (
                    np.array(ex["input"], dtype=np.int8),
                    np.array(ex["output"], dtype=np.int8),
                )
                for ex in HELLO_WORLD_TASK["examples"]
            )
        )
        result = ProgramSynthesisChannel().synthesize(
            SynthesisRequest(spec=spec, budget=Budget(max_depth=2)),
        )
        assert result.status == SynthesisStatus.SUCCESS
        assert result.program is not None

        replay = replay_program(
            result.program,
            spec.examples[0][0],
            spec.examples[0][1],
        )
        assert replay.identical
        # K10 hard-gate: replay must complete in P95 ≤ 100 ms. A single
        # depth-1 program should clear that by orders of magnitude.
        assert replay.duration_ms < 100
