"""Verify the bundled smoke_test.jsonl is well-formed."""

from __future__ import annotations

import json
from pathlib import Path

from cognithor_bench.adapters.base import ScenarioInput

SMOKE = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "cognithor_bench"
    / "scenarios"
    / "smoke_test.jsonl"
)


def test_smoke_file_exists() -> None:
    assert SMOKE.exists(), f"missing smoke scenarios at {SMOKE}"


def test_smoke_file_has_three_to_five_rows() -> None:
    lines = [
        line
        for line in SMOKE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert 3 <= len(lines) <= 5


def test_smoke_rows_are_valid_scenario_inputs() -> None:
    for line in SMOKE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        row = json.loads(line)
        row.setdefault("requires", [])
        row["requires"] = tuple(row["requires"])
        ScenarioInput(**row)


def test_smoke_ids_are_unique() -> None:
    ids = []
    for line in SMOKE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(json.loads(line)["id"])
    assert len(ids) == len(set(ids))
