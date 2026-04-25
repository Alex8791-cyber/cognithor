"""Knowledge seeds — JSONL well-formedness + minimum-content."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

KNOWLEDGE = Path(__file__).resolve().parent.parent / "src" / "insurance_agent_pack" / "knowledge"

SEEDS = ("pkv_grundlagen", "ggf_versorgung", "bav_basics", "bu_grundlagen")


@pytest.mark.parametrize("name", SEEDS)
def test_seed_file_exists(name: str) -> None:
    p = KNOWLEDGE / f"{name}.jsonl"
    assert p.exists(), f"missing {p}"


@pytest.mark.parametrize("name", SEEDS)
def test_seed_lines_parse_as_json(name: str) -> None:
    p = KNOWLEDGE / f"{name}.jsonl"
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        json.loads(line)


@pytest.mark.parametrize("name", SEEDS)
def test_seed_has_required_fields(name: str) -> None:
    p = KNOWLEDGE / f"{name}.jsonl"
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        row = json.loads(line)
        for field in ("topic", "summary", "tags"):
            assert field in row, f"{name} row missing field {field}: {row}"


@pytest.mark.parametrize("name", SEEDS)
def test_seed_minimum_three_rows(name: str) -> None:
    p = KNOWLEDGE / f"{name}.jsonl"
    rows = [
        l for l in p.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")
    ]
    assert len(rows) >= 3, f"{name} should have at least 3 seed rows"
