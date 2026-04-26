"""Tests for cognithor.api.crew_traces — JSONL reader + endpoint helpers."""

from __future__ import annotations

from pathlib import Path

from cognithor.api.crew_traces import read_audit_lines

FIXTURE = Path(__file__).parent / "fixtures" / "sample_audit.jsonl"


def test_read_audit_lines_skips_corrupt_lines() -> None:
    events, skipped = read_audit_lines(FIXTURE)
    assert len(events) == 5
    assert skipped == 1


def test_read_audit_lines_returns_dicts_with_session_id() -> None:
    events, _ = read_audit_lines(FIXTURE)
    assert events[0]["session_id"] == "trace-aaa"
    assert events[-1]["session_id"] == "trace-bbb"


def test_read_audit_lines_returns_zero_skipped_for_clean_file(tmp_path: Path) -> None:
    clean = tmp_path / "clean.jsonl"
    clean.write_text('{"session_id":"x","event_type":"crew_kickoff_started"}\n', encoding="utf-8")
    events, skipped = read_audit_lines(clean)
    assert len(events) == 1
    assert skipped == 0


def test_read_audit_lines_returns_empty_for_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.jsonl"
    events, skipped = read_audit_lines(missing)
    assert events == []
    assert skipped == 0
