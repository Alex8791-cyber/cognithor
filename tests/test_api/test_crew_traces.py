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


def test_group_by_trace_groups_events_by_session_id() -> None:
    from cognithor.api.crew_traces import group_by_trace, read_audit_lines

    events, _ = read_audit_lines(FIXTURE)
    grouped = group_by_trace(events)
    assert "trace-aaa" in grouped
    assert "trace-bbb" in grouped
    assert (
        len(grouped["trace-aaa"]) == 4
    )  # kickoff + task_started + task_completed + kickoff_completed
    assert len(grouped["trace-bbb"]) == 1


def test_derive_trace_meta_computes_status_and_aggregates() -> None:
    from cognithor.api.crew_traces import derive_trace_meta, read_audit_lines

    events, _ = read_audit_lines(FIXTURE)
    aaa_events = [e for e in events if e["session_id"] == "trace-aaa"]
    meta = derive_trace_meta("trace-aaa", aaa_events)
    assert meta["trace_id"] == "trace-aaa"
    assert meta["status"] == "completed"  # crew_kickoff_completed seen
    assert meta["n_tasks"] == 2
    assert meta["total_tokens"] == 1234
    assert meta["agent_count"] == 1
    assert meta["started_at"] == "2026-04-26T10:00:00Z"
    assert meta["ended_at"] == "2026-04-26T10:00:06Z"


def test_derive_trace_meta_returns_running_status_for_unfinished_trace() -> None:
    from cognithor.api.crew_traces import derive_trace_meta, read_audit_lines

    events, _ = read_audit_lines(FIXTURE)
    bbb_events = [e for e in events if e["session_id"] == "trace-bbb"]
    meta = derive_trace_meta("trace-bbb", bbb_events)
    assert meta["status"] == "running"
    assert meta["ended_at"] is None
