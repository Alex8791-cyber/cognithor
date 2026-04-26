"""Tests for cognithor.api.crew_traces — JSONL reader + endpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from cognithor.api.crew_traces import read_audit_lines

if TYPE_CHECKING:
    import pytest

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


def test_derive_trace_stats_aggregates_per_agent_tokens() -> None:
    from cognithor.api.crew_traces import derive_trace_stats, read_audit_lines

    events, _ = read_audit_lines(FIXTURE)
    aaa_events = [e for e in events if e["session_id"] == "trace-aaa"]
    stats = derive_trace_stats(aaa_events)

    assert stats["total_tokens"] == 1234
    assert stats["agent_breakdown"] == {"researcher": 1234}
    assert stats["guardrail_summary"]["pass"] == 0
    assert stats["guardrail_summary"]["fail"] == 0
    assert stats["guardrail_summary"]["retries"] == 0


def test_derive_trace_stats_counts_guardrail_verdicts(tmp_path: Path) -> None:
    from cognithor.api.crew_traces import derive_trace_stats, read_audit_lines

    f = tmp_path / "stats.jsonl"
    f.write_text(
        "\n".join(
            [
                '{"session_id":"x","event_type":"crew_task_started","details":{"agent_role":"a","task_id":"t1"}}',
                '{"session_id":"x","event_type":"crew_guardrail_check","details":{"verdict":"fail","retry_count":1}}',
                '{"session_id":"x","event_type":"crew_guardrail_check","details":{"verdict":"pass","retry_count":0}}',
                '{"session_id":"x","event_type":"crew_task_completed","details":{"task_id":"t1","tokens":500}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    events, _ = read_audit_lines(f)
    stats = derive_trace_stats(events)
    assert stats["guardrail_summary"]["pass"] == 1
    assert stats["guardrail_summary"]["fail"] == 1
    assert stats["guardrail_summary"]["retries"] == 1
    assert stats["agent_breakdown"] == {"a": 500}


def test_list_traces_endpoint_returns_grouped_meta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from cognithor.api.crew_traces import router

    monkeypatch.setattr("cognithor.api.crew_traces._audit_path", lambda: FIXTURE)
    monkeypatch.setenv("COGNITHOR_OWNER_USER_ID", "test-owner")

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    resp = client.get("/api/crew/traces", headers={"X-Cognithor-User": "test-owner"})
    assert resp.status_code == 200
    body = resp.json()
    assert "traces" in body
    trace_ids = [t["trace_id"] for t in body["traces"]]
    assert "trace-aaa" in trace_ids
    assert "trace-bbb" in trace_ids


def test_get_trace_endpoint_returns_events(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from cognithor.api.crew_traces import router

    monkeypatch.setattr("cognithor.api.crew_traces._audit_path", lambda: FIXTURE)
    monkeypatch.setenv("COGNITHOR_OWNER_USER_ID", "test-owner")
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    resp = client.get("/api/crew/trace/trace-aaa", headers={"X-Cognithor-User": "test-owner"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["trace_id"] == "trace-aaa"
    assert len(body["events"]) == 4
    assert body["meta"]["skipped_lines"] == 1


def test_get_trace_endpoint_404_for_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from cognithor.api.crew_traces import router

    monkeypatch.setattr("cognithor.api.crew_traces._audit_path", lambda: FIXTURE)
    monkeypatch.setenv("COGNITHOR_OWNER_USER_ID", "test-owner")
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    resp = client.get(
        "/api/crew/trace/does-not-exist",
        headers={"X-Cognithor-User": "test-owner"},
    )
    assert resp.status_code == 404
    assert resp.json()["detail"]["error"] == "trace_not_found"


def test_get_trace_stats_endpoint_returns_aggregates(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from cognithor.api.crew_traces import router

    monkeypatch.setattr("cognithor.api.crew_traces._audit_path", lambda: FIXTURE)
    monkeypatch.setenv("COGNITHOR_OWNER_USER_ID", "test-owner")
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    resp = client.get(
        "/api/crew/trace/trace-aaa/stats",
        headers={"X-Cognithor-User": "test-owner"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_tokens"] == 1234
    assert "agent_breakdown" in body


def test_list_traces_403_for_non_owner_token(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from cognithor.api.crew_traces import router

    monkeypatch.setattr("cognithor.api.crew_traces._audit_path", lambda: FIXTURE)
    monkeypatch.setenv("COGNITHOR_OWNER_USER_ID", "real-owner")

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    resp = client.get("/api/crew/traces", headers={"X-Cognithor-User": "guest"})
    assert resp.status_code == 403
    assert resp.json()["detail"]["error"] == "owner_only"


def test_list_traces_200_for_owner_token(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from cognithor.api.crew_traces import router

    monkeypatch.setattr("cognithor.api.crew_traces._audit_path", lambda: FIXTURE)
    monkeypatch.setenv("COGNITHOR_OWNER_USER_ID", "owner-x")

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    resp = client.get("/api/crew/traces", headers={"X-Cognithor-User": "owner-x"})
    assert resp.status_code == 200


def test_list_traces_filters_by_status(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from cognithor.api.crew_traces import router

    monkeypatch.setattr("cognithor.api.crew_traces._audit_path", lambda: FIXTURE)
    monkeypatch.setenv("COGNITHOR_OWNER_USER_ID", "owner")
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    resp = client.get(
        "/api/crew/traces?status=running",
        headers={"X-Cognithor-User": "owner"},
    )
    assert resp.status_code == 200
    statuses = [t["status"] for t in resp.json()["traces"]]
    assert "running" in statuses
    assert "completed" not in statuses


def test_list_traces_respects_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from cognithor.api.crew_traces import router

    monkeypatch.setattr("cognithor.api.crew_traces._audit_path", lambda: FIXTURE)
    monkeypatch.setenv("COGNITHOR_OWNER_USER_ID", "owner")
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    resp = client.get(
        "/api/crew/traces?limit=1",
        headers={"X-Cognithor-User": "owner"},
    )
    assert resp.status_code == 200
    assert len(resp.json()["traces"]) == 1


def test_app_smoke_includes_crew_traces_router() -> None:
    """Verify the application factory mounts the crew-traces router."""
    import pytest as _pytest

    try:
        from cognithor.api import build_app  # type: ignore[attr-defined]
    except ImportError:
        _pytest.skip("Application factory not present; skipping mount smoke")
    app = build_app()
    paths = {r.path for r in app.routes if hasattr(r, "path")}
    assert "/api/crew/traces" in paths
    assert "/api/crew/trace/{trace_id}" in paths
