"""Trace-UI REST endpoints — read crew audit events from JSONL.

Endpoints (all owner-gated):
  GET /api/crew/traces?status=&since=&limit=
  GET /api/crew/trace/{trace_id}
  GET /api/crew/trace/{trace_id}/stats

Source: ~/.cognithor/logs/audit.jsonl (Hashline-Guard chain). Corrupt
lines are skipped with a counter surfaced in response meta.
"""

from __future__ import annotations

import contextlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from cognithor.security.owner import OwnerRequiredError, require_owner

log = logging.getLogger(__name__)


def _require_owner_dep(
    x_cognithor_user: str | None = Header(default=None, alias="X-Cognithor-User"),
) -> str:
    """FastAPI dependency: extract user from header, enforce owner gate."""
    try:
        require_owner(x_cognithor_user)
    except OwnerRequiredError as exc:
        raise HTTPException(
            status_code=403,
            detail={"error": "owner_only", "message": str(exc)},
        ) from exc
    return x_cognithor_user or ""


def read_audit_lines(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Read JSONL audit events. Returns (events, skipped_corrupt_count).

    Missing file → ([], 0). Corrupt JSON lines are logged and skipped;
    valid lines are returned in file order.
    """
    if not path.exists():
        return [], 0

    events: list[dict[str, Any]] = []
    skipped = 0
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        log.warning("audit_jsonl_read_failed path=%s", path, exc_info=exc)
        return [], 0

    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            log.error("audit_jsonl_corruption path=%s line_no=%d", path, line_no)
            skipped += 1
            continue
        if isinstance(obj, dict):
            events.append(obj)
        else:
            skipped += 1
    return events, skipped


def group_by_trace(
    events: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group events by `session_id` (== trace_id), preserving file order."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for ev in events:
        tid = ev.get("session_id")
        if not tid:
            continue
        grouped.setdefault(tid, []).append(ev)
    return grouped


def _event_type(ev: dict[str, Any]) -> str:
    return str(ev.get("event_type") or ev.get("event") or "")


def _details(ev: dict[str, Any]) -> dict[str, Any]:
    d = ev.get("details")
    return d if isinstance(d, dict) else {}


def derive_trace_meta(trace_id: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary metadata for one trace's events.

    Returns: {trace_id, status, started_at, ended_at, duration_ms,
              n_tasks, total_tokens, agent_count, n_failed_guardrails}
    """
    started_at: str | None = None
    ended_at: str | None = None
    n_tasks = 0
    total_tokens = 0
    agents: set[str] = set()
    n_failed_guardrails = 0
    has_kickoff_completed = False
    has_kickoff_failed = False

    for ev in events:
        et = _event_type(ev)
        ts = ev.get("timestamp")
        details = _details(ev)
        if et == "crew_kickoff_started":
            started_at = ts if isinstance(ts, str) else started_at
            n_tasks = int(details.get("n_tasks", n_tasks) or n_tasks)
        elif et == "crew_kickoff_completed":
            has_kickoff_completed = True
            ended_at = ts if isinstance(ts, str) else ended_at
        elif et == "crew_kickoff_failed":
            has_kickoff_failed = True
            ended_at = ts if isinstance(ts, str) else ended_at
        elif et == "crew_task_started":
            role = details.get("agent_role")
            if isinstance(role, str):
                agents.add(role)
        elif et == "crew_task_completed":
            tokens = details.get("tokens", 0) or 0
            with contextlib.suppress(TypeError, ValueError):
                total_tokens += int(tokens)
        elif et == "crew_guardrail_check":
            if details.get("verdict") == "fail":
                n_failed_guardrails += 1

    if has_kickoff_failed:
        status = "failed"
    elif has_kickoff_completed:
        status = "completed"
    else:
        status = "running"

    duration_ms: float | None = None
    if started_at and ended_at:
        try:
            t0 = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
            duration_ms = (t1 - t0).total_seconds() * 1000.0
        except ValueError:
            duration_ms = None

    return {
        "trace_id": trace_id,
        "status": status,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_ms": duration_ms,
        "n_tasks": n_tasks,
        "total_tokens": total_tokens,
        "agent_count": len(agents),
        "n_failed_guardrails": n_failed_guardrails,
    }


def derive_trace_stats(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute Stats-Sidebar aggregates: per-agent tokens + guardrail summary."""
    total_tokens = 0
    agent_breakdown: dict[str, int] = {}
    guardrail_pass = 0
    guardrail_fail = 0
    guardrail_retries = 0

    # Track current agent per task_id so we can attribute tokens correctly.
    task_agent: dict[str, str] = {}
    for ev in events:
        et = _event_type(ev)
        details = _details(ev)
        if et == "crew_task_started":
            tid = str(details.get("task_id", ""))
            role = details.get("agent_role")
            if tid and isinstance(role, str):
                task_agent[tid] = role
        elif et == "crew_task_completed":
            tid = str(details.get("task_id", ""))
            tokens_val = details.get("tokens", 0) or 0
            tok = 0
            with contextlib.suppress(TypeError, ValueError):
                tok = int(tokens_val)
            total_tokens += tok
            role = task_agent.get(tid)
            if role:
                agent_breakdown[role] = agent_breakdown.get(role, 0) + tok
        elif et == "crew_guardrail_check":
            verdict = details.get("verdict")
            retry_count_val = details.get("retry_count", 0) or 0
            rc = 0
            with contextlib.suppress(TypeError, ValueError):
                rc = int(retry_count_val)
            guardrail_retries += rc
            if verdict == "pass":
                guardrail_pass += 1
            elif verdict == "fail":
                guardrail_fail += 1

    total_duration_ms: float | None = None
    if events:
        # Attempt to compute total duration from first task_started → last task_completed.
        first_ts: str | None = None
        last_ts: str | None = None
        for ev in events:
            ts = ev.get("timestamp")
            if not isinstance(ts, str):
                continue
            if first_ts is None:
                first_ts = ts
            last_ts = ts
        if first_ts and last_ts:
            with contextlib.suppress(ValueError):
                t0 = datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
                t1 = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
                total_duration_ms = (t1 - t0).total_seconds() * 1000.0

    return {
        "total_tokens": total_tokens,
        "total_duration_ms": total_duration_ms,
        "agent_breakdown": agent_breakdown,
        "guardrail_summary": {
            "pass": guardrail_pass,
            "fail": guardrail_fail,
            "retries": guardrail_retries,
        },
    }


router = APIRouter(prefix="/api/crew", tags=["crew-traces"])


def _audit_path() -> Path:
    """Return the on-disk path to the audit JSONL. Override via monkeypatch in tests."""
    from cognithor.config import load_config

    cfg = load_config()
    return Path(cfg.cognithor_home) / "logs" / "audit.jsonl"


@router.get("/traces")
def list_traces(
    status: str | None = Query(
        default=None, description="Filter by status (running|completed|failed)"
    ),
    since: str | None = Query(
        default=None, description="Filter to traces started after ISO-8601 timestamp"
    ),
    limit: int = Query(default=50, ge=1, le=1000, description="Max number of traces to return"),
    _user: str = Depends(_require_owner_dep),
) -> dict[str, Any]:
    """List all traces with derived metadata."""
    events, skipped = read_audit_lines(_audit_path())
    grouped = group_by_trace(events)
    traces = [derive_trace_meta(tid, evs) for tid, evs in grouped.items()]
    if status:
        traces = [t for t in traces if t["status"] == status]
    if since:
        traces = [t for t in traces if t["started_at"] and t["started_at"] >= since]
    traces.sort(
        key=lambda t: (t["started_at"] is None, t["started_at"] or ""),
        reverse=True,
    )
    traces = traces[:limit]
    return {"traces": traces, "meta": {"skipped_lines": skipped}}


@router.get("/trace/{trace_id}")
def get_trace(trace_id: str, _user: str = Depends(_require_owner_dep)) -> dict[str, Any]:
    """Return full event list for one trace_id."""
    events, skipped = read_audit_lines(_audit_path())
    grouped = group_by_trace(events)
    if trace_id not in grouped:
        raise HTTPException(
            status_code=404,
            detail={"error": "trace_not_found", "trace_id": trace_id},
        )
    return {
        "trace_id": trace_id,
        "events": grouped[trace_id],
        "meta": {"skipped_lines": skipped},
    }


@router.get("/trace/{trace_id}/stats")
def get_trace_stats(trace_id: str, _user: str = Depends(_require_owner_dep)) -> dict[str, Any]:
    """Return derived per-trace aggregates."""
    events, _skipped = read_audit_lines(_audit_path())
    grouped = group_by_trace(events)
    if trace_id not in grouped:
        raise HTTPException(
            status_code=404,
            detail={"error": "trace_not_found", "trace_id": trace_id},
        )
    return derive_trace_stats(grouped[trace_id])
