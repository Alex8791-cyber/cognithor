"""Run Recorder: Records complete agent runs to SQLite for forensic analysis.

Captures every step of an agent run -- plans, gate decisions, tool results,
reflections, and policy snapshots -- into a local SQLite database.  This
enables post-hoc analysis, debugging, and replay of historical runs.

Architecture reference: Phase 2 Intelligence -- Run Recording + Replay.
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jarvis.models import (
    ActionPlan,
    GateDecision,
    ReflectionResult,
    RunRecord,
    RunSummary,
    ToolResult,
)
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# SQL schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS runs (
    id            TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL,
    timestamp     TEXT NOT NULL,
    user_message  TEXT NOT NULL DEFAULT '',
    operation_mode TEXT NOT NULL DEFAULT '',
    success       INTEGER NOT NULL DEFAULT 0,
    final_response TEXT NOT NULL DEFAULT '',
    duration_ms   REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS run_steps (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL REFERENCES runs(id),
    step_order  INTEGER NOT NULL,
    step_type   TEXT NOT NULL,
    data_json   TEXT NOT NULL DEFAULT '{}',
    timestamp   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_policy_snapshots (
    run_id        TEXT PRIMARY KEY REFERENCES runs(id),
    policies_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_run_steps_run_id ON run_steps(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_session_id ON runs(session_id);
CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON runs(timestamp);
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def _parse_iso(ts: str) -> datetime:
    """Parse an ISO-8601 timestamp safely (handles 'Z' suffix and naive datetimes)."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _serialize(obj: Any) -> str:
    """Serialize a Pydantic model (or any JSON-serialisable object) to a JSON string."""
    if hasattr(obj, "model_dump"):
        return json.dumps(obj.model_dump(), default=str, ensure_ascii=False)
    return json.dumps(obj, default=str, ensure_ascii=False)


def _serialize_list(items: list[Any]) -> str:
    """Serialize a list of Pydantic models to a JSON string."""
    dumped = [
        item.model_dump() if hasattr(item, "model_dump") else item
        for item in items
    ]
    return json.dumps(dumped, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# RunRecorder
# ---------------------------------------------------------------------------


class RunRecorder:
    """Records complete agent runs into a local SQLite database.

    Usage::

        recorder = RunRecorder(db_path)
        run_id = recorder.start_run(session_id, user_message, operation_mode)
        recorder.record_plan(run_id, plan)
        recorder.record_gate_decisions(run_id, decisions)
        recorder.record_tool_results(run_id, results)
        recorder.record_reflection(run_id, reflection)
        recorder.record_policy_snapshot(run_id, policies_dict)
        recorder.finish_run(run_id, success=True, final_response="Done.")
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

        # Per-run step counters: run_id -> next step_order
        self._step_counters: dict[str, int] = {}
        # Per-run start timestamps (monotonic) for duration calculation
        self._start_times: dict[str, float] = {}

        log.info("run_recorder_initialized", db_path=str(self._db_path))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_run(
        self,
        session_id: str,
        user_message: str = "",
        operation_mode: str = "",
    ) -> str:
        """Begin recording a new run.  Returns the generated run ID."""
        from jarvis.models import _new_id  # noqa: WPS433

        run_id = _new_id()
        ts = _utc_iso()

        self._conn.execute(
            "INSERT INTO runs (id, session_id, timestamp, user_message, operation_mode) "
            "VALUES (?, ?, ?, ?, ?)",
            (run_id, session_id, ts, user_message, operation_mode),
        )
        self._conn.commit()

        self._step_counters[run_id] = 0
        self._start_times[run_id] = time.monotonic()

        log.info("run_started", run_id=run_id, session_id=session_id)
        return run_id

    def record_plan(self, run_id: str, plan: ActionPlan) -> None:
        """Record an ActionPlan step for the given run."""
        self._record_step(run_id, "plan", _serialize(plan))

    def record_gate_decisions(
        self,
        run_id: str,
        decisions: list[GateDecision],
    ) -> None:
        """Record a list of GateDecisions for the given run."""
        self._record_step(run_id, "gate_decisions", _serialize_list(decisions))

    def record_tool_results(
        self,
        run_id: str,
        results: list[ToolResult],
    ) -> None:
        """Record a list of ToolResults for the given run."""
        self._record_step(run_id, "tool_results", _serialize_list(results))

    def record_reflection(
        self,
        run_id: str,
        reflection: ReflectionResult,
    ) -> None:
        """Record a ReflectionResult for the given run."""
        self._record_step(run_id, "reflection", _serialize(reflection))

    def record_policy_snapshot(
        self,
        run_id: str,
        policies: dict[str, Any],
    ) -> None:
        """Store a snapshot of the active policies for the given run."""
        policies_json = json.dumps(policies, default=str, ensure_ascii=False)
        self._conn.execute(
            "INSERT OR REPLACE INTO run_policy_snapshots (run_id, policies_json) "
            "VALUES (?, ?)",
            (run_id, policies_json),
        )
        self._conn.commit()
        log.debug("policy_snapshot_recorded", run_id=run_id)

    def finish_run(
        self,
        run_id: str,
        success: bool = True,
        final_response: str = "",
    ) -> None:
        """Mark a run as finished, calculating its total duration."""
        start = self._start_times.pop(run_id, None)
        duration_ms = (time.monotonic() - start) * 1000.0 if start is not None else 0.0

        self._conn.execute(
            "UPDATE runs SET success = ?, final_response = ?, duration_ms = ? "
            "WHERE id = ?",
            (int(success), final_response, duration_ms, run_id),
        )
        self._conn.commit()
        self._step_counters.pop(run_id, None)

        log.info(
            "run_finished",
            run_id=run_id,
            success=success,
            duration_ms=round(duration_ms, 1),
        )

    def get_run(self, run_id: str) -> RunRecord | None:
        """Load a complete RunRecord from the database.

        Returns ``None`` if the run does not exist.
        """
        cur = self._conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
        row = cur.fetchone()
        if row is None:
            return None

        col_names = [desc[0] for desc in cur.description]
        run_data = dict(zip(col_names, row))

        # Reconstruct lists from steps
        plans: list[ActionPlan] = []
        gate_decisions: list[list[GateDecision]] = []
        tool_results: list[list[ToolResult]] = []
        reflection: ReflectionResult | None = None

        step_cur = self._conn.execute(
            "SELECT step_type, data_json FROM run_steps "
            "WHERE run_id = ? ORDER BY step_order",
            (run_id,),
        )
        for step_type, data_json in step_cur.fetchall():
            data = json.loads(data_json)
            if step_type == "plan":
                plans.append(ActionPlan.model_validate(data))
            elif step_type == "gate_decisions":
                gate_decisions.append([GateDecision.model_validate(d) for d in data])
            elif step_type == "tool_results":
                tool_results.append([ToolResult.model_validate(d) for d in data])
            elif step_type == "reflection":
                reflection = ReflectionResult.model_validate(data)

        # Policy snapshot
        snap_cur = self._conn.execute(
            "SELECT policies_json FROM run_policy_snapshots WHERE run_id = ?",
            (run_id,),
        )
        snap_row = snap_cur.fetchone()
        policy_snapshot: dict[str, Any] = (
            json.loads(snap_row[0]) if snap_row else {}
        )

        return RunRecord(
            id=run_data["id"],
            session_id=run_data["session_id"],
            timestamp=_parse_iso(run_data["timestamp"]),
            user_message=run_data["user_message"],
            operation_mode=run_data["operation_mode"],
            success=bool(run_data["success"]),
            final_response=run_data["final_response"],
            duration_ms=run_data["duration_ms"],
            plans=plans,
            gate_decisions=gate_decisions,
            tool_results=tool_results,
            reflection=reflection,
            policy_snapshot=policy_snapshot,
        )

    def list_runs(
        self,
        session_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[RunSummary]:
        """List run summaries, optionally filtered by session_id.

        Returns summaries ordered by timestamp descending (newest first).
        """
        if session_id:
            cur = self._conn.execute(
                "SELECT r.id, r.session_id, r.timestamp, r.user_message, "
                "r.success, r.duration_ms, "
                "(SELECT COUNT(*) FROM run_steps s "
                " WHERE s.run_id = r.id AND s.step_type = 'tool_results') AS tool_count "
                "FROM runs r WHERE r.session_id = ? "
                "ORDER BY r.timestamp DESC LIMIT ? OFFSET ?",
                (session_id, limit, offset),
            )
        else:
            cur = self._conn.execute(
                "SELECT r.id, r.session_id, r.timestamp, r.user_message, "
                "r.success, r.duration_ms, "
                "(SELECT COUNT(*) FROM run_steps s "
                " WHERE s.run_id = r.id AND s.step_type = 'tool_results') AS tool_count "
                "FROM runs r "
                "ORDER BY r.timestamp DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )

        summaries: list[RunSummary] = []
        for row in cur.fetchall():
            run_id, sess_id, ts, msg, success, dur, tool_count = row
            summaries.append(
                RunSummary(
                    id=run_id,
                    session_id=sess_id,
                    timestamp=_parse_iso(ts),
                    user_message_preview=msg[:120] if msg else "",
                    success=bool(success),
                    duration_ms=dur,
                    tool_count=tool_count,
                )
            )

        return summaries

    def close(self) -> None:
        """Close the underlying database connection."""
        try:
            self._conn.close()
            log.info("run_recorder_closed")
        except Exception as exc:
            log.warning("run_recorder_close_error", error=str(exc))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _record_step(self, run_id: str, step_type: str, data_json: str) -> None:
        """Insert a step row into run_steps."""
        order = self._step_counters.get(run_id, 0)
        self._step_counters[run_id] = order + 1

        self._conn.execute(
            "INSERT INTO run_steps (run_id, step_order, step_type, data_json, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (run_id, order, step_type, data_json, _utc_iso()),
        )
        self._conn.commit()
        log.debug(
            "run_step_recorded",
            run_id=run_id,
            step_type=step_type,
            step_order=order,
        )
