"""Execution trace data model and SQLite persistence for GEPA.

GEPA (Guided Evolution through Pattern Analysis) uses execution traces
to record every tool call and decision point during an agent execution
cycle (one user message -> response).  The TraceStore persists these
traces in SQLite so that downstream components (CausalAnalyzer,
RewardCalculator, PromptEvolution) can analyse patterns and drive
self-improvement.

Components:
  - TraceStep: A single tool call or decision point.
  - ExecutionTrace: Complete trace of one agent execution cycle.
  - TraceStore: SQLite persistence with WAL mode.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jarvis.db import SQLITE_BUSY_TIMEOUT_MS
from jarvis.security.encrypted_db import encrypted_connect

try:
    from jarvis.security.encrypted_db import compatible_row_factory
except ImportError:
    compatible_row_factory = lambda: sqlite3.Row
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_INPUT_SUMMARY = 500
_MAX_OUTPUT_SUMMARY = 500
_MAX_GOAL_LENGTH = 1000

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class TraceStep:
    """A single step in an execution trace -- one tool call or decision point."""

    step_id: str
    parent_id: str | None
    tool_name: str
    input_summary: str
    output_summary: str
    status: str  # "success", "error", "skipped", "timeout"
    error_detail: str = ""
    duration_ms: int = 0
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionTrace:
    """Complete trace of one agent execution (one user message -> response cycle)."""

    trace_id: str
    session_id: str
    goal: str
    steps: list[TraceStep] = field(default_factory=list)
    total_duration_ms: int = 0
    success_score: float = 0.0
    model_used: str = ""
    created_at: float = 0.0

    # -- derived properties --------------------------------------------------

    @property
    def failed_steps(self) -> list[TraceStep]:
        return [s for s in self.steps if s.status in ("error", "timeout")]

    @property
    def tool_sequence(self) -> list[str]:
        return [s.tool_name for s in self.steps]

    def get_step(self, step_id: str) -> TraceStep | None:
        return next((s for s in self.steps if s.step_id == step_id), None)

    def get_children(self, parent_id: str) -> list[TraceStep]:
        return [s for s in self.steps if s.parent_id == parent_id]

    def get_causal_chain(self, step_id: str) -> list[TraceStep]:
        """Walk up parent chain from step to root."""
        chain: list[TraceStep] = []
        current = self.get_step(step_id)
        while current:
            chain.append(current)
            current = self.get_step(current.parent_id) if current.parent_id else None
        chain.reverse()
        return chain


# ---------------------------------------------------------------------------
# SQL schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS execution_traces (
    trace_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    goal TEXT NOT NULL,
    success_score REAL DEFAULT 0.0,
    model_used TEXT DEFAULT '',
    total_duration_ms INTEGER DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_traces_session ON execution_traces(session_id);
CREATE INDEX IF NOT EXISTS idx_traces_created ON execution_traces(created_at);
CREATE INDEX IF NOT EXISTS idx_traces_score ON execution_traces(success_score);

CREATE TABLE IF NOT EXISTS trace_steps (
    step_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL REFERENCES execution_traces(trace_id) ON DELETE CASCADE,
    parent_id TEXT,
    seq INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    input_summary TEXT DEFAULT '',
    output_summary TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'success',
    error_detail TEXT DEFAULT '',
    duration_ms INTEGER DEFAULT 0,
    ts REAL NOT NULL,
    metadata_json TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_steps_trace ON trace_steps(trace_id);
CREATE INDEX IF NOT EXISTS idx_steps_tool ON trace_steps(tool_name);
CREATE INDEX IF NOT EXISTS idx_steps_status ON trace_steps(status);
"""

# ---------------------------------------------------------------------------
# TraceStore
# ---------------------------------------------------------------------------


class TraceStore:
    """SQLite persistence for execution traces."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    # ------------------------------------------------------------------
    # DB management
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        """Return the DB connection (lazy init)."""
        if self._conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = encrypted_connect(str(self._db_path), check_same_thread=False)
            self._conn.row_factory = compatible_row_factory()
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
            self._conn.execute("PRAGMA foreign_keys = ON")
        return self._conn

    def _init_db(self) -> None:
        """Create tables if they do not exist."""
        try:
            conn = self._get_conn()
            conn.executescript(_SCHEMA_SQL)
            conn.commit()
        except Exception as exc:
            log.error("trace_store_init_failed", error=str(exc))
            raise

    # ------------------------------------------------------------------
    # Public API: save
    # ------------------------------------------------------------------

    def save_trace(self, trace: ExecutionTrace) -> None:
        """Save a complete trace with all steps in a transaction."""
        try:
            conn = self._get_conn()
            goal = trace.goal[:_MAX_GOAL_LENGTH] if trace.goal else ""
            with conn:
                conn.execute(
                    """INSERT OR REPLACE INTO execution_traces
                       (trace_id, session_id, goal, success_score,
                        model_used, total_duration_ms, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        trace.trace_id,
                        trace.session_id,
                        goal,
                        trace.success_score,
                        trace.model_used,
                        trace.total_duration_ms,
                        trace.created_at or time.time(),
                    ),
                )
                # Delete existing steps (for idempotent re-saves)
                conn.execute(
                    "DELETE FROM trace_steps WHERE trace_id = ?",
                    (trace.trace_id,),
                )
                for seq, step in enumerate(trace.steps):
                    input_sum = (step.input_summary or "")[:_MAX_INPUT_SUMMARY]
                    output_sum = (step.output_summary or "")[:_MAX_OUTPUT_SUMMARY]
                    metadata_json = json.dumps(step.metadata, default=str)
                    conn.execute(
                        """INSERT INTO trace_steps
                           (step_id, trace_id, parent_id, seq, tool_name,
                            input_summary, output_summary, status,
                            error_detail, duration_ms, ts, metadata_json)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            step.step_id,
                            trace.trace_id,
                            step.parent_id,
                            seq,
                            step.tool_name,
                            input_sum,
                            output_sum,
                            step.status,
                            step.error_detail,
                            step.duration_ms,
                            step.timestamp or time.time(),
                            metadata_json,
                        ),
                    )
            log.debug(
                "trace_saved",
                trace_id=trace.trace_id,
                steps=len(trace.steps),
            )
        except Exception as exc:
            log.error("trace_save_failed", trace_id=trace.trace_id, error=str(exc))
            raise

    # ------------------------------------------------------------------
    # Public API: read
    # ------------------------------------------------------------------

    def get_trace(self, trace_id: str) -> ExecutionTrace | None:
        """Load a trace with all its steps."""
        try:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT * FROM execution_traces WHERE trace_id = ?",
                (trace_id,),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_trace(row)
        except Exception as exc:
            log.error("trace_get_failed", trace_id=trace_id, error=str(exc))
            return None

    def get_traces_by_session(self, session_id: str) -> list[ExecutionTrace]:
        """All traces for a session, ordered by created_at."""
        try:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT * FROM execution_traces WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            ).fetchall()
            return [self._row_to_trace(r) for r in rows]
        except Exception as exc:
            log.error("traces_by_session_failed", session_id=session_id, error=str(exc))
            return []

    def get_traces_by_tool(self, tool_name: str, limit: int = 50) -> list[ExecutionTrace]:
        """Traces that contain at least one step using this tool."""
        try:
            conn = self._get_conn()
            rows = conn.execute(
                """SELECT DISTINCT e.* FROM execution_traces e
                   JOIN trace_steps s ON e.trace_id = s.trace_id
                   WHERE s.tool_name = ?
                   ORDER BY e.created_at DESC
                   LIMIT ?""",
                (tool_name, limit),
            ).fetchall()
            return [self._row_to_trace(r) for r in rows]
        except Exception as exc:
            log.error("traces_by_tool_failed", tool_name=tool_name, error=str(exc))
            return []

    def get_failed_traces(self, since_hours: int = 24, limit: int = 100) -> list[ExecutionTrace]:
        """Traces with success_score < 0.5 in the last N hours."""
        try:
            conn = self._get_conn()
            cutoff = time.time() - (since_hours * 3600)
            rows = conn.execute(
                """SELECT * FROM execution_traces
                   WHERE success_score < 0.5 AND created_at >= ?
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (cutoff, limit),
            ).fetchall()
            return [self._row_to_trace(r) for r in rows]
        except Exception as exc:
            log.error("failed_traces_failed", error=str(exc))
            return []

    def get_recent_traces(self, limit: int = 50) -> list[ExecutionTrace]:
        """Most recent traces ordered by created_at DESC."""
        try:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT * FROM execution_traces ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._row_to_trace(r) for r in rows]
        except Exception as exc:
            log.error("recent_traces_failed", error=str(exc))
            return []

    def get_traces_since(self, timestamp: float) -> list[ExecutionTrace]:
        """All traces since a given timestamp."""
        try:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT * FROM execution_traces WHERE created_at >= ? ORDER BY created_at",
                (timestamp,),
            ).fetchall()
            return [self._row_to_trace(r) for r in rows]
        except Exception as exc:
            log.error("traces_since_failed", error=str(exc))
            return []

    # ------------------------------------------------------------------
    # Public API: stats
    # ------------------------------------------------------------------

    def get_trace_stats(self, since_hours: int = 24) -> dict[str, Any]:
        """Summary stats: total, success_rate, avg_duration, top_failing_tools, avg_steps."""
        try:
            conn = self._get_conn()
            cutoff = time.time() - (since_hours * 3600)

            # Basic aggregates
            agg = conn.execute(
                """SELECT
                       COUNT(*) AS total,
                       AVG(success_score) AS avg_score,
                       AVG(total_duration_ms) AS avg_duration
                   FROM execution_traces
                   WHERE created_at >= ?""",
                (cutoff,),
            ).fetchone()

            total = agg["total"] or 0
            avg_score = agg["avg_score"] or 0.0
            avg_duration = agg["avg_duration"] or 0.0

            # Success rate (score >= 0.5 counts as success)
            success_row = conn.execute(
                """SELECT COUNT(*) AS cnt FROM execution_traces
                   WHERE created_at >= ? AND success_score >= 0.5""",
                (cutoff,),
            ).fetchone()
            success_count = success_row["cnt"] or 0
            success_rate = (success_count / total) if total > 0 else 0.0

            # Average steps per trace
            avg_steps_row = conn.execute(
                """SELECT AVG(step_count) AS avg_steps FROM (
                       SELECT COUNT(*) AS step_count
                       FROM trace_steps s
                       JOIN execution_traces e ON s.trace_id = e.trace_id
                       WHERE e.created_at >= ?
                       GROUP BY s.trace_id
                   )""",
                (cutoff,),
            ).fetchone()
            avg_steps = (avg_steps_row["avg_steps"] or 0.0) if avg_steps_row else 0.0

            # Top failing tools
            failing_rows = conn.execute(
                """SELECT s.tool_name, COUNT(*) AS fail_count
                   FROM trace_steps s
                   JOIN execution_traces e ON s.trace_id = e.trace_id
                   WHERE e.created_at >= ? AND s.status IN ('error', 'timeout')
                   GROUP BY s.tool_name
                   ORDER BY fail_count DESC
                   LIMIT 10""",
                (cutoff,),
            ).fetchall()
            top_failing_tools = [
                {"tool_name": r["tool_name"], "fail_count": r["fail_count"]} for r in failing_rows
            ]

            return {
                "total": total,
                "success_rate": round(success_rate, 4),
                "avg_score": round(avg_score, 4),
                "avg_duration_ms": round(avg_duration, 1),
                "avg_steps": round(avg_steps, 2),
                "top_failing_tools": top_failing_tools,
            }
        except Exception as exc:
            log.error("trace_stats_failed", error=str(exc))
            return {
                "total": 0,
                "success_rate": 0.0,
                "avg_score": 0.0,
                "avg_duration_ms": 0.0,
                "avg_steps": 0.0,
                "top_failing_tools": [],
            }

    # ------------------------------------------------------------------
    # Public API: cleanup
    # ------------------------------------------------------------------

    def delete_old_traces(self, older_than_days: int = 30) -> int:
        """Cleanup old traces, return count deleted."""
        try:
            conn = self._get_conn()
            cutoff = time.time() - (older_than_days * 86400)
            # Steps are deleted via ON DELETE CASCADE
            cursor = conn.execute(
                "DELETE FROM execution_traces WHERE created_at < ?",
                (cutoff,),
            )
            deleted = cursor.rowcount
            conn.commit()
            if deleted > 0:
                log.info("old_traces_deleted", count=deleted, older_than_days=older_than_days)
            return deleted
        except Exception as exc:
            log.error("delete_old_traces_failed", error=str(exc))
            return 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _row_to_trace(self, row: sqlite3.Row) -> ExecutionTrace:
        """Convert a DB row into an ExecutionTrace with its steps loaded."""
        conn = self._get_conn()
        step_rows = conn.execute(
            "SELECT * FROM trace_steps WHERE trace_id = ? ORDER BY seq",
            (row["trace_id"],),
        ).fetchall()

        steps = [self._row_to_step(sr) for sr in step_rows]

        return ExecutionTrace(
            trace_id=row["trace_id"],
            session_id=row["session_id"],
            goal=row["goal"],
            steps=steps,
            total_duration_ms=row["total_duration_ms"],
            success_score=row["success_score"],
            model_used=row["model_used"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_step(row: sqlite3.Row) -> TraceStep:
        """Convert a DB row into a TraceStep."""
        try:
            metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
        except (json.JSONDecodeError, TypeError):
            metadata = {}

        return TraceStep(
            step_id=row["step_id"],
            parent_id=row["parent_id"],
            tool_name=row["tool_name"],
            input_summary=row["input_summary"] or "",
            output_summary=row["output_summary"] or "",
            status=row["status"],
            error_detail=row["error_detail"] or "",
            duration_ms=row["duration_ms"],
            timestamp=row["ts"],
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the DB connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
