"""Task-level telemetry with success rate and tool latency profile."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from jarvis.db import SQLITE_BUSY_TIMEOUT_MS
from jarvis.security.encrypted_db import encrypted_connect

try:
    from jarvis.security.encrypted_db import compatible_row_factory
except ImportError:
    compatible_row_factory = lambda: sqlite3.Row
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

log = get_logger(__name__)


class TaskTelemetryCollector:
    """Collects task-level telemetry data in SQLite."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = str(db_path) if db_path else ":memory:"
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = encrypted_connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = compatible_row_factory()
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
        return self._conn

    def _ensure_schema(self) -> None:
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS task_telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                success INTEGER NOT NULL,
                duration_ms REAL NOT NULL,
                tools_used TEXT NOT NULL DEFAULT '[]',
                error_type TEXT NOT NULL DEFAULT '',
                error_message TEXT NOT NULL DEFAULT ''
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp
            ON task_telemetry(timestamp)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_telemetry_session
            ON task_telemetry(session_id)
        """)
        conn.commit()

    def record_task(
        self,
        session_id: str,
        success: bool,
        duration_ms: float,
        tool_calls: list[str] | None = None,
        error_type: str = "",
        error_message: str = "",
    ) -> None:
        """Stores a task telemetry measurement."""
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO task_telemetry
               (session_id, timestamp, success, duration_ms, tools_used, error_type, error_message)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                datetime.now(UTC).isoformat(),
                int(success),
                duration_ms,
                json.dumps(tool_calls or []),
                error_type,
                error_message,
            ),
        )
        conn.commit()

    def get_success_rate(self, window_hours: int = 24) -> float:
        """Success rate over a time window."""
        conn = self._get_conn()
        from datetime import timedelta

        cutoff = (datetime.now(UTC) - timedelta(hours=window_hours)).isoformat()

        row = conn.execute(
            """SELECT COUNT(*) as total,
                      SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes
               FROM task_telemetry WHERE timestamp >= ?""",
            (cutoff,),
        ).fetchone()

        total = row["total"] if row else 0
        if total == 0:
            return 0.0
        successes = row["successes"] if row else 0
        return successes / total

    def get_tool_latency_profile(self) -> dict[str, dict[str, float]]:
        """Tool-Latenz-Profil: {tool: {avg, p50, p95, p99}}."""
        conn = self._get_conn()
        rows = conn.execute("SELECT tools_used, duration_ms FROM task_telemetry").fetchall()

        # Collect per-tool latencies
        tool_latencies: dict[str, list[float]] = {}
        for row in rows:
            tools = json.loads(row["tools_used"])
            duration = row["duration_ms"]
            # Distribute duration equally among tools (approximate)
            per_tool = duration / max(len(tools), 1)
            for tool in tools:
                tool_latencies.setdefault(tool, []).append(per_tool)

        result: dict[str, dict[str, float]] = {}
        for tool, latencies in tool_latencies.items():
            latencies.sort()
            n = len(latencies)
            result[tool] = {
                "avg": sum(latencies) / n,
                "p50": latencies[n // 2],
                "p95": latencies[int(n * 0.95)] if n > 1 else latencies[-1],
                "p99": latencies[int(n * 0.99)] if n > 1 else latencies[-1],
            }
        return result

    def get_hourly_stats(self, hours: int = 24) -> list[dict[str, Any]]:
        """Time series for dashboard (hourly aggregation)."""
        conn = self._get_conn()
        from datetime import timedelta

        cutoff = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()

        rows = conn.execute(
            """SELECT
                 substr(timestamp, 1, 13) as hour,
                 COUNT(*) as total,
                 SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes,
                 AVG(duration_ms) as avg_duration
               FROM task_telemetry
               WHERE timestamp >= ?
               GROUP BY hour
               ORDER BY hour""",
            (cutoff,),
        ).fetchall()

        return [
            {
                "hour": row["hour"],
                "total": row["total"],
                "successes": row["successes"],
                "success_rate": row["successes"] / row["total"] if row["total"] > 0 else 0.0,
                "avg_duration_ms": row["avg_duration"],
            }
            for row in rows
        ]

    def get_tool_stats(self) -> dict[str, dict[str, int]]:
        """Tool statistics: {tool_name: {total: N, errors: N}}.

        Aggregated over all recorded tasks.
        """
        conn = self._get_conn()
        rows = conn.execute("SELECT tools_used, success FROM task_telemetry").fetchall()

        stats: dict[str, dict[str, int]] = {}
        for row in rows:
            tools = json.loads(row["tools_used"])
            success = bool(row["success"])
            for tool in tools:
                if tool not in stats:
                    stats[tool] = {"total": 0, "errors": 0}
                stats[tool]["total"] += 1
                if not success:
                    stats[tool]["errors"] += 1
        return stats

    def get_unused_tools(self, since: datetime | None = None) -> list[str]:
        """Tools that have not been used since a given point in time.

        Returns tool names that appear in get_tool_stats()
        but had 0 calls since 'since'. If since=None, returns
        empty list (no reference period).
        """
        if since is None:
            return []

        conn = self._get_conn()
        cutoff = since.isoformat()

        # All tools ever used
        all_rows = conn.execute("SELECT tools_used FROM task_telemetry").fetchall()
        all_tools: set[str] = set()
        for row in all_rows:
            for tool in json.loads(row["tools_used"]):
                all_tools.add(tool)

        # Tools used since cutoff
        recent_rows = conn.execute(
            "SELECT tools_used FROM task_telemetry WHERE timestamp >= ?",
            (cutoff,),
        ).fetchall()
        recent_tools: set[str] = set()
        for row in recent_rows:
            for tool in json.loads(row["tools_used"]):
                recent_tools.add(tool)

        return sorted(all_tools - recent_tools)

    def get_total_tasks(self) -> int:
        """Total number of recorded tasks."""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) as cnt FROM task_telemetry").fetchone()
        return row["cnt"] if row else 0

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
