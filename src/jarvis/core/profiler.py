"""Self-Profiling: Sammelt Tool-Latenzen, Erfolgsraten und Task-Kategorien."""

from __future__ import annotations

import sqlite3
import json
import time
import math
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from jarvis.models import ToolProfile, TaskProfile, CapabilityProfile
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class TaskProfiler:
    """Sammelt und aggregiert Performance-Daten ueber Tools und Tasks."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = str(db_path) if db_path else ":memory:"
        self._conn: sqlite3.Connection | None = None
        self._active_tasks: dict[str, dict[str, Any]] = {}
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
        return self._conn

    def _ensure_schema(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS tool_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                success INTEGER NOT NULL,
                error_type TEXT NOT NULL DEFAULT '',
                timestamp TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_tool_calls_tool ON tool_calls(tool_name);
            CREATE INDEX IF NOT EXISTS idx_tool_calls_session ON tool_calls(session_id);

            CREATE TABLE IF NOT EXISTS task_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL DEFAULT '',
                category TEXT NOT NULL DEFAULT 'general',
                start_time TEXT NOT NULL,
                end_time TEXT,
                success_score REAL,
                tools_used TEXT NOT NULL DEFAULT '[]'
            );
            CREATE INDEX IF NOT EXISTS idx_task_records_category ON task_records(category);
        """)
        conn.commit()

    def start_task(self, session_id: str, task_description: str = "", category: str = "general") -> None:
        """Oeffnet einen Profiling-Kontext fuer einen Task."""
        conn = self._get_conn()
        now = datetime.now(UTC).isoformat()
        self._active_tasks[session_id] = {
            "start_time": time.monotonic(),
            "tools": [],
        }
        conn.execute(
            """INSERT OR REPLACE INTO task_records
               (session_id, description, category, start_time, tools_used)
               VALUES (?, ?, ?, ?, '[]')""",
            (session_id, task_description, category, now),
        )
        conn.commit()

    def record_tool_call(
        self,
        tool_name: str,
        latency_ms: float,
        success: bool,
        error_type: str = "",
        session_id: str = "",
    ) -> None:
        """Zeichnet einen Tool-Call auf."""
        conn = self._get_conn()
        now = datetime.now(UTC).isoformat()
        conn.execute(
            """INSERT INTO tool_calls (session_id, tool_name, latency_ms, success, error_type, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, tool_name, latency_ms, int(success), error_type, now),
        )
        conn.commit()

        # Track in active task
        if session_id in self._active_tasks:
            self._active_tasks[session_id]["tools"].append(tool_name)

    def finish_task(self, session_id: str, success_score: float = 0.0) -> None:
        """Schliesst einen Task-Profiling-Kontext ab."""
        conn = self._get_conn()
        now = datetime.now(UTC).isoformat()

        tools = []
        if session_id in self._active_tasks:
            tools = self._active_tasks.pop(session_id)["tools"]

        conn.execute(
            """UPDATE task_records
               SET end_time = ?, success_score = ?, tools_used = ?
               WHERE session_id = ?""",
            (now, success_score, json.dumps(tools), session_id),
        )
        conn.commit()

    def get_tool_profile(self, tool_name: str) -> ToolProfile:
        """Holt das Profil eines bestimmten Tools."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT latency_ms, success, error_type FROM tool_calls WHERE tool_name = ?",
            (tool_name,),
        ).fetchall()

        if not rows:
            return ToolProfile(
                tool_name=tool_name,
                avg_latency_ms=0.0,
                p95_latency_ms=0.0,
                success_rate=0.0,
                call_count=0,
            )

        latencies = sorted(r["latency_ms"] for r in rows)
        successes = sum(1 for r in rows if r["success"])
        n = len(rows)

        # Error types
        error_types: dict[str, int] = {}
        for r in rows:
            if r["error_type"]:
                error_types[r["error_type"]] = error_types.get(r["error_type"], 0) + 1

        return ToolProfile(
            tool_name=tool_name,
            avg_latency_ms=sum(latencies) / n,
            p95_latency_ms=latencies[int(n * 0.95)] if n > 1 else latencies[-1],
            success_rate=successes / n,
            call_count=n,
            error_types=error_types,
        )

    def get_task_profile(self, category: str) -> TaskProfile:
        """Holt das Profil einer Task-Kategorie."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT success_score, start_time, end_time, tools_used
               FROM task_records WHERE category = ? AND end_time IS NOT NULL""",
            (category,),
        ).fetchall()

        if not rows:
            return TaskProfile(
                category=category,
                avg_score=0.0,
                success_rate=0.0,
                avg_duration_seconds=0.0,
                task_count=0,
            )

        scores = [r["success_score"] or 0.0 for r in rows]
        durations: list[float] = []
        all_tools: list[str] = []

        for r in rows:
            if r["start_time"] and r["end_time"]:
                start = datetime.fromisoformat(r["start_time"])
                end = datetime.fromisoformat(r["end_time"])
                durations.append((end - start).total_seconds())
            tools = json.loads(r["tools_used"])
            all_tools.extend(tools)

        # Common tools (by frequency)
        tool_counts: dict[str, int] = {}
        for t in all_tools:
            tool_counts[t] = tool_counts.get(t, 0) + 1
        common = sorted(tool_counts, key=tool_counts.get, reverse=True)[:5]

        n = len(rows)
        return TaskProfile(
            category=category,
            avg_score=sum(scores) / n,
            success_rate=sum(1 for s in scores if s >= 0.6) / n,
            avg_duration_seconds=sum(durations) / len(durations) if durations else 0.0,
            common_tools=common,
            task_count=n,
        )

    def get_capability_profile(self) -> CapabilityProfile:
        """Gesamtprofil der Faehigkeiten (Staerken/Schwaechen)."""
        conn = self._get_conn()

        # Get all tool names
        tool_names = [r[0] for r in conn.execute(
            "SELECT DISTINCT tool_name FROM tool_calls"
        ).fetchall()]

        tool_profiles = [self.get_tool_profile(t) for t in tool_names]

        # Get all categories
        categories = [r[0] for r in conn.execute(
            "SELECT DISTINCT category FROM task_records WHERE end_time IS NOT NULL"
        ).fetchall()]

        task_profiles = [self.get_task_profile(c) for c in categories]

        # Determine strengths and weaknesses
        strengths: list[str] = []
        weaknesses: list[str] = []

        for tp in tool_profiles:
            if tp.call_count >= 3:
                if tp.success_rate >= 0.8:
                    strengths.append(f"{tp.tool_name} (success: {tp.success_rate:.0%})")
                elif tp.success_rate < 0.5:
                    weaknesses.append(f"{tp.tool_name} (success: {tp.success_rate:.0%})")

        for tkp in task_profiles:
            if tkp.task_count >= 3:
                if tkp.success_rate >= 0.8:
                    strengths.append(f"Task '{tkp.category}' (score: {tkp.avg_score:.1f})")
                elif tkp.success_rate < 0.5:
                    weaknesses.append(f"Task '{tkp.category}' (score: {tkp.avg_score:.1f})")

        # Overall success rate
        total_row = conn.execute(
            "SELECT COUNT(*) as total, SUM(success) as successes FROM tool_calls"
        ).fetchone()
        overall = 0.0
        if total_row and total_row["total"] > 0:
            overall = (total_row["successes"] or 0) / total_row["total"]

        return CapabilityProfile(
            strengths=strengths,
            weaknesses=weaknesses,
            tool_profiles=tool_profiles,
            task_profiles=task_profiles,
            overall_success_rate=overall,
        )

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
