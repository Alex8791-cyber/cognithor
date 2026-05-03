"""Tactical Memory · Tier 3 -- Tool effectiveness & avoidance rules.

Tracks per-tool success/failure rates, derives avoidance rules after
consecutive failures, and exposes LLM-readable insights.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cognithor.utils.logging import get_logger

__all__ = [
    "AvoidanceRule",
    "TacticalMemory",
    "ToolEffectiveness",
    "ToolOutcome",
]

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ToolOutcome:
    """Single recorded tool invocation outcome."""

    tool_name: str
    params_hash: str
    success: bool
    duration_ms: int
    context_hash: str
    timestamp: float
    error_snippet: str | None = None
    caused_replan: bool = False


@dataclass
class ToolEffectiveness:
    """Aggregated effectiveness statistics for one tool."""

    total: int = 0
    successes: int = 0
    failures: int = 0
    avg_duration_ms: float = 0.0
    consecutive_failures: int = 0
    last_success_at: float | None = None
    last_failure_at: float | None = None
    contexts_succeeded: set[str] = field(default_factory=set)
    contexts_failed: set[str] = field(default_factory=set)

    @property
    def effectiveness(self) -> float:
        """Return success ratio, or 0.5 when no data available."""
        if self.total > 0:
            return self.successes / self.total
        return 0.5


@dataclass
class AvoidanceRule:
    """Rule that discourages using a tool in a given context."""

    tool_name: str
    reason: str
    created_at: float
    expires_at: float
    params_pattern: str | None = None
    context_pattern: str | None = None
    trigger_count: int = 3


# ---------------------------------------------------------------------------
# TacticalMemory
# ---------------------------------------------------------------------------

_EMA_ALPHA = 0.3  # Exponential moving average weight for duration


class TacticalMemory:
    """RAM-first store for tool outcomes and auto-generated avoidance rules.

    All writes are in-memory; call :meth:`flush_to_db` to persist and
    :meth:`load_from_db` to restore on restart.
    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        ttl_hours: float = 24.0,
        flush_threshold: float = 0.7,
        max_outcomes: int = 50_000,
        avoidance_consecutive_failures: int = 3,
    ) -> None:
        self._db_path = Path(db_path) if db_path is not None else None
        self._ttl_seconds = ttl_hours * 3600.0
        self._flush_threshold = flush_threshold
        self._max_outcomes = max_outcomes
        self._avoidance_threshold = avoidance_consecutive_failures

        self._outcomes: deque[ToolOutcome] = deque(maxlen=max_outcomes)
        self._effectiveness: dict[str, ToolEffectiveness] = {}
        self._avoidance_rules: list[AvoidanceRule] = []
        self._consecutive_failures: dict[str, int] = {}
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        tool: str,
        params: dict[str, Any],
        success: bool,
        duration_ms: int,
        context: str,
        error: str | None = None,
        caused_replan: bool = False,
    ) -> None:
        """Record one tool invocation and update effectiveness stats."""
        params_hash = self._hash_params(params)
        context_hash = self._hash_context(context)

        outcome = ToolOutcome(
            tool_name=tool,
            params_hash=params_hash,
            success=success,
            duration_ms=duration_ms,
            context_hash=context_hash,
            timestamp=time.time(),
            error_snippet=error,
            caused_replan=caused_replan,
        )
        self._outcomes.append(outcome)

        eff = self._effectiveness.setdefault(tool, ToolEffectiveness())
        eff.total += 1

        # Update EMA for duration
        if eff.avg_duration_ms == 0.0:
            eff.avg_duration_ms = float(duration_ms)
        else:
            eff.avg_duration_ms = _EMA_ALPHA * duration_ms + (1 - _EMA_ALPHA) * eff.avg_duration_ms

        if success:
            eff.successes += 1
            eff.last_success_at = outcome.timestamp
            eff.contexts_succeeded.add(context_hash)
            # Reset consecutive failures
            self._consecutive_failures[tool] = 0
            # Remove any active avoidance rule for this tool
            self._avoidance_rules = [r for r in self._avoidance_rules if r.tool_name != tool]
        else:
            eff.failures += 1
            eff.last_failure_at = outcome.timestamp
            eff.contexts_failed.add(context_hash)
            self._consecutive_failures[tool] = self._consecutive_failures.get(tool, 0) + 1
            count = self._consecutive_failures[tool]
            if count >= self._avoidance_threshold:
                self._maybe_add_avoidance(tool, count, error)

    def get_tool_effectiveness(self, tool: str) -> float:
        """Return effectiveness score (0-1) for *tool*, default 0.5."""
        eff = self._effectiveness.get(tool)
        if eff is None:
            return 0.5
        return eff.effectiveness

    def check_avoidance(self, tool: str, params: dict[str, Any]) -> AvoidanceRule | None:
        """Return the first non-expired avoidance rule matching *tool*."""
        now = time.time()
        for rule in self._avoidance_rules:
            if rule.tool_name != tool:
                continue
            if rule.expires_at <= now:
                continue
            return rule
        return None

    def get_insights_for_llm(self, context: str, max_chars: int = 400) -> str:
        """Build a short German summary of tool performance for the LLM."""
        if not self._effectiveness:
            return ""

        lines: list[str] = ["Taktische Einsichten:"]
        sorted_tools = sorted(self._effectiveness.items(), key=lambda kv: kv[1].total, reverse=True)
        now = time.time()

        for tool, eff in sorted_tools:
            pct = int(eff.effectiveness * 100)
            lines.append(f"  {tool}: {pct}% Erfolg ({eff.total} Aufrufe)")

        # Append avoidance warnings
        for rule in self._avoidance_rules:
            if rule.expires_at > now:
                eff = self._effectiveness.get(rule.tool_name)
                fail_count = eff.failures if eff else rule.trigger_count
                lines.append(f"  WARNUNG: {rule.tool_name} hat {fail_count}x versagt")

        result = "\n".join(lines)
        if len(result) > max_chars:
            result = result[:max_chars]
        return result

    def decay_rules(self) -> int:
        """Remove expired avoidance rules; return count removed."""
        now = time.time()
        before = len(self._avoidance_rules)
        self._avoidance_rules = [r for r in self._avoidance_rules if r.expires_at > now]
        return before - len(self._avoidance_rules)

    def flush_to_db(self) -> int:
        """Persist effectiveness + avoidance rules to SQLite; return row count."""
        if self._db_path is None:
            return 0

        conn = self._get_conn()
        self._ensure_tables(conn)

        rows_written = 0
        cur = conn.cursor()

        # Upsert effectiveness
        for tool, eff in self._effectiveness.items():
            cur.execute(
                """INSERT OR REPLACE INTO tool_effectiveness
                   (tool_name, total, successes, failures, avg_duration_ms,
                    effectiveness, last_updated)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    tool,
                    eff.total,
                    eff.successes,
                    eff.failures,
                    eff.avg_duration_ms,
                    eff.effectiveness,
                    time.time(),
                ),
            )
            rows_written += 1

        # Rebuild avoidance_rules table
        cur.execute("DELETE FROM avoidance_rules")
        for rule in self._avoidance_rules:
            cur.execute(
                """INSERT INTO avoidance_rules
                   (tool_name, params_pattern, context_pattern, reason,
                    trigger_count, created_at, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    rule.tool_name,
                    rule.params_pattern,
                    rule.context_pattern,
                    rule.reason,
                    rule.trigger_count,
                    rule.created_at,
                    rule.expires_at,
                ),
            )
            rows_written += 1

        conn.commit()
        log.debug("tactical_db_flushed", rows=rows_written)
        return rows_written

    def load_from_db(self) -> int:
        """Load effectiveness + avoidance rules from SQLite; return row count."""
        if self._db_path is None:
            return 0
        if not self._db_path.exists():
            return 0

        conn = self._get_conn()
        self._ensure_tables(conn)

        rows_loaded = 0
        cur = conn.cursor()

        try:
            cur.execute(
                "SELECT tool_name, total, successes, failures, avg_duration_ms "
                "FROM tool_effectiveness"
            )
            for row in cur.fetchall():
                tool = row[0]
                eff = ToolEffectiveness(
                    total=row[1],
                    successes=row[2],
                    failures=row[3],
                    avg_duration_ms=row[4],
                )
                self._effectiveness[tool] = eff
                rows_loaded += 1
        except sqlite3.OperationalError:
            pass

        try:
            cur.execute(
                "SELECT tool_name, params_pattern, context_pattern, reason, "
                "trigger_count, created_at, expires_at FROM avoidance_rules"
            )
            now = time.time()
            for row in cur.fetchall():
                if row[6] > now:  # only non-expired
                    rule = AvoidanceRule(
                        tool_name=row[0],
                        params_pattern=row[1],
                        context_pattern=row[2],
                        reason=row[3],
                        trigger_count=row[4],
                        created_at=row[5],
                        expires_at=row[6],
                    )
                    self._avoidance_rules.append(rule)
                    rows_loaded += 1
        except sqlite3.OperationalError:
            pass

        log.debug("tactical_db_loaded", rows=rows_loaded)
        return rows_loaded

    def close(self) -> None:
        """Flush to DB and close the connection."""
        try:
            self.flush_to_db()
        finally:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def get_stats(self) -> dict[str, Any]:
        """Return a snapshot of current memory statistics."""
        now = time.time()
        active_rules = sum(1 for r in self._avoidance_rules if r.expires_at > now)
        return {
            "outcomes_count": len(self._outcomes),
            "tools_tracked": len(self._effectiveness),
            "avoidance_rules_active": active_rules,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hash_params(self, params: dict[str, Any]) -> str:
        """MD5 of JSON-serialised params (first 12 chars)."""
        raw = json.dumps(params, sort_keys=True, default=str)
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def _hash_context(self, context: str) -> str:
        """MD5 of context string (first 12 chars)."""
        return hashlib.md5(context.encode()).hexdigest()[:12]

    def _maybe_add_avoidance(self, tool: str, count: int, error: str | None) -> None:
        """Create or refresh an avoidance rule after repeated failures."""
        # Remove existing rule for this tool to refresh it
        self._avoidance_rules = [r for r in self._avoidance_rules if r.tool_name != tool]
        reason = f"{tool} hat {count} Mal hintereinander versagt" + (
            f": {error[:80]}" if error else ""
        )
        now = time.time()
        rule = AvoidanceRule(
            tool_name=tool,
            reason=reason,
            created_at=now,
            expires_at=now + self._ttl_seconds,
            trigger_count=count,
        )
        self._avoidance_rules.append(rule)
        log.warning("avoidance_rule_created", tool=tool, consecutive_failures=count)

    def _get_conn(self) -> sqlite3.Connection:
        """Return (and lazily open) the SQLite connection."""
        if self._conn is None:
            assert self._db_path is not None
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            # Try encrypted_connect first; fall back to plain sqlite3
            try:
                from cognithor.security.encrypted_db import encrypted_connect

                self._conn = encrypted_connect(str(self._db_path))
            except Exception:
                self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        return self._conn

    @staticmethod
    def _ensure_tables(conn: sqlite3.Connection) -> None:
        """Create tables if they don't exist yet."""
        conn.execute(
            """CREATE TABLE IF NOT EXISTS tool_effectiveness (
                tool_name TEXT PRIMARY KEY,
                total INTEGER,
                successes INTEGER,
                failures INTEGER,
                avg_duration_ms REAL,
                effectiveness REAL,
                last_updated REAL
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS avoidance_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name TEXT NOT NULL,
                params_pattern TEXT,
                context_pattern TEXT,
                reason TEXT NOT NULL,
                trigger_count INTEGER,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL
            )"""
        )
        conn.commit()
