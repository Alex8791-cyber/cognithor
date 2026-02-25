from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from jarvis.models import PolicyChange, PolicyProposal
from jarvis.utils.logging import get_logger

logger = get_logger(__name__)


class GovernanceAgent:
    """Analyzes telemetry data and proposes governance policy changes."""

    def __init__(
        self,
        task_telemetry: Any = None,
        error_clusterer: Any = None,
        task_profiler: Any = None,
        cost_tracker: Any = None,
        run_recorder: Any = None,
        db_path: str = "governance.db",
    ) -> None:
        self.task_telemetry = task_telemetry
        self.error_clusterer = error_clusterer
        self.task_profiler = task_profiler
        self.cost_tracker = cost_tracker
        self.run_recorder = run_recorder
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._init_schema()
        logger.info("GovernanceAgent initialized with db_path=%s", db_path)

    def _init_schema(self) -> None:
        """Create the proposals table if it does not exist."""
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS proposals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                category TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                evidence TEXT NOT NULL,
                suggested_change_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                decision_reason TEXT
            )
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Proposal helpers
    # ------------------------------------------------------------------

    def _create_proposal(
        self,
        category: str,
        title: str,
        description: str,
        evidence: dict[str, Any],
        suggested_change: dict[str, Any],
    ) -> int:
        """Insert a new proposal and return its id.

        Deduplicates: if a pending proposal with the same category and title
        already exists, its id is returned instead of creating a duplicate.
        """
        existing = self._conn.execute(
            "SELECT id FROM proposals WHERE category = ? AND title = ? AND status = 'pending'",
            (category, title),
        ).fetchone()
        if existing:
            return existing[0]

        now = datetime.now(timezone.utc).isoformat()
        cursor = self._conn.execute(
            """
            INSERT INTO proposals (timestamp, category, title, description,
                                   evidence, suggested_change_json, status)
            VALUES (?, ?, ?, ?, ?, ?, 'pending')
            """,
            (
                now,
                category,
                title,
                description,
                json.dumps(evidence),
                json.dumps(suggested_change),
            ),
        )
        self._conn.commit()
        proposal_id = cursor.lastrowid
        logger.info("Created proposal #%d: %s", proposal_id, title)
        return proposal_id

    def _row_to_proposal(self, row: sqlite3.Row) -> PolicyProposal:
        """Convert a database row to a PolicyProposal model."""
        ts = row["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return PolicyProposal(
            id=row["id"],
            timestamp=ts,
            category=row["category"],
            title=row["title"],
            description=row["description"],
            evidence=json.loads(row["evidence"]),
            suggested_change=json.loads(row["suggested_change_json"]),
            status=row["status"],
            decision_reason=row["decision_reason"] or "",
        )

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze(self) -> list[PolicyProposal]:
        """Run all governance checks and return newly created proposals."""
        created: list[PolicyProposal] = []

        created.extend(self._check_error_rate())
        created.extend(self._check_budget())
        created.extend(self._check_recurring_error_clusters())
        created.extend(self._check_tool_latency())
        created.extend(self._check_unused_tools())

        logger.info("Analysis complete -- %d new proposal(s) created", len(created))
        return created

    def _check_error_rate(self) -> list[PolicyProposal]:
        """Error-rate > 30% for a tool -> propose timeout increase or disable."""
        proposals: list[PolicyProposal] = []
        if self.task_telemetry is None:
            return proposals

        try:
            tool_stats = getattr(self.task_telemetry, "get_tool_stats", lambda: {})()
            for tool_name, stats in tool_stats.items():
                total = stats.get("total", 0)
                errors = stats.get("errors", 0)
                if total == 0:
                    continue
                error_rate = errors / total
                if error_rate > 0.30:
                    evidence = {
                        "tool": tool_name,
                        "total_calls": total,
                        "error_count": errors,
                        "error_rate": round(error_rate, 4),
                    }
                    suggested_change = {
                        "action": "increase_timeout_or_disable",
                        "tool": tool_name,
                        "reason": f"Error rate {error_rate:.1%} exceeds 30% threshold",
                    }
                    pid = self._create_proposal(
                        category="error_rate",
                        title=f"High error rate for tool '{tool_name}'",
                        description=(
                            f"Tool '{tool_name}' has an error rate of {error_rate:.1%} "
                            f"({errors}/{total} calls). Consider increasing its timeout "
                            f"or disabling it."
                        ),
                        evidence=evidence,
                        suggested_change=suggested_change,
                    )
                    row = self._conn.execute(
                        "SELECT * FROM proposals WHERE id = ?", (pid,)
                    ).fetchone()
                    proposals.append(self._row_to_proposal(row))
        except Exception:
            logger.exception("Error checking tool error rates")

        return proposals

    def _check_budget(self) -> list[PolicyProposal]:
        """Budget > 80% consumed -> propose cheaper model."""
        proposals: list[PolicyProposal] = []
        if self.cost_tracker is None:
            return proposals

        try:
            budget_info = getattr(self.cost_tracker, "get_budget_info", lambda: None)()
            if budget_info is None:
                return proposals

            budget_limit = budget_info.get("limit", 0)
            budget_used = budget_info.get("used", 0)
            if budget_limit <= 0:
                return proposals

            usage_pct = budget_used / budget_limit
            if usage_pct > 0.80:
                evidence = {
                    "budget_limit": budget_limit,
                    "budget_used": round(budget_used, 4),
                    "usage_percentage": round(usage_pct * 100, 2),
                }
                suggested_change = {
                    "action": "switch_to_cheaper_model",
                    "reason": f"Budget usage at {usage_pct:.1%}, exceeds 80% threshold",
                }
                pid = self._create_proposal(
                    category="budget",
                    title="Budget usage exceeds 80%",
                    description=(
                        f"Current budget usage is {usage_pct:.1%} "
                        f"(${budget_used:.2f} / ${budget_limit:.2f}). "
                        f"Consider switching to a cheaper model to stay within budget."
                    ),
                    evidence=evidence,
                    suggested_change=suggested_change,
                )
                row = self._conn.execute(
                    "SELECT * FROM proposals WHERE id = ?", (pid,)
                ).fetchone()
                proposals.append(self._row_to_proposal(row))
        except Exception:
            logger.exception("Error checking budget")

        return proposals

    def _check_recurring_error_clusters(self) -> list[PolicyProposal]:
        """Recurring error cluster (>5x) -> propose policy rule."""
        proposals: list[PolicyProposal] = []
        if self.error_clusterer is None:
            return proposals

        try:
            clusters = getattr(self.error_clusterer, "get_clusters", lambda: [])()
            for cluster in clusters:
                count = cluster.get("count", 0)
                if count > 5:
                    cluster_id = cluster.get("id", "unknown")
                    pattern = cluster.get("pattern", "unknown")
                    evidence = {
                        "cluster_id": cluster_id,
                        "pattern": pattern,
                        "occurrence_count": count,
                    }
                    suggested_change = {
                        "action": "add_policy_rule",
                        "cluster_id": cluster_id,
                        "pattern": pattern,
                        "reason": f"Error cluster seen {count} times (>5 threshold)",
                    }
                    pid = self._create_proposal(
                        category="recurring_error",
                        title=f"Recurring error cluster: {cluster_id}",
                        description=(
                            f"Error cluster '{cluster_id}' with pattern '{pattern}' "
                            f"has occurred {count} times. Consider adding a policy rule "
                            f"to handle this automatically."
                        ),
                        evidence=evidence,
                        suggested_change=suggested_change,
                    )
                    row = self._conn.execute(
                        "SELECT * FROM proposals WHERE id = ?", (pid,)
                    ).fetchone()
                    proposals.append(self._row_to_proposal(row))
        except Exception:
            logger.exception("Error checking recurring error clusters")

        return proposals

    def _check_tool_latency(self) -> list[PolicyProposal]:
        """Tool latency p95 > 10s -> propose timeout adjustment."""
        proposals: list[PolicyProposal] = []
        if self.task_profiler is None:
            return proposals

        try:
            latency_stats = getattr(
                self.task_profiler, "get_latency_stats", lambda: {}
            )()
            for tool_name, stats in latency_stats.items():
                p95 = stats.get("p95", 0)
                if p95 > 10.0:
                    evidence = {
                        "tool": tool_name,
                        "p95_latency_seconds": round(p95, 3),
                        "threshold_seconds": 10.0,
                    }
                    suggested_change = {
                        "action": "adjust_timeout",
                        "tool": tool_name,
                        "suggested_timeout": round(p95 * 1.5, 1),
                        "reason": f"p95 latency {p95:.1f}s exceeds 10s threshold",
                    }
                    pid = self._create_proposal(
                        category="tool_latency",
                        title=f"High p95 latency for tool '{tool_name}'",
                        description=(
                            f"Tool '{tool_name}' has a p95 latency of {p95:.1f}s, "
                            f"exceeding the 10s threshold. Consider adjusting its "
                            f"timeout to {p95 * 1.5:.1f}s."
                        ),
                        evidence=evidence,
                        suggested_change=suggested_change,
                    )
                    row = self._conn.execute(
                        "SELECT * FROM proposals WHERE id = ?", (pid,)
                    ).fetchone()
                    proposals.append(self._row_to_proposal(row))
        except Exception:
            logger.exception("Error checking tool latency")

        return proposals

    def _check_unused_tools(self) -> list[PolicyProposal]:
        """Unused tools (0 calls in 7 days) -> propose removal from allowed_tools."""
        proposals: list[PolicyProposal] = []
        if self.task_telemetry is None:
            return proposals

        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=7)
            unused_tools = getattr(
                self.task_telemetry,
                "get_unused_tools",
                lambda since: [],
            )(cutoff)

            for tool_name in unused_tools:
                evidence = {
                    "tool": tool_name,
                    "days_inactive": 7,
                    "since": cutoff.isoformat(),
                }
                suggested_change = {
                    "action": "remove_from_allowed_tools",
                    "tool": tool_name,
                    "reason": f"No calls in the last 7 days",
                }
                pid = self._create_proposal(
                    category="unused_tool",
                    title=f"Unused tool: '{tool_name}'",
                    description=(
                        f"Tool '{tool_name}' has had 0 calls in the last 7 days. "
                        f"Consider removing it from allowed_tools to reduce attack "
                        f"surface and configuration complexity."
                    ),
                    evidence=evidence,
                    suggested_change=suggested_change,
                )
                row = self._conn.execute(
                    "SELECT * FROM proposals WHERE id = ?", (pid,)
                ).fetchone()
                proposals.append(self._row_to_proposal(row))
        except Exception:
            logger.exception("Error checking unused tools")

        return proposals

    # ------------------------------------------------------------------
    # Proposal management
    # ------------------------------------------------------------------

    def get_pending_proposals(self) -> list[PolicyProposal]:
        """Return all proposals with status 'pending'."""
        rows = self._conn.execute(
            "SELECT * FROM proposals WHERE status = 'pending' ORDER BY id"
        ).fetchall()
        return [self._row_to_proposal(row) for row in rows]

    def approve_proposal(self, proposal_id: int) -> PolicyChange:
        """Approve a proposal and return a PolicyChange to be applied."""
        row = self._conn.execute(
            "SELECT * FROM proposals WHERE id = ?", (proposal_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Proposal {proposal_id} not found")

        self._conn.execute(
            "UPDATE proposals SET status = 'approved', decision_reason = 'Approved' "
            "WHERE id = ?",
            (proposal_id,),
        )
        self._conn.commit()
        logger.info("Proposal #%d approved", proposal_id)

        suggested_change = json.loads(row["suggested_change_json"])
        return PolicyChange(
            proposal_id=proposal_id,
            category=row["category"],
            title=row["title"],
            change=suggested_change,
            timestamp=datetime.now(timezone.utc),
        )

    def reject_proposal(self, proposal_id: int, reason: str) -> None:
        """Reject a proposal with a given reason."""
        row = self._conn.execute(
            "SELECT * FROM proposals WHERE id = ?", (proposal_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Proposal {proposal_id} not found")

        self._conn.execute(
            "UPDATE proposals SET status = 'rejected', decision_reason = ? "
            "WHERE id = ?",
            (reason, proposal_id),
        )
        self._conn.commit()
        logger.info("Proposal #%d rejected: %s", proposal_id, reason)

    def get_proposal_history(self, limit: int = 50) -> list[PolicyProposal]:
        """Return recent proposals regardless of status, ordered newest first."""
        rows = self._conn.execute(
            "SELECT * FROM proposals ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_proposal(row) for row in rows]

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            logger.info("GovernanceAgent database connection closed")
