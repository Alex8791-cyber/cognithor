"""Cost tracking for LLM API calls.

Tracks costs per session and globally with SQLite persistence.
Budget enforcement with daily and monthly limits.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from cognithor.db import SQLITE_BUSY_TIMEOUT_MS
from cognithor.models import AgentBudgetStatus, BudgetStatus, CostRecord, CostReport
from cognithor.security.encrypted_db import encrypted_connect
from cognithor.utils.logging import get_logger

log = get_logger(__name__)


class CostTracker:
    """Tracks LLM API costs per session and globally."""

    # Pricing table (USD per 1M tokens, as of 2025)
    DEFAULT_PRICING: dict[str, dict[str, float]] = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
        "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
        "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
        "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
        "claude-3-opus": {"input": 15.00, "output": 75.00},
    }

    # Fallback price for unknown models
    FALLBACK_PRICING: dict[str, float] = {"input": 5.00, "output": 15.00}

    def __init__(
        self,
        db_path: str,
        daily_budget: float = 0.0,
        monthly_budget: float = 0.0,
    ) -> None:
        self._db_path = db_path
        self._daily_budget = daily_budget
        self._monthly_budget = monthly_budget
        self._conn = encrypted_connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
        # Crew-Layer (Task 11) reads the most recent record via ``last_call()``
        # to surface per-task token usage without an extra DB roundtrip.
        self._last_record: CostRecord | None = None
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_costs (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                session_id TEXT,
                model TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                cost_usd REAL NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_costs_timestamp ON llm_costs(timestamp)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_costs_session ON llm_costs(session_id)
        """)
        # Migration: add agent_name column if missing
        try:
            self._conn.execute("ALTER TABLE llm_costs ADD COLUMN agent_name TEXT DEFAULT ''")
            self._conn.commit()
        except Exception:
            pass  # Column already exists (catches both sqlite3 and sqlcipher3)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_costs_agent ON llm_costs(agent_name)
        """)
        self._conn.commit()

    def _get_pricing(self, model: str) -> dict[str, float]:
        """Finds price for a model. Ollama models are free."""
        # Ollama / local models
        if "/" not in model and ":" in model:
            return {"input": 0.0, "output": 0.0}
        # Exact match
        if model in self.DEFAULT_PRICING:
            return self.DEFAULT_PRICING[model]
        # Prefix match (e.g. "gpt-4o-2024-11-20" -> "gpt-4o")
        for known_model, pricing in self.DEFAULT_PRICING.items():
            if model.startswith(known_model):
                return pricing
        return self.FALLBACK_PRICING

    def record_llm_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        session_id: str = "",
        agent_name: str = "",
    ) -> CostRecord:
        """Records an LLM call and calculates costs."""
        pricing = self._get_pricing(model)
        cost = (
            input_tokens * pricing["input"] / 1_000_000
            + output_tokens * pricing["output"] / 1_000_000
        )

        record = CostRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            session_id=session_id,
            agent_name=agent_name,
        )

        self._conn.execute(
            "INSERT INTO llm_costs "
            "(id, timestamp, session_id, model, "
            "input_tokens, output_tokens, cost_usd, agent_name) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.id,
                record.timestamp.isoformat(),
                record.session_id,
                record.model,
                record.input_tokens,
                record.output_tokens,
                record.cost_usd,
                record.agent_name,
            ),
        )
        self._conn.commit()
        self._last_record = record
        return record

    def last_call(self) -> CostRecord | None:
        """Return the most recently recorded ``CostRecord`` (or None).

        Additive helper for the Crew-Layer (Task 11): once the Planner
        completes ``formulate_response``, the Crew-Layer reads this to
        populate ``TaskOutput.token_usage`` without an extra DB query.
        """
        return self._last_record

    def get_session_cost(self, session_id: str) -> float:
        """Total costs of a session."""
        row = self._conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM llm_costs WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return float(row[0]) if row else 0.0

    def get_daily_cost(self, day: date | None = None) -> float:
        """Daily costs."""
        day = day or datetime.now(UTC).date()
        day_str = day.isoformat()
        row = self._conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM llm_costs WHERE timestamp LIKE ?",
            (f"{day_str}%",),
        ).fetchone()
        return float(row[0]) if row else 0.0

    def get_monthly_cost(self, year: int, month: int) -> float:
        """Monthly costs."""
        prefix = f"{year:04d}-{month:02d}"
        row = self._conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM llm_costs WHERE timestamp LIKE ?",
            (f"{prefix}%",),
        ).fetchone()
        return float(row[0]) if row else 0.0

    def check_budget(self) -> BudgetStatus:
        """Checks if budget is exceeded."""
        today = datetime.now(UTC).date()
        daily_cost = self.get_daily_cost(today)
        monthly_cost = self.get_monthly_cost(today.year, today.month)

        daily_remaining = -1.0
        monthly_remaining = -1.0
        warnings: list[str] = []

        if self._daily_budget > 0:
            daily_remaining = self._daily_budget - daily_cost
            if daily_remaining <= 0:
                warnings.append(f"Tageslimit ({self._daily_budget:.2f} USD) erreicht")
            elif daily_remaining < self._daily_budget * 0.2:
                warnings.append(f"Tageslimit fast erreicht ({daily_remaining:.2f} USD uebrig)")

        if self._monthly_budget > 0:
            monthly_remaining = self._monthly_budget - monthly_cost
            if monthly_remaining <= 0:
                warnings.append(f"Monatslimit ({self._monthly_budget:.2f} USD) erreicht")
            elif monthly_remaining < self._monthly_budget * 0.2:
                warnings.append(f"Monatslimit fast erreicht ({monthly_remaining:.2f} USD uebrig)")

        ok = True
        if self._daily_budget > 0 and daily_remaining <= 0:
            ok = False
        if self._monthly_budget > 0 and monthly_remaining <= 0:
            ok = False

        return BudgetStatus(
            ok=ok,
            daily_remaining=daily_remaining,
            monthly_remaining=monthly_remaining,
            warning="; ".join(warnings),
        )

    def get_budget_info(self) -> dict[str, float] | None:
        """Budget info for GovernanceAgent: {limit, used}.

        Returns None if no budget limit is configured.
        """
        # Effective limit: prefer monthly limit, then daily limit
        limit = self._monthly_budget if self._monthly_budget > 0 else self._daily_budget
        if limit <= 0:
            return None

        if self._monthly_budget > 0:
            today = datetime.now(UTC).date()
            used = self.get_monthly_cost(today.year, today.month)
        else:
            used = self.get_daily_cost()

        return {"limit": limit, "used": used}

    def get_agent_costs(self, days: int = 1) -> dict[str, float]:
        """Get cost breakdown by agent for the last N days."""
        from datetime import timedelta

        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        rows = self._conn.execute(
            "SELECT agent_name, COALESCE(SUM(cost_usd), 0) "
            "FROM llm_costs WHERE timestamp >= ? "
            "GROUP BY agent_name ORDER BY SUM(cost_usd) DESC",
            (cutoff,),
        ).fetchall()
        return {name or "(unknown)": cost for name, cost in rows}

    def check_agent_budget(self, agent_name: str, daily_limit: float = 0.0) -> AgentBudgetStatus:
        """Check budget for a specific agent."""
        from cognithor.models import AgentBudgetStatus

        today = datetime.now(UTC).date().isoformat()
        row = self._conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM llm_costs "
            "WHERE agent_name = ? AND timestamp LIKE ?",
            (agent_name, f"{today}%"),
        ).fetchone()
        daily_cost = float(row[0]) if row else 0.0

        ok = True
        warning = ""
        if daily_limit > 0:
            if daily_cost >= daily_limit:
                ok = False
                warning = f"Agent '{agent_name}' Tageslimit ({daily_limit:.2f} USD) erreicht"
            elif daily_cost >= daily_limit * 0.8:
                warning = (
                    f"Agent '{agent_name}' Tageslimit fast erreicht "
                    f"({daily_limit - daily_cost:.2f} USD uebrig)"
                )

        return AgentBudgetStatus(
            agent_name=agent_name,
            daily_cost_usd=daily_cost,
            daily_limit_usd=daily_limit,
            ok=ok,
            warning=warning,
        )

    def get_cost_report(self, days: int = 30) -> CostReport:
        """Aggregated cost report over the last N days."""
        from datetime import timedelta

        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()

        rows = self._conn.execute(
            "SELECT model, cost_usd, timestamp FROM llm_costs "
            "WHERE timestamp >= ? ORDER BY timestamp DESC",
            (cutoff,),
        ).fetchall()

        total_cost = 0.0
        cost_by_model: dict[str, float] = {}
        cost_by_day: dict[str, float] = {}

        for model, cost, ts in rows:
            total_cost += cost
            cost_by_model[model] = cost_by_model.get(model, 0.0) + cost
            day_key = ts[:10]
            cost_by_day[day_key] = cost_by_day.get(day_key, 0.0) + cost

        total_calls = len(rows)
        avg = total_cost / total_calls if total_calls > 0 else 0.0

        agent_rows = self._conn.execute(
            "SELECT agent_name, COALESCE(SUM(cost_usd), 0) "
            "FROM llm_costs WHERE timestamp >= ? "
            "GROUP BY agent_name",
            (cutoff,),
        ).fetchall()
        cost_by_agent = {name or "(unknown)": cost for name, cost in agent_rows}

        return CostReport(
            total_cost_usd=total_cost,
            total_calls=total_calls,
            cost_by_model=cost_by_model,
            cost_by_day=cost_by_day,
            cost_by_agent=cost_by_agent,
            avg_cost_per_call=avg,
        )

    def close(self) -> None:
        """Closes the database connection."""
        self._conn.close()
