"""``CostEntry`` foundation (TRUST-6, operational-trust audit, 2026-05-04).

Reviewer-feedback gap: cognithor already has a ``DomainCostTracker``
(Sprint-26.1) for PSE domains and an ad-hoc cost field on the
``EscalationEvent`` (TRUST-8). What is missing is a **uniform
cross-cutting ledger** that lets the operator answer one budget
question — "how much have I spent today, broken down by tool /
domain / channel / run?" — without grepping three subsystems.

This module ships the **foundation**: a frozen ``CostEntry``
dataclass + an in-memory ``CostLedger`` with multi-axis aggregation
and budget alerting. Wiring this into the existing PSE cost tracker
+ TRUST-8 escalation log + every other tool that bills tokens is a
deliberate follow-up; this layer stays storage-free.

Design choices:

* **Integer micro-USD** (``cost_usd_micro = round(usd * 1_000_000)``)
  is the canonical cost unit. Floats accumulate drift; integers
  don't. The display property ``cost_usd`` divides on read.
* **Optional vs required axes**: ``tool``, ``backend``,
  ``run_id``, ``channel``, ``domain`` — every entry has at least
  ``tool`` set; the rest are best-effort. Aggregation falls back to
  ``"_unknown"`` for missing values so the histogram never silently
  drops rows.
* **Budget alerts** are passive: the ledger doesn't enforce, it
  just answers ``budget_status(limit)`` so the caller can decide
  whether to escalate, throttle, or alert.
* **No prompt content stored** — same privacy invariant as TRUST-8.
  Notes are advisory free-text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from cognithor.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Source typing
# ---------------------------------------------------------------------------


class CostKind(StrEnum):
    """What the cost paid for.

    Closed set so the Trace-UI can render distinct icons. Each kind
    decides how an entry is grouped in summary tiles.
    """

    LLM_INFERENCE = "llm_inference"  # tokens billed by an LLM provider
    EMBEDDING = "embedding"  # vector embedding API
    TOOL_API = "tool_api"  # external tool API (search, scraping, …)
    STORAGE = "storage"  # persisted bytes-per-month
    NETWORK = "network"  # bandwidth / egress
    OTHER = "other"


_UNKNOWN = "_unknown"


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostEntry:
    """Immutable single billable event.

    Frozen so the audit log can hash + embed it without copy.

    Attributes
    ----------
    kind:
        :class:`CostKind`.
    tool:
        Stable tool / capability name. ``"web_fetch"``, ``"qwen3:30b"``,
        ``"openai:embed-3"``. Non-empty.
    cost_usd_micro:
        Integer micro-USD. ``round(usd * 1_000_000)`` at the call
        site to avoid float drift in the ledger sum. Non-negative.
    backend:
        Optional backend id — ``"ollama"``, ``"anthropic"``, ``"local"``.
        Empty → aggregations bucket under ``_unknown``.
    run_id:
        Optional TRUST-1 run-receipt id. Empty for background work
        without a tracked run.
    channel:
        Optional channel id — ``"telegram"``, ``"cli"``, ``"webui"``.
    domain:
        Optional cross-axis: PSE domain (``"sql"``, ``"json"``, …),
        memory tier, etc. Empty when the cost isn't domain-scoped.
    prompt_tokens / response_tokens / unit_count:
        Best-effort counts. -1 when the cost isn't token- or
        unit-bearing (e.g. monthly storage charge).
    occurred_at:
        UTC timestamp the cost was incurred.
    notes:
        Short free-text. MUST NOT contain prompt or response
        content; the module trusts the caller to obey.
    """

    kind: CostKind
    tool: str
    cost_usd_micro: int
    backend: str = ""
    run_id: str = ""
    channel: str = ""
    domain: str = ""
    prompt_tokens: int = -1
    response_tokens: int = -1
    unit_count: int = -1
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.tool:
            msg = "CostEntry.tool must be a non-empty string"
            raise ValueError(msg)
        if self.cost_usd_micro < 0:
            msg = f"CostEntry.cost_usd_micro must be non-negative, got {self.cost_usd_micro}"
            raise ValueError(msg)
        for label, value in (
            ("prompt_tokens", self.prompt_tokens),
            ("response_tokens", self.response_tokens),
            ("unit_count", self.unit_count),
        ):
            if value < -1:
                msg = f"CostEntry.{label} must be -1 (unknown) or non-negative, got {value}"
                raise ValueError(msg)

    @property
    def cost_usd(self) -> float:
        """USD as float — for display only; use ``cost_usd_micro`` for sums."""
        return self.cost_usd_micro / 1_000_000.0


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostSummary:
    """Aggregated view over an :class:`CostLedger` window.

    All histograms map a string key (with ``_unknown`` for missing
    axes) to an integer micro-USD total.
    """

    entry_count: int
    total_cost_usd_micro: int
    by_kind: dict[CostKind, int]
    by_tool: dict[str, int]
    by_backend: dict[str, int]
    by_channel: dict[str, int]
    by_domain: dict[str, int]
    by_run: dict[str, int]

    @property
    def total_cost_usd(self) -> float:
        return self.total_cost_usd_micro / 1_000_000.0

    def top_n(
        self,
        axis: str,
        *,
        n: int = 5,
    ) -> list[tuple[str, int]]:
        """Return the top-N keys for ``axis`` sorted by cost descending.

        ``axis`` is one of ``"tool"``, ``"backend"``, ``"channel"``,
        ``"domain"``, ``"run"``. ``"kind"`` is supported separately
        because the keys are :class:`CostKind` values, not strings.
        """
        bucket = self._axis_bucket(axis)
        if n < 1:
            return []
        return sorted(bucket.items(), key=lambda kv: (-kv[1], kv[0]))[:n]

    def _axis_bucket(self, axis: str) -> dict[str, int]:
        # Support the str-keyed axes; ``kind`` has its own helper below.
        mapping: dict[str, dict[str, int]] = {
            "tool": self.by_tool,
            "backend": self.by_backend,
            "channel": self.by_channel,
            "domain": self.by_domain,
            "run": self.by_run,
        }
        if axis not in mapping:
            valid = sorted(mapping)
            msg = f"top_n: axis must be one of {valid}, got {axis!r}"
            raise ValueError(msg)
        return mapping[axis]


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


class BudgetStatus(StrEnum):
    """Outcome of comparing total cost against a budget."""

    UNDER = "under"
    APPROACHING = "approaching"  # >= 80 % of budget
    EXCEEDED = "exceeded"


@dataclass(frozen=True)
class BudgetReport:
    """Snapshot of a single budget check."""

    limit_usd_micro: int
    spent_usd_micro: int
    status: BudgetStatus
    remaining_usd_micro: int  # may be negative when EXCEEDED

    @property
    def utilisation(self) -> float:
        """Fraction of the budget spent (>1.0 when exceeded). 0 when limit=0."""
        if self.limit_usd_micro == 0:
            return 0.0
        return self.spent_usd_micro / self.limit_usd_micro


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


class CostLedger:
    """Append-only in-memory ledger of cost entries.

    Production code uses :data:`COST_LEDGER`; tests construct fresh
    ledgers for isolation.
    """

    def __init__(self) -> None:
        self._entries: list[CostEntry] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def record(self, entry: CostEntry) -> None:
        self._entries.append(entry)

    def clear(self) -> None:
        self._entries.clear()

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def entries(self) -> tuple[CostEntry, ...]:
        return tuple(self._entries)

    def by_run(self, run_id: str) -> tuple[CostEntry, ...]:
        if not run_id:
            return ()
        return tuple(e for e in self._entries if e.run_id == run_id)

    def by_tool(self, tool: str) -> tuple[CostEntry, ...]:
        if not tool:
            return ()
        return tuple(e for e in self._entries if e.tool == tool)

    def by_kind(self, kind: CostKind) -> tuple[CostEntry, ...]:
        return tuple(e for e in self._entries if e.kind == kind)

    def in_window(self, *, start: datetime, end: datetime) -> tuple[CostEntry, ...]:
        if start > end:
            msg = "in_window: start must be <= end"
            raise ValueError(msg)
        return tuple(e for e in self._entries if start <= e.occurred_at <= end)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def summarise(
        self,
        entries: tuple[CostEntry, ...] | None = None,
    ) -> CostSummary:
        scope = entries if entries is not None else tuple(self._entries)
        by_kind: dict[CostKind, int] = {}
        by_tool: dict[str, int] = {}
        by_backend: dict[str, int] = {}
        by_channel: dict[str, int] = {}
        by_domain: dict[str, int] = {}
        by_run: dict[str, int] = {}
        total = 0
        for e in scope:
            by_kind[e.kind] = by_kind.get(e.kind, 0) + e.cost_usd_micro
            by_tool[e.tool] = by_tool.get(e.tool, 0) + e.cost_usd_micro
            backend = e.backend or _UNKNOWN
            by_backend[backend] = by_backend.get(backend, 0) + e.cost_usd_micro
            channel = e.channel or _UNKNOWN
            by_channel[channel] = by_channel.get(channel, 0) + e.cost_usd_micro
            domain = e.domain or _UNKNOWN
            by_domain[domain] = by_domain.get(domain, 0) + e.cost_usd_micro
            run = e.run_id or _UNKNOWN
            by_run[run] = by_run.get(run, 0) + e.cost_usd_micro
            total += e.cost_usd_micro
        return CostSummary(
            entry_count=len(scope),
            total_cost_usd_micro=total,
            by_kind=by_kind,
            by_tool=by_tool,
            by_backend=by_backend,
            by_channel=by_channel,
            by_domain=by_domain,
            by_run=by_run,
        )

    # ------------------------------------------------------------------
    # Budget
    # ------------------------------------------------------------------

    def budget_status(
        self,
        *,
        limit_usd_micro: int,
        scope: tuple[CostEntry, ...] | None = None,
        approaching_threshold: float = 0.80,
    ) -> BudgetReport:
        """Compare cumulative cost against a budget.

        ``approaching_threshold`` (0..1) decides when the report
        flips from ``UNDER`` to ``APPROACHING``.
        """
        if limit_usd_micro < 0:
            msg = f"budget_status: limit_usd_micro must be non-negative, got {limit_usd_micro}"
            raise ValueError(msg)
        if not 0.0 < approaching_threshold < 1.0:
            msg = (
                "budget_status: approaching_threshold must be in (0, 1), "
                f"got {approaching_threshold}"
            )
            raise ValueError(msg)
        summary = self.summarise(scope)
        spent = summary.total_cost_usd_micro
        if limit_usd_micro == 0:
            # No budget configured — anything spent is EXCEEDED.
            status = BudgetStatus.EXCEEDED if spent > 0 else BudgetStatus.UNDER
        elif spent > limit_usd_micro:
            status = BudgetStatus.EXCEEDED
        elif spent >= int(limit_usd_micro * approaching_threshold):
            status = BudgetStatus.APPROACHING
        else:
            status = BudgetStatus.UNDER
        return BudgetReport(
            limit_usd_micro=limit_usd_micro,
            spent_usd_micro=spent,
            status=status,
            remaining_usd_micro=limit_usd_micro - spent,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def snapshot(self) -> list[dict[str, object]]:
        """JSON-serialisable insertion-order snapshot."""
        return [
            {
                "kind": e.kind.value,
                "tool": e.tool,
                "cost_usd_micro": e.cost_usd_micro,
                "cost_usd": round(e.cost_usd, 6),
                "backend": e.backend,
                "run_id": e.run_id,
                "channel": e.channel,
                "domain": e.domain,
                "prompt_tokens": e.prompt_tokens,
                "response_tokens": e.response_tokens,
                "unit_count": e.unit_count,
                "occurred_at": e.occurred_at.isoformat(),
                "notes": e.notes,
            }
            for e in self._entries
        ]


# ---------------------------------------------------------------------------
# Process-local default
# ---------------------------------------------------------------------------

# Production-wiring lands in a follow-up PR (PSE cost tracker,
# TRUST-8 escalation, every billed tool). Tests use fresh ledgers.
COST_LEDGER: CostLedger = CostLedger()


def _record_cost_ledger_migration() -> None:
    """TRUST-10 self-audit: announce the cost-ledger schema."""
    from contextlib import suppress

    from cognithor.security.migration_ledger import (
        MIGRATION_LEDGER,
        MigrationChainError,
        MigrationDomain,
        MigrationStatus,
        MigrationStep,
    )

    with suppress(MigrationChainError, ValueError):
        MIGRATION_LEDGER.record(
            MigrationStep(
                domain=MigrationDomain.COST_LEDGER,
                source_version="v0-no-ledger",
                target_version="v1-micro-usd-ledger",
                status=MigrationStatus.APPLIED,
                applied_by="system",
                item_count=-1,
                migration_id="cost_ledger:v0-no-ledger:v1-micro-usd-ledger",
                notes=(
                    "TRUST-6 CostLedger schema active "
                    "(integer micro-USD cost unit, CostKind enum, "
                    "BudgetReport status taxonomy)"
                ),
            )
        )


_record_cost_ledger_migration()
