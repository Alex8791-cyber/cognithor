"""``EscalationEvent`` foundation (TRUST-8, operational-trust audit, 2026-05-04).

Reviewer-feedback gap: cognithor's selling point is "local-first", and
the gateway routes most calls to Ollama on the operator's machine.
But there are paths where a request *does* leave the machine — context
overflow falling back to a 200k-window cloud model, an explicit
owner override (``/cloud claude please summarise``), or a graceful
degradation when the local backend is down. Today there is no
**uniform contract** to log those escalations: they happen scattered
across the backend layer with different log shapes (or no log at all).

For an operator who needs to answer privacy / cost / compliance
questions, that is unacceptable. "Did this query leave the machine?"
must be a single ledger lookup, not a grep across N log files.

This module ships the **foundation**: a frozen ``EscalationEvent``
dataclass + an in-memory ``EscalationLedger`` with append-only
semantics + summary queries. Wiring this into the backend dispatch
layer (one event per cloud call, fired from
``cognithor.backends.dispatch``) is a deliberate follow-up.

**Privacy contract**: the event records *metadata only* — backend ids,
token counts, USD cost, timestamps. **No prompt content, no model
output**. The operator trusts cognithor's own audit log; this module
must not become a back-channel that re-leaks the very content
escalation was logged to track.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from cognithor.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Reason taxonomy
# ---------------------------------------------------------------------------


class EscalationReason(StrEnum):
    """Why a request was escalated from local to cloud.

    Closed set so consumers can switch on it for downstream UX
    (the Trace-UI renders different icons per reason). Adding a new
    value is an intentional review point — privacy implications.
    """

    LOCAL_BACKEND_DOWN = "local_backend_down"
    CONTEXT_TOO_LARGE = "context_too_large"
    MODEL_NOT_AVAILABLE_LOCALLY = "model_not_available_locally"
    OWNER_OVERRIDE = "owner_override"
    RATE_LIMITED_LOCAL = "rate_limited_local"
    COST_BUDGET_DECISION = "cost_budget_decision"
    QUALITY_THRESHOLD = "quality_threshold"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EscalationEvent:
    """Metadata-only record of a single local→cloud escalation.

    Frozen so the audit log can hash + embed it without copy.

    Privacy contract — these fields are intentionally sufficient for
    audit but insufficient for content-reconstruction:

    * ``from_backend`` / ``to_backend`` — backend ids only
      (``"ollama:qwen3:30b"``, ``"anthropic:claude-opus-4-7"``).
    * ``prompt_tokens`` / ``response_tokens`` — counts only.
    * ``cost_usd_micro`` — integer micro-USD (1/1_000_000 USD) so the
      ledger can sum without float drift.
    * ``owner_consented`` — boolean derived from explicit override or
      pre-approved auto-escalation policy.
    * ``run_id`` / ``request_id`` — primary keys to cross-reference
      with the TRUST-1 run-receipt and audit hash-chain. Empty when
      the call happened outside a tracked run (e.g. background
      summarisation).
    * ``notes`` — short free-text. Must NOT contain prompt or
      response content; the ledger trusts the caller to obey this
      and is documented as such.

    The shape is intentionally narrow — TRUST-1 receipts already
    carry the full per-step audit trail; this module exists to
    answer "did the request leave the machine?" in O(1).
    """

    reason: EscalationReason
    from_backend: str
    to_backend: str
    prompt_tokens: int
    response_tokens: int
    cost_usd_micro: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    owner_consented: bool = False
    run_id: str = ""
    request_id: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.from_backend:
            msg = "EscalationEvent.from_backend must be a non-empty string"
            raise ValueError(msg)
        if not self.to_backend:
            msg = "EscalationEvent.to_backend must be a non-empty string"
            raise ValueError(msg)
        if self.prompt_tokens < 0:
            msg = f"EscalationEvent.prompt_tokens must be non-negative, got {self.prompt_tokens}"
            raise ValueError(msg)
        if self.response_tokens < 0:
            msg = (
                f"EscalationEvent.response_tokens must be non-negative, got {self.response_tokens}"
            )
            raise ValueError(msg)
        if self.cost_usd_micro < 0:
            msg = f"EscalationEvent.cost_usd_micro must be non-negative, got {self.cost_usd_micro}"
            raise ValueError(msg)
        if self.completed_at is not None and self.completed_at < self.started_at:
            msg = (
                f"EscalationEvent.completed_at ({self.completed_at.isoformat()}) "
                f"must be >= started_at ({self.started_at.isoformat()})"
            )
            raise ValueError(msg)

    @property
    def cost_usd(self) -> float:
        """USD cost as a float — for display, not for accumulation.

        Use the integer ``cost_usd_micro`` for sums to avoid float
        drift; this property is for the Trace-UI cell.
        """
        return self.cost_usd_micro / 1_000_000.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.response_tokens


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EscalationSummary:
    """Aggregate over an :class:`EscalationLedger` window.

    Used by the Trace-UI cost-overview tile and the optional CLI
    ``cognithor cloud-escalations`` report.
    """

    event_count: int
    total_prompt_tokens: int
    total_response_tokens: int
    total_cost_usd_micro: int
    by_reason: dict[EscalationReason, int]
    by_destination: dict[str, int]

    @property
    def total_cost_usd(self) -> float:
        return self.total_cost_usd_micro / 1_000_000.0

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_response_tokens


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


class EscalationLedger:
    """Append-only in-memory ledger of cloud-escalation events.

    Production code uses :data:`ESCALATION_LEDGER`; tests construct
    fresh ledgers for isolation. The follow-up wiring PR fires one
    ``record()`` call per cloud completion; nothing here actually
    issues HTTP requests.
    """

    def __init__(self) -> None:
        self._events: list[EscalationEvent] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def record(self, event: EscalationEvent) -> None:
        """Append ``event`` to the ledger."""
        self._events.append(event)

    def clear(self) -> None:
        self._events.clear()

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._events)

    def events(self) -> tuple[EscalationEvent, ...]:
        """Return all events in insertion order."""
        return tuple(self._events)

    def by_reason(self, reason: EscalationReason) -> tuple[EscalationEvent, ...]:
        return tuple(e for e in self._events if e.reason == reason)

    def by_destination(self, to_backend: str) -> tuple[EscalationEvent, ...]:
        return tuple(e for e in self._events if e.to_backend == to_backend)

    def by_run(self, run_id: str) -> tuple[EscalationEvent, ...]:
        """Events tied to ``run_id`` — the TRUST-1 cross-reference query."""
        if not run_id:
            return ()
        return tuple(e for e in self._events if e.run_id == run_id)

    def in_window(self, *, start: datetime, end: datetime) -> tuple[EscalationEvent, ...]:
        """Events with ``start <= started_at <= end``."""
        if start > end:
            msg = "in_window: start must be <= end"
            raise ValueError(msg)
        return tuple(e for e in self._events if start <= e.started_at <= end)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summarise(
        self,
        events: tuple[EscalationEvent, ...] | None = None,
    ) -> EscalationSummary:
        """Aggregate ``events`` (or the full ledger if ``None``).

        Works with the result of any ``by_*`` / ``in_window`` query
        for sub-window summaries.
        """
        scope = events if events is not None else tuple(self._events)
        by_reason: dict[EscalationReason, int] = {}
        by_destination: dict[str, int] = {}
        total_prompt = 0
        total_response = 0
        total_cost = 0
        for ev in scope:
            by_reason[ev.reason] = by_reason.get(ev.reason, 0) + 1
            by_destination[ev.to_backend] = by_destination.get(ev.to_backend, 0) + 1
            total_prompt += ev.prompt_tokens
            total_response += ev.response_tokens
            total_cost += ev.cost_usd_micro
        return EscalationSummary(
            event_count=len(scope),
            total_prompt_tokens=total_prompt,
            total_response_tokens=total_response,
            total_cost_usd_micro=total_cost,
            by_reason=by_reason,
            by_destination=by_destination,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def snapshot(self) -> list[dict[str, object]]:
        """JSON-serialisable snapshot in insertion order.

        Embedded in TRUST-1 run receipts when the run had any
        escalations; the Trace-UI renders one row per event.
        """
        return [
            {
                "reason": ev.reason.value,
                "from_backend": ev.from_backend,
                "to_backend": ev.to_backend,
                "prompt_tokens": ev.prompt_tokens,
                "response_tokens": ev.response_tokens,
                "cost_usd_micro": ev.cost_usd_micro,
                "cost_usd": round(ev.cost_usd, 6),
                "started_at": ev.started_at.isoformat(),
                "completed_at": (
                    ev.completed_at.isoformat() if ev.completed_at is not None else None
                ),
                "owner_consented": ev.owner_consented,
                "run_id": ev.run_id,
                "request_id": ev.request_id,
                "notes": ev.notes,
            }
            for ev in self._events
        ]


# ---------------------------------------------------------------------------
# Process-local default
# ---------------------------------------------------------------------------

# Backend dispatch wires into this instance via a follow-up PR; the
# ledger stays empty in production until that wiring lands. Tests
# construct their own :class:`EscalationLedger` for isolation.
ESCALATION_LEDGER: EscalationLedger = EscalationLedger()


def _record_escalation_ledger_migration() -> None:
    """TRUST-10 self-audit: announce the escalation-ledger schema."""
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
                domain=MigrationDomain.ESCALATION_LEDGER,
                source_version="v0-no-ledger",
                target_version="v1-metadata-only-ledger",
                status=MigrationStatus.APPLIED,
                applied_by="system",
                item_count=-1,
                migration_id="escalation_ledger:v0-no-ledger:v1-metadata-only-ledger",
                notes=(
                    "TRUST-8 EscalationLedger schema active "
                    "(metadata-only privacy contract, EscalationReason enum)"
                ),
            )
        )


_record_escalation_ledger_migration()
