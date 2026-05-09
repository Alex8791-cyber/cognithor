"""``BackendDispatchEvent`` foundation (TRUST-8 backend-dispatch tracking, 2026-05-09).

Companion to :mod:`cognithor.security.cloud_escalation`. Where
``cloud_escalation`` answers *"did the request leave the machine"* in
O(1), this module answers the broader *"which backend actually
served this completion, and how long did it take"* — for **every**
LLM call (local + cloud). The two are deliberately separate:

* ``cloud_escalation`` is a **privacy / cost** surface — every entry
  represents money spent + bytes leaving the box. Tight schema, tight
  contract.
* ``backend_dispatch`` is a **performance / reliability** surface —
  every entry represents one round-trip to whichever backend the
  ``UnifiedLLMClient`` selected. Local Ollama hits land here. vLLM
  hits land here. So do cloud calls (which ALSO land in
  ``cloud_escalation``).

The ledger lets an operator answer:

* "What's the p50/p95 latency on the planner Ollama right now?"
* "Did vLLM circuit-break in the last hour?"
* "How many of yesterday's 1.2k completions actually went to vLLM
  vs. fell back to Ollama?"

Today the gateway has scattered structured logs that try to answer
those — this module collects them in one place with a uniform shape,
so the Trace-UI / CLI can render the answer without grep.

Privacy contract (same as cloud_escalation): metadata only. Backend
ids, model names, timestamps, success/failure shape, optional token
counts. **No prompt content. No response content.** Tests assert this
explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from cognithor.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Outcome taxonomy
# ---------------------------------------------------------------------------


class DispatchOutcome(StrEnum):
    """Closed taxonomy of how a backend dispatch ended.

    Closed so consumers can render different icons / colour-codes per
    outcome without an Unknown-fallback branch. New values are an
    intentional review point.
    """

    SUCCESS = "success"
    """Backend returned a usable response within its timeout."""

    BACKEND_ERROR = "backend_error"
    """Backend responded but with a structured error (4xx/5xx, schema
    violation, content-policy refusal). Distinct from a transport
    failure — the backend was reachable."""

    TRANSPORT_ERROR = "transport_error"
    """Network-level failure: timeout, connection refused, DNS, etc.
    The backend never responded."""

    CIRCUIT_OPEN = "circuit_open"
    """The ``UnifiedLLMClient`` circuit-breaker rejected the call
    before reaching the backend. Counts as an "attempted dispatch"
    (the operator wants to see breaker activity in the ledger) but no
    bytes left the machine."""

    BAD_REQUEST = "bad_request"
    """Request was malformed before transport (missing model, invalid
    message shape). The backend never saw the request; useful for
    distinguishing client bugs from backend availability issues."""

    CANCELLED = "cancelled"
    """Caller cancelled mid-flight (asyncio.CancelledError, user
    abort). Distinct from errors — no remediation needed."""


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendDispatchEvent:
    """Metadata-only record of a single backend chat dispatch.

    Frozen so consumers can hash / embed it in audit trails without
    copy. All timestamps are UTC.

    Privacy: token counts are optional (the backend ABI doesn't always
    surface them); when the count is unknown, the field stays at -1
    rather than 0 so summaries can distinguish "no tokens reported"
    from "0 tokens emitted".
    """

    backend_type: str
    """Backend identifier (``"ollama"``, ``"vllm"``, ``"anthropic"``,
    ``"openai"``, ``"gemini"``, ``"claude_code"``, ``"vllm_inprocess"``,
    ...). Stable across releases — the Trace-UI builds bucket charts
    keyed off this."""

    model: str
    """Model name passed to the backend (``"qwen3:30b"``,
    ``"claude-opus-4-7"``). Empty when the dispatch failed before the
    model was selected (e.g. circuit-open before model resolution)."""

    outcome: DispatchOutcome
    """Closed-taxonomy outcome — see :class:`DispatchOutcome`."""

    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    """When the dispatch left the unified-client layer."""

    completed_at: datetime | None = None
    """When the dispatch resolved (success OR error). ``None`` only
    when the recorder had no chance to capture completion (cancelled
    process, midway-crash). Production code always sets this."""

    prompt_tokens: int = -1
    """Prompt token count if known (-1 when the backend ABI didn't
    surface it; counts will be 0 for empty prompts)."""

    response_tokens: int = -1
    """Response token count if known (-1 when unknown)."""

    error_kind: str = ""
    """Class name of the exception that ended a non-SUCCESS dispatch,
    or empty for SUCCESS. Stays short — the UI surfaces this as a
    short tag, not a stack trace."""

    error_msg: str = ""
    """First line of the exception message, capped at 200 chars. Empty
    on SUCCESS."""

    is_fallback: bool = False
    """True when this dispatch was a fallback after a primary backend
    failed (e.g. vLLM → Ollama). Lets the operator see how often the
    primary serves vs. how often the system limps along on the
    backup."""

    run_id: str = ""
    """Cross-reference key into TRUST-1 run-receipts and the audit
    hash-chain. Empty when the dispatch happened outside a tracked
    run (e.g. background summarisation or a /probe call)."""

    request_id: str = ""
    """Optional secondary key — distinguishes individual dispatches
    inside the same run (e.g. multi-step Planner → Executor)."""

    notes: str = ""
    """Free-text breadcrumb. MUST NOT contain prompt or response
    content; the privacy contract carries through the dispatch surface."""

    def __post_init__(self) -> None:
        if not self.backend_type:
            msg = "BackendDispatchEvent.backend_type must be a non-empty string"
            raise ValueError(msg)
        if self.prompt_tokens < -1:
            msg = f"BackendDispatchEvent.prompt_tokens must be >= -1 (got {self.prompt_tokens})"
            raise ValueError(msg)
        if self.response_tokens < -1:
            msg = f"BackendDispatchEvent.response_tokens must be >= -1 (got {self.response_tokens})"
            raise ValueError(msg)
        if self.completed_at is not None and self.completed_at < self.started_at:
            msg = (
                f"BackendDispatchEvent.completed_at ({self.completed_at.isoformat()}) "
                f"must be >= started_at ({self.started_at.isoformat()})"
            )
            raise ValueError(msg)
        # Cap the error_msg at the documented length so a verbose
        # backend banner can't bloat the ledger memory footprint.
        # Frozen dataclass: object.__setattr__ via internal escape.
        if len(self.error_msg) > 200:
            object.__setattr__(self, "error_msg", self.error_msg[:200])

    @property
    def latency_s(self) -> float | None:
        """Wall-clock dispatch latency in seconds, or ``None`` when
        the dispatch is still in flight (``completed_at is None``)."""
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds()

    @property
    def succeeded(self) -> bool:
        return self.outcome == DispatchOutcome.SUCCESS

    @property
    def total_tokens(self) -> int:
        """Sum of prompt + response tokens, or ``-1`` when either is unknown."""
        if self.prompt_tokens < 0 or self.response_tokens < 0:
            return -1
        return self.prompt_tokens + self.response_tokens


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DispatchSummary:
    """Aggregate over a :class:`BackendDispatchLedger` window.

    The Trace-UI renders one chip per backend. CLI ``cognithor
    backends-status`` builds a table from the ``by_backend`` /
    ``by_outcome`` cross-buckets.
    """

    event_count: int
    success_count: int
    by_backend: dict[str, int]
    by_outcome: dict[DispatchOutcome, int]
    fallback_count: int
    total_prompt_tokens: int
    """-1 when ANY contributing event reported -1; otherwise the sum.
    Mixed-known/unknown is treated as unknown to avoid silent under-
    counting."""
    total_response_tokens: int
    """Same -1-propagation rule as ``total_prompt_tokens``."""

    @property
    def success_rate(self) -> float:
        """Fraction of dispatches in [0.0, 1.0]; 1.0 for an empty ledger
        (vacuously successful) so empty buckets don't flag red in the UI."""
        if self.event_count == 0:
            return 1.0
        return self.success_count / self.event_count


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


class BackendDispatchLedger:
    """Append-only in-memory ledger of backend dispatch events.

    Production code uses :data:`BACKEND_DISPATCH_LEDGER`; tests
    construct fresh ledgers for isolation. The ledger never trims —
    callers that want a bounded window snapshot via :meth:`in_window`.
    """

    def __init__(self) -> None:
        self._events: list[BackendDispatchEvent] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def record(self, event: BackendDispatchEvent) -> None:
        """Append ``event`` to the ledger."""
        self._events.append(event)
        log.debug(
            "backend_dispatch_recorded",
            backend=event.backend_type,
            model=event.model,
            outcome=event.outcome.value,
            latency_s=event.latency_s,
            run_id=event.run_id or None,
        )

    def clear(self) -> None:
        """Drop all events (test helper; production code never calls this)."""
        self._events.clear()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._events)

    def events(self) -> tuple[BackendDispatchEvent, ...]:
        """Return all events in insertion order."""
        return tuple(self._events)

    def by_run(self, run_id: str) -> tuple[BackendDispatchEvent, ...]:
        """Events whose ``run_id`` exactly matches ``run_id``."""
        return tuple(e for e in self._events if e.run_id == run_id)

    def by_backend(self, backend_type: str) -> tuple[BackendDispatchEvent, ...]:
        """Events whose ``backend_type`` exactly matches."""
        return tuple(e for e in self._events if e.backend_type == backend_type)

    def by_outcome(self, outcome: DispatchOutcome) -> tuple[BackendDispatchEvent, ...]:
        return tuple(e for e in self._events if e.outcome == outcome)

    def in_window(self, *, start: datetime, end: datetime) -> tuple[BackendDispatchEvent, ...]:
        """Events with ``start <= started_at <= end``."""
        return tuple(e for e in self._events if start <= e.started_at <= end)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def summarise(
        self,
        events: tuple[BackendDispatchEvent, ...] | None = None,
    ) -> DispatchSummary:
        """Compute a :class:`DispatchSummary` over ``events`` (default: all).

        Use the per-axis filters (:meth:`by_run`, :meth:`by_backend`,
        :meth:`in_window`) to scope the input.
        """
        ev = events if events is not None else tuple(self._events)
        by_backend: dict[str, int] = {}
        by_outcome: dict[DispatchOutcome, int] = {}
        success_count = 0
        fallback_count = 0
        total_prompt = 0
        total_response = 0
        prompt_known = True
        response_known = True

        for e in ev:
            by_backend[e.backend_type] = by_backend.get(e.backend_type, 0) + 1
            by_outcome[e.outcome] = by_outcome.get(e.outcome, 0) + 1
            if e.outcome == DispatchOutcome.SUCCESS:
                success_count += 1
            if e.is_fallback:
                fallback_count += 1
            if e.prompt_tokens < 0:
                prompt_known = False
            else:
                total_prompt += e.prompt_tokens
            if e.response_tokens < 0:
                response_known = False
            else:
                total_response += e.response_tokens

        return DispatchSummary(
            event_count=len(ev),
            success_count=success_count,
            by_backend=by_backend,
            by_outcome=by_outcome,
            fallback_count=fallback_count,
            total_prompt_tokens=total_prompt if prompt_known else -1,
            total_response_tokens=total_response if response_known else -1,
        )

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> list[dict[str, object]]:
        """JSON-serialisable list of all events for embedding into the
        run-receipt or REST surface. Lossless w.r.t. the public fields
        of :class:`BackendDispatchEvent`."""
        rows: list[dict[str, object]] = []
        for e in self._events:
            rows.append(
                {
                    "backend_type": e.backend_type,
                    "model": e.model,
                    "outcome": e.outcome.value,
                    "started_at": e.started_at.isoformat(),
                    "completed_at": (e.completed_at.isoformat() if e.completed_at else None),
                    "latency_s": e.latency_s,
                    "prompt_tokens": e.prompt_tokens,
                    "response_tokens": e.response_tokens,
                    "error_kind": e.error_kind,
                    "error_msg": e.error_msg,
                    "is_fallback": e.is_fallback,
                    "run_id": e.run_id,
                    "request_id": e.request_id,
                    "notes": e.notes,
                }
            )
        return rows


# ---------------------------------------------------------------------------
# Process-local default
# ---------------------------------------------------------------------------

# Production callers (UnifiedLLMClient.chat, vllm fallback path)
# write into this instance. Tests construct their own
# :class:`BackendDispatchLedger` for isolation.
BACKEND_DISPATCH_LEDGER: BackendDispatchLedger = BackendDispatchLedger()


# ---------------------------------------------------------------------------
# Convenience builder for the call-site wiring
# ---------------------------------------------------------------------------


def record_backend_dispatch(
    *,
    backend_type: str,
    model: str,
    outcome: DispatchOutcome,
    started_at: datetime,
    completed_at: datetime | None = None,
    prompt_tokens: int = -1,
    response_tokens: int = -1,
    error_kind: str = "",
    error_msg: str = "",
    is_fallback: bool = False,
    run_id: str = "",
    request_id: str = "",
    notes: str = "",
    ledger: BackendDispatchLedger | None = None,
) -> BackendDispatchEvent:
    """Build + record a :class:`BackendDispatchEvent` in one call.

    Convenience wrapper for the ``UnifiedLLMClient`` hook so the call
    site doesn't have to import the dataclass + ledger separately.
    Returns the recorded event for downstream callers that want to
    embed it in their own audit trail.

    ``completed_at`` defaults to ``datetime.now(UTC)`` when omitted —
    the common case where the recorder fires immediately after the
    backend returns.

    The default ``ledger`` is the canonical
    :data:`BACKEND_DISPATCH_LEDGER`; tests pass a fresh instance for
    isolation.
    """
    target_ledger = ledger if ledger is not None else BACKEND_DISPATCH_LEDGER
    actual_completed = completed_at if completed_at is not None else datetime.now(UTC)
    event = BackendDispatchEvent(
        backend_type=backend_type,
        model=model,
        outcome=outcome,
        started_at=started_at,
        completed_at=actual_completed,
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
        error_kind=error_kind,
        error_msg=error_msg,
        is_fallback=is_fallback,
        run_id=run_id,
        request_id=request_id,
        notes=notes,
    )
    target_ledger.record(event)
    return event


__all__ = [
    "BACKEND_DISPATCH_LEDGER",
    "BackendDispatchEvent",
    "BackendDispatchLedger",
    "DispatchOutcome",
    "DispatchSummary",
    "record_backend_dispatch",
]
