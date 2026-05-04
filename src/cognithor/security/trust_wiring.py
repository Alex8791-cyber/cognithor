"""Cross-wiring helpers between the TRUST-5..10 ledgers (2026-05-04).

The six foundations (#395-#401) ship as independent in-memory
ledgers. Several pairs naturally reconcile on a shared key:

* TRUST-8 ``EscalationEvent.run_id`` ↔ TRUST-6 ``CostEntry.run_id`` —
  every cloud completion incurs a USD cost; the two ledgers should
  agree on the total when scoped to the same run.

This module ships *passive* helpers that emit cross-ledger entries
on a single call, so callers don't have to remember to record into
both ledgers and risk drift. Pure functions of the input event +
ledger references — no HTTP, no IO, no mutation outside the
explicit `record(...)` calls.

Privacy contract carries through: only metadata (token counts,
USD micro, backend ids, run_id) flows. No prompt or response
content.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cognithor.security.cloud_escalation import (
    ESCALATION_LEDGER,
    EscalationEvent,
)
from cognithor.security.cost_ledger import (
    COST_LEDGER,
    CostEntry,
    CostKind,
)

if TYPE_CHECKING:
    from cognithor.security.cloud_escalation import EscalationLedger
    from cognithor.security.cost_ledger import CostLedger


def record_escalation_with_cost(
    event: EscalationEvent,
    *,
    escalation_ledger: EscalationLedger | None = None,
    cost_ledger: CostLedger | None = None,
) -> CostEntry | None:
    """Record ``event`` to the escalation ledger AND mirror its cost.

    The cost-mirror only fires when ``event.cost_usd_micro > 0`` —
    a free / cached escalation has nothing to bill. Returns the
    derived :class:`CostEntry` (or ``None`` when no cost was
    mirrored), so callers can attach it to a richer audit log if
    they want.

    The mirrored entry maps:

    * ``kind = LLM_INFERENCE`` (escalations are by definition LLM
      calls — the foundation doesn't model non-LLM escalations).
    * ``tool = event.to_backend`` — the cloud backend that
      *received* the escalation, so cost-by-tool aggregation lines
      up with provider billing.
    * ``backend = event.to_backend`` (same).
    * ``run_id`` / ``request_id`` / ``prompt_tokens`` /
      ``response_tokens`` / ``occurred_at`` / ``cost_usd_micro``
      threaded straight through.
    * ``notes`` documents the cross-ledger origin.

    Both ledgers default to the canonical singletons; tests inject
    fresh instances for isolation.
    """
    actual_escalation = escalation_ledger if escalation_ledger is not None else ESCALATION_LEDGER
    actual_cost = cost_ledger if cost_ledger is not None else COST_LEDGER

    actual_escalation.record(event)

    if event.cost_usd_micro <= 0:
        return None

    entry = CostEntry(
        kind=CostKind.LLM_INFERENCE,
        tool=event.to_backend,
        cost_usd_micro=event.cost_usd_micro,
        backend=event.to_backend,
        run_id=event.run_id,
        channel="",
        domain="",
        prompt_tokens=event.prompt_tokens,
        response_tokens=event.response_tokens,
        unit_count=-1,
        occurred_at=event.started_at,
        notes=f"escalation:{event.reason.value}",
    )
    actual_cost.record(entry)
    return entry


__all__ = ["record_escalation_with_cost"]
