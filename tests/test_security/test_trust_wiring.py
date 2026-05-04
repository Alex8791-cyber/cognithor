"""Tests for the TRUST-6/8 cross-wiring helper."""

from __future__ import annotations

from datetime import UTC, datetime

from cognithor.security.cloud_escalation import (
    EscalationEvent,
    EscalationLedger,
    EscalationReason,
)
from cognithor.security.cost_ledger import (
    CostKind,
    CostLedger,
)
from cognithor.security.trust_wiring import record_escalation_with_cost


def _utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=UTC)


def _event(
    *,
    cost_usd_micro: int = 15_000,
    run_id: str = "run-42",
    prompt_tokens: int = 1000,
    response_tokens: int = 200,
    reason: EscalationReason = EscalationReason.OWNER_OVERRIDE,
) -> EscalationEvent:
    return EscalationEvent(
        reason=reason,
        from_backend="ollama:qwen3:30b",
        to_backend="anthropic:claude-opus-4-7",
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
        cost_usd_micro=cost_usd_micro,
        started_at=_utc(2026, 5, 4, 12),
        run_id=run_id,
    )


class TestRecordEscalationWithCost:
    def test_records_to_both_ledgers(self) -> None:
        esc = EscalationLedger()
        cost = CostLedger()
        ev = _event()
        entry = record_escalation_with_cost(ev, escalation_ledger=esc, cost_ledger=cost)
        assert entry is not None
        # Escalation ledger has the event.
        assert esc.events() == (ev,)
        # Cost ledger has the mirror.
        assert len(cost) == 1
        recorded = cost.entries()[0]
        assert recorded.tool == "anthropic:claude-opus-4-7"
        assert recorded.backend == "anthropic:claude-opus-4-7"
        assert recorded.kind == CostKind.LLM_INFERENCE
        assert recorded.cost_usd_micro == 15_000
        assert recorded.run_id == "run-42"
        assert recorded.prompt_tokens == 1000
        assert recorded.response_tokens == 200
        assert recorded.occurred_at == _utc(2026, 5, 4, 12)
        assert recorded.notes == "escalation:owner_override"

    def test_zero_cost_skips_mirror(self) -> None:
        # A cached / free escalation has nothing to bill — the
        # escalation event still flows, but no cost entry.
        esc = EscalationLedger()
        cost = CostLedger()
        ev = _event(cost_usd_micro=0)
        entry = record_escalation_with_cost(ev, escalation_ledger=esc, cost_ledger=cost)
        assert entry is None
        assert len(esc) == 1
        assert len(cost) == 0

    def test_run_id_consistency_across_ledgers(self) -> None:
        # Cross-ledger consistency is the value-add: by_run should
        # surface the same run on both sides.
        esc = EscalationLedger()
        cost = CostLedger()
        record_escalation_with_cost(
            _event(run_id="run-42", cost_usd_micro=10_000),
            escalation_ledger=esc,
            cost_ledger=cost,
        )
        record_escalation_with_cost(
            _event(run_id="run-99", cost_usd_micro=99_999),
            escalation_ledger=esc,
            cost_ledger=cost,
        )
        record_escalation_with_cost(
            _event(run_id="run-42", cost_usd_micro=20_000),
            escalation_ledger=esc,
            cost_ledger=cost,
        )
        # Same run_id surfaces aligned totals on both sides.
        esc_run42 = esc.summarise(esc.by_run("run-42"))
        cost_run42 = cost.summarise(cost.by_run("run-42"))
        assert esc_run42.event_count == 2
        assert cost_run42.entry_count == 2
        assert esc_run42.total_cost_usd_micro == cost_run42.total_cost_usd_micro == 30_000

    def test_uses_canonical_ledgers_when_omitted(self) -> None:
        # No injection ⇒ writes to the process-local singletons. We
        # don't assert specific contents (production state may have
        # been wired) — just that the helper does not raise.
        ev = _event(cost_usd_micro=0)  # zero cost ⇒ no cost-side mutation
        result = record_escalation_with_cost(ev)
        assert result is None

    def test_reason_threaded_into_notes(self) -> None:
        esc = EscalationLedger()
        cost = CostLedger()
        record_escalation_with_cost(
            _event(reason=EscalationReason.CONTEXT_TOO_LARGE),
            escalation_ledger=esc,
            cost_ledger=cost,
        )
        recorded = cost.entries()[0]
        assert recorded.notes == "escalation:context_too_large"
