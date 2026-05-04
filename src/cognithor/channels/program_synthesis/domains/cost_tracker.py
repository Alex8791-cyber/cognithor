"""``DomainCostTracker`` — per-domain token + wall-time accounting.

Owner-Decision D6 (Sprint-26 memo): Cost-Tracker is *not* a stretch
goal but a core selling argument for the Insurance-Advisor pack
("locally executed — at OpenAI prices this would have cost X €"). It
ships with §26.1 Foundation.

The tracker is intentionally minimal:

* ``record(domain, tokens_in, tokens_out, wall_ms, model, escalated)``
* ``snapshot()`` returns a per-domain aggregate dict suitable for
  `cost_summary` REST endpoints
* USD/EUR price-lookup is plug-in via a simple ``ModelPricing`` map so
  tests don't depend on live exchange rates

This file does not import the global ``cognithor.telemetry.cost_tracker``;
that module covers cross-channel cost. The Sprint-26 tracker is
explicitly per-PSE-domain so the Insurance-Pack-Marketing copy can
quote "synthesis cost €" cleanly without mixing in chat / planner
costs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelPricing:
    """USD price per 1k tokens for a given model.

    Sprint-26 only stores the two prices we currently quote against
    (Qwen3 local = 0.0, GPT-4o cloud = canonical OpenAI price). The
    map is deliberately small; expanding it is a 2-line PR.
    """

    input_per_1k_usd: float
    output_per_1k_usd: float


# Conservative defaults — local runs are free, cloud comparison uses
# 2024-published GPT-4o list price. Source-of-truth lives outside this
# module (config or pricing.json) so we don't bake numbers into Python.
DEFAULT_PRICING: dict[str, ModelPricing] = {
    "qwen3:27b": ModelPricing(input_per_1k_usd=0.0, output_per_1k_usd=0.0),
    "qwen3.6:27b": ModelPricing(input_per_1k_usd=0.0, output_per_1k_usd=0.0),
    "ollama/local": ModelPricing(input_per_1k_usd=0.0, output_per_1k_usd=0.0),
    # Reference cloud price for marketing comparison (USD).
    "gpt-4o": ModelPricing(input_per_1k_usd=0.005, output_per_1k_usd=0.015),
    "gpt-4o-mini": ModelPricing(input_per_1k_usd=0.00015, output_per_1k_usd=0.0006),
    "claude-sonnet-4-5": ModelPricing(input_per_1k_usd=0.003, output_per_1k_usd=0.015),
}


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class DomainCostRecord:
    """Mutable aggregate of cost stats for one domain."""

    domain: str
    call_count: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    wall_ms_total: float = 0.0
    escalation_count: int = 0  # how many calls escalated to a larger budget
    cost_usd_local: float = 0.0
    cost_usd_cloud_reference: float = 0.0
    by_model: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "call_count": self.call_count,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "wall_ms_total": round(self.wall_ms_total, 2),
            "escalation_count": self.escalation_count,
            "cost_usd_local": round(self.cost_usd_local, 6),
            "cost_usd_cloud_reference": round(self.cost_usd_cloud_reference, 6),
            "savings_usd_vs_cloud": round(self.cost_usd_cloud_reference - self.cost_usd_local, 6),
            "by_model": dict(self.by_model),
        }


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class DomainCostTracker:
    """Per-domain in-memory cost aggregator.

    The tracker is intentionally process-local. Persistence (writing
    to ``~/.cognithor/cost.db``) is the responsibility of the global
    ``cognithor.telemetry.cost_tracker.CostTracker`` — this Sprint-26
    layer just provides the per-domain breakdown the Flutter
    Cost-Summary widget and the Insurance-Pack-Marketing copy quote.
    """

    def __init__(
        self,
        pricing: dict[str, ModelPricing] | None = None,
        cloud_reference_model: str = "gpt-4o",
    ) -> None:
        self._pricing = pricing or DEFAULT_PRICING
        self._records: dict[str, DomainCostRecord] = {}
        self._cloud_ref = cloud_reference_model

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        *,
        domain: str,
        tokens_in: int,
        tokens_out: int,
        wall_ms: float,
        model: str,
        escalated: bool = False,
    ) -> DomainCostRecord:
        """Append one synthesis call to the tracker.

        Returns the (mutated) :class:`DomainCostRecord` so callers can
        embed it directly in audit-log entries without a second lookup.
        """
        if tokens_in < 0 or tokens_out < 0:
            msg = "token counts must be non-negative"
            raise ValueError(msg)
        if wall_ms < 0:
            msg = "wall_ms must be non-negative"
            raise ValueError(msg)

        record = self._records.setdefault(domain, DomainCostRecord(domain=domain))
        record.call_count += 1
        record.tokens_in += tokens_in
        record.tokens_out += tokens_out
        record.wall_ms_total += wall_ms
        if escalated:
            record.escalation_count += 1
        record.by_model[model] = record.by_model.get(model, 0) + 1

        local_pricing = self._pricing.get(model)
        if local_pricing is not None:
            record.cost_usd_local += (tokens_in / 1000.0) * local_pricing.input_per_1k_usd + (
                tokens_out / 1000.0
            ) * local_pricing.output_per_1k_usd

        cloud_pricing = self._pricing.get(self._cloud_ref)
        if cloud_pricing is not None:
            record.cost_usd_cloud_reference += (
                tokens_in / 1000.0
            ) * cloud_pricing.input_per_1k_usd + (
                tokens_out / 1000.0
            ) * cloud_pricing.output_per_1k_usd
        return record

    # ------------------------------------------------------------------
    # Read-only views
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Return a stable, JSON-serialisable per-domain snapshot."""
        return {name: rec.to_dict() for name, rec in self._records.items()}

    def get(self, domain: str) -> DomainCostRecord | None:
        return self._records.get(domain)

    def reset(self) -> None:
        """Clear all records — primarily for tests."""
        self._records.clear()

    # ------------------------------------------------------------------
    # Convenience: timing context manager
    # ------------------------------------------------------------------

    def time(self, domain: str, *, model: str) -> _TimingContext:
        """Return a context manager that auto-records wall_ms.

        Usage::

            with tracker.time("sql", model="qwen3:27b") as scope:
                scope.set_tokens(in_=120, out_=50)
                scope.set_escalated(False)
                # ... run synthesis ...

        The context manager captures wall-time in ``__exit__`` and
        forwards everything to :meth:`record`.
        """
        return _TimingContext(self, domain=domain, model=model)


class _TimingContext:
    """Helper class for :meth:`DomainCostTracker.time`."""

    def __init__(self, tracker: DomainCostTracker, *, domain: str, model: str) -> None:
        self._tracker = tracker
        self._domain = domain
        self._model = model
        self._tokens_in = 0
        self._tokens_out = 0
        self._escalated = False
        self._t0 = 0.0

    def set_tokens(self, *, in_: int, out_: int) -> None:
        self._tokens_in = in_
        self._tokens_out = out_

    def set_escalated(self, value: bool) -> None:
        self._escalated = value

    def __enter__(self) -> _TimingContext:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        wall_ms = (time.perf_counter() - self._t0) * 1000.0
        self._tracker.record(
            domain=self._domain,
            tokens_in=self._tokens_in,
            tokens_out=self._tokens_out,
            wall_ms=wall_ms,
            model=self._model,
            escalated=self._escalated,
        )
