# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-15 — MTP (multi-token prediction) speculative-decoding stats.

When ``build_inprocess_vllm_*_choice_fn`` is given a ``speculative_config``,
vLLM runs draft-and-verify inside ``LLM.generate``: each forward pass
emits up to ``num_speculative_tokens`` candidate tokens that the
larger model verifies. Tokens that match are accepted in a single
forward pass — the practical speedup depends on **acceptance rate**
(fraction of drafts that survive verification).

Without instrumentation we can't tell whether MTP is paying off:

* High acceptance (≥ 0.7) → near the theoretical 1.9× decode lift.
* Low acceptance (< 0.3) → drafts are wrong half the time, the
  speculative pass is wasted compute.

vLLM exposes spec-decoding stats two ways:

1. **Per-request:** ``RequestOutput.metrics.spec_token_acceptance_counts``
   (vLLM 0.20+; tuple of len ``num_speculative_tokens+1`` showing how
   many tokens were accepted at each draft position).
2. **Per-engine cumulative:** ``LLM.llm_engine.get_metrics()`` →
   list of ``Metric`` objects including
   ``vllm:spec_decode_num_drafts_total``,
   ``vllm:spec_decode_num_accepted_tokens_total``,
   ``vllm:spec_decode_num_emitted_tokens_total``.

This module ships :class:`MTPStats`, a sticky aggregator that polls
either source and exposes a clean :meth:`acceptance_rate` plus
:meth:`spec_token_efficiency` (accepted-emitted ratio).

The poll is **non-invasive** and **safe** — if MTP isn't configured
or the metrics aren't accessible, ``acceptance_rate`` returns ``None``
and the caller can ignore.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "MTPSnapshot",
    "MTPStats",
    "extract_per_request_acceptance",
    "poll_engine_mtp_metrics",
]


@dataclass(frozen=True)
class MTPSnapshot:
    """One aggregated snapshot of speculative-decoding metrics."""

    drafts_proposed: int  # how many speculative drafts the model emitted
    drafts_accepted: int  # how many of those drafts the verifier accepted
    tokens_emitted: int  # total tokens output (incl. drafts that got accepted)
    num_speculative_tokens: int  # config: drafts proposed per forward pass

    @property
    def acceptance_rate(self) -> float | None:
        """Fraction of drafts that survived verification (0..1)."""
        if self.drafts_proposed == 0:
            return None
        return self.drafts_accepted / self.drafts_proposed

    @property
    def spec_token_efficiency(self) -> float | None:
        """Mean accepted-tokens-per-forward-pass.

        ``1.0`` means MTP gave no speedup (only the base token survived).
        ``num_speculative_tokens + 1`` is the theoretical max.
        """
        if self.tokens_emitted == 0:
            return None
        # tokens_emitted = drafts_accepted + base_tokens; "passes" =
        # tokens_emitted - drafts_accepted (the base tokens, one per pass)
        passes = self.tokens_emitted - self.drafts_accepted
        if passes <= 0:
            return None
        return self.tokens_emitted / passes


def extract_per_request_acceptance(request_output: Any) -> MTPSnapshot | None:
    """Pull spec-decoding stats out of a ``vllm.RequestOutput``.

    Reads ``request_output.metrics.spec_token_acceptance_counts`` —
    a tuple where index ``i`` holds the count of forward passes that
    accepted exactly ``i`` speculative tokens. Aggregates into a
    snapshot. Returns ``None`` if the field isn't present (MTP off
    or older vLLM).
    """
    metrics = getattr(request_output, "metrics", None)
    if metrics is None:
        return None
    counts = getattr(metrics, "spec_token_acceptance_counts", None)
    if not counts:
        return None
    # counts is e.g. (12, 8, 5, 3) — 12 passes accepted 0 drafts,
    # 8 accepted 1, 5 accepted 2, 3 accepted 3.
    num_spec = len(counts) - 1
    drafts_proposed = sum(counts) * num_spec  # passes × drafts/pass
    drafts_accepted = sum(i * c for i, c in enumerate(counts))
    # base tokens = sum(counts) (one per forward pass), accepted on top.
    tokens_emitted = sum(counts) + drafts_accepted
    return MTPSnapshot(
        drafts_proposed=drafts_proposed,
        drafts_accepted=drafts_accepted,
        tokens_emitted=tokens_emitted,
        num_speculative_tokens=num_spec,
    )


def poll_engine_mtp_metrics(llm: Any) -> MTPSnapshot | None:
    """Read cumulative spec-decoding metrics off ``LLM.llm_engine``.

    Iterates ``llm.llm_engine.get_metrics()`` looking for the
    spec-decoding gauges. Falls back to ``stat_logger.spec_decode_stats``
    on older vLLM. Returns ``None`` if neither path is available.
    """
    engine = getattr(llm, "llm_engine", None)
    if engine is None:
        return None

    drafts = accepted = emitted = 0
    found_any = False

    # Path A — modern get_metrics().
    get_metrics = getattr(engine, "get_metrics", None)
    if callable(get_metrics):
        try:
            for metric in get_metrics():
                name = getattr(metric, "name", "") or ""
                value = getattr(metric, "value", None)
                if value is None:
                    continue
                if name.endswith("spec_decode_num_drafts_total"):
                    drafts = int(value)
                    found_any = True
                elif name.endswith("spec_decode_num_accepted_tokens_total"):
                    accepted = int(value)
                    found_any = True
                elif name.endswith("spec_decode_num_emitted_tokens_total"):
                    emitted = int(value)
                    found_any = True
        except Exception:
            pass

    # Path B — older stat_logger fallback.
    if not found_any:
        stat_logger = getattr(engine, "stat_logger", None)
        spec = getattr(stat_logger, "spec_decode_stats", None) if stat_logger else None
        if spec is not None:
            try:
                drafts = int(getattr(spec, "num_drafts", 0))
                accepted = int(getattr(spec, "num_accepted_tokens", 0))
                emitted = int(getattr(spec, "num_emitted_tokens", 0))
                found_any = drafts + accepted + emitted > 0
            except Exception:
                pass

    if not found_any:
        return None

    # num_speculative_tokens isn't on the metrics; infer from drafts/passes
    # if we can. Default 0 (unknown) — caller still gets acceptance_rate.
    num_spec = 0
    return MTPSnapshot(
        drafts_proposed=drafts,
        drafts_accepted=accepted,
        tokens_emitted=emitted,
        num_speculative_tokens=num_spec,
    )


@dataclass
class MTPStats:
    """Sticky aggregator of MTP snapshots across an episode.

    Call :meth:`add_request` after every ``llm.chat`` call (cheap
    pure-Python). Call :meth:`update_from_engine` periodically (e.g.
    once per game) to capture cumulative engine totals as a sanity
    check against the per-request aggregation.

    :attr:`acceptance_rate` and :attr:`spec_token_efficiency` are
    properties that return ``None`` until at least one snapshot is in.
    """

    snapshots: list[MTPSnapshot] = field(default_factory=list)

    def add_request(self, request_output: Any) -> MTPSnapshot | None:
        """Pull a per-request snapshot off a ``RequestOutput`` and store it.

        Returns the snapshot (or ``None`` if MTP wasn't on for this call).
        """
        snap = extract_per_request_acceptance(request_output)
        if snap is not None:
            self.snapshots.append(snap)
        return snap

    def update_from_engine(self, llm: Any) -> MTPSnapshot | None:
        """Read cumulative engine metrics and store as a snapshot."""
        snap = poll_engine_mtp_metrics(llm)
        if snap is not None:
            self.snapshots.append(snap)
        return snap

    @property
    def total_drafts_proposed(self) -> int:
        return sum(s.drafts_proposed for s in self.snapshots)

    @property
    def total_drafts_accepted(self) -> int:
        return sum(s.drafts_accepted for s in self.snapshots)

    @property
    def acceptance_rate(self) -> float | None:
        """Mean acceptance rate across all recorded snapshots (0..1)."""
        if self.total_drafts_proposed == 0:
            return None
        return self.total_drafts_accepted / self.total_drafts_proposed

    @property
    def spec_token_efficiency(self) -> float | None:
        """Mean tokens-emitted-per-forward-pass."""
        emitted = sum(s.tokens_emitted for s in self.snapshots)
        accepted = self.total_drafts_accepted
        passes = emitted - accepted
        if passes <= 0:
            return None
        return emitted / passes

    def summary(self) -> dict[str, Any]:
        """JSON-friendly dump of the aggregated metrics."""
        return {
            "snapshots": len(self.snapshots),
            "total_drafts_proposed": self.total_drafts_proposed,
            "total_drafts_accepted": self.total_drafts_accepted,
            "acceptance_rate": self.acceptance_rate,
            "spec_token_efficiency": self.spec_token_efficiency,
        }
