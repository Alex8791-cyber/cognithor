# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Spec §6.5.2 Zone-2 — Hybrid-Repair (Sprint-1 plan task 9 slice).

When the Refiner mode controller sits in the α-grey-band
(``0.35 ≤ α < 0.45``), trust in the LLM is unclear: a single bad
LLM-call could push performance-α into Symbolic territory, but
purely symbolic repairs may lack the breadth the LLM offers.

Hybrid-Repair runs **both backends in parallel**:

* the **symbolic** advisor (``advise_repairs`` from
  :mod:`refiner.symbolic_repair`) — instant, rule-based, no
  network round-trip;
* a **single-stage LLM** call (no CoT preamble — Stage 2 only,
  per spec §6.5.2 latency note) that proposes additional
  candidate replacement sources.

Both produce a uniform :class:`HybridRepairCandidate`. Each
candidate is scored by an injected ``scorer`` callable (the
Verifier in production); the highest-scoring candidate wins.

This module is the *orchestrator*. The actual evaluator is
dependency-injected so the loop is testable without a live
verifier or LLM.

The key contract: Hybrid-Repair is **always** at least as good
as either backend alone — when the LLM hangs or the symbolic
advisor returns nothing, the other side still produces a result.
``await asyncio.gather(..., return_exceptions=True)`` ensures one
side's failure never sinks the other.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from cognithor.channels.program_synthesis.refiner.llm_repair_two_stage import (
        LLMRepairResult,
    )
    from cognithor.channels.program_synthesis.refiner.symbolic_repair import (
        RepairSuggestion,
    )


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


CandidateOrigin = Literal["symbolic", "llm"]


@dataclass(frozen=True)
class HybridRepairCandidate:
    """One repair candidate from either backend, in a uniform shape.

    ``source`` is the DSL source string the candidate represents —
    either from the symbolic advisor's ``primitive_hint`` (lifted
    into a source by the caller) or from the LLM's
    ``replacement_source``.

    ``confidence`` is the backend's self-reported confidence,
    clamped to ``[0, 1]``. The orchestrator does *not* re-rank by
    confidence — it re-ranks by the scorer's verdict — but
    confidence is preserved for tie-breaks and telemetry.

    ``origin`` tags which backend produced the candidate, so the
    Refiner can attribute wins / losses for the α-controller's
    performance-α update.

    ``detail`` is a free-form note for telemetry.
    """

    source: str
    confidence: float
    origin: CandidateOrigin
    detail: str = ""


@dataclass(frozen=True)
class HybridRepairResult:
    """Outcome of one :func:`run_hybrid_repair` call.

    ``winner`` is the highest-scoring candidate, or ``None`` when
    no candidate beat the ``threshold`` baseline (so the Refiner
    falls back to the next escalation step).

    ``winner_score`` is the absolute score the verifier assigned
    to ``winner``.

    ``candidates_evaluated`` is every candidate that was scored,
    in the order they were scored — used by tests / telemetry to
    audit how the orchestrator weighed alternatives.

    ``symbolic_failed`` / ``llm_failed`` carry the exception
    *types* (as string tags) when one side blew up, so the caller
    can decide whether to penalise that backend's α.
    """

    winner: HybridRepairCandidate | None
    winner_score: float
    candidates_evaluated: tuple[HybridRepairCandidate, ...]
    symbolic_failed: str | None = None
    llm_failed: str | None = None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


SymbolicSupplier = Callable[[], Iterable["RepairSuggestion"]]
LLMSupplier = Callable[[], Awaitable["LLMRepairResult"]]
Scorer = Callable[[HybridRepairCandidate], Awaitable[float]]


async def run_hybrid_repair(
    *,
    symbolic_supplier: SymbolicSupplier,
    llm_supplier: LLMSupplier,
    scorer: Scorer,
    symbolic_to_source: Callable[[RepairSuggestion], str | None] | None = None,
    threshold: float = 0.0,
) -> HybridRepairResult:
    """Run symbolic + LLM repair in parallel, score every candidate, pick the best.

    ``symbolic_supplier`` returns the :class:`RepairSuggestion` list
    from :func:`advise_repairs`. The synchronous interface keeps the
    advisor cheap; the orchestrator wraps it in ``asyncio.to_thread``
    so it runs concurrent with the LLM round-trip.

    ``llm_supplier`` runs the actual LLM repair (typically a
    :class:`LLMRepairTwoStageClient.repair` configured with
    ``llm_json_max_retries=0`` for single-stage latency).

    ``symbolic_to_source`` lifts a :class:`RepairSuggestion` into a
    DSL source string. Defaults to using the suggestion's
    ``primitive_hint`` directly when present (so e.g.
    ``primitive_hint="rotate90"`` becomes ``source="rotate90(input)"``);
    if ``primitive_hint`` is ``None`` (R5 local-edit) the suggestion
    is dropped from the candidate set.

    ``scorer`` is the verifier callable; it returns a score in
    ``[0, 1]`` (or any float — the orchestrator sorts descending).

    ``threshold`` is the floor the winner must clear. When no
    candidate beats it, ``HybridRepairResult.winner`` is ``None``.

    The function never raises on backend failures — it captures
    the exception type and continues with the surviving backend.
    """
    if symbolic_to_source is None:
        symbolic_to_source = _default_symbolic_to_source

    sym_task = asyncio.create_task(asyncio.to_thread(_collect_symbolic, symbolic_supplier))
    llm_task = asyncio.create_task(_safe_llm(llm_supplier))

    sym_outcome, llm_outcome = await asyncio.gather(sym_task, llm_task)

    candidates: list[HybridRepairCandidate] = []
    symbolic_failed = sym_outcome.error
    if sym_outcome.suggestions is not None:
        for sym_sug in sym_outcome.suggestions:
            source = symbolic_to_source(sym_sug)
            if source is None:
                continue
            candidates.append(
                HybridRepairCandidate(
                    source=source,
                    confidence=_clamp_unit(sym_sug.confidence),
                    origin="symbolic",
                    detail=sym_sug.detail,
                )
            )

    llm_failed = llm_outcome.error
    if llm_outcome.result is not None:
        for llm_sug in llm_outcome.result.suggestions:
            candidates.append(
                HybridRepairCandidate(
                    source=llm_sug.replacement_source,
                    confidence=_clamp_unit(llm_sug.confidence),
                    origin="llm",
                    detail=llm_sug.reasoning,
                )
            )

    scored: list[tuple[HybridRepairCandidate, float]] = []
    for candidate in candidates:
        try:
            score = await scorer(candidate)
        except Exception:
            continue
        if not math.isfinite(score):
            continue
        scored.append((candidate, score))

    # Pick the highest score above threshold; ties broken by descending confidence.
    scored.sort(key=lambda pair: (pair[1], pair[0].confidence), reverse=True)
    winner: HybridRepairCandidate | None = None
    winner_score = float("-inf")
    for cand, score in scored:
        if score >= threshold:
            winner = cand
            winner_score = score
            break

    return HybridRepairResult(
        winner=winner,
        winner_score=winner_score if winner is not None else 0.0,
        candidates_evaluated=tuple(c for c, _ in scored),
        symbolic_failed=symbolic_failed,
        llm_failed=llm_failed,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SymbolicOutcome:
    suggestions: tuple[RepairSuggestion, ...] | None
    error: str | None


@dataclass(frozen=True)
class _LLMOutcome:
    result: LLMRepairResult | None
    error: str | None


def _collect_symbolic(supplier: SymbolicSupplier) -> _SymbolicOutcome:
    try:
        return _SymbolicOutcome(suggestions=tuple(supplier()), error=None)
    except Exception as exc:
        return _SymbolicOutcome(suggestions=None, error=type(exc).__name__)


async def _safe_llm(supplier: LLMSupplier) -> _LLMOutcome:
    try:
        return _LLMOutcome(result=await supplier(), error=None)
    except Exception as exc:
        return _LLMOutcome(result=None, error=type(exc).__name__)


def _default_symbolic_to_source(sug: RepairSuggestion) -> str | None:
    """Lift a RepairSuggestion to a source string via its primitive_hint."""
    hint = sug.primitive_hint
    if hint is None:
        return None
    # The primitive hint is bare ("rotate90", "scale_up_2x", "recolor"); we
    # don't know the right argument shape without registry awareness, so
    # default to the simplest case: ``<hint>(input)``. The Refiner driver
    # is free to override ``symbolic_to_source`` with a registry-aware
    # version when more context is needed.
    return f"{hint}(input)"


def _clamp_unit(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


__all__ = [
    "CandidateOrigin",
    "HybridRepairCandidate",
    "HybridRepairResult",
    "LLMSupplier",
    "Scorer",
    "SymbolicSupplier",
    "run_hybrid_repair",
]
