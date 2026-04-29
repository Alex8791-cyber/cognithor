# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""PGE-Trinity adapter — public synthesis entrypoint (spec §12).

The Planner asks "is this synthesizable?" via :func:`is_synthesizable`
and, if yes, builds a :class:`SynthesisRequest` that the Gatekeeper
validates and routes through this channel.

Phase-1 wiring of the dataflow (Happy Path):

1. Cache lookup via :class:`PSECache`.
2. NumPy fast-path via :class:`NumpySolverBridge`.
3. Bottom-up enumerative search.
4. Cache write on a cacheable status.
5. Result returned.

The channel is sandbox-strategy aware: it uses
:func:`select_sandbox_strategy` to pick the right :class:`Executor`
implementation, which the search engine and the equivalence pruner
share.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from cognithor.channels.program_synthesis.core.types import (
    Budget,
    SynthesisResult,
    SynthesisStatus,
    TaskSpec,
)
from cognithor.channels.program_synthesis.integration.numpy_solver_bridge import (
    NumpySolverBridge,
)
from cognithor.channels.program_synthesis.integration.state_graph_bridge import (
    StateGraphBridge,
)
from cognithor.channels.program_synthesis.integration.tactical_memory import (
    PSECache,
)
from cognithor.channels.program_synthesis.sandbox.strategies import (
    _BaseStrategy,
    select_sandbox_strategy,
)
from cognithor.channels.program_synthesis.search.enumerative import (
    EnumerativeSearch,
)

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Auto-classification — spec §12.1
# ---------------------------------------------------------------------------


def is_synthesizable(task: dict[str, Any]) -> bool:
    """Heuristic the Planner uses to decide whether a task should route to PSE.

    Phase 1 is rule-based, not ML — the spec calls out that an ML
    classifier is a Phase-2 concern. The rules are:

    - Has at least 2 demo ``examples`` (single-demo tasks are too
      under-specified to enumerate against).
    - Every example has both ``input`` and ``output`` keys.
    - The first example's ``input`` looks like a 2-D grid (list of
      lists of ints).
    """
    if not isinstance(task, dict):
        return False
    examples = task.get("examples")
    if not isinstance(examples, list) or len(examples) < 2:
        return False
    for ex in examples:
        if not isinstance(ex, dict):
            return False
        if "input" not in ex or "output" not in ex:
            return False
    return _looks_like_grid(examples[0]["input"])


def _looks_like_grid(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    first = value[0]
    if not isinstance(first, list) or not first:
        return False
    return all(isinstance(c, int) for c in first)


# ---------------------------------------------------------------------------
# SynthesisRequest — public API surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SynthesisRequest:
    """A single synthesis call's input.

    The Planner builds this from a parsed task; the Gatekeeper performs
    capability + size validation before forwarding to the channel.
    """

    spec: TaskSpec
    # ``Budget`` is itself a frozen dataclass so a default-factory keeps
    # ruff happy and matches the pattern the rest of PSE uses.
    budget: Budget = field(default_factory=Budget)
    # Optional SGN hints — keyed by primitive name (see spec §15).
    sgn_hints: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Channel
# ---------------------------------------------------------------------------


class ProgramSynthesisChannel:
    """Synchronous PSE channel.

    Wires together the cache, the NumPy fast-path, the SGN bridge, and
    the enumerative search engine. All collaborators are
    dependency-injectable so tests can construct a minimal channel
    without booting Cognithor.
    """

    def __init__(
        self,
        *,
        cache: PSECache | None = None,
        numpy_bridge: NumpySolverBridge | None = None,
        sgn_bridge: StateGraphBridge | None = None,
        sandbox_strategy: _BaseStrategy | None = None,
        engine: EnumerativeSearch | None = None,
    ) -> None:
        self._cache = cache if cache is not None else PSECache()
        self._numpy = numpy_bridge if numpy_bridge is not None else NumpySolverBridge()
        self._sgn = sgn_bridge if sgn_bridge is not None else StateGraphBridge()
        # Pick a sandbox strategy unless the caller provides one.
        self._sandbox = (
            sandbox_strategy
            if sandbox_strategy is not None
            else select_sandbox_strategy(emit_warning=False)
        )
        # The engine reuses the sandbox strategy as its executor — same
        # Executor protocol, single point of policy enforcement.
        self._engine = engine if engine is not None else EnumerativeSearch(executor=self._sandbox)

    # -- Public API --------------------------------------------------

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        """End-to-end synthesis: cache → fast-path → enumerator → cache."""
        spec = self._apply_sgn(request)
        budget = request.budget

        # 1. Cache lookup.
        if budget.cache_lookup:
            entry = self._cache.get(spec, budget)
            if entry is not None and entry.status == SynthesisStatus.SUCCESS:
                LOG.info("PSE cache hit for %s", entry.spec_hash)
                return self._cache_entry_to_result(entry)

        # 2. NumPy fast-path.
        if self._numpy.is_available():
            t0 = time.monotonic()
            fast = self._numpy.try_solve(spec)
            if fast is not None:
                elapsed = time.monotonic() - t0
                # Re-stamp the fast-path's cost_seconds with our wall-clock
                # so callers see a non-zero (informative) duration.
                stamped = SynthesisResult(
                    status=fast.status,
                    program=fast.program,
                    score=fast.score,
                    confidence=fast.confidence,
                    cost_seconds=elapsed,
                    cost_candidates=fast.cost_candidates,
                    verifier_trace=fast.verifier_trace,
                    cache_hit=False,
                    annotations=fast.annotations,
                )
                self._cache.put(spec, budget, stamped)
                return stamped

        # 3. Enumerative search.
        result = self._engine.search(spec, budget)

        # 4. Cache write — the tactical-memory cache filters
        # non-cacheable statuses internally.
        self._cache.put(spec, budget, result)
        return result

    # -- Internals ---------------------------------------------------

    def _apply_sgn(self, request: SynthesisRequest) -> TaskSpec:
        if not request.sgn_hints:
            return request.spec
        return self._sgn.annotate(request.spec, request.sgn_hints)

    @staticmethod
    def _cache_entry_to_result(entry: object) -> SynthesisResult:
        """Materialise a cache entry as a SynthesisResult.

        The cached form stores ``program_source`` (a string), not the
        live :class:`Program` tree — Phase 1 returns the source plus
        ``cache_hit=True`` so the caller can decide whether to re-run
        the search to recover the live tree.
        """
        return SynthesisResult(
            status=entry.status,  # type: ignore[attr-defined]
            program=None,
            score=entry.score,  # type: ignore[attr-defined]
            confidence=entry.confidence,  # type: ignore[attr-defined]
            cost_seconds=entry.cost_seconds,  # type: ignore[attr-defined]
            cost_candidates=0,
            verifier_trace=(),
            cache_hit=True,
            annotations=(
                ("cached_program_source", entry.program_source or ""),  # type: ignore[attr-defined]
                ("cached_program_hash", entry.program_hash or ""),  # type: ignore[attr-defined]
            ),
        )


__all__ = [
    "ProgramSynthesisChannel",
    "SynthesisRequest",
    "is_synthesizable",
]
