# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-2 Track E — MCTS extensions (parallel + restart + diversity).

Three modules of optional complexity layered on top of the
Sprint-1 :class:`MCTSController` (PR #258), each gated by a
config flag:

* :class:`ParallelMCTSDriver` — runs ``N`` independent
  :class:`MCTSController` instances concurrently via
  ``asyncio.gather``; merges the best trajectory across workers.
  Each worker has its own tree (no locking; Sprint-3 will look at
  shared-tree variants when the parallel-correctness payoff
  exceeds the implementation cost).
* :class:`RestartController` — watches the Sprint-1
  :class:`MCTSController.run` outcomes; when the search plateaus
  early in the budget without crossing a configurable score
  threshold, it restarts with a bumped ``c_puct`` (more exploration).
  Spec §5.7.
* :func:`apply_diversity_bonus` — given a list of
  :class:`MCTSActionCandidate` priors and the path-prefixes of
  already-visited subtrees, dampens priors whose path is similar
  to existing visits. Spec §5.6, similarity_metric=
  ``edit_distance``.

All three honour the existing :class:`Phase2Config` surface; the
constants live in the ``mcts_*`` block already added in PR #258
(``mcts_parallelism_workers`` is added here).

Sprint-2 directive acceptance for Track E:
"4 Workers stabil, Restart-Trigger empirisch kalibriert,
Diversity-Bonus messbar reduziert lokale Optima."
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cognithor.channels.program_synthesis.phase2.config import (
    DEFAULT_PHASE2_CONFIG,
    Phase2Config,
)
from cognithor.channels.program_synthesis.phase2.mcts_controller import (
    MCTSActionCandidate,
    MCTSController,
    MCTSResult,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from cognithor.channels.program_synthesis.phase2.datatypes import MCTSNode
    from cognithor.channels.program_synthesis.phase2.mcts_controller import (
        ActionSupplier,
        ValueEstimator,
    )


# ---------------------------------------------------------------------------
# Parallel driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParallelMCTSResult:
    """Aggregate outcome of one :meth:`ParallelMCTSDriver.run` call.

    ``best`` is the worker result with the highest ``best_value``
    (ties broken by deeper trajectory). ``per_worker`` is the full
    list — useful for telemetry on how much agreement / disagreement
    there was across the parallel pool.

    ``workers_used`` reports how many workers the driver actually
    spun up (capped at ``max(1, config.mcts_parallelism_workers)``).
    """

    best: MCTSResult
    per_worker: tuple[MCTSResult, ...]
    workers_used: int


class ParallelMCTSDriver:
    """Run N independent MCTS controllers concurrently; merge best.

    Each worker gets its own freshly-built root and runs to its own
    budget / iteration cap. The driver returns the worker with the
    highest ``best_value``; the others are kept in
    ``ParallelMCTSResult.per_worker`` for telemetry.

    The current implementation is *embarrassingly parallel*: workers
    don't share a tree. Sprint-3 may add a shared-tree mode using
    asyncio locks + virtual-loss; for Sprint-2 the directive only
    requires "4 Workers stabil" — N independent trees deliver that
    with zero risk of cross-worker corruption.

    The ``root_factory`` is a zero-arg callable so each worker gets
    a fresh root (rather than sharing a mutable :class:`MCTSNode`
    across coroutines).
    """

    def __init__(
        self,
        action_supplier: ActionSupplier,
        value_estimator: ValueEstimator,
        *,
        config: Phase2Config = DEFAULT_PHASE2_CONFIG,
    ) -> None:
        self._supplier = action_supplier
        self._estimator = value_estimator
        self._config = config

    async def run(
        self,
        root_factory: Callable[[], MCTSNode],
        *,
        wall_clock_budget_seconds: float,
        max_iterations: int | None = None,
    ) -> ParallelMCTSResult:
        """Spin up workers, gather their results, return the best."""
        n_workers = max(1, self._config.mcts_parallelism_workers)

        async def _run_one() -> MCTSResult:
            controller = MCTSController(
                action_supplier=self._supplier,
                value_estimator=self._estimator,
                config=self._config,
            )
            # MCTSController.run is sync; offload to a thread so
            # multiple workers genuinely run concurrently.
            return await asyncio.to_thread(
                controller.run,
                root_factory(),
                wall_clock_budget_seconds=wall_clock_budget_seconds,
                max_iterations=max_iterations,
            )

        per_worker = await asyncio.gather(*[_run_one() for _ in range(n_workers)])
        best = max(
            per_worker,
            key=lambda r: (r.best_value, len(r.best_path), -r.iterations_completed),
        )
        return ParallelMCTSResult(
            best=best,
            per_worker=tuple(per_worker),
            workers_used=n_workers,
        )


# ---------------------------------------------------------------------------
# Restart controller
# ---------------------------------------------------------------------------


@dataclass
class RestartController:
    """Spec §5.7 — restart MCTS with bumped c_puct on early plateau.

    The controller wraps :class:`MCTSController`. After each ``run``
    it inspects the result: if the search consumed *less* than
    ``mcts_restart_budget_fraction`` of its budget AND
    ``best_value < mcts_restart_score_threshold``, the next ``run``
    is invoked with ``c_puct *= mcts_restart_c_puct_multiplier``
    (achieved by spawning a fresh controller with an overridden
    config — :class:`Phase2Config` is frozen).

    ``run_count`` and ``last_c_puct`` are exposed so callers /
    telemetry can audit how often the restart fired.
    """

    action_supplier: ActionSupplier
    value_estimator: ValueEstimator
    config: Phase2Config = DEFAULT_PHASE2_CONFIG

    run_count: int = 0
    restarts_triggered: int = 0
    last_c_puct: float = 0.0

    _next_c_puct_override: float | None = field(default=None, init=False, repr=False)

    async def run(
        self,
        root: MCTSNode,
        *,
        wall_clock_budget_seconds: float,
        max_iterations: int | None = None,
    ) -> MCTSResult:
        """Run a single MCTS pass; record whether the next pass should restart."""
        c_puct = (
            self._next_c_puct_override
            if self._next_c_puct_override is not None
            else self.config.mcts_c_puct
        )
        cfg = (
            self.config
            if self._next_c_puct_override is None
            else _config_with_c_puct(self.config, c_puct)
        )
        controller = MCTSController(
            action_supplier=self.action_supplier,
            value_estimator=self.value_estimator,
            config=cfg,
        )
        # Sync MCTSController.run — runs in caller's thread; the
        # caller may wrap in asyncio.to_thread if they're in async land.
        result = await asyncio.to_thread(
            controller.run,
            root,
            wall_clock_budget_seconds=wall_clock_budget_seconds,
            max_iterations=max_iterations,
        )
        self.run_count += 1
        self.last_c_puct = c_puct
        # Decide whether to flag a restart for the *next* run.
        budget_fraction_used = result.elapsed_seconds / max(wall_clock_budget_seconds, 1e-9)
        if (
            budget_fraction_used < self.config.mcts_restart_budget_fraction
            and result.best_value < self.config.mcts_restart_score_threshold
        ):
            self._next_c_puct_override = c_puct * self.config.mcts_restart_c_puct_multiplier
            self.restarts_triggered += 1
        else:
            self._next_c_puct_override = None
        return result


def _config_with_c_puct(base: Phase2Config, c_puct: float) -> Phase2Config:
    """Build a new ``Phase2Config`` with only ``mcts_c_puct`` overridden.

    ``Phase2Config`` is frozen, so we use :func:`dataclasses.replace`
    semantics inline (the dataclass exposes all fields by name).
    """
    from dataclasses import replace

    return replace(base, mcts_c_puct=c_puct)


# ---------------------------------------------------------------------------
# Diversity bonus
# ---------------------------------------------------------------------------


def apply_diversity_bonus(
    candidates: Iterable[MCTSActionCandidate],
    visited_paths: Iterable[tuple[str, ...]],
    *,
    config: Phase2Config = DEFAULT_PHASE2_CONFIG,
) -> tuple[MCTSActionCandidate, ...]:
    """Spec §5.6 — dampen priors of candidates similar to visited subtrees.

    For each candidate, compute the *minimum* edit-distance from the
    candidate's primitive name to any token in any visited path (the
    cheapest existing similarity check that captures "we've explored
    this primitive lately"). Candidates with low edit distance get a
    multiplicative penalty:

        prior' = prior · (1 - λ · (1 - d / max_len))

    where ``λ = config.mcts_diversity_bonus_lambda`` and ``max_len``
    normalises the penalty to ``[0, 1]``. Higher d (more dissimilar)
    → smaller penalty → higher prior retention.

    When no paths have been visited the candidates are returned
    unchanged.
    """
    visited = list(visited_paths)
    if not visited:
        return tuple(candidates)
    visited_tokens: set[str] = set()
    for path in visited:
        visited_tokens.update(path)
    if not visited_tokens:
        return tuple(candidates)
    lam = config.mcts_diversity_bonus_lambda
    out: list[MCTSActionCandidate] = []
    for cand in candidates:
        max_len = max(len(cand.primitive), max(len(t) for t in visited_tokens), 1)
        min_dist = min(_edit_distance(cand.primitive, t) for t in visited_tokens)
        # Similarity in [0, 1]: 1.0 = identical, 0.0 = completely different.
        similarity = 1.0 - (min_dist / max_len)
        penalty = lam * similarity
        adjusted_prior = max(0.0, cand.prior * (1.0 - penalty))
        out.append(MCTSActionCandidate(primitive=cand.primitive, prior=adjusted_prior))
    return tuple(out)


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance — small inputs (primitive names ≤ 30 chars)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


__all__ = [
    "ParallelMCTSDriver",
    "ParallelMCTSResult",
    "RestartController",
    "apply_diversity_bonus",
]
