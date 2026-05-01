# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Module B — MCTS controller (spec §5, Sprint-1 plan task 7).

Sprint-1 ships a clean, single-threaded PUCT controller that
satisfies the spec §5 contract:

* **Selection** — descend from root via PUCT (``MCTSNode.puct_score``)
  until reaching a leaf or the depth limit.
* **Expansion** — ask the injected ``ActionSupplier`` for the legal
  expansions at the leaf; convert them to child :class:`MCTSNode`
  with the supplier's prior probabilities.
* **Evaluation** — call the injected ``ValueEstimator`` to score
  the leaf (or a freshly-expanded child).
* **Backpropagation** — walk back to the root, calling
  :meth:`MCTSNode.record_visit` with the leaf's value.

Around the four phases the controller supports:

* **Anytime termination** — caller passes a wall-clock budget and an
  optional max-iterations cap; the loop returns the best-known
  trajectory whenever either expires (spec §5.4).
* **Virtual loss** — opt-in negative offset added to ``total_value``
  during selection so a future parallel-worker upgrade doesn't
  re-pile on the same subtree (spec §5.5).
* **Fallback controller** — :class:`FallbackGuard` watches the
  best-score trajectory + the mean node depth; when the search
  plateaus or stays too shallow, it signals the caller to switch
  to a simpler controller (spec §5.10).

The Sprint-1 controller is single-threaded; the parallel ``workers``
upgrade (configurable via ``Phase2Config.mcts_virtual_loss > 0``) is
a Sprint-2 PR that reuses every interface defined here. Production
wiring will inject a Phase-1 ``EnumerativeSearch`` instance as the
``ActionSupplier`` and a Phase-2 verifier as the ``ValueEstimator``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from cognithor.channels.program_synthesis.phase2.config import (
    DEFAULT_PHASE2_CONFIG,
    Phase2Config,
)
from cognithor.channels.program_synthesis.phase2.datatypes import MCTSNode

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MCTSActionCandidate:
    """One legal expansion at a node — what the supplier returns.

    ``primitive`` is the action label the parent edge carries
    (mirrors :attr:`MCTSNode.primitive`). ``prior`` is the
    probability the mixer assigned this action; the controller
    feeds it into PUCT directly without re-normalisation (the
    supplier is expected to hand back a sensible distribution).
    """

    primitive: str
    prior: float


@dataclass(frozen=True)
class MCTSResult:
    """Outcome of one :meth:`MCTSController.run` call.

    ``best_node`` is the highest-mean-value descendant of the root
    found across all iterations (or the root itself when no
    expansion happened — degenerate but well-defined).

    ``best_value`` is the value the supplier reported for
    ``best_node``'s primitive path on the iteration where it
    became best.

    ``best_path`` lists the primitives along the trajectory from
    the root to ``best_node`` (root excluded). Used by the engine
    to lift the trajectory into a Program tree.

    ``terminated_by`` reports why the loop stopped:
    ``"budget"`` / ``"iterations"`` / ``"fallback"`` / ``"no_actions"``.

    ``iterations_completed`` and ``elapsed_seconds`` are anytime
    metrics — the caller's telemetry sink reads them.
    """

    best_node: MCTSNode
    best_value: float
    best_path: tuple[str, ...]
    terminated_by: str
    iterations_completed: int
    elapsed_seconds: float
    fallback_signalled: bool = False


# ---------------------------------------------------------------------------
# Strategy callables
# ---------------------------------------------------------------------------


# Action supplier: given a node, return its legal expansions plus priors.
# The controller calls this once per leaf — repeated calls on the same
# node are the supplier's responsibility to memoise.
ActionSupplier = Callable[[MCTSNode], Iterable[MCTSActionCandidate]]

# Value estimator: given a leaf node + the path of primitives leading
# from the root, return a scalar reward in any range (the controller
# only uses the value relatively; absolute scale is the caller's
# concern). 0 ≤ value ≤ 1 is recommended for the verifier integration.
ValueEstimator = Callable[[MCTSNode, tuple[str, ...]], float]


# ---------------------------------------------------------------------------
# Fallback guard
# ---------------------------------------------------------------------------


@dataclass
class FallbackGuard:
    """Spec §5.10 — watch the search for plateau + shallow-tree signals.

    The guard has two checks, both gated by a warmup window so the
    controller has time to populate the tree before any verdict is
    rendered:

    1. **Plateau** — over the last ``plateau_iters`` iterations the
       best score moved by less than ``plateau_delta``. Indicates
       the search has converged on a local minimum and the caller
       should switch to a simpler controller (or hand off to CEGIS).
    2. **Shallow tree** — the mean depth of the visited frontier is
       below ``min_node_depth_mean``. Indicates the prior was too
       narrow and the search is wasting iterations near the root.

    Both signals are advisory — the controller still returns its
    best-known trajectory; the *caller* decides what to do with the
    fallback flag. (In production: switch to the v1.2 D6
    "fallback controller" that ignores the LLM prior entirely.)
    """

    warmup_iters: int
    plateau_iters: int
    plateau_delta: float
    min_node_depth_mean: float

    _score_history: list[float] = field(default_factory=list)
    _depth_sum: float = 0.0
    _depth_count: int = 0

    def record(self, *, best_value: float, leaf_depth: int) -> None:
        self._score_history.append(best_value)
        self._depth_sum += leaf_depth
        self._depth_count += 1

    def should_fall_back(self, *, iteration: int) -> bool:
        if iteration < self.warmup_iters:
            return False
        if self._plateau_signal():
            return True
        return self._shallow_tree_signal()

    def _plateau_signal(self) -> bool:
        if len(self._score_history) < self.plateau_iters + 1:
            return False
        window = self._score_history[-self.plateau_iters - 1 :]
        spread = max(window) - min(window)
        return spread < self.plateau_delta

    def _shallow_tree_signal(self) -> bool:
        if self._depth_count == 0:
            return False
        return (self._depth_sum / self._depth_count) < self.min_node_depth_mean


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class MCTSController:
    """Single-threaded PUCT controller — spec §5 Sprint-1 minimal.

    Strategies (action supplier + value estimator) are
    dependency-injected; the controller knows nothing about the DSL
    or the verifier. The :class:`MCTSNode` tree it builds satisfies
    the existing ``MCTSNode.puct_score`` contract.

    Telemetry-relevant invariants:

    * ``MCTSResult.iterations_completed`` is monotone non-decreasing.
    * Every iteration either expands at least one child or signals
      ``"no_actions"`` and exits early — the loop never spins
      without making progress.
    * ``MCTSResult.fallback_signalled`` reflects the
      :class:`FallbackGuard` verdict at termination time.
    """

    def __init__(
        self,
        action_supplier: ActionSupplier,
        value_estimator: ValueEstimator,
        *,
        config: Phase2Config = DEFAULT_PHASE2_CONFIG,
        fallback: FallbackGuard | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._supplier = action_supplier
        self._estimator = value_estimator
        self._config = config
        self._fallback = fallback or _default_fallback_from(config)
        self._clock = clock

    def run(
        self,
        root: MCTSNode,
        *,
        wall_clock_budget_seconds: float,
        max_iterations: int | None = None,
    ) -> MCTSResult:
        """Run the PUCT loop until the budget / iteration cap fires."""
        if wall_clock_budget_seconds <= 0:
            raise ValueError(
                f"wall_clock_budget_seconds must be > 0; got {wall_clock_budget_seconds!r}"
            )
        cap = (
            max_iterations
            if max_iterations is not None
            else (self._config.mcts_max_iterations or None)
        )
        start = self._clock()
        deadline = start + wall_clock_budget_seconds

        best_node = root
        best_value = float("-inf")
        best_path: tuple[str, ...] = ()
        terminated_by = "budget"
        iterations = 0
        fallback_signalled = False

        while True:
            now = self._clock()
            if now >= deadline:
                terminated_by = "budget"
                break
            if cap is not None and iterations >= cap:
                terminated_by = "iterations"
                break

            iterations += 1

            # ── Selection ────────────────────────────────────────
            leaf, leaf_path = self._select(root)

            # ── Expansion ────────────────────────────────────────
            expanded = self._expand(leaf)
            if not expanded and leaf is root and not root.children:
                # Empty action space at root — no work possible.
                terminated_by = "no_actions"
                break

            # ── Evaluation ───────────────────────────────────────
            target = expanded if expanded is not None else leaf
            target_path = (*leaf_path, target.primitive) if expanded is not None else leaf_path
            value = self._estimator(target, target_path)

            # ── Backpropagation ──────────────────────────────────
            self._backpropagate(target, value)

            # ── Best-trajectory bookkeeping ───────────────────────
            if value > best_value:
                best_value = value
                best_node = target
                best_path = target_path

            # ── Fallback guard ───────────────────────────────────
            self._fallback.record(best_value=best_value, leaf_depth=len(target_path))
            if self._fallback.should_fall_back(iteration=iterations):
                fallback_signalled = True
                terminated_by = "fallback"
                break

        return MCTSResult(
            best_node=best_node,
            best_value=best_value if best_value != float("-inf") else 0.0,
            best_path=best_path,
            terminated_by=terminated_by,
            iterations_completed=iterations,
            elapsed_seconds=self._clock() - start,
            fallback_signalled=fallback_signalled,
        )

    # ── Selection / expansion / backprop ─────────────────────────────

    def _select(self, root: MCTSNode) -> tuple[MCTSNode, tuple[str, ...]]:
        """Descend via PUCT until reaching a leaf (no children)."""
        node = root
        path: list[str] = []
        while node.children:
            best_child = self._pick_best_child(node)
            path.append(best_child.primitive)
            node = best_child
        return node, tuple(path)

    def _pick_best_child(self, parent: MCTSNode) -> MCTSNode:
        """PUCT-pick the child with the highest score (with virtual-loss adjust)."""
        c_puct = self._config.mcts_c_puct
        n_parent = parent.visit_count
        vloss = self._config.mcts_virtual_loss
        if vloss > 0:
            return max(
                parent.children.values(),
                key=lambda c: _puct_with_virtual_loss(c, c_puct, n_parent, vloss),
            )
        return max(
            parent.children.values(),
            key=lambda c: c.puct_score(c_puct=c_puct, parent_visit_count=n_parent),
        )

    def _expand(self, leaf: MCTSNode) -> MCTSNode | None:
        """Ask the supplier for actions, materialise children, return one to evaluate.

        Returns the first newly-added child for evaluation. When the
        leaf already has children (re-visit), returns ``None`` so the
        caller evaluates the leaf itself rather than re-expanding.
        """
        if leaf.children:
            return None
        candidates = list(self._supplier(leaf))
        if not candidates:
            return None
        for cand in candidates:
            child = MCTSNode(primitive=cand.primitive, prior=cand.prior, parent=leaf)
            leaf.children[cand.primitive] = child
        # Pick the child with the highest prior to evaluate first —
        # cheap heuristic that biases evaluation toward what the
        # supplier already thinks is plausible.
        first_eval = max(leaf.children.values(), key=lambda c: c.prior)
        return first_eval

    def _backpropagate(self, leaf: MCTSNode, value: float) -> None:
        """Walk leaf → root, recording the visit on every node."""
        node: MCTSNode | None = leaf
        while node is not None:
            node.record_visit(value)
            node = node.parent


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _puct_with_virtual_loss(
    child: MCTSNode,
    c_puct: float,
    parent_visit_count: int,
    virtual_loss: float,
) -> float:
    """PUCT with a virtual-loss penalty — used during selection only.

    The penalty is the spec §5.5 mechanism for parallel workers: each
    worker subtracts ``virtual_loss`` from a node's running mean while
    it's "in flight", so a second worker picking PUCT-best is biased
    away from the same subtree. Sprint-1 is single-threaded so the
    penalty has no operational effect; tests assert it nonetheless
    converges to the same answer as the no-virtual-loss path.
    """
    base = child.puct_score(c_puct=c_puct, parent_visit_count=parent_visit_count)
    if child.visit_count == 0:
        # No virtual loss applies until the node has been visited; the
        # PUCT exploration term already favours unvisited children.
        return base
    return base - virtual_loss / max(child.visit_count, 1)


def _default_fallback_from(config: Phase2Config) -> FallbackGuard:
    return FallbackGuard(
        warmup_iters=config.mcts_fallback_warmup_iters,
        plateau_iters=config.mcts_fallback_plateau_iters,
        plateau_delta=config.mcts_fallback_plateau_delta,
        min_node_depth_mean=config.mcts_fallback_min_node_depth_mean,
    )


__all__ = [
    "ActionSupplier",
    "FallbackGuard",
    "MCTSActionCandidate",
    "MCTSController",
    "MCTSResult",
    "ValueEstimator",
]
