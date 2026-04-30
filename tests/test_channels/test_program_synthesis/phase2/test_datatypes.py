# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Phase-2 datatypes tests (Sprint-1 plan task 2, spec §9)."""

from __future__ import annotations

import pytest

from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.phase2 import (
    FeatureWithConfidence,
    MCTSNode,
    MCTSState,
    MixedPolicy,
    PartitionedBudget,
    Phase2Config,
)

# ---------------------------------------------------------------------------
# FeatureWithConfidence
# ---------------------------------------------------------------------------


class TestFeatureWithConfidence:
    def test_construction_round_trip(self) -> None:
        f = FeatureWithConfidence(name="size_ratio", value=2.0, n_demos=4)
        assert f.name == "size_ratio"
        assert f.value == 2.0
        assert f.n_demos == 4

    def test_is_hashable(self) -> None:
        # Frozen dataclass + immutable value — safe as a dict key.
        f1 = FeatureWithConfidence(name="x", value=1, n_demos=3)
        f2 = FeatureWithConfidence(name="x", value=1, n_demos=3)
        assert hash(f1) == hash(f2)
        assert {f1: "ok"}[f2] == "ok"

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            FeatureWithConfidence(name="", value=1, n_demos=1)

    def test_negative_n_demos_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_demos must be >= 0"):
            FeatureWithConfidence(name="x", value=1, n_demos=-1)

    def test_effective_confidence_dampens_with_n(self) -> None:
        # Spec §4.4: at n=n0=4, factor = 0.5.
        f = FeatureWithConfidence(name="x", value=1, n_demos=4)
        assert f.effective_confidence() == 0.5

    def test_effective_confidence_zero_at_no_demos(self) -> None:
        f = FeatureWithConfidence(name="x", value=1, n_demos=0)
        assert f.effective_confidence() == 0.0

    def test_effective_confidence_uses_config_n0(self) -> None:
        config = Phase2Config(sample_size_dampening_n0=8)
        f = FeatureWithConfidence(name="x", value=1, n_demos=8)
        assert f.effective_confidence(config=config) == 0.5

    def test_base_confidence_scales_through(self) -> None:
        f = FeatureWithConfidence(name="x", value=1, n_demos=4)
        assert f.effective_confidence(base_confidence=0.6) == 0.3


# ---------------------------------------------------------------------------
# PartitionedBudget
# ---------------------------------------------------------------------------


class TestPartitionedBudget:
    def test_spec_default_sums_to_one(self) -> None:
        b = PartitionedBudget.from_spec_default()
        assert b.pre_processing == 0.07
        assert b.mcts == 0.70
        assert b.refiner == 0.18
        assert b.cegis == 0.05

    def test_construction_rejects_non_unit_sum(self) -> None:
        with pytest.raises(ValueError, match="must sum to 1.0"):
            PartitionedBudget(pre_processing=0.10, mcts=0.70, refiner=0.10, cegis=0.05)

    def test_construction_rejects_negative_fraction(self) -> None:
        with pytest.raises(ValueError, match="must be >= 0"):
            PartitionedBudget(
                pre_processing=-0.05,
                mcts=0.85,
                refiner=0.15,
                cegis=0.05,
            )

    def test_floating_point_slack_tolerated(self) -> None:
        # Sum = 1.0 ± 1e-9 — must construct cleanly.
        PartitionedBudget(
            pre_processing=0.07,
            mcts=0.70 + 1e-12,
            refiner=0.18,
            cegis=0.05,
        )

    def test_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        b = PartitionedBudget.from_spec_default()
        with pytest.raises(FrozenInstanceError):
            b.mcts = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MixedPolicy
# ---------------------------------------------------------------------------


class TestMixedPolicy:
    def test_from_dict_orders_keys(self) -> None:
        p = MixedPolicy.from_dict({"b": 0.5, "a": 0.3, "c": 0.2}, alpha=0.5)
        # Sorted by key for determinism.
        assert p.primitive_scores == (("a", 0.3), ("b", 0.5), ("c", 0.2))
        assert p.alpha == 0.5

    def test_round_trip_through_dict(self) -> None:
        original = {"a": 0.4, "b": 0.6}
        p = MixedPolicy.from_dict(original, alpha=0.7)
        assert p.as_dict() == original

    def test_is_hashable(self) -> None:
        p1 = MixedPolicy.from_dict({"a": 1.0}, alpha=0.5)
        p2 = MixedPolicy.from_dict({"a": 1.0}, alpha=0.5)
        assert hash(p1) == hash(p2)


# ---------------------------------------------------------------------------
# MCTSNode + MCTSState
# ---------------------------------------------------------------------------


class TestMCTSNode:
    def test_default_visit_count_zero(self) -> None:
        n = MCTSNode(primitive="rotate90")
        assert n.visit_count == 0
        assert n.total_value == 0.0
        assert n.mean_value == 0.0

    def test_record_visit_accumulates(self) -> None:
        n = MCTSNode(primitive="rotate90")
        n.record_visit(0.5)
        n.record_visit(0.7)
        assert n.visit_count == 2
        assert n.total_value == 1.2
        assert abs(n.mean_value - 0.6) < 1e-9

    def test_puct_score_blends_value_and_exploration(self) -> None:
        n = MCTSNode(primitive="rotate90", prior=0.4)
        # No visits yet → mean_value=0; exploration term dominates.
        score = n.puct_score(c_puct=3.5, parent_visit_count=10)
        # 3.5 · 0.4 · sqrt(10) / 1 ≈ 4.427
        assert 4.4 < score < 4.5

    def test_puct_handles_zero_parent_visits(self) -> None:
        n = MCTSNode(primitive="x", prior=0.6)
        # Degenerate case: parent_visit_count=0 → sqrt(0)=0, fall
        # back to the prior alone (the spec rationale: a freshly-
        # spawned root has no exploration bonus to compute).
        score = n.puct_score(c_puct=3.5, parent_visit_count=0)
        assert score == 0.6  # prior, no mean_value

    def test_children_dict_is_per_instance(self) -> None:
        # Sanity: dataclass field(default_factory=dict) avoids the
        # mutable-default-argument trap.
        a = MCTSNode(primitive="x")
        b = MCTSNode(primitive="y")
        a.children["k"] = b
        assert "k" not in MCTSNode(primitive="z").children


class TestMCTSState:
    def test_step_increments_iteration(self) -> None:
        root = MCTSNode(primitive="root")
        s = MCTSState(root=root, budget=PartitionedBudget.from_spec_default())
        assert s.iteration == 0
        s.step()
        s.step()
        assert s.iteration == 2

    def test_best_so_far_starts_none(self) -> None:
        root = MCTSNode(primitive="root")
        s = MCTSState(root=root, budget=PartitionedBudget.from_spec_default())
        assert s.best_so_far is None
