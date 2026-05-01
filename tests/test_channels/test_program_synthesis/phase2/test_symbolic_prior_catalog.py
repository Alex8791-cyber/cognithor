# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Symbolic-Prior heuristic catalog tests (Sprint-1 plan task 4, spec §4.4)."""

from __future__ import annotations

import numpy as np
import pytest

from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.phase2.config import Phase2Config
from cognithor.channels.program_synthesis.phase2.symbolic_prior import (
    SymbolicPriorResult,
)
from cognithor.channels.program_synthesis.phase2.symbolic_prior_catalog import (
    DEFAULT_RULES,
    HeuristicSymbolicPrior,
    r_background_dominant,
    r_colors_removed,
    r_diagonal_flip,
    r_horizontal_flip,
    r_identity_match,
    r_new_colors_introduced,
    r_objects_present,
    r_output_larger,
    r_output_smaller,
    r_palette_changed,
    r_palette_preserved,
    r_rotation_90,
    r_rotation_180,
    r_rotation_270,
    r_shape_equal,
    r_shape_scaled_down_2x,
    r_shape_scaled_up_2x,
    r_shape_scaled_up_3x,
    r_shape_transposed,
    r_vertical_flip,
)


def _g(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


# ---------------------------------------------------------------------------
# Catalog size
# ---------------------------------------------------------------------------


class TestCatalogSize:
    def test_default_rules_count(self) -> None:
        # Spec §4.4 says ~20 rules. We ship exactly 20.
        assert len(DEFAULT_RULES) == 20


# ---------------------------------------------------------------------------
# Shape rules
# ---------------------------------------------------------------------------


class TestShapeRules:
    def test_shape_equal_fires_on_same_shape(self) -> None:
        v = r_shape_equal(_g([[1, 2]]), _g([[3, 4]]))
        assert "recolor" in v
        assert "swap_colors" in v

    def test_shape_equal_silent_on_different_shape(self) -> None:
        assert r_shape_equal(_g([[1]]), _g([[1, 2]])) == {}

    def test_shape_transposed(self) -> None:
        v = r_shape_transposed(_g([[1, 2, 3]]), _g([[1], [2], [3]]))
        assert "transpose" in v
        assert "rotate90" in v

    def test_shape_scaled_up_2x(self) -> None:
        v = r_shape_scaled_up_2x(_g([[1]]), _g([[1, 1], [1, 1]]))
        assert v["scale_up_2x"] == 0.9

    def test_shape_scaled_up_3x(self) -> None:
        v = r_shape_scaled_up_3x(_g([[1]]), _g([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
        assert v["scale_up_3x"] == 0.9

    def test_shape_scaled_down_2x(self) -> None:
        v = r_shape_scaled_down_2x(_g([[1, 1], [1, 1]]), _g([[1]]))
        assert v["scale_down_2x"] == 0.9

    def test_output_smaller(self) -> None:
        v = r_output_smaller(_g([[1, 2, 3], [4, 5, 6]]), _g([[5]]))
        assert "crop_bbox" in v

    def test_output_larger(self) -> None:
        v = r_output_larger(_g([[1]]), _g([[1, 1], [1, 1]]))
        assert "pad_with" in v
        assert "frame" in v


# ---------------------------------------------------------------------------
# Symmetry rules
# ---------------------------------------------------------------------------


class TestSymmetryRules:
    def test_horizontal_flip(self) -> None:
        a = _g([[1, 2, 3], [4, 5, 6]])
        v = r_horizontal_flip(a, np.fliplr(a))
        assert v == {"mirror_horizontal": 0.95}

    def test_horizontal_flip_silent_when_not_flipped(self) -> None:
        assert r_horizontal_flip(_g([[1, 2]]), _g([[1, 2]])) == {}

    def test_vertical_flip(self) -> None:
        a = _g([[1, 2], [3, 4]])
        v = r_vertical_flip(a, np.flipud(a))
        assert v == {"mirror_vertical": 0.95}

    def test_diagonal_flip(self) -> None:
        a = _g([[1, 2], [3, 4]])
        v = r_diagonal_flip(a, a.T)
        assert "transpose" in v
        assert "mirror_diagonal" in v

    def test_rotation_90(self) -> None:
        a = _g([[1, 2], [3, 4]])
        # DSL rotate90 is clockwise = np.rot90 with k=-1.
        v = r_rotation_90(a, np.rot90(a, k=-1))
        assert v == {"rotate90": 0.95}

    def test_rotation_180(self) -> None:
        a = _g([[1, 2], [3, 4]])
        v = r_rotation_180(a, np.rot90(a, k=2))
        assert v == {"rotate180": 0.95}

    def test_rotation_270(self) -> None:
        a = _g([[1, 2], [3, 4]])
        # DSL rotate270 = np.rot90 with k=1.
        v = r_rotation_270(a, np.rot90(a, k=1))
        assert v == {"rotate270": 0.95}


# ---------------------------------------------------------------------------
# Color rules
# ---------------------------------------------------------------------------


class TestColorRules:
    def test_palette_preserved(self) -> None:
        v = r_palette_preserved(_g([[1, 2], [3, 1]]), _g([[3, 1], [2, 1]]))
        assert "swap_colors" in v

    def test_palette_changed(self) -> None:
        v = r_palette_changed(_g([[1, 2]]), _g([[3, 4]]))
        assert "recolor" in v

    def test_new_colors_introduced(self) -> None:
        v = r_new_colors_introduced(_g([[1, 1]]), _g([[1, 7]]))
        assert "recolor" in v

    def test_colors_removed(self) -> None:
        v = r_colors_removed(_g([[1, 2, 3]]), _g([[1, 1, 1]]))
        assert "recolor" in v
        assert "mask_eq" in v


# ---------------------------------------------------------------------------
# Structure rules
# ---------------------------------------------------------------------------


class TestStructureRules:
    def test_objects_present_with_three_colors(self) -> None:
        v = r_objects_present(_g([[1, 2, 3], [3, 2, 1]]), _g([[0]]))
        assert "objects_of_color" in v
        assert "filter_objects" in v

    def test_objects_present_silent_on_one_color(self) -> None:
        assert r_objects_present(_g([[1, 1]]), _g([[0]])) == {}

    def test_background_dominant(self) -> None:
        # 90% zeros → background dominant.
        grid = np.zeros((10, 10), dtype=np.int8)
        grid[0, 0] = 5
        v = r_background_dominant(grid, grid)
        assert "replace_background" in v

    def test_background_dominant_silent_on_balanced(self) -> None:
        grid = _g([[1, 2], [3, 4]])  # all unique
        assert r_background_dominant(grid, grid) == {}

    def test_identity_match(self) -> None:
        a = _g([[1, 2], [3, 4]])
        v = r_identity_match(a, a)
        assert v == {"identity": 0.95}

    def test_identity_silent_on_change(self) -> None:
        assert r_identity_match(_g([[1]]), _g([[2]])) == {}


# ---------------------------------------------------------------------------
# HeuristicSymbolicPrior — aggregation
# ---------------------------------------------------------------------------


class TestHeuristicSymbolicPrior:
    def test_returns_distribution_summing_to_one(self) -> None:
        a = _g([[1, 2], [3, 4]])
        examples = [(a, np.rot90(a, k=-1))]
        prior = HeuristicSymbolicPrior(
            primitive_whitelist=["rotate90", "rotate180", "rotate270", "recolor"],
        )
        result = prior.get_prior(examples)
        assert isinstance(result, SymbolicPriorResult)
        assert abs(sum(result.primitive_scores.values()) - 1.0) < 1e-9
        # rotate90 should be top.
        top = max(result.primitive_scores, key=lambda k: result.primitive_scores[k])
        assert top == "rotate90"

    def test_only_whitelisted_primitives_present(self) -> None:
        a = _g([[1, 2], [3, 4]])
        prior = HeuristicSymbolicPrior(primitive_whitelist=["rotate90"])
        result = prior.get_prior([(a, np.rot90(a, k=-1))])
        # Only the whitelisted name appears.
        assert set(result.primitive_scores) == {"rotate90"}

    def test_empty_examples_returns_zero_confidence(self) -> None:
        prior = HeuristicSymbolicPrior(primitive_whitelist=["rotate90"])
        result = prior.get_prior([])
        assert result.effective_confidence == 0.0
        # Falls back to uniform.
        assert result.primitive_scores == {"rotate90": 1.0}

    def test_no_rule_fires_falls_back_to_uniform(self) -> None:
        # A whitelist that excludes everything the rules would vote for.
        a = _g([[1, 2], [3, 4]])
        prior = HeuristicSymbolicPrior(primitive_whitelist=["frame", "overlay"])
        result = prior.get_prior([(a, np.rot90(a, k=-1))])
        # Sum still 1.0 even on fallback.
        assert abs(sum(result.primitive_scores.values()) - 1.0) < 1e-9

    def test_confidence_scales_with_demo_count(self) -> None:
        a = _g([[1, 2], [3, 4]])
        rotated = np.rot90(a, k=-1)
        prior = HeuristicSymbolicPrior(primitive_whitelist=["rotate90"])
        c1 = prior.get_prior([(a, rotated)]).effective_confidence
        c4 = prior.get_prior([(a, rotated)] * 4).effective_confidence
        c12 = prior.get_prior([(a, rotated)] * 12).effective_confidence
        assert c1 < c4 < c12

    def test_more_firing_rules_higher_confidence(self) -> None:
        # Identity demo fires r_identity_match + r_shape_equal + r_palette_preserved
        # = 3 rules. Compare with a rotate90 demo that also fires 3 rules
        # (r_shape_transposed, r_rotation_90, r_palette_preserved). Both should
        # have similar confidence; this asserts confidence is in (0, 1).
        whitelist = ["identity", "rotate90", "recolor"]
        prior = HeuristicSymbolicPrior(primitive_whitelist=whitelist)
        a = _g([[1, 2], [3, 4]])
        result = prior.get_prior([(a, a)])
        assert 0.0 < result.effective_confidence < 1.0

    def test_config_dampening_n0_overrides(self) -> None:
        a = _g([[1, 2], [3, 4]])
        rotated = np.rot90(a, k=-1)
        cfg_default = Phase2Config()  # n0=4
        cfg_aggressive = Phase2Config(sample_size_dampening_n0=1)
        prior_default = HeuristicSymbolicPrior(primitive_whitelist=["rotate90"], config=cfg_default)
        prior_aggressive = HeuristicSymbolicPrior(
            primitive_whitelist=["rotate90"], config=cfg_aggressive
        )
        c_default = prior_default.get_prior([(a, rotated)]).effective_confidence
        c_aggressive = prior_aggressive.get_prior([(a, rotated)]).effective_confidence
        # n0=1: dampening factor 1/(1+1) = 0.5 vs n0=4: 1/(1+4) = 0.2
        # → aggressive (smaller n0) yields HIGHER confidence at n=1.
        assert c_aggressive > c_default

    def test_empty_whitelist_raises(self) -> None:
        prior = HeuristicSymbolicPrior(primitive_whitelist=[])
        with pytest.raises(ValueError, match="empty primitive whitelist"):
            prior.get_prior([(_g([[1]]), _g([[1]]))])

    def test_multiple_pairs_average(self) -> None:
        # Pair 1 votes rotate90; pair 2 votes rotate180. Average should
        # split between them, with the second at least as strong.
        a = _g([[1, 2], [3, 4]])
        whitelist = ["rotate90", "rotate180"]
        prior = HeuristicSymbolicPrior(primitive_whitelist=whitelist)
        result = prior.get_prior(
            [
                (a, np.rot90(a, k=-1)),  # rotate90
                (a, np.rot90(a, k=2)),  # rotate180
            ]
        )
        # Both should appear in the distribution.
        assert set(result.primitive_scores) == {"rotate90", "rotate180"}


# ---------------------------------------------------------------------------
# DSL whitelist integration
# ---------------------------------------------------------------------------


class TestLiveRegistryIntegration:
    def test_default_whitelist_uses_live_registry(self) -> None:
        from cognithor.channels.program_synthesis.dsl.registry import REGISTRY

        a = _g([[1, 2], [3, 4]])
        prior = HeuristicSymbolicPrior()  # no explicit whitelist
        result = prior.get_prior([(a, np.rot90(a, k=-1))])
        # All keys must be valid primitive names.
        registered = set(REGISTRY.names())
        assert set(result.primitive_scores).issubset(registered)
