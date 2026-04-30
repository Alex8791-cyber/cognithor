# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""F1 — Suspicion-Score with two-class multipliers (spec v1.4 §7.3.2)."""

from __future__ import annotations

from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.phase2 import (
    DEFAULT_PHASE2_CONFIG,
    Phase2Config,
)
from cognithor.channels.program_synthesis.phase2.suspicion import (
    compute_syntactic_complexity,
    effective_token_count,
)
from cognithor.channels.program_synthesis.search.candidate import (
    InputRef,
    Program,
)


def _g(name: str) -> Program:
    """Build a one-token program: ``<name>(input)``."""
    return Program(
        primitive=name,
        children=(InputRef(),),
        output_type="Grid",
    )


class TestEffectiveTokenCount:
    """Multiplier weighting per token class."""

    def test_high_impact_one_tokener_weighs_three(self) -> None:
        assert effective_token_count(_g("tile")) == 3.0

    def test_structural_abstraction_one_tokener_weighs_one_point_five(self) -> None:
        # Central F1 change: objects() drops from 3.0 to 1.5.
        assert effective_token_count(_g("objects")) == 1.5

    def test_regular_one_tokener_weighs_one(self) -> None:
        assert effective_token_count(_g("recolor")) == 1.0

    def test_input_ref_alone_weighs_zero(self) -> None:
        assert effective_token_count(InputRef()) == 0.0

    def test_nested_program_sums_class_weights(self) -> None:
        # mirror(rotate90(input)) — both High-Impact → 3.0 + 3.0 = 6.0.
        prog = Program(
            primitive="mirror",
            children=(Program("rotate90", (InputRef(),), "Grid"),),
            output_type="Grid",
        )
        assert effective_token_count(prog) == 6.0

    def test_mixed_classes_in_one_tree(self) -> None:
        # filter_objects(connected_components_4(input))
        # = 1.5 (filter_objects) + 1.5 (connected_components_4)
        prog = Program(
            primitive="filter_objects",
            children=(Program("connected_components_4", (InputRef(),), "ObjectSet"),),
            output_type="ObjectSet",
        )
        assert effective_token_count(prog) == 3.0


class TestComputeSyntacticComplexity:
    """Spec v1.4 §7.3.2 — graduated [0, 1] complexity score."""

    def test_objects_alone_lower_complexity_than_tile_alone(self) -> None:
        # The central F1 invariant: ``objects()`` alone now has a
        # *lower* complexity than ``tile()`` alone, so its suspicion
        # for a high partial_score will be greater (more triviality
        # penalty).
        assert compute_syntactic_complexity(_g("objects")) < compute_syntactic_complexity(
            _g("tile")
        )

    def test_recolor_lowest_complexity(self) -> None:
        # Regular < structural-abstraction < high-impact.
        c_recolor = compute_syntactic_complexity(_g("recolor"))
        c_objects = compute_syntactic_complexity(_g("objects"))
        c_tile = compute_syntactic_complexity(_g("tile"))
        assert c_recolor < c_objects < c_tile

    def test_pure_input_ref_has_zero_complexity(self) -> None:
        assert compute_syntactic_complexity(InputRef()) == 0.0

    def test_score_capped_at_one(self) -> None:
        # 12 high-impact tokens stacked would weigh 36 — well over the
        # length budget of 12. Score must clamp to 1.0.
        prog: Program = _g("tile")
        for _ in range(20):
            prog = Program("tile", (prog,), "Grid")
        assert compute_syntactic_complexity(prog) == 1.0

    def test_config_override_scales_through(self) -> None:
        # Doubling the high-impact multiplier in the config doubles the
        # effective token count of a tile-only program.
        config = Phase2Config(
            high_impact_multiplier=6.0,
            structural_abstraction_multiplier=3.0,
            regular_primitive_multiplier=2.0,
        )
        assert effective_token_count(_g("tile"), config=config) == 2.0 * effective_token_count(
            _g("tile"), config=DEFAULT_PHASE2_CONFIG
        )
