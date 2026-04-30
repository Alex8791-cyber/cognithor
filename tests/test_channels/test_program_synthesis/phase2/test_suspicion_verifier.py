# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""F1 — compute_suspicion ordering invariants (spec v1.4 §7.3.2).

Spec rationale: ``objects()`` alone is more suspect than ``tile()``
alone, which is more suspect than ``recolor()`` alone — when all three
get the same partial score. This file pins that ordering so any future
formula change still has to satisfy F1.

We deliberately do not pin numeric suspicion values to specific
constants — the exact §7.3.3 score-effect formula is reserved for a
later sprint. Sprint-1 ships the qualitative contract.
"""

from __future__ import annotations

import pytest

# Load integration first to avoid the existing PSE import-cycle.
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.phase2 import (
    DEFAULT_PHASE2_CONFIG,
    Phase2Config,
    SuspicionScore,
    compute_suspicion,
)
from cognithor.channels.program_synthesis.search.candidate import InputRef, Program


def _g(name: str) -> Program:
    return Program(primitive=name, children=(InputRef(),), output_type="Grid")


class TestSuspicionScoreShape:
    def test_returns_dataclass_with_three_fields(self) -> None:
        s = compute_suspicion(_g("tile"), partial_score=0.85)
        assert isinstance(s, SuspicionScore)
        assert isinstance(s.value, float)
        assert isinstance(s.syntactic_complexity, float)
        assert s.partial_score == 0.85

    def test_value_in_unit_interval(self) -> None:
        for partial in (0.0, 0.25, 0.5, 0.85, 1.0):
            for prog in (_g("tile"), _g("objects"), _g("recolor")):
                s = compute_suspicion(prog, partial_score=partial)
                assert 0.0 <= s.value <= 1.0

    def test_partial_score_must_be_in_unit_interval(self) -> None:
        with pytest.raises(ValueError, match="partial_score must be in"):
            compute_suspicion(_g("tile"), partial_score=-0.1)
        with pytest.raises(ValueError, match="partial_score must be in"):
            compute_suspicion(_g("tile"), partial_score=1.5)


class TestF1OrderingInvariant:
    """Spec §7.3.2 verbatim ordering invariant."""

    def test_objects_more_suspect_than_tile_at_same_partial_score(self) -> None:
        p = 0.85
        s_tile = compute_suspicion(_g("tile"), partial_score=p)
        s_objects = compute_suspicion(_g("objects"), partial_score=p)
        # objects has lower syntactic_complexity → higher suspicion.
        assert s_objects.syntactic_complexity < s_tile.syntactic_complexity
        assert s_objects.value > s_tile.value

    def test_recolor_more_suspect_than_objects_at_same_partial_score(self) -> None:
        # Regular primitives have the lowest multiplier, so a 1-token
        # recolor() with high score is the most suspect of all.
        p = 0.85
        s_recolor = compute_suspicion(_g("recolor"), partial_score=p)
        s_objects = compute_suspicion(_g("objects"), partial_score=p)
        assert s_recolor.value > s_objects.value

    def test_full_chain_ordering(self) -> None:
        p = 0.85
        s_tile = compute_suspicion(_g("tile"), partial_score=p).value
        s_objects = compute_suspicion(_g("objects"), partial_score=p).value
        s_recolor = compute_suspicion(_g("recolor"), partial_score=p).value
        # Spec §7.3.2 verbatim: tile < objects < recolor (in suspicion).
        assert s_tile < s_objects < s_recolor


class TestEdgeCases:
    def test_pure_input_ref_max_suspicion_equals_partial_score(self) -> None:
        # No primitives → zero complexity → full partial_score is
        # suspicious. A program that's just InputRef cannot honestly
        # produce a partial_score = 0.85.
        s = compute_suspicion(InputRef(), partial_score=0.85)
        assert s.syntactic_complexity == 0.0
        assert s.value == 0.85

    def test_zero_partial_score_yields_zero_suspicion(self) -> None:
        s = compute_suspicion(_g("tile"), partial_score=0.0)
        assert s.value == 0.0

    def test_perfect_partial_score_still_below_one_for_complex_program(self) -> None:
        # A long high-impact chain has near-1 complexity → suspicion
        # near 0 even with a perfect partial_score.
        prog: Program = _g("tile")
        for _ in range(20):
            prog = Program("tile", (prog,), "Grid")
        s = compute_suspicion(prog, partial_score=1.0)
        assert s.syntactic_complexity == 1.0
        assert s.value == 0.0


class TestConfigOverride:
    def test_collapsing_multipliers_collapses_ordering(self) -> None:
        # If all classes get a 1.0 multiplier, the F1 distinction
        # disappears: objects(), tile(), and recolor() all weigh the
        # same. The Sprint-1 A/B knob in Phase2Config must pass
        # through to compute_suspicion.
        flat = Phase2Config(
            high_impact_multiplier=1.0,
            structural_abstraction_multiplier=1.0,
            regular_primitive_multiplier=1.0,
        )
        p = 0.85
        s_tile = compute_suspicion(_g("tile"), partial_score=p, config=flat)
        s_objects = compute_suspicion(_g("objects"), partial_score=p, config=flat)
        s_recolor = compute_suspicion(_g("recolor"), partial_score=p, config=flat)
        # All three identical under the flat config.
        assert s_tile.value == s_objects.value == s_recolor.value

    def test_default_config_used_when_none_passed(self) -> None:
        s_default = compute_suspicion(_g("tile"), partial_score=0.85)
        s_explicit = compute_suspicion(
            _g("tile"),
            partial_score=0.85,
            config=DEFAULT_PHASE2_CONFIG,
        )
        assert s_default.value == s_explicit.value
