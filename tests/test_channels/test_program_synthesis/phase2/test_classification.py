# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""F1 — DSL primitive classification tests (spec v1.4 §7.3.2)."""

from __future__ import annotations

import pytest

from cognithor.channels.program_synthesis.core.exceptions import DSLError
from cognithor.channels.program_synthesis.dsl.registry import PrimitiveSpec
from cognithor.channels.program_synthesis.dsl.signatures import Signature

# Load integration first to avoid the existing PSE import-cycle.
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.phase2 import (
    HIGH_IMPACT_PRIMITIVES,
    STRUCTURAL_ABSTRACTION_PRIMITIVES,
    classify_primitive_name,
)


class TestClassifyPrimitiveName:
    """spec v1.4 §7.3.2 — three-way classification."""

    def test_objects_in_structural_abstraction_set(self) -> None:
        # Central F1 reclassification: ``objects`` is no longer
        # high_impact (it produces an object set, not an output).
        assert "objects" in STRUCTURAL_ABSTRACTION_PRIMITIVES
        assert "objects" not in HIGH_IMPACT_PRIMITIVES
        assert classify_primitive_name("objects") == "structural_abstraction"

    def test_tile_only_in_high_impact(self) -> None:
        assert "tile" in HIGH_IMPACT_PRIMITIVES
        assert "tile" not in STRUCTURAL_ABSTRACTION_PRIMITIVES
        assert classify_primitive_name("tile") == "high_impact"

    def test_recolor_is_regular(self) -> None:
        assert classify_primitive_name("recolor") == "regular"

    def test_unknown_primitive_is_regular(self) -> None:
        assert classify_primitive_name("never_registered_xyz") == "regular"

    def test_whitelists_are_disjoint(self) -> None:
        assert HIGH_IMPACT_PRIMITIVES.isdisjoint(STRUCTURAL_ABSTRACTION_PRIMITIVES)

    def test_rotate_family_high_impact(self) -> None:
        for name in ("rotate90", "rotate180", "rotate270", "transpose"):
            assert classify_primitive_name(name) == "high_impact"

    def test_object_extractors_structural_abstraction(self) -> None:
        for name in ("connected_components_4", "filter_objects", "bounding_box"):
            assert classify_primitive_name(name) == "structural_abstraction"


class TestPrimitiveSpecMutualExclusion:
    """F1 — spec §18.2 + §18.3: assertion at construction time."""

    def _spec(
        self,
        *,
        is_high_impact: bool = False,
        is_structural_abstraction: bool = False,
    ) -> PrimitiveSpec:
        return PrimitiveSpec(
            name="x",
            signature=Signature(inputs=("Grid",), output="Grid"),
            cost=1.0,
            fn=lambda g: g,
            is_high_impact=is_high_impact,
            is_structural_abstraction=is_structural_abstraction,
        )

    def test_default_neither_flag_set(self) -> None:
        s = self._spec()
        assert s.is_high_impact is False
        assert s.is_structural_abstraction is False

    def test_high_impact_only_constructs(self) -> None:
        s = self._spec(is_high_impact=True)
        assert s.is_high_impact is True
        assert s.is_structural_abstraction is False

    def test_structural_abstraction_only_constructs(self) -> None:
        s = self._spec(is_structural_abstraction=True)
        assert s.is_structural_abstraction is True
        assert s.is_high_impact is False

    def test_both_flags_at_once_raises(self) -> None:
        with pytest.raises(DSLError, match="mutually exclusive"):
            self._spec(is_high_impact=True, is_structural_abstraction=True)
