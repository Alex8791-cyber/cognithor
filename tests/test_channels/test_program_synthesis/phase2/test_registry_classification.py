# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Drift gate: live REGISTRY agrees with Phase-2 classification whitelists.

Spec v1.4 §7.3.2 says the suspicion-score reads two mutually-exclusive
flags off ``PrimitiveSpec``. Sprint-1 wires those flags from the
phase2.classification whitelists into the @primitive decorator, so
every primitive registered through the decorator picks up its
classification automatically.

This test pins the agreement: if a future PR registers a primitive
whose name is in HIGH_IMPACT_PRIMITIVES, its spec must carry
``is_high_impact=True`` — and the whitelists themselves must remain
disjoint.
"""

from __future__ import annotations

from cognithor.channels.program_synthesis.dsl.registry import REGISTRY

# Load integration first to avoid the existing PSE import-cycle.
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.phase2 import (
    HIGH_IMPACT_PRIMITIVES,
    STRUCTURAL_ABSTRACTION_PRIMITIVES,
    classify_primitive_name,
)


class TestRegistryAutoClassification:
    """Decorator-driven flag-setting matches the static whitelists."""

    def test_every_registered_high_impact_primitive_has_flag_set(self) -> None:
        for spec in REGISTRY.all_primitives():
            if spec.name in HIGH_IMPACT_PRIMITIVES:
                assert spec.is_high_impact, (
                    f"Primitive {spec.name!r} is in HIGH_IMPACT_PRIMITIVES "
                    f"but its registered spec has is_high_impact=False."
                )
                assert not spec.is_structural_abstraction

    def test_every_registered_structural_abstraction_primitive_has_flag_set(
        self,
    ) -> None:
        for spec in REGISTRY.all_primitives():
            if spec.name in STRUCTURAL_ABSTRACTION_PRIMITIVES:
                assert spec.is_structural_abstraction, (
                    f"Primitive {spec.name!r} is in "
                    f"STRUCTURAL_ABSTRACTION_PRIMITIVES but its "
                    f"registered spec has is_structural_abstraction=False."
                )
                assert not spec.is_high_impact

    def test_flagged_primitives_are_in_the_corresponding_whitelist(self) -> None:
        # Reverse direction: if a primitive carries a flag, its name
        # must appear in the matching whitelist. Catches accidental
        # hand-added flags in @primitive(...) calls that bypass the
        # auto-classification path.
        for spec in REGISTRY.all_primitives():
            if spec.is_high_impact:
                assert spec.name in HIGH_IMPACT_PRIMITIVES, (
                    f"Primitive {spec.name!r} has is_high_impact=True "
                    f"but is not in HIGH_IMPACT_PRIMITIVES."
                )
            if spec.is_structural_abstraction:
                assert spec.name in STRUCTURAL_ABSTRACTION_PRIMITIVES, (
                    f"Primitive {spec.name!r} has "
                    f"is_structural_abstraction=True but is not in "
                    f"STRUCTURAL_ABSTRACTION_PRIMITIVES."
                )

    def test_classify_function_agrees_with_registered_specs(self) -> None:
        # Final sanity: the pure helper and the decorator-driven path
        # report the same answer for every registered primitive.
        for spec in REGISTRY.all_primitives():
            cls = classify_primitive_name(spec.name)
            if cls == "high_impact":
                assert spec.is_high_impact
                assert not spec.is_structural_abstraction
            elif cls == "structural_abstraction":
                assert spec.is_structural_abstraction
                assert not spec.is_high_impact
            else:
                assert not spec.is_high_impact
                assert not spec.is_structural_abstraction

    def test_at_least_one_primitive_per_class(self) -> None:
        # Spec rationale: if either class is empty in the registry,
        # the suspicion multipliers can never differentiate, which
        # defeats F1's purpose.
        any_high = any(s.is_high_impact for s in REGISTRY.all_primitives())
        any_struct = any(s.is_structural_abstraction for s in REGISTRY.all_primitives())
        assert any_high, (
            "no registered primitive carries is_high_impact=True; "
            "F1 multipliers cannot discriminate."
        )
        assert any_struct, (
            "no registered primitive carries "
            "is_structural_abstraction=True; F1 multipliers cannot "
            "discriminate."
        )
