# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""DSL signature tests (spec §7.4)."""

from __future__ import annotations

import pytest

from cognithor.channels.program_synthesis.dsl.signatures import (
    ALLOWED_TYPES,
    Signature,
)


class TestSignature:
    def test_simple_unary_grid_signature(self) -> None:
        sig = Signature(inputs=("Grid",), output="Grid")
        assert sig.arity == 1
        assert sig.matches(("Grid",))
        assert not sig.matches(("Color",))

    def test_ternary_signature(self) -> None:
        sig = Signature(inputs=("Grid", "Color", "Color"), output="Grid")
        assert sig.arity == 3
        assert sig.matches(("Grid", "Color", "Color"))
        assert not sig.matches(("Grid", "Color"))

    def test_predicate_output_allowed(self) -> None:
        sig = Signature(inputs=("Object", "Color"), output="Bool")
        assert sig.output == "Bool"

    def test_higher_order_signature(self) -> None:
        # filter_objects: (ObjectSet, Predicate) -> ObjectSet
        sig = Signature(inputs=("ObjectSet", "Predicate"), output="ObjectSet")
        assert sig.arity == 2

    def test_unknown_input_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown type tag"):
            Signature(inputs=("Foo",), output="Grid")

    def test_unknown_output_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown type tag"):
            Signature(inputs=("Grid",), output="Bar")

    def test_allowed_types_includes_phase_1_5_additions(self) -> None:
        # spec §6.4 + §7.5 introduced these in Phase 1.5
        for t in ("Predicate", "Lambda", "AlignMode", "SortKey"):
            assert t in ALLOWED_TYPES

    def test_zero_arity_constant_signature(self) -> None:
        sig = Signature(inputs=(), output="Color")
        assert sig.arity == 0
        assert sig.matches(())
