# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""DSL primitive registry tests (spec §7.1)."""

from __future__ import annotations

import numpy as np
import pytest

from cognithor.channels.program_synthesis.core.exceptions import (
    DSLError,
    UnknownPrimitiveError,
)
from cognithor.channels.program_synthesis.dsl.registry import (
    PrimitiveRegistry,
    PrimitiveSpec,
    primitive,
)
from cognithor.channels.program_synthesis.dsl.signatures import Signature


def _rotate90(grid: np.ndarray) -> np.ndarray:
    return np.rot90(grid, k=-1).copy()


def _make_spec(name: str = "rotate90", cost: float = 1.0) -> PrimitiveSpec:
    return PrimitiveSpec(
        name=name,
        signature=Signature(inputs=("Grid",), output="Grid"),
        cost=cost,
        fn=_rotate90,
        description="Rotate the grid 90° clockwise.",
        examples=(("[[1,2],[3,4]]", "[[3,1],[4,2]]"),),
    )


class TestPrimitiveSpec:
    def test_valid_spec_constructs(self) -> None:
        spec = _make_spec()
        assert spec.name == "rotate90"
        assert spec.signature.arity == 1
        assert spec.cost == 1.0

    def test_negative_cost_rejected(self) -> None:
        with pytest.raises(DSLError, match=">= 0"):
            _make_spec(cost=-0.1)

    def test_invalid_name_rejected(self) -> None:
        with pytest.raises(DSLError, match="Invalid primitive name"):
            _make_spec(name="rotate-90")

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(DSLError, match="Invalid primitive name"):
            _make_spec(name="")


class TestPrimitiveRegistry:
    def test_register_and_lookup(self) -> None:
        reg = PrimitiveRegistry()
        spec = _make_spec()
        reg.register(spec)
        assert reg.get("rotate90") is spec
        assert "rotate90" in reg
        assert len(reg) == 1

    def test_double_register_rejected(self) -> None:
        reg = PrimitiveRegistry()
        reg.register(_make_spec())
        with pytest.raises(DSLError, match="already registered"):
            reg.register(_make_spec())

    def test_unknown_primitive_raises(self) -> None:
        reg = PrimitiveRegistry()
        with pytest.raises(UnknownPrimitiveError):
            reg.get("nope")

    def test_primitives_by_arity(self) -> None:
        reg = PrimitiveRegistry()
        reg.register(_make_spec(name="rotate90"))
        reg.register(
            PrimitiveSpec(
                name="recolor",
                signature=Signature(inputs=("Grid", "Color", "Color"), output="Grid"),
                cost=1.5,
                fn=lambda g, a, b: g,
            )
        )
        unary = reg.primitives_with_arity(1)
        ternary = reg.primitives_with_arity(3)
        assert {p.name for p in unary} == {"rotate90"}
        assert {p.name for p in ternary} == {"recolor"}
        assert reg.primitives_with_arity(2) == ()

    def test_primitives_by_output_type(self) -> None:
        reg = PrimitiveRegistry()
        reg.register(_make_spec(name="rotate90"))
        reg.register(
            PrimitiveSpec(
                name="most_common_color",
                signature=Signature(inputs=("Grid",), output="Color"),
                cost=1.0,
                fn=lambda g: 0,
            )
        )
        grids = reg.primitives_by_output_type("Grid")
        colors = reg.primitives_by_output_type("Color")
        assert {p.name for p in grids} == {"rotate90"}
        assert {p.name for p in colors} == {"most_common_color"}

    def test_decorator_registers_into_explicit_registry(self) -> None:
        reg = PrimitiveRegistry()

        @primitive(
            name="identity",
            signature=Signature(inputs=("Grid",), output="Grid"),
            cost=0.1,
            registry=reg,
        )
        def _identity(grid: np.ndarray) -> np.ndarray:
            return grid

        assert "identity" in reg
        assert reg.get("identity").cost == 0.1

    def test_names_returns_registered_order(self) -> None:
        reg = PrimitiveRegistry()
        reg.register(_make_spec(name="rotate90"))
        reg.register(_make_spec(name="rotate180"))
        assert reg.names() == ("rotate90", "rotate180")
