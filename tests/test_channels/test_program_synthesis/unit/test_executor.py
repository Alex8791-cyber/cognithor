# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""InProcessExecutor tests (spec §11 — Phase-1 in-process variant)."""

from __future__ import annotations

import numpy as np

from cognithor.channels.program_synthesis.dsl.registry import (
    PrimitiveRegistry,
    PrimitiveSpec,
)
from cognithor.channels.program_synthesis.dsl.signatures import Signature
from cognithor.channels.program_synthesis.search.candidate import (
    Const,
    InputRef,
    Program,
)
from cognithor.channels.program_synthesis.search.executor import (
    ExecutionResult,
    InProcessExecutor,
)


def _g(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


def _registry_with(name: str, fn) -> PrimitiveRegistry:
    reg = PrimitiveRegistry()
    reg.register(
        PrimitiveSpec(
            name=name,
            signature=Signature(inputs=("Grid",), output="Grid"),
            cost=1.0,
            fn=fn,
        )
    )
    return reg


class TestInProcessExecutor:
    def test_input_ref_returns_input_grid(self) -> None:
        ex = InProcessExecutor()
        g = _g([[1, 2], [3, 4]])
        result = ex.execute(InputRef(), g)
        assert result.ok is True
        assert np.array_equal(result.value, g)

    def test_const_returns_value(self) -> None:
        ex = InProcessExecutor()
        result = ex.execute(Const(value=5, output_type="Color"), _g([[0]]))
        assert result.ok is True
        assert result.value == 5

    def test_unary_program(self) -> None:
        # Use the real REGISTRY's rotate90.
        ex = InProcessExecutor()
        prog = Program(
            primitive="rotate90",
            children=(InputRef(),),
            output_type="Grid",
        )
        result = ex.execute(prog, _g([[1, 2], [3, 4]]))
        assert result.ok is True
        assert np.array_equal(result.value, _g([[3, 1], [4, 2]]))

    def test_ternary_program(self) -> None:
        ex = InProcessExecutor()
        prog = Program(
            primitive="recolor",
            children=(
                InputRef(),
                Const(value=1, output_type="Color"),
                Const(value=9, output_type="Color"),
            ),
            output_type="Grid",
        )
        result = ex.execute(prog, _g([[1, 2], [1, 3]]))
        assert result.ok is True
        assert np.array_equal(result.value, _g([[9, 2], [9, 3]]))

    def test_nested_program(self) -> None:
        # mirror_horizontal(rotate90(input))
        ex = InProcessExecutor()
        inner = Program("rotate90", (InputRef(),), "Grid")
        outer = Program("mirror_horizontal", (inner,), "Grid")
        result = ex.execute(outer, _g([[1, 2], [3, 4]]))
        assert result.ok is True
        # rotate90: [[3,1],[4,2]] -> mirror_h: [[1,3],[2,4]]
        assert np.array_equal(result.value, _g([[1, 3], [2, 4]]))

    def test_failed_primitive_returns_error_tag(self) -> None:
        # rotate90 expects an int8 grid; pass a string and confirm the
        # exception surfaces as an ExecutionResult with ok=False.
        ex = InProcessExecutor()
        prog = Program("rotate90", (InputRef(),), "Grid")
        result = ex.execute(prog, "not a grid")
        assert result.ok is False
        # The TypeMismatchError class name should be in the error tag.
        assert result.error == "TypeMismatchError"

    def test_unknown_primitive_returns_error(self) -> None:
        # Use a fresh registry that doesn't know "rotate90".
        ex = InProcessExecutor(registry=PrimitiveRegistry())
        prog = Program("rotate90", (InputRef(),), "Grid")
        result = ex.execute(prog, _g([[1]]))
        assert result.ok is False
        assert result.error == "UnknownPrimitiveError"

    def test_custom_registry_isolates_from_global(self) -> None:
        # Register a "double" that returns the input multiplied by 2.
        reg = _registry_with("double", lambda g: (g * 2).astype(np.int8))
        ex = InProcessExecutor(registry=reg)
        prog = Program("double", (InputRef(),), "Grid")
        result = ex.execute(prog, _g([[1, 2]]))
        assert result.ok is True
        assert np.array_equal(result.value, _g([[2, 4]]))

    def test_execution_result_is_frozen_dataclass(self) -> None:
        r = ExecutionResult(ok=True, value=42)
        # Frozen — assignment must fail.
        from dataclasses import FrozenInstanceError

        import pytest as _pytest

        with _pytest.raises(FrozenInstanceError):
            r.value = 99  # type: ignore[misc]
