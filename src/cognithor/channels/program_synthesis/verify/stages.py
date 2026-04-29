# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Verifier pipeline stages (spec §10.1).

Five stages, sequenced strictly. Each stage checks one invariant of a
candidate program:

1. **Syntax**  — the program tree is well-formed (every primitive
   reference points to a registered spec, child counts match arity).
2. **Type**    — recursive signature consistency: each primitive's
   declared input types match the output types of its children.
3. **Demo**    — the program produces the expected output on every
   training example. Hard fail.
4. **Property**— Phase-1 property tests (size, color, no-NaN, ...). Hard fail.
5. **Held-Out**— same demo check on held-out pairs. Soft fail (lowers
   ``confidence``, doesn't drop the candidate).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from cognithor.channels.program_synthesis.core.types import StageResult, TaskSpec
from cognithor.channels.program_synthesis.dsl.registry import (
    REGISTRY,
    PrimitiveRegistry,
)
from cognithor.channels.program_synthesis.search.candidate import (
    Const,
    InputRef,
    ProgramNode,
)
from cognithor.channels.program_synthesis.search.executor import (
    Executor,
    InProcessExecutor,
)
from cognithor.channels.program_synthesis.verify.properties import (
    DEFAULT_PROPERTIES,
    PropertyFn,
)


@dataclass
class _StageContext:
    """Per-stage convenience: holds program + spec + executor + registry."""

    program: ProgramNode
    spec: TaskSpec
    executor: Executor
    registry: PrimitiveRegistry


class Stage(ABC):
    """One pipeline stage.

    ``fail_fast`` controls whether a non-passing return short-circuits
    the rest of the pipeline. Demo and Property are fail-fast; Held-Out
    is not (it adjusts confidence instead of dropping the candidate).
    """

    name: str
    fail_fast: bool = True

    @abstractmethod
    def run(self, ctx: _StageContext) -> StageResult: ...


# ---------------------------------------------------------------------------
# Stage 1: Syntax
# ---------------------------------------------------------------------------


class SyntaxStage(Stage):
    name = "syntax"
    fail_fast = True

    def run(self, ctx: _StageContext) -> StageResult:
        t0 = time.monotonic()
        ok, detail = self._check(ctx.program, ctx.registry)
        return StageResult(
            stage="syntax",
            passed=ok,
            detail=detail,
            duration_ms=(time.monotonic() - t0) * 1000.0,
        )

    @staticmethod
    def _check(node: ProgramNode, registry: PrimitiveRegistry) -> tuple[bool, str]:
        if isinstance(node, InputRef):
            return True, ""
        if isinstance(node, Const):
            return True, ""
        # Program
        if node.primitive not in registry:
            return False, f"unknown primitive {node.primitive!r}"
        spec = registry.get(node.primitive)
        if len(node.children) != spec.signature.arity:
            return (
                False,
                f"{node.primitive}: expected arity {spec.signature.arity}, "
                f"got {len(node.children)} children",
            )
        for child in node.children:
            ok, detail = SyntaxStage._check(child, registry)
            if not ok:
                return False, detail
        return True, ""


# ---------------------------------------------------------------------------
# Stage 2: Type
# ---------------------------------------------------------------------------


class TypeStage(Stage):
    name = "type"
    fail_fast = True

    def run(self, ctx: _StageContext) -> StageResult:
        t0 = time.monotonic()
        ok, detail = self._check(ctx.program, ctx.registry)
        return StageResult(
            stage="type",
            passed=ok,
            detail=detail,
            duration_ms=(time.monotonic() - t0) * 1000.0,
        )

    @staticmethod
    def _check(node: ProgramNode, registry: PrimitiveRegistry) -> tuple[bool, str]:
        if isinstance(node, InputRef):
            return True, ""
        if isinstance(node, Const):
            return True, ""
        # Program
        if node.primitive not in registry:
            return False, f"unknown primitive {node.primitive!r}"
        spec = registry.get(node.primitive)
        # Output type consistency.
        if node.output_type != spec.signature.output:
            return (
                False,
                f"{node.primitive}: declared output {node.output_type!r} "
                f"!= signature output {spec.signature.output!r}",
            )
        # Child output types must match the input slots.
        for slot, expected_type in enumerate(spec.signature.inputs):
            child = node.children[slot]
            child_out = child.output_type
            if child_out != expected_type:
                return (
                    False,
                    f"{node.primitive}: arg {slot} type {child_out!r} "
                    f"!= expected {expected_type!r}",
                )
            # Recurse.
            ok, detail = TypeStage._check(child, registry)
            if not ok:
                return False, detail
        return True, ""


# ---------------------------------------------------------------------------
# Stage 3: Demo
# ---------------------------------------------------------------------------


def _outputs_match(actual: object, expected: object) -> bool:
    if isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray):
        return actual.shape == expected.shape and np.array_equal(actual, expected)
    return actual == expected


class DemoStage(Stage):
    name = "demo"
    fail_fast = True

    def run(self, ctx: _StageContext) -> StageResult:
        t0 = time.monotonic()
        if not ctx.spec.examples:
            return StageResult(
                stage="demo",
                passed=False,
                detail="no demo examples in spec",
                duration_ms=(time.monotonic() - t0) * 1000.0,
            )
        for i, (inp, expected) in enumerate(ctx.spec.examples):
            r = ctx.executor.execute(ctx.program, inp)
            if not r.ok:
                return StageResult(
                    stage="demo",
                    passed=False,
                    detail=f"demo {i}: execution failed ({r.error})",
                    duration_ms=(time.monotonic() - t0) * 1000.0,
                )
            if not _outputs_match(r.value, expected):
                return StageResult(
                    stage="demo",
                    passed=False,
                    detail=f"demo {i}: output mismatch",
                    duration_ms=(time.monotonic() - t0) * 1000.0,
                )
        return StageResult(
            stage="demo",
            passed=True,
            detail=f"all {len(ctx.spec.examples)} demos matched",
            duration_ms=(time.monotonic() - t0) * 1000.0,
        )


# ---------------------------------------------------------------------------
# Stage 4: Property
# ---------------------------------------------------------------------------


class PropertyStage(Stage):
    name = "property"
    fail_fast = True

    def __init__(self, properties: tuple[tuple[str, PropertyFn], ...] = DEFAULT_PROPERTIES) -> None:
        self._properties = properties

    def run(self, ctx: _StageContext) -> StageResult:
        t0 = time.monotonic()
        for inp, expected in ctx.spec.examples:
            r = ctx.executor.execute(ctx.program, inp)
            if not r.ok:
                # Demo stage already checked execution. If we got here
                # under fail_fast=True, demo passed. So a fresh failure
                # here is anomalous — flag and stop.
                return StageResult(
                    stage="property",
                    passed=False,
                    detail=f"unexpected execution failure: {r.error}",
                    duration_ms=(time.monotonic() - t0) * 1000.0,
                )
            for prop_name, prop_fn in self._properties:
                ok, prop_detail = prop_fn(r.value, expected, inp)
                if not ok:
                    return StageResult(
                        stage="property",
                        passed=False,
                        detail=f"{prop_name}: {prop_detail}",
                        duration_ms=(time.monotonic() - t0) * 1000.0,
                    )
        return StageResult(
            stage="property",
            passed=True,
            detail=f"{len(self._properties)} properties verified",
            duration_ms=(time.monotonic() - t0) * 1000.0,
        )


# ---------------------------------------------------------------------------
# Stage 5: Held-Out (soft)
# ---------------------------------------------------------------------------


class HeldOutStage(Stage):
    name = "held_out"
    fail_fast = False  # spec §10.1: lowers confidence, doesn't drop candidate

    def run(self, ctx: _StageContext) -> StageResult:
        t0 = time.monotonic()
        if not ctx.spec.held_out:
            return StageResult(
                stage="held_out",
                passed=True,
                detail="no held-out pairs",
                duration_ms=(time.monotonic() - t0) * 1000.0,
            )
        passed = 0
        for inp, expected in ctx.spec.held_out:
            r = ctx.executor.execute(ctx.program, inp)
            if r.ok and _outputs_match(r.value, expected):
                passed += 1
        total = len(ctx.spec.held_out)
        return StageResult(
            stage="held_out",
            passed=passed == total,
            detail=f"{passed}/{total} held-out pairs matched",
            duration_ms=(time.monotonic() - t0) * 1000.0,
        )


def default_pipeline() -> tuple[Stage, ...]:
    """Stage tuple in the spec's mandated order."""
    return (
        SyntaxStage(),
        TypeStage(),
        DemoStage(),
        PropertyStage(),
        HeldOutStage(),
    )


# Re-export REGISTRY + InProcessExecutor so the pipeline module can
# build a default :class:`_StageContext` without consumers importing
# from three places.
__all__ = [
    "DemoStage",
    "HeldOutStage",
    "PropertyStage",
    "Stage",
    "SyntaxStage",
    "TypeStage",
    "default_pipeline",
]
_ = REGISTRY  # type: ignore[unreachable]  — keep name resolvable for re-export tests
_ = InProcessExecutor  # type: ignore[unreachable]
