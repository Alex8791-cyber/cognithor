# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Executor protocol + an in-process implementation (spec §11).

The full sandboxed executor with subprocess + AST whitelist + WSL2
worker (spec §11.6) lands in Week 4-5. For the search engine plumbing —
observational-equivalence pruning, demo verification — we need the
*interface* now plus a direct in-process variant suitable for unit
tests and the trivial-task end-to-end.

Both implementations satisfy :class:`Executor`; the pruner and the
enumerator stay agnostic to which one is wired in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from cognithor.channels.program_synthesis.dsl.registry import (
    REGISTRY,
    PrimitiveRegistry,
)
from cognithor.channels.program_synthesis.search.candidate import (
    Const,
    InputRef,
    ProgramNode,
)


@dataclass(frozen=True)
class ExecutionResult:
    """Outcome of running one program on one input.

    ``ok`` is True iff the program ran to completion and produced an
    output. ``error`` is a *type tag* not a full message, so two
    candidates that fail the same way share a fingerprint.
    """

    ok: bool
    value: Any = None
    error: str | None = None


class Executor(Protocol):
    """Minimal contract: execute a program on one input, return a result."""

    def execute(self, program: ProgramNode, input_grid: Any) -> ExecutionResult: ...


class InProcessExecutor:
    """Direct call into the registered Python function via REGISTRY.

    Phase 1 fallback used by tests and the trivial-task path. The real
    sandboxed worker (Week 4-5) implements the same interface in a
    subprocess with resource limits.
    """

    def __init__(self, registry: PrimitiveRegistry | None = None) -> None:
        self._registry = registry if registry is not None else REGISTRY

    def execute(self, program: ProgramNode, input_grid: Any) -> ExecutionResult:
        try:
            value = self._eval(program, input_grid)
        except Exception as exc:
            return ExecutionResult(ok=False, error=type(exc).__name__)
        return ExecutionResult(ok=True, value=value)

    # -- internals ---------------------------------------------------

    def _eval(self, node: ProgramNode, input_grid: Any) -> Any:
        if isinstance(node, InputRef):
            return input_grid
        if isinstance(node, Const):
            return node.value
        # Program
        spec = self._registry.get(node.primitive)
        args = tuple(self._eval(child, input_grid) for child in node.children)
        return spec.fn(*args)
