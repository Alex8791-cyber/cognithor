# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Trace-builder — turns a Program into a step-by-step pseudo-code trace.

For every primitive call (ProgramNode in the tree) we record:

* an auto-generated variable name
* the source-line equivalent (``var = primitive(args...)``)
* a short summary of the produced value (shape for grids, size + colors
  for ObjectSets, integer/string verbatim, error tag on crash)
* the wall-clock duration of the call

The trace is intentionally deterministic — the same Program against the
same input grid always yields the same trace bytes, which is the
foundation for K9 (every solved program has a trace) and K10 (replay
matches exactly).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from cognithor.channels.program_synthesis.dsl.types_grid import Object, ObjectSet
from cognithor.channels.program_synthesis.search.candidate import (
    Const,
    InputRef,
    Program,
    ProgramNode,
)
from cognithor.channels.program_synthesis.search.executor import (
    Executor,
    InProcessExecutor,
)


@dataclass(frozen=True)
class TraceLine:
    """One step in the trace.

    ``index`` is 1-based for human display. ``var`` is the name bound
    to this step's result (e.g. ``step3``); inner Const / InputRef
    nodes don't get their own line — they're inlined into the parent's
    source.
    """

    index: int
    var: str
    source: str
    summary: str
    duration_ms: float
    ok: bool


@dataclass(frozen=True)
class TraceResult:
    """Aggregate of every :class:`TraceLine` plus the program's final value."""

    program_source: str
    program_hash: str | None
    final_value_summary: str
    final_value_ok: bool
    lines: tuple[TraceLine, ...]

    @property
    def total_duration_ms(self) -> float:
        return sum(line.duration_ms for line in self.lines)

    @property
    def all_ok(self) -> bool:
        return self.final_value_ok and all(line.ok for line in self.lines)


# ---------------------------------------------------------------------------
# Value-summary helpers — short human strings for each ARC value type.
# ---------------------------------------------------------------------------


def _summarise(value: Any) -> str:
    if isinstance(value, np.ndarray) and value.dtype == np.bool_:
        return f"Mask {value.shape} ({int(value.sum())} true)"
    if isinstance(value, np.ndarray):
        return f"Grid {value.shape}"
    if isinstance(value, ObjectSet):
        if len(value) == 0:
            return "ObjectSet (empty)"
        colors = sorted({o.color for o in value})
        return f"ObjectSet of {len(value)} objects, colors={colors}"
    if isinstance(value, Object):
        return f"Object color={value.color} cells={value.size} bbox={value.bbox}"
    if isinstance(value, bool):
        return f"Bool {value}"
    if isinstance(value, int):
        return f"Int {value}"
    if isinstance(value, str):
        return f"Str {value!r}"
    return f"{type(value).__name__} {value!r}"


# ---------------------------------------------------------------------------
# Tree-walking trace builder.
# ---------------------------------------------------------------------------


def _is_terminal(node: ProgramNode) -> bool:
    """A node we don't render as its own trace line.

    ``InputRef`` is the literal input — no need to repeat it.
    ``Const`` values are inlined as arguments in the parent's source.
    Zero-arity Program nodes (e.g. ``const_color_5()``) DO get their
    own line so the trace shows where each constant came from.
    """
    return isinstance(node, InputRef | Const)


def _next_var_name(counter: list[int]) -> str:
    counter[0] += 1
    return f"step{counter[0]}"


def _node_source_with_vars(node: ProgramNode, var_for: dict[int, str]) -> str:
    """Render *node* with its child Program references replaced by step-vars.

    Inner Const / InputRef nodes are still rendered inline. Inner
    Program nodes are replaced by their bound step-var so the trace
    reads as a sequence of assignments rather than nested calls.
    """
    if isinstance(node, InputRef):
        return "input"
    if isinstance(node, Const):
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            return str(node.value)
        return repr(node.value)
    # Program
    args: list[str] = []
    for child in node.children:
        if isinstance(child, Program) and id(child) in var_for:
            args.append(var_for[id(child)])
        else:
            args.append(_node_source_with_vars(child, var_for))
    return f"{node.primitive}({', '.join(args)})"


def build_trace(
    program: ProgramNode,
    input_grid: Any,
    executor: Executor | None = None,
) -> TraceResult:
    """Walk *program* bottom-up; record every primitive call.

    Children are evaluated before their parents, so the trace reads
    naturally as ``step1 = ...; step2 = ...; result = stepN``.
    """
    ex = executor if executor is not None else InProcessExecutor()
    counter: list[int] = [0]
    var_for: dict[int, str] = {}
    lines: list[TraceLine] = []

    def visit(node: ProgramNode) -> tuple[Any, bool]:
        # Recurse into children first so inner steps are emitted before
        # outer steps — bottom-up DFS.
        if isinstance(node, Program):
            for child in node.children:
                if isinstance(child, Program):
                    visit(child)

        if _is_terminal(node):
            # Terminal nodes don't get their own line; the executor
            # handles them as part of the outer call.
            return _eval_terminal(node, input_grid), True

        # Time and execute this Program node via the executor.
        t0 = time.monotonic()
        result = ex.execute(node, input_grid)
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        var = _next_var_name(counter)
        var_for[id(node)] = var
        source = _node_source_with_vars(node, var_for)
        if result.ok:
            summary = _summarise(result.value)
        else:
            summary = f"<error: {result.error}>"
        lines.append(
            TraceLine(
                index=counter[0],
                var=var,
                source=source,
                summary=summary,
                duration_ms=elapsed_ms,
                ok=result.ok,
            )
        )
        return result.value, result.ok

    final_value, final_ok = visit(program)
    if final_ok:
        final_summary = _summarise(final_value)
    else:
        final_summary = "<error during execution>"

    program_hash: str | None = None
    if isinstance(program, Program):
        program_hash = program.stable_hash()
    program_source = program.to_source() if hasattr(program, "to_source") else repr(program)

    return TraceResult(
        program_source=program_source,
        program_hash=program_hash,
        final_value_summary=final_summary,
        final_value_ok=final_ok,
        lines=tuple(lines),
    )


def _eval_terminal(node: ProgramNode, input_grid: Any) -> Any:
    if isinstance(node, InputRef):
        return input_grid
    if isinstance(node, Const):
        return node.value
    raise TypeError(f"_eval_terminal: not a terminal node: {type(node).__name__}")


# ---------------------------------------------------------------------------
# Pretty-printer (the K9 deliverable: human-readable pseudo-code).
# ---------------------------------------------------------------------------


def format_trace(trace: TraceResult, *, header: dict[str, str] | None = None) -> str:
    """Render *trace* as a multi-line string matching the spec §24.5 example.

    ``header`` is an optional dict of name → value pairs for the ``#``-
    prefixed metadata block at the top (spec hash, search time, etc.).
    Callers that have additional context add entries here.
    """
    out_lines: list[str] = ["# PSE Solution Trace"]
    if trace.program_hash:
        out_lines.append(f"# Program hash: {trace.program_hash}")
    if header:
        for k, v in header.items():
            out_lines.append(f"# {k}: {v}")
    out_lines.append("")
    if not trace.lines:
        out_lines.append(f"result = {trace.program_source}")
        out_lines.append(f"        # → {trace.final_value_summary}")
    else:
        for line in trace.lines:
            out_lines.append(f"Step {line.index}: {line.var} = {line.source}")
            out_lines.append(f"        # → {line.summary}")
        out_lines.append("")
        out_lines.append(f"# Final: {trace.final_value_summary}")
    return "\n".join(out_lines)
