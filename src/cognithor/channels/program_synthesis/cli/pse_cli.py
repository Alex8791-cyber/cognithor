# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""``cognithor pse`` CLI — sub-command dispatch (spec §19.2).

Phase-1 ships the most useful subset:

* ``pse dsl list``           — list every registered primitive.
* ``pse dsl describe <name>``— show signature + cost + description.
* ``pse sandbox doctor``     — print the selected sandbox strategy
                               + the capabilities the host platform
                               grants.
* ``pse run <task.json>``    — synthesize one task end-to-end and
                               print the verdict + trace.
* ``pse explain <result.json>`` — re-render the trace from a saved
                                  SynthesisResult.

The CLI is dependency-injection-friendly: every command takes its
output stream so unit tests can capture stdout without monkeypatching
``sys.stdout``.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import IO, Any

import numpy as np

from cognithor.channels.program_synthesis.cli.dsl_reference import (
    cmd_dsl_reference,
)
from cognithor.channels.program_synthesis.core.types import (
    Budget,
    SynthesisStatus,
    TaskSpec,
)
from cognithor.channels.program_synthesis.core.version import (
    DSL_VERSION,
    PSE_VERSION,
)
from cognithor.channels.program_synthesis.dsl.registry import REGISTRY
from cognithor.channels.program_synthesis.integration.pge_adapter import (
    ProgramSynthesisChannel,
    SynthesisRequest,
)
from cognithor.channels.program_synthesis.sandbox.strategies import (
    capabilities_for_strategy,
    select_sandbox_strategy,
)
from cognithor.channels.program_synthesis.trace import (
    build_trace,
    format_trace,
)

# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------


def cmd_dsl_list(stream: IO[str]) -> int:
    """``pse dsl list`` — table of (name, arity, cost, output)."""
    rows: list[tuple[str, int, float, str]] = []
    for spec in REGISTRY.all_primitives():
        rows.append(
            (
                spec.name,
                spec.signature.arity,
                spec.cost,
                spec.signature.output,
            )
        )
    rows.sort(key=lambda r: (r[1], r[0]))  # by arity, then name
    name_w = max(len(r[0]) for r in rows) if rows else 4
    print(f"{'name':<{name_w}}  arity  cost   output", file=stream)
    print(f"{'-' * name_w}  -----  -----  --------", file=stream)
    for name, arity, cost, output in rows:
        print(
            f"{name:<{name_w}}  {arity:>5}  {cost:>5.2f}  {output}",
            file=stream,
        )
    print(f"\n{len(rows)} primitives registered.", file=stream)
    return 0


def cmd_dsl_describe(name: str, stream: IO[str]) -> int:
    """``pse dsl describe <name>`` — full record for one primitive."""
    if name not in REGISTRY:
        print(f"error: unknown primitive {name!r}", file=stream)
        return 2
    spec = REGISTRY.get(name)
    print(f"name        : {spec.name}", file=stream)
    print(f"arity       : {spec.signature.arity}", file=stream)
    print(f"signature   : {spec.signature.inputs} -> {spec.signature.output}", file=stream)
    print(f"cost        : {spec.cost}", file=stream)
    if spec.description:
        print(f"description : {spec.description}", file=stream)
    if spec.examples:
        print("examples:", file=stream)
        for inp, out in spec.examples:
            print(f"  {inp}  →  {out}", file=stream)
    return 0


def cmd_sandbox_doctor(stream: IO[str]) -> int:
    """``pse sandbox doctor`` — platform detection + capability allow-list."""
    strategy = select_sandbox_strategy(emit_warning=False)
    info = strategy.info
    caps = capabilities_for_strategy(strategy)
    print(f"strategy    : {info.name}", file=stream)
    print(f"description : {info.description}", file=stream)
    print(
        f"limits      : wall_clock={info.limits.wall_clock_seconds}s, "
        f"memory={info.limits.memory_mb}MB, "
        f"per_candidate={info.limits.per_candidate_ms}ms",
        file=stream,
    )
    print(f"research    : {info.research_mode}", file=stream)
    print(f"production  : {info.allows_production_capability}", file=stream)
    print("capabilities:", file=stream)
    for cap in caps:
        print(f"  • {cap.value}", file=stream)
    return 0


def cmd_run(task_path: str, stream: IO[str]) -> int:
    """``pse run <task.json>`` — synthesize from a JSON file.

    Expected JSON shape::

        {
          "examples": [
            {"input": [[1,2],[3,4]], "output": [[3,1],[4,2]]},
            ...
          ],
          "budget": {"max_depth": 4, "wall_clock_seconds": 30.0}
        }
    """
    try:
        with open(task_path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot read {task_path}: {exc}", file=stream)
        return 2
    spec = _spec_from_payload(payload)
    if spec is None:
        print(
            "error: payload does not look like a synthesis task "
            "(missing 'examples' or wrong shape)",
            file=stream,
        )
        return 2
    budget = _budget_from_payload(payload)
    channel = ProgramSynthesisChannel()
    result = channel.synthesize(SynthesisRequest(spec=spec, budget=budget))

    print(f"status       : {result.status.value}", file=stream)
    print(f"score        : {result.score:.2f}", file=stream)
    print(f"confidence   : {result.confidence:.2f}", file=stream)
    print(f"cost_seconds : {result.cost_seconds:.3f}", file=stream)
    print(f"candidates   : {result.cost_candidates}", file=stream)
    print(f"cache_hit    : {result.cache_hit}", file=stream)
    if result.program is not None and hasattr(result.program, "to_source"):
        print(f"\nprogram: {result.program.to_source()}", file=stream)
        # Render the step-by-step trace against the first demo input.
        if spec.examples:
            inp = spec.examples[0][0]
            trace = build_trace(result.program, inp)
            print(file=stream)
            print(
                format_trace(
                    trace,
                    header={
                        "PSE version": PSE_VERSION,
                        "DSL version": DSL_VERSION,
                        "Search time": (
                            f"{result.cost_seconds:.3f}s, {result.cost_candidates} candidates"
                        ),
                    },
                ),
                file=stream,
            )
    return 0 if result.status == SynthesisStatus.SUCCESS else 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spec_from_payload(payload: object) -> TaskSpec | None:
    if not isinstance(payload, dict):
        return None
    examples_raw = payload.get("examples")
    if not isinstance(examples_raw, list) or len(examples_raw) < 1:
        return None
    examples: list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]] = []
    for ex in examples_raw:
        if not isinstance(ex, dict):
            return None
        inp = ex.get("input")
        out = ex.get("output")
        if not isinstance(inp, list) or not isinstance(out, list):
            return None
        try:
            inp_arr = np.array(inp, dtype=np.int8)
            out_arr = np.array(out, dtype=np.int8)
        except (ValueError, TypeError):
            return None
        if inp_arr.ndim != 2 or out_arr.ndim != 2:
            return None
        examples.append((inp_arr, out_arr))
    return TaskSpec(examples=tuple(examples))


def _budget_from_payload(payload: dict[str, Any]) -> Budget:
    raw = payload.get("budget", {})
    if not isinstance(raw, dict):
        return Budget()
    return Budget(
        max_depth=int(raw.get("max_depth", 4)),
        wall_clock_seconds=float(raw.get("wall_clock_seconds", 30.0)),
        max_candidates=int(raw.get("max_candidates", 50_000)),
    )


# ---------------------------------------------------------------------------
# Argparse front-end
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cognithor pse",
        description="Cognithor Program Synthesis Engine CLI (spec §19.2).",
    )
    parser.add_argument(
        "--version", action="version", version=f"PSE {PSE_VERSION} (DSL {DSL_VERSION})"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # dsl
    dsl = sub.add_parser("dsl", help="DSL inspection")
    dsl_sub = dsl.add_subparsers(dest="dsl_command", required=True)
    dsl_sub.add_parser("list", help="list every registered primitive")
    dsl_describe = dsl_sub.add_parser("describe", help="show signature + cost + description")
    dsl_describe.add_argument("name", help="primitive name (e.g. rotate90)")
    dsl_ref = dsl_sub.add_parser(
        "reference",
        help="emit the auto-generated DSL reference Markdown",
    )
    dsl_ref.add_argument(
        "--output",
        default=None,
        help="path to write reference to (default: stdout)",
    )

    # sandbox
    sandbox = sub.add_parser("sandbox", help="sandbox utilities")
    sandbox_sub = sandbox.add_subparsers(dest="sandbox_command", required=True)
    sandbox_sub.add_parser(
        "doctor",
        help="show platform-detected strategy + capability allow-list",
    )

    # run
    run = sub.add_parser("run", help="synthesize from a task JSON file")
    run.add_argument("task_path", help="path to task.json")

    return parser


def main(argv: list[str] | None = None, stream: IO[str] | None = None) -> int:
    """Entry point. Returns the process exit code.

    ``stream`` defaults to ``sys.stdout``; tests can pass an
    ``io.StringIO`` to capture the output.
    """
    out = stream if stream is not None else sys.stdout
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "dsl":
        if args.dsl_command == "list":
            return cmd_dsl_list(out)
        if args.dsl_command == "describe":
            return cmd_dsl_describe(args.name, out)
        if args.dsl_command == "reference":
            return cmd_dsl_reference(out, args.output)
    if args.command == "sandbox" and args.sandbox_command == "doctor":
        return cmd_sandbox_doctor(out)
    if args.command == "run":
        return cmd_run(args.task_path, out)

    parser.print_help(out)
    return 2


__all__ = ["main"]
