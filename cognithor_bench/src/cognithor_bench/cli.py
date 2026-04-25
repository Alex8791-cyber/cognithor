"""cognithor-bench CLI — `run` and `tabulate` subcommands."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from cognithor_bench.adapters.autogen_adapter import AutoGenAdapter
from cognithor_bench.adapters.base import ScenarioResult
from cognithor_bench.adapters.cognithor_adapter import CognithorAdapter
from cognithor_bench.reporter import tabulate_results
from cognithor_bench.runner import BenchRunner


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cognithor-bench",
        description="Reproducible Multi-Agent benchmark scaffold for Cognithor.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run a JSONL scenario file through an adapter.")
    run.add_argument("scenario", type=Path, help="Path to a .jsonl scenario file.")
    run.add_argument("--repeat", type=int, default=1, help="Repetitions per scenario.")
    run.add_argument("--subsample", type=float, default=1.0, help="Fraction of rows to sample.")
    run.add_argument("--adapter", choices=("cognithor", "autogen"), default="cognithor")
    run.add_argument(
        "--model", default="ollama/qwen3:8b", help="Model spec (e.g. ollama/qwen3:8b)."
    )
    run.add_argument("--output-dir", type=Path, default=Path("results"))
    run.add_argument("--seed", type=int, default=None)
    iso = run.add_mutually_exclusive_group()
    iso.add_argument("--native", action="store_true", help="Run in-process (default).")
    iso.add_argument("--docker", action="store_true", help="Run inside Docker (opt-in).")

    tab = sub.add_parser("tabulate", help="Aggregate a results directory into Markdown.")
    tab.add_argument("results_dir", type=Path)
    return p


def _run_under_docker(args: argparse.Namespace) -> int:
    """Stub for opt-in Docker execution. Real implementation is post-v0.94.0."""
    print(
        "[cognithor-bench] --docker isolation is post-v0.94.0; falling back to --native",
        file=sys.stderr,
    )
    return _run_native(args)


def _run_native(args: argparse.Namespace) -> int:
    adapter = (
        AutoGenAdapter(model=args.model)
        if args.adapter == "autogen"
        else CognithorAdapter(model=args.model)
    )
    runner = BenchRunner(adapter=adapter, seed=args.seed)
    results = asyncio.run(
        runner.run_file(
            args.scenario,
            repeat=args.repeat,
            subsample=args.subsample,
            output_dir=args.output_dir,
        )
    )
    print(tabulate_results(results))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    if not args.scenario.exists():
        print(f"error: scenario file not found: {args.scenario}", file=sys.stderr)
        sys.exit(2)
    if args.docker:
        return _run_under_docker(args)
    return _run_native(args)


def _cmd_tabulate(args: argparse.Namespace) -> int:
    if not args.results_dir.exists():
        print(f"error: results directory not found: {args.results_dir}", file=sys.stderr)
        sys.exit(2)
    rows: list[ScenarioResult] = []
    for path in sorted(args.results_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(ScenarioResult(**json.loads(line)))
    print(tabulate_results(rows))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "run":
        return _cmd_run(args)
    if args.cmd == "tabulate":
        return _cmd_tabulate(args)
    return 1
