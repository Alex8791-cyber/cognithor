"""Reporter — Markdown table for ScenarioResult lists."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cognithor_bench.adapters.base import ScenarioResult


def _format_pass_rate(passed: int, total: int) -> str:
    if total == 0:
        return "—"
    if passed == total:
        return "✅"
    if passed == 0:
        return "❌"
    return f"{(passed / total) * 100.0:.0f}%"


def tabulate_results(results: list[ScenarioResult]) -> str:
    if not results:
        return "_no results to report_\n"

    grouped: dict[str, list[ScenarioResult]] = defaultdict(list)
    for r in results:
        grouped[r.id].append(r)

    lines = [
        "| Scenario | Runs | Pass-Rate | Avg Duration (s) | Sample Output |",
        "| --- | --- | --- | --- | --- |",
    ]
    total_runs = 0
    total_pass = 0
    for sid in sorted(grouped):
        runs = grouped[sid]
        n = len(runs)
        passed = sum(1 for r in runs if r.success)
        avg = mean(r.duration_sec for r in runs)
        sample = (runs[0].output or "")[:40].replace("\n", " ")
        sample_md = sample if sample else (runs[0].error or "—")
        lines.append(f"| {sid} | {n} | {_format_pass_rate(passed, n)} | {avg:.3f} | {sample_md} |")
        total_runs += n
        total_pass += passed

    total_pct = _format_pass_rate(total_pass, total_runs)
    lines.append(f"| **Total** | **{total_runs}** | **{total_pct}** | — | — |")
    return "\n".join(lines) + "\n"
