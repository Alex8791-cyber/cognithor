"""Reporter — aggregate ScenarioResult lists into Markdown tables."""

from __future__ import annotations

from cognithor_bench.adapters.base import ScenarioResult
from cognithor_bench.reporter import tabulate_results


def test_tabulate_empty() -> None:
    md = tabulate_results([])
    assert "no results" in md.lower()


def test_tabulate_single_result() -> None:
    results = [
        ScenarioResult(id="s1", output="4", success=True, duration_sec=0.12, error=None),
    ]
    md = tabulate_results(results)
    assert "| s1 |" in md
    assert "| ✅ |" in md or "| pass |" in md


def test_tabulate_aggregates_repeated_ids() -> None:
    """Two runs of the same id show as one row with pass-rate 50%."""
    results = [
        ScenarioResult(id="s1", output="4", success=True, duration_sec=0.1, error=None),
        ScenarioResult(id="s1", output="x", success=False, duration_sec=0.2, error=None),
    ]
    md = tabulate_results(results)
    # Pass rate column shows 50% (1/2)
    assert "50" in md
    # Average duration ~0.15s appears (rounded)
    assert "0.15" in md or "0.150" in md


def test_tabulate_includes_summary_row() -> None:
    results = [
        ScenarioResult(id="s1", output="4", success=True, duration_sec=0.1, error=None),
        ScenarioResult(id="s2", output="x", success=False, duration_sec=0.2, error=None),
    ]
    md = tabulate_results(results)
    assert "Total" in md or "Overall" in md
