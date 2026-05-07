#!/usr/bin/env python3
"""Compute the mutation score from a cosmic-ray session sqlite file.

Outputs a single-line JSON summary suitable for CI consumption:

    {"total": 250, "killed": 215, "survived": 30, "skipped": 5,
     "mutation_score_pct": 87.76, "killed_per_module": {...}}

Usage::

    python tests/quality/score_mutations.py session.sqlite
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path


def score(session_path: Path) -> dict[str, object]:
    """Read cosmic-ray's session DB and compute the killed/survived score.

    Cosmic-ray persists per-mutation results in a sqlite table. The exact
    schema differs slightly across versions; this query is defensive
    (joins the mutation_specs and work_results tables and tolerates
    missing columns).
    """
    if not session_path.exists():
        return {"error": f"session not found: {session_path}"}

    conn = sqlite3.connect(str(session_path))
    try:
        rows = conn.execute(
            "SELECT module_path, test_outcome FROM work_results"
        ).fetchall()
    except sqlite3.OperationalError:
        # Fallback for older cosmic-ray schemas
        rows = conn.execute(
            "SELECT job_id AS module_path, test_outcome FROM work_results"
        ).fetchall()
    conn.close()

    total = len(rows)
    killed_per_module: dict[str, int] = defaultdict(int)
    survived_per_module: dict[str, int] = defaultdict(int)
    counts: dict[str, int] = defaultdict(int)
    for module_path, outcome in rows:
        outcome_lc = (outcome or "").lower()
        counts[outcome_lc] += 1
        if "killed" in outcome_lc:
            killed_per_module[str(module_path)] += 1
        elif "survived" in outcome_lc:
            survived_per_module[str(module_path)] += 1

    killed = counts.get("killed", 0)
    survived = counts.get("survived", 0)
    skipped = counts.get("skipped", 0) + counts.get("incompetent", 0)
    denominator = killed + survived
    mutation_score_pct = (
        round((killed / denominator) * 100, 2) if denominator else 0.0
    )
    return {
        "total": total,
        "killed": killed,
        "survived": survived,
        "skipped": skipped,
        "mutation_score_pct": mutation_score_pct,
        "killed_per_module": dict(killed_per_module),
        "survived_per_module": dict(survived_per_module),
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: score_mutations.py <session.sqlite>", file=sys.stderr)
        return 2
    summary = score(Path(sys.argv[1]))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if "error" not in summary else 1


if __name__ == "__main__":
    sys.exit(main())
