"""``Scorecard`` — public PSE benchmark scorecard (Sprint-26 §26.1).

The scorecard is the public face of Owner-Decision D1: every domain's
score on its external benchmark, written nightly into
``docs/pse/scorecard.json`` and rendered on
``cognithor.ai/pse/scorecard``.

This module is intentionally schema-first — the JSON shape is the
contract the cognithor.ai site reads, so we lock it down with a
frozen dataclass and a strict ``to_dict``. The CI workflow in
``.github/workflows/pse-scorecard-nightly.yml`` calls
:meth:`Scorecard.write_json` after each domain's eval finishes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

SCORECARD_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ScorecardEntry:
    """One domain's latest benchmark result.

    ``score`` is normalised to [0, 1] regardless of the underlying
    benchmark unit so the cognithor.ai / pse table can render a
    single uniform progress bar. ``raw_score`` keeps the original
    string the benchmark emits (e.g. "30 % EX") for forensic display.
    """

    domain: str
    benchmark: str
    score: float  # 0..1
    raw_score: str
    target: float  # 0..1, sprint-end goal
    sample_count: int
    seconds_total: float = 0.0
    cost_usd_local: float = 0.0
    cost_usd_cloud_reference: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["score"] = round(self.score, 4)
        d["target"] = round(self.target, 4)
        d["seconds_total"] = round(self.seconds_total, 2)
        d["cost_usd_local"] = round(self.cost_usd_local, 6)
        d["cost_usd_cloud_reference"] = round(self.cost_usd_cloud_reference, 6)
        # Derive a friendly delta-vs-target for the UI.
        d["delta_vs_target"] = round(self.score - self.target, 4)
        return d


@dataclass
class Scorecard:
    """Full scorecard payload — all domain entries plus metadata."""

    entries: list[ScorecardEntry] = field(default_factory=list)
    git_sha: str = ""
    notes: str = ""

    def add(self, entry: ScorecardEntry) -> None:
        # Replace any prior entry for the same (domain, benchmark) so
        # the JSON only ever has the most recent run per pair.
        self.entries = [
            e
            for e in self.entries
            if not (e.domain == entry.domain and e.benchmark == entry.benchmark)
        ]
        self.entries.append(entry)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCORECARD_SCHEMA_VERSION,
            "generated_at": datetime.now(UTC).isoformat(),
            "git_sha": self.git_sha,
            "entries": [
                e.to_dict() for e in sorted(self.entries, key=lambda e: (e.domain, e.benchmark))
            ],
            "notes": self.notes,
        }

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------

    def write_json(self, path: Path) -> None:
        """Write the scorecard atomically to ``path``.

        The CI workflow calls this once per nightly run. Using
        ``write_text`` after building the full string keeps the file
        self-consistent; readers never see a partially-written entry.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        path.write_text(payload + "\n", encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path) -> Scorecard:
        """Load a previously-written scorecard back into memory.

        Skips entries that don't conform to the current schema rather
        than raising — keeps a stale CI artefact from blocking a
        regen. Schema-version drift is logged elsewhere by the CI.
        """
        if not path.is_file():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()

        out = cls(git_sha=str(data.get("git_sha", "")))
        for raw in data.get("entries", []):
            try:
                out.entries.append(
                    ScorecardEntry(
                        domain=str(raw["domain"]),
                        benchmark=str(raw["benchmark"]),
                        score=float(raw["score"]),
                        raw_score=str(raw.get("raw_score", "")),
                        target=float(raw.get("target", 0.0)),
                        sample_count=int(raw.get("sample_count", 0)),
                        seconds_total=float(raw.get("seconds_total", 0.0)),
                        cost_usd_local=float(raw.get("cost_usd_local", 0.0)),
                        cost_usd_cloud_reference=float(raw.get("cost_usd_cloud_reference", 0.0)),
                        notes=str(raw.get("notes", "")),
                    )
                )
            except (KeyError, TypeError, ValueError):
                # Skip malformed entry rather than break the whole load
                continue
        return out

    # ------------------------------------------------------------------
    # Regression helpers (consumed by CI gate)
    # ------------------------------------------------------------------

    def regression_check(self, baseline: Scorecard, *, tolerance: float = 0.005) -> list[str]:
        """Return a list of regression descriptions vs ``baseline``.

        Empty list = no regressions. Each string is human-readable and
        ready to drop into a PR comment via the nightly CI workflow.
        ``tolerance`` lets tiny variance (e.g. floating-point noise on
        a fixed dataset) not trigger a failure — Sprint-26 default
        0.5 percentage-points.
        """
        out: list[str] = []
        baseline_map = {(e.domain, e.benchmark): e.score for e in baseline.entries}
        for current in self.entries:
            key = (current.domain, current.benchmark)
            if key not in baseline_map:
                continue
            previous = baseline_map[key]
            if current.score + tolerance < previous:
                drop = previous - current.score
                out.append(
                    f"{current.domain}/{current.benchmark} regressed "
                    f"from {previous:.3f} to {current.score:.3f} "
                    f"(-{drop:.3f}). Tolerance {tolerance:.3f}."
                )
        return out
