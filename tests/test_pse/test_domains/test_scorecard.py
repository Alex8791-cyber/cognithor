"""Tests for ``Scorecard`` + ``ScorecardEntry``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cognithor.channels.program_synthesis.domains.scorecard import (
    SCORECARD_SCHEMA_VERSION,
    Scorecard,
    ScorecardEntry,
)


def _entry(
    domain: str = "sql",
    benchmark: str = "spider-easy",
    score: float = 0.2,
    target: float = 0.3,
) -> ScorecardEntry:
    return ScorecardEntry(
        domain=domain,
        benchmark=benchmark,
        score=score,
        raw_score=f"{score * 100:.1f} % EX",
        target=target,
        sample_count=470,
    )


class TestScorecardEntry:
    def test_to_dict_rounds(self) -> None:
        e = _entry(score=0.123456789)
        d = e.to_dict()
        assert d["score"] == 0.1235
        assert d["delta_vs_target"] == round(0.123456789 - 0.3, 4)

    def test_includes_raw_score(self) -> None:
        d = _entry().to_dict()
        assert d["raw_score"] == "20.0 % EX"


class TestScorecard:
    def test_empty_payload(self) -> None:
        c = Scorecard()
        out = c.to_dict()
        assert out["schema_version"] == SCORECARD_SCHEMA_VERSION
        assert out["entries"] == []

    def test_add_replaces_same_pair(self) -> None:
        c = Scorecard()
        c.add(_entry(score=0.10))
        c.add(_entry(score=0.20))
        assert len(c.entries) == 1
        assert c.entries[0].score == 0.20

    def test_add_keeps_distinct_pairs(self) -> None:
        c = Scorecard()
        c.add(_entry("sql", "spider-easy"))
        c.add(_entry("sql", "spider-medium"))
        c.add(_entry("json", "jq-cookbook"))
        assert len(c.entries) == 3

    def test_to_dict_sorted(self) -> None:
        c = Scorecard()
        c.add(_entry("sql", "spider-easy"))
        c.add(_entry("ast", "humaneval-plus"))
        c.add(_entry("json", "jq-cookbook"))
        out = c.to_dict()
        domains = [e["domain"] for e in out["entries"]]
        assert domains == ["ast", "json", "sql"]

    def test_write_and_load_roundtrip(self, tmp_path: Path) -> None:
        c = Scorecard(git_sha="abc123")
        c.add(_entry("sql", "spider-easy", score=0.31))
        c.add(_entry("ast", "humaneval-plus", score=0.45, target=0.45))
        path = tmp_path / "scorecard.json"
        c.write_json(path)

        loaded = Scorecard.load_json(path)
        assert len(loaded.entries) == 2
        assert loaded.git_sha == "abc123"
        sql = next(e for e in loaded.entries if e.domain == "sql")
        assert sql.score == 0.31

    def test_load_missing_file_returns_empty(self, tmp_path: Path) -> None:
        loaded = Scorecard.load_json(tmp_path / "nope.json")
        assert loaded.entries == []

    def test_load_malformed_json_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        loaded = Scorecard.load_json(path)
        assert loaded.entries == []

    def test_load_skips_malformed_entry(self, tmp_path: Path) -> None:
        path = tmp_path / "mixed.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "git_sha": "x",
                    "entries": [
                        {  # valid
                            "domain": "sql",
                            "benchmark": "spider-easy",
                            "score": 0.2,
                            "raw_score": "20 %",
                            "target": 0.3,
                            "sample_count": 470,
                        },
                        {"domain": "broken"},  # missing required fields
                    ],
                }
            ),
            encoding="utf-8",
        )
        loaded = Scorecard.load_json(path)
        assert len(loaded.entries) == 1
        assert loaded.entries[0].domain == "sql"

    def test_regression_check_no_regressions(self) -> None:
        baseline = Scorecard()
        baseline.add(_entry("sql", "spider-easy", score=0.20))
        current = Scorecard()
        current.add(_entry("sql", "spider-easy", score=0.25))
        assert current.regression_check(baseline) == []

    def test_regression_check_detects_drop(self) -> None:
        baseline = Scorecard()
        baseline.add(_entry("sql", "spider-easy", score=0.30))
        current = Scorecard()
        current.add(_entry("sql", "spider-easy", score=0.20))
        regs = current.regression_check(baseline)
        assert len(regs) == 1
        assert "sql/spider-easy" in regs[0]
        assert "0.300" in regs[0] and "0.200" in regs[0]

    def test_regression_check_within_tolerance(self) -> None:
        baseline = Scorecard()
        baseline.add(_entry("sql", "spider-easy", score=0.300))
        current = Scorecard()
        current.add(_entry("sql", "spider-easy", score=0.298))
        # 0.002 drop is within default 0.005 tolerance
        assert current.regression_check(baseline) == []

    def test_regression_check_ignores_new_pairs(self) -> None:
        baseline = Scorecard()
        baseline.add(_entry("sql", "spider-easy", score=0.20))
        current = Scorecard()
        current.add(_entry("sql", "spider-easy", score=0.25))
        current.add(_entry("ast", "humaneval-plus", score=0.10))
        assert current.regression_check(baseline) == []

    def test_real_baseline_loads(self) -> None:
        # Sanity: the baseline JSON we ship in the repo must parse.
        repo_baseline = Path("docs/pse/scorecard.json")
        if not repo_baseline.is_file():
            pytest.skip("scorecard baseline not in repo (CI checkout root)")
        loaded = Scorecard.load_json(repo_baseline)
        # 10 domain rows are committed in Sprint-26.1
        assert len(loaded.entries) >= 10
