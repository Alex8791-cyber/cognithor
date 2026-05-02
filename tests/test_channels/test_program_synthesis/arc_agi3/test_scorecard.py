# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-11 Wave-6 — scorecard parser + aggregator tests."""

from __future__ import annotations

import json

import pytest

from cognithor.channels.program_synthesis.arc_agi3.scorecard import (
    GameResult,
    ScorecardSummary,
    parse_scorecard,
    summarise,
)
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)

_SAMPLE_SCORECARD = {
    "ls20": {
        "game_id": "ls20",
        "agent_name": "ls20.cognithorrandom.80",
        "won": False,
        "levels_completed": 0,
        "win_levels": 3,
        "actions_taken": 80,
        "elapsed_seconds": 12.34,
        "card_id": "abc123",
    },
    "locksmith": {
        "game_id": "locksmith",
        "agent_name": "locksmith.cognithorrandom.80",
        "won": True,
        "levels_completed": 5,
        "win_levels": 5,
        "actions_taken": 47,
        "elapsed_seconds": 8.21,
        "card_id": "abc124",
    },
}


class TestParseScorecard:
    def test_parses_dict_payload(self) -> None:
        results = parse_scorecard(_SAMPLE_SCORECARD)
        assert len(results) == 2
        ls20 = next(r for r in results if r.game_id == "ls20")
        assert ls20.won is False
        assert ls20.levels_completed == 0
        assert ls20.actions_taken == 80
        locksmith = next(r for r in results if r.game_id == "locksmith")
        assert locksmith.won is True
        assert locksmith.levels_completed == 5

    def test_parses_json_string(self) -> None:
        results = parse_scorecard(json.dumps(_SAMPLE_SCORECARD))
        assert len(results) == 2

    def test_top_level_non_dict_raises(self) -> None:
        with pytest.raises(ValueError, match="expected dict"):
            parse_scorecard("[1, 2, 3]")

    def test_skips_non_dict_entries(self) -> None:
        payload = {
            "ls20": _SAMPLE_SCORECARD["ls20"],
            "locksmith": None,
            "skipped_game": "this is a string, not a dict",
        }
        results = parse_scorecard(payload)
        assert len(results) == 1
        assert results[0].game_id == "ls20"

    def test_missing_fields_get_defaults(self) -> None:
        payload = {"sparse_game": {"game_id": "sparse_game"}}
        results = parse_scorecard(payload)
        assert results[0].won is False
        assert results[0].levels_completed == 0
        assert results[0].win_levels == 0
        assert results[0].actions_taken == 0
        assert results[0].elapsed_seconds == 0.0
        assert results[0].agent_name == ""

    def test_unknown_extra_fields_ignored(self) -> None:
        payload = {
            "g": {
                "game_id": "g",
                "won": True,
                "levels_completed": 1,
                "win_levels": 1,
                "actions_taken": 5,
                "elapsed_seconds": 0.5,
                "future_field": "ignored",
                "another": [1, 2, 3],
            }
        }
        results = parse_scorecard(payload)
        assert len(results) == 1
        assert results[0].won is True


class TestGameResult:
    def test_progress_ratio(self) -> None:
        r = GameResult(
            game_id="g",
            agent_name="",
            won=False,
            levels_completed=2,
            win_levels=4,
            actions_taken=0,
            elapsed_seconds=0.0,
        )
        assert r.progress_ratio == 0.5

    def test_progress_ratio_clamps_to_one(self) -> None:
        r = GameResult(
            game_id="g",
            agent_name="",
            won=True,
            levels_completed=10,
            win_levels=4,  # over-reported by harness
            actions_taken=0,
            elapsed_seconds=0.0,
        )
        assert r.progress_ratio == 1.0

    def test_progress_ratio_zero_if_no_win_levels(self) -> None:
        r = GameResult(
            game_id="g",
            agent_name="",
            won=False,
            levels_completed=0,
            win_levels=0,
            actions_taken=0,
            elapsed_seconds=0.0,
        )
        assert r.progress_ratio == 0.0


class TestSummarise:
    def test_empty(self) -> None:
        s = summarise([])
        assert s.n_games == 0
        assert s.n_won == 0
        assert s.win_rate == 0.0

    def test_aggregate(self) -> None:
        results = parse_scorecard(_SAMPLE_SCORECARD)
        s = summarise(results)
        assert s.n_games == 2
        assert s.n_won == 1
        assert s.win_rate == 0.5
        assert s.total_actions == 127  # 80 + 47
        assert abs(s.total_seconds - 20.55) < 1e-9
        # Mean progress: (0/3 + 5/5) / 2 = 0.5
        assert abs(s.mean_progress_ratio - 0.5) < 1e-9

    def test_summary_is_dataclass(self) -> None:
        # ScorecardSummary is frozen so it can be hashed/cached.
        s = ScorecardSummary(
            n_games=1,
            n_won=1,
            total_actions=10,
            total_seconds=1.0,
            mean_progress_ratio=1.0,
        )
        assert s.win_rate == 1.0
