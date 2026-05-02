# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-11 Wave-6 — Scorecard parser + aggregator.

The ARC-AGI-3 harness emits a *scorecard* per agent run — a structured
JSON payload describing the result on each of the games the agent
played. This module parses that payload and aggregates across multiple
runs (different agents, different games) for offline analysis.

The data shape is the official one from
``arcprize.ARC-AGI-3-Agents``'s ``EnvironmentScorecard.model_dump()``:

.. code-block:: json

    {
      "ls20": {
        "game_id": "ls20",
        "agent_name": "ls20.cognithorrandom.80",
        "won": false,
        "levels_completed": 0,
        "win_levels": 3,
        "actions_taken": 80,
        "elapsed_seconds": 12.34,
        "card_id": "abc123"
      },
      "locksmith": { ... }
    }

The parser is permissive (extra fields are ignored, missing fields
get default values) so it tracks upstream schema changes gracefully.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GameResult:
    """One agent run on one game."""

    game_id: str
    agent_name: str
    won: bool
    levels_completed: int
    win_levels: int
    actions_taken: int
    elapsed_seconds: float

    @property
    def progress_ratio(self) -> float:
        """Levels-completed normalised to [0, 1]. Returns 0.0 if win_levels is 0."""
        if self.win_levels <= 0:
            return 0.0
        return min(self.levels_completed / self.win_levels, 1.0)


@dataclass(frozen=True)
class ScorecardSummary:
    """Aggregate across all games in a single scorecard."""

    n_games: int
    n_won: int
    total_actions: int
    total_seconds: float
    mean_progress_ratio: float

    @property
    def win_rate(self) -> float:
        return self.n_won / self.n_games if self.n_games else 0.0


def parse_scorecard(payload: dict[str, Any] | str) -> list[GameResult]:
    """Parse a raw scorecard JSON dict (or string) into a list of results.

    Accepts either:

    * the canonical ``{game_id: {fields...}, ...}`` mapping
    * a JSON string that decodes to such a mapping

    Missing fields fall back to safe defaults (``False`` / ``0`` / ``0.0``).
    Unknown extra fields are ignored. Non-dict entries are skipped with
    no warning — the harness occasionally emits ``null`` for skipped
    games and we don't want a noisy parser.
    """
    if isinstance(payload, str):
        decoded = json.loads(payload)
    else:
        decoded = payload
    if not isinstance(decoded, dict):
        raise ValueError(
            f"parse_scorecard: expected dict at top-level, got {type(decoded).__name__}"
        )
    results: list[GameResult] = []
    for game_id, entry in decoded.items():
        if not isinstance(entry, dict):
            continue
        results.append(
            GameResult(
                game_id=str(entry.get("game_id", game_id)),
                agent_name=str(entry.get("agent_name", "")),
                won=bool(entry.get("won", False)),
                levels_completed=int(entry.get("levels_completed", 0)),
                win_levels=int(entry.get("win_levels", 0)),
                actions_taken=int(entry.get("actions_taken", 0)),
                elapsed_seconds=float(entry.get("elapsed_seconds", 0.0)),
            )
        )
    return results


def summarise(results: list[GameResult]) -> ScorecardSummary:
    """Aggregate a list of :class:`GameResult` into a summary."""
    if not results:
        return ScorecardSummary(
            n_games=0,
            n_won=0,
            total_actions=0,
            total_seconds=0.0,
            mean_progress_ratio=0.0,
        )
    n_games = len(results)
    n_won = sum(1 for r in results if r.won)
    total_actions = sum(r.actions_taken for r in results)
    total_seconds = sum(r.elapsed_seconds for r in results)
    mean_progress = sum(r.progress_ratio for r in results) / n_games
    return ScorecardSummary(
        n_games=n_games,
        n_won=n_won,
        total_actions=total_actions,
        total_seconds=total_seconds,
        mean_progress_ratio=mean_progress,
    )


__all__ = [
    "GameResult",
    "ScorecardSummary",
    "parse_scorecard",
    "summarise",
]
