# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-12 — GameProfile (lifted from cognithor.arc.game_profile).

Persistent per-game mechanic profile with cross-episode learning
metrics. Each game-id gets a JSON file at
``~/.cognithor/arc/game_profiles/<id>.json`` that accumulates
``total_runs``, ``best_score``, per-strategy ``win_rate``, and analysis
results (``click_zones``, ``movement_effects``, ``win_condition``).

The new Sprint-11 ``Sprint10DSLAgent``'s ``EpisodeMemory`` is volatile
(in-memory, capacity 16). ``GameProfile`` is the **persistent**
counterpart — what was learned this run carries to the next.

Drop-in import; the disk format is unchanged from ``cognithor.arc``,
so an existing ``~/.cognithor/arc/game_profiles/`` populated by the
legacy stack is forward-compatible.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from cognithor.utils.logging import get_logger

__all__ = ["GameProfile", "StrategyMetrics"]

log = get_logger(__name__)

_DEFAULT_BASE_DIR = Path.home() / ".cognithor" / "arc"


@dataclass
class StrategyMetrics:
    """Tracks success metrics for a single solver strategy."""

    attempts: int = 0
    wins: int = 0
    total_levels_solved: int = 0
    avg_steps_to_win: float = 0.0
    avg_budget_ratio: float = 0.0

    @property
    def win_rate(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.wins / self.attempts


@dataclass
class GameProfile:
    """Persistent per-game mechanic profile."""

    game_id: str
    game_type: Literal["click", "keyboard", "mixed"]
    available_actions: list[int]

    # Analysis results
    click_zones: list[tuple[int, int]]
    target_colors: list[int]
    movement_effects: dict[int, str]
    win_condition: str
    vision_description: str
    vision_strategy: str

    # Learning metrics
    strategy_metrics: dict[str, StrategyMetrics]
    total_runs: int = 0
    best_score: int = 0

    # Meta
    analyzed_at: str = ""
    profile_version: int = 1
    has_toggles: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = {
            "game_id": self.game_id,
            "game_type": self.game_type,
            "available_actions": self.available_actions,
            "click_zones": [list(z) for z in self.click_zones],
            "target_colors": self.target_colors,
            "movement_effects": {str(k): v for k, v in self.movement_effects.items()},
            "win_condition": self.win_condition,
            "vision_description": self.vision_description,
            "vision_strategy": self.vision_strategy,
            "strategy_metrics": {name: asdict(m) for name, m in self.strategy_metrics.items()},
            "total_runs": self.total_runs,
            "best_score": self.best_score,
            "analyzed_at": self.analyzed_at,
            "profile_version": self.profile_version,
            "has_toggles": self.has_toggles,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GameProfile:
        metrics = {}
        for name, m in d.get("strategy_metrics", {}).items():
            metrics[name] = StrategyMetrics(**m)
        movement = {int(k): v for k, v in d.get("movement_effects", {}).items()}
        click_zones = [tuple(z) for z in d.get("click_zones", [])]
        return cls(
            game_id=d["game_id"],
            game_type=d["game_type"],
            available_actions=d.get("available_actions", []),
            click_zones=click_zones,
            target_colors=d.get("target_colors", []),
            movement_effects=movement,
            win_condition=d.get("win_condition", "unknown"),
            vision_description=d.get("vision_description", ""),
            vision_strategy=d.get("vision_strategy", ""),
            strategy_metrics=metrics,
            total_runs=d.get("total_runs", 0),
            best_score=d.get("best_score", 0),
            analyzed_at=d.get("analyzed_at", ""),
            profile_version=d.get("profile_version", 1),
            has_toggles=d.get("has_toggles", False),
        )

    def save(self, base_dir: Path | None = None) -> None:
        base = base_dir or _DEFAULT_BASE_DIR
        profile_dir = base / "game_profiles"
        profile_dir.mkdir(parents=True, exist_ok=True)
        path = profile_dir / f"{self.game_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        log.info("arc.profile_saved", game_id=self.game_id, path=str(path))

    @classmethod
    def load(cls, game_id: str, base_dir: Path | None = None) -> GameProfile | None:
        base = base_dir or _DEFAULT_BASE_DIR
        path = base / "game_profiles" / f"{game_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            log.warning("arc.profile_load_failed", game_id=game_id, error=str(exc))
            return None

    def update_metrics(
        self,
        strategy_name: str,
        *,
        won: bool,
        levels_solved: int,
        steps: int,
        budget_ratio: float,
    ) -> None:
        if strategy_name not in self.strategy_metrics:
            self.strategy_metrics[strategy_name] = StrategyMetrics()
        m = self.strategy_metrics[strategy_name]
        m.attempts += 1
        m.total_levels_solved += levels_solved
        if won:
            m.wins += 1
            if m.wins == 1:
                m.avg_steps_to_win = float(steps)
                m.avg_budget_ratio = budget_ratio
            else:
                n = m.wins
                m.avg_steps_to_win = m.avg_steps_to_win * (n - 1) / n + steps / n
                m.avg_budget_ratio = m.avg_budget_ratio * (n - 1) / n + budget_ratio / n

    def update_run(self, score: int) -> None:
        self.total_runs += 1
        if score > self.best_score:
            self.best_score = score

    def ranked_strategies(self) -> list[str]:
        if not self.strategy_metrics:
            return []

        def score(name: str) -> float:
            m = self.strategy_metrics[name]
            if m.attempts == 0:
                return 1.0  # exploration bonus
            return m.win_rate

        return sorted(self.strategy_metrics.keys(), key=score, reverse=True)

    def default_strategies(self) -> list[tuple[str, float]]:
        if self.game_type == "click":
            if self.has_toggles:
                return [("cluster_click", 0.6), ("sequence_click", 0.3), ("targeted_click", 0.1)]
            return [("sequence_click", 0.6), ("cluster_click", 0.3), ("targeted_click", 0.1)]
        if self.game_type == "keyboard":
            return [("keyboard_explore", 0.5), ("keyboard_sequence", 0.3), ("hybrid", 0.2)]
        return [("hybrid", 0.5), ("targeted_click", 0.3), ("keyboard_explore", 0.2)]

    @classmethod
    def exists(cls, game_id: str, base_dir: Path | None = None) -> bool:
        base = base_dir or _DEFAULT_BASE_DIR
        return (base / "game_profiles" / f"{game_id}.json").exists()
