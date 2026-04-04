"""ARC-AGI-3 task data model — grids, tasks, solutions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["ArcTask", "GameResult", "Grid", "Solution"]

Grid = list[list[int]]


@dataclass
class ArcTask:
    """An ARC-AGI-3 task with example pairs and a test input."""

    task_id: str
    examples: list[tuple[Grid, Grid]]
    test_input: Grid


@dataclass
class Solution:
    """A candidate solution for an ARC task."""

    output: Grid
    method: str
    description: str
    complexity: int
    transform_fn: Any | None = None


@dataclass
class GameResult:
    """Result of playing one ARC game."""

    win: bool
    attempts: int
    task_id: str
    solutions_tried: list[Solution] = field(default_factory=list)
