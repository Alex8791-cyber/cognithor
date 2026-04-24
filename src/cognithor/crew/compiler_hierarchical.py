"""Hierarchical compiler stub.

Task 8-9 may route HIERARCHICAL-process Crews through this module's entry
point before Task 10 lands the real manager-LLM integration. The stub
returns declaration order so imports resolve and HIERARCHICAL Crews run
deterministically without a manager. Task 10 replaces this wholesale.
"""

from __future__ import annotations

from typing import Any

from cognithor.crew.agent import CrewAgent
from cognithor.crew.task import CrewTask


def order_tasks_hierarchical(
    tasks: list[CrewTask],
    agents: list[CrewAgent],
    **_: Any,
) -> list[CrewTask]:
    """Stub: declaration order. Real manager-LLM routing lands in Task 10."""
    return list(tasks)
