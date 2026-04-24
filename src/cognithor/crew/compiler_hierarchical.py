"""Hierarchical compiler: determines task execution order under a manager
LLM. When manager_llm is not available (offline tests, no model set) the
routing falls back to declaration order, keeping behaviour deterministic.

Task 11 will replace the placeholder branch with a real call through
``cognithor.core.delegation`` — at that point a synthetic manager agent
is instantiated and the manager_llm actually steers the choice.
"""

from __future__ import annotations

from cognithor.crew.agent import CrewAgent
from cognithor.crew.task import CrewTask


def order_tasks_hierarchical(
    tasks: list[CrewTask],
    agents: list[CrewAgent],
    *,
    manager_llm: str | None = None,
) -> list[CrewTask]:
    """Return tasks in the order the manager agent chose.

    Deterministic fallback: when no manager_llm is set or the delegation
    module is unavailable, return the declaration order. Production
    hierarchical routing uses the existing ``cognithor.core.delegation``
    module — wiring arrives in Task 11 once the PGE integration lands.
    """
    if manager_llm is None:
        return list(tasks)

    # Placeholder — integration with cognithor.core.delegation lands in Task 11.
    # For now the offline default is identical to sequential. This keeps the
    # test contract tight while leaving the wiring-point explicit.
    return list(tasks)
