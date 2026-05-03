"""Exploration task executor -- autonomously researches knowledge gaps."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.learning.curiosity import CuriosityEngine, ExplorationTask
    from cognithor.memory.manager import MemoryManager

log = get_logger(__name__)


@dataclass
class ExplorationResult:
    """Result of executing a single exploration task."""

    gap_id: str
    query: str
    found_answer: bool = False
    answer_summary: str = ""
    sources_checked: list[str] = field(default_factory=list)
    entities_updated: int = 0
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(UTC),
    )


class ExplorationExecutor:
    """Execute exploration tasks proposed by CuriosityEngine."""

    def __init__(
        self,
        curiosity: CuriosityEngine | None = None,
        memory: MemoryManager | None = None,
    ) -> None:
        self._curiosity = curiosity
        self._memory = memory
        self._results: list[ExplorationResult] = []
        self._running = False

    async def execute_task(
        self,
        task: ExplorationTask,
    ) -> ExplorationResult:
        """Execute a single exploration task by searching memory."""
        result = ExplorationResult(
            gap_id=task.gap_id,
            query=task.query,
        )

        if self._curiosity:
            self._curiosity.mark_exploring(task.gap_id)

        # Search memory for answers
        if self._memory and "memory" in task.sources:
            try:
                search_results = await self._search_memory(
                    task.query,
                )
                result.sources_checked.append("memory")
                if search_results:
                    result.found_answer = True
                    result.answer_summary = search_results[:500]
                    result.entities_updated = 1
            except Exception:
                log.debug(
                    "exploration_memory_search_failed",
                    exc_info=True,
                )

        # Update gap status
        if self._curiosity:
            if result.found_answer:
                self._curiosity.mark_answered(task.gap_id)
            else:
                # Reset to open if nothing found
                for g in self._curiosity._gaps:
                    if g.id == task.gap_id and g.status == "exploring":
                        g.status = "open"
                        break

        self._results.append(result)
        return result

    async def execute_batch(
        self,
        max_tasks: int = 3,
    ) -> list[ExplorationResult]:
        """Execute a batch of exploration tasks."""
        if not self._curiosity:
            return []

        tasks = self._curiosity.propose_exploration(
            max_tasks=max_tasks,
        )
        results = []
        for task in tasks:
            try:
                r = await self.execute_task(task)
                results.append(r)
            except Exception:
                log.debug(
                    "exploration_task_failed",
                    gap_id=task.gap_id,
                    exc_info=True,
                )
        return results

    async def _search_memory(self, query: str) -> str:
        """Search memory system for information about a query."""
        if not self._memory:
            return ""
        try:
            # Use the memory manager's search if available
            if hasattr(self._memory, "search"):
                results = await self._memory.search(
                    query,
                    top_k=5,
                )
                if results:
                    return "\n".join(str(r) for r in results[:3])
            # Fallback: search semantic memory directly
            sem = getattr(self._memory, "semantic", None)
            if sem and hasattr(sem, "search"):
                results = sem.search(query, top_k=5)
                if results:
                    return "\n".join(str(r) for r in results[:3])
        except Exception:
            log.debug(
                "memory_search_error",
                query=query[:50],
                exc_info=True,
            )
        return ""

    @property
    def results(self) -> list[ExplorationResult]:
        """Return a copy of accumulated results."""
        return list(self._results)

    def stats(self) -> dict[str, Any]:
        """Return exploration statistics."""
        total = len(self._results)
        answered = sum(1 for r in self._results if r.found_answer)
        return {
            "total_explorations": total,
            "answered": answered,
            "unanswered": total - answered,
            "entities_updated": sum(r.entities_updated for r in self._results),
        }
