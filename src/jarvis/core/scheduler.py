"""Priority-based agent scheduling with orchestrator/worker quotas.

Agents run on a min-heap priority queue (1-10, higher = more important).
Orchestrators and workers share compute via configurable quotas (default 50/50).
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass

from jarvis.utils.logging import get_logger
from jarvis.utils.platform import get_max_concurrent_agents

log = get_logger(__name__)


@dataclass
class ScheduledAgent:
    """Metadata for a scheduled agent."""

    agent_id: str
    role: str = "worker"  # "orchestrator" | "worker" | "monitor"
    priority: int = 5  # 1-10, higher = more important


class AgentScheduler:
    """Priority-based agent scheduler with role quotas.

    Args:
        orchestrator_quota: Fraction of ticks allocated to orchestrators (0.0-1.0).
        worker_quota: Fraction of ticks allocated to workers (0.0-1.0).
        default_priority: Default priority for new agents.
        max_concurrent_agents: Max agents running simultaneously.
    """

    def __init__(
        self,
        orchestrator_quota: float = 0.5,
        worker_quota: float = 0.5,
        default_priority: int = 5,
        max_concurrent_agents: int | None = None,
    ) -> None:
        self.orchestrator_quota = orchestrator_quota
        self.worker_quota = worker_quota
        self.default_priority = default_priority
        self.max_concurrent_agents = (
            max_concurrent_agents
            if max_concurrent_agents is not None
            else get_max_concurrent_agents()
        )

        # Min-heap: (-priority, timestamp, agent_id, role)
        self._queue: list[tuple[int, float, str, str]] = []
        self._running: dict[str, ScheduledAgent] = {}

        # Quota tracking (rolling window)
        self._tick_count = 0
        self._orchestrator_ticks = 0
        self._worker_ticks = 0

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    @property
    def running_count(self) -> int:
        return len(self._running)

    def enqueue(self, agent: ScheduledAgent) -> None:
        """Add an agent to the scheduling queue."""
        priority = max(1, min(10, agent.priority))
        heapq.heappush(
            self._queue,
            (-priority, time.monotonic(), agent.agent_id, agent.role),
        )
        log.debug(
            "agent_enqueued",
            agent_id=agent.agent_id,
            role=agent.role,
            priority=priority,
        )

    async def tick(self) -> ScheduledAgent | None:
        """Return the next agent to run, respecting quotas and priority.

        Returns None if the queue is empty or max concurrent agents reached.
        """
        if not self._queue:
            return None

        if self.running_count >= self.max_concurrent_agents:
            return None

        self._tick_count += 1

        # Check quota: if orchestrator quota is exhausted, skip orchestrator entries
        orch_ratio = self._orchestrator_ticks / self._tick_count if self._tick_count > 0 else 0.0
        worker_ratio = self._worker_ticks / self._tick_count if self._tick_count > 0 else 0.0

        # Try to find a suitable agent from the queue
        skipped: list[tuple[int, float, str, str]] = []
        result: ScheduledAgent | None = None

        while self._queue:
            neg_pri, ts, agent_id, role = heapq.heappop(self._queue)

            # Quota enforcement
            if role == "orchestrator" and orch_ratio > self.orchestrator_quota:
                skipped.append((neg_pri, ts, agent_id, role))
                continue
            if role in ("worker", "monitor") and worker_ratio > self.worker_quota:
                skipped.append((neg_pri, ts, agent_id, role))
                continue

            # Found a valid agent
            agent = ScheduledAgent(
                agent_id=agent_id,
                role=role,
                priority=-neg_pri,
            )
            self._running[agent_id] = agent

            if role == "orchestrator":
                self._orchestrator_ticks += 1
            else:
                self._worker_ticks += 1

            result = agent
            break

        # Put skipped items back
        for item in skipped:
            heapq.heappush(self._queue, item)

        # If nothing was selected due to quota, just take the highest priority
        if result is None and self._queue:
            neg_pri, ts, agent_id, role = heapq.heappop(self._queue)
            agent = ScheduledAgent(
                agent_id=agent_id,
                role=role,
                priority=-neg_pri,
            )
            self._running[agent_id] = agent
            if role == "orchestrator":
                self._orchestrator_ticks += 1
            else:
                self._worker_ticks += 1
            result = agent

        return result

    def complete(self, agent_id: str) -> None:
        """Mark an agent as finished, freeing a concurrency slot."""
        self._running.pop(agent_id, None)

    def reset_quotas(self) -> None:
        """Reset quota counters (e.g. at start of new scheduling epoch)."""
        self._tick_count = 0
        self._orchestrator_ticks = 0
        self._worker_ticks = 0
