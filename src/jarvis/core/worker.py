"""Distributed Worker Runtime — Worker nodes, job distribution, and failover.

Provides:
  - WorkerNode:        Individual worker that pulls and executes jobs
  - WorkerPool:        Manages a fleet of worker nodes
  - JobDistributor:    Routes jobs to workers (round-robin, capability-based)
  - HealthMonitor:     Tracks heartbeats, detects failures
  - FailoverManager:   Re-queues jobs from failed workers

Architecture: §15.2 (Distributed Agent Runtime)
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable

from jarvis.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class WorkerState(StrEnum):
    """Worker lifecycle states."""

    IDLE = "idle"
    BUSY = "busy"
    DRAINING = "draining"    # Finishing current job, then shutting down
    OFFLINE = "offline"
    FAILED = "failed"


class JobState(StrEnum):
    """Job lifecycle states."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    REQUEUED = "requeued"
    DEAD = "dead"            # Exhausted all retries


class RoutingStrategy(StrEnum):
    """Job routing strategies."""

    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    CAPABILITY_BASED = "capability_based"
    RANDOM = "random"


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------


@dataclass
class Job:
    """A unit of work to be executed by a worker."""

    job_id: str
    task_type: str                          # e.g. "agent_turn", "tool_call", "batch"
    payload: dict[str, Any] = field(default_factory=dict)
    priority: int = 5                       # 1 (low) - 10 (critical)
    state: JobState = JobState.PENDING
    assigned_worker: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300
    required_capabilities: list[str] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def elapsed_ms(self) -> float:
        if self.started_at <= 0:
            return 0.0
        end = self.completed_at if self.completed_at > 0 else time.time()
        return (end - self.started_at) * 1000

    @property
    def can_retry(self) -> bool:
        return self.retry_count < self.max_retries

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "task_type": self.task_type,
            "state": self.state.value,
            "priority": self.priority,
            "assigned_worker": self.assigned_worker,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Worker Node
# ---------------------------------------------------------------------------


@dataclass
class WorkerNode:
    """An individual worker node that executes jobs.

    Each worker has capabilities that determine which job types it can handle,
    and sends heartbeats to signal liveness.
    """

    worker_id: str
    name: str = ""
    state: WorkerState = WorkerState.IDLE
    capabilities: list[str] = field(default_factory=list)
    queue_name: str = "default"
    max_concurrent_jobs: int = 1
    current_jobs: list[str] = field(default_factory=list)  # job_ids
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    total_completed: int = 0
    total_failed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def load(self) -> float:
        """Current load as 0.0-1.0 ratio."""
        if self.max_concurrent_jobs <= 0:
            return 1.0
        return len(self.current_jobs) / self.max_concurrent_jobs

    @property
    def available_slots(self) -> int:
        return max(0, self.max_concurrent_jobs - len(self.current_jobs))

    @property
    def is_available(self) -> bool:
        return (
            self.state in (WorkerState.IDLE, WorkerState.BUSY)
            and self.available_slots > 0
        )

    def has_capability(self, capability: str) -> bool:
        """Check if worker can handle a specific capability."""
        if not self.capabilities:
            return True  # No capabilities = handles everything
        return capability in self.capabilities

    def heartbeat(self) -> None:
        """Update heartbeat timestamp."""
        self.last_heartbeat = time.time()

    def assign_job(self, job_id: str) -> bool:
        """Assign a job to this worker."""
        if not self.is_available:
            return False
        self.current_jobs.append(job_id)
        if len(self.current_jobs) >= self.max_concurrent_jobs:
            self.state = WorkerState.BUSY
        return True

    def complete_job(self, job_id: str) -> bool:
        """Mark a job as completed on this worker."""
        if job_id not in self.current_jobs:
            return False
        self.current_jobs.remove(job_id)
        self.total_completed += 1
        if self.state == WorkerState.BUSY and len(self.current_jobs) < self.max_concurrent_jobs:
            self.state = WorkerState.IDLE
        return True

    def fail_job(self, job_id: str) -> bool:
        """Mark a job as failed on this worker."""
        if job_id not in self.current_jobs:
            return False
        self.current_jobs.remove(job_id)
        self.total_failed += 1
        if self.state == WorkerState.BUSY and len(self.current_jobs) < self.max_concurrent_jobs:
            self.state = WorkerState.IDLE
        return True

    def drain(self) -> None:
        """Start draining: finish current jobs but accept no new ones."""
        self.state = WorkerState.DRAINING

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "name": self.name or self.worker_id,
            "state": self.state.value,
            "capabilities": self.capabilities,
            "queue_name": self.queue_name,
            "load": round(self.load, 2),
            "current_jobs": len(self.current_jobs),
            "max_concurrent": self.max_concurrent_jobs,
            "total_completed": self.total_completed,
            "total_failed": self.total_failed,
            "last_heartbeat": self.last_heartbeat,
            "uptime_seconds": round(time.time() - self.registered_at, 1),
        }


# ---------------------------------------------------------------------------
# Health Monitor
# ---------------------------------------------------------------------------


class HealthMonitor:
    """Monitors worker health via heartbeats.

    Detects failed workers and triggers failover when
    heartbeats are missed beyond the configured threshold.
    """

    def __init__(
        self,
        heartbeat_interval: float = 10.0,
        failure_threshold: int = 3,
    ) -> None:
        self._heartbeat_interval = heartbeat_interval
        self._failure_threshold = failure_threshold
        self._failure_callbacks: list[Callable[[str], None]] = []

    @property
    def heartbeat_interval(self) -> float:
        return self._heartbeat_interval

    @property
    def failure_threshold(self) -> int:
        return self._failure_threshold

    @property
    def timeout_seconds(self) -> float:
        """Time after which a worker is considered failed."""
        return self._heartbeat_interval * self._failure_threshold

    def on_failure(self, callback: Callable[[str], None]) -> None:
        """Register a callback for worker failure events."""
        self._failure_callbacks.append(callback)

    def check_workers(self, workers: dict[str, WorkerNode]) -> list[str]:
        """Check all workers for missed heartbeats.

        Returns list of worker_ids that have failed.
        """
        now = time.time()
        failed: list[str] = []

        for worker_id, worker in workers.items():
            if worker.state in (WorkerState.OFFLINE, WorkerState.FAILED):
                continue

            elapsed = now - worker.last_heartbeat
            if elapsed > self.timeout_seconds:
                worker.state = WorkerState.FAILED
                failed.append(worker_id)
                log.warning(
                    "worker_failed",
                    worker_id=worker_id,
                    last_heartbeat_ago=round(elapsed, 1),
                    threshold=self.timeout_seconds,
                )
                for cb in self._failure_callbacks:
                    try:
                        cb(worker_id)
                    except Exception:
                        log.error("failure_callback_error", worker_id=worker_id, exc_info=True)

        return failed

    def worker_health(self, worker: WorkerNode) -> dict[str, Any]:
        """Get health status for a single worker."""
        now = time.time()
        elapsed = now - worker.last_heartbeat
        healthy = elapsed <= self.timeout_seconds

        return {
            "worker_id": worker.worker_id,
            "healthy": healthy,
            "last_heartbeat_ago_s": round(elapsed, 1),
            "state": worker.state.value,
            "load": round(worker.load, 2),
        }


# ---------------------------------------------------------------------------
# Failover Manager
# ---------------------------------------------------------------------------


class FailoverManager:
    """Handles job failover when workers crash.

    Re-queues jobs from failed workers, respecting retry limits.
    Jobs that exceed max_retries are moved to dead state.
    """

    def __init__(self) -> None:
        self._requeued_count: int = 0
        self._dead_count: int = 0
        self._failover_log: list[dict[str, Any]] = []

    def handle_worker_failure(
        self,
        worker: WorkerNode,
        jobs: dict[str, Job],
    ) -> list[Job]:
        """Re-queue all jobs from a failed worker.

        Returns list of jobs that were successfully re-queued.
        Jobs exceeding max_retries are marked dead.
        """
        requeued: list[Job] = []

        for job_id in list(worker.current_jobs):
            job = jobs.get(job_id)
            if not job:
                continue

            job.retry_count += 1
            job.assigned_worker = ""

            if job.can_retry:
                job.state = JobState.REQUEUED
                self._requeued_count += 1
                requeued.append(job)
                log.info(
                    "job_requeued",
                    job_id=job.job_id,
                    retry=job.retry_count,
                    max=job.max_retries,
                )
            else:
                job.state = JobState.DEAD
                job.error = f"Worker {worker.worker_id} failed; retries exhausted"
                self._dead_count += 1
                log.warning(
                    "job_dead",
                    job_id=job.job_id,
                    retries=job.retry_count,
                )

            self._failover_log.append({
                "job_id": job.job_id,
                "worker_id": worker.worker_id,
                "action": "requeued" if job.can_retry or job.state == JobState.REQUEUED else "dead",
                "retry_count": job.retry_count,
                "timestamp": time.time(),
            })

        worker.current_jobs.clear()
        return requeued

    @property
    def requeued_count(self) -> int:
        return self._requeued_count

    @property
    def dead_count(self) -> int:
        return self._dead_count

    @property
    def failover_log(self) -> list[dict[str, Any]]:
        return list(self._failover_log)

    def stats(self) -> dict[str, Any]:
        return {
            "total_requeued": self._requeued_count,
            "total_dead": self._dead_count,
            "failover_events": len(self._failover_log),
        }


# ---------------------------------------------------------------------------
# Job Distributor
# ---------------------------------------------------------------------------


class JobDistributor:
    """Routes jobs to workers based on configurable strategy.

    Supports multiple routing strategies:
    - round_robin: Cycle through available workers
    - least_loaded: Pick worker with lowest load
    - capability_based: Match job requirements to worker capabilities
    - random: Random selection among available workers
    """

    def __init__(self, strategy: RoutingStrategy = RoutingStrategy.LEAST_LOADED) -> None:
        self._strategy = strategy
        self._rr_index: int = 0  # Round-robin counter
        self._assignments: int = 0
        self._rejections: int = 0

    @property
    def strategy(self) -> RoutingStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, value: RoutingStrategy) -> None:
        self._strategy = value

    def assign(
        self,
        job: Job,
        workers: dict[str, WorkerNode],
    ) -> WorkerNode | None:
        """Find the best worker for a job and assign it.

        Returns the assigned worker, or None if no suitable worker found.
        """
        candidates = self._find_candidates(job, workers)
        if not candidates:
            self._rejections += 1
            return None

        if self._strategy == RoutingStrategy.ROUND_ROBIN:
            worker = self._round_robin(candidates)
        elif self._strategy == RoutingStrategy.LEAST_LOADED:
            worker = self._least_loaded(candidates)
        elif self._strategy == RoutingStrategy.CAPABILITY_BASED:
            worker = self._capability_based(job, candidates)
        else:  # RANDOM
            import random
            worker = random.choice(candidates)

        if worker.assign_job(job.job_id):
            job.state = JobState.ASSIGNED
            job.assigned_worker = worker.worker_id
            self._assignments += 1
            log.debug(
                "job_assigned",
                job_id=job.job_id,
                worker_id=worker.worker_id,
                strategy=self._strategy.value,
            )
            return worker

        self._rejections += 1
        return None

    def _find_candidates(
        self,
        job: Job,
        workers: dict[str, WorkerNode],
    ) -> list[WorkerNode]:
        """Filter workers that can accept this job."""
        candidates = []
        for w in workers.values():
            if not w.is_available:
                continue
            # Check capabilities if job requires specific ones
            if job.required_capabilities:
                if not all(w.has_capability(c) for c in job.required_capabilities):
                    continue
            candidates.append(w)
        return candidates

    def _round_robin(self, candidates: list[WorkerNode]) -> WorkerNode:
        """Cycle through candidates."""
        idx = self._rr_index % len(candidates)
        self._rr_index += 1
        return candidates[idx]

    def _least_loaded(self, candidates: list[WorkerNode]) -> WorkerNode:
        """Pick the worker with the lowest load."""
        return min(candidates, key=lambda w: w.load)

    def _capability_based(
        self,
        job: Job,
        candidates: list[WorkerNode],
    ) -> WorkerNode:
        """Pick worker with best explicit capability match, then lowest load."""
        def score(w: WorkerNode) -> tuple[int, int, float]:
            # Explicit matches (capability listed and present)
            explicit = sum(1 for c in job.required_capabilities if c in w.capabilities)
            # Penalty for generalist workers (no capabilities defined)
            specificity = 1 if w.capabilities else 0
            return (-explicit, -specificity, w.load)
        return min(candidates, key=score)

    def stats(self) -> dict[str, Any]:
        return {
            "strategy": self._strategy.value,
            "total_assignments": self._assignments,
            "total_rejections": self._rejections,
        }


# ---------------------------------------------------------------------------
# Worker Pool
# ---------------------------------------------------------------------------


class WorkerPool:
    """Manages a fleet of worker nodes with health monitoring and failover.

    Central orchestrator that:
    - Registers/deregisters workers
    - Distributes jobs via JobDistributor
    - Monitors health via HealthMonitor
    - Handles failover via FailoverManager
    """

    def __init__(
        self,
        strategy: RoutingStrategy = RoutingStrategy.LEAST_LOADED,
        heartbeat_interval: float = 10.0,
        failure_threshold: int = 3,
    ) -> None:
        self._workers: dict[str, WorkerNode] = {}
        self._jobs: dict[str, Job] = {}
        self._pending_queue: list[str] = []  # job_ids in priority order
        self._distributor = JobDistributor(strategy)
        self._health = HealthMonitor(heartbeat_interval, failure_threshold)
        self._failover = FailoverManager()

        # Wire up health → failover
        self._health.on_failure(self._on_worker_failure)

    @property
    def distributor(self) -> JobDistributor:
        return self._distributor

    @property
    def health_monitor(self) -> HealthMonitor:
        return self._health

    @property
    def failover(self) -> FailoverManager:
        return self._failover

    # -- Worker management --

    def register_worker(
        self,
        worker_id: str,
        *,
        name: str = "",
        capabilities: list[str] | None = None,
        queue_name: str = "default",
        max_concurrent: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> WorkerNode:
        """Register a new worker node."""
        worker = WorkerNode(
            worker_id=worker_id,
            name=name or worker_id,
            capabilities=capabilities or [],
            queue_name=queue_name,
            max_concurrent_jobs=max_concurrent,
            metadata=metadata or {},
        )
        self._workers[worker_id] = worker
        log.info("worker_registered", worker_id=worker_id, capabilities=capabilities)
        return worker

    def deregister_worker(self, worker_id: str) -> bool:
        """Remove a worker from the pool."""
        worker = self._workers.get(worker_id)
        if not worker:
            return False
        worker.state = WorkerState.OFFLINE
        # Re-queue any remaining jobs
        if worker.current_jobs:
            self._failover.handle_worker_failure(worker, self._jobs)
            self._reprioritize_requeued()
        del self._workers[worker_id]
        log.info("worker_deregistered", worker_id=worker_id)
        return True

    def get_worker(self, worker_id: str) -> WorkerNode | None:
        return self._workers.get(worker_id)

    def drain_worker(self, worker_id: str) -> bool:
        """Put worker into draining mode."""
        worker = self._workers.get(worker_id)
        if not worker:
            return False
        worker.drain()
        return True

    def worker_heartbeat(self, worker_id: str) -> bool:
        """Record heartbeat from a worker."""
        worker = self._workers.get(worker_id)
        if not worker:
            return False
        worker.heartbeat()
        return True

    def list_workers(self) -> list[WorkerNode]:
        return list(self._workers.values())

    # -- Job management --

    def submit_job(
        self,
        job_id: str,
        task_type: str,
        *,
        payload: dict[str, Any] | None = None,
        priority: int = 5,
        max_retries: int = 3,
        timeout_seconds: int = 300,
        required_capabilities: list[str] | None = None,
    ) -> Job:
        """Submit a new job to the pool."""
        job = Job(
            job_id=job_id,
            task_type=task_type,
            payload=payload or {},
            priority=priority,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            required_capabilities=required_capabilities or [],
        )
        self._jobs[job_id] = job
        self._insert_pending(job_id, priority)
        log.info("job_submitted", job_id=job_id, task_type=task_type, priority=priority)
        return job

    def dispatch_pending(self) -> list[tuple[str, str]]:
        """Try to assign all pending/requeued jobs to available workers.

        Returns list of (job_id, worker_id) pairs that were assigned.
        """
        assigned: list[tuple[str, str]] = []
        remaining: list[str] = []

        for job_id in self._pending_queue:
            job = self._jobs.get(job_id)
            if not job or job.state not in (JobState.PENDING, JobState.REQUEUED):
                continue

            worker = self._distributor.assign(job, self._workers)
            if worker:
                assigned.append((job_id, worker.worker_id))
            else:
                remaining.append(job_id)

        self._pending_queue = remaining
        return assigned

    def start_job(self, job_id: str) -> bool:
        """Mark a job as running (called by the worker)."""
        job = self._jobs.get(job_id)
        if not job or job.state != JobState.ASSIGNED:
            return False
        job.state = JobState.RUNNING
        job.started_at = time.time()
        return True

    def complete_job(self, job_id: str, result: dict[str, Any] | None = None) -> bool:
        """Mark a job as completed."""
        job = self._jobs.get(job_id)
        if not job:
            return False
        job.state = JobState.COMPLETED
        job.completed_at = time.time()
        job.result = result or {}

        worker = self._workers.get(job.assigned_worker)
        if worker:
            worker.complete_job(job_id)

        log.info("job_completed", job_id=job_id, elapsed_ms=round(job.elapsed_ms, 1))
        return True

    def fail_job(self, job_id: str, error: str = "") -> bool:
        """Mark a job as failed and potentially re-queue it."""
        job = self._jobs.get(job_id)
        if not job:
            return False

        worker = self._workers.get(job.assigned_worker)
        if worker:
            worker.fail_job(job_id)

        job.retry_count += 1
        job.error = error

        if job.can_retry:
            job.state = JobState.REQUEUED
            job.assigned_worker = ""
            self._insert_pending(job.job_id, job.priority)
            log.info("job_requeued", job_id=job_id, retry=job.retry_count)
        else:
            job.state = JobState.DEAD
            log.warning("job_dead", job_id=job_id, error=error)

        return True

    def get_job(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def pending_count(self) -> int:
        return len(self._pending_queue)

    # -- Health checks --

    def check_health(self) -> list[str]:
        """Run health check on all workers. Returns list of failed worker IDs."""
        return self._health.check_workers(self._workers)

    def worker_health(self, worker_id: str) -> dict[str, Any] | None:
        """Get health status for a specific worker."""
        worker = self._workers.get(worker_id)
        if not worker:
            return None
        return self._health.worker_health(worker)

    # -- Stats --

    def stats(self) -> dict[str, Any]:
        workers = list(self._workers.values())
        jobs = list(self._jobs.values())

        return {
            "workers": {
                "total": len(workers),
                "idle": sum(1 for w in workers if w.state == WorkerState.IDLE),
                "busy": sum(1 for w in workers if w.state == WorkerState.BUSY),
                "draining": sum(1 for w in workers if w.state == WorkerState.DRAINING),
                "failed": sum(1 for w in workers if w.state == WorkerState.FAILED),
                "total_completed": sum(w.total_completed for w in workers),
                "total_failed": sum(w.total_failed for w in workers),
            },
            "jobs": {
                "total": len(jobs),
                "pending": sum(1 for j in jobs if j.state == JobState.PENDING),
                "assigned": sum(1 for j in jobs if j.state == JobState.ASSIGNED),
                "running": sum(1 for j in jobs if j.state == JobState.RUNNING),
                "completed": sum(1 for j in jobs if j.state == JobState.COMPLETED),
                "failed": sum(1 for j in jobs if j.state in (JobState.FAILED, JobState.DEAD)),
                "requeued": sum(1 for j in jobs if j.state == JobState.REQUEUED),
            },
            "pending_queue_depth": self.pending_count(),
            "distributor": self._distributor.stats(),
            "failover": self._failover.stats(),
        }

    # -- Internal --

    def _on_worker_failure(self, worker_id: str) -> None:
        """Handle a worker failure detected by health monitor."""
        worker = self._workers.get(worker_id)
        if not worker:
            return
        requeued = self._failover.handle_worker_failure(worker, self._jobs)
        for job in requeued:
            self._insert_pending(job.job_id, job.priority)
        if requeued:
            log.info(
                "failover_requeued",
                worker_id=worker_id,
                requeued=len(requeued),
            )

    def _insert_pending(self, job_id: str, priority: int) -> None:
        """Insert job into pending queue in priority order (highest first)."""
        if job_id in self._pending_queue:
            return
        # Find insertion point (descending priority)
        for i, existing_id in enumerate(self._pending_queue):
            existing_job = self._jobs.get(existing_id)
            if existing_job and existing_job.priority < priority:
                self._pending_queue.insert(i, job_id)
                return
        self._pending_queue.append(job_id)

    def _reprioritize_requeued(self) -> None:
        """Re-insert requeued jobs into the pending queue."""
        for job_id, job in self._jobs.items():
            if job.state == JobState.REQUEUED and job_id not in self._pending_queue:
                self._insert_pending(job_id, job.priority)
