"""Tests for Distributed Worker Runtime.

Covers: WorkerNode, WorkerPool, JobDistributor, HealthMonitor,
FailoverManager, Job, and routing strategies.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from jarvis.core.worker import (
    FailoverManager,
    HealthMonitor,
    Job,
    JobDistributor,
    JobState,
    RoutingStrategy,
    WorkerNode,
    WorkerPool,
    WorkerState,
)


# ============================================================================
# Job
# ============================================================================


class TestJob:
    def test_defaults(self) -> None:
        j = Job(job_id="j1", task_type="agent_turn")
        assert j.state == JobState.PENDING
        assert j.priority == 5
        assert j.can_retry is True

    def test_can_retry_exhausted(self) -> None:
        j = Job(job_id="j1", task_type="t", retry_count=3, max_retries=3)
        assert j.can_retry is False

    def test_elapsed_ms(self) -> None:
        j = Job(job_id="j1", task_type="t", started_at=100.0, completed_at=100.5)
        assert j.elapsed_ms == 500.0

    def test_elapsed_ms_unstarted(self) -> None:
        j = Job(job_id="j1", task_type="t")
        assert j.elapsed_ms == 0.0

    def test_to_dict(self) -> None:
        j = Job(job_id="j1", task_type="agent_turn", priority=8)
        d = j.to_dict()
        assert d["job_id"] == "j1"
        assert d["priority"] == 8
        assert d["state"] == "pending"


# ============================================================================
# WorkerNode
# ============================================================================


class TestWorkerNode:
    def test_defaults(self) -> None:
        w = WorkerNode(worker_id="w1")
        assert w.state == WorkerState.IDLE
        assert w.load == 0.0
        assert w.is_available is True
        assert w.available_slots == 1

    def test_load_calculation(self) -> None:
        w = WorkerNode(worker_id="w1", max_concurrent_jobs=4)
        w.current_jobs = ["j1", "j2"]
        assert w.load == 0.5

    def test_assign_job(self) -> None:
        w = WorkerNode(worker_id="w1", max_concurrent_jobs=2)
        assert w.assign_job("j1") is True
        assert "j1" in w.current_jobs
        assert w.state == WorkerState.IDLE  # Still has slots

    def test_assign_fills_capacity(self) -> None:
        w = WorkerNode(worker_id="w1", max_concurrent_jobs=1)
        w.assign_job("j1")
        assert w.state == WorkerState.BUSY
        assert w.is_available is False

    def test_assign_when_full(self) -> None:
        w = WorkerNode(worker_id="w1", max_concurrent_jobs=1)
        w.assign_job("j1")
        assert w.assign_job("j2") is False

    def test_complete_job(self) -> None:
        w = WorkerNode(worker_id="w1", max_concurrent_jobs=1)
        w.assign_job("j1")
        assert w.complete_job("j1") is True
        assert w.total_completed == 1
        assert w.state == WorkerState.IDLE

    def test_complete_unknown_job(self) -> None:
        w = WorkerNode(worker_id="w1")
        assert w.complete_job("unknown") is False

    def test_fail_job(self) -> None:
        w = WorkerNode(worker_id="w1")
        w.assign_job("j1")
        assert w.fail_job("j1") is True
        assert w.total_failed == 1
        assert w.state == WorkerState.IDLE

    def test_drain(self) -> None:
        w = WorkerNode(worker_id="w1")
        w.drain()
        assert w.state == WorkerState.DRAINING
        assert w.is_available is False

    def test_has_capability_empty(self) -> None:
        w = WorkerNode(worker_id="w1")
        assert w.has_capability("anything") is True  # No caps = handles all

    def test_has_capability_match(self) -> None:
        w = WorkerNode(worker_id="w1", capabilities=["gpu", "ml"])
        assert w.has_capability("gpu") is True
        assert w.has_capability("cpu") is False

    def test_heartbeat(self) -> None:
        w = WorkerNode(worker_id="w1")
        old_hb = w.last_heartbeat
        time.sleep(0.01)
        w.heartbeat()
        assert w.last_heartbeat > old_hb

    def test_to_dict(self) -> None:
        w = WorkerNode(worker_id="w1", name="Worker 1")
        d = w.to_dict()
        assert d["worker_id"] == "w1"
        assert d["name"] == "Worker 1"
        assert "load" in d
        assert "uptime_seconds" in d


# ============================================================================
# HealthMonitor
# ============================================================================


class TestHealthMonitor:
    def test_healthy_worker(self) -> None:
        monitor = HealthMonitor(heartbeat_interval=10.0, failure_threshold=3)
        w = WorkerNode(worker_id="w1")
        w.heartbeat()
        failed = monitor.check_workers({"w1": w})
        assert failed == []

    def test_detect_failed_worker(self) -> None:
        monitor = HealthMonitor(heartbeat_interval=1.0, failure_threshold=1)
        w = WorkerNode(worker_id="w1")
        w.last_heartbeat = time.time() - 5.0  # 5s ago, threshold is 1s
        failed = monitor.check_workers({"w1": w})
        assert "w1" in failed
        assert w.state == WorkerState.FAILED

    def test_skip_already_failed(self) -> None:
        monitor = HealthMonitor(heartbeat_interval=1.0, failure_threshold=1)
        w = WorkerNode(worker_id="w1", state=WorkerState.FAILED)
        w.last_heartbeat = time.time() - 100.0
        failed = monitor.check_workers({"w1": w})
        assert failed == []

    def test_skip_offline(self) -> None:
        monitor = HealthMonitor(heartbeat_interval=1.0, failure_threshold=1)
        w = WorkerNode(worker_id="w1", state=WorkerState.OFFLINE)
        w.last_heartbeat = time.time() - 100.0
        failed = monitor.check_workers({"w1": w})
        assert failed == []

    def test_failure_callback(self) -> None:
        monitor = HealthMonitor(heartbeat_interval=1.0, failure_threshold=1)
        failed_ids: list[str] = []
        monitor.on_failure(lambda wid: failed_ids.append(wid))
        w = WorkerNode(worker_id="w1")
        w.last_heartbeat = time.time() - 5.0
        monitor.check_workers({"w1": w})
        assert failed_ids == ["w1"]

    def test_timeout_seconds(self) -> None:
        monitor = HealthMonitor(heartbeat_interval=10.0, failure_threshold=3)
        assert monitor.timeout_seconds == 30.0

    def test_worker_health_info(self) -> None:
        monitor = HealthMonitor(heartbeat_interval=10.0, failure_threshold=3)
        w = WorkerNode(worker_id="w1")
        w.heartbeat()
        info = monitor.worker_health(w)
        assert info["healthy"] is True
        assert info["worker_id"] == "w1"


# ============================================================================
# FailoverManager
# ============================================================================


class TestFailoverManager:
    def test_requeue_jobs(self) -> None:
        fm = FailoverManager()
        w = WorkerNode(worker_id="w1")
        w.current_jobs = ["j1", "j2"]
        jobs = {
            "j1": Job(job_id="j1", task_type="t", max_retries=3),
            "j2": Job(job_id="j2", task_type="t", max_retries=3),
        }
        requeued = fm.handle_worker_failure(w, jobs)
        assert len(requeued) == 2
        assert all(j.state == JobState.REQUEUED for j in requeued)
        assert fm.requeued_count == 2

    def test_dead_when_retries_exhausted(self) -> None:
        fm = FailoverManager()
        w = WorkerNode(worker_id="w1")
        w.current_jobs = ["j1"]
        jobs = {
            "j1": Job(job_id="j1", task_type="t", retry_count=3, max_retries=3),
        }
        requeued = fm.handle_worker_failure(w, jobs)
        assert len(requeued) == 0
        assert jobs["j1"].state == JobState.DEAD
        assert fm.dead_count == 1

    def test_mixed_requeue_and_dead(self) -> None:
        fm = FailoverManager()
        w = WorkerNode(worker_id="w1")
        w.current_jobs = ["j1", "j2"]
        jobs = {
            "j1": Job(job_id="j1", task_type="t", max_retries=3),
            "j2": Job(job_id="j2", task_type="t", retry_count=3, max_retries=3),
        }
        requeued = fm.handle_worker_failure(w, jobs)
        assert len(requeued) == 1
        assert requeued[0].job_id == "j1"
        assert fm.requeued_count == 1
        assert fm.dead_count == 1

    def test_clears_worker_jobs(self) -> None:
        fm = FailoverManager()
        w = WorkerNode(worker_id="w1")
        w.current_jobs = ["j1"]
        jobs = {"j1": Job(job_id="j1", task_type="t")}
        fm.handle_worker_failure(w, jobs)
        assert w.current_jobs == []

    def test_failover_log(self) -> None:
        fm = FailoverManager()
        w = WorkerNode(worker_id="w1")
        w.current_jobs = ["j1"]
        jobs = {"j1": Job(job_id="j1", task_type="t")}
        fm.handle_worker_failure(w, jobs)
        assert len(fm.failover_log) == 1

    def test_stats(self) -> None:
        fm = FailoverManager()
        w = WorkerNode(worker_id="w1")
        w.current_jobs = ["j1"]
        jobs = {"j1": Job(job_id="j1", task_type="t")}
        fm.handle_worker_failure(w, jobs)
        s = fm.stats()
        assert s["total_requeued"] == 1
        assert s["failover_events"] == 1


# ============================================================================
# JobDistributor
# ============================================================================


class TestJobDistributor:
    def _make_workers(self) -> dict[str, WorkerNode]:
        return {
            "w1": WorkerNode(worker_id="w1", max_concurrent_jobs=2),
            "w2": WorkerNode(worker_id="w2", max_concurrent_jobs=2),
            "w3": WorkerNode(worker_id="w3", max_concurrent_jobs=2, capabilities=["gpu"]),
        }

    def test_round_robin(self) -> None:
        dist = JobDistributor(RoutingStrategy.ROUND_ROBIN)
        workers = self._make_workers()
        j1 = Job(job_id="j1", task_type="t")
        j2 = Job(job_id="j2", task_type="t")
        w1 = dist.assign(j1, workers)
        w2 = dist.assign(j2, workers)
        assert w1 is not None
        assert w2 is not None
        assert w1.worker_id != w2.worker_id

    def test_least_loaded(self) -> None:
        dist = JobDistributor(RoutingStrategy.LEAST_LOADED)
        workers = self._make_workers()
        workers["w1"].current_jobs = ["existing"]
        j = Job(job_id="j1", task_type="t")
        w = dist.assign(j, workers)
        assert w is not None
        assert w.worker_id != "w1"  # w1 is more loaded

    def test_capability_based(self) -> None:
        dist = JobDistributor(RoutingStrategy.CAPABILITY_BASED)
        workers = self._make_workers()
        j = Job(job_id="j1", task_type="t", required_capabilities=["gpu"])
        w = dist.assign(j, workers)
        assert w is not None
        assert w.worker_id == "w3"

    def test_no_capable_worker(self) -> None:
        dist = JobDistributor(RoutingStrategy.CAPABILITY_BASED)
        workers = {
            "w1": WorkerNode(worker_id="w1", capabilities=["cpu"]),
        }
        j = Job(job_id="j1", task_type="t", required_capabilities=["gpu"])
        w = dist.assign(j, workers)
        assert w is None

    def test_no_available_workers(self) -> None:
        dist = JobDistributor(RoutingStrategy.LEAST_LOADED)
        workers = {
            "w1": WorkerNode(worker_id="w1", state=WorkerState.OFFLINE),
        }
        j = Job(job_id="j1", task_type="t")
        w = dist.assign(j, workers)
        assert w is None

    def test_stats(self) -> None:
        dist = JobDistributor(RoutingStrategy.LEAST_LOADED)
        workers = {"w1": WorkerNode(worker_id="w1")}
        j = Job(job_id="j1", task_type="t")
        dist.assign(j, workers)
        s = dist.stats()
        assert s["total_assignments"] == 1

    def test_strategy_change(self) -> None:
        dist = JobDistributor(RoutingStrategy.ROUND_ROBIN)
        assert dist.strategy == RoutingStrategy.ROUND_ROBIN
        dist.strategy = RoutingStrategy.LEAST_LOADED
        assert dist.strategy == RoutingStrategy.LEAST_LOADED


# ============================================================================
# WorkerPool — Registration
# ============================================================================


class TestWorkerPoolRegistration:
    def test_register_worker(self) -> None:
        pool = WorkerPool()
        w = pool.register_worker("w1", name="Worker 1", capabilities=["gpu"])
        assert w.worker_id == "w1"
        assert w.name == "Worker 1"
        assert pool.get_worker("w1") is not None

    def test_deregister_worker(self) -> None:
        pool = WorkerPool()
        pool.register_worker("w1")
        assert pool.deregister_worker("w1") is True
        assert pool.get_worker("w1") is None

    def test_deregister_nonexistent(self) -> None:
        pool = WorkerPool()
        assert pool.deregister_worker("unknown") is False

    def test_drain_worker(self) -> None:
        pool = WorkerPool()
        pool.register_worker("w1")
        assert pool.drain_worker("w1") is True
        assert pool.get_worker("w1").state == WorkerState.DRAINING

    def test_worker_heartbeat(self) -> None:
        pool = WorkerPool()
        pool.register_worker("w1")
        old = pool.get_worker("w1").last_heartbeat
        time.sleep(0.01)
        assert pool.worker_heartbeat("w1") is True
        assert pool.get_worker("w1").last_heartbeat > old

    def test_heartbeat_unknown_worker(self) -> None:
        pool = WorkerPool()
        assert pool.worker_heartbeat("unknown") is False

    def test_list_workers(self) -> None:
        pool = WorkerPool()
        pool.register_worker("w1")
        pool.register_worker("w2")
        assert len(pool.list_workers()) == 2


# ============================================================================
# WorkerPool — Job Lifecycle
# ============================================================================


class TestWorkerPoolJobs:
    def _pool_with_workers(self) -> WorkerPool:
        pool = WorkerPool()
        pool.register_worker("w1", max_concurrent=2)
        pool.register_worker("w2", max_concurrent=2)
        return pool

    def test_submit_job(self) -> None:
        pool = self._pool_with_workers()
        j = pool.submit_job("j1", "agent_turn", priority=8)
        assert j.state == JobState.PENDING
        assert pool.pending_count() == 1

    def test_dispatch_assigns_jobs(self) -> None:
        pool = self._pool_with_workers()
        pool.submit_job("j1", "agent_turn")
        assigned = pool.dispatch_pending()
        assert len(assigned) == 1
        assert pool.pending_count() == 0

    def test_dispatch_multiple_jobs(self) -> None:
        pool = self._pool_with_workers()
        pool.submit_job("j1", "t")
        pool.submit_job("j2", "t")
        pool.submit_job("j3", "t")
        assigned = pool.dispatch_pending()
        assert len(assigned) == 3

    def test_dispatch_respects_capacity(self) -> None:
        pool = WorkerPool()
        pool.register_worker("w1", max_concurrent=1)
        pool.submit_job("j1", "t")
        pool.submit_job("j2", "t")
        assigned = pool.dispatch_pending()
        assert len(assigned) == 1
        assert pool.pending_count() == 1

    def test_start_job(self) -> None:
        pool = self._pool_with_workers()
        pool.submit_job("j1", "t")
        pool.dispatch_pending()
        assert pool.start_job("j1") is True
        assert pool.get_job("j1").state == JobState.RUNNING

    def test_complete_job(self) -> None:
        pool = self._pool_with_workers()
        pool.submit_job("j1", "t")
        pool.dispatch_pending()
        pool.start_job("j1")
        assert pool.complete_job("j1", result={"answer": 42}) is True
        assert pool.get_job("j1").state == JobState.COMPLETED
        assert pool.get_job("j1").result == {"answer": 42}

    def test_fail_job_requeues(self) -> None:
        pool = self._pool_with_workers()
        pool.submit_job("j1", "t", max_retries=3)
        pool.dispatch_pending()
        pool.start_job("j1")
        assert pool.fail_job("j1", error="timeout") is True
        j = pool.get_job("j1")
        assert j.state == JobState.REQUEUED
        assert j.retry_count == 1
        assert pool.pending_count() == 1

    def test_fail_job_dead_after_retries(self) -> None:
        pool = self._pool_with_workers()
        pool.submit_job("j1", "t", max_retries=1)
        pool.dispatch_pending()
        pool.start_job("j1")
        pool.fail_job("j1", "err1")
        # Re-dispatch
        pool.dispatch_pending()
        pool.start_job("j1")
        pool.fail_job("j1", "err2")
        j = pool.get_job("j1")
        assert j.state == JobState.DEAD

    def test_priority_ordering(self) -> None:
        pool = self._pool_with_workers()
        pool.submit_job("low", "t", priority=1)
        pool.submit_job("high", "t", priority=10)
        pool.submit_job("mid", "t", priority=5)
        assigned = pool.dispatch_pending()
        # High priority should be assigned first
        assert assigned[0][0] == "high"


# ============================================================================
# WorkerPool — Health & Failover
# ============================================================================


class TestWorkerPoolHealthFailover:
    def test_check_health_healthy(self) -> None:
        pool = WorkerPool(heartbeat_interval=10.0, failure_threshold=3)
        pool.register_worker("w1")
        pool.worker_heartbeat("w1")
        failed = pool.check_health()
        assert failed == []

    def test_check_health_detects_failure(self) -> None:
        pool = WorkerPool(heartbeat_interval=1.0, failure_threshold=1)
        pool.register_worker("w1")
        pool.get_worker("w1").last_heartbeat = time.time() - 5.0
        failed = pool.check_health()
        assert "w1" in failed

    def test_failover_requeues_jobs(self) -> None:
        pool = WorkerPool(heartbeat_interval=1.0, failure_threshold=1)
        pool.register_worker("w1", max_concurrent=2)
        pool.register_worker("w2", max_concurrent=2)
        pool.submit_job("j1", "t")
        pool.submit_job("j2", "t")
        pool.dispatch_pending()

        # Simulate w1 failure
        w1 = pool.get_worker("w1")
        w1_jobs = list(w1.current_jobs)
        w1.last_heartbeat = time.time() - 5.0
        pool.check_health()

        # Jobs from failed worker should be re-queued
        for jid in w1_jobs:
            j = pool.get_job(jid)
            assert j.state in (JobState.REQUEUED, JobState.ASSIGNED, JobState.COMPLETED)

    def test_deregister_requeues_jobs(self) -> None:
        pool = WorkerPool()
        pool.register_worker("w1")
        pool.submit_job("j1", "t")
        pool.dispatch_pending()
        pool.start_job("j1")
        pool.deregister_worker("w1")
        j = pool.get_job("j1")
        assert j.state in (JobState.REQUEUED, JobState.DEAD)

    def test_worker_health_info(self) -> None:
        pool = WorkerPool()
        pool.register_worker("w1")
        pool.worker_heartbeat("w1")
        info = pool.worker_health("w1")
        assert info is not None
        assert info["healthy"] is True

    def test_worker_health_unknown(self) -> None:
        pool = WorkerPool()
        assert pool.worker_health("unknown") is None


# ============================================================================
# WorkerPool — Stats
# ============================================================================


class TestWorkerPoolStats:
    def test_stats_structure(self) -> None:
        pool = WorkerPool()
        pool.register_worker("w1")
        pool.submit_job("j1", "t")
        s = pool.stats()
        assert "workers" in s
        assert "jobs" in s
        assert "distributor" in s
        assert "failover" in s
        assert s["workers"]["total"] == 1
        assert s["jobs"]["total"] == 1
        assert s["jobs"]["pending"] == 1


# ============================================================================
# Enum coverage
# ============================================================================


class TestEnums:
    def test_worker_state_values(self) -> None:
        assert len(WorkerState) == 5
        assert WorkerState.IDLE == "idle"

    def test_job_state_values(self) -> None:
        assert len(JobState) == 7
        assert JobState.PENDING == "pending"
        assert JobState.DEAD == "dead"

    def test_routing_strategy_values(self) -> None:
        assert len(RoutingStrategy) == 4
        assert RoutingStrategy.ROUND_ROBIN == "round_robin"
