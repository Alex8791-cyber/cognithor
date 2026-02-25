"""Tests für Per-Agent-Heartbeat."""

from __future__ import annotations

import pytest

from jarvis.core.agent_heartbeat import (
    AgentHeartbeatConfig,
    AgentHeartbeatScheduler,
    AgentTask,
    TaskStatus,
)


class TestAgentTask:
    def test_to_dict(self) -> None:
        task = AgentTask(task_id="t1", agent_id="coder", name="Build")
        d = task.to_dict()
        assert d["task_id"] == "t1"
        assert d["success_rate"] == 0.0

    def test_success_rate(self) -> None:
        task = AgentTask(task_id="t1", agent_id="a", name="X", run_count=10, fail_count=3)
        d = task.to_dict()
        assert d["success_rate"] == 70.0


class TestAgentHeartbeatScheduler:
    def test_configure_agent(self) -> None:
        sched = AgentHeartbeatScheduler()
        sched.configure_agent(AgentHeartbeatConfig(agent_id="coder", interval_minutes=15))
        assert sched.get_config("coder") is not None
        assert sched.get_config("coder").interval_minutes == 15

    def test_add_and_get_task(self) -> None:
        sched = AgentHeartbeatScheduler()
        sched.add_task(AgentTask(task_id="t1", agent_id="coder", name="Build"))
        assert sched.get_task("coder", "t1") is not None

    def test_remove_task(self) -> None:
        sched = AgentHeartbeatScheduler()
        sched.add_task(AgentTask(task_id="t1", agent_id="coder", name="Build"))
        assert sched.remove_task("coder", "t1")
        assert not sched.remove_task("coder", "t1")

    def test_agent_tasks(self) -> None:
        sched = AgentHeartbeatScheduler()
        sched.add_task(AgentTask(task_id="t1", agent_id="coder", name="Build"))
        sched.add_task(AgentTask(task_id="t2", agent_id="coder", name="Test"))
        sched.add_task(AgentTask(task_id="t3", agent_id="researcher", name="Search"))
        assert len(sched.agent_tasks("coder")) == 2
        assert len(sched.agent_tasks("researcher")) == 1

    def test_enabled_tasks(self) -> None:
        sched = AgentHeartbeatScheduler()
        sched.add_task(AgentTask(task_id="t1", agent_id="a", name="X", enabled=True))
        sched.add_task(AgentTask(task_id="t2", agent_id="a", name="Y", enabled=False))
        assert len(sched.enabled_tasks("a")) == 1

    def test_start_and_complete_task(self) -> None:
        sched = AgentHeartbeatScheduler()
        sched.add_task(AgentTask(task_id="t1", agent_id="a", name="Build"))
        run = sched.start_task("a", "t1")
        assert run is not None
        sched.complete_task(run, success=True)
        task = sched.get_task("a", "t1")
        assert task.run_count == 1
        assert task.last_status == TaskStatus.COMPLETED

    def test_start_disabled_task(self) -> None:
        sched = AgentHeartbeatScheduler()
        sched.add_task(AgentTask(task_id="t1", agent_id="a", name="X", enabled=False))
        assert sched.start_task("a", "t1") is None

    def test_start_nonexistent_task(self) -> None:
        sched = AgentHeartbeatScheduler()
        assert sched.start_task("a", "nope") is None

    def test_failed_task(self) -> None:
        sched = AgentHeartbeatScheduler()
        sched.add_task(AgentTask(task_id="t1", agent_id="a", name="X"))
        run = sched.start_task("a", "t1")
        sched.complete_task(run, success=False, error="timeout")
        task = sched.get_task("a", "t1")
        assert task.fail_count == 1
        assert task.last_error == "timeout"
        assert task.last_status == TaskStatus.FAILED

    def test_agent_summary(self) -> None:
        sched = AgentHeartbeatScheduler()
        sched.configure_agent(AgentHeartbeatConfig(agent_id="coder"))
        sched.add_task(AgentTask(task_id="t1", agent_id="coder", name="Build"))
        summary = sched.agent_summary("coder")
        assert summary["agent_id"] == "coder"
        assert summary["task_count"] == 1

    def test_global_dashboard(self) -> None:
        sched = AgentHeartbeatScheduler()
        sched.configure_agent(AgentHeartbeatConfig(agent_id="a"))
        sched.configure_agent(AgentHeartbeatConfig(agent_id="b"))
        sched.add_task(AgentTask(task_id="t1", agent_id="a", name="X"))
        sched.add_task(AgentTask(task_id="t2", agent_id="b", name="Y"))
        dash = sched.global_dashboard()
        assert dash["agent_count"] == 2
        assert dash["total_tasks"] == 2

    def test_agent_count(self) -> None:
        sched = AgentHeartbeatScheduler()
        sched.configure_agent(AgentHeartbeatConfig(agent_id="a"))
        sched.add_task(AgentTask(task_id="t1", agent_id="b", name="X"))
        assert sched.agent_count == 2
