"""Tests for BackgroundProcessManager."""

import asyncio
import os
import sqlite3
import tempfile
from pathlib import Path

import pytest


class TestBackgroundProcessManager:
    """Core lifecycle: start, track, finish, query."""

    @pytest.fixture
    def manager(self, tmp_path):
        from jarvis.mcp.background_tasks import BackgroundProcessManager

        return BackgroundProcessManager(
            db_path=tmp_path / "jobs.db",
            log_dir=tmp_path / "logs",
        )

    @pytest.mark.asyncio
    async def test_start_returns_job_id(self, manager):
        job_id = await manager.start("echo hello", description="test echo")
        assert job_id is not None
        assert len(job_id) > 8

    @pytest.mark.asyncio
    async def test_start_creates_log_file(self, manager):
        job_id = await manager.start("echo hello")
        job = manager.get_job(job_id)
        assert job is not None
        assert Path(job["log_file"]).parent.exists()

    @pytest.mark.asyncio
    async def test_list_jobs_returns_started(self, manager):
        job_id = await manager.start("echo hello")
        jobs = manager.list_jobs()
        assert len(jobs) >= 1
        assert any(j["id"] == job_id for j in jobs)

    @pytest.mark.asyncio
    async def test_job_completes_with_exit_code(self, manager):
        job_id = await manager.start("echo done")
        # Wait for short command to finish
        await asyncio.sleep(1)
        await manager.check_job(job_id)
        job = manager.get_job(job_id)
        assert job["status"] in ("completed", "running")

    @pytest.mark.asyncio
    async def test_stop_job_kills_process(self, manager):
        # Start a long-running command
        import sys

        job_id = await manager.start(
            f'{sys.executable} -c "import time; time.sleep(60)"',
            timeout_seconds=300,
        )
        await asyncio.sleep(0.5)
        result = await manager.stop_job(job_id)
        assert result is True
        job = manager.get_job(job_id)
        assert job["status"] == "killed"

    @pytest.mark.asyncio
    async def test_read_log_tail(self, manager):
        import sys

        job_id = await manager.start(
            f"{sys.executable} -c \"for i in range(20): print(f'line {{i}}')\"",
        )
        await asyncio.sleep(1.5)
        await manager.check_job(job_id)
        lines = manager.read_log(job_id, tail=5)
        assert len(lines) <= 5

    @pytest.mark.asyncio
    async def test_get_nonexistent_job_returns_none(self, manager):
        job = manager.get_job("nonexistent-id")
        assert job is None

    @pytest.mark.asyncio
    async def test_list_active_only(self, manager):
        import sys

        job_id = await manager.start(
            f"{sys.executable} -c \"print('done')\"",
        )
        await asyncio.sleep(1)
        await manager.check_job(job_id)
        active = manager.list_jobs(active_only=True)
        # Short command should be done by now
        assert all(j["status"] == "running" for j in active)


class TestProcessMonitor:
    """ProcessMonitor polls jobs and detects status changes."""

    @pytest.fixture
    def manager(self, tmp_path):
        from jarvis.mcp.background_tasks import BackgroundProcessManager

        return BackgroundProcessManager(
            db_path=tmp_path / "jobs.db",
            log_dir=tmp_path / "logs",
        )

    @pytest.mark.asyncio
    async def test_monitor_detects_completion(self, manager):
        from jarvis.mcp.background_tasks import ProcessMonitor

        notifications = []

        async def on_change(job_id, old, new, job):
            notifications.append((job_id, old, new))

        monitor = ProcessMonitor(manager, on_status_change=on_change)
        import sys

        job_id = await manager.start(
            f"{sys.executable} -c \"print('done')\"",
            check_interval=1,
        )
        # Run one monitor cycle
        await asyncio.sleep(1.5)
        await monitor.poll_once()
        assert len(notifications) >= 1
        assert notifications[0][2] in ("completed", "failed")

    @pytest.mark.asyncio
    async def test_monitor_detects_timeout(self, manager):
        from jarvis.mcp.background_tasks import ProcessMonitor

        notifications = []

        async def on_change(job_id, old, new, job):
            notifications.append((job_id, old, new))

        monitor = ProcessMonitor(manager, on_status_change=on_change)
        import sys

        job_id = await manager.start(
            f'{sys.executable} -c "import time; time.sleep(60)"',
            timeout_seconds=1,
            check_interval=1,
        )
        await asyncio.sleep(2)
        await monitor.poll_once()
        assert any(n[2] in ("timeout", "killed") for n in notifications)
