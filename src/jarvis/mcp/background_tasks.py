"""Background Task Manager — long-running shell commands with monitoring.

Enables the Planner to start shell commands in the background, monitor
their progress via 5 health-check methods, and read/tail output logs.

Architecture:
  BackgroundProcessManager: SQLite registry + subprocess spawning
  ProcessMonitor: Async polling loop with 5 verification methods
  6 MCP Tools: start, list, check, read_log, stop, wait
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import signal
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.audit import AuditLogger

log = get_logger(__name__)

__all__ = [
    "BackgroundProcessManager",
    "ProcessMonitor",
    "register_background_tools",
]

# Limits
MAX_LOG_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB per log file
LOG_CLEANUP_DAYS = 7  # Delete logs of finished jobs older than 7 days
DEFAULT_TIMEOUT = 3600  # 1 hour
DEFAULT_CHECK_INTERVAL = 30  # seconds


# ============================================================================
# SQLite Schema
# ============================================================================

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS background_jobs (
    id TEXT PRIMARY KEY,
    command TEXT NOT NULL,
    description TEXT DEFAULT '',
    agent_name TEXT DEFAULT 'jarvis',
    session_id TEXT DEFAULT '',
    channel TEXT DEFAULT '',
    pid INTEGER,
    status TEXT DEFAULT 'running',
    exit_code INTEGER,
    started_at REAL NOT NULL,
    finished_at REAL,
    timeout_seconds INTEGER DEFAULT 3600,
    check_interval INTEGER DEFAULT 30,
    log_file TEXT NOT NULL,
    last_check_at REAL,
    last_output_size INTEGER DEFAULT 0,
    working_dir TEXT DEFAULT ''
);
"""


# ============================================================================
# BackgroundProcessManager
# ============================================================================


class BackgroundProcessManager:
    """Manages background shell processes with SQLite persistence."""

    def __init__(
        self,
        db_path: Path | str,
        log_dir: Path | str,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._audit = audit_logger
        self._processes: dict[str, subprocess.Popen] = {}
        self._init_db()

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(_SCHEMA)
            # Mark orphaned jobs from previous sessions
            conn.execute(
                "UPDATE background_jobs SET status = 'orphaned' "
                "WHERE status = 'running'"
            )
            conn.commit()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    # -- Start --------------------------------------------------------------

    async def start(
        self,
        command: str,
        *,
        description: str = "",
        timeout_seconds: int = DEFAULT_TIMEOUT,
        check_interval: int = DEFAULT_CHECK_INTERVAL,
        working_dir: str = "",
        agent_name: str = "jarvis",
        session_id: str = "",
        channel: str = "",
    ) -> str:
        """Start a command in the background. Returns job_id."""
        job_id = f"bg_{uuid.uuid4().hex[:12]}"
        log_file = self._log_dir / f"{job_id}.log"
        cwd = working_dir or None

        loop = asyncio.get_running_loop()

        def _spawn() -> tuple[subprocess.Popen, Any]:
            lf = open(log_file, "w", encoding="utf-8", errors="replace")
            kwargs: dict[str, Any] = {
                "stdout": lf,
                "stderr": subprocess.STDOUT,
                "cwd": cwd,
                "start_new_session": True,
            }
            if sys.platform == "win32":
                kwargs["creationflags"] = (
                    subprocess.CREATE_NEW_PROCESS_GROUP
                    | getattr(subprocess, "CREATE_NO_WINDOW", 0)
                )
                kwargs.pop("start_new_session", None)
            if sys.platform == "win32":
                p = subprocess.Popen(command, shell=True, **kwargs)
            else:
                p = subprocess.Popen(["bash", "-c", command], **kwargs)
            return p, lf

        proc, log_handle = await loop.run_in_executor(None, _spawn)
        # Close parent's copy of the log handle — subprocess inherited the fd
        log_handle.close()
        self._processes[job_id] = proc

        with self._conn() as conn:
            conn.execute(
                "INSERT INTO background_jobs "
                "(id, command, description, agent_name, session_id, channel, "
                " pid, status, started_at, timeout_seconds, check_interval, "
                " log_file, last_check_at, last_output_size, working_dir) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    job_id, command, description, agent_name, session_id,
                    channel, proc.pid, "running", time.time(),
                    timeout_seconds, check_interval, str(log_file),
                    time.time(), 0, working_dir or "",
                ),
            )

        if self._audit:
            self._audit.log_tool_call(
                "start_background",
                {"command": command[:200], "job_id": job_id},
                agent_name=agent_name,
                result=f"Started as PID {proc.pid}",
                success=True,
            )

        log.info(
            "background_job_started",
            job_id=job_id,
            pid=proc.pid,
            command=command[:100],
        )
        return job_id

    # -- Query --------------------------------------------------------------

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM background_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_jobs(self, active_only: bool = False) -> list[dict[str, Any]]:
        with self._conn() as conn:
            if active_only:
                rows = conn.execute(
                    "SELECT * FROM background_jobs WHERE status = 'running' "
                    "ORDER BY started_at DESC"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM background_jobs ORDER BY started_at DESC LIMIT 50"
                ).fetchall()
        return [dict(r) for r in rows]

    # -- Check (single job) -------------------------------------------------

    async def check_job(self, job_id: str) -> dict[str, Any] | None:
        """Check a job's status using multiple verification methods."""
        job = self.get_job(job_id)
        if not job or job["status"] != "running":
            return job

        proc = self._processes.get(job_id)
        now = time.time()
        new_status = None
        exit_code = None

        # Method 1: Process alive check
        if proc is not None:
            retcode = proc.poll()
            if retcode is not None:
                exit_code = retcode
                new_status = "completed" if retcode == 0 else "failed"
        else:
            # Orphaned -- check via OS
            pid = job.get("pid")
            if pid and not self._pid_alive(pid):
                new_status = "orphaned"

        # Method 2: Timeout detection
        if new_status is None:
            elapsed = now - job["started_at"]
            if elapsed > job["timeout_seconds"]:
                await self.stop_job(job_id, force=True)
                new_status = "timeout"

        # Method 3: Output stall detection
        log_path = Path(job["log_file"])
        current_size = log_path.stat().st_size if log_path.exists() else 0
        stalled = False
        if (
            new_status is None
            and job["last_output_size"] == current_size
            and (now - (job["last_check_at"] or now)) > job["check_interval"] * 2
        ):
            stalled = True

        # Update DB
        with self._conn() as conn:
            if new_status:
                conn.execute(
                    "UPDATE background_jobs SET status=?, exit_code=?, "
                    "finished_at=?, last_check_at=?, last_output_size=? "
                    "WHERE id=?",
                    (new_status, exit_code, now, now, current_size, job_id),
                )
                # Cleanup process ref
                self._processes.pop(job_id, None)

                if self._audit:
                    self._audit.log_system(
                        f"Background job {job_id} {new_status} "
                        f"(exit_code={exit_code})",
                    )
            else:
                conn.execute(
                    "UPDATE background_jobs SET last_check_at=?, "
                    "last_output_size=? WHERE id=?",
                    (now, current_size, job_id),
                )

        result = self.get_job(job_id)
        if result and stalled:
            result["_stalled"] = True
        return result

    # -- Stop ---------------------------------------------------------------

    async def stop_job(self, job_id: str, force: bool = False) -> bool:
        """Stop a running job. Returns True if killed."""
        job = self.get_job(job_id)
        if not job or job["status"] != "running":
            return False

        proc = self._processes.get(job_id)
        if proc is not None:
            try:
                if force or sys.platform == "win32":
                    proc.kill()
                else:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            except OSError:
                pass
        else:
            # Try OS-level kill
            pid = job.get("pid")
            if pid:
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError:
                    pass

        self._processes.pop(job_id, None)
        with self._conn() as conn:
            conn.execute(
                "UPDATE background_jobs SET status='killed', finished_at=? "
                "WHERE id=?",
                (time.time(), job_id),
            )

        if self._audit:
            self._audit.log_tool_call(
                "stop_background_job",
                {"job_id": job_id, "force": force},
                result="killed",
                success=True,
            )

        log.info("background_job_killed", job_id=job_id, force=force)
        return True

    # -- Wait ---------------------------------------------------------------

    async def wait_job(
        self, job_id: str, timeout: int = 300
    ) -> dict[str, Any] | None:
        """Wait for a job to complete. Returns final status."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = await self.check_job(job_id)
            if not job or job["status"] != "running":
                return job
            await asyncio.sleep(2)
        return self.get_job(job_id)

    # -- Log reading --------------------------------------------------------

    def read_log(
        self,
        job_id: str,
        *,
        tail: int = 0,
        head: int = 0,
        offset: int = 0,
        limit: int = 100,
        grep: str = "",
    ) -> list[str]:
        """Read log file lines with tail/head/grep support."""
        job = self.get_job(job_id)
        if not job:
            return []
        log_path = Path(job["log_file"])
        if not log_path.exists():
            return []

        try:
            all_lines = log_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError:
            return []

        if grep:
            pattern = re.compile(grep, re.IGNORECASE)
            all_lines = [ln for ln in all_lines if pattern.search(ln)]

        if tail > 0:
            return all_lines[-tail:]
        if head > 0:
            return all_lines[:head]
        return all_lines[offset : offset + limit]

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def cleanup_old_logs(self, max_age_days: int = LOG_CLEANUP_DAYS) -> int:
        """Delete log files for finished jobs older than max_age_days."""
        cutoff = time.time() - max_age_days * 86400
        removed = 0
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, log_file FROM background_jobs "
                "WHERE status != 'running' AND finished_at < ?",
                (cutoff,),
            ).fetchall()
            for row in rows:
                p = Path(row["log_file"])
                if p.exists():
                    p.unlink(missing_ok=True)
                    removed += 1
                conn.execute(
                    "DELETE FROM background_jobs WHERE id = ?", (row["id"],)
                )
        return removed


# ============================================================================
# ProcessMonitor
# ============================================================================


class ProcessMonitor:
    """Async polling loop that checks background jobs periodically.

    5 verification methods per job:
      1. Process-Alive (os.waitpid / proc.poll)
      2. Exit-Code (success vs failure)
      3. Output-Stall (log size unchanged for 2+ intervals)
      4. Timeout (elapsed > timeout_seconds)
      5. Resource-Check (optional, via psutil if available)

    Fires on_status_change callback when a job transitions.
    """

    def __init__(
        self,
        manager: BackgroundProcessManager,
        *,
        on_status_change: Any = None,
        default_interval: int = DEFAULT_CHECK_INTERVAL,
    ) -> None:
        self._manager = manager
        self._on_change = on_status_change
        self._default_interval = default_interval
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the monitor loop as an asyncio task."""
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="bg-process-monitor")
        log.info("process_monitor_started")

    async def stop(self) -> None:
        """Stop the monitor loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        log.info("process_monitor_stopped")

    async def poll_once(self) -> int:
        """Run a single poll cycle. Returns number of status changes."""
        changes = 0
        jobs = self._manager.list_jobs(active_only=True)
        for job in jobs:
            old_status = job["status"]
            updated = await self._manager.check_job(job["id"])
            if updated and updated["status"] != old_status:
                changes += 1
                if self._on_change:
                    try:
                        await self._on_change(
                            job["id"], old_status, updated["status"], updated
                        )
                    except Exception:
                        log.debug("monitor_callback_error", exc_info=True)

                # Method 5: Resource check (optional)
                self._check_resources(job)

            # Method 3 supplement: warn on stall
            if updated and updated.get("_stalled"):
                log.warning(
                    "background_job_stalled",
                    job_id=job["id"],
                    command=job["command"][:80],
                )
        return changes

    def _check_resources(self, job: dict) -> None:
        """Optional resource check via psutil."""
        try:
            import psutil
            pid = job.get("pid")
            if pid and psutil.pid_exists(pid):
                proc = psutil.Process(pid)
                mem_mb = proc.memory_info().rss / (1024 * 1024)
                cpu = proc.cpu_percent(interval=0.1)
                if mem_mb > 2048:
                    log.warning(
                        "background_job_high_memory",
                        job_id=job["id"],
                        mem_mb=round(mem_mb),
                    )
                if cpu > 95:
                    log.warning(
                        "background_job_high_cpu",
                        job_id=job["id"],
                        cpu_percent=round(cpu),
                    )
        except ImportError:
            pass  # psutil not installed, skip
        except Exception:
            pass  # Process may have exited

    async def _loop(self) -> None:
        """Main polling loop."""
        while self._running:
            try:
                await self.poll_once()
            except Exception:
                log.debug("process_monitor_poll_error", exc_info=True)
            await asyncio.sleep(self._default_interval)


# ============================================================================
# MCP Tool Registration
# ============================================================================


def register_background_tools(
    mcp_client: Any,
    config: Any,
    audit_logger: Any = None,
) -> BackgroundProcessManager:
    """Register background task MCP tools.

    Returns:
        BackgroundProcessManager instance (for gateway to start monitor).
    """
    jarvis_home = getattr(config, "jarvis_home", Path.home() / ".jarvis")
    manager = BackgroundProcessManager(
        db_path=jarvis_home / "background_jobs.db",
        log_dir=jarvis_home / "workspace" / "background_logs",
        audit_logger=audit_logger,
    )

    # Tool 1: start_background
    async def _start_background(**kwargs: Any) -> str:
        command = kwargs.get("command", "")
        if not command.strip():
            return "Error: 'command' is required."
        description = kwargs.get("description", "")
        timeout = int(kwargs.get("timeout_seconds", DEFAULT_TIMEOUT))
        interval = int(kwargs.get("check_interval", DEFAULT_CHECK_INTERVAL))
        job_id = await manager.start(
            command,
            description=description,
            timeout_seconds=timeout,
            check_interval=interval,
        )
        return (
            f"Background job started: {job_id}\n"
            f"Command: {command[:100]}\n"
            f"Timeout: {timeout}s, Check interval: {interval}s\n"
            f"Use check_background_job('{job_id}') to monitor."
        )

    mcp_client.register_builtin_handler(
        "start_background",
        _start_background,
        description=(
            "Start a shell command in the background. Returns a job_id for monitoring. "
            "Use for long-running tasks like downloads, builds, or training runs."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"},
                "description": {"type": "string", "description": "Human-readable description"},
                "timeout_seconds": {"type": "integer", "description": "Max runtime (default: 3600)", "default": 3600},
                "check_interval": {"type": "integer", "description": "Monitor poll interval in seconds (default: 30)", "default": 30},
            },
            "required": ["command"],
        },
    )

    # Tool 2: list_background_jobs
    async def _list_jobs(**kwargs: Any) -> str:
        active_only = kwargs.get("active_only", False)
        jobs = manager.list_jobs(active_only=bool(active_only))
        if not jobs:
            return "No background jobs found."
        lines = []
        for j in jobs:
            elapsed = time.time() - j["started_at"]
            elapsed_str = f"{elapsed:.0f}s" if elapsed < 3600 else f"{elapsed/3600:.1f}h"
            lines.append(
                f"  {j['id']} | {j['status']:10s} | {elapsed_str:>8s} | {j['command'][:60]}"
            )
        header = f"{'ID':>18s} | {'Status':10s} | {'Elapsed':>8s} | Command"
        return f"{header}\n" + "\n".join(lines)

    mcp_client.register_builtin_handler(
        "list_background_jobs",
        _list_jobs,
        description="List all background jobs (active and recent completed).",
        input_schema={
            "type": "object",
            "properties": {
                "active_only": {"type": "boolean", "description": "Only show running jobs", "default": False},
            },
        },
    )

    # Tool 3: check_background_job
    async def _check_job(**kwargs: Any) -> str:
        job_id = kwargs.get("job_id", "")
        if not job_id:
            return "Error: 'job_id' is required."
        job = await manager.check_job(job_id)
        if not job:
            return f"Job '{job_id}' not found."
        # Get last 20 lines of output
        recent = manager.read_log(job_id, tail=20)
        output_section = "\n".join(recent) if recent else "(no output yet)"
        stalled = " [STALLED - no new output]" if job.get("_stalled") else ""
        elapsed = time.time() - job["started_at"]
        return (
            f"Job: {job_id}\n"
            f"Status: {job['status']}{stalled}\n"
            f"PID: {job.get('pid', '?')}\n"
            f"Elapsed: {elapsed:.0f}s\n"
            f"Exit code: {job.get('exit_code', '-')}\n"
            f"--- Last 20 lines ---\n{output_section}"
        )

    mcp_client.register_builtin_handler(
        "check_background_job",
        _check_job,
        description="Check status and recent output of a background job.",
        input_schema={
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "Job ID from start_background"},
            },
            "required": ["job_id"],
        },
    )

    # Tool 4: read_background_log
    async def _read_log(**kwargs: Any) -> str:
        job_id = kwargs.get("job_id", "")
        if not job_id:
            return "Error: 'job_id' is required."
        tail = int(kwargs.get("tail", 0))
        head = int(kwargs.get("head", 0))
        offset = int(kwargs.get("offset", 0))
        limit = int(kwargs.get("limit", 100))
        grep = kwargs.get("grep", "")
        lines = manager.read_log(
            job_id, tail=tail, head=head, offset=offset, limit=limit, grep=grep
        )
        if not lines:
            return f"No output for job '{job_id}'."
        return "\n".join(lines)

    mcp_client.register_builtin_handler(
        "read_background_log",
        _read_log,
        description=(
            "Read the log/output of a background job. Supports tail (last N lines), "
            "head (first N lines), offset+limit pagination, and grep filtering."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "Job ID"},
                "tail": {"type": "integer", "description": "Last N lines (default: 0 = disabled)", "default": 0},
                "head": {"type": "integer", "description": "First N lines (default: 0 = disabled)", "default": 0},
                "offset": {"type": "integer", "description": "Skip first N lines", "default": 0},
                "limit": {"type": "integer", "description": "Max lines to return (default: 100)", "default": 100},
                "grep": {"type": "string", "description": "Regex filter pattern", "default": ""},
            },
            "required": ["job_id"],
        },
    )

    # Tool 5: stop_background_job
    async def _stop_job(**kwargs: Any) -> str:
        job_id = kwargs.get("job_id", "")
        if not job_id:
            return "Error: 'job_id' is required."
        force = bool(kwargs.get("force", False))
        ok = await manager.stop_job(job_id, force=force)
        return f"Job {job_id} stopped." if ok else f"Job {job_id} not running or not found."

    mcp_client.register_builtin_handler(
        "stop_background_job",
        _stop_job,
        description="Stop a running background job. Use force=true for immediate SIGKILL.",
        input_schema={
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "Job ID to stop"},
                "force": {"type": "boolean", "description": "Force kill (SIGKILL)", "default": False},
            },
            "required": ["job_id"],
        },
    )

    # Tool 6: wait_background_job
    async def _wait_job(**kwargs: Any) -> str:
        job_id = kwargs.get("job_id", "")
        if not job_id:
            return "Error: 'job_id' is required."
        timeout = int(kwargs.get("timeout", 300))
        job = await manager.wait_job(job_id, timeout=timeout)
        if not job:
            return f"Job '{job_id}' not found."
        return (
            f"Job {job_id}: {job['status']} (exit_code={job.get('exit_code', '-')})"
        )

    mcp_client.register_builtin_handler(
        "wait_background_job",
        _wait_job,
        description="Wait for a background job to complete (with timeout).",
        input_schema={
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "Job ID to wait for"},
                "timeout": {"type": "integer", "description": "Max wait time in seconds (default: 300)", "default": 300},
            },
            "required": ["job_id"],
        },
    )

    log.info(
        "background_tools_registered",
        tools=["start_background", "list_background_jobs", "check_background_job",
               "read_background_log", "stop_background_job", "wait_background_job"],
    )
    return manager
