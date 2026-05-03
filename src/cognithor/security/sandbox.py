"""Sandbox: Isolated execution environments.

Provides different isolation levels in which the executor
can run code. The higher the level, the stronger the isolation.

Isolation levels [B§3.3]:
  L0 (PROCESS):   subprocess + ulimit (Unix) / Job Objects (Windows)
  L-JOB (JOBOBJECT): Windows Job Objects -- native Windows isolation
  L1 (NAMESPACE): nsjail/bubblewrap (Linux namespaces)
  L2 (CONTAINER): Docker with resource limits
  L3 (VM):        Reserved (not implemented)

Bible reference: §3.3 (Sandbox), §11.1 (Security Architecture)
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from cognithor.models import SandboxConfig, SandboxLevel
from cognithor.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class SandboxResult:
    """Result of a sandbox execution."""

    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    sandbox_level: SandboxLevel
    killed: bool = False
    oom_killed: bool = False
    timed_out: bool = False
    isolation_degraded: bool = False


class Sandbox:
    """Multi-level sandbox for isolated execution. [B§3.3]

    Automatically selects the highest available isolation level
    or uses the explicitly configured one.
    """

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self._config = config or SandboxConfig()
        self._capabilities = self._detect_capabilities()

    def _detect_capabilities(self) -> dict[str, bool]:
        """Detects available sandbox tools."""
        return {
            "process": True,  # Always available
            "jobobject": sys.platform == "win32",  # Windows Job Objects
            "bwrap": shutil.which("bwrap") is not None,
            "nsjail": shutil.which("nsjail") is not None,
            "docker": shutil.which("docker") is not None,
        }

    @property
    def available_levels(self) -> list[SandboxLevel]:
        """Available isolation levels."""
        levels = [SandboxLevel.PROCESS]
        if self._capabilities["jobobject"]:
            levels.append(SandboxLevel.JOBOBJECT)
        if self._capabilities["bwrap"] or self._capabilities["nsjail"]:
            levels.append(SandboxLevel.NAMESPACE)
        if self._capabilities["docker"]:
            levels.append(SandboxLevel.CONTAINER)
        return levels

    @property
    def max_level(self) -> SandboxLevel:
        """Highest available isolation level."""
        levels = self.available_levels
        return levels[-1]

    @property
    def capabilities(self) -> dict[str, bool]:
        """Detected sandbox capabilities."""
        return dict(self._capabilities)

    async def execute(
        self,
        command: str,
        *,
        level: SandboxLevel | None = None,
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
        network: bool | None = None,
    ) -> SandboxResult:
        """Executes a command in the sandbox.

        Args:
            command: Shell command.
            level: Desired isolation level (None = config default).
            working_dir: Working directory.
            env: Additional environment variables.
            timeout: Timeout in seconds (None = config default).
            network: Allow network access (None = config default).

        Returns:
            SandboxResult.
        """
        effective_level = level or self._config.level
        effective_timeout = timeout or self._config.timeout_seconds
        effective_network = network if network is not None else self._config.network_access

        # Downgrade if level is not available
        if effective_level not in self.available_levels:
            old = effective_level
            effective_level = self.max_level
            log.warning(
                "sandbox_level_downgrade",
                requested=old.value,
                actual=effective_level.value,
            )

        log.debug(
            "sandbox_execute",
            level=effective_level.value,
            command=command[:100],
            timeout=effective_timeout,
        )

        start = datetime.now(UTC)

        if effective_level == SandboxLevel.CONTAINER:
            result = await self._exec_docker(
                command,
                working_dir=working_dir,
                env=env,
                timeout=effective_timeout,
                network=effective_network,
            )
        elif effective_level == SandboxLevel.NAMESPACE:
            result = await self._exec_namespace(
                command,
                working_dir=working_dir,
                env=env,
                timeout=effective_timeout,
                network=effective_network,
            )
        elif effective_level == SandboxLevel.JOBOBJECT:
            result = await self._exec_with_jobobject(
                command,
                working_dir=working_dir,
                env=env,
                timeout=effective_timeout,
            )
        else:
            result = await self._exec_process(
                command,
                working_dir=working_dir,
                env=env,
                timeout=effective_timeout,
            )

        elapsed = (datetime.now(UTC) - start).total_seconds()
        result.duration_ms = int(elapsed * 1000)
        result.sandbox_level = effective_level

        log.info(
            "sandbox_result",
            level=effective_level.value,
            exit_code=result.exit_code,
            duration_ms=result.duration_ms,
            killed=result.killed,
            timed_out=result.timed_out,
        )
        return result

    # ========================================================================
    # L0: Process-Level (subprocess + ulimit)
    # ========================================================================

    async def _exec_process(
        self,
        command: str,
        *,
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int = 30,
    ) -> SandboxResult:
        """Executes command as subprocess with resource limits.

        On Windows, Job Objects are used for resource limits
        (ulimit is not available). On Unix, ulimit is used.
        """
        merged_env = self._build_env(env)

        if sys.platform == "win32":
            # Windows: Job Objects fuer Resource-Limits nutzen
            return await self._exec_process_with_jobobject(
                command,
                working_dir=working_dir,
                env=merged_env,
                timeout=timeout,
            )

        # Unix: Resource-Limits
        import resource as _resource

        mem_bytes = self._config.max_memory_mb * 1024 * 1024
        cpu_seconds = self._config.max_cpu_seconds

        cmd_args = shlex.split(command)

        # Prefer prlimit (Linux, fork-safe) over preexec_fn (macOS fallback)
        _has_prlimit = hasattr(_resource, "prlimit")

        def _set_limits() -> None:
            """preexec_fn fallback for macOS where prlimit is unavailable.
            Only calls setrlimit which is async-signal-safe."""
            with contextlib.suppress(ValueError, OSError):
                _resource.setrlimit(_resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            with contextlib.suppress(ValueError, OSError):
                _resource.setrlimit(_resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))

        extra_kwargs: dict[str, Any] = {"process_group": 0}
        if not _has_prlimit:
            extra_kwargs["preexec_fn"] = _set_limits

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
                env=merged_env,
                **extra_kwargs,
            )
            # Apply resource limits via prlimit after fork (Linux only, fork-safe)
            if _has_prlimit:
                try:
                    _resource.prlimit(proc.pid, _resource.RLIMIT_AS, (mem_bytes, mem_bytes))
                    _resource.prlimit(proc.pid, _resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
                except (ValueError, OSError, PermissionError):
                    pass  # Best-effort: limits may fail for non-root
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
                return SandboxResult(
                    exit_code=proc.returncode or 0,
                    stdout=stdout_bytes.decode(errors="replace"),
                    stderr=stderr_bytes.decode(errors="replace"),
                    duration_ms=0,
                    sandbox_level=SandboxLevel.PROCESS,
                )
            except TimeoutError:
                proc.kill()
                await proc.wait()
                return SandboxResult(
                    exit_code=-1,
                    stdout="",
                    stderr=f"Timeout after {timeout}s",
                    duration_ms=timeout * 1000,
                    sandbox_level=SandboxLevel.PROCESS,
                    killed=True,
                    timed_out=True,
                )
        except OSError as exc:
            return SandboxResult(
                exit_code=-1,
                stdout="",
                stderr=str(exc),
                duration_ms=0,
                sandbox_level=SandboxLevel.PROCESS,
            )

    async def _exec_process_with_jobobject(
        self,
        command: str,
        *,
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int = 30,
    ) -> SandboxResult:
        """Executes command as subprocess with Windows Job Object resource limits.

        Used internally by _exec_process on Windows to replace ulimit.
        """
        import ctypes

        from cognithor.utils.win32_job import (
            JOB_OBJECT_LIMIT_JOB_TIME,
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
            JOB_OBJECT_LIMIT_PROCESS_MEMORY,
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
            PROCESS_ALL_ACCESS,
            JobObjectExtendedLimitInformation,
        )

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined, unused-ignore]
        job_handle = None
        proc_handle = None

        try:
            # 1. Create Job Object
            job_handle = kernel32.CreateJobObjectW(None, None)
            if not job_handle:
                last_err = ctypes.get_last_error()  # type: ignore[attr-defined, unused-ignore]
                if not self._config.allow_degraded_sandbox:
                    log.error(
                        "jobobject_create_failed_execution_refused",
                        error=last_err,
                    )
                    return SandboxResult(
                        exit_code=-1,
                        stdout="",
                        stderr=(
                            "CreateJobObjectW failed and "
                            "allow_degraded_sandbox=False -- "
                            "execution refused"
                        ),
                        duration_ms=0,
                        sandbox_level=SandboxLevel.PROCESS,
                        isolation_degraded=True,
                    )
                log.warning(
                    "jobobject_create_failed_degraded_fallback",
                    error=last_err,
                )
                # Fallback: execute without Job Object (timeout protection only)
                result = await self._exec_process_bare(
                    command, working_dir=working_dir, env=env, timeout=timeout
                )
                result.isolation_degraded = True
                return result

            # 2. Configure limits
            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = (
                JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                | JOB_OBJECT_LIMIT_PROCESS_MEMORY
                | JOB_OBJECT_LIMIT_JOB_TIME
            )

            # Memory limit
            info.ProcessMemoryLimit = self._config.max_memory_mb * 1024 * 1024

            # CPU time limit (100-nanosecond units)
            info.BasicLimitInformation.PerJobUserTimeLimit = (
                self._config.max_cpu_seconds * 10_000_000
            )

            # 3. Set limits on job
            kernel32.SetInformationJobObject(
                job_handle,
                JobObjectExtendedLimitInformation,
                ctypes.byref(info),
                ctypes.sizeof(info),
            )

            # 4. Start subprocess
            cmd_args = shlex.split(command)
            proc = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
                env=env,
            )

            # 5. Assign process to job
            proc_handle = kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, proc.pid)
            if proc_handle:
                kernel32.AssignProcessToJobObject(job_handle, proc_handle)

            # 6. Wait for completion
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
                return SandboxResult(
                    exit_code=proc.returncode or 0,
                    stdout=stdout_bytes.decode(errors="replace"),
                    stderr=stderr_bytes.decode(errors="replace"),
                    duration_ms=0,
                    sandbox_level=SandboxLevel.PROCESS,
                )
            except TimeoutError:
                proc.kill()
                await proc.wait()
                return SandboxResult(
                    exit_code=-1,
                    stdout="",
                    stderr=f"Timeout nach {timeout}s",
                    duration_ms=timeout * 1000,
                    sandbox_level=SandboxLevel.PROCESS,
                    killed=True,
                    timed_out=True,
                )
        except OSError as exc:
            return SandboxResult(
                exit_code=-1,
                stdout="",
                stderr=str(exc),
                duration_ms=0,
                sandbox_level=SandboxLevel.PROCESS,
            )
        finally:
            if proc_handle:
                kernel32.CloseHandle(proc_handle)
            if job_handle:
                kernel32.CloseHandle(job_handle)

    async def _exec_process_bare(
        self,
        command: str,
        *,
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int = 30,
    ) -> SandboxResult:
        """Bare subprocess without resource limits (Windows fallback)."""
        try:
            cmd_args = shlex.split(command)
            proc = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
                env=env,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
                return SandboxResult(
                    exit_code=proc.returncode or 0,
                    stdout=stdout_bytes.decode(errors="replace"),
                    stderr=stderr_bytes.decode(errors="replace"),
                    duration_ms=0,
                    sandbox_level=SandboxLevel.PROCESS,
                )
            except TimeoutError:
                proc.kill()
                await proc.wait()
                return SandboxResult(
                    exit_code=-1,
                    stdout="",
                    stderr=f"Timeout nach {timeout}s",
                    duration_ms=timeout * 1000,
                    sandbox_level=SandboxLevel.PROCESS,
                    killed=True,
                    timed_out=True,
                )
        except OSError as exc:
            return SandboxResult(
                exit_code=-1,
                stdout="",
                stderr=str(exc),
                duration_ms=0,
                sandbox_level=SandboxLevel.PROCESS,
            )

    # ========================================================================
    # L1: Namespace-Level (bubblewrap)
    # ========================================================================

    async def _exec_namespace(
        self,
        command: str,
        *,
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int = 30,
        network: bool = False,
    ) -> SandboxResult:
        """Executes command with bubblewrap (Linux namespaces)."""
        if not self._capabilities.get("bwrap"):
            log.warning("bwrap_not_available_fallback_to_process")
            return await self._exec_process(
                command, working_dir=working_dir, env=env, timeout=timeout
            )

        _created_tmp = False
        work_dir = working_dir
        if not work_dir:
            work_dir = tempfile.mkdtemp(prefix="jarvis_ns_")
            _created_tmp = True

        bwrap_args = [
            "bwrap",
        ]

        # System directories read-only (only if they exist)
        for sys_path in ["/usr", "/lib", "/lib64", "/bin", "/sbin"]:
            if os.path.isdir(sys_path):
                bwrap_args += ["--ro-bind", sys_path, sys_path]

        bwrap_args += [
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--bind",
            work_dir,
            work_dir,
            "--chdir",
            work_dir,
            "--die-with-parent",
            "--new-session",
        ]

        # Isolate network
        if not network:
            bwrap_args.append("--unshare-net")

        # Add allowed paths
        for allowed in self._config.allowed_paths:
            expanded = os.path.expanduser(allowed)
            if os.path.exists(expanded):
                bwrap_args.extend(["--bind", expanded, expanded])

        bwrap_args.extend(["--", "sh", "-c", command])

        merged_env = self._build_env(env)

        try:
            proc = await asyncio.create_subprocess_exec(
                *bwrap_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=merged_env,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
                return SandboxResult(
                    exit_code=proc.returncode or 0,
                    stdout=stdout_bytes.decode(errors="replace"),
                    stderr=stderr_bytes.decode(errors="replace"),
                    duration_ms=0,
                    sandbox_level=SandboxLevel.NAMESPACE,
                )
            except TimeoutError:
                proc.kill()
                await proc.wait()
                return SandboxResult(
                    exit_code=-1,
                    stdout="",
                    stderr=f"Namespace-Timeout nach {timeout}s",
                    duration_ms=timeout * 1000,
                    sandbox_level=SandboxLevel.NAMESPACE,
                    killed=True,
                    timed_out=True,
                )
        except OSError as exc:
            return SandboxResult(
                exit_code=-1,
                stdout="",
                stderr=str(exc),
                duration_ms=0,
                sandbox_level=SandboxLevel.NAMESPACE,
            )
        finally:
            if _created_tmp:
                import shutil

                shutil.rmtree(work_dir, ignore_errors=True)

    # ========================================================================
    # L2: Container-Level (Docker)
    # ========================================================================

    async def _exec_docker(
        self,
        command: str,
        *,
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int = 30,
        network: bool = False,
    ) -> SandboxResult:
        """Executes command in a Docker container."""
        if not self._capabilities.get("docker"):
            log.warning("docker_not_available_fallback")
            return await self._exec_namespace(
                command,
                working_dir=working_dir,
                env=env,
                timeout=timeout,
                network=network,
            )

        docker_args = [
            "docker",
            "run",
            "--rm",
            "--memory",
            f"{self._config.max_memory_mb}m",
            "--cpus",
            str(self._config.max_cpu_cores),
            "--pids-limit",
            "64",
            "--read-only",
            "--tmpfs",
            "/tmp:size=64m",
            "--security-opt",
            "no-new-privileges",
        ]

        if not network:
            docker_args.extend(["--network", "none"])

        # Mount working directory
        if working_dir and os.path.exists(working_dir):
            docker_args.extend(["-v", f"{working_dir}:/workspace", "-w", "/workspace"])

        # Environment variables
        merged_env = env or {}
        for k, v in merged_env.items():
            docker_args.extend(["-e", f"{k}={v}"])

        docker_args.extend(["python:3.12-slim", "sh", "-c", command])

        try:
            proc = await asyncio.create_subprocess_exec(
                *docker_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
                return SandboxResult(
                    exit_code=proc.returncode or 0,
                    stdout=stdout_bytes.decode(errors="replace"),
                    stderr=stderr_bytes.decode(errors="replace"),
                    duration_ms=0,
                    sandbox_level=SandboxLevel.CONTAINER,
                )
            except TimeoutError:
                proc.kill()
                await proc.wait()
                return SandboxResult(
                    exit_code=-1,
                    stdout="",
                    stderr=f"Container-Timeout nach {timeout}s",
                    duration_ms=timeout * 1000,
                    sandbox_level=SandboxLevel.CONTAINER,
                    killed=True,
                    timed_out=True,
                )
        except OSError as exc:
            return SandboxResult(
                exit_code=-1,
                stdout="",
                stderr=str(exc),
                duration_ms=0,
                sandbox_level=SandboxLevel.CONTAINER,
            )

    # ========================================================================
    # L-JOB: Windows Job Object Level
    # ========================================================================

    async def _exec_with_jobobject(
        self,
        command: str,
        *,
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int = 30,
    ) -> SandboxResult:
        """Executes command with Windows Job Object isolation.

        Provides memory, CPU, and process limits via Win32 Job Objects.
        Stronger than PROCESS level, but weaker than NAMESPACE/CONTAINER.
        """
        import ctypes

        from cognithor.utils.win32_job import (
            JOB_OBJECT_LIMIT_ACTIVE_PROCESS,
            JOB_OBJECT_LIMIT_JOB_TIME,
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
            JOB_OBJECT_LIMIT_PROCESS_MEMORY,
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
            PROCESS_ALL_ACCESS,
            JobObjectExtendedLimitInformation,
        )

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined, unused-ignore]
        job_handle = None
        proc_handle = None
        merged_env = self._build_env(env)

        try:
            # 1. Job Object erstellen
            job_handle = kernel32.CreateJobObjectW(None, None)
            if not job_handle:
                log.warning("jobobject_create_failed", error=ctypes.get_last_error())  # type: ignore[attr-defined, unused-ignore]
                return await self._exec_process(
                    command, working_dir=working_dir, env=env, timeout=timeout
                )

            # 2. Limits konfigurieren
            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = (
                JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                | JOB_OBJECT_LIMIT_PROCESS_MEMORY
                | JOB_OBJECT_LIMIT_ACTIVE_PROCESS
                | JOB_OBJECT_LIMIT_JOB_TIME
            )

            # Memory-Limit (Bytes)
            info.ProcessMemoryLimit = self._config.max_memory_mb * 1024 * 1024

            # CPU-Zeit-Limit (100-Nanosekunden-Einheiten)
            info.BasicLimitInformation.PerJobUserTimeLimit = (
                self._config.max_cpu_seconds * 10_000_000
            )

            # Prozess-Limit
            info.BasicLimitInformation.ActiveProcessLimit = 64

            # 3. Limits auf Job setzen
            success = kernel32.SetInformationJobObject(
                job_handle,
                JobObjectExtendedLimitInformation,
                ctypes.byref(info),
                ctypes.sizeof(info),
            )
            if not success:
                log.warning("jobobject_setinfo_failed", error=ctypes.get_last_error())  # type: ignore[attr-defined, unused-ignore]

            # 4. Subprocess starten
            _created_job_tmp = not working_dir
            work_dir = working_dir or tempfile.mkdtemp(prefix="jarvis_job_")
            cmd_args = shlex.split(command)

            proc = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=work_dir,
                env=merged_env,
            )

            # 5. Prozess dem Job zuweisen
            proc_handle = kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, proc.pid)
            if proc_handle:
                kernel32.AssignProcessToJobObject(job_handle, proc_handle)
            else:
                log.warning(
                    "jobobject_openprocess_failed",
                    pid=proc.pid,
                    error=ctypes.get_last_error(),  # type: ignore[attr-defined, unused-ignore]
                )

            # 6. Auf Abschluss warten
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
                return SandboxResult(
                    exit_code=proc.returncode or 0,
                    stdout=stdout_bytes.decode(errors="replace"),
                    stderr=stderr_bytes.decode(errors="replace"),
                    duration_ms=0,
                    sandbox_level=SandboxLevel.JOBOBJECT,
                )
            except TimeoutError:
                proc.kill()
                await proc.wait()
                return SandboxResult(
                    exit_code=-1,
                    stdout="",
                    stderr=f"Job Object Timeout nach {timeout}s",
                    duration_ms=timeout * 1000,
                    sandbox_level=SandboxLevel.JOBOBJECT,
                    killed=True,
                    timed_out=True,
                )
        except OSError as exc:
            return SandboxResult(
                exit_code=-1,
                stdout="",
                stderr=str(exc),
                duration_ms=0,
                sandbox_level=SandboxLevel.JOBOBJECT,
            )
        finally:
            if proc_handle:
                kernel32.CloseHandle(proc_handle)
            if job_handle:
                kernel32.CloseHandle(job_handle)
            if _created_job_tmp:
                import shutil

                shutil.rmtree(work_dir, ignore_errors=True)

    # ========================================================================
    # Helper methods
    # ========================================================================

    def _build_env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        """Builds safe environment variables."""
        # Minimal environment
        if sys.platform == "win32":
            _sysroot = os.environ.get("SYSTEMROOT", r"C:\Windows")
            _minimal_path = os.pathsep.join(
                [
                    os.path.dirname(sys.executable),
                    os.path.join(_sysroot, "System32"),
                    _sysroot,
                ]
            )
            safe_env: dict[str, str] = {
                "PATH": _minimal_path,
                "HOME": str(os.path.expanduser("~")),
                "SYSTEMROOT": _sysroot,
            }
        else:
            safe_env = {
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "HOME": "/tmp",
                "LANG": "C.UTF-8",
            }
        # Config-Env
        for k, v in self._config.env_vars.items():
            safe_env[k] = v
        # Extra-Env
        if extra:
            safe_env.update(extra)
        return safe_env
