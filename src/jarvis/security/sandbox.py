"""Sandbox: Isolierte Ausführungsumgebungen.

Stellt verschiedene Isolierungsstufen bereit, in denen
der Executor Code ausführen kann. Je höher die Stufe,
desto stärker die Isolation.

Isolierungsstufen [B§3.3]:
  L0 (PROCESS):   subprocess + ulimit (Unix) / Job Objects (Windows)
  L-JOB (JOBOBJECT): Windows Job Objects -- native Windows-Isolation
  L1 (NAMESPACE): nsjail/bubblewrap (Linux-Namespaces)
  L2 (CONTAINER): Docker mit Resource-Limits
  L3 (VM):        Reserved (nicht implementiert)

Bibel-Referenz: §3.3 (Sandbox), §11.1 (Sicherheitsarchitektur)
"""

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime

from jarvis.models import SandboxConfig, SandboxLevel
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class SandboxResult:
    """Ergebnis einer Sandbox-Ausführung."""

    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    sandbox_level: SandboxLevel
    killed: bool = False
    oom_killed: bool = False
    timed_out: bool = False


class Sandbox:
    """Multi-Level Sandbox für isolierte Ausführung. [B§3.3]

    Wählt automatisch die höchste verfügbare Isolierungsstufe
    oder verwendet die explizit konfigurierte.
    """

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self._config = config or SandboxConfig()
        self._capabilities = self._detect_capabilities()

    def _detect_capabilities(self) -> dict[str, bool]:
        """Erkennt verfügbare Sandbox-Tools."""
        return {
            "process": True,  # Immer verfügbar
            "jobobject": sys.platform == "win32",  # Windows Job Objects
            "bwrap": shutil.which("bwrap") is not None,
            "nsjail": shutil.which("nsjail") is not None,
            "docker": shutil.which("docker") is not None,
        }

    @property
    def available_levels(self) -> list[SandboxLevel]:
        """Verfügbare Isolierungsstufen."""
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
        """Höchste verfügbare Isolierungsstufe."""
        levels = self.available_levels
        return levels[-1]

    @property
    def capabilities(self) -> dict[str, bool]:
        """Erkannte Sandbox-Fähigkeiten."""
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
        """Führt einen Befehl in der Sandbox aus.

        Args:
            command: Shell-Befehl.
            level: Gewünschte Isolierungsstufe (None = Config-Default).
            working_dir: Arbeitsverzeichnis.
            env: Zusätzliche Umgebungsvariablen.
            timeout: Timeout in Sekunden (None = Config-Default).
            network: Netzwerkzugriff erlauben (None = Config-Default).

        Returns:
            SandboxResult.
        """
        effective_level = level or self._config.level
        effective_timeout = timeout or self._config.timeout_seconds
        effective_network = network if network is not None else self._config.network_access

        # Downgrade wenn Level nicht verfügbar
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
        """Führt Befehl als subprocess mit Resource-Limits aus.

        Auf Windows werden Job Objects für Resource-Limits genutzt
        (ulimit ist nicht verfügbar). Auf Unix wird ulimit verwendet.
        """
        merged_env = self._build_env(env)

        if sys.platform == "win32":
            # Windows: Job Objects für Resource-Limits nutzen
            return await self._exec_process_with_jobobject(
                command,
                working_dir=working_dir,
                env=merged_env,
                timeout=timeout,
            )

        # Unix: Resource-Limits via preexec_fn (resource module)
        import resource as _resource

        mem_bytes = self._config.max_memory_mb * 1024 * 1024
        cpu_seconds = self._config.max_cpu_seconds

        def _set_limits() -> None:
            try:
                _resource.setrlimit(_resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            except (ValueError, OSError):
                pass
            try:
                _resource.setrlimit(_resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
            except (ValueError, OSError):
                pass

        cmd_args = shlex.split(command)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
                env=merged_env,
                preexec_fn=_set_limits,
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

    async def _exec_process_with_jobobject(
        self,
        command: str,
        *,
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int = 30,
    ) -> SandboxResult:
        """Führt Befehl als subprocess mit Windows Job Object Resource-Limits aus.

        Wird intern von _exec_process auf Windows verwendet, um ulimit zu ersetzen.
        """
        import ctypes
        import ctypes.wintypes

        # Win32 Constants
        JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
        JOB_OBJECT_LIMIT_JOB_TIME = 0x00000004
        JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
        JobObjectExtendedLimitInformation = 9
        PROCESS_ALL_ACCESS = 0x001F0FFF

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", ctypes.wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", ctypes.wintypes.DWORD),
                ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
                ("PriorityClass", ctypes.wintypes.DWORD),
                ("SchedulingClass", ctypes.wintypes.DWORD),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.windll.kernel32
        job_handle = None
        proc_handle = None

        try:
            # 1. Job Object erstellen
            job_handle = kernel32.CreateJobObjectW(None, None)
            if not job_handle:
                log.warning("jobobject_create_failed", error=ctypes.get_last_error())
                # Fallback: ohne Job Object ausführen
                return await self._exec_process_bare(
                    command, working_dir=working_dir, env=env, timeout=timeout
                )

            # 2. Limits konfigurieren
            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = (
                JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                | JOB_OBJECT_LIMIT_PROCESS_MEMORY
                | JOB_OBJECT_LIMIT_JOB_TIME
            )

            # Memory-Limit
            info.ProcessMemoryLimit = self._config.max_memory_mb * 1024 * 1024

            # CPU-Zeit-Limit (100-Nanosekunden-Einheiten)
            info.BasicLimitInformation.PerJobUserTimeLimit = (
                self._config.max_cpu_seconds * 10_000_000
            )

            # 3. Limits auf Job setzen
            kernel32.SetInformationJobObject(
                job_handle,
                JobObjectExtendedLimitInformation,
                ctypes.byref(info),
                ctypes.sizeof(info),
            )

            # 4. Subprocess starten
            cmd_args = shlex.split(command)
            proc = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
                env=env,
            )

            # 5. Prozess dem Job zuweisen
            proc_handle = kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, proc.pid)
            if proc_handle:
                kernel32.AssignProcessToJobObject(job_handle, proc_handle)

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
        """Bare subprocess ohne Resource-Limits (Windows-Fallback)."""
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
        """Führt Befehl mit bubblewrap (Linux-Namespaces) aus."""
        if not self._capabilities.get("bwrap"):
            log.warning("bwrap_not_available_fallback_to_process")
            return await self._exec_process(
                command, working_dir=working_dir, env=env, timeout=timeout
            )

        work_dir = working_dir or tempfile.mkdtemp(prefix="jarvis_ns_")

        bwrap_args = [
            "bwrap",
        ]

        # Systemverzeichnisse read-only (nur wenn vorhanden)
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

        # Netzwerk isolieren
        if not network:
            bwrap_args.append("--unshare-net")

        # Erlaubte Pfade hinzufügen
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
        """Führt Befehl in einem Docker-Container aus."""
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
            str(self._config.max_cpu_seconds / 10),
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

        # Arbeitsverzeichnis mounten
        if working_dir and os.path.exists(working_dir):
            docker_args.extend(["-v", f"{working_dir}:/workspace", "-w", "/workspace"])

        # Env-Variablen
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
        """Führt Befehl mit Windows Job Object Isolation aus.

        Bietet Memory-, CPU- und Prozess-Limits via Win32 Job Objects.
        Stärker als PROCESS-Level, aber schwächer als NAMESPACE/CONTAINER.
        """
        import ctypes
        import ctypes.wintypes

        # Win32 Constants
        JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
        JOB_OBJECT_LIMIT_JOB_TIME = 0x00000004
        JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
        JobObjectExtendedLimitInformation = 9
        PROCESS_ALL_ACCESS = 0x001F0FFF

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", ctypes.wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", ctypes.wintypes.DWORD),
                ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
                ("PriorityClass", ctypes.wintypes.DWORD),
                ("SchedulingClass", ctypes.wintypes.DWORD),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.windll.kernel32
        job_handle = None
        proc_handle = None
        merged_env = self._build_env(env)

        try:
            # 1. Job Object erstellen
            job_handle = kernel32.CreateJobObjectW(None, None)
            if not job_handle:
                log.warning("jobobject_create_failed", error=ctypes.get_last_error())
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
                log.warning(
                    "jobobject_setinfo_failed", error=ctypes.get_last_error()
                )

            # 4. Subprocess starten
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
                    error=ctypes.get_last_error(),
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

    # ========================================================================
    # Hilfsmethoden
    # ========================================================================

    def _build_env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        """Baut sichere Umgebungsvariablen zusammen."""
        # Minimales Environment
        if sys.platform == "win32":
            safe_env: dict[str, str] = {
                "PATH": os.environ.get("PATH", ""),
                "HOME": str(os.path.expanduser("~")),
                "SYSTEMROOT": os.environ.get("SYSTEMROOT", r"C:\Windows"),
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
