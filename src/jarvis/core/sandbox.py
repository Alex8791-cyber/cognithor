"""Prozess-Sandbox: Echte Isolation für Shell-Befehle.

Drei Sandbox-Level (automatisches Fallback):
  1. bubblewrap (bwrap) — Linux-Namespaces, kein Root nötig
  2. firejail — Application Sandboxing
  3. bare — Kein Sandbox (nur Timeout + Output-Limit)

bubblewrap isoliert:
  - Dateisystem: Nur Workspace + /usr + /bin + /lib sichtbar
  - Netzwerk: Optional deaktivierbar
  - PID-Namespace: Prozesse sehen nur sich selbst
  - /tmp: Eigenes tmpfs pro Befehl
  - /home: Nur Workspace-Verzeichnis gemountet

Bibel-Referenz: §4.3 (Sandbox-Ausführung)
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import shutil
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from jarvis.utils.logging import get_logger

log = get_logger(__name__)

MAX_OUTPUT_BYTES = 50_000


# ============================================================================
# Konfiguration
# ============================================================================


class SandboxLevel(StrEnum):
    """Verfügbare Sandbox-Level."""

    BWRAP = "bwrap"  # bubblewrap — stärkste Isolation
    FIREJAIL = "firejail"  # Firejail — gute Isolation
    JOBOBJECT = "jobobject"  # Windows Job Objects — Windows-Isolation
    BARE = "bare"  # Kein Sandbox — nur Timeout


class NetworkPolicy(StrEnum):
    """Netzwerk-Richtlinien für die Sandbox."""

    ALLOW = "allow"  # Netzwerk erlaubt
    BLOCK = "block"  # Kein Netzwerk in der Sandbox


@dataclass
class SandboxConfig:
    """Sandbox-Konfiguration."""

    # Bevorzugtes Level (Fallback automatisch)
    preferred_level: SandboxLevel = SandboxLevel.BWRAP

    # Dateisystem
    workspace_dir: Path = field(default_factory=lambda: Path.home() / ".jarvis" / "workspace")
    allowed_read_paths: list[str] = field(default_factory=lambda: (
        [] if sys.platform == "win32" else
        [p for p in [
            "/usr", "/bin", "/sbin", "/lib",
            "/lib64",           # Linux only
            "/etc/alternatives",# Linux only
            "/etc/ssl", "/etc/resolv.conf", "/etc/hosts",
        ] if Path(p).exists()]
    ))
    allowed_write_paths: list[str] = field(default_factory=list)

    # Netzwerk
    network: NetworkPolicy = NetworkPolicy.ALLOW

    # Resource-Limits
    max_memory_mb: int = 512
    max_file_size_mb: int = 100
    max_processes: int = 64
    default_timeout: int = 30

    # Umgebungsvariablen die durchgereicht werden
    env_passthrough: list[str] = field(default_factory=lambda: [
        "PATH", "HOME", "LANG", "LC_ALL", "TERM",
    ])


# ============================================================================
# Sandbox-Ergebnis
# ============================================================================


@dataclass
class SandboxResult:
    """Ergebnis einer Sandbox-Ausführung."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    timed_out: bool = False
    sandbox_level: str = "bare"
    truncated: bool = False
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and self.error is None

    @property
    def output(self) -> str:
        """Kombinierter Output für den Agent."""
        parts: list[str] = []

        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(f"[STDERR]\n{self.stderr}")
        if self.exit_code != 0:
            parts.append(f"[EXIT CODE: {self.exit_code}]")
        if self.timed_out:
            parts.append("[TIMEOUT]")
        if self.truncated:
            parts.append("[... Output gekürzt]")
        if self.error:
            parts.append(f"[FEHLER: {self.error}]")

        return "\n".join(parts) if parts else "(Keine Ausgabe)"


# ============================================================================
# Sandbox-Implementierungen
# ============================================================================


class BwrapSandbox:
    """bubblewrap (bwrap) Sandbox — stärkste Isolation.

    Nutzt Linux-Namespaces für:
    - Mount-Namespace: Isoliertes Dateisystem
    - PID-Namespace: Eigener Prozessbaum
    - Network-Namespace: Optional kein Netzwerk
    - User-Namespace: Kein Root in der Sandbox
    """

    def __init__(self, config: SandboxConfig) -> None:
        self._config = config

    @staticmethod
    def is_available() -> bool:
        """Prüft ob bwrap installiert ist."""
        return shutil.which("bwrap") is not None

    def build_command(self, command: str, working_dir: str) -> list[str]:
        """Baut den bwrap-Befehl mit allen Mounts und Flags.

        Args:
            command: Auszuführender Shell-Befehl.
            working_dir: Arbeitsverzeichnis innerhalb der Sandbox.
        """
        cfg = self._config
        workspace = str(cfg.workspace_dir.resolve())

        args = ["bwrap"]

        # --- Dateisystem-Mounts ---

        # Basis: leeres Root
        args += ["--unshare-all"]

        # Nur PID unsharen wenn kein Netzwerk benötigt wird
        # (--unshare-all macht alles auf einmal)
        if cfg.network == NetworkPolicy.ALLOW:
            args += ["--share-net"]

        # /proc und /dev (minimal, für Prozess-Info)
        args += ["--proc", "/proc"]
        args += ["--dev", "/dev"]

        # Eigenes tmpfs für /tmp
        args += ["--tmpfs", "/tmp"]

        # Systemverzeichnisse read-only
        for path in cfg.allowed_read_paths:
            if Path(path).exists():
                args += ["--ro-bind", path, path]

        # Workspace read-write (das einzige beschreibbare Verzeichnis)
        Path(workspace).mkdir(parents=True, exist_ok=True)
        args += ["--bind", workspace, workspace]

        # Zusätzliche Write-Pfade
        for path in cfg.allowed_write_paths:
            if Path(path).exists():
                args += ["--bind", path, path]

        # HOME auf Workspace setzen
        args += ["--setenv", "HOME", workspace]

        # --- Umgebungsvariablen ---
        for var in cfg.env_passthrough:
            val = os.environ.get(var)
            if val:
                args += ["--setenv", var, val]

        # Deutsche Locale
        args += ["--setenv", "LANG", "de_DE.UTF-8"]
        args += ["--setenv", "LC_ALL", "de_DE.UTF-8"]

        # --- Resource-Limits via ulimit im Shell-Befehl ---
        ulimits = (
            f"ulimit -v {cfg.max_memory_mb * 1024} 2>/dev/null; "
            f"ulimit -f {cfg.max_file_size_mb * 1024} 2>/dev/null; "
            f"ulimit -u {cfg.max_processes} 2>/dev/null; "
        )

        # Working Directory
        args += ["--chdir", working_dir]

        # Shell mit Befehl
        args += ["--", "/bin/sh", "-c", ulimits + command]

        return args


class FirejailSandbox:
    """Firejail-basierte Sandbox — gute Isolation."""

    def __init__(self, config: SandboxConfig) -> None:
        self._config = config

    @staticmethod
    def is_available() -> bool:
        return shutil.which("firejail") is not None

    def build_command(self, command: str, working_dir: str) -> list[str]:
        cfg = self._config
        workspace = str(cfg.workspace_dir.resolve())

        args = [
            "firejail",
            "--quiet",
            "--noprofile",
            f"--private={workspace}",
            "--noroot",
            "--nosound",
            "--no3d",
            "--nodvd",
            "--notv",
            "--novideo",
        ]

        # Netzwerk
        if cfg.network == NetworkPolicy.BLOCK:
            args.append("--net=none")

        # Resource-Limits
        args.append(f"--rlimit-as={cfg.max_memory_mb * 1024 * 1024}")
        args.append(f"--rlimit-fsize={cfg.max_file_size_mb * 1024 * 1024}")
        args.append(f"--rlimit-nproc={cfg.max_processes}")

        # Nur Workspace beschreibbar
        for path in cfg.allowed_read_paths:
            if Path(path).exists():
                args.append(f"--whitelist={path}")

        args += ["--", "/bin/sh", "-c", f"cd {shlex.quote(working_dir)} && {command}"]

        return args


# ============================================================================
# Windows Job Object Sandbox
# ============================================================================


class WindowsJobObjectSandbox:
    """Windows Job Object Sandbox — Windows-native Prozess-Isolation.

    Nutzt Win32 Job Objects für:
    - Memory-Limit: Beschränkt den Arbeitsspeicher pro Prozess
    - CPU-Zeit-Limit: Begrenzt CPU-Nutzung
    - Prozess-Limit: Maximale Anzahl aktiver Prozesse
    - Kill-on-Close: Alle Prozesse werden beendet wenn der Job geschlossen wird
    """

    # Win32 Constants
    JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
    JOB_OBJECT_LIMIT_JOB_TIME = 0x00000004
    JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    JobObjectExtendedLimitInformation = 9
    PROCESS_ALL_ACCESS = 0x001F0FFF

    def __init__(self, config: SandboxConfig) -> None:
        self._config = config

    @staticmethod
    def is_available() -> bool:
        """Prüft ob Windows Job Objects verfügbar sind."""
        return sys.platform == "win32"

    async def execute(
        self,
        command: str,
        working_dir: str,
        timeout: int = 30,
        max_memory_mb: int = 512,
        max_processes: int = 64,
        max_cpu_seconds: int = 10,
    ) -> SandboxResult:
        """Führt einen Befehl mit Windows Job Object Isolation aus.

        Args:
            command: Shell-Befehl.
            working_dir: Arbeitsverzeichnis.
            timeout: Timeout in Sekunden.
            max_memory_mb: Memory-Limit in MB.
            max_processes: Maximale Prozess-Anzahl.
            max_cpu_seconds: CPU-Zeit-Limit in Sekunden.

        Returns:
            SandboxResult mit Ausführungsergebnis.
        """
        import ctypes
        import ctypes.wintypes

        # ----- ctypes Strukturen (lokal definiert, nur auf Windows) -----

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
                return SandboxResult(
                    error=f"CreateJobObjectW fehlgeschlagen: {ctypes.get_last_error()}",
                    sandbox_level="jobobject",
                    exit_code=-1,
                )

            # 2. Limits konfigurieren
            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = (
                self.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                | self.JOB_OBJECT_LIMIT_PROCESS_MEMORY
                | self.JOB_OBJECT_LIMIT_ACTIVE_PROCESS
                | self.JOB_OBJECT_LIMIT_JOB_TIME
            )

            # Memory-Limit (Bytes)
            info.ProcessMemoryLimit = max_memory_mb * 1024 * 1024

            # CPU-Zeit-Limit (100-Nanosekunden-Einheiten)
            info.BasicLimitInformation.PerJobUserTimeLimit = max_cpu_seconds * 10_000_000

            # Prozess-Limit
            info.BasicLimitInformation.ActiveProcessLimit = max_processes

            # 3. Limits auf Job setzen
            success = kernel32.SetInformationJobObject(
                job_handle,
                self.JobObjectExtendedLimitInformation,
                ctypes.byref(info),
                ctypes.sizeof(info),
            )
            if not success:
                kernel32.CloseHandle(job_handle)
                return SandboxResult(
                    error=f"SetInformationJobObject fehlgeschlagen: {ctypes.get_last_error()}",
                    sandbox_level="jobobject",
                    exit_code=-1,
                )

            # 4. Subprocess starten (exec statt shell, um Injection zu vermeiden)
            env = {
                **os.environ,
                "HOME": str(Path.home()),
                "PATH": os.environ.get("PATH", ""),
            }

            # Windows: cmd.exe /S /c "..." für Shell-Features + korrekte Quotes
            exec_args = ["cmd.exe", "/c", command]

            proc = await asyncio.create_subprocess_exec(
                *exec_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
                env=env,
            )

            # 5. Prozess-Handle holen und dem Job zuweisen
            proc_handle = kernel32.OpenProcess(
                self.PROCESS_ALL_ACCESS, False, proc.pid
            )
            if proc_handle:
                kernel32.AssignProcessToJobObject(job_handle, proc_handle)
            else:
                log.warning(
                    "jobobject_assign_failed",
                    pid=proc.pid,
                    error=ctypes.get_last_error(),
                )

            # 6. Auf Abschluss warten (mit Timeout)
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=float(timeout),
                )
            except TimeoutError:
                proc.kill()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                return SandboxResult(
                    timed_out=True,
                    sandbox_level="jobobject",
                    exit_code=-1,
                )

            # 7. Output verarbeiten
            stdout, stderr, truncated = SandboxExecutor._decode_and_truncate(
                stdout_bytes, stderr_bytes
            )

            log.info(
                "jobobject_exec_done",
                exit_code=proc.returncode,
                stdout_len=len(stdout),
                stderr_len=len(stderr),
            )

            return SandboxResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=proc.returncode or 0,
                sandbox_level="jobobject",
                truncated=truncated,
            )

        except OSError as exc:
            return SandboxResult(
                error=f"Windows Job Object Fehler: {exc}",
                sandbox_level="jobobject",
                exit_code=-1,
            )
        finally:
            # 8. Handles schließen
            if proc_handle:
                kernel32.CloseHandle(proc_handle)
            if job_handle:
                kernel32.CloseHandle(job_handle)


# ============================================================================
# Sandbox-Executor
# ============================================================================


class SandboxExecutor:
    """Führt Befehle in der sichersten verfügbaren Sandbox aus.

    Automatisches Fallback:
        bwrap → firejail → bare (kein Sandbox)

    Usage:
        executor = SandboxExecutor(config)
        result = await executor.execute("ls -la", working_dir="/workspace")
    """

    def __init__(self, config: SandboxConfig | None = None) -> None:
        import asyncio
        self._config = config or SandboxConfig()
        self._config_lock = asyncio.Lock()
        self._level: SandboxLevel = SandboxLevel.BARE
        self._bwrap: BwrapSandbox | None = None
        self._firejail: FirejailSandbox | None = None
        self._jobobject: WindowsJobObjectSandbox | None = None

        # Bestes verfügbares Level ermitteln
        self._detect_sandbox()

    def _detect_sandbox(self) -> None:
        """Ermittelt das beste verfügbare Sandbox-Level."""
        preferred = self._config.preferred_level

        if preferred == SandboxLevel.BWRAP and BwrapSandbox.is_available():
            self._bwrap = BwrapSandbox(self._config)
            self._level = SandboxLevel.BWRAP
            log.info("sandbox_detected", level="bwrap", isolation="full")
            return

        if preferred in (SandboxLevel.BWRAP, SandboxLevel.FIREJAIL) and FirejailSandbox.is_available():
            self._firejail = FirejailSandbox(self._config)
            self._level = SandboxLevel.FIREJAIL
            log.info("sandbox_detected", level="firejail", isolation="good")
            return

        # Fallback: bwrap trotzdem versuchen wenn nicht explizit firejail gewählt
        if BwrapSandbox.is_available():
            self._bwrap = BwrapSandbox(self._config)
            self._level = SandboxLevel.BWRAP
            log.info("sandbox_detected", level="bwrap", isolation="full")
            return

        if FirejailSandbox.is_available():
            self._firejail = FirejailSandbox(self._config)
            self._level = SandboxLevel.FIREJAIL
            log.info("sandbox_detected", level="firejail", isolation="good")
            return

        # Windows: Job Objects als Sandbox nutzen (vor bare-Fallback)
        if WindowsJobObjectSandbox.is_available():
            self._jobobject = WindowsJobObjectSandbox(self._config)
            self._level = SandboxLevel.JOBOBJECT
            log.info("sandbox_detected", level="jobobject", isolation="windows-native")
            return

        self._level = SandboxLevel.BARE
        log.warning(
            "no_sandbox_available",
            message="Weder bwrap noch firejail noch Windows Job Objects gefunden. Befehle laufen UNGESCHÜTZT!",
            install_hint="apt install bubblewrap  # Empfohlen (Linux)",
        )

    @property
    def level(self) -> SandboxLevel:
        """Aktives Sandbox-Level."""
        return self._level

    async def execute(
        self,
        command: str,
        *,
        working_dir: str | None = None,
        timeout: int | None = None,
        network: NetworkPolicy | None = None,
        max_memory_mb: int | None = None,
        max_processes: int | None = None,
    ) -> SandboxResult:
        """Führt einen Befehl in der Sandbox aus.

        Args:
            command: Shell-Befehl.
            working_dir: Arbeitsverzeichnis (Default: Workspace).
            timeout: Timeout in Sekunden.
            network: Netzwerk-Policy Override.
            max_memory_mb: Memory-Limit Override (MB).
            max_processes: Prozess-Limit Override.
        """
        if not command.strip():
            return SandboxResult(error="Kein Befehl angegeben")

        timeout = timeout or self._config.default_timeout
        cwd = working_dir or str(self._config.workspace_dir)

        # Working-Dir sicherstellen
        Path(cwd).mkdir(parents=True, exist_ok=True)

        # Lokale Kopie fuer thread-sichere Overrides (Config wird nicht mutiert)
        eff_network = network if network is not None else self._config.network
        eff_memory = max_memory_mb if max_memory_mb is not None else self._config.max_memory_mb
        eff_processes = max_processes if max_processes is not None else self._config.max_processes

        # Lock fuer concurrency-sichere Config-Mutation (Sandbox-Builder lesen self._config)
        async with self._config_lock:
            saved = (self._config.network, self._config.max_memory_mb, self._config.max_processes)
            self._config.network = eff_network
            self._config.max_memory_mb = eff_memory
            self._config.max_processes = eff_processes

            try:
                if self._level == SandboxLevel.BWRAP and self._bwrap:
                    return await self._exec_sandboxed(
                        self._bwrap.build_command(command, cwd),
                        timeout,
                        SandboxLevel.BWRAP,
                    )

                if self._level == SandboxLevel.FIREJAIL and self._firejail:
                    return await self._exec_sandboxed(
                        self._firejail.build_command(command, cwd),
                        timeout,
                        SandboxLevel.FIREJAIL,
                    )

                if self._level == SandboxLevel.JOBOBJECT and self._jobobject:
                    return await self._exec_with_jobobject(
                        command, cwd, timeout, eff_memory, eff_processes,
                    )

                # Bare-Modus (kein Sandbox)
                return await self._exec_bare(command, cwd, timeout)

            finally:
                self._config.network, self._config.max_memory_mb, self._config.max_processes = saved

    async def _exec_sandboxed(
        self,
        full_args: list[str],
        timeout: int,
        level: SandboxLevel,
    ) -> SandboxResult:
        """Führt einen gesandboxten Befehl aus."""
        log.info(
            "sandbox_exec_start",
            level=level.value,
            args_preview=" ".join(full_args)[:200],
            timeout=timeout,
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                *full_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=float(timeout),
                )
            except TimeoutError:
                proc.kill()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                return SandboxResult(
                    timed_out=True,
                    sandbox_level=level.value,
                    exit_code=-1,
                )

            stdout, stderr, truncated = self._decode_and_truncate(stdout_bytes, stderr_bytes)

            log.info(
                "sandbox_exec_done",
                level=level.value,
                exit_code=proc.returncode,
                stdout_len=len(stdout),
                stderr_len=len(stderr),
            )

            return SandboxResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=proc.returncode or 0,
                sandbox_level=level.value,
                truncated=truncated,
            )

        except FileNotFoundError:
            log.error("sandbox_binary_not_found", level=level.value)
            # Fallback auf bare
            return await self._exec_bare(
                full_args[-1] if full_args else "",
                str(self._config.workspace_dir),
                timeout,
            )
        except Exception as exc:
            return SandboxResult(
                error=f"Sandbox-Fehler ({level.value}): {exc}",
                sandbox_level=level.value,
                exit_code=-1,
            )

    async def _exec_with_jobobject(
        self,
        command: str,
        cwd: str,
        timeout: int,
        max_memory_mb: int,
        max_processes: int,
    ) -> SandboxResult:
        """Führt Befehl mit Windows Job Object Isolation aus.

        Args:
            command: Shell-Befehl.
            cwd: Arbeitsverzeichnis.
            timeout: Timeout in Sekunden.
            max_memory_mb: Memory-Limit in MB.
            max_processes: Prozess-Limit.
        """
        if not self._jobobject:
            return await self._exec_bare(command, cwd, timeout)

        log.info(
            "jobobject_exec_start",
            command_preview=command[:200],
            timeout=timeout,
            max_memory_mb=max_memory_mb,
            max_processes=max_processes,
        )

        return await self._jobobject.execute(
            command,
            working_dir=cwd,
            timeout=timeout,
            max_memory_mb=max_memory_mb,
            max_processes=max_processes,
            max_cpu_seconds=self._config.default_timeout,
        )

    async def _exec_bare(
        self,
        command: str,
        cwd: str,
        timeout: int,
    ) -> SandboxResult:
        """Bare-Ausführung ohne Sandbox (Fallback)."""
        log.warning("bare_exec_no_sandbox", command=command[:200], cwd=cwd)

        env = {
            **os.environ,
            "HOME": str(Path.home()),
            "PATH": os.environ.get("PATH", "/usr/bin:/bin" if sys.platform != "win32" else ""),
        }
        if sys.platform != "win32":
            env["LANG"] = "de_DE.UTF-8"
            env["LC_ALL"] = "de_DE.UTF-8"

        try:
            # Exec statt Shell um Injection zu vermeiden.
            # Shell-Features (Pipes, Redirects) werden über sh -c / cmd /c bereitgestellt.
            if sys.platform == "win32":
                exec_args = ["cmd.exe", "/c", command]
            else:
                exec_args = ["sh", "-c", command]

            proc = await asyncio.create_subprocess_exec(
                *exec_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=float(timeout),
                )
            except TimeoutError:
                proc.kill()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                return SandboxResult(timed_out=True, sandbox_level="bare", exit_code=-1)

            stdout, stderr, truncated = self._decode_and_truncate(stdout_bytes, stderr_bytes)

            return SandboxResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=proc.returncode or 0,
                sandbox_level="bare",
                truncated=truncated,
            )

        except OSError as exc:
            return SandboxResult(error=f"Befehl konnte nicht gestartet werden: {exc}", sandbox_level="bare")

    @staticmethod
    def _decode_and_truncate(
        stdout_bytes: bytes,
        stderr_bytes: bytes,
    ) -> tuple[str, str, bool]:
        """Decodiert und beschränkt Output-Größe."""
        def decode(data: bytes) -> str:
            if not data:
                return ""
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError:
                return data.decode("latin-1", errors="replace")

        stdout = decode(stdout_bytes)
        stderr = decode(stderr_bytes)
        truncated = False

        if len(stdout) + len(stderr) > MAX_OUTPUT_BYTES:
            truncated = True
            if stdout:
                stdout = stdout[:MAX_OUTPUT_BYTES // 2]
            if stderr:
                stderr = stderr[:MAX_OUTPUT_BYTES // 2]

        return stdout, stderr, truncated
