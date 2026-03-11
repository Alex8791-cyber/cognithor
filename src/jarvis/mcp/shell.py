"""Shell-Tool für Jarvis -- mit echter Sandbox-Isolation.

Führt Shell-Befehle in einer isolierten Umgebung aus:
  - bubblewrap (bwrap): Linux-Namespaces, stärkste Isolation
  - firejail: Application Sandboxing, gute Isolation
  - bare: Fallback ohne Sandbox (nur Timeout + Output-Limit)

Der Gatekeeper blockiert destruktive Befehle VOR der Ausführung.
Die Sandbox isoliert die Ausführung zusätzlich auf OS-Level.
Zusammen bilden sie ein Defense-in-Depth-System.

Bibel-Referenz: §5.3 (jarvis-shell Server), §4.3 (Sandbox)
"""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jarvis.core.sandbox import (
    NetworkPolicy,
    SandboxConfig,
    SandboxExecutor,
    SandboxLevel,
)
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.config import JarvisConfig

log = get_logger(__name__)

# Log-Limits
MAX_LOG_COMMAND_LENGTH = 200
MAX_REDACTED_LOG_PREFIX = 50

# Pfad-Validierung (Layer 0)
_NULL_BYTE_RE = re.compile(r"\x00")
_PATH_TRAVERSAL_RE = re.compile(r"(?:^|[\s;|&])(?:\.\.[/\\]){2,}")
_FILE_COMMANDS = frozenset(
    {
        "cat",
        "head",
        "tail",
        "less",
        "more",
        "cp",
        "mv",
        "rm",
        "chmod",
        "chown",
        "ln",
        "readlink",
        "stat",
        "file",
        "touch",
        "mkdir",
        "rmdir",
    }
)

__all__ = [
    "ShellTools",
    "ShellError",
    "register_shell_tools",
]


class ShellError(Exception):
    """Fehler bei Shell-Ausführung."""


class ShellTools:
    """Shell-Befehlsausführung mit echter Sandbox-Isolation. [B§5.3]

    Security-Architektur (Defense in Depth):
      Layer 1: Gatekeeper -- Regex-Blocklist + Policy-Regeln
      Layer 2: Sandbox -- OS-Level Prozess-Isolation (bwrap/firejail)
      Layer 3: Resource-Limits -- Timeout, Memory, Disk, Processes
    """

    def __init__(self, config: "JarvisConfig") -> None:
        """Initialisiert ShellTools mit Sandbox.

        Erkennt automatisch das beste verfügbare Sandbox-Level.
        """
        self._config = config

        # Konfigurierbare Limits aus config.shell (mit sicheren Defaults)
        _shell_cfg = getattr(config, "shell", None)
        self._default_timeout: int = getattr(_shell_cfg, "default_timeout_seconds", 30)
        self._max_log_command_length: int = getattr(
            _shell_cfg, "max_log_command_length", MAX_LOG_COMMAND_LENGTH
        )
        self._max_redacted_log_prefix: int = getattr(
            _shell_cfg, "max_redacted_log_prefix", MAX_REDACTED_LOG_PREFIX
        )

        # Sandbox-Konfiguration aus JarvisConfig ableiten
        sandbox_config = SandboxConfig(
            workspace_dir=config.workspace_dir,
            default_timeout=self._default_timeout,
        )

        # Sandbox-Level aus Config übernehmen (wenn vorhanden)
        sandbox_level = getattr(config, "sandbox_level", "bwrap")
        if sandbox_level in ("bwrap", "firejail", "bare"):
            sandbox_config.preferred_level = SandboxLevel(sandbox_level)

        # Netzwerk-Policy aus Config
        sandbox_network = getattr(config, "sandbox_network", "allow")
        if sandbox_network in ("allow", "block"):
            sandbox_config.network = NetworkPolicy(sandbox_network)

        self._sandbox = SandboxExecutor(sandbox_config)
        self._default_cwd = str(config.workspace_dir)

        log.info(
            "shell_tools_init",
            sandbox_level=self._sandbox.level.value,
            workspace=self._default_cwd,
        )

    @property
    def sandbox_level(self) -> str:
        """Aktives Sandbox-Level."""
        return self._sandbox.level.value

    @staticmethod
    def _validate_command(command: str, workspace_root: str) -> str | None:
        """Layer-0-Validierung: Null-Bytes, Path-Traversal, File-Path-Escape.

        Returns:
            Fehlermeldung bei Hard Block, None wenn ok.
        """
        # 1. Null-Byte → Hard Block
        if _NULL_BYTE_RE.search(command):
            log.warning("shell_null_byte_blocked", command_prefix=command[:50])
            return "Befehl blockiert: Null-Byte erkannt."

        # 2. Path Traversal (../../..) → Hard Block
        if _PATH_TRAVERSAL_RE.search(command):
            log.warning("shell_path_traversal_blocked", command_prefix=command[:80])
            return "Befehl blockiert: Path-Traversal erkannt."

        # 3. File-Path-Escape: Prüfe ob File-Commands Pfade ausserhalb Workspace nutzen
        try:
            parts = shlex.split(command)
        except ValueError:
            return None  # Unparsbares Command → Sandbox entscheidet

        if not parts:
            return None

        base_cmd = Path(parts[0]).name  # Nur Basisname (z.B. /usr/bin/cat → cat)
        if base_cmd in _FILE_COMMANDS:
            ws_root = Path(workspace_root).resolve()
            for arg in parts[1:]:
                if arg.startswith("-"):
                    continue
                try:
                    resolved = (ws_root / arg).resolve()
                    resolved.relative_to(ws_root)
                except (ValueError, OSError):
                    log.warning(
                        "shell_path_escape_blocked",
                        command=base_cmd,
                        argument=arg[:100],
                        workspace=str(ws_root),
                    )
                    return f"Befehl blockiert: Pfad '{arg[:100]}' liegt außerhalb des Workspace."

        return None

    async def exec_command(
        self,
        command: str,
        working_dir: str | None = None,
        timeout: int | None = None,
        _sandbox_network: str | None = None,
        _sandbox_max_memory_mb: int | None = None,
        _sandbox_max_processes: int | None = None,
    ) -> str:
        """Führt einen Shell-Befehl in der Sandbox aus.

        Args:
            command: Shell-Befehl als String.
            working_dir: Arbeitsverzeichnis (Default: ~/.jarvis/workspace/).
            timeout: Timeout in Sekunden (Default: 30).
            _sandbox_network: Per-Agent Netzwerk-Override ("allow"/"block").
            _sandbox_max_memory_mb: Per-Agent Memory-Limit in MB.
            _sandbox_max_processes: Per-Agent Prozess-Limit.

        Returns:
            Kombinierter stdout + stderr Output.
        """
        if not command.strip():
            return "Kein Befehl angegeben."

        cwd = working_dir or self._default_cwd

        # Working-Directory validieren -- muss unter Workspace liegen
        cwd_path = Path(cwd).expanduser().resolve()
        workspace_root = Path(self._default_cwd).expanduser().resolve()
        try:
            cwd_path.relative_to(workspace_root)
        except ValueError:
            return (
                f"Zugriff verweigert: Arbeitsverzeichnis '{cwd}' liegt ausserhalb "
                f"des Workspace ({workspace_root})"
            )
        cwd_path.mkdir(parents=True, exist_ok=True)

        # Layer-0-Validierung: Null-Bytes, Path-Traversal
        if getattr(self._config, "security", None) and getattr(
            self._config.security, "shell_validate_paths", True
        ):
            validation_error = self._validate_command(command, str(workspace_root))
            if validation_error:
                return validation_error

        # Per-Agent Overrides
        network_override = None
        if _sandbox_network:
            try:
                network_override = NetworkPolicy(_sandbox_network)
            except ValueError:
                pass  # Invalid network policy value, fall back to default (None)

        # Befehls-Logging: Kuerzen und sensitive Muster maskieren
        _log_cmd = command[: self._max_log_command_length]
        for _pattern in ("API_KEY=", "TOKEN=", "PASSWORD=", "SECRET=", "BEARER "):
            if _pattern.lower() in _log_cmd.lower():
                _log_cmd = _log_cmd[: self._max_redacted_log_prefix] + " [REDACTED]"
                break

        log.info(
            "shell_exec_start",
            command=_log_cmd,
            cwd=str(cwd_path),
            sandbox=self._sandbox.level.value,
            timeout=timeout,
            network_override=_sandbox_network,
            memory_override=_sandbox_max_memory_mb,
            processes_override=_sandbox_max_processes,
        )

        # In Sandbox ausführen (mit per-Agent Overrides)
        result = await self._sandbox.execute(
            command,
            working_dir=str(cwd_path),
            timeout=timeout,
            network=network_override,
            max_memory_mb=_sandbox_max_memory_mb,
            max_processes=_sandbox_max_processes,
        )

        log_method = log.warning if result.truncated else log.info
        log_method(
            "shell_exec_done",
            command=_log_cmd,
            exit_code=result.exit_code,
            sandbox=result.sandbox_level,
            stdout_len=len(result.stdout),
            stderr_len=len(result.stderr),
            timed_out=result.timed_out,
            truncated=result.truncated,
        )

        return result.output


def register_shell_tools(
    mcp_client: Any,
    config: "JarvisConfig",
) -> ShellTools:
    """Registriert Shell-Tools beim MCP-Client.

    Returns:
        ShellTools-Instanz.
    """
    shell = ShellTools(config)

    mcp_client.register_builtin_handler(
        "exec_command",
        shell.exec_command,
        description=(
            f"Führt einen Shell-Befehl in einer Sandbox ({shell.sandbox_level}) aus. "
            "Arbeitsverzeichnis: ~/.jarvis/workspace/. "
            "Destruktive Befehle werden vom Gatekeeper blockiert."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell-Befehl"},
                "working_dir": {
                    "type": "string",
                    "description": "Arbeitsverzeichnis (Default: ~/.jarvis/workspace/)",
                    "default": None,
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in Sekunden (Default: 30)",
                    "default": 30,
                },
            },
            "required": ["command"],
        },
    )

    log.info(
        "shell_tools_registered",
        tools=["exec_command"],
        sandbox_level=shell.sandbox_level,
    )
    return shell
