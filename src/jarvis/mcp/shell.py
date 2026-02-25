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

        # Sandbox-Konfiguration aus JarvisConfig ableiten
        sandbox_config = SandboxConfig(
            workspace_dir=config.workspace_dir,
            default_timeout=30,
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

        # Per-Agent Overrides
        network_override = None
        if _sandbox_network:
            try:
                network_override = NetworkPolicy(_sandbox_network)
            except ValueError:
                pass

        # Befehls-Logging: Kuerzen und sensitive Muster maskieren
        _log_cmd = command[:MAX_LOG_COMMAND_LENGTH]
        for _pattern in ("API_KEY=", "TOKEN=", "PASSWORD=", "SECRET=", "BEARER "):
            if _pattern.lower() in _log_cmd.lower():
                _log_cmd = _log_cmd[:MAX_REDACTED_LOG_PREFIX] + " [REDACTED]"
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

        log.info(
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
