"""Remote shell execution via SSH for Cognithor.

Supports executing commands on remote hosts via SSH subprocess.
Uses the system's ssh client (no paramiko dependency needed).
"""

from __future__ import annotations

import asyncio
import re
import shlex
import time
from dataclasses import dataclass
from typing import Any

from cognithor.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class RemoteHost:
    """SSH connection target."""

    name: str
    host: str
    user: str = "root"
    port: int = 22
    key_path: str | None = None
    working_dir: str = "/tmp"
    max_memory_mb: int = 512
    timeout: int = 60


class RemoteShellTools:
    """MCP tools for remote command execution via SSH."""

    # Commands blocked on remote hosts too
    _BLOCKED_PATTERNS = [
        r"rm\s+-rf\s+/[^.]",  # rm -rf /anything (not relative)
        r"mkfs\.",
        r"dd\s+.*of=/dev/",
        r"shutdown",
        r"reboot",
        r"init\s+[06]",
        r":()\{.*\|.*&.*\}",  # fork bomb
    ]

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._hosts: dict[str, RemoteHost] = {}
        self._config = config or {}
        # Load hosts from config if provided
        for name, hcfg in self._config.get("hosts", {}).items():
            self._hosts[name] = RemoteHost(
                name=name,
                host=hcfg.get("host", "localhost"),
                user=hcfg.get("user", "root"),
                port=hcfg.get("port", 22),
                key_path=hcfg.get("key_path"),
                working_dir=hcfg.get("working_dir", "/tmp"),
                timeout=hcfg.get("timeout", 60),
            )

    def add_host(self, host: RemoteHost) -> None:
        """Register a remote host for execution."""
        self._hosts[host.name] = host
        log.info("remote_host_added", name=host.name, host=host.host)

    def remove_host(self, name: str) -> bool:
        """Remove a registered host."""
        if name in self._hosts:
            del self._hosts[name]
            return True
        return False

    def list_hosts(self) -> list[dict[str, Any]]:
        """List all registered remote hosts."""
        return [
            {
                "name": h.name,
                "host": h.host,
                "user": h.user,
                "port": h.port,
                "working_dir": h.working_dir,
            }
            for h in self._hosts.values()
        ]

    def _validate_command(self, command: str) -> str | None:
        """Validate command is safe. Returns error message or None."""
        if "\x00" in command:
            return "Null bytes not allowed in commands"
        for pattern in self._BLOCKED_PATTERNS:
            if re.search(pattern, command):
                return f"Blocked dangerous command pattern: {pattern}"
        return None

    def _build_ssh_command(self, host: RemoteHost, command: str) -> list[str]:
        """Build the ssh subprocess command."""
        ssh_cmd = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "BatchMode=yes",
            "-p",
            str(host.port),
        ]
        if host.key_path:
            ssh_cmd.extend(["-i", host.key_path])
        ssh_cmd.append(f"{host.user}@{host.host}")
        # PASS-3 SEC-CRIT: the command body must NOT be interpolated raw
        # into the outer SSH shell string. Doing so lets a caller include
        # ``;``, ``$()``, ``|sh``, ``echo$(IFS)foo`` etc. and bypass the
        # ``_BLOCKED_PATTERNS`` regex blocklist by using shell tokens the
        # blocklist never anticipated. Wrap the user command as a single
        # shlex-quoted argument to ``bash -c`` so the outer shell sees it
        # as one literal token; bash then parses the user's intent in a
        # contained sub-shell. ``working_dir`` was already shlex-quoted.
        wrapped = (
            f"cd {shlex.quote(host.working_dir)} && "
            f"timeout {int(host.timeout)} bash -c {shlex.quote(command)}"
        )
        ssh_cmd.append(wrapped)
        return ssh_cmd

    async def exec_remote(
        self,
        host_name: str,
        command: str,
        working_dir: str | None = None,
        timeout: int | None = None,
    ) -> str:
        """Execute a command on a remote host via SSH."""
        host = self._hosts.get(host_name)
        if host is None:
            available = ", ".join(self._hosts.keys()) or "none"
            return f"Error: Unknown host '{host_name}'. Available: {available}"

        # Validate
        error = self._validate_command(command)
        if error:
            return f"Error: {error}"

        # Override working dir/timeout if specified
        effective_host = RemoteHost(
            name=host.name,
            host=host.host,
            user=host.user,
            port=host.port,
            key_path=host.key_path,
            working_dir=working_dir or host.working_dir,
            timeout=timeout or host.timeout,
        )

        ssh_cmd = self._build_ssh_command(effective_host, command)
        log.info(
            "remote_exec_start",
            host=host_name,
            command=command[:200],
        )

        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=effective_host.timeout + 15,  # Extra buffer for SSH overhead
            )
        except TimeoutError:
            return f"Error: Command timed out after {effective_host.timeout}s on {host_name}"
        except FileNotFoundError:
            return "Error: SSH client not found. Install OpenSSH."

        duration_ms = int((time.monotonic() - start) * 1000)
        output = stdout.decode("utf-8", errors="replace")
        err_output = stderr.decode("utf-8", errors="replace")

        # Truncate
        max_len = 50_000
        if len(output) > max_len:
            output = output[:max_len] + "\n[... truncated]"

        log.info(
            "remote_exec_complete",
            host=host_name,
            exit_code=proc.returncode,
            duration_ms=duration_ms,
        )

        result = output
        if err_output.strip() and proc.returncode != 0:
            result += f"\n[stderr]: {err_output[:5000]}"
        if proc.returncode != 0:
            result += f"\n[exit code: {proc.returncode}]"

        return result.strip() or "(no output)"

    async def test_connection(self, host_name: str) -> str:
        """Test SSH connection to a host."""
        return await self.exec_remote(host_name, "echo 'Connection OK' && uname -a")


def register_remote_shell_tools(
    mcp_client: Any,
    config: dict[str, Any] | None = None,
) -> RemoteShellTools | None:
    """Register remote shell MCP tools."""
    tools = RemoteShellTools(config)

    mcp_client.register_builtin_handler(
        "remote_exec",
        tools.exec_remote,
        description=(
            "Execute a command on a remote host via SSH. "
            "Use for running commands on servers, Docker containers, "
            "or VPS instances. Requires host to be registered first."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "host_name": {
                    "type": "string",
                    "description": "Name of the registered remote host",
                },
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                },
                "working_dir": {
                    "type": "string",
                    "description": "Working directory on remote (optional)",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (optional, default from host config)",
                },
            },
            "required": ["host_name", "command"],
        },
    )

    mcp_client.register_builtin_handler(
        "remote_list_hosts",
        lambda **_kwargs: tools.list_hosts(),
        description="List all registered remote SSH hosts.",
        input_schema={"type": "object", "properties": {}},
    )

    mcp_client.register_builtin_handler(
        "remote_test_connection",
        tools.test_connection,
        description="Test SSH connection to a registered host.",
        input_schema={
            "type": "object",
            "properties": {
                "host_name": {
                    "type": "string",
                    "description": "Name of the host to test",
                },
            },
            "required": ["host_name"],
        },
    )

    log.info(
        "remote_shell_tools_registered",
        tools=["remote_exec", "remote_list_hosts", "remote_test_connection"],
    )
    return tools
