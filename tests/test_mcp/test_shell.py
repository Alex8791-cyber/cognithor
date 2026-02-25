"""Tests für ShellTools – Sandbox-gesicherte Shell-Ausführung.

Testet:
  - Einfache Befehle (echo, ls, cat)
  - Working-Directory
  - Exit-Codes und stderr
  - Timeout-Enforcing
  - Leerer Befehl
  - Output-Decodierung
  - Tool-Registrierung beim MCP-Client
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from jarvis.config import JarvisConfig, SecurityConfig, ensure_directory_structure
from jarvis.mcp.shell import ShellTools, register_shell_tools


def _path_in_result(expected_path: Path, result: str) -> bool:
    """Check if a path is contained in command output, handling MSYS/Git-Bash path differences on Windows."""
    expected = str(expected_path)
    if expected in result:
        return True
    if sys.platform == "win32":
        # Git-Bash pwd gives /c/Users/... instead of C:\Users\...
        posix = expected_path.as_posix()
        if posix in result:
            return True
        # MSYS maps /tmp to C:\Users\...\AppData\Local\Temp
        # Compare by matching the unique test-specific path suffix
        # e.g. both paths end with "pytest-NNN/test_pwd0/.jarvis/workspace"
        stripped = result.strip().replace("\\", "/")
        expected_posix = posix
        # Find longest common suffix by path components
        result_parts = stripped.rstrip("/").split("/")
        expected_parts = expected_posix.rstrip("/").split("/")
        common = 0
        for rp, ep in zip(reversed(result_parts), reversed(expected_parts)):
            if rp == ep:
                common += 1
            else:
                break
        # At least 2 components should match (e.g. ".jarvis/workspace")
        if common >= 2:
            return True
    return False

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
def config(tmp_path: Path) -> JarvisConfig:
    cfg = JarvisConfig(
        jarvis_home=tmp_path / ".jarvis",
        security=SecurityConfig(
            allowed_paths=[str(tmp_path)],
        ),
    )
    ensure_directory_structure(cfg)
    return cfg


@pytest.fixture()
def shell(config: JarvisConfig) -> ShellTools:
    return ShellTools(config)


# =============================================================================
# Einfache Befehle
# =============================================================================


class TestBasicCommands:
    @pytest.mark.asyncio
    async def test_echo(self, shell: ShellTools) -> None:
        """echo gibt den Text zurück."""
        result = await shell.exec_command("echo Hello Jarvis")
        assert "Hello Jarvis" in result

    @pytest.mark.asyncio
    async def test_pwd(self, shell: ShellTools, config: JarvisConfig) -> None:
        """pwd gibt das Arbeitsverzeichnis zurück."""
        result = await shell.exec_command("pwd")
        assert _path_in_result(config.workspace_dir, result)

    @pytest.mark.asyncio
    async def test_ls(self, shell: ShellTools, config: JarvisConfig) -> None:
        """ls in einem Verzeichnis mit Dateien funktioniert."""
        (config.workspace_dir / "test.txt").write_text("x")
        result = await shell.exec_command("ls")
        assert "test.txt" in result

    @pytest.mark.asyncio
    async def test_cat_file(self, shell: ShellTools, config: JarvisConfig) -> None:
        """cat liest Dateiinhalt."""
        (config.workspace_dir / "read_me.txt").write_text("Hallo Welt")
        result = await shell.exec_command("cat read_me.txt")
        assert "Hallo Welt" in result

    @pytest.mark.asyncio
    async def test_empty_command(self, shell: ShellTools) -> None:
        """Leerer Befehl gibt Hinweis zurück."""
        result = await shell.exec_command("")
        assert "Kein Befehl" in result

    @pytest.mark.asyncio
    async def test_whitespace_command(self, shell: ShellTools) -> None:
        """Nur-Whitespace-Befehl gibt Hinweis zurück."""
        result = await shell.exec_command("   ")
        assert "Kein Befehl" in result


# =============================================================================
# Working Directory
# =============================================================================


class TestWorkingDirectory:
    @pytest.mark.asyncio
    async def test_custom_working_dir(self, shell: ShellTools, config: JarvisConfig) -> None:
        """Benutzerdefiniertes Working Directory wird verwendet (inside workspace)."""
        custom_dir = config.workspace_dir / "custom_wd"
        custom_dir.mkdir()
        result = await shell.exec_command("pwd", working_dir=str(custom_dir))
        assert _path_in_result(custom_dir, result)

    @pytest.mark.asyncio
    async def test_working_dir_created_if_missing(self, shell: ShellTools, config: JarvisConfig) -> None:
        """Fehlendes Working Directory wird automatisch erstellt (inside workspace)."""
        new_dir = config.workspace_dir / "auto_created"
        assert not new_dir.exists()
        await shell.exec_command("echo test", working_dir=str(new_dir))
        assert new_dir.exists()

    @pytest.mark.asyncio
    async def test_working_dir_outside_workspace_rejected(self, shell: ShellTools, tmp_path: Path) -> None:
        """Arbeitsverzeichnis ausserhalb Workspace wird abgelehnt."""
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        result = await shell.exec_command("echo test", working_dir=str(outside_dir))
        assert "Zugriff verweigert" in result

    @pytest.mark.asyncio
    async def test_default_working_dir(self, shell: ShellTools, config: JarvisConfig) -> None:
        """Default Working Directory ist ~/.jarvis/workspace."""
        result = await shell.exec_command("pwd")
        assert _path_in_result(config.workspace_dir, result)


# =============================================================================
# Exit-Code und stderr
# =============================================================================


class TestExitCodeAndStderr:
    @pytest.mark.asyncio
    async def test_nonzero_exit_code_reported(self, shell: ShellTools) -> None:
        """Nicht-Null Exit-Code wird angezeigt."""
        result = await shell.exec_command("exit 42")
        assert "EXIT CODE: 42" in result

    @pytest.mark.asyncio
    async def test_stderr_captured(self, shell: ShellTools) -> None:
        """stderr wird separat erfasst."""
        result = await shell.exec_command("echo error >&2")
        assert "STDERR" in result
        assert "error" in result

    @pytest.mark.asyncio
    async def test_combined_stdout_stderr(self, shell: ShellTools) -> None:
        """stdout und stderr werden beide angezeigt."""
        result = await shell.exec_command("echo out; echo err >&2")
        assert "out" in result
        assert "err" in result

    @pytest.mark.asyncio
    async def test_failed_command(self, shell: ShellTools) -> None:
        """Fehlgeschlagener Befehl zeigt Fehler."""
        result = await shell.exec_command("ls /verzeichnis_das_nicht_existiert")
        assert "EXIT CODE" in result or "STDERR" in result

    @pytest.mark.asyncio
    async def test_no_output_command(self, shell: ShellTools) -> None:
        """Befehl ohne Ausgabe gibt Hinweis."""
        result = await shell.exec_command("true")
        assert result == "(Keine Ausgabe)"


# =============================================================================
# Timeout
# =============================================================================


class TestTimeout:
    @pytest.mark.asyncio
    async def test_timeout_kills_command(self, shell: ShellTools) -> None:
        """Befehl wird nach Timeout beendet."""
        result = await shell.exec_command("sleep 60", timeout=1)
        assert "TIMEOUT" in result

    @pytest.mark.asyncio
    async def test_fast_command_no_timeout(self, shell: ShellTools) -> None:
        """Schneller Befehl wird nicht durch Timeout beeinflusst."""
        result = await shell.exec_command("echo schnell", timeout=10)
        assert "schnell" in result
        assert "TIMEOUT" not in result


# =============================================================================
# Output-Decodierung
# =============================================================================


class TestOutputDecoding:
    """Decoding ist jetzt in SandboxExecutor._decode_and_truncate."""

    def test_decode_utf8(self) -> None:
        from jarvis.core.sandbox import SandboxExecutor

        stdout, stderr, truncated = SandboxExecutor._decode_and_truncate(
            "Ärzte und Über".encode(), b""
        )
        assert "Ärzte" in stdout

    def test_decode_empty(self) -> None:
        from jarvis.core.sandbox import SandboxExecutor

        stdout, stderr, truncated = SandboxExecutor._decode_and_truncate(b"", b"")
        assert stdout == ""
        assert stderr == ""

    def test_decode_latin1_fallback(self) -> None:
        from jarvis.core.sandbox import SandboxExecutor

        # 0x80 ist kein gültiges UTF-8
        data = b"Hello \x80 World"
        stdout, stderr, truncated = SandboxExecutor._decode_and_truncate(data, b"")
        assert "Hello" in stdout
        assert "World" in stdout


# =============================================================================
# register_shell_tools
# =============================================================================


class TestRegisterShellTools:
    def test_registers_exec_command(self, config: JarvisConfig) -> None:
        from jarvis.mcp.client import JarvisMCPClient

        client = JarvisMCPClient(config)
        shell = register_shell_tools(client, config)

        assert isinstance(shell, ShellTools)
        tools = client.get_tool_list()
        assert "exec_command" in tools
        assert len(tools) == 1

    def test_schema_contains_command_param(self, config: JarvisConfig) -> None:
        from jarvis.mcp.client import JarvisMCPClient

        client = JarvisMCPClient(config)
        register_shell_tools(client, config)

        schemas = client.get_tool_schemas()
        assert "exec_command" in schemas
        props = schemas["exec_command"]["inputSchema"]["properties"]
        assert "command" in props
        assert "working_dir" in props
        assert "timeout" in props
