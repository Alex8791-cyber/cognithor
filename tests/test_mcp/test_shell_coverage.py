"""Coverage-Tests fuer shell.py -- fehlende Pfade abdecken."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.mcp.shell import ShellTools, ShellError, register_shell_tools


@pytest.fixture
def config(tmp_path: Path) -> MagicMock:
    cfg = MagicMock()
    cfg.workspace_dir = tmp_path
    cfg.shell = None
    cfg.sandbox_level = "bare"
    cfg.sandbox_network = "allow"
    cfg.security.shell_validate_paths = True
    return cfg


class TestValidateCommand:
    def test_null_byte_blocked(self) -> None:
        result = ShellTools._validate_command("echo \x00test", "/tmp")
        assert result is not None
        assert "Null-Byte" in result

    def test_path_traversal_blocked(self) -> None:
        # Path traversal is a hard block
        result = ShellTools._validate_command("cat ../../etc/passwd", "/workspace")
        assert result is not None
        assert "Path-Traversal" in result

    def test_normal_command(self) -> None:
        result = ShellTools._validate_command("ls -la", "/workspace")
        assert result is None

    def test_unparsable_command(self) -> None:
        result = ShellTools._validate_command("echo 'unterminated", "/workspace")
        assert result is None

    def test_file_command_escape(self) -> None:
        result = ShellTools._validate_command("cat /etc/passwd", "/workspace")
        assert result is not None  # Blocked: path outside workspace
        assert "außerhalb" in result or "ausserhalb" in result or "blockiert" in result


class TestExecCommand:
    @pytest.mark.asyncio
    async def test_empty_command(self, config: MagicMock) -> None:
        shell = ShellTools(config)
        result = await shell.exec_command("   ")
        assert "Kein Befehl" in result

    @pytest.mark.asyncio
    async def test_outside_workspace(self, config: MagicMock) -> None:
        shell = ShellTools(config)
        result = await shell.exec_command("ls", working_dir="/etc")
        assert "Zugriff verweigert" in result

    @pytest.mark.asyncio
    async def test_null_byte_blocked(self, config: MagicMock) -> None:
        shell = ShellTools(config)
        result = await shell.exec_command("echo \x00test")
        assert "Null-Byte" in result

    @pytest.mark.asyncio
    async def test_redacted_logging(self, config: MagicMock) -> None:
        shell = ShellTools(config)
        # Just verify it doesn't crash on sensitive commands
        mock_result = MagicMock()
        mock_result.output = "ok"
        mock_result.exit_code = 0
        mock_result.stdout = "ok"
        mock_result.stderr = ""
        mock_result.timed_out = False
        mock_result.truncated = False
        mock_result.sandbox_level = "bare"
        shell._sandbox.execute = AsyncMock(return_value=mock_result)
        result = await shell.exec_command("export API_KEY=secret123")
        assert result == "ok"


class TestSandboxLevel:
    def test_property(self, config: MagicMock) -> None:
        shell = ShellTools(config)
        assert isinstance(shell.sandbox_level, str)


class TestRegisterShellTools:
    def test_registers_one_tool(self, config: MagicMock) -> None:
        mock_client = MagicMock()
        shell = register_shell_tools(mock_client, config)
        assert isinstance(shell, ShellTools)
        assert mock_client.register_builtin_handler.call_count == 1
        name = mock_client.register_builtin_handler.call_args_list[0].args[0]
        assert name == "exec_command"


# ============================================================================
# TestExecCommandExtended
# ============================================================================


class TestExecCommandExtended:
    """Erweiterte Tests fuer exec_command -- Sandbox-Ergebnisverarbeitung."""

    @pytest.mark.asyncio
    async def test_timeout_behavior(self, config: MagicMock) -> None:
        """Sandbox meldet Timeout -> Ergebnis wird korrekt durchgereicht."""
        shell = ShellTools(config)
        mock_result = MagicMock()
        mock_result.output = "[TIMEOUT] Befehl nach 30s abgebrochen"
        mock_result.exit_code = 124
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_result.timed_out = True
        mock_result.truncated = False
        mock_result.sandbox_level = "bare"
        shell._sandbox.execute = AsyncMock(return_value=mock_result)

        result = await shell.exec_command("sleep 999")
        assert "TIMEOUT" in result or "abgebrochen" in result

    @pytest.mark.asyncio
    async def test_truncated_output(self, config: MagicMock) -> None:
        """Sandbox meldet truncated -> Ergebnis enthaelt Truncation-Hinweis."""
        shell = ShellTools(config)
        mock_result = MagicMock()
        mock_result.output = "partial output... [TRUNCATED]"
        mock_result.exit_code = 0
        mock_result.stdout = "partial output... [TRUNCATED]"
        mock_result.stderr = ""
        mock_result.timed_out = False
        mock_result.truncated = True
        mock_result.sandbox_level = "bare"
        shell._sandbox.execute = AsyncMock(return_value=mock_result)

        result = await shell.exec_command("cat largefile.bin")
        assert "TRUNCATED" in result or "partial" in result

    @pytest.mark.asyncio
    async def test_stderr_output(self, config: MagicMock) -> None:
        """Sandbox liefert stderr-Inhalt -> wird im output zurueckgegeben."""
        shell = ShellTools(config)
        mock_result = MagicMock()
        mock_result.output = "Error: file not found\n"
        mock_result.exit_code = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error: file not found"
        mock_result.timed_out = False
        mock_result.truncated = False
        mock_result.sandbox_level = "bare"
        shell._sandbox.execute = AsyncMock(return_value=mock_result)

        result = await shell.exec_command("cat nonexistent.txt")
        assert "file not found" in result

    @pytest.mark.asyncio
    async def test_successful_execution(self, config: MagicMock) -> None:
        """Happy-Path: Befehl laeuft durch und liefert stdout zurueck."""
        shell = ShellTools(config)
        mock_result = MagicMock()
        mock_result.output = "hello world\n"
        mock_result.exit_code = 0
        mock_result.stdout = "hello world\n"
        mock_result.stderr = ""
        mock_result.timed_out = False
        mock_result.truncated = False
        mock_result.sandbox_level = "bare"
        shell._sandbox.execute = AsyncMock(return_value=mock_result)

        result = await shell.exec_command("echo hello world")
        assert result == "hello world\n"

    @pytest.mark.asyncio
    async def test_sandbox_overrides(self, config: MagicMock) -> None:
        """Per-Agent Sandbox-Overrides (_sandbox_network, _sandbox_max_memory_mb) werden weitergegeben."""
        shell = ShellTools(config)
        mock_result = MagicMock()
        mock_result.output = "ok"
        mock_result.exit_code = 0
        mock_result.stdout = "ok"
        mock_result.stderr = ""
        mock_result.timed_out = False
        mock_result.truncated = False
        mock_result.sandbox_level = "bare"
        shell._sandbox.execute = AsyncMock(return_value=mock_result)

        result = await shell.exec_command(
            "echo test",
            _sandbox_network="block",
            _sandbox_max_memory_mb=256,
        )
        assert result == "ok"

        # Pruefen dass execute mit den Overrides aufgerufen wurde
        call_kwargs = shell._sandbox.execute.call_args
        assert call_kwargs.kwargs.get("max_memory_mb") == 256
        # network wird als NetworkPolicy-Enum uebergeben
        assert call_kwargs.kwargs.get("network") is not None


# ============================================================================
# TestValidateCommandExtended
# ============================================================================


class TestValidateCommandExtended:
    """Erweiterte Tests fuer _validate_command."""

    def test_multiple_path_traversals(self) -> None:
        """Drei-Level-Path-Traversal auf sensitive Datei -> Hard Block."""
        result = ShellTools._validate_command("cat ../../../etc/shadow", "/workspace")
        assert result is not None
        assert "Path-Traversal" in result

    def test_safe_file_command(self) -> None:
        """Datei-Zugriff innerhalb des Workspace -> kein Warning, kein Block."""
        import tempfile

        with tempfile.TemporaryDirectory() as ws:
            # Erstelle eine lokale Datei im Workspace
            local_file = Path(ws) / "local_file.txt"
            local_file.touch()
            result = ShellTools._validate_command(
                f"cat {local_file}",
                ws,
            )
            assert result is None


# ============================================================================
# TestSandboxLevelExtended
# ============================================================================


class TestSandboxLevelExtended:
    """Erweiterte Tests fuer sandbox_level-Property mit verschiedenen Konfigurationen."""

    def test_different_sandbox_levels(self) -> None:
        """ShellTools mit verschiedenen sandbox_level-Werten initialisieren."""
        for level in ("bare", "process", "jobobject"):
            cfg = MagicMock()
            cfg.workspace_dir = Path(tempfile.gettempdir()) / "test_workspace"
            cfg.shell = None
            cfg.sandbox_level = level
            cfg.sandbox_network = "allow"
            cfg.security.shell_validate_paths = True

            try:
                shell = ShellTools(cfg)
                # sandbox_level Property muss einen String zurueckgeben
                assert isinstance(shell.sandbox_level, str)
                assert len(shell.sandbox_level) > 0
            except ValueError:
                # Nicht alle Levels sind auf jeder Plattform verfuegbar.
                # Das ist ok -- der Test prueft nur dass die Konstruktion
                # entweder funktioniert oder einen sauberen ValueError wirft.
                pass
