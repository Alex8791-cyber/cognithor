"""Tests für die Prozess-Sandbox."""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.core.sandbox import (
    BwrapSandbox,
    FirejailSandbox,
    NetworkPolicy,
    SandboxConfig,
    SandboxExecutor,
    SandboxLevel,
    SandboxResult,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    d = tmp_path / "workspace"
    d.mkdir()
    return d


@pytest.fixture
def config(workspace: Path) -> SandboxConfig:
    return SandboxConfig(
        workspace_dir=workspace,
        preferred_level=SandboxLevel.BARE,  # Tests laufen ohne bwrap/firejail
        default_timeout=10,
    )


@pytest.fixture
def executor(config: SandboxConfig) -> SandboxExecutor:
    return SandboxExecutor(config)


# ============================================================================
# SandboxResult
# ============================================================================


class TestSandboxResult:
    def test_success(self) -> None:
        r = SandboxResult(stdout="ok", exit_code=0)
        assert r.success is True
        assert r.output == "ok"

    def test_failure_exit_code(self) -> None:
        r = SandboxResult(exit_code=1)
        assert r.success is False
        assert "[EXIT CODE: 1]" in r.output

    def test_timeout(self) -> None:
        r = SandboxResult(timed_out=True, exit_code=-1)
        assert r.success is False
        assert "[TIMEOUT]" in r.output

    def test_error(self) -> None:
        r = SandboxResult(error="Sandbox kaputt")
        assert r.success is False
        assert "Sandbox kaputt" in r.output

    def test_combined_output(self) -> None:
        r = SandboxResult(stdout="out", stderr="err", exit_code=1, truncated=True)
        assert "out" in r.output
        assert "[STDERR]" in r.output
        assert "err" in r.output
        assert "[EXIT CODE: 1]" in r.output
        assert "gekürzt" in r.output

    def test_empty_output(self) -> None:
        r = SandboxResult()
        assert r.output == "(Keine Ausgabe)"


# ============================================================================
# SandboxConfig
# ============================================================================


class TestSandboxConfig:
    def test_defaults(self) -> None:
        cfg = SandboxConfig()
        assert cfg.preferred_level == SandboxLevel.BWRAP
        assert cfg.network == NetworkPolicy.ALLOW
        assert cfg.max_memory_mb == 512
        assert cfg.default_timeout == 30
        if sys.platform != "win32":
            assert "/usr" in cfg.allowed_read_paths
        else:
            assert cfg.allowed_read_paths == []

    def test_custom(self, workspace: Path) -> None:
        cfg = SandboxConfig(
            workspace_dir=workspace,
            preferred_level=SandboxLevel.FIREJAIL,
            network=NetworkPolicy.BLOCK,
            max_memory_mb=256,
        )
        assert cfg.workspace_dir == workspace
        assert cfg.preferred_level == SandboxLevel.FIREJAIL
        assert cfg.network == NetworkPolicy.BLOCK


# ============================================================================
# BwrapSandbox
# ============================================================================


class TestBwrapSandbox:
    def test_build_command_basic(self, workspace: Path) -> None:
        cfg = SandboxConfig(workspace_dir=workspace)
        bwrap = BwrapSandbox(cfg)
        args = bwrap.build_command("echo hello", str(workspace))

        assert args[0] == "bwrap"
        assert "--unshare-all" in args
        assert "--share-net" in args  # Default: Netzwerk erlaubt
        assert "--proc" in args
        assert "--dev" in args
        assert "--tmpfs" in args
        assert str(workspace) in args  # Workspace gemountet
        assert "echo hello" in args[-1]  # Befehl am Ende

    def test_build_command_no_network(self, workspace: Path) -> None:
        cfg = SandboxConfig(workspace_dir=workspace, network=NetworkPolicy.BLOCK)
        bwrap = BwrapSandbox(cfg)
        args = bwrap.build_command("curl evil.com", str(workspace))

        assert "--share-net" not in args

    def test_build_command_ulimits(self, workspace: Path) -> None:
        cfg = SandboxConfig(workspace_dir=workspace, max_memory_mb=256, max_processes=32)
        bwrap = BwrapSandbox(cfg)
        args = bwrap.build_command("stress-test", str(workspace))

        shell_cmd = args[-1]
        assert "ulimit -v 262144" in shell_cmd  # 256 * 1024
        assert "ulimit -u 32" in shell_cmd


# ============================================================================
# FirejailSandbox
# ============================================================================


class TestFirejailSandbox:
    def test_build_command_basic(self, workspace: Path) -> None:
        cfg = SandboxConfig(workspace_dir=workspace)
        fj = FirejailSandbox(cfg)
        args = fj.build_command("echo hello", str(workspace))

        assert args[0] == "firejail"
        assert "--quiet" in args
        assert "--noprofile" in args
        assert "--noroot" in args
        assert f"--private={workspace}" in args

    def test_build_command_no_network(self, workspace: Path) -> None:
        cfg = SandboxConfig(workspace_dir=workspace, network=NetworkPolicy.BLOCK)
        fj = FirejailSandbox(cfg)
        args = fj.build_command("wget evil.com", str(workspace))

        assert "--net=none" in args


# ============================================================================
# SandboxExecutor — Bare-Modus
# ============================================================================


class TestSandboxExecutorBare:
    @pytest.mark.asyncio
    async def test_exec_echo(self, executor: SandboxExecutor, workspace: Path) -> None:
        result = await executor.execute("echo 'Hallo Sandbox'", working_dir=str(workspace))
        assert result.success is True
        assert "Hallo Sandbox" in result.stdout
        assert result.sandbox_level in ("bare", "jobobject")

    @pytest.mark.asyncio
    async def test_exec_empty_command(self, executor: SandboxExecutor) -> None:
        result = await executor.execute("")
        assert result.success is False
        assert "Kein Befehl" in result.error

    @pytest.mark.asyncio
    async def test_exec_exit_code(self, executor: SandboxExecutor, workspace: Path) -> None:
        result = await executor.execute("exit 42", working_dir=str(workspace))
        assert result.exit_code == 42
        assert result.success is False

    @pytest.mark.asyncio
    async def test_exec_stderr(self, executor: SandboxExecutor, workspace: Path) -> None:
        result = await executor.execute("echo err >&2", working_dir=str(workspace))
        assert "err" in result.stderr

    @pytest.mark.asyncio
    async def test_exec_timeout(self, executor: SandboxExecutor, workspace: Path) -> None:
        result = await executor.execute("sleep 30", working_dir=str(workspace), timeout=1)
        assert result.timed_out is True
        assert result.success is False

    @pytest.mark.asyncio
    async def test_exec_creates_working_dir(self, executor: SandboxExecutor, tmp_path: Path) -> None:
        new_dir = tmp_path / "sub" / "dir"
        result = await executor.execute("pwd", working_dir=str(new_dir))
        assert result.success is True
        assert new_dir.exists()

    @pytest.mark.asyncio
    async def test_exec_large_output_truncated(self, executor: SandboxExecutor, workspace: Path) -> None:
        # Generiere Output > 50KB via temporaeres Script (vermeidet cmd.exe Quote-Probleme)
        script = workspace / "_large_output.py"
        script.write_text("print('A' * 60000)", encoding="utf-8")
        python = sys.executable
        result = await executor.execute(
            f"{python} {script}",
            working_dir=str(workspace),
        )
        assert result.truncated is True

    @pytest.mark.asyncio
    async def test_exec_unicode(self, executor: SandboxExecutor, workspace: Path) -> None:
        script = workspace / "_unicode_test.py"
        script.write_text("print('Ü Ö Ä ß')", encoding="utf-8")
        python = sys.executable
        result = await executor.execute(
            f"{python} {script}",
            working_dir=str(workspace),
        )
        assert result.success is True
        assert "Ü" in result.stdout or "\\xdc" in repr(result.stdout)


# ============================================================================
# SandboxExecutor — Level-Erkennung
# ============================================================================


class TestSandboxDetection:
    def test_bare_when_nothing_available(self) -> None:
        with patch("jarvis.core.sandbox.BwrapSandbox.is_available", return_value=False), \
             patch("jarvis.core.sandbox.FirejailSandbox.is_available", return_value=False):
            cfg = SandboxConfig(preferred_level=SandboxLevel.BWRAP)
            executor = SandboxExecutor(cfg)
            # Auf Windows fällt der Sandbox auf JOBOBJECT zurück, auf Linux auf BARE
            assert executor.level in (SandboxLevel.BARE, SandboxLevel.JOBOBJECT)

    def test_bwrap_when_available(self) -> None:
        with patch("jarvis.core.sandbox.BwrapSandbox.is_available", return_value=True):
            cfg = SandboxConfig(preferred_level=SandboxLevel.BWRAP)
            executor = SandboxExecutor(cfg)
            assert executor.level == SandboxLevel.BWRAP

    def test_firejail_when_preferred_and_available(self) -> None:
        with patch("jarvis.core.sandbox.BwrapSandbox.is_available", return_value=False), \
             patch("jarvis.core.sandbox.FirejailSandbox.is_available", return_value=True):
            cfg = SandboxConfig(preferred_level=SandboxLevel.FIREJAIL)
            executor = SandboxExecutor(cfg)
            assert executor.level == SandboxLevel.FIREJAIL

    def test_bwrap_fallback_when_firejail_preferred_but_unavailable(self) -> None:
        with patch("jarvis.core.sandbox.BwrapSandbox.is_available", return_value=True), \
             patch("jarvis.core.sandbox.FirejailSandbox.is_available", return_value=False):
            cfg = SandboxConfig(preferred_level=SandboxLevel.FIREJAIL)
            executor = SandboxExecutor(cfg)
            assert executor.level == SandboxLevel.BWRAP


# ============================================================================
# SandboxExecutor — Network-Policy Override
# ============================================================================


class TestNetworkPolicyOverride:
    @pytest.mark.asyncio
    async def test_network_override_restored(self, config: SandboxConfig) -> None:
        executor = SandboxExecutor(config)
        assert config.network == NetworkPolicy.ALLOW

        await executor.execute("echo test", network=NetworkPolicy.BLOCK)

        # Original sollte wiederhergestellt sein
        assert config.network == NetworkPolicy.ALLOW


# ============================================================================
# Shell-Integration
# ============================================================================


class TestShellToolsIntegration:
    @pytest.mark.asyncio
    async def test_shell_uses_sandbox(self, tmp_path: Path) -> None:
        from jarvis.mcp.shell import ShellTools

        config = MagicMock()
        config.workspace_dir = tmp_path
        config.sandbox_level = "bare"
        config.sandbox_network = "allow"

        shell = ShellTools(config)
        assert shell.sandbox_level in ("bare", "bwrap", "firejail", "jobobject")

        result = await shell.exec_command("echo 'Sandbox aktiv'")
        assert "Sandbox aktiv" in result

    @pytest.mark.asyncio
    async def test_shell_empty_command(self, tmp_path: Path) -> None:
        from jarvis.mcp.shell import ShellTools

        config = MagicMock()
        config.workspace_dir = tmp_path
        config.sandbox_level = "bare"
        config.sandbox_network = "allow"

        shell = ShellTools(config)
        result = await shell.exec_command("")
        assert "Kein Befehl" in result

    @pytest.mark.asyncio
    async def test_shell_register(self, tmp_path: Path) -> None:
        from jarvis.mcp.shell import register_shell_tools

        config = MagicMock()
        config.workspace_dir = tmp_path
        config.sandbox_level = "bare"
        config.sandbox_network = "allow"

        mcp_client = MagicMock()
        shell = register_shell_tools(mcp_client, config)

        mcp_client.register_builtin_handler.assert_called_once()
        assert shell.sandbox_level in ("bare", "bwrap", "firejail", "jobobject")


class TestSandboxOverrides:
    """Per-Agent Sandbox-Overrides werden korrekt angewendet und zurückgesetzt."""

    @pytest.mark.asyncio
    async def test_memory_override_applied_and_restored(self) -> None:
        config = SandboxConfig(max_memory_mb=512, max_processes=64)
        executor = SandboxExecutor(config)

        original_memory = executor._config.max_memory_mb
        original_procs = executor._config.max_processes

        # Execute with overrides
        await executor.execute(
            "echo test",
            max_memory_mb=1024,
            max_processes=128,
        )

        # Config wurde zurückgesetzt
        assert executor._config.max_memory_mb == original_memory
        assert executor._config.max_processes == original_procs

    @pytest.mark.asyncio
    async def test_network_override_restored_on_error(self) -> None:
        config = SandboxConfig(network=NetworkPolicy.ALLOW)
        executor = SandboxExecutor(config)

        # Selbst bei einem fehlerhaften Befehl wird die Config wiederhergestellt
        await executor.execute(
            "exit 1",
            network=NetworkPolicy.BLOCK,
        )

        assert executor._config.network == NetworkPolicy.ALLOW

    @pytest.mark.asyncio
    async def test_no_override_keeps_defaults(self) -> None:
        config = SandboxConfig(max_memory_mb=256, max_processes=32)
        executor = SandboxExecutor(config)

        await executor.execute("echo test")

        assert executor._config.max_memory_mb == 256
        assert executor._config.max_processes == 32
