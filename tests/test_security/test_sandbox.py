"""Tests für security/sandbox.py – Sandbox-Isolierung."""

from __future__ import annotations

import sys

import pytest

from jarvis.models import SandboxConfig, SandboxLevel
from jarvis.security.sandbox import Sandbox, SandboxResult

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sandbox() -> Sandbox:
    return Sandbox(
        SandboxConfig(
            level=SandboxLevel.PROCESS,
            timeout_seconds=10,
            max_memory_mb=128,
            max_cpu_seconds=5,
        )
    )


# ============================================================================
# Capabilities
# ============================================================================


class TestCapabilities:
    def test_process_always_available(self, sandbox: Sandbox):
        assert SandboxLevel.PROCESS in sandbox.available_levels

    def test_capabilities_dict(self, sandbox: Sandbox):
        caps = sandbox.capabilities
        assert "process" in caps
        assert caps["process"] is True
        assert "docker" in caps
        assert "bwrap" in caps

    def test_max_level_at_least_process(self, sandbox: Sandbox):
        assert sandbox.max_level in (
            SandboxLevel.PROCESS,
            SandboxLevel.NAMESPACE,
            SandboxLevel.CONTAINER,
        )


# ============================================================================
# Process-Level Execution
# ============================================================================


class TestProcessExecution:
    @pytest.mark.asyncio
    async def test_simple_command(self, sandbox: Sandbox):
        result = await sandbox.execute(
            "echo hello",
            level=SandboxLevel.PROCESS,
        )
        assert result.exit_code == 0
        assert "hello" in result.stdout
        assert result.sandbox_level == SandboxLevel.PROCESS

    @pytest.mark.asyncio
    async def test_exit_code(self, sandbox: Sandbox):
        result = await sandbox.execute(
            "python -c \"import sys; sys.exit(42)\"",
            level=SandboxLevel.PROCESS,
        )
        assert result.exit_code == 42

    @pytest.mark.asyncio
    async def test_stderr(self, sandbox: Sandbox):
        result = await sandbox.execute(
            "python -c \"import sys; sys.stderr.write('error\\n')\"",
            level=SandboxLevel.PROCESS,
        )
        assert "error" in result.stderr

    @pytest.mark.asyncio
    async def test_timeout(self):
        sb = Sandbox(SandboxConfig(timeout_seconds=1))
        cmd = "sleep 10" if sys.platform != "win32" else "ping -n 11 127.0.0.1"
        result = await sb.execute(
            cmd,
            level=SandboxLevel.PROCESS,
            timeout=1,
        )
        assert result.timed_out is True
        assert result.killed is True
        assert result.exit_code == -1

    @pytest.mark.asyncio
    async def test_working_directory(self, sandbox: Sandbox, tmp_path):
        (tmp_path / "test.txt").write_text("found")
        result = await sandbox.execute(
            "python -c \"print(open('test.txt').read())\"",
            level=SandboxLevel.PROCESS,
            working_dir=str(tmp_path),
        )
        assert result.exit_code == 0
        assert "found" in result.stdout

    @pytest.mark.asyncio
    async def test_env_vars(self, sandbox: Sandbox):
        result = await sandbox.execute(
            "python -c \"import os; print(os.environ.get('MY_VAR', ''))\"",
            level=SandboxLevel.PROCESS,
            env={"MY_VAR": "custom_value"},
        )
        assert "custom_value" in result.stdout

    @pytest.mark.asyncio
    async def test_duration_tracked(self, sandbox: Sandbox):
        cmd = "sleep 0.1" if sys.platform != "win32" else "ping -n 2 127.0.0.1"
        result = await sandbox.execute(
            cmd,
            level=SandboxLevel.PROCESS,
        )
        assert result.duration_ms >= 50  # At least some time passed

    @pytest.mark.asyncio
    async def test_multi_line_output(self, sandbox: Sandbox):
        result = await sandbox.execute(
            "echo line1; echo line2; echo line3",
            level=SandboxLevel.PROCESS,
        )
        assert result.exit_code == 0
        assert "line1" in result.stdout
        assert "line3" in result.stdout


class TestLevelDowngrade:
    @pytest.mark.asyncio
    async def test_downgrades_unavailable_level(self, sandbox: Sandbox):
        # CONTAINER level may not be available (no Docker in test env)
        result = await sandbox.execute(
            "echo test",
            level=SandboxLevel.CONTAINER,
        )
        # Should still work (downgraded to PROCESS, or Docker found but not running)
        assert result.exit_code == 0 or "docker" in result.stderr.lower()


class TestSandboxResult:
    def test_result_fields(self):
        r = SandboxResult(
            exit_code=0,
            stdout="out",
            stderr="err",
            duration_ms=100,
            sandbox_level=SandboxLevel.PROCESS,
        )
        assert r.exit_code == 0
        assert r.killed is False
        assert r.oom_killed is False
        assert r.timed_out is False

    def test_result_timeout_flags(self):
        r = SandboxResult(
            exit_code=-1,
            stdout="",
            stderr="timeout",
            duration_ms=5000,
            sandbox_level=SandboxLevel.PROCESS,
            killed=True,
            timed_out=True,
        )
        assert r.killed is True
        assert r.timed_out is True

    def test_result_oom_killed(self):
        r = SandboxResult(
            exit_code=-1,
            stdout="",
            stderr="oom",
            duration_ms=0,
            sandbox_level=SandboxLevel.PROCESS,
            oom_killed=True,
        )
        assert r.oom_killed is True
        assert r.killed is False


# ============================================================================
# Default-Config und _build_env
# ============================================================================


class TestDefaultConfig:
    def test_no_config_uses_defaults(self):
        sb = Sandbox()
        assert SandboxLevel.PROCESS in sb.available_levels

    @pytest.mark.asyncio
    async def test_default_timeout_from_config(self):
        sb = Sandbox(SandboxConfig(timeout_seconds=5))
        result = await sb.execute("echo ok")
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_default_level_from_config(self):
        sb = Sandbox(SandboxConfig(level=SandboxLevel.PROCESS))
        result = await sb.execute("echo ok")
        assert result.sandbox_level == SandboxLevel.PROCESS

    @pytest.mark.asyncio
    async def test_network_default_from_config(self):
        sb = Sandbox(SandboxConfig(network_access=True))
        # Sollte nicht crashen — network-Flag wird durchgereicht
        result = await sb.execute("echo ok")
        assert result.exit_code == 0


class TestBuildEnv:
    def test_minimal_safe_env(self, sandbox: Sandbox):
        env = sandbox._build_env()
        assert "PATH" in env
        assert "HOME" in env
        if sys.platform != "win32":
            assert "LANG" in env
            assert env["LANG"] == "C.UTF-8"
        else:
            assert "SYSTEMROOT" in env

    def test_config_env_vars_merged(self):
        sb = Sandbox(SandboxConfig(env_vars={"CUSTOM": "from_config"}))
        env = sb._build_env()
        assert env["CUSTOM"] == "from_config"

    def test_extra_env_overrides_config(self):
        sb = Sandbox(SandboxConfig(env_vars={"KEY": "config_val"}))
        env = sb._build_env(extra={"KEY": "extra_val"})
        assert env["KEY"] == "extra_val"

    def test_extra_env_none(self, sandbox: Sandbox):
        env = sandbox._build_env(extra=None)
        assert "PATH" in env

    def test_extra_env_adds_new_keys(self, sandbox: Sandbox):
        env = sandbox._build_env(extra={"NEW_KEY": "new_val"})
        assert env["NEW_KEY"] == "new_val"
        assert "PATH" in env  # Basis-Keys noch da


# ============================================================================
# Namespace/Docker Fallback-Pfade
# ============================================================================


class TestNamespaceFallback:
    @pytest.mark.asyncio
    async def test_namespace_falls_back_to_process_without_bwrap(self):
        sb = Sandbox(SandboxConfig(level=SandboxLevel.NAMESPACE))
        # bwrap ist in der Testumgebung typischerweise nicht da
        if not sb.capabilities.get("bwrap"):
            result = await sb._exec_namespace("echo fallback_test")
            assert result.exit_code == 0
            assert "fallback_test" in result.stdout

    @pytest.mark.asyncio
    async def test_namespace_with_network_flag(self):
        sb = Sandbox(SandboxConfig(level=SandboxLevel.NAMESPACE))
        if not sb.capabilities.get("bwrap"):
            result = await sb._exec_namespace("echo net", network=True)
            assert result.exit_code == 0


class TestDockerFallback:
    @pytest.mark.asyncio
    async def test_docker_falls_back_without_docker(self):
        sb = Sandbox(SandboxConfig(level=SandboxLevel.CONTAINER))
        if not sb.capabilities.get("docker"):
            result = await sb._exec_docker("echo docker_fallback")
            assert result.exit_code == 0
            assert "docker_fallback" in result.stdout

    @pytest.mark.asyncio
    async def test_docker_with_network_flag(self):
        sb = Sandbox(SandboxConfig(level=SandboxLevel.CONTAINER))
        if not sb.capabilities.get("docker"):
            result = await sb._exec_docker("echo net", network=True)
            assert result.exit_code == 0


# ============================================================================
# OSError-Handling
# ============================================================================


class TestOSErrorHandling:
    @pytest.mark.asyncio
    async def test_process_oserror_returns_result(self, sandbox: Sandbox, monkeypatch):
        import asyncio as _asyncio

        async def _boom(*a, **kw):
            raise OSError("spawn failed")

        monkeypatch.setattr(_asyncio, "create_subprocess_exec", _boom)
        result = await sandbox._exec_process("echo unreachable")
        assert result.exit_code == -1
        assert "spawn failed" in result.stderr

    @pytest.mark.asyncio
    async def test_namespace_oserror_returns_result(self, monkeypatch):
        import asyncio as _asyncio

        sb = Sandbox()
        # Simuliere bwrap vorhanden aber exec schlägt fehl
        sb._capabilities["bwrap"] = True

        async def _boom(*a, **kw):
            raise OSError("bwrap crashed")

        monkeypatch.setattr(_asyncio, "create_subprocess_exec", _boom)
        result = await sb._exec_namespace("echo unreachable")
        assert result.exit_code == -1
        assert "bwrap crashed" in result.stderr

    @pytest.mark.asyncio
    async def test_docker_oserror_returns_result(self, monkeypatch):
        import asyncio as _asyncio

        sb = Sandbox()
        sb._capabilities["docker"] = True

        async def _boom(*a, **kw):
            raise OSError("docker daemon gone")

        monkeypatch.setattr(_asyncio, "create_subprocess_exec", _boom)
        result = await sb._exec_docker("echo unreachable")
        assert result.exit_code == -1
        assert "docker daemon gone" in result.stderr


# ============================================================================
# Timeout in Namespace/Docker (mit mocked capabilities)
# ============================================================================


class TestNamespaceTimeout:
    @pytest.mark.asyncio
    async def test_namespace_timeout_with_bwrap(self, monkeypatch):
        import asyncio as _asyncio

        sb = Sandbox(SandboxConfig(timeout_seconds=1))
        sb._capabilities["bwrap"] = True

        class FakeProc:
            returncode = None

            async def communicate(self):
                await _asyncio.sleep(10)
                return b"", b""

            def kill(self):
                pass

            async def wait(self):
                self.returncode = -9

        async def _fake_exec(*a, **kw):
            return FakeProc()

        monkeypatch.setattr(_asyncio, "create_subprocess_exec", _fake_exec)
        result = await sb._exec_namespace("sleep 100", timeout=1)
        assert result.timed_out is True
        assert result.killed is True

    @pytest.mark.asyncio
    async def test_docker_timeout_with_docker(self, monkeypatch):
        import asyncio as _asyncio

        sb = Sandbox(SandboxConfig(timeout_seconds=1))
        sb._capabilities["docker"] = True

        class FakeProc:
            returncode = None

            async def communicate(self):
                await _asyncio.sleep(10)
                return b"", b""

            def kill(self):
                pass

            async def wait(self):
                self.returncode = -9

        async def _fake_exec(*a, **kw):
            return FakeProc()

        monkeypatch.setattr(_asyncio, "create_subprocess_exec", _fake_exec)
        result = await sb._exec_docker("sleep 100", timeout=1)
        assert result.timed_out is True
        assert result.killed is True
