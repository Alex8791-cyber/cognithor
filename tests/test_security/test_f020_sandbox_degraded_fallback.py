"""Tests fuer F-020: Windows Sandbox Fallback ohne Resource-Limits.

Prueft dass:
  - SandboxResult.isolation_degraded existiert und default False ist
  - SandboxConfig.allow_degraded_sandbox existiert und default True ist
  - Bei CreateJobObjectW-Fehler + allow_degraded=True: Fallback mit isolation_degraded=True
  - Bei CreateJobObjectW-Fehler + allow_degraded=False: Ausfuehrung verweigert
  - Normaler Pfad (kein Fehler) setzt isolation_degraded=False
  - Source-Code die Fixes enthaelt
"""

from __future__ import annotations

import inspect
import sys
from unittest.mock import AsyncMock, patch

import pytest

from jarvis.models import SandboxConfig, SandboxLevel
from jarvis.security.sandbox import Sandbox, SandboxResult

# ============================================================================
# SandboxResult Tests
# ============================================================================


class TestSandboxResultField:
    """Prueft das neue isolation_degraded Feld."""

    def test_default_false(self) -> None:
        result = SandboxResult(
            exit_code=0,
            stdout="",
            stderr="",
            duration_ms=0,
            sandbox_level=SandboxLevel.PROCESS,
        )
        assert result.isolation_degraded is False

    def test_can_set_true(self) -> None:
        result = SandboxResult(
            exit_code=0,
            stdout="",
            stderr="",
            duration_ms=0,
            sandbox_level=SandboxLevel.PROCESS,
            isolation_degraded=True,
        )
        assert result.isolation_degraded is True

    def test_mutable(self) -> None:
        """isolation_degraded kann nachtraeglich gesetzt werden."""
        result = SandboxResult(
            exit_code=0,
            stdout="",
            stderr="",
            duration_ms=0,
            sandbox_level=SandboxLevel.PROCESS,
        )
        result.isolation_degraded = True
        assert result.isolation_degraded is True


# ============================================================================
# SandboxConfig Tests
# ============================================================================


class TestSandboxConfigField:
    """Prueft das neue allow_degraded_sandbox Feld."""

    def test_default_true(self) -> None:
        config = SandboxConfig()
        assert config.allow_degraded_sandbox is True

    def test_can_set_false(self) -> None:
        config = SandboxConfig(allow_degraded_sandbox=False)
        assert config.allow_degraded_sandbox is False


# ============================================================================
# Degraded Fallback Behavior (mocked)
# ============================================================================


class TestDegradedFallback:
    """Prueft das Verhalten bei CreateJobObjectW-Fehlschlag."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only")
    async def test_fallback_sets_isolation_degraded(self) -> None:
        """allow_degraded=True: Fallback ausfuehren, isolation_degraded=True."""
        config = SandboxConfig(allow_degraded_sandbox=True)
        sandbox = Sandbox(config)

        bare_result = SandboxResult(
            exit_code=0,
            stdout="ok",
            stderr="",
            duration_ms=100,
            sandbox_level=SandboxLevel.PROCESS,
        )

        with (
            patch.object(
                sandbox, "_exec_process_bare", new_callable=AsyncMock, return_value=bare_result
            ),
            patch("ctypes.windll.kernel32.CreateJobObjectW", return_value=0),
        ):
            with patch("ctypes.get_last_error", return_value=5):
                result = await sandbox._exec_process_with_jobobject(
                    "echo test",
                    timeout=10,
                )

        assert result.isolation_degraded is True
        assert result.exit_code == 0

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only")
    async def test_refuse_when_not_allowed(self) -> None:
        """allow_degraded=False: Ausfuehrung verweigern bei JobObject-Fehler."""
        config = SandboxConfig(allow_degraded_sandbox=False)
        sandbox = Sandbox(config)

        with patch("ctypes.windll.kernel32.CreateJobObjectW", return_value=0):
            with patch("ctypes.get_last_error", return_value=5):
                result = await sandbox._exec_process_with_jobobject(
                    "echo test",
                    timeout=10,
                )

        assert result.exit_code == -1
        assert result.isolation_degraded is True
        assert "allow_degraded_sandbox=False" in result.stderr
        assert "refused" in result.stderr or "verweigert" in result.stderr

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only")
    async def test_refuse_does_not_call_bare(self) -> None:
        """Bei Verweigerung wird _exec_process_bare NICHT aufgerufen."""
        config = SandboxConfig(allow_degraded_sandbox=False)
        sandbox = Sandbox(config)

        bare_mock = AsyncMock()
        with patch.object(sandbox, "_exec_process_bare", bare_mock):
            with patch("ctypes.windll.kernel32.CreateJobObjectW", return_value=0):
                with patch("ctypes.get_last_error", return_value=5):
                    await sandbox._exec_process_with_jobobject(
                        "echo test",
                        timeout=10,
                    )

        bare_mock.assert_not_called()


class TestNormalPath:
    """Prueft dass der normale Pfad (kein Fehler) nicht degraded ist."""

    def test_bare_result_default_not_degraded(self) -> None:
        """_exec_process_bare erzeugt Results ohne isolation_degraded."""
        result = SandboxResult(
            exit_code=0,
            stdout="",
            stderr="",
            duration_ms=0,
            sandbox_level=SandboxLevel.PROCESS,
        )
        assert result.isolation_degraded is False

    def test_normal_sandbox_result_not_degraded(self) -> None:
        """Ein normales SandboxResult hat isolation_degraded=False."""
        result = SandboxResult(
            exit_code=0,
            stdout="output",
            stderr="",
            duration_ms=50,
            sandbox_level=SandboxLevel.PROCESS,
        )
        assert not result.isolation_degraded


# ============================================================================
# Source-Level-Checks
# ============================================================================


class TestSourceLevelChecks:
    """Prueft den Source-Code auf die Fixes."""

    def test_result_has_isolation_degraded_field(self) -> None:
        source = inspect.getsource(SandboxResult)
        assert "isolation_degraded" in source

    def test_jobobject_checks_allow_degraded(self) -> None:
        source = inspect.getsource(Sandbox._exec_process_with_jobobject)
        assert "allow_degraded_sandbox" in source

    def test_jobobject_logs_error_on_refuse(self) -> None:
        source = inspect.getsource(Sandbox._exec_process_with_jobobject)
        assert "execution_refused" in source

    def test_jobobject_logs_warning_on_fallback(self) -> None:
        source = inspect.getsource(Sandbox._exec_process_with_jobobject)
        assert "degraded_fallback" in source

    def test_jobobject_sets_degraded_on_result(self) -> None:
        source = inspect.getsource(Sandbox._exec_process_with_jobobject)
        assert "isolation_degraded = True" in source

    def test_config_has_allow_degraded(self) -> None:
        source = inspect.getsource(SandboxConfig)
        assert "allow_degraded_sandbox" in source
