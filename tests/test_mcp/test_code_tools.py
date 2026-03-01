"""Tests für Code-Tools -- run_python und analyze_code.

Testet:
  - TestRunPython: print, syntax error, empty code, timeout, workspace confinement
  - TestAnalyzeCode: clean code, smell, security issue, no input
  - TestRegistration: 2 Tools registriert
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from jarvis.config import JarvisConfig, SecurityConfig, ensure_directory_structure
from jarvis.mcp.code_tools import MAX_CODE_SIZE, CodeTools, register_code_tools

if TYPE_CHECKING:
    from pathlib import Path


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
def config(tmp_path: "Path") -> JarvisConfig:
    cfg = JarvisConfig(
        jarvis_home=tmp_path / ".jarvis",
        security=SecurityConfig(allowed_paths=[str(tmp_path)]),
    )
    ensure_directory_structure(cfg)
    return cfg


@pytest.fixture()
def code_tools(config: JarvisConfig) -> CodeTools:
    return CodeTools(config)


# =============================================================================
# TestRunPython
# =============================================================================


class TestRunPython:
    """Tests für run_python."""

    @pytest.mark.asyncio()
    async def test_print_output(self, code_tools: CodeTools) -> None:
        """Einfacher print-Befehl gibt Output zurück."""
        result = await code_tools.run_python('print("Hello Jarvis")')
        assert "Hello Jarvis" in result

    @pytest.mark.asyncio()
    async def test_syntax_error(self, code_tools: CodeTools) -> None:
        """Syntax-Fehler gibt stderr/exit code zurück."""
        result = await code_tools.run_python("def foo(:\n  pass")
        # Sollte einen Fehler enthalten (SyntaxError oder exit code)
        assert "SyntaxError" in result or "EXIT CODE" in result or "Error" in result

    @pytest.mark.asyncio()
    async def test_empty_code(self, code_tools: CodeTools) -> None:
        """Leerer Code gibt Fehlermeldung zurück."""
        result = await code_tools.run_python("")
        assert "Kein Code" in result

    @pytest.mark.asyncio()
    async def test_whitespace_only(self, code_tools: CodeTools) -> None:
        """Nur Whitespace gibt Fehlermeldung zurück."""
        result = await code_tools.run_python("   \n  \n  ")
        assert "Kein Code" in result

    @pytest.mark.asyncio()
    async def test_multiline_code(self, code_tools: CodeTools) -> None:
        """Mehrzeiliger Code funktioniert."""
        code = "x = 5\ny = 10\nprint(x + y)"
        result = await code_tools.run_python(code)
        assert "15" in result

    @pytest.mark.asyncio()
    async def test_workspace_confinement(self, code_tools: CodeTools) -> None:
        """Arbeitsverzeichnis außerhalb Workspace wird abgelehnt."""
        result = await code_tools.run_python(
            'print("test")', working_dir="/tmp/evil"
        )
        assert "Zugriff verweigert" in result

    @pytest.mark.asyncio()
    async def test_temp_file_cleanup(self, code_tools: CodeTools, config: JarvisConfig) -> None:
        """Temp-Datei wird nach Ausführung gelöscht."""
        import os
        workspace = config.workspace_dir
        before = set(os.listdir(workspace)) if workspace.exists() else set()
        await code_tools.run_python('print("cleanup test")')
        after = set(os.listdir(workspace)) if workspace.exists() else set()
        # Keine _jarvis_run_*.py Dateien übrig
        leftover = {f for f in (after - before) if f.startswith("_jarvis_run_")}
        assert len(leftover) == 0


# =============================================================================
# TestAnalyzeCode
# =============================================================================


class TestAnalyzeCode:
    """Tests für analyze_code."""

    @pytest.mark.asyncio()
    async def test_clean_code(self, code_tools: CodeTools) -> None:
        """Sauberer Code gibt positiven Bericht."""
        code = 'def add(a, b):\n    """Addiert zwei Zahlen."""\n    return a + b\n'
        result = await code_tools.analyze_code(code=code)
        assert "Code-Analyse" in result
        assert "Keine gefunden" in result or "0 gefunden" in result

    @pytest.mark.asyncio()
    async def test_smell_detection(self, code_tools: CodeTools) -> None:
        """Code-Smell (zu viele Parameter) wird erkannt."""
        code = "def foo(a, b, c, d, e, f, g, h):\n    pass\n"
        result = await code_tools.analyze_code(code=code)
        assert "too_many_params" in result or "Parameter" in result

    @pytest.mark.asyncio()
    async def test_security_finding(self, code_tools: CodeTools) -> None:
        """Security-Finding (eval) wird erkannt."""
        code = 'x = eval("2 + 2")\nprint(x)\n'
        result = await code_tools.analyze_code(code=code)
        assert "eval" in result.lower() or "CRITICAL" in result or "Sicherheit" in result

    @pytest.mark.asyncio()
    async def test_no_input(self, code_tools: CodeTools) -> None:
        """Kein Input gibt Fehlermeldung."""
        result = await code_tools.analyze_code()
        assert "Fehler" in result

    @pytest.mark.asyncio()
    async def test_file_not_found(self, code_tools: CodeTools) -> None:
        """Nicht existierende Datei gibt Fehler."""
        result = await code_tools.analyze_code(file_path="/nonexistent/file.py")
        assert "nicht gefunden" in result

    @pytest.mark.asyncio()
    async def test_non_python_file(self, code_tools: CodeTools) -> None:
        """Nicht-Python-Datei wird abgelehnt."""
        result = await code_tools.analyze_code(file_path="/some/file.txt")
        assert "Nur Python" in result or "nicht gefunden" in result

    @pytest.mark.asyncio()
    async def test_file_analysis(
        self, code_tools: CodeTools, config: JarvisConfig,
    ) -> None:
        """Datei-basierte Analyse funktioniert."""
        test_file = config.workspace_dir / "test_analyze.py"
        test_file.write_text(
            "def simple():\n    return 42\n",
            encoding="utf-8",
        )
        try:
            result = await code_tools.analyze_code(file_path=str(test_file))
            assert "Code-Analyse" in result
        finally:
            test_file.unlink(missing_ok=True)


# =============================================================================
# TestRegistration
# =============================================================================


class TestRegistration:
    """Tests für register_code_tools."""

    def test_registers_two_tools(self, config: JarvisConfig) -> None:
        """Registriert genau 2 Tools (run_python, analyze_code)."""
        mock_client = MagicMock()
        register_code_tools(mock_client, config)

        assert mock_client.register_builtin_handler.call_count == 2

        registered_names = [
            call.args[0]
            for call in mock_client.register_builtin_handler.call_args_list
        ]
        assert "run_python" in registered_names
        assert "analyze_code" in registered_names

    def test_returns_code_tools_instance(self, config: JarvisConfig) -> None:
        """Gibt eine CodeTools-Instanz zurück."""
        mock_client = MagicMock()
        result = register_code_tools(mock_client, config)
        assert isinstance(result, CodeTools)


# =============================================================================
# File-Size-Limits (Security Hardening)
# =============================================================================


class TestCodeSizeLimits:
    """Tests für Code-Größen-Limits."""

    @pytest.mark.asyncio()
    async def test_run_python_code_too_large(self, code_tools: CodeTools) -> None:
        """Code über MAX_CODE_SIZE wird abgelehnt."""
        huge_code = "x = 1\n" * (MAX_CODE_SIZE // 6 + 1)
        assert len(huge_code.encode("utf-8")) > MAX_CODE_SIZE
        result = await code_tools.run_python(huge_code)
        assert "zu gross" in result.lower() or "too large" in result.lower() or "gross" in result.lower()

    @pytest.mark.asyncio()
    async def test_run_python_within_limit(self, code_tools: CodeTools) -> None:
        """Code innerhalb des Limits wird normal ausgeführt."""
        result = await code_tools.run_python('print("ok")')
        assert "ok" in result
