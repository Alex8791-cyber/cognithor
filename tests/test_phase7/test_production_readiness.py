"""Phase 7 Tests: Production Readiness.

Testet:
  1. Healthcheck-Modul
  2. Smoke-Test importierbar (kein Ollama nötig)
  3. Deployment-Dateien vorhanden und korrekt
  4. Install-Skript Syntax
  5. systemd Service-Datei Format
  6. Requirements.txt parsierbar
  7. Planner-Prompts enthalten erwartete Schlüsselwörter
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# =============================================================================
# 1. Healthcheck
# =============================================================================


class TestHealthcheck:
    """Healthcheck-Modul Funktionalität."""

    def test_healthy_status(self):
        """Gesundes System meldet 'healthy'."""
        from jarvis.healthcheck import health_status

        result = health_status(
            ollama_available=True,
            channels_active=["cli", "telegram"],
            models_loaded=["qwen3:32b"],
        )
        assert result["status"] == "healthy"
        assert result["ollama"] is True
        assert "cli" in result["channels"]
        assert result["errors"] == []
        assert "uptime_seconds" in result
        assert "started_at" in result
        assert "timestamp" in result

    def test_degraded_without_ollama(self):
        """Ohne Ollama: Status 'degraded'."""
        from jarvis.healthcheck import health_status

        result = health_status(ollama_available=False)
        assert result["status"] == "degraded"
        assert "nicht erreichbar" in result["errors"][0]

    def test_degraded_with_errors(self):
        """Mit expliziten Fehlern: Status 'degraded'."""
        from jarvis.healthcheck import health_status

        result = health_status(
            ollama_available=True,
            errors=["Memory-Index korrupt"],
        )
        assert result["status"] == "degraded"
        assert len(result["errors"]) == 1

    def test_uptime_is_positive(self):
        """Uptime ist immer >= 0."""
        from jarvis.healthcheck import health_status

        result = health_status(ollama_available=True)
        assert result["uptime_seconds"] >= 0

    def test_timestamp_is_iso8601(self):
        """Timestamps sind ISO 8601 konform."""
        from jarvis.healthcheck import health_status

        result = health_status(ollama_available=True)
        # Einfacher ISO-Check
        assert "T" in result["timestamp"]
        assert "T" in result["started_at"]


# =============================================================================
# 2. Smoke-Test Skript
# =============================================================================


class TestSmokeTestScript:
    """Smoke-Test Skript existiert und ist importierbar."""

    def test_smoke_test_exists(self):
        """smoke_test.py existiert."""
        script = PROJECT_ROOT / "scripts" / "smoke_test.py"
        assert script.exists()

    def test_smoke_test_syntax(self):
        """smoke_test.py hat gültige Python-Syntax."""
        script = PROJECT_ROOT / "scripts" / "smoke_test.py"
        result = subprocess.run(
            [sys.executable, "-c", f"import ast; ast.parse(open('{script}').read()); print('OK')"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "OK" in result.stdout

    def test_smoke_test_has_main(self):
        """smoke_test.py hat eine main() Funktion."""
        script = PROJECT_ROOT / "scripts" / "smoke_test.py"
        content = script.read_text(encoding="utf-8")
        assert "def main()" in content
        assert "test_python_imports" in content
        assert "test_ollama" in content
        assert "test_gatekeeper" in content


# =============================================================================
# 3. Deployment-Dateien
# =============================================================================


class TestDeploymentFiles:
    """Deployment-Dateien sind vorhanden und korrekt formatiert."""

    def test_systemd_service_exists(self):
        """jarvis.service existiert."""
        service = PROJECT_ROOT / "deploy" / "jarvis.service"
        assert service.exists()

    def test_systemd_service_format(self):
        """jarvis.service hat korrektes systemd-Format."""
        service = PROJECT_ROOT / "deploy" / "jarvis.service"
        content = service.read_text(encoding="utf-8")

        # Pflicht-Sektionen
        assert "[Unit]" in content
        assert "[Service]" in content
        assert "[Install]" in content

        # Wichtige Direktiven
        assert "ExecStart=" in content
        assert "Restart=" in content
        assert "Type=" in content
        assert "WorkingDirectory=" in content

    def test_systemd_restart_policy(self):
        """Service startet bei Absturz automatisch neu."""
        service = PROJECT_ROOT / "deploy" / "jarvis.service"
        content = service.read_text(encoding="utf-8")
        assert "Restart=on-failure" in content
        assert "RestartSec=" in content

    def test_systemd_security_hardening(self):
        """Service hat Sicherheits-Härtung."""
        service = PROJECT_ROOT / "deploy" / "jarvis.service"
        content = service.read_text(encoding="utf-8")
        assert "NoNewPrivileges=true" in content
        assert "ProtectSystem=" in content

    def test_install_script_exists(self):
        """install.sh existiert."""
        script = PROJECT_ROOT / "install.sh"
        assert script.exists()

    @pytest.mark.skipif(sys.platform == "win32", reason="install.sh is not relevant on Windows")
    def test_install_script_syntax(self):
        """install.sh hat gültige Bash-Syntax."""
        script = PROJECT_ROOT / "install.sh"
        result = subprocess.run(
            ["bash", "-n", str(script)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Bash-Syntaxfehler: {result.stderr}"

    def test_install_script_modes(self):
        """install.sh unterstützt --minimal und --full Modi."""
        script = PROJECT_ROOT / "install.sh"
        content = script.read_text(encoding="utf-8")
        assert "--minimal" in content
        assert "--full" in content

    def test_requirements_txt_exists(self):
        """requirements.txt existiert."""
        req = PROJECT_ROOT / "requirements.txt"
        assert req.exists()

    def test_requirements_txt_parseable(self):
        """requirements.txt enthält gültige Paketnamen."""
        req = PROJECT_ROOT / "requirements.txt"
        content = req.read_text(encoding="utf-8")

        # Mindestens die Core-Deps
        assert "pydantic" in content
        assert "httpx" in content
        assert "structlog" in content
        assert "pyyaml" in content.lower() or "PyYAML" in content
        assert "cryptography" in content


# =============================================================================
# 4. Planner-Prompt Qualität
# =============================================================================


class TestPlannerPrompts:
    """Planner-Prompts enthalten die nötigen Schlüsselwörter."""

    def test_system_prompt_has_tools_placeholder(self):
        """System-Prompt enthält {tools_section} Placeholder."""
        from jarvis.core.planner import SYSTEM_PROMPT

        assert "{tools_section}" in SYSTEM_PROMPT

    def test_system_prompt_has_context_placeholder(self):
        """System-Prompt enthält {context_section} Placeholder."""
        from jarvis.core.planner import SYSTEM_PROMPT

        assert "{context_section}" in SYSTEM_PROMPT

    def test_system_prompt_mentions_json_format(self):
        """System-Prompt erklärt JSON-Output-Format."""
        from jarvis.core.planner import SYSTEM_PROMPT

        assert "json" in SYSTEM_PROMPT.lower()
        assert '"goal"' in SYSTEM_PROMPT
        assert '"steps"' in SYSTEM_PROMPT
        assert '"tool"' in SYSTEM_PROMPT

    def test_system_prompt_explains_direct_answer(self):
        """System-Prompt erklärt direkte Antwort-Option."""
        from jarvis.core.planner import SYSTEM_PROMPT

        lower = SYSTEM_PROMPT.lower()
        assert "direkt" in lower
        assert "kein json" in lower or "kein tool" in lower

    def test_system_prompt_is_german(self):
        """System-Prompt ist auf Deutsch."""
        from jarvis.core.planner import SYSTEM_PROMPT

        # Deutsche Schlüsselwörter
        assert "Deutsch" in SYSTEM_PROMPT or "deutsch" in SYSTEM_PROMPT

    def test_replan_prompt_has_placeholders(self):
        """Replan-Prompt enthält nötige Placeholders."""
        from jarvis.core.planner import REPLAN_PROMPT

        assert "{results_section}" in REPLAN_PROMPT
        assert "{original_goal}" in REPLAN_PROMPT

    def test_escalation_prompt_has_placeholders(self):
        """Escalation-Prompt enthält {tool} und {reason}."""
        from jarvis.core.planner import ESCALATION_PROMPT

        assert "{tool}" in ESCALATION_PROMPT
        assert "{reason}" in ESCALATION_PROMPT

    def test_planner_extracts_json_from_codeblock(self):
        """Planner kann JSON aus ```json ... ``` Blöcken extrahieren."""
        from unittest.mock import AsyncMock, MagicMock

        from jarvis.config import JarvisConfig
        from jarvis.core.planner import Planner

        config = JarvisConfig()
        planner = Planner(config, AsyncMock(), MagicMock())

        text = """Ich werde die Datei erstellen.

```json
{
  "goal": "Datei erstellen",
  "reasoning": "User will eine Datei",
  "steps": [
    {
      "tool": "write_file",
      "params": {"path": "/tmp/test.txt", "content": "Hallo"},
      "rationale": "Datei schreiben"
    }
  ],
  "confidence": 0.9
}
```"""

        plan = planner._extract_plan(text, "test")
        assert plan.has_actions
        assert len(plan.steps) == 1
        assert plan.steps[0].tool == "write_file"
        assert plan.confidence == 0.9

    def test_planner_handles_direct_text(self):
        """Planner erkennt direkte Antwort (kein JSON)."""
        from unittest.mock import AsyncMock, MagicMock

        from jarvis.config import JarvisConfig
        from jarvis.core.planner import Planner

        config = JarvisConfig()
        planner = Planner(config, AsyncMock(), MagicMock())

        text = "Berlin ist die Hauptstadt von Deutschland."
        plan = planner._extract_plan(text, "Hauptstadt?")
        assert not plan.has_actions
        assert plan.direct_response == text

    def test_planner_handles_malformed_json(self):
        """Planner überlebt kaputtes JSON gracefully."""
        from unittest.mock import AsyncMock, MagicMock

        from jarvis.config import JarvisConfig
        from jarvis.core.planner import Planner

        config = JarvisConfig()
        planner = Planner(config, AsyncMock(), MagicMock())

        text = '```json\n{"goal": "test", "steps": [KAPUTT]}\n```'
        plan = planner._extract_plan(text, "test")
        # Sollte als direkte Antwort interpretiert werden, nicht crashen
        assert plan is not None


# =============================================================================
# 5. Projekt-Integrität
# =============================================================================


class TestProjectIntegrity:
    """Gesamtprojekt-Struktur ist vollständig."""

    def test_all_init_files_exist(self):
        """Alle Python-Packages haben __init__.py."""
        src = PROJECT_ROOT / "src" / "jarvis"
        packages = [
            src,
            src / "channels",
            src / "core",
            src / "gateway",
            src / "mcp",
            src / "memory",
            src / "security",
            src / "utils",
        ]
        for pkg in packages:
            init = pkg / "__init__.py"
            assert init.exists(), f"Fehlt: {init}"

    def test_version_importable(self):
        """Jarvis-Version ist importierbar."""
        from jarvis import __version__

        assert __version__
        # Semantic Versioning
        parts = __version__.split(".")
        assert len(parts) >= 2

    def test_main_entry_point(self):
        """python -m jarvis --version funktioniert."""
        result = subprocess.run(
            [sys.executable, "-m", "jarvis", "--version"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT / "src"),
            env={**__import__("os").environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
        )
        assert result.returncode == 0
        assert "Cognithor" in result.stdout or "cognithor" in result.stdout.lower()

    def test_no_circular_imports(self):
        """Keine zirkulären Imports in den Kern-Modulen."""
        modules = [
            "jarvis.config",
            "jarvis.models",
            "jarvis.utils.logging",
            "jarvis.core.planner",
            "jarvis.core.gatekeeper",
            "jarvis.core.executor",
            "jarvis.memory.manager",
            "jarvis.security.audit",
            "jarvis.healthcheck",
        ]
        for mod in modules:
            result = subprocess.run(
                [sys.executable, "-c", f"import {mod}; print('OK')"],
                capture_output=True,
                text=True,
                env={**__import__("os").environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
            )
            assert result.returncode == 0, f"Import-Fehler in {mod}: {result.stderr}"
