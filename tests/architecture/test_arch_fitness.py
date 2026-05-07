"""Architecture fitness tests — Sprint 2.3.

The heavy lifting is done by ``import-linter`` (config in
``.importlinter``). This file provides a Python-level smoke test that
runs lint-imports as a subprocess and asserts a clean exit, plus a
handful of project-specific invariants that import-linter does not
express directly.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_import_linter_passes() -> None:
    """Run `lint-imports` and assert clean exit.

    If this fails locally with a `lint-imports: command not found`
    error, install the quality extras: `pip install -e ".[dev,quality]"`.
    """
    try:
        importlib_spec = subprocess.run(
            [sys.executable, "-c", "import importlinter"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        pytest.skip("could not probe for import-linter")
    if importlib_spec.returncode != 0:
        pytest.skip("import-linter not installed (install via [quality] extra)")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "importlinter.cli", "lint"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError:
        pytest.skip("import-linter not installed (install via [quality] extra)")
    if result.returncode != 0:
        pytest.fail(
            "import-linter contracts failed:\n"
            + result.stdout
            + "\n----- stderr -----\n"
            + result.stderr
        )


def test_no_circular_imports_in_critical_paths() -> None:
    """Hard sanity: importing the core public surface must not loop.

    A circular-import bug between cognithor.core and cognithor.video
    would surface here as an ImportError.
    """
    # Import the public surface in a single fresh subprocess so module
    # caches do not mask a real circular dep.
    code = (
        "import cognithor.core.workflow; "
        "import cognithor.core.vlm_router; "
        "import cognithor.core.vllm_backend; "
        "import cognithor.core.vllm_orchestrator; "
        "import cognithor.video.routing; "
        "import cognithor.audit; "
        "import cognithor.security; "
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        "Critical-path import loop detected:\n"
        + result.stdout
        + "\n----- stderr -----\n"
        + result.stderr
    )
    assert "ok" in result.stdout


def test_security_modules_have_no_outbound_dependencies_into_packs() -> None:
    """Security must be the bottom layer; packs cannot pollute it.

    Grep-based check (cheap to run, complementary to import-linter).
    If any cognithor.security.* module imports cognithor.packs, this
    fails.
    """
    sec_dir = REPO_ROOT / "src" / "cognithor" / "security"
    offenders: list[str] = []
    for py in sec_dir.rglob("*.py"):
        text = py.read_text(encoding="utf-8", errors="ignore")
        if "from cognithor.packs" in text or "import cognithor.packs" in text:
            offenders.append(str(py.relative_to(REPO_ROOT)))
    assert not offenders, (
        f"Security layer imports from packs: {offenders}. "
        f"Security must not depend on user-loadable code."
    )
