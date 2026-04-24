"""Edge-case coverage for the scaffolded-project-internal `cognithor run` helper.

`run_cmd.run_project_crew` is the entry point that every scaffolded template
installs as its own `cognithor run`. These tests cover the error paths so the
generated-project scripts fail loudly instead of silently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cognithor.crew.cli.run_cmd import run_project_crew

if TYPE_CHECKING:
    from pathlib import Path


def test_run_project_crew_errors_when_no_src_dir(tmp_path: Path, capsys) -> None:
    rc = run_project_crew(tmp_path)
    captured = capsys.readouterr()
    assert rc == 2
    assert "No src/" in captured.err


def test_run_project_crew_errors_when_src_has_no_package(tmp_path: Path, capsys) -> None:
    (tmp_path / "src").mkdir()
    rc = run_project_crew(tmp_path)
    captured = capsys.readouterr()
    assert rc == 2
    assert "No Python package" in captured.err


def test_run_project_crew_errors_on_missing_main_module(tmp_path: Path, capsys) -> None:
    pkg = tmp_path / "src" / "demo_pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    # intentionally no main.py
    rc = run_project_crew(tmp_path)
    captured = capsys.readouterr()
    assert rc == 2
    assert "cannot import" in captured.err


def test_run_project_crew_errors_when_build_crew_missing(tmp_path: Path, capsys) -> None:
    pkg = tmp_path / "src" / "demo_pkg2"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "main.py").write_text("# no build_crew defined\n")
    rc = run_project_crew(tmp_path)
    captured = capsys.readouterr()
    assert rc == 2
    assert "build_crew" in captured.err


def test_run_project_crew_defaults_to_cwd(monkeypatch, tmp_path: Path, capsys) -> None:
    """Passing None falls back to Path.cwd()."""
    monkeypatch.chdir(tmp_path)
    rc = run_project_crew(None)
    captured = capsys.readouterr()
    assert rc == 2
    assert "No src/" in captured.err


@pytest.fixture(autouse=True)
def _clear_sys_path(monkeypatch):
    """Avoid polluting sys.path between tests that insert fake src dirs."""
    import sys

    snapshot = list(sys.path)
    yield
    sys.path[:] = snapshot
    # Drop any stray imported demo_pkg* modules so repeated tests re-import cleanly.
    for name in list(sys.modules):
        if name.startswith(("demo_pkg", "demo_pkg2")):
            sys.modules.pop(name, None)
