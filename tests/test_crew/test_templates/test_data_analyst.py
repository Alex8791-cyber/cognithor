"""Task 41 — data-analyst template E2E test."""

from pathlib import Path

from cognithor.crew.cli.init_cmd import run_init


def test_data_analyst_template_ships_all_required_files(tmp_path: Path):
    project = tmp_path / "da"
    rc = run_init(name="da", template="data-analyst", directory=project, lang="de")
    assert rc == 0

    # 8 user-editable files from the template tree
    assert (project / "pyproject.toml").exists()
    assert (project / "src" / "da" / "crew.py").exists()
    assert (project / "src" / "da" / "main.py").exists()
    assert (project / "src" / "da" / "__init__.py").exists()
    assert (project / "tests" / "test_crew.py").exists()
    assert (project / "README.md").exists()
    assert (project / ".env.example").exists()
    assert (project / "config" / "agents.yaml").exists()
    assert (project / "config" / "tasks.yaml").exists()
    # Auto-injected gitignore
    assert (project / ".gitignore").exists()


def test_data_analyst_template_wires_python_sandbox_tool(tmp_path: Path):
    """Rendered crew.py must expose ``tools=["python_sandbox"]`` on the analyst."""
    project = tmp_path / "da"
    rc = run_init(name="da", template="data-analyst", directory=project, lang="de")
    assert rc == 0

    crew_file = (project / "src" / "da" / "crew.py").read_text(encoding="utf-8")
    assert 'tools=["python_sandbox"]' in crew_file, (
        "Analyst agent must be wired with the python_sandbox tool"
    )
