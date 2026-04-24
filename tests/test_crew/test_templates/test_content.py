"""Task 42 — content template E2E test."""

from pathlib import Path

from cognithor.crew.cli.init_cmd import run_init


def test_content_template_ships_all_required_files(tmp_path: Path):
    project = tmp_path / "cn"
    rc = run_init(name="cn", template="content", directory=project, lang="de")
    assert rc == 0

    # 8 user-editable files from the template tree
    assert (project / "pyproject.toml").exists()
    assert (project / "src" / "cn" / "crew.py").exists()
    assert (project / "src" / "cn" / "main.py").exists()
    assert (project / "src" / "cn" / "__init__.py").exists()
    assert (project / "tests" / "test_crew.py").exists()
    assert (project / "README.md").exists()
    assert (project / ".env.example").exists()
    assert (project / "config" / "agents.yaml").exists()
    assert (project / "config" / "tasks.yaml").exists()
    # Auto-injected gitignore
    assert (project / ".gitignore").exists()


def test_content_template_uses_hierarchical_process_with_manager_llm(tmp_path: Path):
    """Rendered crew.py must use HIERARCHICAL process + manager_llm."""
    project = tmp_path / "cn"
    rc = run_init(name="cn", template="content", directory=project, lang="de")
    assert rc == 0

    crew_file = (project / "src" / "cn" / "crew.py").read_text(encoding="utf-8")
    assert "HIERARCHICAL" in crew_file, "Content crew must use CrewProcess.HIERARCHICAL"
    assert "manager_llm" in crew_file, "Content crew must configure manager_llm"
