"""Task 39 — research template E2E test."""

from pathlib import Path

from cognithor.crew.cli.init_cmd import run_init


def test_research_template_renders_and_smoke_tests_pass(tmp_path: Path):
    project = tmp_path / "rc"
    rc = run_init(name="rc", template="research", directory=project, lang="de")
    assert rc == 0

    # Required artifacts — 8 user-editable files from the template tree
    assert (project / "pyproject.toml").exists()
    assert (project / "src" / "rc" / "crew.py").exists()
    assert (project / "src" / "rc" / "main.py").exists()
    assert (project / "src" / "rc" / "__init__.py").exists()
    assert (project / "tests" / "test_crew.py").exists()
    assert (project / "README.md").exists()
    assert (project / ".env.example").exists()
    assert (project / "config" / "agents.yaml").exists()
    assert (project / "config" / "tasks.yaml").exists()

    # Plus the auto-injected .gitignore (scaffolder writes this regardless of
    # which template was picked; not part of the template tree itself).
    assert (project / ".gitignore").exists(), (
        "Scaffolder must auto-inject .gitignore to prevent accidental secret commits"
    )
