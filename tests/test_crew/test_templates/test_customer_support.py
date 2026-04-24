"""Task 40 — customer-support template E2E test."""

import ast
from pathlib import Path

from cognithor.crew.cli.init_cmd import run_init


def test_customer_support_template_has_three_agents(tmp_path: Path):
    """AST-parse the rendered crew.py and assert exactly 3 @agent methods."""
    project = tmp_path / "cs"
    rc = run_init(name="cs", template="customer-support", directory=project, lang="de")
    assert rc == 0

    crew_path = project / "src" / "cs" / "crew.py"
    tree = ast.parse(crew_path.read_text(encoding="utf-8"))

    agent_methods: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "CustomerSupportCrew":
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    for deco in item.decorator_list:
                        deco_name = deco.id if isinstance(deco, ast.Name) else None
                        if deco_name == "agent":
                            agent_methods.append(item.name)

    assert len(agent_methods) == 3, (
        f"Expected 3 @agent methods, got {len(agent_methods)}: {agent_methods}"
    )
    assert set(agent_methods) == {"intake", "classifier", "response_writer"}


def test_customer_support_template_ships_all_required_files(tmp_path: Path):
    project = tmp_path / "cs"
    rc = run_init(name="cs", template="customer-support", directory=project, lang="de")
    assert rc == 0

    # 8 user-editable files from the template tree
    assert (project / "pyproject.toml").exists()
    assert (project / "src" / "cs" / "crew.py").exists()
    assert (project / "src" / "cs" / "main.py").exists()
    assert (project / "src" / "cs" / "__init__.py").exists()
    assert (project / "tests" / "test_crew.py").exists()
    assert (project / "README.md").exists()
    assert (project / ".env.example").exists()
    assert (project / "config" / "agents.yaml").exists()
    assert (project / "config" / "tasks.yaml").exists()
    # Auto-injected gitignore
    assert (project / ".gitignore").exists()
