"""Task 36 — `cognithor init` core handler tests."""

from pathlib import Path

import pytest

from cognithor.crew.cli.init_cmd import InitCommandError, run_init


@pytest.fixture
def mock_templates(tmp_path: Path, monkeypatch):
    """Plant a minimal mock template at src/{{project_name}}/main.py.jinja."""
    tpl_root = tmp_path / "templates"
    research = tpl_root / "research"
    research.mkdir(parents=True)
    (research / "template.yaml").write_text(
        "name: research\ndescription_de: Mock\ndescription_en: Mock\n"
    )
    (research / "README.md.jinja").write_text("# {{ project_name }}")
    src_dir = research / "src" / "{{ project_name }}"
    src_dir.mkdir(parents=True)
    (src_dir / "__init__.py").write_text("")
    (src_dir / "main.py.jinja").write_text("PROJECT = '{{ project_name }}'")

    monkeypatch.setattr("cognithor.crew.cli.list_templates_cmd.TEMPLATES_ROOT", tpl_root)
    monkeypatch.setattr("cognithor.crew.cli.init_cmd.TEMPLATES_ROOT", tpl_root)
    return tpl_root


def test_creates_project_from_template(tmp_path: Path, mock_templates):
    project_dir = tmp_path / "my_project"
    rc = run_init(name="My Project", template="research", directory=project_dir, lang="en")
    assert rc == 0
    assert (project_dir / "README.md").read_text() == "# my_project"
    # R4-C2: main.py lives inside the package
    assert (project_dir / "src" / "my_project" / "main.py").read_text() == (
        "PROJECT = 'my_project'"
    )
    assert (project_dir / "src" / "my_project" / "__init__.py").exists()


def test_refuses_nonempty_directory(tmp_path: Path, mock_templates):
    project_dir = tmp_path / "existing"
    project_dir.mkdir()
    (project_dir / "file.txt").write_text("hello")
    with pytest.raises(InitCommandError):
        run_init(name="existing", template="research", directory=project_dir, lang="en")


def test_unknown_template_raises(tmp_path: Path, mock_templates):
    with pytest.raises(InitCommandError, match="unknown"):
        run_init(
            name="x",
            template="does_not_exist",
            directory=tmp_path / "x",
            lang="en",
        )


def test_init_force_overwrites_existing_dir(tmp_path: Path, mock_templates, capsys):
    """R4-I5: `--force` removes an existing non-empty target."""
    project_dir = tmp_path / "my_project"
    project_dir.mkdir()
    (project_dir / "stale.txt").write_text("pre-existing junk")

    rc = run_init(
        name="My Project",
        template="research",
        directory=project_dir,
        lang="en",
        force=True,
    )
    assert rc == 0
    assert not (project_dir / "stale.txt").exists()
    assert (project_dir / "README.md").read_text() == "# my_project"
    captured = capsys.readouterr()
    assert "--force" in captured.out
    assert "removing existing" in captured.out
