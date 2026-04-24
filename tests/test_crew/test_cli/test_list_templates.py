"""Task 35 — template metadata discovery tests."""

from pathlib import Path
from unittest.mock import patch

from cognithor.crew.cli.list_templates_cmd import TemplateMeta, list_templates


def test_discovers_template_from_template_yaml(tmp_path: Path):
    t_dir = tmp_path / "research"
    t_dir.mkdir()
    (t_dir / "template.yaml").write_text(
        "name: research\n"
        "description_de: Zwei-Agenten-Research-Crew\n"
        "description_en: Two-agent research crew\n"
        "required_models: ['ollama/qwen3:8b']\n"
        "tags: [demo, quickstart]\n"
    )
    with patch("cognithor.crew.cli.list_templates_cmd.TEMPLATES_ROOT", tmp_path):
        templates = list_templates()

    assert len(templates) == 1
    t = templates[0]
    assert isinstance(t, TemplateMeta)
    assert t.name == "research"
    assert t.description_de.startswith("Zwei")


def test_skips_dirs_without_template_yaml(tmp_path: Path):
    (tmp_path / "broken").mkdir()
    with patch("cognithor.crew.cli.list_templates_cmd.TEMPLATES_ROOT", tmp_path):
        templates = list_templates()
    assert templates == []
