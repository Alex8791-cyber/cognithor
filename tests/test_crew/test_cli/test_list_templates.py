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


def test_list_templates_cli_lists_all_five():
    """After Tasks 39-43, all 5 first-party templates are discoverable."""
    from cognithor.crew.cli.list_templates_cmd import list_templates

    names = {t.name for t in list_templates()}
    assert names == {
        "research",
        "customer-support",
        "data-analyst",
        "content",
        "versicherungs-vergleich",
    }


def test_list_templates_respects_order_field():
    """Templates sort by `order` ascending. Tasks 39-43 assign 1-5 in this order."""
    from cognithor.crew.cli.list_templates_cmd import list_templates

    names = [t.name for t in list_templates()]
    assert names == [
        "research",
        "customer-support",
        "data-analyst",
        "content",
        "versicherungs-vergleich",
    ]


def test_list_templates_via_cli_subprocess():
    """Full CLI invocation — verifies spec §3.2 flag syntax works end-to-end."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "cognithor", "init", "--list-templates"],
        capture_output=True,
        text=True,
        check=True,
    )
    expected = [
        "research",
        "customer-support",
        "data-analyst",
        "content",
        "versicherungs-vergleich",
    ]
    positions = [result.stdout.find(n) for n in expected]
    assert all(p >= 0 for p in positions), f"Template missing from CLI output: {result.stdout}"
    assert positions == sorted(positions), f"Templates listed out of order: {result.stdout}"
