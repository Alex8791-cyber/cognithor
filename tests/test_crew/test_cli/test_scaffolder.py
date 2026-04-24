"""Task 34 — cli.scaffolder tests (sanitize + render + security + i18n)."""

from pathlib import Path

import pytest

from cognithor.crew.cli.scaffolder import render_tree, sanitize_project_name


class TestSanitize:
    def test_spaces_to_underscore(self):
        assert sanitize_project_name("My Research Crew") == "my_research_crew"

    def test_hyphens_to_underscore(self):
        assert sanitize_project_name("my-crew") == "my_crew"

    def test_leading_digit_prefixed(self):
        assert sanitize_project_name("123abc") == "project_123abc"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            sanitize_project_name("")


class TestRenderTree:
    def test_renders_jinja_templates(self, tmp_path: Path):
        src = tmp_path / "src_templates"
        src.mkdir()
        (src / "hello.py.jinja").write_text("print('{{ project_name }}')")
        (src / "README.md.jinja").write_text("# {{ project_name | title }}")
        (src / "plain.txt").write_text("no substitution")
        dest = tmp_path / "out"

        render_tree(src, dest, context={"project_name": "my_crew"})

        assert (dest / "hello.py").read_text() == "print('my_crew')"
        assert (dest / "README.md").read_text() == "# My_crew"
        assert (dest / "plain.txt").read_text() == "no substitution"

    def test_refuses_non_empty_dest(self, tmp_path: Path):
        (tmp_path / "out").mkdir()
        (tmp_path / "out" / "existing.txt").write_text("already here")
        with pytest.raises(FileExistsError):
            render_tree(tmp_path / "src", tmp_path / "out", context={})


def test_scaffolder_blocks_path_traversal_in_filename(tmp_path):
    """Regression for NC3: ``{{ '../../etc/passwd' }}`` in filename MUST raise."""
    src = tmp_path / "tmpl"
    (src / "subdir").mkdir(parents=True)
    traversal = src / "subdir" / "{{ payload }}.jinja"
    traversal.write_text("pwned", encoding="utf-8")

    dest = tmp_path / "out"
    with pytest.raises(ValueError, match="traversal|Forbidden"):
        render_tree(src, dest, context={"payload": "../../etc/passwd"})


def test_scaffolder_blocks_backslash_traversal_on_windows(tmp_path):
    """Backslash-based traversal payloads are also rejected."""
    src = tmp_path / "tmpl"
    src.mkdir()
    traversal = src / "{{ payload }}.jinja"
    traversal.write_text("pwned", encoding="utf-8")

    dest = tmp_path / "out"
    with pytest.raises(ValueError, match="traversal|Forbidden"):
        render_tree(src, dest, context={"payload": r"..\..\secrets"})


def test_sanitize_project_name_rejects_CON_on_all_platforms():
    """Windows reserved device names rejected even on Linux (NI4)."""
    for reserved in ("CON", "con", "nul", "COM1", "lpt9", "prn", "aux"):
        with pytest.raises(ValueError, match="reserved Windows device name"):
            sanitize_project_name(reserved)


def test_scaffolder_renders_language_specific_readme_de(tmp_path):
    """R4-C1: ``lang='de'`` renders the ``.de`` variant to ``README.md``."""
    src = tmp_path / "tmpl"
    src.mkdir()
    (src / "README.md.jinja.de").write_text("# {{ project_name }} (DE)")
    (src / "README.md.jinja.en").write_text("# {{ project_name }} (EN)")
    dest = tmp_path / "out"

    render_tree(src, dest, context={"project_name": "demo", "lang": "de"})

    assert (dest / "README.md").read_text() == "# demo (DE)"
    assert not (dest / "README.md.jinja.en").exists()
    assert not (dest / "README.md.jinja.de").exists()


def test_scaffolder_renders_language_specific_readme_en(tmp_path):
    """Same contract as DE but ``lang='en'``."""
    src = tmp_path / "tmpl"
    src.mkdir()
    (src / "README.md.jinja.de").write_text("# {{ project_name }} (DE)")
    (src / "README.md.jinja.en").write_text("# {{ project_name }} (EN)")
    dest = tmp_path / "out"

    render_tree(src, dest, context={"project_name": "demo", "lang": "en"})

    assert (dest / "README.md").read_text() == "# demo (EN)"
    assert not (dest / "README.md.jinja.en").exists()
    assert not (dest / "README.md.jinja.de").exists()


def test_scaffolder_falls_back_to_en_when_requested_lang_missing(tmp_path):
    """R5 regression: ``lang='zh'`` with only .de/.en variants falls back to EN."""
    src = tmp_path / "tmpl"
    src.mkdir()
    (src / "README.md.jinja.de").write_text("# {{ project_name }} (DE)")
    (src / "README.md.jinja.en").write_text("# {{ project_name }} (EN)")
    dest = tmp_path / "out"

    render_tree(src, dest, context={"project_name": "demo", "lang": "zh"})

    assert (dest / "README.md").read_text() == "# demo (EN)"
    assert not (dest / "README.md.jinja.en").exists()
    assert not (dest / "README.md.jinja.de").exists()


def test_scaffolder_falls_back_to_first_sorted_when_no_en_variant(tmp_path):
    """No requested lang, no 'en' — pick first alphabetical (here .de)."""
    src = tmp_path / "tmpl"
    src.mkdir()
    (src / "README.md.jinja.de").write_text("# {{ project_name }} (DE)")
    (src / "README.md.jinja.zh").write_text("# {{ project_name }} (ZH)")
    dest = tmp_path / "out"

    render_tree(src, dest, context={"project_name": "demo", "lang": "ar"})

    assert (dest / "README.md").read_text() == "# demo (DE)"


def test_scaffolder_auto_injects_gitignore(tmp_path):
    """Scaffolder must write a default .gitignore when the template doesn't
    ship one — prevents accidental .env / __pycache__ commits."""
    from cognithor.crew.cli.scaffolder import render_tree

    src = tmp_path / "tmpl"
    src.mkdir()
    (src / "foo.txt").write_text("bar")
    dest = tmp_path / "out"

    render_tree(src, dest, context={})

    gi = dest / ".gitignore"
    assert gi.exists()
    content = gi.read_text()
    assert "__pycache__" in content
    assert ".env" in content
