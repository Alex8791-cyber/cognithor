"""Tests for document creation and template MCP tools (read_xlsx, document_create,
typst_render, template_list, template_render)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from jarvis.documents.templates import DocumentTemplate, TemplateManager
from jarvis.mcp.media import MediaPipeline

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    d = tmp_path / "workspace"
    d.mkdir()
    return d


@pytest.fixture()
def pipeline(workspace: Path) -> MediaPipeline:
    return MediaPipeline(workspace_dir=workspace)


# ── read_xlsx ─────────────────────────────────────────────────────


class TestReadXlsx:
    @pytest.mark.asyncio
    async def test_file_not_found(self, pipeline: MediaPipeline) -> None:
        result = await pipeline.read_xlsx("/nonexistent/file.xlsx")
        assert not result.success

    @pytest.mark.asyncio
    async def test_read_valid_xlsx(self, pipeline: MediaPipeline, tmp_path: Path) -> None:
        """Create a real xlsx with openpyxl, read back via pipeline, verify markdown table."""
        openpyxl = pytest.importorskip("openpyxl")

        xlsx_path = tmp_path / "test_data.xlsx"
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Sheet1"
        ws.append(["Name", "Age", "City"])
        ws.append(["Alice", 30, "Berlin"])
        ws.append(["Bob", 25, "Munich"])
        wb.save(xlsx_path)
        wb.close()

        result = await pipeline.read_xlsx(str(xlsx_path))
        assert result.success
        assert "Name" in result.text
        assert "Alice" in result.text
        assert "Bob" in result.text
        # Markdown table structure
        assert "|" in result.text
        assert "---" in result.text

    @pytest.mark.asyncio
    async def test_wrong_extension(self, pipeline: MediaPipeline, tmp_path: Path) -> None:
        bad = tmp_path / "data.csv"
        bad.write_text("a,b\n1,2")
        result = await pipeline.read_xlsx(str(bad))
        assert not result.success


# ── document_create ───────────────────────────────────────────────


class TestDocumentCreate:
    @pytest.mark.asyncio
    async def test_create_docx(self, pipeline: MediaPipeline) -> None:
        """JSON structure to DOCX, verify file is created."""
        pytest.importorskip("docx")

        structure = json.dumps(
            {
                "title": "Test Report",
                "sections": [
                    {"heading": "Introduction", "content": "Hello World."},
                    {"heading": "Data", "table": {"headers": ["A", "B"], "rows": [["1", "2"]]}},
                ],
            }
        )
        result = await pipeline.create_document(structure, fmt="docx", filename="test_report")
        assert result.success
        assert result.output_path
        from pathlib import Path as P

        assert P(result.output_path).exists()
        assert P(result.output_path).suffix == ".docx"

    @pytest.mark.asyncio
    async def test_invalid_json(self, pipeline: MediaPipeline) -> None:
        result = await pipeline.create_document("not json at all", fmt="docx")
        assert not result.success
        assert "JSON" in result.error

    @pytest.mark.asyncio
    async def test_unsupported_format(self, pipeline: MediaPipeline) -> None:
        result = await pipeline.create_document('{"title": "x"}', fmt="rtf")
        assert not result.success


# ── typst_render ──────────────────────────────────────────────────


class TestTypstRender:
    @pytest.mark.asyncio
    async def test_render_simple_pdf(self, pipeline: MediaPipeline) -> None:
        """Simple Typst source compiled to PDF, verify file created."""
        typst = pytest.importorskip("typst")  # noqa: F841

        source = "#set page(width: 10cm, height: 5cm)\n= Hello\nThis is a test."
        result = await pipeline.typst_render(source, filename="test_typst")
        assert result.success
        assert result.output_path
        from pathlib import Path as P

        assert P(result.output_path).exists()
        assert P(result.output_path).suffix == ".pdf"

    @pytest.mark.asyncio
    async def test_empty_source(self, pipeline: MediaPipeline) -> None:
        result = await pipeline.typst_render("", filename="empty")
        assert not result.success


# ── template_list ─────────────────────────────────────────────────


class TestTemplateList:
    def test_no_templates_dir(self, tmp_path: Path) -> None:
        """TemplateManager with non-existent dir returns empty list."""
        tm = TemplateManager(templates_dir=tmp_path / "nonexistent")
        assert tm.list_templates() == []

    def test_loads_typ_files(self, tmp_path: Path) -> None:
        """Templates loaded from .typ files with frontmatter."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()

        typ_content = (
            "// template: Brief\n"
            "// description: A formal letter\n"
            "// category: correspondence\n"
            "// variables: name, address\n"
            "\n"
            "Dear {{name}},\n"
            "Your address is {{address}}.\n"
        )
        (templates_dir / "brief.typ").write_text(typ_content, encoding="utf-8")

        tm = TemplateManager(templates_dir=templates_dir)
        templates = tm.list_templates()
        assert len(templates) == 1
        assert templates[0].slug == "brief"
        assert "name" in templates[0].variables
        assert "address" in templates[0].variables


# ── template_render ───────────────────────────────────────────────


class TestTemplateRender:
    def test_render_replaces_placeholders(self, tmp_path: Path) -> None:
        """Render a brief template and verify placeholders are filled."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()

        typ_content = (
            "// template: Brief\n"
            "// description: A formal letter\n"
            "// variables: name, city\n"
            "\n"
            "Dear {{name}},\n"
            "Welcome to {{city}}.\n"
        )
        (templates_dir / "brief.typ").write_text(typ_content, encoding="utf-8")

        tm = TemplateManager(templates_dir=templates_dir)
        rendered = tm.render_template("brief", {"name": "Alice", "city": "Berlin"})
        assert "Dear Alice," in rendered
        assert "Welcome to Berlin." in rendered
        assert "{{name}}" not in rendered
        assert "{{city}}" not in rendered

    def test_render_unknown_slug_raises(self, tmp_path: Path) -> None:
        tm = TemplateManager(templates_dir=tmp_path)
        with pytest.raises(KeyError, match="not found"):
            tm.render_template("nonexistent", {})

    @pytest.mark.asyncio
    async def test_render_to_pdf(self, tmp_path: Path) -> None:
        """Full pipeline: template render -> typst compile -> PDF."""
        typst = pytest.importorskip("typst")  # noqa: F841

        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()

        typ_content = (
            "// template: Brief\n"
            "// description: A test letter\n"
            "// variables: name\n"
            "\n"
            "#set page(width: 10cm, height: 5cm)\n"
            "Hello {{name}}.\n"
        )
        (templates_dir / "brief.typ").write_text(typ_content, encoding="utf-8")

        tm = TemplateManager(templates_dir=templates_dir)
        rendered = tm.render_template("brief", {"name": "World"})

        workspace = tmp_path / "ws"
        workspace.mkdir()
        pipeline = MediaPipeline(workspace_dir=workspace)
        result = await pipeline.typst_render(rendered, filename="brief_test")
        assert result.success
        assert result.output_path
        from pathlib import Path as P

        assert P(result.output_path).exists()
