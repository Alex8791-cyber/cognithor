"""Tests for read_pdf, read_ppt, read_docx MCP tools."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

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


# ── read_pdf ───────────────────────────────────────────────────────


class TestReadPdf:
    @pytest.mark.asyncio
    async def test_file_not_found(self, pipeline: MediaPipeline) -> None:
        result = await pipeline.read_pdf("/nonexistent/file.pdf")
        assert not result.success
        assert "nicht gefunden" in result.error

    @pytest.mark.asyncio
    async def test_extract_text(self, pipeline: MediaPipeline, workspace: Path) -> None:
        """PDF text extraction via mocked fitz."""
        pdf_path = workspace / "test.pdf"
        pdf_path.write_bytes(b"dummy")

        mock_page = MagicMock()
        mock_page.get_text.return_value = "Seite 1 Inhalt"
        mock_page.get_images.return_value = []

        mock_doc = MagicMock()
        mock_doc.__len__ = lambda s: 1
        mock_doc.__getitem__ = lambda s, idx: mock_page
        mock_doc.metadata = {"title": "Test", "author": "Author"}
        mock_doc.close = MagicMock()

        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            result = await pipeline.read_pdf(str(pdf_path))

        assert result.success
        assert "Seite 1 Inhalt" in result.text
        assert result.metadata["page_count"] == 1

    @pytest.mark.asyncio
    async def test_page_range(self, pipeline: MediaPipeline, workspace: Path) -> None:
        """Page range parameter filters pages correctly."""
        pdf_path = workspace / "multi.pdf"
        pdf_path.write_bytes(b"dummy")

        pages = []
        for i in range(5):
            p = MagicMock()
            p.get_text.return_value = f"Page {i + 1}"
            p.get_images.return_value = []
            pages.append(p)

        mock_doc = MagicMock()
        mock_doc.__len__ = lambda s: 5
        mock_doc.__getitem__ = lambda s, idx: pages[idx]
        mock_doc.metadata = {}
        mock_doc.close = MagicMock()

        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            result = await pipeline.read_pdf(str(pdf_path), pages="2-3")

        assert result.success
        assert "Page 2" in result.text
        assert "Page 3" in result.text
        assert "Page 1" not in result.text
        assert "Page 4" not in result.text


# ── read_ppt ───────────────────────────────────────────────────────


class TestReadPpt:
    @pytest.mark.asyncio
    async def test_file_not_found(self, pipeline: MediaPipeline) -> None:
        result = await pipeline.read_ppt("/nonexistent/file.pptx")
        assert not result.success
        assert "nicht gefunden" in result.error

    @pytest.mark.asyncio
    async def test_extract_slides(self, pipeline: MediaPipeline, workspace: Path) -> None:
        """PPTX slide extraction via mocked pptx."""
        pptx_path = workspace / "test.pptx"
        pptx_path.write_bytes(b"dummy")

        mock_title = MagicMock()
        mock_title.text = "Slide Title"

        mock_run = MagicMock()
        mock_run.text = "Content text"
        mock_run.font = MagicMock()
        mock_run.font.bold = False
        mock_run.font.italic = False

        mock_para = MagicMock()
        mock_para.runs = [mock_run]

        mock_tf = MagicMock()
        mock_tf.paragraphs = [mock_para]

        mock_shape = MagicMock()
        mock_shape.text_frame = mock_tf
        mock_shape.shape_type = 1  # Not PICTURE

        mock_slide = MagicMock()
        mock_slide.shapes = MagicMock()
        mock_slide.shapes.title = mock_title
        mock_slide.shapes.__iter__ = lambda s: iter([mock_shape])
        mock_slide.has_notes_slide = False

        mock_prs = MagicMock()
        mock_prs.slides = [mock_slide]

        mock_pptx = MagicMock()
        mock_pptx.Presentation.return_value = mock_prs

        mock_mso = MagicMock()
        mock_mso.MSO_SHAPE_TYPE.PICTURE = 13

        with patch.dict("sys.modules", {"pptx": mock_pptx, "pptx.enum.shapes": mock_mso}):
            result = await pipeline.read_ppt(str(pptx_path))

        assert result.success
        assert "Content text" in result.text
        assert result.metadata["slide_count"] == 1


# ── read_docx ──────────────────────────────────────────────────────


class TestReadDocx:
    @pytest.mark.asyncio
    async def test_file_not_found(self, pipeline: MediaPipeline) -> None:
        result = await pipeline.read_docx("/nonexistent/file.docx")
        assert not result.success
        assert "nicht gefunden" in result.error

    @pytest.mark.asyncio
    async def test_extract_paragraphs(self, pipeline: MediaPipeline, workspace: Path) -> None:
        """DOCX paragraph extraction via mocked docx."""
        docx_path = workspace / "test.docx"
        docx_path.write_bytes(b"dummy")

        mock_run = MagicMock()
        mock_run.text = "Hello World"
        mock_run.bold = False
        mock_run.italic = False

        mock_para = MagicMock()
        mock_para.text = "Hello World"
        mock_para.runs = [mock_run]
        mock_para.style = MagicMock()
        mock_para.style.name = "Normal"

        mock_props = MagicMock()
        mock_props.author = "Test Author"
        mock_props.title = "Test Doc"
        mock_props.created = None
        mock_props.modified = None

        mock_doc = MagicMock()
        mock_doc.paragraphs = [mock_para]
        mock_doc.tables = []
        mock_doc.core_properties = mock_props
        mock_doc.part = MagicMock()
        mock_doc.part.rels = {}

        mock_docx_mod = MagicMock()
        mock_docx_mod.Document.return_value = mock_doc

        with patch.dict("sys.modules", {"docx": mock_docx_mod, "docx.opc.constants": MagicMock()}):
            result = await pipeline.read_docx(str(docx_path))

        assert result.success
        assert "Hello World" in result.text
        assert result.metadata["author"] == "Test Author"


# ── Tool registration ──────────────────────────────────────────────


class TestToolSchemas:
    def test_schemas_include_new_tools(self) -> None:
        from jarvis.mcp.media import MEDIA_TOOL_SCHEMAS

        assert "read_pdf" in MEDIA_TOOL_SCHEMAS
        assert "read_ppt" in MEDIA_TOOL_SCHEMAS
        assert "read_docx" in MEDIA_TOOL_SCHEMAS

    def test_schema_count(self) -> None:
        from jarvis.mcp.media import MEDIA_TOOL_SCHEMAS

        # 8 original + 3 read_* + 5 new (typst_render, template_list, template_render, read_xlsx, read_ppt) = 16
        assert len(MEDIA_TOOL_SCHEMAS) == 16
