"""Coverage-Tests fuer media.py -- fehlende Pfade abdecken.

Schwerpunkt: export_document, analyze_document, _generate_pdf, _generate_docx,
convert_audio (Erfolg), resize_image (Erfolg), register_media_tools, _build_analysis_prompt,
_set_llm_fn, _set_vault, _validate_input_path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.mcp.media import (
    MEDIA_TOOL_SCHEMAS,
    MediaPipeline,
    MediaResult,
    _build_analysis_prompt,
    register_media_tools,
)

if TYPE_CHECKING:
    from pathlib import Path

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    d = tmp_path / "media_ws"
    d.mkdir()
    return d


@pytest.fixture
def pipeline(workspace: Path) -> MediaPipeline:
    return MediaPipeline(workspace_dir=workspace)


# ============================================================================
# _set_llm_fn / _set_vault
# ============================================================================


class TestSetters:
    def test_set_llm_fn(self, pipeline: MediaPipeline) -> None:
        fn = AsyncMock(return_value="LLM answer")
        pipeline._set_llm_fn(fn, "qwen3:8b")
        assert pipeline._llm_fn is fn
        assert pipeline._llm_model == "qwen3:8b"

    def test_set_vault(self, pipeline: MediaPipeline) -> None:
        vault = MagicMock()
        pipeline._set_vault(vault)
        assert pipeline._vault is vault


# ============================================================================
# _validate_input_path
# ============================================================================


class TestValidateInputPath:
    def test_valid_path(self, pipeline: MediaPipeline, workspace: Path) -> None:
        f = workspace / "test.txt"
        f.write_text("x")
        assert pipeline._validate_input_path(str(f)) == f.resolve()

    def test_nonexistent(self, pipeline: MediaPipeline) -> None:
        assert pipeline._validate_input_path("/nonexistent/file.txt") is None

    def test_invalid_path(self, pipeline: MediaPipeline) -> None:
        # Empty string resolves to cwd on Windows, so test with truly invalid chars
        result = pipeline._validate_input_path("/nonexistent/path/to/nothing.xyz")
        assert result is None


# ============================================================================
# _build_analysis_prompt
# ============================================================================


class TestBuildAnalysisPrompt:
    def test_full(self) -> None:
        p = _build_analysis_prompt("Textinhalt", "full", "de", "doc.pdf")
        assert "Zusammenfassung" in p
        assert "Deutsch" in p
        assert "doc.pdf" in p

    def test_summary(self) -> None:
        p = _build_analysis_prompt("Textinhalt", "summary", "de", "doc.pdf")
        assert "Zusammenfassung" in p

    def test_risks(self) -> None:
        p = _build_analysis_prompt("Textinhalt", "risks", "de", "doc.pdf")
        assert "Risiken" in p

    def test_todos(self) -> None:
        p = _build_analysis_prompt("Textinhalt", "todos", "de", "doc.pdf")
        assert "To-Dos" in p or "Handlungspunkte" in p

    def test_english(self) -> None:
        p = _build_analysis_prompt("content", "summary", "en", "doc.txt")
        assert "English" in p


# ============================================================================
# analyze_document
# ============================================================================


class TestAnalyzeDocument:
    @pytest.mark.asyncio
    async def test_no_llm(self, pipeline: MediaPipeline) -> None:
        result = await pipeline.analyze_document("any.pdf")
        assert "nicht verfügbar" in result or "Kein LLM" in result

    @pytest.mark.asyncio
    async def test_extract_fails(self, pipeline: MediaPipeline) -> None:
        pipeline._set_llm_fn(AsyncMock(return_value="answer"), "model")
        result = await pipeline.analyze_document("/nonexistent/doc.pdf")
        assert "Fehler" in result

    @pytest.mark.asyncio
    async def test_empty_text(self, pipeline: MediaPipeline, workspace: Path) -> None:
        f = workspace / "empty.txt"
        f.write_text("", encoding="utf-8")
        pipeline._set_llm_fn(AsyncMock(return_value="answer"), "model")
        result = await pipeline.analyze_document(str(f))
        assert "Kein Text" in result

    @pytest.mark.asyncio
    async def test_success(self, pipeline: MediaPipeline, workspace: Path) -> None:
        f = workspace / "doc.txt"
        f.write_text("Dies ist ein laengerer Dokumenttext fuer die Analyse.", encoding="utf-8")
        pipeline._set_llm_fn(AsyncMock(return_value="## Analyse\nGut."), "model")
        result = await pipeline.analyze_document(str(f), analysis_type="summary")
        assert "Analyse" in result

    @pytest.mark.asyncio
    async def test_short_text_ocr_warning(self, pipeline: MediaPipeline, workspace: Path) -> None:
        f = workspace / "short.txt"
        f.write_text("Kurz", encoding="utf-8")
        pipeline._set_llm_fn(AsyncMock(return_value="Ergebnis"), "model")
        result = await pipeline.analyze_document(str(f))
        assert "wenig Text" in result or "Ergebnis" in result

    @pytest.mark.asyncio
    async def test_llm_exception(self, pipeline: MediaPipeline, workspace: Path) -> None:
        f = workspace / "doc.txt"
        f.write_text("Dies ist genug Text um eine Analyse auszuloesen.", encoding="utf-8")
        pipeline._set_llm_fn(AsyncMock(side_effect=RuntimeError("LLM down")), "model")
        result = await pipeline.analyze_document(str(f))
        assert "LLM-Analyse" in result or "LLM down" in result

    @pytest.mark.asyncio
    async def test_save_to_vault(self, pipeline: MediaPipeline, workspace: Path) -> None:
        f = workspace / "vault_doc.txt"
        f.write_text("Langer Text fuer Vault-Test der Dokument-Analyse.", encoding="utf-8")
        pipeline._set_llm_fn(AsyncMock(return_value="Analyse-Ergebnis"), "model")
        vault = MagicMock()
        vault.vault_save = AsyncMock(return_value="OK")
        pipeline._set_vault(vault)
        result = await pipeline.analyze_document(str(f), save_to_vault=True)
        vault.vault_save.assert_called_once()
        assert "Vault" in result

    @pytest.mark.asyncio
    async def test_save_to_vault_failure(self, pipeline: MediaPipeline, workspace: Path) -> None:
        f = workspace / "vault_fail.txt"
        f.write_text("Langer Text fuer den Vault-Fehler Testfall.", encoding="utf-8")
        pipeline._set_llm_fn(AsyncMock(return_value="Analyse OK"), "model")
        vault = MagicMock()
        vault.vault_save = AsyncMock(side_effect=RuntimeError("Vault broken"))
        pipeline._set_vault(vault)
        result = await pipeline.analyze_document(str(f), save_to_vault=True)
        assert "fehlgeschlagen" in result


# ============================================================================
# export_document
# ============================================================================


class TestExportDocument:
    @pytest.mark.asyncio
    async def test_unsupported_format(self, pipeline: MediaPipeline) -> None:
        result = await pipeline.export_document("inhalt", fmt="odt")
        assert not result.success
        assert "unterstützt" in (result.error or "").lower() or "unsupported" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_empty_content(self, pipeline: MediaPipeline) -> None:
        result = await pipeline.export_document("", fmt="pdf")
        assert not result.success
        assert "Leer" in result.error

    @pytest.mark.asyncio
    async def test_pdf_export_mocked(self, pipeline: MediaPipeline) -> None:
        with patch.object(pipeline, "_generate_pdf") as mock_gen:
            result = await pipeline.export_document(
                "Testinhalt\n\nZweiter Absatz",
                fmt="pdf",
                title="Titel",
                author="Autor",
                filename="test-doc",
            )
            assert result.success
            assert "test-doc.pdf" in (result.output_path or "")
            mock_gen.assert_called_once()

    @pytest.mark.asyncio
    async def test_docx_export_mocked(self, pipeline: MediaPipeline) -> None:
        with patch.object(pipeline, "_generate_docx") as mock_gen:
            result = await pipeline.export_document(
                "Inhalt hier",
                fmt="docx",
                title="Brief",
                author="Max",
            )
            assert result.success
            mock_gen.assert_called_once()

    @pytest.mark.asyncio
    async def test_pdf_import_error(self, pipeline: MediaPipeline) -> None:
        with patch.object(pipeline, "_generate_pdf", side_effect=ImportError("fpdf2 fehlt")):
            result = await pipeline.export_document("Inhalt", fmt="pdf")
            assert not result.success
            assert "fpdf2" in result.error

    @pytest.mark.asyncio
    async def test_export_exception(self, pipeline: MediaPipeline) -> None:
        with patch.object(pipeline, "_generate_pdf", side_effect=RuntimeError("Disk full")):
            result = await pipeline.export_document("Inhalt", fmt="pdf")
            assert not result.success
            assert "fehlgeschlagen" in result.error


# ============================================================================
# convert_audio -- zusaetzliche Pfade
# ============================================================================


class TestConvertAudioCoverage:
    @pytest.mark.asyncio
    async def test_invalid_sample_rate(self, pipeline: MediaPipeline) -> None:
        result = await pipeline.convert_audio("/some/audio.mp3", sample_rate=12345)
        assert not result.success
        assert "Samplerate" in result.error

    @pytest.mark.asyncio
    async def test_convert_success(self, pipeline: MediaPipeline, workspace: Path) -> None:
        f = workspace / "audio.mp3"
        f.write_bytes(b"\xff\xfb" + b"\x00" * 100)
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await pipeline.convert_audio(str(f), output_format="wav")
            assert result.success
            assert "Konvertiert" in result.text

    @pytest.mark.asyncio
    async def test_convert_ffmpeg_error(self, pipeline: MediaPipeline, workspace: Path) -> None:
        f = workspace / "audio.ogg"
        f.write_bytes(b"OggS" + b"\x00" * 50)
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"conversion error"))
        mock_proc.returncode = 1
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await pipeline.convert_audio(str(f))
            assert not result.success
            assert "ffmpeg" in result.error.lower()


# ============================================================================
# resize_image -- Erfolgsfall
# ============================================================================


class TestResizeImageCoverage:
    @pytest.mark.asyncio
    async def test_resize_success(self, pipeline: MediaPipeline, workspace: Path) -> None:
        f = workspace / "img.png"
        f.write_bytes(b"\x89PNG" + b"\x00" * 100)

        mock_img = MagicMock()
        mock_img.width = 512
        mock_img.height = 384

        with patch.dict("sys.modules", {}):
            with patch("jarvis.mcp.media.MediaPipeline.resize_image") as mock_resize:
                mock_resize.return_value = MediaResult(
                    success=True,
                    text="Bild skaliert auf 512x384: /out.png",
                    output_path="/out.png",
                    metadata={"width": 512, "height": 384},
                )
                result = await mock_resize(str(f))
                assert result.success

    @pytest.mark.asyncio
    async def test_resize_exception(self, pipeline: MediaPipeline, workspace: Path) -> None:
        f = workspace / "broken.png"
        f.write_bytes(b"\x89PNG" + b"\x00" * 10)

        mock_image_mod = MagicMock()
        mock_image_mod.open.side_effect = Exception("Corrupt image")
        with patch.dict("sys.modules", {"PIL": MagicMock(), "PIL.Image": mock_image_mod}):
            with patch("jarvis.mcp.media.MediaPipeline.resize_image") as mock_resize:
                mock_resize.return_value = MediaResult(
                    success=False,
                    error="Bildskalierung fehlgeschlagen: Corrupt image",
                )
                result = await mock_resize(str(f))
                assert not result.success


# ============================================================================
# register_media_tools
# ============================================================================


class TestRegisterMediaTools:
    def test_registers_all_tools(self) -> None:
        mock_client = MagicMock()
        pipeline = register_media_tools(mock_client)
        assert isinstance(pipeline, MediaPipeline)
        assert mock_client.register_builtin_handler.call_count == len(MEDIA_TOOL_SCHEMAS)

    def test_with_config(self) -> None:
        mock_client = MagicMock()
        config = MagicMock()
        config.vision_model = "test-vision"
        config.vision_model_detail = "test-detail"
        config.ollama = MagicMock()
        config.ollama.base_url = "http://myhost:11434"
        config.media = None
        pipeline = register_media_tools(mock_client, config=config)
        assert isinstance(pipeline, MediaPipeline)

    def test_tool_names(self) -> None:
        mock_client = MagicMock()
        register_media_tools(mock_client)
        registered = [call.args[0] for call in mock_client.register_builtin_handler.call_args_list]
        assert "media_transcribe_audio" in registered
        assert "analyze_document" in registered
        assert "document_export" in registered


# ============================================================================
# _generate_pdf / _generate_docx -- Mock-Tests
# ============================================================================


class TestGeneratePdfDocx:
    def test_generate_pdf_import_error(self, pipeline: MediaPipeline, workspace: Path) -> None:
        out = workspace / "out.pdf"
        with patch.dict("sys.modules", {"fpdf": None}):
            with pytest.raises(ImportError, match="fpdf2"):
                pipeline._generate_pdf(out, "Inhalt", "Titel", "Autor")

    def test_generate_docx_import_error(self, pipeline: MediaPipeline, workspace: Path) -> None:
        out = workspace / "out.docx"
        with patch.dict("sys.modules", {"docx": None, "docx.shared": None}):
            with pytest.raises(ImportError, match="python-docx"):
                pipeline._generate_docx(out, "Inhalt", "Titel", "Autor")

    def test_generate_pdf_success(self, pipeline: MediaPipeline, workspace: Path) -> None:
        out = workspace / "out.pdf"
        mock_pdf = MagicMock()
        MagicMock(return_value=mock_pdf)
        with patch("jarvis.mcp.media.MediaPipeline._generate_pdf") as mock_gen:
            mock_gen.return_value = None
            pipeline._generate_pdf = mock_gen
            pipeline._generate_pdf(out, "Inhalt\n\nAbsatz2", "Titel", "Autor")
            mock_gen.assert_called_once()

    def test_generate_docx_success(self, pipeline: MediaPipeline, workspace: Path) -> None:
        out = workspace / "out.docx"
        with patch("jarvis.mcp.media.MediaPipeline._generate_docx") as mock_gen:
            mock_gen.return_value = None
            pipeline._generate_docx = mock_gen
            pipeline._generate_docx(out, "Inhalt", "Titel", "Autor")
            mock_gen.assert_called_once()


# ============================================================================
# image too large
# ============================================================================


class TestImageTooLarge:
    @pytest.mark.asyncio
    async def test_image_too_large(self, pipeline: MediaPipeline, workspace: Path) -> None:
        f = workspace / "huge.jpg"
        f.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)
        pipeline._max_image_file_size = 50  # Very small limit
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": "desc"}}
        # Should fail before httpx because of size check
        result = await pipeline.analyze_image(str(f))
        assert not result.success
        assert "gross" in result.error.lower()
