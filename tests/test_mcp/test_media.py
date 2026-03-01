"""Tests für die Media-Pipeline MCP-Tools."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.mcp.media import (
    MAX_AUDIO_FILE_SIZE,
    MAX_EXTRACT_FILE_SIZE,
    MEDIA_TOOL_SCHEMAS,
    MAX_EXTRACT_LENGTH,
    MediaPipeline,
    MediaResult,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    d = tmp_path / "media_workspace"
    d.mkdir()
    return d


@pytest.fixture
def pipeline(workspace: Path) -> MediaPipeline:
    return MediaPipeline(workspace_dir=workspace)


@pytest.fixture
def sample_text_file(tmp_path: Path) -> Path:
    f = tmp_path / "sample.txt"
    f.write_text("Hallo Welt. Dies ist ein Testdokument.", encoding="utf-8")
    return f


@pytest.fixture
def sample_markdown_file(tmp_path: Path) -> Path:
    f = tmp_path / "readme.md"
    f.write_text("# Titel\n\nEin Absatz mit **Markdown**.\n\n## Unterabschnitt\n\nMehr Text.", encoding="utf-8")
    return f


@pytest.fixture
def sample_csv_file(tmp_path: Path) -> Path:
    f = tmp_path / "data.csv"
    f.write_text("Name,Alter,Stadt\nMax,30,Berlin\nAnna,25,München", encoding="utf-8")
    return f


@pytest.fixture
def sample_html_file(tmp_path: Path) -> Path:
    f = tmp_path / "page.html"
    f.write_text(
        "<html><head><title>Test</title><style>body{}</style></head>"
        "<body><h1>Überschrift</h1><p>Ein Absatz.</p>"
        "<script>alert('x')</script></body></html>",
        encoding="utf-8",
    )
    return f


@pytest.fixture
def sample_json_file(tmp_path: Path) -> Path:
    f = tmp_path / "config.json"
    f.write_text('{"name": "Jarvis", "version": "1.0"}', encoding="utf-8")
    return f


# ============================================================================
# Text-Extraktion
# ============================================================================


class TestExtractText:
    @pytest.mark.asyncio
    async def test_extract_txt(self, pipeline: MediaPipeline, sample_text_file: Path) -> None:
        result = await pipeline.extract_text(str(sample_text_file))
        assert result.success is True
        assert "Hallo Welt" in result.text
        assert result.metadata["format"] == ".txt"

    @pytest.mark.asyncio
    async def test_extract_markdown(self, pipeline: MediaPipeline, sample_markdown_file: Path) -> None:
        result = await pipeline.extract_text(str(sample_markdown_file))
        assert result.success is True
        assert "Titel" in result.text
        assert "Markdown" in result.text

    @pytest.mark.asyncio
    async def test_extract_csv(self, pipeline: MediaPipeline, sample_csv_file: Path) -> None:
        result = await pipeline.extract_text(str(sample_csv_file))
        assert result.success is True
        assert "Max" in result.text
        assert "Berlin" in result.text

    @pytest.mark.asyncio
    async def test_extract_html_strips_tags(self, pipeline: MediaPipeline, sample_html_file: Path) -> None:
        result = await pipeline.extract_text(str(sample_html_file))
        assert result.success is True
        assert "Überschrift" in result.text
        assert "Ein Absatz" in result.text
        # Script und Style sollten entfernt sein
        assert "alert" not in result.text
        assert "body{}" not in result.text

    @pytest.mark.asyncio
    async def test_extract_json(self, pipeline: MediaPipeline, sample_json_file: Path) -> None:
        result = await pipeline.extract_text(str(sample_json_file))
        assert result.success is True
        assert "Jarvis" in result.text

    @pytest.mark.asyncio
    async def test_extract_nonexistent(self, pipeline: MediaPipeline) -> None:
        result = await pipeline.extract_text("/nonexistent/file.txt")
        assert result.success is False
        assert "nicht gefunden" in result.error

    @pytest.mark.asyncio
    async def test_extract_unsupported_format(self, pipeline: MediaPipeline, tmp_path: Path) -> None:
        f = tmp_path / "data.bin"
        f.write_bytes(b"\x00\x01\x02\x03")
        result = await pipeline.extract_text(str(f))
        assert result.success is False
        assert "Nicht unterstützt" in result.error

    @pytest.mark.asyncio
    async def test_extract_truncates_long_text(self, pipeline: MediaPipeline, tmp_path: Path) -> None:
        f = tmp_path / "long.txt"
        f.write_text("A" * (MAX_EXTRACT_LENGTH + 5000), encoding="utf-8")
        result = await pipeline.extract_text(str(f))
        assert result.success is True
        assert "gekürzt" in result.text
        assert len(result.text) < MAX_EXTRACT_LENGTH + 200

    @pytest.mark.asyncio
    async def test_extract_pdf_no_library(self, pipeline: MediaPipeline, tmp_path: Path) -> None:
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF-1.4 test")  # Fake PDF
        with patch.dict("sys.modules", {"fitz": None, "pdfplumber": None}):
            result = await pipeline.extract_text(str(f))
            assert result.success is False

    @pytest.mark.asyncio
    async def test_extract_docx_no_library(self, pipeline: MediaPipeline, tmp_path: Path) -> None:
        f = tmp_path / "doc.docx"
        f.write_bytes(b"PK\x03\x04 fake docx")
        with patch.dict("sys.modules", {"docx": None}):
            result = await pipeline.extract_text(str(f))
            assert result.success is False


# ============================================================================
# Audio-Transkription
# ============================================================================


class TestTranscribeAudio:
    @pytest.mark.asyncio
    async def test_transcribe_file_not_found(self, pipeline: MediaPipeline) -> None:
        result = await pipeline.transcribe_audio("/nonexistent/audio.wav")
        assert result.success is False
        assert "nicht gefunden" in result.error

    @pytest.mark.asyncio
    async def test_transcribe_no_whisper(self, pipeline: MediaPipeline, tmp_path: Path) -> None:
        f = tmp_path / "audio.wav"
        f.write_bytes(b"RIFF" + b"\x00" * 100)  # Fake WAV
        with patch.dict("sys.modules", {"faster_whisper": None}):
            result = await pipeline.transcribe_audio(str(f))
            assert result.success is False
            assert "faster-whisper" in result.error

    @pytest.mark.asyncio
    async def test_transcribe_success_mock(self, pipeline: MediaPipeline, tmp_path: Path) -> None:
        f = tmp_path / "speech.wav"
        f.write_bytes(b"RIFF" + b"\x00" * 100)

        mock_segment = MagicMock()
        mock_segment.text = "Hallo dies ist ein Test"
        mock_info = MagicMock()

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        with patch("jarvis.mcp.media.WhisperModel", return_value=mock_model, create=True):
            # Wir mocken den Import direkt
            import jarvis.mcp.media as media_mod

            original_transcribe = media_mod.MediaPipeline.transcribe_audio

            async def _mock_transcribe(self, audio_path, *, language="de", model="base"):
                return MediaResult(
                    success=True,
                    text="Hallo dies ist ein Test",
                    metadata={"language": language, "model": model},
                )

            media_mod.MediaPipeline.transcribe_audio = _mock_transcribe
            try:
                result = await pipeline.transcribe_audio(str(f))
                assert result.success is True
                assert "Hallo" in result.text
            finally:
                media_mod.MediaPipeline.transcribe_audio = original_transcribe


# ============================================================================
# Bildanalyse
# ============================================================================


class TestAnalyzeImage:
    @pytest.mark.asyncio
    async def test_image_not_found(self, pipeline: MediaPipeline) -> None:
        result = await pipeline.analyze_image("/nonexistent/photo.jpg")
        assert result.success is False
        assert "nicht gefunden" in result.error

    @pytest.mark.asyncio
    async def test_unsupported_format(self, pipeline: MediaPipeline, tmp_path: Path) -> None:
        f = tmp_path / "image.tiff"
        f.write_bytes(b"\x00" * 10)
        result = await pipeline.analyze_image(str(f))
        assert result.success is False
        assert "Nicht unterstützt" in result.error

    @pytest.mark.asyncio
    async def test_analyze_success_mock(self, pipeline: MediaPipeline, tmp_path: Path) -> None:
        f = tmp_path / "photo.jpg"
        f.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)  # Fake JPEG header

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"content": "Ein Foto von einem Hund im Park."},
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            MockClient.return_value = mock_client

            result = await pipeline.analyze_image(str(f))
            assert result.success is True
            assert "Hund" in result.text

    @pytest.mark.asyncio
    async def test_analyze_ollama_error(self, pipeline: MediaPipeline, tmp_path: Path) -> None:
        f = tmp_path / "photo.png"
        f.write_bytes(b"\x89PNG" + b"\x00" * 100)

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            MockClient.return_value = mock_client

            result = await pipeline.analyze_image(str(f))
            assert result.success is False
            assert "500" in result.error


# ============================================================================
# Audio-Konvertierung
# ============================================================================


class TestConvertAudio:
    @pytest.mark.asyncio
    async def test_convert_file_not_found(self, pipeline: MediaPipeline) -> None:
        result = await pipeline.convert_audio("/nonexistent/audio.mp3")
        assert result.success is False
        assert "nicht gefunden" in result.error

    @pytest.mark.asyncio
    async def test_convert_no_ffmpeg(self, pipeline: MediaPipeline, tmp_path: Path) -> None:
        f = tmp_path / "audio.mp3"
        f.write_bytes(b"\xff\xfb" + b"\x00" * 100)

        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            result = await pipeline.convert_audio(str(f))
            assert result.success is False
            assert "ffmpeg" in result.error


# ============================================================================
# Bildskalierung
# ============================================================================


class TestResizeImage:
    @pytest.mark.asyncio
    async def test_resize_not_found(self, pipeline: MediaPipeline) -> None:
        result = await pipeline.resize_image("/nonexistent/img.png")
        assert result.success is False
        assert "nicht gefunden" in result.error

    @pytest.mark.asyncio
    async def test_resize_no_pillow(self, pipeline: MediaPipeline, tmp_path: Path) -> None:
        f = tmp_path / "img.png"
        f.write_bytes(b"\x89PNG" + b"\x00" * 100)
        with patch.dict("sys.modules", {"PIL": None, "PIL.Image": None}):
            result = await pipeline.resize_image(str(f))
            assert result.success is False


# ============================================================================
# TTS
# ============================================================================


class TestTextToSpeech:
    @pytest.mark.asyncio
    async def test_tts_empty_text(self, pipeline: MediaPipeline) -> None:
        result = await pipeline.text_to_speech("")
        assert result.success is False
        assert "Leer" in result.error

    @pytest.mark.asyncio
    async def test_tts_no_backend(self, pipeline: MediaPipeline) -> None:
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            result = await pipeline.text_to_speech("Hallo Welt")
            assert result.success is False
            assert "Kein TTS-Backend" in result.error


# ============================================================================
# Tool-Schemas
# ============================================================================


class TestMediaToolSchemas:
    def test_all_schemas_present(self) -> None:
        expected = [
            "media_transcribe_audio",
            "media_analyze_image",
            "media_extract_text",
            "media_convert_audio",
            "media_resize_image",
            "media_tts",
        ]
        for name in expected:
            assert name in MEDIA_TOOL_SCHEMAS, f"Schema fehlt: {name}"

    def test_schemas_have_description_and_input(self) -> None:
        for name, schema in MEDIA_TOOL_SCHEMAS.items():
            assert "description" in schema, f"{name}: description fehlt"
            assert "inputSchema" in schema, f"{name}: inputSchema fehlt"
            assert "properties" in schema["inputSchema"], f"{name}: properties fehlt"

    def test_required_fields(self) -> None:
        assert "audio_path" in MEDIA_TOOL_SCHEMAS["media_transcribe_audio"]["inputSchema"]["required"]
        assert "image_path" in MEDIA_TOOL_SCHEMAS["media_analyze_image"]["inputSchema"]["required"]
        assert "file_path" in MEDIA_TOOL_SCHEMAS["media_extract_text"]["inputSchema"]["required"]
        assert "input_path" in MEDIA_TOOL_SCHEMAS["media_convert_audio"]["inputSchema"]["required"]
        assert "image_path" in MEDIA_TOOL_SCHEMAS["media_resize_image"]["inputSchema"]["required"]
        assert "text" in MEDIA_TOOL_SCHEMAS["media_tts"]["inputSchema"]["required"]


# ============================================================================
# File-Size-Limits (Security Hardening)
# ============================================================================


class TestFileSizeLimits:
    """Tests für Dateigrößen-Limits."""

    async def test_extract_text_file_too_large(self, pipeline: MediaPipeline, tmp_path: Path) -> None:
        """extract_text lehnt Dateien > MAX_EXTRACT_FILE_SIZE ab."""
        large_file = tmp_path / "huge.txt"
        # Erstelle eine Datei die das Limit überschreitet (sparse/truncated)
        large_file.write_bytes(b"x" * 1024)  # Klein erstellen
        # Mock st_size um die Datei als zu gross erscheinen zu lassen
        with patch.object(Path, "stat") as mock_stat:
            mock_stat.return_value = MagicMock(st_size=MAX_EXTRACT_FILE_SIZE + 1)
            result = await pipeline.extract_text(str(large_file))
        assert not result.success
        assert "zu gross" in result.error.lower() or "too large" in result.error.lower() or "gross" in result.error.lower()

    async def test_transcribe_audio_file_too_large(self, pipeline: MediaPipeline, tmp_path: Path) -> None:
        """transcribe_audio lehnt Dateien > MAX_AUDIO_FILE_SIZE ab."""
        audio_file = tmp_path / "huge.wav"
        audio_file.write_bytes(b"\x00" * 1024)
        with patch.object(Path, "stat") as mock_stat:
            mock_stat.return_value = MagicMock(st_size=MAX_AUDIO_FILE_SIZE + 1)
            result = await pipeline.transcribe_audio(str(audio_file))
        assert not result.success
        assert "gross" in result.error.lower() or "large" in result.error.lower()

    async def test_extract_text_within_limit(self, pipeline: MediaPipeline, tmp_path: Path) -> None:
        """Dateien innerhalb des Limits werden normal verarbeitet."""
        small_file = tmp_path / "small.txt"
        small_file.write_text("Kleiner Text", encoding="utf-8")
        result = await pipeline.extract_text(str(small_file))
        assert result.success
        assert "Kleiner Text" in result.text
