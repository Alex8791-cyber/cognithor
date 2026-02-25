"""Tests für Multimodal Memory.

Testet:
  - MediaType Erkennung
  - MediaAsset Datenmodell
  - MultimodalMemory: Ingest, Suche, Duplikat-Erkennung, Directory-Ingest
  - Extraktion ohne Pipeline (Fallbacks)
  - Cross-modale Suche
  - Asset-Verwaltung (Remove, Stats)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.memory.multimodal import (
    MediaAsset,
    MediaType,
    MultimodalMemory,
    detect_media_type,
    file_hash,
)


# ============================================================================
# MediaType Erkennung
# ============================================================================


class TestMediaTypeDetection:
    """Erkennung von Medientypen anhand Dateiendung."""

    @pytest.mark.parametrize(
        "ext,expected",
        [
            (".jpg", MediaType.IMAGE),
            (".jpeg", MediaType.IMAGE),
            (".png", MediaType.IMAGE),
            (".gif", MediaType.IMAGE),
            (".webp", MediaType.IMAGE),
            (".bmp", MediaType.IMAGE),
            (".svg", MediaType.IMAGE),
            (".wav", MediaType.AUDIO),
            (".mp3", MediaType.AUDIO),
            (".ogg", MediaType.AUDIO),
            (".flac", MediaType.AUDIO),
            (".m4a", MediaType.AUDIO),
            (".pdf", MediaType.DOCUMENT),
            (".docx", MediaType.DOCUMENT),
            (".txt", MediaType.DOCUMENT),
            (".md", MediaType.DOCUMENT),
            (".html", MediaType.DOCUMENT),
            (".csv", MediaType.DOCUMENT),
            (".json", MediaType.DOCUMENT),
        ],
    )
    def test_known_extensions(self, ext: str, expected: MediaType) -> None:
        assert detect_media_type(f"test{ext}") == expected

    def test_unknown_extension(self) -> None:
        assert detect_media_type("file.xyz") is None
        assert detect_media_type("file.exe") is None
        assert detect_media_type("file") is None

    def test_case_insensitive(self) -> None:
        assert detect_media_type("image.JPG") == MediaType.IMAGE
        assert detect_media_type("doc.PDF") == MediaType.DOCUMENT

    def test_path_object(self) -> None:
        assert detect_media_type(Path("/home/user/photo.png")) == MediaType.IMAGE


# ============================================================================
# File Hash
# ============================================================================


class TestFileHash:
    """SHA-256 Hash für Mediendateien."""

    def test_hash_existing_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("Hello World")
        h = file_hash(f)
        assert len(h) == 64  # SHA-256 hex
        assert h.isalnum()

    def test_hash_nonexistent(self) -> None:
        assert file_hash("/nonexistent/file.txt") == ""

    def test_hash_deterministic(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("Deterministic Content")
        assert file_hash(f) == file_hash(f)

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("Content A")
        f2.write_text("Content B")
        assert file_hash(f1) != file_hash(f2)


# ============================================================================
# MediaAsset
# ============================================================================


class TestMediaAsset:
    """MediaAsset Datenmodell."""

    def test_basic_creation(self) -> None:
        asset = MediaAsset(
            id="test123",
            media_type=MediaType.IMAGE,
            file_path="/photos/sunset.jpg",
            file_hash="abc123",
        )
        assert asset.filename == "sunset.jpg"
        assert asset.media_type == MediaType.IMAGE
        assert asset.chunk_ids == []

    def test_filename_from_path(self) -> None:
        asset = MediaAsset(
            id="x", media_type=MediaType.DOCUMENT,
            file_path="/deep/nested/path/report.pdf", file_hash="h",
        )
        assert asset.filename == "report.pdf"


# ============================================================================
# MultimodalMemory
# ============================================================================


class TestMultimodalMemoryBasic:
    """Basis-Tests ohne Pipeline und Manager."""

    @pytest.fixture
    def mm(self) -> MultimodalMemory:
        return MultimodalMemory()

    def test_initial_empty(self, mm: MultimodalMemory) -> None:
        assert mm.asset_count == 0
        assert mm.list_assets() == []
        assert mm.stats()["total_assets"] == 0

    @pytest.mark.asyncio
    async def test_ingest_nonexistent_file(self, mm: MultimodalMemory) -> None:
        result = await mm.ingest_media("/nonexistent/file.jpg")
        assert result is None

    @pytest.mark.asyncio
    async def test_ingest_unsupported_type(
        self, mm: MultimodalMemory, tmp_path: Path,
    ) -> None:
        f = tmp_path / "file.xyz"
        f.write_text("data")
        result = await mm.ingest_media(f)
        assert result is None

    @pytest.mark.asyncio
    async def test_ingest_text_file_direct_read(
        self, mm: MultimodalMemory, tmp_path: Path,
    ) -> None:
        """Textdateien werden direkt gelesen (ohne Pipeline)."""
        f = tmp_path / "notes.txt"
        f.write_text("Wichtige Notizen über BU-Tarife")
        asset = await mm.ingest_media(f)

        assert asset is not None
        assert asset.media_type == MediaType.DOCUMENT
        assert "BU-Tarife" in asset.text_representation
        assert asset.extraction_method == "direct_read"
        assert mm.asset_count == 1

    @pytest.mark.asyncio
    async def test_ingest_with_text_override(
        self, mm: MultimodalMemory, tmp_path: Path,
    ) -> None:
        f = tmp_path / "photo.jpg"
        f.write_bytes(b"\xff\xd8\xff\xe0")  # JPEG magic bytes
        asset = await mm.ingest_media(f, text_override="Ein Sonnenuntergang am Strand")

        assert asset is not None
        assert "Sonnenuntergang" in asset.text_representation
        assert asset.extraction_method == "manual"

    @pytest.mark.asyncio
    async def test_ingest_markdown_direct_read(
        self, mm: MultimodalMemory, tmp_path: Path,
    ) -> None:
        f = tmp_path / "readme.md"
        f.write_text("# Jarvis Agent OS\n\nArchitektur-Überblick")
        asset = await mm.ingest_media(f)
        assert asset is not None
        assert "Jarvis" in asset.text_representation


class TestMultimodalDuplication:
    """Duplikat-Erkennung."""

    @pytest.mark.asyncio
    async def test_same_file_not_duplicated(self, tmp_path: Path) -> None:
        mm = MultimodalMemory()
        f = tmp_path / "notes.txt"
        f.write_text("Content")

        asset1 = await mm.ingest_media(f)
        asset2 = await mm.ingest_media(f)

        assert asset1 is not None
        assert asset2 is not None
        assert asset1.id == asset2.id  # Gleiche Datei = gleicher Asset
        assert mm.asset_count == 1  # Nicht dupliziert


class TestMultimodalSearch:
    """Cross-modale Suche."""

    @pytest.fixture
    async def mm_with_assets(self, tmp_path: Path) -> MultimodalMemory:
        mm = MultimodalMemory()

        f1 = tmp_path / "insurance.txt"
        f1.write_text("BU-Tarife Vergleich zwischen WWK und Allianz")
        await mm.ingest_media(f1)

        f2 = tmp_path / "recipe.txt"
        f2.write_text("Pasta Carbonara Rezept mit Speck und Ei")
        await mm.ingest_media(f2)

        f3 = tmp_path / "code.md"
        f3.write_text("Python Flask API für Jarvis WebChat")
        await mm.ingest_media(f3)

        return mm

    @pytest.mark.asyncio
    async def test_search_finds_relevant(
        self, mm_with_assets: MultimodalMemory,
    ) -> None:
        results = mm_with_assets.search_media("BU Tarife WWK")
        assert len(results) >= 1
        assert "insurance" in results[0][0].filename

    @pytest.mark.asyncio
    async def test_search_with_type_filter(
        self, mm_with_assets: MultimodalMemory,
    ) -> None:
        results = mm_with_assets.search_media("Python", media_type=MediaType.DOCUMENT)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_search_no_results(
        self, mm_with_assets: MultimodalMemory,
    ) -> None:
        results = mm_with_assets.search_media("Quantenphysik")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_top_k(
        self, mm_with_assets: MultimodalMemory,
    ) -> None:
        results = mm_with_assets.search_media("a", top_k=1)
        assert len(results) <= 1


class TestMultimodalWithPipeline:
    """Integration mit Mock-MediaPipeline."""

    @pytest.mark.asyncio
    async def test_image_analysis_via_pipeline(self, tmp_path: Path) -> None:
        mock_pipeline = AsyncMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.text = "Ein Diagramm mit BU-Tarifvergleich, Balkendiagramm"
        mock_result.metadata = {"model": "llava:13b"}
        mock_pipeline.analyze_image = AsyncMock(return_value=mock_result)

        mm = MultimodalMemory(media_pipeline=mock_pipeline)

        f = tmp_path / "chart.png"
        f.write_bytes(b"\x89PNG\r\n")  # PNG magic
        asset = await mm.ingest_media(f)

        assert asset is not None
        assert "BU-Tarifvergleich" in asset.text_representation
        assert "image_analysis" in asset.extraction_method
        mock_pipeline.analyze_image.assert_called_once()

    @pytest.mark.asyncio
    async def test_audio_transcription_via_pipeline(self, tmp_path: Path) -> None:
        mock_pipeline = AsyncMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.text = "Guten Tag, ich möchte eine Berufsunfähigkeitsversicherung abschließen"
        mock_result.metadata = {"model": "base"}
        mock_pipeline.transcribe_audio = AsyncMock(return_value=mock_result)

        mm = MultimodalMemory(media_pipeline=mock_pipeline)

        f = tmp_path / "call.wav"
        f.write_bytes(b"RIFF" + b"\x00" * 100)
        asset = await mm.ingest_media(f)

        assert asset is not None
        assert "Berufsunfähigkeitsversicherung" in asset.text_representation
        mock_pipeline.transcribe_audio.assert_called_once()

    @pytest.mark.asyncio
    async def test_pipeline_failure_graceful(self, tmp_path: Path) -> None:
        mock_pipeline = AsyncMock()
        mock_result = MagicMock()
        mock_result.success = False
        mock_pipeline.analyze_image = AsyncMock(return_value=mock_result)

        mm = MultimodalMemory(media_pipeline=mock_pipeline)
        f = tmp_path / "broken.jpg"
        f.write_bytes(b"\xff\xd8")
        asset = await mm.ingest_media(f)

        assert asset is None  # Kein Text extrahiert → kein Asset


class TestMultimodalWithManager:
    """Integration mit Mock-MemoryManager."""

    @pytest.mark.asyncio
    async def test_chunks_indexed_via_manager(self, tmp_path: Path) -> None:
        mock_manager = MagicMock()
        mock_manager.index_text.return_value = 3

        mock_chunk = MagicMock()
        mock_chunk.id = "chunk_1"
        mock_manager.index.get_chunks_by_source.return_value = [mock_chunk]

        mm = MultimodalMemory(memory_manager=mock_manager)

        f = tmp_path / "doc.txt"
        f.write_text("Wichtige Informationen über Policen")
        asset = await mm.ingest_media(f)

        assert asset is not None
        assert "chunk_1" in asset.chunk_ids
        mock_manager.index_text.assert_called_once()


class TestMultimodalDirectoryIngest:
    """Verzeichnis-basiertes Ingest."""

    @pytest.mark.asyncio
    async def test_ingest_directory(self, tmp_path: Path) -> None:
        mm = MultimodalMemory()

        (tmp_path / "notes.txt").write_text("Notiz 1")
        (tmp_path / "data.csv").write_text("a,b,c\n1,2,3")
        (tmp_path / "readme.md").write_text("# README")
        (tmp_path / "ignored.exe").write_text("binary")  # Nicht unterstützt

        assets = await mm.ingest_directory(tmp_path)
        assert len(assets) == 3  # txt + csv + md, nicht exe

    @pytest.mark.asyncio
    async def test_ingest_empty_directory(self, tmp_path: Path) -> None:
        mm = MultimodalMemory()
        assets = await mm.ingest_directory(tmp_path)
        assert assets == []

    @pytest.mark.asyncio
    async def test_ingest_nonexistent_directory(self) -> None:
        mm = MultimodalMemory()
        assets = await mm.ingest_directory("/nonexistent")
        assert assets == []


class TestMultimodalRemoveAndStats:
    """Asset-Verwaltung."""

    @pytest.mark.asyncio
    async def test_remove_asset(self, tmp_path: Path) -> None:
        mm = MultimodalMemory()
        f = tmp_path / "test.txt"
        f.write_text("Content")
        asset = await mm.ingest_media(f)
        assert asset is not None
        assert mm.asset_count == 1

        removed = mm.remove_asset(asset.id)
        assert removed is True
        assert mm.asset_count == 0

    def test_remove_nonexistent(self) -> None:
        mm = MultimodalMemory()
        assert mm.remove_asset("ghost") is False

    @pytest.mark.asyncio
    async def test_stats_by_type(self, tmp_path: Path) -> None:
        mm = MultimodalMemory()
        (tmp_path / "a.txt").write_text("A")
        (tmp_path / "b.md").write_text("B")
        await mm.ingest_directory(tmp_path)

        stats = mm.stats()
        assert stats["total_assets"] == 2
        assert "document" in stats["by_type"]

    @pytest.mark.asyncio
    async def test_get_assets_by_type(self, tmp_path: Path) -> None:
        mm = MultimodalMemory()
        (tmp_path / "a.txt").write_text("A")
        await mm.ingest_media(tmp_path / "a.txt")

        docs = mm.get_assets_by_type(MediaType.DOCUMENT)
        assert len(docs) == 1
        images = mm.get_assets_by_type(MediaType.IMAGE)
        assert len(images) == 0
