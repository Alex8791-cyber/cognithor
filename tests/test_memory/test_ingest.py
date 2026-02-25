"""Tests für die Knowledge Ingest Pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.memory.ingest import (
    IngestConfig,
    IngestPipeline,
    IngestResult,
    TextExtractor,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def ingest_dir(tmp_path: Path) -> Path:
    """Erstellt Watch-Verzeichnisstruktur."""
    watch = tmp_path / "ingest"
    watch.mkdir()
    (tmp_path / "ingest" / "processed").mkdir()
    (tmp_path / "ingest" / "failed").mkdir()
    return watch


@pytest.fixture
def config(ingest_dir: Path) -> IngestConfig:
    return IngestConfig(
        watch_dir=ingest_dir,
        processed_dir=ingest_dir / "processed",
        failed_dir=ingest_dir / "failed",
        generate_embeddings=False,
    )


@pytest.fixture
def pipeline(config: IngestConfig) -> IngestPipeline:
    """Pipeline ohne MemoryManager (nur Extraktion)."""
    return IngestPipeline(config, memory_manager=None)


@pytest.fixture
def pipeline_with_memory(config: IngestConfig) -> IngestPipeline:
    """Pipeline mit Mock-MemoryManager."""
    mock_memory = MagicMock()
    mock_memory.index_text.return_value = 5  # 5 Chunks
    return IngestPipeline(config, memory_manager=mock_memory)


# ============================================================================
# IngestResult
# ============================================================================


class TestIngestResult:
    def test_success_summary(self) -> None:
        r = IngestResult(
            file_path="/tmp/test.md",
            file_name="test.md",
            success=True,
            chunks_created=5,
            text_length=2000,
        )
        assert "✅" in r.summary
        assert "5 Chunks" in r.summary

    def test_error_summary(self) -> None:
        r = IngestResult(
            file_path="/tmp/bad.pdf",
            file_name="bad.pdf",
            error="PDF-Extraktion fehlgeschlagen",
        )
        assert "❌" in r.summary
        assert "fehlgeschlagen" in r.summary


# ============================================================================
# TextExtractor
# ============================================================================


class TestTextExtractor:
    @pytest.fixture
    def extractor(self) -> TextExtractor:
        return TextExtractor()

    @pytest.mark.asyncio
    async def test_extract_txt(self, extractor: TextExtractor, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("Hallo Welt", encoding="utf-8")
        text = await extractor.extract(f)
        assert text == "Hallo Welt"

    @pytest.mark.asyncio
    async def test_extract_markdown(self, extractor: TextExtractor, tmp_path: Path) -> None:
        f = tmp_path / "test.md"
        f.write_text("# Titel\n\nInhalt hier", encoding="utf-8")
        text = await extractor.extract(f)
        assert "Titel" in text
        assert "Inhalt" in text

    @pytest.mark.asyncio
    async def test_extract_csv(self, extractor: TextExtractor, tmp_path: Path) -> None:
        f = tmp_path / "data.csv"
        f.write_text("name,wert\nAlpha,1\nBeta,2", encoding="utf-8")
        text = await extractor.extract(f)
        assert "Alpha" in text
        assert "Beta" in text

    @pytest.mark.asyncio
    async def test_extract_json(self, extractor: TextExtractor, tmp_path: Path) -> None:
        f = tmp_path / "data.json"
        f.write_text('{"key": "value"}', encoding="utf-8")
        text = await extractor.extract(f)
        assert "key" in text

    @pytest.mark.asyncio
    async def test_extract_html(self, extractor: TextExtractor, tmp_path: Path) -> None:
        f = tmp_path / "page.html"
        f.write_text(
            "<html><head><style>body{}</style></head>"
            "<body><h1>Titel</h1><p>Inhalt</p>"
            "<script>alert('hi')</script></body></html>",
            encoding="utf-8",
        )
        text = await extractor.extract(f)
        assert "Titel" in text
        assert "Inhalt" in text
        assert "script" not in text.lower()
        assert "style" not in text.lower()

    @pytest.mark.asyncio
    async def test_extract_unsupported(self, extractor: TextExtractor, tmp_path: Path) -> None:
        f = tmp_path / "data.exe"
        f.write_bytes(b"\x00\x01\x02")
        with pytest.raises(ValueError, match="Nicht unterstützt"):
            await extractor.extract(f)


# ============================================================================
# IngestPipeline — Einzeldateien
# ============================================================================


class TestIngestFile:
    @pytest.mark.asyncio
    async def test_ingest_txt_success(self, pipeline: IngestPipeline, ingest_dir: Path) -> None:
        f = ingest_dir / "test.txt"
        f.write_text("Dies ist ein Testdokument mit genug Text.", encoding="utf-8")

        result = await pipeline.ingest_file(f)
        assert result.success is True
        assert result.text_length > 0
        assert result.chunks_created >= 1

        # Datei sollte nach processed/ verschoben sein
        assert not f.exists()
        assert len(list((ingest_dir / "processed").iterdir())) == 1

    @pytest.mark.asyncio
    async def test_ingest_with_memory_manager(
        self, pipeline_with_memory: IngestPipeline, ingest_dir: Path
    ) -> None:
        f = ingest_dir / "doc.md"
        f.write_text("# Dokumentation\n\nWichtiger Inhalt.", encoding="utf-8")

        result = await pipeline_with_memory.ingest_file(f)
        assert result.success is True
        assert result.chunks_created == 5  # Mock gibt 5 zurück

    @pytest.mark.asyncio
    async def test_ingest_nonexistent_file(self, pipeline: IngestPipeline) -> None:
        result = await pipeline.ingest_file(Path("/tmp/nonexistent.txt"))
        assert result.success is False
        assert "nicht gefunden" in result.error

    @pytest.mark.asyncio
    async def test_ingest_unsupported_format(self, pipeline: IngestPipeline, ingest_dir: Path) -> None:
        f = ingest_dir / "binary.exe"
        f.write_bytes(b"\x00\x01")
        result = await pipeline.ingest_file(f)
        assert result.success is False
        assert "Nicht unterstützt" in result.error

    @pytest.mark.asyncio
    async def test_ingest_too_large(self, pipeline: IngestPipeline, ingest_dir: Path) -> None:
        pipeline._config.max_file_size_bytes = 100  # 100 Bytes Limit
        f = ingest_dir / "large.txt"
        f.write_text("X" * 200, encoding="utf-8")

        result = await pipeline.ingest_file(f)
        assert result.success is False
        assert "zu groß" in result.error

    @pytest.mark.asyncio
    async def test_ingest_empty_text(self, pipeline: IngestPipeline, ingest_dir: Path) -> None:
        f = ingest_dir / "empty.txt"
        f.write_text("", encoding="utf-8")

        result = await pipeline.ingest_file(f)
        assert result.success is False
        assert "Kein Text" in result.error

    @pytest.mark.asyncio
    async def test_ingest_duplicate_detection(self, pipeline: IngestPipeline, ingest_dir: Path) -> None:
        # Erste Datei
        f1 = ingest_dir / "doc1.txt"
        f1.write_text("Gleicher Inhalt", encoding="utf-8")
        result1 = await pipeline.ingest_file(f1)
        assert result1.success is True

        # Gleiche Datei nochmal (gleicher Name + Größe + Anfang)
        f2 = ingest_dir / "doc1.txt"
        f2.write_text("Gleicher Inhalt", encoding="utf-8")
        result2 = await pipeline.ingest_file(f2)
        assert result2.success is False
        assert "Bereits verarbeitet" in result2.error


# ============================================================================
# IngestPipeline — Batch
# ============================================================================


class TestScanAndIngest:
    @pytest.mark.asyncio
    async def test_scan_empty(self, pipeline: IngestPipeline) -> None:
        results = await pipeline.scan_and_ingest()
        assert results == []

    @pytest.mark.asyncio
    async def test_scan_multiple_files(self, pipeline: IngestPipeline, ingest_dir: Path) -> None:
        (ingest_dir / "a.txt").write_text("Datei A Inhalt", encoding="utf-8")
        (ingest_dir / "b.md").write_text("# Datei B\nInhalt", encoding="utf-8")
        (ingest_dir / "c.csv").write_text("col1,col2\n1,2", encoding="utf-8")

        results = await pipeline.scan_and_ingest()
        assert len(results) == 3
        success = [r for r in results if r.success]
        assert len(success) == 3

    @pytest.mark.asyncio
    async def test_scan_ignores_unsupported(self, pipeline: IngestPipeline, ingest_dir: Path) -> None:
        (ingest_dir / "good.txt").write_text("Guter Inhalt", encoding="utf-8")
        (ingest_dir / "bad.exe").write_bytes(b"\x00")  # Wird ignoriert

        results = await pipeline.scan_and_ingest()
        assert len(results) == 1


# ============================================================================
# Statistiken
# ============================================================================


class TestStats:
    def test_initial_stats(self, pipeline: IngestPipeline) -> None:
        stats = pipeline.stats()
        assert stats["pending"] == 0
        assert stats["processed"] == 0
        assert stats["failed"] == 0
        assert ".md" in stats["supported_formats"]
        assert ".pdf" in stats["supported_formats"]

    @pytest.mark.asyncio
    async def test_stats_after_ingest(self, pipeline: IngestPipeline, ingest_dir: Path) -> None:
        (ingest_dir / "test.txt").write_text("Inhalt", encoding="utf-8")
        await pipeline.scan_and_ingest()

        stats = pipeline.stats()
        assert stats["pending"] == 0
        assert stats["processed"] == 1


# ============================================================================
# Watch (Stop-Mechanismus)
# ============================================================================


class TestWatch:
    def test_stop(self, pipeline: IngestPipeline) -> None:
        pipeline._running = True
        pipeline.stop()
        assert pipeline._running is False
