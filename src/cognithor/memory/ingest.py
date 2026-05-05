"""Knowledge Ingest Pipeline: Automatic ingestion of documents into RAG.

Ueberwacht ein Ingest-Verzeichnis (~/.cognithor/ingest/) und verarbeitet
neue Dateien automatisch:

  1. Datei erkannt (neue/geaenderte Datei im Watch-Verzeichnis)
  2. Text extrahiert (PDF, DOCX, HTML, Markdown, TXT, CSV, JSON)
  3. In Chunks aufgeteilt (Token-basiert mit Overlap)
  4. In Memory-Index gespeichert (FTS5 + optional Embeddings)
  5. Datei in „processed" verschoben

Unterstuetzte Formate:
  - .md, .txt, .csv, .json, .xml → Direkt lesen
  - .html → HTML-Tags strippen
  - .pdf → PyMuPDF/pdfplumber (via MediaPipeline)
  - .docx → python-docx (via MediaPipeline)

Integration:
  - Nutzt bestehenden MemoryManager.index_text() fuer Chunking + DB
  - Nutzt MediaPipeline.extract_text() fuer PDF/DOCX
  - Optionale Embedding-Generierung via MemoryManager.index_with_embeddings()

Bibel-Referenz: §7.1 (Knowledge Base), §4.3 (Memory Ingest)
"""

from __future__ import annotations

import asyncio
import hashlib
import shutil
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from cognithor.utils.logging import get_logger

log = get_logger(__name__)


# ============================================================================
# Konfiguration
# ============================================================================


SUPPORTED_EXTENSIONS = {
    ".md",
    ".txt",
    ".csv",
    ".json",
    ".xml",
    ".html",
    ".htm",
    ".pdf",
    ".docx",
}

# Maximale Dateigroesse fuer Ingest (10 MB)
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024


@dataclass
class IngestConfig:
    """Konfiguration der Ingest-Pipeline."""

    # Verzeichnisse
    watch_dir: Path = field(default_factory=lambda: Path.home() / ".cognithor" / "ingest")
    processed_dir: Path = field(
        default_factory=lambda: Path.home() / ".cognithor" / "ingest" / "processed"
    )
    failed_dir: Path = field(
        default_factory=lambda: Path.home() / ".cognithor" / "ingest" / "failed"
    )

    # Verarbeitung
    max_file_size_bytes: int = MAX_FILE_SIZE_BYTES
    generate_embeddings: bool = False  # Braucht laufendes Ollama
    poll_interval_seconds: float = 5.0  # Polling-Intervall für Watch

    # Chunking (Defaults aus MemoryManager)
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64


# ============================================================================
# Ingest-Ergebnis
# ============================================================================


@dataclass
class IngestResult:
    """Ergebnis der Verarbeitung einer einzelnen Datei."""

    file_path: str
    file_name: str
    success: bool = False
    chunks_created: int = 0
    text_length: int = 0
    error: str | None = None
    duration_seconds: float = 0.0
    content_hash: str = ""

    @property
    def summary(self) -> str:
        if self.success:
            return f"✅ {self.file_name}: {self.chunks_created} Chunks ({self.text_length} Zeichen)"
        return f"❌ {self.file_name}: {self.error}"


# ============================================================================
# Text-Extraktion
# ============================================================================


class TextExtractor:
    """Extrahiert Text aus verschiedenen Dateiformaten.

    Nutzt die MediaPipeline fuer PDF/DOCX wenn verfuegbar,
    sonst direkte Extraktion fuer einfache Formate.
    """

    def __init__(self) -> None:
        self._media_pipeline: Any | None = None
        self._init_media_pipeline()

    def _init_media_pipeline(self) -> None:
        """Versucht MediaPipeline zu laden (optional)."""
        try:
            from cognithor.mcp.media import MediaPipeline

            self._media_pipeline = MediaPipeline()
        except ImportError:
            log.debug("media_pipeline_not_available_for_ingest")

    async def extract(self, file_path: Path) -> str:
        """Extrahiert Text aus einer Datei.

        Args:
            file_path: Pfad zur Datei.

        Returns:
            Extrahierter Text.

        Raises:
            ValueError: Wenn Format nicht unterstuetzt.
            IOError: Wenn Datei nicht lesbar.
        """
        suffix = file_path.suffix.lower()

        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Nicht unterstütztes Format: {suffix}")

        if suffix in (".md", ".txt"):
            return file_path.read_text(encoding="utf-8", errors="replace")

        if suffix == ".csv":
            return file_path.read_text(encoding="utf-8", errors="replace")

        if suffix == ".json":
            return file_path.read_text(encoding="utf-8", errors="replace")

        if suffix == ".xml":
            return file_path.read_text(encoding="utf-8", errors="replace")

        if suffix in (".html", ".htm"):
            return self._extract_html(file_path)

        if suffix == ".pdf":
            return await self._extract_pdf(file_path)

        if suffix == ".docx":
            return await self._extract_docx(file_path)

        raise ValueError(f"Kein Extraktor für {suffix}")

    def _extract_html(self, file_path: Path) -> str:
        """Extrahiert Text aus HTML (Tags strippen)."""
        from bs4 import BeautifulSoup

        content = file_path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(content, "html.parser")

        # Script und Style komplett entfernen
        for tag in soup(["script", "style"]):
            tag.decompose()

        # Text extrahieren und Whitespace normalisieren
        text = soup.get_text(separator=" ", strip=True)
        return " ".join(text.split())

    async def _extract_pdf(self, file_path: Path) -> str:
        """Extrahiert Text aus PDF via MediaPipeline."""
        if self._media_pipeline:
            result = await self._media_pipeline.extract_text(str(file_path))
            if result.success:
                return cast("str", result.text)

            raise OSError(f"PDF-Extraktion fehlgeschlagen: {result.error}")

        # Fallback: PyMuPDF direkt (blocking I/O → thread)
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise OSError("PDF-Extraktion benötigt PyMuPDF: pip install pymupdf") from exc

        def _read_pdf() -> str:
            doc = fitz.open(str(file_path))
            texts = [page.get_text() for page in doc]
            doc.close()
            return "\n\n".join(texts)

        return await asyncio.to_thread(_read_pdf)

    async def _extract_docx(self, file_path: Path) -> str:
        """Extrahiert Text aus DOCX via MediaPipeline."""
        if self._media_pipeline:
            result = await self._media_pipeline.extract_text(str(file_path))
            if result.success:
                return cast("str", result.text)

            raise OSError(f"DOCX-Extraktion fehlgeschlagen: {result.error}")

        # Fallback: python-docx direkt (blocking I/O → thread)
        try:
            import docx
        except ImportError as exc:
            raise OSError("DOCX-Extraktion benötigt python-docx: pip install python-docx") from exc

        def _read_docx() -> str:
            doc = docx.Document(str(file_path))
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())

        return await asyncio.to_thread(_read_docx)


# ============================================================================
# Ingest Pipeline
# ============================================================================


class IngestPipeline:
    """Verarbeitet Dateien und fuegt sie der Knowledge Base hinzu.

    Usage:
        pipeline = IngestPipeline(config, memory_manager)

        # Einzelne Datei verarbeiten:
        result = await pipeline.ingest_file(Path("dokument.pdf"))

        # Verzeichnis scannen und alles verarbeiten:
        results = await pipeline.scan_and_ingest()

        # Kontinuierlich ueberwachen (blocking):
        await pipeline.watch()
    """

    def __init__(
        self,
        config: IngestConfig | None = None,
        memory_manager: Any | None = None,
    ) -> None:
        self._config = config or IngestConfig()
        self._memory = memory_manager
        self._extractor = TextExtractor()
        self._processed_hashes: set[str] = set()
        # PASS-3: protect the dedup-set check-then-add against parallel
        # ingest_file() calls (the gateway can call this concurrently
        # while the watch loop is also polling). Without the lock, two
        # callers with the same content_hash both pass the membership
        # test, both index the file, and the second shutil.move on the
        # same file raises FileNotFoundError that the bare except
        # silently swallows.
        self._hash_lock = asyncio.Lock()
        self._running = False

        # Verzeichnisse sicherstellen
        self._config.watch_dir.mkdir(parents=True, exist_ok=True)
        self._config.processed_dir.mkdir(parents=True, exist_ok=True)
        self._config.failed_dir.mkdir(parents=True, exist_ok=True)

        # Bereits verarbeitete Dateien laden
        self._load_processed_hashes()

    def _load_processed_hashes(self) -> None:
        """Laedt Hashes bereits verarbeiteter Dateien."""
        for f in self._config.processed_dir.iterdir():
            if f.is_file():
                self._processed_hashes.add(f.stem.split("_", 1)[0])

    @staticmethod
    def _file_hash(file_path: Path) -> str:
        """Compute a stable hash for a file."""
        hasher = hashlib.sha256()
        hasher.update(file_path.name.encode())
        hasher.update(str(file_path.stat().st_size).encode())
        # Erste 4096 Bytes fuer schnelle Erkennung
        with open(file_path, "rb") as f:
            hasher.update(f.read(4096))
        return hasher.hexdigest()[:16]

    # ========================================================================
    # Einzeldatei-Verarbeitung
    # ========================================================================

    async def ingest_file(self, file_path: Path) -> IngestResult:
        """Verarbeitet eine einzelne Datei.

        1. Validierung (Groesse, Format)
        2. Text-Extraktion
        3. Memory-Indexierung
        4. Verschieben nach processed/

        Args:
            file_path: Pfad zur Datei.

        Returns:
            IngestResult mit Statistiken.
        """
        start = time.monotonic()
        file_name = file_path.name

        log.info("ingest_start", file=file_name)

        # Validierung
        if not file_path.exists():
            return IngestResult(
                file_path=str(file_path),
                file_name=file_name,
                error="Datei nicht gefunden",
            )

        file_size = file_path.stat().st_size
        log.debug("ingest_file_size", file=file_name, size=file_size)

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return IngestResult(
                file_path=str(file_path),
                file_name=file_name,
                error=f"Nicht unterstütztes Format: {file_path.suffix}",
            )

        if file_size > self._config.max_file_size_bytes:
            return IngestResult(
                file_path=str(file_path),
                file_name=file_name,
                error=f"Datei zu groß: {file_size / 1024 / 1024:.1f} MB",
            )

        # Duplikat-Erkennung — protected by ``_hash_lock`` for the full
        # check-then-claim pattern. Reserve the hash atomically so a
        # concurrent caller racing on the same file gets the dedup hit.
        # On failure paths below we discard the reservation.
        content_hash = self._file_hash(file_path)
        async with self._hash_lock:
            if content_hash in self._processed_hashes:
                return IngestResult(
                    file_path=str(file_path),
                    file_name=file_name,
                    error="Bereits verarbeitet (Hash bekannt)",
                    content_hash=content_hash,
                )
            # Reserve the hash up-front. If the indexing pipeline below
            # fails we discard the reservation so the caller can retry.
            self._processed_hashes.add(content_hash)
        _hash_reserved = True

        try:
            # Text extrahieren
            text = await self._extractor.extract(file_path)

            if not text.strip():
                return IngestResult(
                    file_path=str(file_path),
                    file_name=file_name,
                    error="Kein Text extrahiert",
                )

            # In Memory indexieren
            chunks_created = 0
            if self._memory is not None:
                source_path = f"ingest://{file_name}"

                if self._config.generate_embeddings and hasattr(
                    self._memory, "index_with_embeddings"
                ):
                    chunks_created = await self._memory.index_with_embeddings(file_path)
                elif hasattr(self._memory, "index_text"):
                    chunks_created = self._memory.index_text(text, source_path)
                else:
                    log.warning("memory_manager_missing_index_method")
            else:
                # Ohne MemoryManager: nur extrahieren, nicht speichern
                # Chunks schaetzen (1 Chunk pro ~512 Tokens ≈ ~2000 Zeichen)
                chunks_created = max(1, len(text) // 2000)

            # Erfolgreich → nach processed/ verschieben.
            # Reservation already done up-front (PASS-3 race fix).
            self._move_to_processed(file_path, content_hash)

            duration = time.monotonic() - start

            result = IngestResult(
                file_path=str(file_path),
                file_name=file_name,
                success=True,
                chunks_created=chunks_created,
                text_length=len(text),
                duration_seconds=round(duration, 2),
                content_hash=content_hash,
            )

            log.info(
                "ingest_success",
                file=file_name,
                chunks=chunks_created,
                text_len=len(text),
                duration=round(duration, 2),
            )

            return result

        except Exception as exc:
            # Fehler → nach failed/ verschieben + Reservation freigeben,
            # damit der Aufrufer retry-en kann (sonst Geister-Hash blockt).
            if _hash_reserved:
                async with self._hash_lock:
                    self._processed_hashes.discard(content_hash)
            self._move_to_failed(file_path)
            duration = time.monotonic() - start

            log.error("ingest_error", file=file_name, error=str(exc))

            return IngestResult(
                file_path=str(file_path),
                file_name=file_name,
                error=str(exc),
                duration_seconds=round(duration, 2),
            )

    def _move_to_processed(self, file_path: Path, content_hash: str) -> None:
        """Verschiebt eine verarbeitete Datei nach processed/."""
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        dest = self._config.processed_dir / f"{content_hash}_{timestamp}_{file_path.name}"
        try:
            shutil.move(str(file_path), str(dest))
        except Exception as exc:
            log.warning("move_to_processed_failed", error=str(exc))

    def _move_to_failed(self, file_path: Path) -> None:
        """Verschiebt eine fehlgeschlagene Datei nach failed/."""
        dest = self._config.failed_dir / file_path.name
        try:
            shutil.move(str(file_path), str(dest))
        except Exception as exc:
            log.warning("move_to_failed_failed", error=str(exc))

    # ========================================================================
    # Batch-Verarbeitung
    # ========================================================================

    async def scan_and_ingest(self) -> list[IngestResult]:
        """Scannt das Watch-Verzeichnis und verarbeitet alle neuen Dateien.

        Returns:
            Liste von IngestResults.
        """
        results: list[IngestResult] = []

        if not self._config.watch_dir.exists():
            return results

        files = sorted(
            f
            for f in self._config.watch_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not files:
            return results

        log.info("ingest_scan", files_found=len(files))

        for file_path in files:
            result = await self.ingest_file(file_path)
            results.append(result)

        success_count = sum(1 for r in results if r.success)
        total_chunks = sum(r.chunks_created for r in results)

        log.info(
            "ingest_scan_complete",
            total=len(results),
            success=success_count,
            chunks=total_chunks,
        )

        return results

    # ========================================================================
    # Continuous Watch
    # ========================================================================

    async def watch(self) -> None:
        """Ueberwacht das Ingest-Verzeichnis kontinuierlich.

        Pollt in regelmaessigen Abstaenden. Kann mit stop() beendet werden.
        """
        self._running = True
        interval = self._config.poll_interval_seconds

        log.info(
            "ingest_watch_started",
            watch_dir=str(self._config.watch_dir),
            interval=interval,
        )

        while self._running:
            try:
                await self.scan_and_ingest()
            except Exception as exc:
                log.error("ingest_watch_error", error=str(exc))

            await asyncio.sleep(interval)

        log.info("ingest_watch_stopped")

    def stop(self) -> None:
        """Stoppt den Watch-Loop."""
        self._running = False

    # ========================================================================
    # Statistiken
    # ========================================================================

    def stats(self) -> dict[str, Any]:
        """Pipeline-Statistiken."""
        pending = 0
        processed = 0
        failed = 0

        if self._config.watch_dir.exists():
            pending = sum(
                1
                for f in self._config.watch_dir.iterdir()
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        if self._config.processed_dir.exists():
            processed = sum(1 for f in self._config.processed_dir.iterdir() if f.is_file())
        if self._config.failed_dir.exists():
            failed = sum(1 for f in self._config.failed_dir.iterdir() if f.is_file())

        return {
            "watch_dir": str(self._config.watch_dir),
            "pending": pending,
            "processed": processed,
            "failed": failed,
            "known_hashes": len(self._processed_hashes),
            "supported_formats": sorted(SUPPORTED_EXTENSIONS),
        }
