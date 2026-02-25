"""Multimodal Memory: Medien-Embeddings für Bilder, Audio und Dokumente.

Erweitert das Memory-System um multimodale Suche:
  - Bilder → Beschreibung → Text-Embedding → durchsuchbar
  - Audio → Transkript → Text-Embedding → durchsuchbar
  - Dokumente → Extrahierter Text → Chunks → Embeddings

Cross-modale Suche: Eine Textfrage findet relevante Bilder,
Audio-Clips und Dokumente über ihre textuelle Repräsentation.

Architektur:
  MediaPipeline.analyze_image() → Beschreibung
  MediaPipeline.transcribe_audio() → Transkript
  MediaPipeline.extract_text() → Dokumenttext
       ↓
  MultimodalMemory.ingest_media() → Chunk + Embedding → Index
       ↓
  HybridSearch/EnhancedSearch findet diese Chunks normal

Bibel-Referenz: §4.9 (Multimodal Memory)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from jarvis.models import Chunk, MemoryTier

logger = logging.getLogger("jarvis.memory.multimodal")


# ============================================================================
# Datenmodelle
# ============================================================================


class MediaType(Enum):
    """Unterstützte Medientypen."""

    IMAGE = "image"
    AUDIO = "audio"
    DOCUMENT = "document"
    VIDEO = "video"  # Zukunft


@dataclass
class MediaAsset:
    """Ein indexiertes Medien-Asset.

    Speichert die Verknüpfung zwischen Original-Datei,
    textueller Repräsentation und den daraus erzeugten Chunks.
    """

    id: str
    media_type: MediaType
    file_path: str
    file_hash: str  # SHA-256 der Originaldatei

    # Textuelle Repräsentation
    text_representation: str = ""  # Beschreibung, Transkript oder extrahierter Text
    extraction_method: str = ""  # "llava", "whisper", "pymupdf" etc.

    # Verknüpfung zu Memory-Chunks
    chunk_ids: list[str] = field(default_factory=list)

    # Metadaten
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )

    @property
    def filename(self) -> str:
        return Path(self.file_path).name


# ============================================================================
# MIME-Type Erkennung
# ============================================================================

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".svg"}
_AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm", ".aac"}
_DOC_EXTS = {".pdf", ".docx", ".txt", ".md", ".html", ".htm", ".csv", ".json", ".xml"}


def detect_media_type(file_path: str | Path) -> MediaType | None:
    """Erkennt den Medientyp anhand der Dateiendung.

    Returns:
        MediaType oder None wenn nicht unterstützt.
    """
    suffix = Path(file_path).suffix.lower()
    if suffix in _IMAGE_EXTS:
        return MediaType.IMAGE
    if suffix in _AUDIO_EXTS:
        return MediaType.AUDIO
    if suffix in _DOC_EXTS:
        return MediaType.DOCUMENT
    return None


def file_hash(file_path: str | Path) -> str:
    """Berechnet SHA-256 Hash einer Datei."""
    h = hashlib.sha256()
    path = Path(file_path)
    if not path.exists():
        return ""
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# ============================================================================
# Multimodal Memory
# ============================================================================


class MultimodalMemory:
    """Verwaltet multimodale Medien-Assets im Memory-System.

    Workflow:
      1. Media-Datei empfangen
      2. Text extrahieren (via MediaPipeline oder direkt)
      3. Text chunken + in Memory-Index speichern
      4. Asset-Metadaten tracken
      5. Cross-modale Suche über normale HybridSearch

    Args:
        memory_manager: MemoryManager für Index + Embedding-Zugriff.
        media_pipeline: MediaPipeline für Bild/Audio/Dokument-Verarbeitung.
    """

    def __init__(
        self,
        memory_manager: Any = None,
        media_pipeline: Any = None,
    ) -> None:
        self._manager = memory_manager
        self._pipeline = media_pipeline
        self._assets: dict[str, MediaAsset] = {}  # asset_id → MediaAsset
        self._file_hash_map: dict[str, str] = {}  # file_hash → asset_id (Dedup)

    @property
    def asset_count(self) -> int:
        return len(self._assets)

    def get_asset(self, asset_id: str) -> MediaAsset | None:
        return self._assets.get(asset_id)

    def get_assets_by_type(self, media_type: MediaType) -> list[MediaAsset]:
        return [a for a in self._assets.values() if a.media_type == media_type]

    def list_assets(self) -> list[MediaAsset]:
        return list(self._assets.values())

    # ========================================================================
    # Ingest: Medien in Memory aufnehmen
    # ========================================================================

    async def ingest_media(
        self,
        file_path: str | Path,
        *,
        text_override: str = "",
        media_type: MediaType | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MediaAsset | None:
        """Nimmt eine Mediendatei in das Memory-System auf.

        1. Medientyp erkennen
        2. Text extrahieren (oder Override verwenden)
        3. Text chunken + indexieren
        4. Optional Embeddings erzeugen
        5. Asset tracken

        Args:
            file_path: Pfad zur Mediendatei.
            text_override: Wenn gesetzt, wird dieser Text statt Extraktion verwendet.
            media_type: Expliziter Medientyp (sonst Auto-Detect).
            metadata: Zusätzliche Metadaten.

        Returns:
            MediaAsset oder None bei Fehler.
        """
        path = Path(file_path)
        if not path.exists():
            logger.warning("media_file_not_found: %s", file_path)
            return None

        # Medientyp erkennen
        mtype = media_type or detect_media_type(path)
        if mtype is None:
            logger.warning("unsupported_media_type: %s", path.suffix)
            return None

        # Duplikat-Check via File-Hash
        fhash = file_hash(path)
        if fhash in self._file_hash_map:
            existing = self._assets.get(self._file_hash_map[fhash])
            if existing:
                logger.info("media_duplicate_skipped: %s", path.name)
                return existing

        # Text extrahieren
        text = text_override
        extraction_method = "manual" if text_override else ""

        if not text:
            text, extraction_method = await self._extract_text(path, mtype)

        if not text:
            logger.warning("media_no_text_extracted: %s", path.name)
            return None

        # Asset erstellen
        asset_id = hashlib.sha256(f"{fhash}:{path.name}".encode()).hexdigest()[:16]
        asset = MediaAsset(
            id=asset_id,
            media_type=mtype,
            file_path=str(path.resolve()),
            file_hash=fhash,
            text_representation=text,
            extraction_method=extraction_method,
            metadata=metadata or {},
        )

        # Text in Memory indexieren
        chunk_ids = await self._index_media_text(asset, text)
        asset.chunk_ids = chunk_ids

        # Asset tracken
        self._assets[asset_id] = asset
        self._file_hash_map[fhash] = asset_id

        logger.info(
            "media_ingested: asset=%s type=%s file=%s chunks=%d method=%s",
            asset_id, mtype.value, path.name, len(chunk_ids), extraction_method,
        )
        return asset

    async def ingest_directory(
        self,
        directory: str | Path,
        *,
        recursive: bool = True,
    ) -> list[MediaAsset]:
        """Nimmt alle Mediendateien eines Verzeichnisses auf.

        Args:
            directory: Verzeichnispfad.
            recursive: Auch Unterverzeichnisse durchsuchen.

        Returns:
            Liste der aufgenommenen Assets.
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            return []

        all_exts = _IMAGE_EXTS | _AUDIO_EXTS | _DOC_EXTS
        glob_fn = dir_path.rglob if recursive else dir_path.glob

        assets: list[MediaAsset] = []
        for f in sorted(glob_fn("*")):
            if f.is_file() and f.suffix.lower() in all_exts:
                asset = await self.ingest_media(f)
                if asset:
                    assets.append(asset)

        logger.info("directory_ingested: %s → %d assets", directory, len(assets))
        return assets

    # ========================================================================
    # Cross-modale Suche
    # ========================================================================

    def search_media(
        self,
        query: str,
        *,
        media_type: MediaType | None = None,
        top_k: int = 5,
    ) -> list[tuple[MediaAsset, float]]:
        """Sucht in Media-Assets via BM25 über deren Text-Repräsentation.

        Schnelle, synchrone Suche ohne Embedding-Server.

        Args:
            query: Suchtext.
            media_type: Optional nur bestimmten Medientyp durchsuchen.
            top_k: Max Ergebnisse.

        Returns:
            [(MediaAsset, score)] sortiert nach Relevanz.
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored: list[tuple[MediaAsset, float]] = []
        for asset in self._assets.values():
            if media_type and asset.media_type != media_type:
                continue

            # Einfacher Wort-Overlap Score
            text_lower = asset.text_representation.lower()
            text_words = set(text_lower.split())
            overlap = len(query_words & text_words)

            if overlap == 0:
                continue

            # Score: Overlap-Ratio + Keyword-Dichte
            score = overlap / max(len(query_words), 1)

            # Bonus wenn Query-Wörter im Dateinamen
            if any(w in asset.filename.lower() for w in query_words):
                score += 0.2

            scored.append((asset, min(score, 1.0)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ========================================================================
    # Text-Extraktion (delegiert an MediaPipeline)
    # ========================================================================

    async def _extract_text(
        self,
        path: Path,
        media_type: MediaType,
    ) -> tuple[str, str]:
        """Extrahiert Text aus einer Mediendatei.

        Returns:
            (text, extraction_method) Tupel.
        """
        if self._pipeline is None:
            # Ohne Pipeline: Nur Textdateien lesen
            if media_type == MediaType.DOCUMENT and path.suffix.lower() in {
                ".txt", ".md", ".csv", ".json", ".xml",
            }:
                text = path.read_text(encoding="utf-8", errors="replace")
                return text[:10000], "direct_read"
            return "", ""

        try:
            if media_type == MediaType.IMAGE:
                result = await self._pipeline.analyze_image(str(path))
                if result.success:
                    return result.text, f"image_analysis:{result.metadata.get('model', 'unknown')}"

            elif media_type == MediaType.AUDIO:
                result = await self._pipeline.transcribe_audio(str(path))
                if result.success:
                    return result.text, f"whisper:{result.metadata.get('model', 'base')}"

            elif media_type == MediaType.DOCUMENT:
                result = await self._pipeline.extract_text(str(path))
                if result.success:
                    return result.text, f"extract:{result.metadata.get('format', 'unknown')}"

        except Exception as exc:
            logger.warning("media_extraction_failed: %s — %s", path.name, exc)

        return "", ""

    # ========================================================================
    # Indexierung (delegiert an MemoryManager)
    # ========================================================================

    async def _index_media_text(
        self,
        asset: MediaAsset,
        text: str,
    ) -> list[str]:
        """Indexiert den extrahierten Text als Memory-Chunks.

        Returns:
            Liste der erzeugten Chunk-IDs.
        """
        if self._manager is None:
            # Ohne Manager: Nur IDs generieren
            chunk_id = hashlib.sha256(text[:200].encode()).hexdigest()[:16]
            return [chunk_id]

        source_path = f"media:{asset.media_type.value}:{asset.filename}"

        try:
            # index_text gibt Anzahl zurück, wir brauchen IDs
            count = self._manager.index_text(
                text,
                source_path,
                tier=MemoryTier.SEMANTIC,
            )

            # Chunk-IDs aus Index holen
            chunks = self._manager.index.get_chunks_by_source(source_path)
            return [c.id for c in chunks]

        except Exception as exc:
            logger.warning("media_indexing_failed: %s — %s", asset.filename, exc)
            return []

    # ========================================================================
    # Verwaltung
    # ========================================================================

    def remove_asset(self, asset_id: str) -> bool:
        """Entfernt ein Media-Asset und seine Chunks.

        Args:
            asset_id: Asset-ID.

        Returns:
            True wenn erfolgreich.
        """
        asset = self._assets.pop(asset_id, None)
        if not asset:
            return False

        # Chunks aus Index entfernen
        if self._manager:
            source_path = f"media:{asset.media_type.value}:{asset.filename}"
            self._manager.index.delete_chunks_by_source(source_path)

        # File-Hash Map aufräumen
        self._file_hash_map.pop(asset.file_hash, None)

        logger.info("media_asset_removed: asset=%s file=%s", asset_id, asset.filename)
        return True

    def stats(self) -> dict[str, Any]:
        """Multimodal Memory Statistiken."""
        by_type: dict[str, int] = {}
        total_chunks = 0
        for asset in self._assets.values():
            key = asset.media_type.value
            by_type[key] = by_type.get(key, 0) + 1
            total_chunks += len(asset.chunk_ids)

        return {
            "total_assets": len(self._assets),
            "by_type": by_type,
            "total_chunks": total_chunks,
            "unique_files": len(self._file_hash_map),
        }
