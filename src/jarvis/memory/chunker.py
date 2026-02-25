"""Sliding-Window Chunker mit Markdown-Awareness. [B§4.8]

Teilt Markdown-Dateien in überlappende Chunks auf.
Bricht nie mitten in einer Zeile.
Bevorzugt Markdown-Überschriften am Chunk-Anfang.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime

from jarvis.config import MemoryConfig
from jarvis.models import Chunk, MemoryTier

# Approximation: 1 Token ≈ 4 Zeichen (für Deutsch etwas konservativer)
CHARS_PER_TOKEN = 4

# Markdown header pattern
_HEADER_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)

# Date pattern in filenames like 2026-02-21.md
_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def _estimate_tokens(text: str) -> int:
    """Grobe Token-Schätzung. 1 Token ≈ 4 Zeichen."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def _content_hash(text: str) -> str:
    """SHA-256 Hash für Embedding-Cache."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_date_from_path(path: str) -> datetime | None:
    """Extrahiert Datum aus Dateipfad (z.B. episodes/2026-02-21.md)."""
    match = _DATE_RE.search(path)
    if match:
        try:
            return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return None
    return None


def _find_header_positions(lines: list[str]) -> set[int]:
    """Findet Zeilen-Indizes die Markdown-Überschriften sind."""
    return {i for i, line in enumerate(lines) if _HEADER_RE.match(line)}


def _detect_tier(source_path: str) -> MemoryTier:
    """Erkennt den Memory-Tier anhand des Dateipfads."""
    path_lower = source_path.lower().replace("\\", "/")
    if "core.md" in path_lower or "/core" in path_lower:
        return MemoryTier.CORE
    if "/episodes/" in path_lower or path_lower.startswith("episodes/"):
        return MemoryTier.EPISODIC
    if "/procedures/" in path_lower or path_lower.startswith("procedures/"):
        return MemoryTier.PROCEDURAL
    if "/knowledge/" in path_lower or "/semantic/" in path_lower:
        return MemoryTier.SEMANTIC
    return MemoryTier.SEMANTIC  # Default


def chunk_text(
    text: str,
    source_path: str,
    *,
    chunk_size_tokens: int = 400,
    chunk_overlap_tokens: int = 80,
    tier: MemoryTier | None = None,
) -> list[Chunk]:
    """Teilt Text in überlappende Chunks auf.

    Args:
        text: Der zu teilende Text.
        source_path: Quell-Dateipfad.
        chunk_size_tokens: Maximale Chunk-Größe in Tokens.
        chunk_overlap_tokens: Überlappung zwischen Chunks in Tokens.
        tier: Expliziter Memory-Tier (wird sonst aus Pfad abgeleitet).

    Returns:
        Liste von Chunk-Objekten.
    """
    if not text or not text.strip():
        return []

    lines = text.split("\n")
    memory_tier = tier if tier is not None else _detect_tier(source_path)
    timestamp = _extract_date_from_path(source_path)
    header_positions = _find_header_positions(lines)

    chunk_size_chars = chunk_size_tokens * CHARS_PER_TOKEN
    overlap_chars = chunk_overlap_tokens * CHARS_PER_TOKEN

    chunks: list[Chunk] = []
    current_lines: list[str] = []
    current_chars = 0
    chunk_start_line = 0

    def _flush(end_line: int) -> None:
        """Aktuellen Buffer als Chunk speichern."""
        nonlocal current_lines, current_chars, chunk_start_line
        if not current_lines:
            return

        chunk_text = "\n".join(current_lines)
        if not chunk_text.strip():
            current_lines = []
            current_chars = 0
            return

        chunks.append(
            Chunk(
                text=chunk_text,
                source_path=source_path,
                line_start=chunk_start_line,
                line_end=end_line,
                content_hash=_content_hash(chunk_text),
                memory_tier=memory_tier,
                timestamp=timestamp,
                token_count=_estimate_tokens(chunk_text),
            )
        )

    for i, line in enumerate(lines):
        line_chars = len(line) + 1  # +1 für \n

        # Würde die aktuelle Zeile den Chunk überschreiten?
        if current_chars + line_chars > chunk_size_chars and current_lines:
            # Chunk abschließen
            _flush(i - 1)

            # Overlap berechnen: Letzte N Zeichen als Überlappung behalten
            if overlap_chars > 0 and current_lines:
                overlap_lines: list[str] = []
                overlap_count = 0
                for prev_line in reversed(current_lines):
                    if overlap_count + len(prev_line) + 1 > overlap_chars:
                        break
                    overlap_lines.insert(0, prev_line)
                    overlap_count += len(prev_line) + 1

                current_lines = overlap_lines
                current_chars = overlap_count
                chunk_start_line = i - len(overlap_lines)
            else:
                current_lines = []
                current_chars = 0
                chunk_start_line = i

        # Wenn aktuelle Zeile ein Header ist UND wir schon Content haben,
        # starte neuen Chunk am Header (Header-Aware Splitting)
        if i in header_positions and current_lines and current_chars > overlap_chars * 2:
            _flush(i - 1)
            current_lines = []
            current_chars = 0
            chunk_start_line = i

        current_lines.append(line)
        current_chars += line_chars

    # Letzten Chunk flushen
    if current_lines:
        _flush(len(lines) - 1)

    return chunks


def chunk_file(
    source_path: str,
    *,
    config: MemoryConfig | None = None,
    tier: MemoryTier | None = None,
) -> list[Chunk]:
    """Liest eine Datei und teilt sie in Chunks auf.

    Args:
        source_path: Pfad zur Markdown-Datei.
        config: Memory-Konfiguration (optional).
        tier: Expliziter Memory-Tier.

    Returns:
        Liste von Chunk-Objekten.

    Raises:
        FileNotFoundError: Wenn die Datei nicht existiert.
    """
    with open(source_path, encoding="utf-8") as f:
        text = f.read()

    if config is None:
        config = MemoryConfig()

    return chunk_text(
        text,
        source_path,
        chunk_size_tokens=config.chunk_size_tokens,
        chunk_overlap_tokens=config.chunk_overlap_tokens,
        tier=tier,
    )
