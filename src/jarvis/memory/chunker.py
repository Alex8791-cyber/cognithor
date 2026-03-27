"""Sliding-window chunker with Markdown awareness. [B§4.8]

Teilt Markdown-Dateien in überlappende Chunks auf.
Never breaks in the middle of a line.
Bevorzugt Markdown-Überschriften am Chunk-Anfang.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime

from jarvis.config import MemoryConfig
from jarvis.models import Chunk, MemoryTier

# ── Token Estimation ──────────────────────────────────────────

#
# Language-dependent heuristic: German texts have longer words
# and compound nouns (e.g. "Berufsunfähigkeitsversicherung") that
# BPE tokenizers split into more tokens than a simple
# chars/4 approach would suggest.
#
# Benchmark (GPT-4 / Llama Tokenizer):
# German:   1 Token ≈ 3.2 chars (average mixed text)
# English:  1 Token ≈ 4.0 chars
#
# We use a word-based estimation as fallback for
# mixed-language texts, as it is more robust than chars/N.

# Pattern for German compound noun detection (>=16 chars, typically German)
_COMPOUND_RE = re.compile(r"\b[A-ZÄÖÜa-zäöüß]{16,}\b")

# Markdown header pattern
_HEADER_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)

# Date pattern in filenames like 2026-02-21.md
_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def _estimate_tokens(text: str) -> int:
    """Sprachbewusste Token-Schätzung.

    Kombiniert zwei Heuristiken:
      1. Wort-basiert: Jedes Wort ≈ 1.3 Tokens (BPE-Durchschnitt)
      2. Komposita-Korrektur: Lange Wörter (≥16 Zeichen) werden
         typischerweise in 3-6 Tokens zerlegt statt 1-2

    Genauer als chars/4, besonders für deutsche Texte mit
    Komposita wie "Berufsunfähigkeitsversicherung" (≈8 Tokens)
    oder "Haftpflichtversicherungsgesellschaft" (≈7 Tokens).

    Returns:
        Geschätzte Anzahl Tokens (Minimum 1).
    """
    if not text:
        return 1

    words = text.split()
    word_count = len(words)

    if word_count == 0:
        return 1

    # Base estimate: each word ≈ 1.3 tokens
    base_tokens = int(word_count * 1.3)

    # Compound noun correction: Long words produce more tokens
    # than the base estimate assumes
    compounds = _COMPOUND_RE.findall(text)
    compound_extra = 0
    for compound in compounds:
        # A 20-char word ≈ 5 tokens instead of 1.3
        estimated_subtokens = len(compound) // 4
        compound_extra += max(0, estimated_subtokens - 1)  # -1 because base already counts 1.3

    return max(1, base_tokens + compound_extra)


def _content_hash(text: str) -> str:
    """SHA-256 Hash für Embedding-Cache."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_date_from_path(path: str) -> datetime | None:
    """Extract date from file path (z.B. episodes/2026-02-21.md)."""
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
    """Detect the memory tier based on the file path."""
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
        text: The text to split.
        source_path: Source file path.
        chunk_size_tokens: Maximale Chunk-Größe in Tokens.
        chunk_overlap_tokens: Überlappung zwischen Chunks in Tokens.
        tier: Explicit memory tier (otherwise derived from path).

    Returns:
        List of Chunk objects.
    """
    if not text or not text.strip():
        return []

    lines = text.split("\n")
    memory_tier = tier if tier is not None else _detect_tier(source_path)
    timestamp = _extract_date_from_path(source_path)
    header_positions = _find_header_positions(lines)

    # Chars-per-Token ratio for chunk boundary calculation
    # German: ~3.2 chars/token (more conservative than English ~4.0)
    # Conservative (3.2) produces slightly smaller chunks, which is
    # better for retrieval quality than chunks that are too large
    _CHARS_PER_TOKEN_RATIO = 3.2
    chunk_size_chars = int(chunk_size_tokens * _CHARS_PER_TOKEN_RATIO)
    overlap_chars = int(chunk_overlap_tokens * _CHARS_PER_TOKEN_RATIO)

    chunks: list[Chunk] = []
    current_lines: list[str] = []
    current_chars = 0
    chunk_start_line = 0

    def _flush(end_line: int) -> None:
        """Save current buffer as chunk."""
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

        # Would the current line exceed the chunk size?
        if current_chars + line_chars > chunk_size_chars and current_lines:
            # Finalize chunk
            _flush(i - 1)

            # Calculate overlap: Keep last N chars as overlap
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

        # If current line is a header AND we already have content,
        # start new chunk at header (Header-Aware Splitting)
        if i in header_positions and current_lines and current_chars > overlap_chars * 2:
            _flush(i - 1)
            current_lines = []
            current_chars = 0
            chunk_start_line = i

        current_lines.append(line)
        current_chars += line_chars

    # Flush the last chunk
    if current_lines:
        _flush(len(lines) - 1)

    return chunks


def chunk_file(
    source_path: str,
    *,
    config: MemoryConfig | None = None,
    tier: MemoryTier | None = None,
) -> list[Chunk]:
    """Read a file and split it into chunks.

    Args:
        source_path: Path to the Markdown file.
        config: Memory configuration (optional).
        tier: Expliziter Memory-Tier.

    Returns:
        List of Chunk objects.

    Raises:
        FileNotFoundError: If the file does not exist.
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
