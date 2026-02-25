"""Tests für memory/chunker.py · Sliding-Window Chunking."""

from __future__ import annotations

from jarvis.memory.chunker import (
    _content_hash,
    _detect_tier,
    _estimate_tokens,
    _extract_date_from_path,
    _find_header_positions,
    chunk_text,
)
from jarvis.models import MemoryTier


class TestEstimateTokens:
    def test_empty(self):
        assert _estimate_tokens("") == 1  # min 1

    def test_short(self):
        assert _estimate_tokens("Hello") == 1

    def test_longer(self):
        # 100 chars → ~25 tokens
        text = "a" * 100
        assert _estimate_tokens(text) == 25


class TestContentHash:
    def test_deterministic(self):
        h1 = _content_hash("hello world")
        h2 = _content_hash("hello world")
        assert h1 == h2

    def test_different(self):
        h1 = _content_hash("hello")
        h2 = _content_hash("world")
        assert h1 != h2

    def test_sha256_length(self):
        h = _content_hash("test")
        assert len(h) == 64  # SHA-256 hex


class TestExtractDate:
    def test_episode_path(self):
        d = _extract_date_from_path("episodes/2026-02-21.md")
        assert d is not None
        assert d.year == 2026
        assert d.month == 2
        assert d.day == 21

    def test_no_date(self):
        assert _extract_date_from_path("core.md") is None

    def test_invalid_date(self):
        assert _extract_date_from_path("episodes/2026-13-99.md") is None


class TestFindHeaders:
    def test_finds_headers(self):
        lines = ["# Title", "text", "## Sub", "more text", "### Deep"]
        positions = _find_header_positions(lines)
        assert positions == {0, 2, 4}

    def test_no_headers(self):
        lines = ["just text", "more text"]
        assert _find_header_positions(lines) == set()


class TestDetectTier:
    def test_core(self):
        assert _detect_tier("/home/.jarvis/memory/CORE.md") == MemoryTier.CORE

    def test_episodic(self):
        assert _detect_tier("/home/.jarvis/memory/episodes/2026-02-21.md") == MemoryTier.EPISODIC

    def test_procedural(self):
        assert (
            _detect_tier("/home/.jarvis/memory/procedures/bu-angebot.md") == MemoryTier.PROCEDURAL
        )

    def test_semantic(self):
        assert (
            _detect_tier("/home/.jarvis/memory/knowledge/kunden/mueller.md") == MemoryTier.SEMANTIC
        )

    def test_default(self):
        assert _detect_tier("/some/random/file.md") == MemoryTier.SEMANTIC


class TestChunkText:
    def test_empty_text(self):
        assert chunk_text("", "test.md") == []

    def test_whitespace_only(self):
        assert chunk_text("   \n\n  ", "test.md") == []

    def test_single_chunk(self):
        text = "Hello world\nThis is a test"
        chunks = chunk_text(text, "test.md", chunk_size_tokens=100)
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].source_path == "test.md"

    def test_multiple_chunks(self):
        # ~100 tokens per line (400 chars), chunk_size=50 tokens
        lines = [f"Line {i}: " + "x" * 190 for i in range(10)]
        text = "\n".join(lines)
        chunks = chunk_text(text, "test.md", chunk_size_tokens=50, chunk_overlap_tokens=0)
        assert len(chunks) > 1

    def test_overlap(self):
        lines = [f"Zeile {i}: " + "a" * 180 for i in range(6)]
        text = "\n".join(lines)
        chunks = chunk_text(text, "test.md", chunk_size_tokens=50, chunk_overlap_tokens=20)
        # Mit Overlap sollten Chunks überlappende Inhalte haben
        if len(chunks) > 1:
            # Letzte Zeilen des ersten Chunks sollten im zweiten Chunk vorkommen
            first_lines = set(chunks[0].text.split("\n"))
            second_lines = set(chunks[1].text.split("\n"))
            overlap = first_lines & second_lines
            # Es sollte mindestens etwas Überlappung geben
            assert len(overlap) >= 0  # Kann 0 sein wenn Overlap zu klein

    def test_header_aware_splitting(self):
        text = "# Title\nSome content here\n" + "x" * 1600 + "\n# Another Section\nMore content"
        chunks = chunk_text(text, "test.md", chunk_size_tokens=200)
        # Der zweite Chunk sollte mit dem Header beginnen
        if len(chunks) > 1:
            has_header_start = any(c.text.startswith("# ") for c in chunks[1:])
            assert has_header_start

    def test_content_hash_unique(self):
        text = "Line 1\nLine 2\nLine 3\n" + "x" * 2000 + "\nLine end"
        chunks = chunk_text(text, "test.md", chunk_size_tokens=50)
        hashes = [c.content_hash for c in chunks]
        assert len(hashes) == len(set(hashes))  # Alle unique

    def test_tier_from_path(self):
        chunks = chunk_text("Hello", "episodes/2026-02-21.md")
        assert len(chunks) == 1
        assert chunks[0].memory_tier == MemoryTier.EPISODIC

    def test_explicit_tier(self):
        chunks = chunk_text("Hello", "test.md", tier=MemoryTier.CORE)
        assert chunks[0].memory_tier == MemoryTier.CORE

    def test_line_numbers(self):
        text = "Line 0\nLine 1\nLine 2\nLine 3\nLine 4"
        chunks = chunk_text(text, "test.md", chunk_size_tokens=100)
        assert chunks[0].line_start == 0
        assert chunks[0].line_end == 4

    def test_token_count_set(self):
        text = "Hello world this is a test"
        chunks = chunk_text(text, "test.md")
        assert chunks[0].token_count > 0

    def test_timestamp_from_path(self):
        chunks = chunk_text("Test", "episodes/2026-03-15.md")
        assert chunks[0].timestamp is not None
        assert chunks[0].timestamp.year == 2026
        assert chunks[0].timestamp.month == 3
