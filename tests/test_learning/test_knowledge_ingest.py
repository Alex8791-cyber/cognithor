"""Tests for Priority enum and IngestResult fields in knowledge_ingest.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognithor.learning.knowledge_ingest import (
    IngestResult,
    KnowledgeIngestService,
    Priority,
    _QueueItem,
)

# ---------------------------------------------------------------------------
# Priority enum
# ---------------------------------------------------------------------------


class TestPriorityEnum:
    def test_ordering_high_less_than_normal(self):
        assert Priority.HIGH < Priority.NORMAL

    def test_ordering_normal_less_than_low(self):
        assert Priority.NORMAL < Priority.LOW

    def test_ordering_high_less_than_low(self):
        assert Priority.HIGH < Priority.LOW

    def test_int_values(self):
        assert int(Priority.HIGH) == 0
        assert int(Priority.NORMAL) == 1
        assert int(Priority.LOW) == 2

    def test_from_string_high(self):
        assert Priority.from_string("high") == Priority.HIGH

    def test_from_string_normal(self):
        assert Priority.from_string("normal") == Priority.NORMAL

    def test_from_string_low(self):
        assert Priority.from_string("low") == Priority.LOW

    def test_from_string_uppercase(self):
        assert Priority.from_string("HIGH") == Priority.HIGH

    def test_from_string_mixed_case(self):
        assert Priority.from_string("Normal") == Priority.NORMAL

    def test_from_string_invalid_returns_normal(self):
        assert Priority.from_string("bogus") == Priority.NORMAL

    def test_from_string_empty_returns_normal(self):
        assert Priority.from_string("") == Priority.NORMAL


# ---------------------------------------------------------------------------
# IngestResult dataclass
# ---------------------------------------------------------------------------


class TestIngestResult:
    def _make(self, **kwargs) -> IngestResult:
        defaults = dict(id="test-id", source_type="file", source_name="test.txt", status="success")
        defaults.update(kwargs)
        return IngestResult(**defaults)

    def test_default_priority_is_normal(self):
        result = self._make()
        assert result.priority == Priority.NORMAL

    def test_explicit_priority_high(self):
        result = self._make(priority=Priority.HIGH)
        assert result.priority == Priority.HIGH

    def test_explicit_priority_low(self):
        result = self._make(priority=Priority.LOW)
        assert result.priority == Priority.LOW

    def test_chunks_alias_returns_chunks_created(self):
        result = self._make(chunks_created=42)
        assert result.chunks == 42

    def test_chunks_alias_zero(self):
        result = self._make(chunks_created=0)
        assert result.chunks == 0

    def test_chunks_alias_mirrors_chunks_created(self):
        result = self._make(chunks_created=7)
        result.chunks_created = 99
        assert result.chunks == 99

    def test_deep_learn_status_default_is_pending(self):
        result = self._make()
        assert result.deep_learn_status == "pending"

    def test_deep_learn_status_can_be_set(self):
        for status in ("queued", "skipped", "completed", "failed"):
            result = self._make(deep_learn_status=status)
            assert result.deep_learn_status == status


# ---------------------------------------------------------------------------
# IngestQueue
# ---------------------------------------------------------------------------


class TestIngestQueue:
    def test_enqueue_dequeue_priority_order(self):
        from cognithor.learning.knowledge_ingest import IngestQueue, _QueueItem

        q = IngestQueue()
        q.enqueue(
            _QueueItem(
                result_id="low1", text="low", source="f1.pdf", priority=Priority.LOW, page_images=[]
            )
        )
        q.enqueue(
            _QueueItem(
                result_id="high1",
                text="high",
                source="f2.pdf",
                priority=Priority.HIGH,
                page_images=[],
            )
        )
        q.enqueue(
            _QueueItem(
                result_id="norm1",
                text="norm",
                source="f3.pdf",
                priority=Priority.NORMAL,
                page_images=[],
            )
        )
        assert not q.empty
        assert q.dequeue().result_id == "high1"
        assert q.dequeue().result_id == "norm1"
        assert q.dequeue().result_id == "low1"
        assert q.empty

    def test_queue_size(self):
        from cognithor.learning.knowledge_ingest import IngestQueue, _QueueItem

        q = IngestQueue()
        assert len(q) == 0
        q.enqueue(
            _QueueItem(
                result_id="x", text="t", source="s", priority=Priority.NORMAL, page_images=[]
            )
        )
        assert len(q) == 1

    def test_pending_returns_sorted_list(self):
        from cognithor.learning.knowledge_ingest import IngestQueue, _QueueItem

        q = IngestQueue()
        q.enqueue(
            _QueueItem(
                result_id="a", text="t", source="s1", priority=Priority.NORMAL, page_images=[]
            )
        )
        q.enqueue(
            _QueueItem(result_id="b", text="t", source="s2", priority=Priority.HIGH, page_images=[])
        )
        pending = q.pending()
        assert len(pending) == 2
        assert pending[0]["id"] == "b"  # HIGH first
        assert pending[0]["priority"] == "HIGH"
        assert pending[1]["id"] == "a"


# ---------------------------------------------------------------------------
# Deep-learn integration (Task 3)
# ---------------------------------------------------------------------------


class TestDeepLearn:
    @pytest.mark.asyncio
    async def test_ingest_file_queues_deep_learn(self):
        memory = MagicMock()
        memory.index_text = MagicMock(return_value=3)
        svc = KnowledgeIngestService(memory=memory)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(svc, "_extract_text", AsyncMock(return_value="Hello world content"))
            mp.setattr(svc, "_ensure_worker", MagicMock())
            result = await svc.ingest_file(
                "test.txt",
                b"Hello world content",
                priority=Priority.NORMAL,
            )
        assert result.status == "success"
        assert result.chunks == 3
        assert result.deep_learn_status == "queued"
        assert len(svc._queue) == 1

    @pytest.mark.asyncio
    async def test_ingest_file_low_priority_skips(self):
        memory = MagicMock()
        memory.index_text = MagicMock(return_value=2)
        svc = KnowledgeIngestService(memory=memory)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(svc, "_extract_text", AsyncMock(return_value="Some text"))
            result = await svc.ingest_file("test.txt", b"Some text", priority=Priority.LOW)
        assert result.deep_learn_status == "skipped"
        assert len(svc._queue) == 0

    @pytest.mark.asyncio
    async def test_deep_learn_calls_builder(self):
        builder = AsyncMock()
        builder.build = AsyncMock()
        svc = KnowledgeIngestService(knowledge_builder=builder)
        svc._results.append(
            IngestResult(id="v1", source_type="file", source_name="test.pdf", status="success")
        )
        item = _QueueItem(
            result_id="v1",
            text="Original text",
            source="upload://test.pdf",
            priority=Priority.HIGH,
            page_images=[],
        )
        await svc._deep_learn(item)
        builder.build.assert_called_once()
        assert builder.build.call_args[0][0].text == "Original text"
        assert builder.build.call_args[0][0].source_type == "user_upload"

    @pytest.mark.asyncio
    async def test_deep_learn_skipped_without_builder(self):
        svc = KnowledgeIngestService()
        svc._results.append(
            IngestResult(id="v2", source_type="file", source_name="t.pdf", status="success")
        )
        item = _QueueItem(
            result_id="v2",
            text="text",
            source="upload://t.pdf",
            priority=Priority.NORMAL,
            page_images=[],
        )
        await svc._deep_learn(item)
        assert svc._results[0].deep_learn_status == "skipped"


# ---------------------------------------------------------------------------
# PDF Vision pipeline (Task 5)
# ---------------------------------------------------------------------------


class TestPdfVision:
    def test_extract_page_images_non_pdf(self):
        """Non-PDF files return empty list."""
        svc = KnowledgeIngestService()
        result = svc._extract_page_images(b"not a pdf", ".txt")
        assert result == []

    def test_extract_page_images_invalid_pdf(self):
        """Invalid PDF content returns empty list gracefully."""
        svc = KnowledgeIngestService()
        result = svc._extract_page_images(b"not a real pdf", ".pdf")
        assert result == []

    def test_extract_page_images_no_pypdf(self):
        """Graceful fallback when pypdf is not installed."""
        from unittest.mock import patch

        svc = KnowledgeIngestService()
        with patch.dict("sys.modules", {"pypdf": None}):
            result = svc._extract_page_images(b"%PDF-1.4", ".pdf")
            assert result == []

    @pytest.mark.asyncio
    async def test_deep_learn_calls_builder_with_correct_fetch_result(self):
        """Deep learn creates FetchResult with user_upload source_type."""
        builder = AsyncMock()
        builder.build = AsyncMock()
        svc = KnowledgeIngestService(knowledge_builder=builder)
        svc._results.append(
            IngestResult(
                id="dl1",
                source_type="file",
                source_name="doc.pdf",
                status="success",
            )
        )
        item = _QueueItem(
            result_id="dl1",
            text="Document content here",
            source="upload://doc.pdf",
            priority=Priority.HIGH,
            page_images=[],
        )
        await svc._deep_learn(item)
        builder.build.assert_called_once()
        fetch = builder.build.call_args[0][0]
        assert fetch.text == "Document content here"
        assert fetch.source_type == "user_upload"
        assert fetch.url == "upload://doc.pdf"
        # Result status updated
        assert svc._results[0].deep_learn_status == "completed"


class TestOcrPdf:
    def test_ocr_returns_empty_without_deps(self):
        """OCR gracefully returns empty when pytesseract/pdf2image not installed."""
        svc = KnowledgeIngestService()
        with patch.dict("sys.modules", {"pytesseract": None}):
            result = svc._ocr_pdf(b"%PDF-1.4 fake")
            assert result == ""

    def test_ocr_returns_empty_on_invalid_pdf(self):
        """OCR returns empty for non-PDF content."""
        svc = KnowledgeIngestService()
        # pdf2image will fail on invalid content
        result = svc._ocr_pdf(b"not a pdf at all")
        assert result == ""

    @pytest.mark.asyncio
    async def test_extract_text_uses_ocr_fallback_for_sparse_pdf(self):
        """If PDF text extraction yields < 100 chars, OCR fallback is tried."""
        svc = KnowledgeIngestService()
        # Mock TextExtractor to return sparse text
        with patch("cognithor.memory.ingest.TextExtractor") as MockExtractor:
            mock_inst = MagicMock()
            mock_inst.extract = AsyncMock(return_value="ab")  # < 100 chars
            MockExtractor.return_value = mock_inst
            with patch.object(svc, "_ocr_pdf", return_value="OCR extracted full text here"):
                text = await svc._extract_text(b"%PDF-1.4", "scan.pdf")
                assert text == "OCR extracted full text here"


class TestYoutubeFrames:
    def test_no_ffmpeg_returns_empty(self):
        """Without ffmpeg binary, returns empty list."""
        svc = KnowledgeIngestService()
        with patch("shutil.which", return_value=None):
            result = svc._extract_youtube_frames("dQw4w9WgXcQ")
            assert result == []

    def test_no_ytdl_returns_empty(self):
        """Without yt-dlp/youtube-dl, returns empty list."""
        svc = KnowledgeIngestService()

        def _which(name: str) -> str | None:
            if name == "ffmpeg":
                return "/usr/bin/ffmpeg"
            return None

        with patch("shutil.which", side_effect=_which):
            result = svc._extract_youtube_frames("dQw4w9WgXcQ")
            assert result == []


class TestKnowledgeIngestProvenance:
    """TRUST-9 wiring: ``ingest_file(provenance_source_type=...,
    provenance_source_id=...)`` writes a tag to the canonical
    PROVENANCE_LEDGER keyed by ``ingest:<result.id>``.
    """

    @pytest.mark.asyncio
    async def test_ingest_without_provenance_does_not_tag(self) -> None:
        import cognithor.memory.provenance as prov_mod
        from cognithor.memory.provenance import ProvenanceLedger

        isolated = ProvenanceLedger()
        original = prov_mod.PROVENANCE_LEDGER
        prov_mod.PROVENANCE_LEDGER = isolated  # type: ignore[misc]
        try:
            memory = MagicMock()
            memory.index_text = MagicMock(return_value=1)
            svc = KnowledgeIngestService(memory=memory)
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(svc, "_extract_text", AsyncMock(return_value="content"))
                mp.setattr(svc, "_ensure_worker", MagicMock())
                result = await svc.ingest_file("a.txt", b"c", priority=Priority.LOW)
            assert f"ingest:{result.id}" not in isolated
        finally:
            prov_mod.PROVENANCE_LEDGER = original  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_ingest_with_provenance_tags_ledger(self) -> None:
        import cognithor.memory.provenance as prov_mod
        from cognithor.memory.provenance import ProvenanceLedger, SourceType

        isolated = ProvenanceLedger()
        original = prov_mod.PROVENANCE_LEDGER
        prov_mod.PROVENANCE_LEDGER = isolated  # type: ignore[misc]
        try:
            memory = MagicMock()
            memory.index_text = MagicMock(return_value=1)
            svc = KnowledgeIngestService(memory=memory)
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(svc, "_extract_text", AsyncMock(return_value="content"))
                mp.setattr(svc, "_ensure_worker", MagicMock())
                result = await svc.ingest_file(
                    "a.pdf",
                    b"content",
                    priority=Priority.NORMAL,
                    provenance_source_type="user_directive",
                    provenance_source_id="upload-7",
                    provenance_notes="owner-uploaded PDF",
                )
            tag = isolated.current(f"ingest:{result.id}")
            assert tag is not None
            assert tag.source_type == SourceType.USER_DIRECTIVE
            assert tag.source_id == "upload-7"
            assert tag.notes == "owner-uploaded PDF"
        finally:
            prov_mod.PROVENANCE_LEDGER = original  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_partial_provenance_args_skip_tag(self) -> None:
        import cognithor.memory.provenance as prov_mod
        from cognithor.memory.provenance import ProvenanceLedger

        isolated = ProvenanceLedger()
        original = prov_mod.PROVENANCE_LEDGER
        prov_mod.PROVENANCE_LEDGER = isolated  # type: ignore[misc]
        try:
            memory = MagicMock()
            memory.index_text = MagicMock(return_value=1)
            svc = KnowledgeIngestService(memory=memory)
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(svc, "_extract_text", AsyncMock(return_value="c"))
                mp.setattr(svc, "_ensure_worker", MagicMock())
                await svc.ingest_file(
                    "a.txt",
                    b"c",
                    priority=Priority.LOW,
                    provenance_source_type="user_directive",
                )
                await svc.ingest_file(
                    "b.txt",
                    b"c",
                    priority=Priority.LOW,
                    provenance_source_id="upload-7",
                )
            assert len(isolated) == 0
        finally:
            prov_mod.PROVENANCE_LEDGER = original  # type: ignore[misc]
