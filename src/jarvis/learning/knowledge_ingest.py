"""Unified knowledge ingestion -- files, URLs, YouTube videos.

Accepts content from the Flutter UI and processes it into the memory system.
Supports: PDF, DOCX, images (via vision), websites (via trafilatura), YouTube (via transcript API).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.memory.manager import MemoryManager

log = get_logger(__name__)


@dataclass
class IngestResult:
    """Result of a single knowledge ingestion operation."""

    id: str
    source_type: str  # file, url, youtube
    source_name: str
    status: str  # processing, success, failed
    chunks_created: int = 0
    text_length: int = 0
    error: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class KnowledgeIngestService:
    """Processes files, URLs, and YouTube links into Jarvis memory."""

    def __init__(self, memory: MemoryManager | None = None) -> None:
        self._memory = memory
        self._results: list[IngestResult] = []

    async def ingest_file(self, filename: str, content: bytes) -> IngestResult:
        """Ingest a file (PDF, DOCX, TXT, MD, images)."""
        result = IngestResult(
            id=str(uuid4()),
            source_type="file",
            source_name=filename,
            status="processing",
        )
        try:
            import tempfile

            suffix = Path(filename).suffix.lower()

            # Write to temp file for extraction
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(content)
                tmp_path = Path(f.name)

            text = ""

            # Image: use vision analysis
            if suffix in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"):
                text = await self._extract_image_text(tmp_path)
            else:
                # Use TextExtractor for documents
                from jarvis.memory.ingest import TextExtractor

                extractor = TextExtractor()
                text = await extractor.extract(tmp_path)

            # Clean up temp file
            tmp_path.unlink(missing_ok=True)

            if not text or not text.strip():
                result.status = "failed"
                result.error = "No text could be extracted"
                self._results.append(result)
                return result

            # Index into memory
            chunks = 0
            if self._memory and hasattr(self._memory, "index_text"):
                chunks = self._memory.index_text(text, f"upload://{filename}")

            result.status = "success"
            result.chunks_created = chunks
            result.text_length = len(text)

        except Exception as exc:
            result.status = "failed"
            result.error = str(exc)
            log.warning("ingest_file_failed", file=filename, error=str(exc))

        self._results.append(result)
        return result

    async def ingest_url(self, url: str) -> IngestResult:
        """Ingest a website URL (extracts main content via trafilatura)."""
        result = IngestResult(
            id=str(uuid4()),
            source_type="url",
            source_name=url,
            status="processing",
        )
        try:
            # Check if YouTube
            if _is_youtube_url(url):
                return await self.ingest_youtube(url)

            # Fetch and extract
            text = await self._fetch_url_text(url)

            if not text or not text.strip():
                result.status = "failed"
                result.error = "No text could be extracted from URL"
                self._results.append(result)
                return result

            chunks = 0
            if self._memory and hasattr(self._memory, "index_text"):
                chunks = self._memory.index_text(text, f"web://{url}")

            result.status = "success"
            result.chunks_created = chunks
            result.text_length = len(text)

        except Exception as exc:
            result.status = "failed"
            result.error = str(exc)

        self._results.append(result)
        return result

    async def ingest_youtube(self, url: str) -> IngestResult:
        """Ingest a YouTube video (extracts transcript/captions)."""
        result = IngestResult(
            id=str(uuid4()),
            source_type="youtube",
            source_name=url,
            status="processing",
        )
        try:
            video_id = _extract_youtube_id(url)
            if not video_id:
                result.status = "failed"
                result.error = "Invalid YouTube URL"
                self._results.append(result)
                return result

            text = await self._fetch_youtube_transcript(video_id)

            if not text:
                result.status = "failed"
                result.error = "No transcript available for this video"
                self._results.append(result)
                return result

            chunks = 0
            if self._memory and hasattr(self._memory, "index_text"):
                chunks = self._memory.index_text(text, f"youtube://{video_id}")

            result.status = "success"
            result.chunks_created = chunks
            result.text_length = len(text)

        except Exception as exc:
            result.status = "failed"
            result.error = str(exc)

        self._results.append(result)
        return result

    async def _extract_image_text(self, path: Path) -> str:
        """Extract text from image using vision LLM."""
        try:
            from jarvis.mcp.media import MediaPipeline

            media = MediaPipeline()
            result = await media.analyze_image(
                str(path),
                prompt="Extract all text and describe the content of this image in detail.",
            )
            if result.success and result.text:
                return result.text
            return ""
        except Exception:
            return ""

    async def _fetch_url_text(self, url: str) -> str:
        """Fetch and extract main text from a URL."""
        try:
            import httpx
            import trafilatura

            async with httpx.AsyncClient(
                timeout=30,
                follow_redirects=True,
            ) as client:
                resp = await client.get(
                    url,
                    headers={"User-Agent": "Cognithor/1.0"},
                )
                resp.raise_for_status()
                html = resp.text
            text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
            )
            return text or ""
        except ImportError:
            # Fallback without trafilatura
            import httpx

            async with httpx.AsyncClient(
                timeout=30,
                follow_redirects=True,
            ) as client:
                resp = await client.get(url)
                return resp.text[:50000]
        except Exception as exc:
            log.warning("url_fetch_failed", url=url, error=str(exc))
            return ""

    async def _fetch_youtube_transcript(self, video_id: str) -> str:
        """Fetch YouTube transcript via the free timedtext API."""
        try:
            import httpx

            # Fetch YouTube page to extract captions track URL
            api_url = f"https://www.youtube.com/watch?v={video_id}"
            async with httpx.AsyncClient(
                timeout=30,
                follow_redirects=True,
            ) as client:
                resp = await client.get(
                    api_url,
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                html = resp.text

            # Extract captions track URL from page source
            import re as _re

            match = _re.search(
                r'"captionTracks":\[.*?"baseUrl":"(.*?)"',
                html,
            )
            if not match:
                # Try alternative: look for playerCaptionsTracklistRenderer
                match = _re.search(
                    r'"baseUrl":"(https://www\.youtube\.com/api/timedtext[^"]*)"',
                    html,
                )

            if not match:
                return ""

            caption_url = match.group(1).replace("\\u0026", "&")

            # Fetch the captions XML
            async with httpx.AsyncClient(timeout=15) as client:
                cap_resp = await client.get(caption_url)
                cap_xml = cap_resp.text

            # Parse XML captions to plain text
            lines = []
            for text_match in _re.finditer(
                r"<text[^>]*>(.*?)</text>",
                cap_xml,
                _re.DOTALL,
            ):
                line = text_match.group(1)
                # Decode HTML entities
                line = line.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                line = line.replace("&#39;", "'").replace("&quot;", '"')
                lines.append(line.strip())

            return "\n".join(lines)

        except Exception as exc:
            log.warning(
                "youtube_transcript_failed",
                video_id=video_id,
                error=str(exc),
            )
            return ""

    @property
    def results(self) -> list[IngestResult]:
        """Return a copy of all ingestion results."""
        return list(self._results)

    def stats(self) -> dict[str, Any]:
        """Return aggregate ingestion statistics."""
        total = len(self._results)
        success = sum(1 for r in self._results if r.status == "success")
        return {
            "total": total,
            "success": success,
            "failed": total - success,
            "total_chunks": sum(r.chunks_created for r in self._results),
            "total_text": sum(r.text_length for r in self._results),
        }


def _is_youtube_url(url: str) -> bool:
    """Check if a URL is a YouTube URL."""
    return bool(re.search(r"(youtube\.com|youtu\.be)", url, re.I))


def _extract_youtube_id(url: str) -> str:
    """Extract a YouTube video ID from a URL."""
    patterns = [
        r"(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:embed/)([a-zA-Z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return ""
