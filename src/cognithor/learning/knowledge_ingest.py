"""Unified knowledge ingestion -- files, URLs, YouTube videos.

Accepts content from the Flutter UI and processes it into the memory system.
Supports: PDF, DOCX, images (via vision), websites (via trafilatura), YouTube (via transcript API).
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
import heapq
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.memory.manager import MemoryManager

log = get_logger(__name__)


class Priority(enum.IntEnum):
    """Upload learning priority."""

    HIGH = 0
    NORMAL = 1
    LOW = 2

    @classmethod
    def from_string(cls, s: str) -> Priority:
        try:
            return cls[s.upper()]
        except KeyError:
            return cls.NORMAL


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
    priority: Priority = Priority.NORMAL
    deep_learn_status: str = "pending"  # pending, queued, skipped, completed, failed

    @property
    def chunks(self) -> int:
        """Alias for chunks_created — Flutter compat."""
        return self.chunks_created


@dataclass
class _QueueItem:
    """Item in the deep-learn priority queue."""

    result_id: str
    text: str
    source: str
    priority: Priority
    page_images: list[Path]

    def __lt__(self, other: _QueueItem) -> bool:
        return self.priority < other.priority


class IngestQueue:
    """Priority queue for background deep-learning tasks."""

    def __init__(self) -> None:
        self._heap: list[_QueueItem] = []

    def enqueue(self, item: _QueueItem) -> None:
        heapq.heappush(self._heap, item)

    def dequeue(self) -> _QueueItem:
        return heapq.heappop(self._heap)

    @property
    def empty(self) -> bool:
        return len(self._heap) == 0

    def __len__(self) -> int:
        return len(self._heap)

    def pending(self) -> list[dict[str, str]]:
        """Return queue contents for the API (without consuming)."""
        return [
            {"id": item.result_id, "source": item.source, "priority": item.priority.name}
            for item in sorted(self._heap)
        ]


class KnowledgeIngestService:
    """Processes files, URLs, and YouTube links into Jarvis memory."""

    def __init__(
        self,
        memory: MemoryManager | None = None,
        knowledge_builder: Any | None = None,
        llm_fn: Any | None = None,
        evolution_loop: Any | None = None,
        on_progress: Any | None = None,
    ) -> None:
        self._memory = memory
        self._knowledge_builder = knowledge_builder
        self._llm_fn = llm_fn
        self._evolution_loop = evolution_loop
        self._on_progress = on_progress
        self._results: list[IngestResult] = []
        self._queue = IngestQueue()
        self._worker_task: asyncio.Task | None = None

    async def _notify(self, event: str, source: str, **kwargs: Any) -> None:
        """Fire progress callback and emit a structured log entry."""
        log.info(event, source=source, **kwargs)
        if self._on_progress:
            with contextlib.suppress(Exception):
                await self._on_progress(event, source, **kwargs)

    async def ingest_file(
        self,
        filename: str,
        content: bytes,
        priority: Priority = Priority.NORMAL,
    ) -> IngestResult:
        """Ingest a file (PDF, DOCX, TXT, MD, images)."""
        result = IngestResult(
            id=str(uuid4()),
            source_type="file",
            source_name=filename,
            status="processing",
        )
        try:
            text = await self._extract_text(content, filename)

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
            result.priority = priority

            # Queue for deep learning
            if priority == Priority.LOW:
                result.deep_learn_status = "skipped"
            else:
                result.deep_learn_status = "queued"
                suffix = Path(filename).suffix.lower()
                page_images = self._extract_page_images(content, suffix)
                item = _QueueItem(
                    result_id=result.id,
                    text=text,
                    source=f"upload://{filename}",
                    priority=priority,
                    page_images=page_images,
                )
                self._queue.enqueue(item)
                self._ensure_worker()

        except Exception as exc:
            result.status = "failed"
            result.error = str(exc)
            log.warning("ingest_file_failed", file=filename, error=str(exc))

        self._results.append(result)
        return result

    async def ingest_url(
        self,
        url: str,
        priority: Priority = Priority.NORMAL,
    ) -> IngestResult:
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
                return await self.ingest_youtube(url, priority=priority)

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
            result.priority = priority

            # Queue for deep learning
            if priority == Priority.LOW:
                result.deep_learn_status = "skipped"
            else:
                result.deep_learn_status = "queued"
                item = _QueueItem(
                    result_id=result.id,
                    text=text,
                    source=f"web://{url}",
                    priority=priority,
                    page_images=[],
                )
                self._queue.enqueue(item)
                self._ensure_worker()

        except Exception as exc:
            result.status = "failed"
            result.error = str(exc)

        self._results.append(result)
        return result

    async def ingest_youtube(
        self,
        url: str,
        priority: Priority = Priority.NORMAL,
    ) -> IngestResult:
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
            result.priority = priority

            # Queue for deep learning (with optional video frame extraction)
            if priority == Priority.LOW:
                result.deep_learn_status = "skipped"
            else:
                result.deep_learn_status = "queued"
                frames = self._extract_youtube_frames(video_id) if priority == Priority.HIGH else []
                item = _QueueItem(
                    result_id=result.id,
                    text=text,
                    source=f"youtube://{video_id}",
                    priority=priority,
                    page_images=frames,
                )
                self._queue.enqueue(item)
                self._ensure_worker()

        except Exception as exc:
            result.status = "failed"
            result.error = str(exc)

        self._results.append(result)
        return result

    async def _extract_text(self, content: bytes, filename: str) -> str:
        """Extract text from file content.

        For images: uses vision model.
        For PDFs: uses TextExtractor, then falls back to OCR if text is sparse.
        For other documents: uses TextExtractor.
        """
        import tempfile

        suffix = Path(filename).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(content)
            tmp_path = Path(f.name)
        try:
            if suffix in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"):
                return await self._extract_image_text(tmp_path)
            from cognithor.memory.ingest import TextExtractor

            extractor = TextExtractor()
            text = await extractor.extract(tmp_path)

            # OCR fallback for scanned PDFs: if extracted text is very sparse
            if suffix == ".pdf" and len((text or "").strip()) < 100:
                ocr_text = self._ocr_pdf(content)
                if ocr_text and len(ocr_text) > len(text or ""):
                    log.info("ocr_fallback_used", file=filename, ocr_len=len(ocr_text))
                    return ocr_text

            return text
        finally:
            tmp_path.unlink(missing_ok=True)

    def _ocr_pdf(self, content: bytes) -> str:
        """OCR a scanned PDF using Tesseract. Returns empty string on failure."""
        try:
            from pdf2image import convert_from_bytes
            from pytesseract import image_to_string
        except ImportError:
            return ""

        try:
            images = convert_from_bytes(content, dpi=200, fmt="png")
            pages: list[str] = []
            for img in images[:50]:  # Cap at 50 pages
                text = image_to_string(img, lang="eng+deu")
                if text and text.strip():
                    pages.append(text.strip())
            return "\n\n".join(pages)
        except Exception as exc:
            log.debug("ocr_pdf_failed", error=str(exc))
            return ""

    def _extract_youtube_frames(self, video_id: str, max_frames: int = 5) -> list[Path]:
        """Extract key frames from a YouTube video using yt-dlp + ffmpeg.

        Downloads a low-res version, extracts frames at regular intervals.
        Returns list of PNG paths. Gracefully returns [] if tools missing.
        """
        try:
            import shutil
            import subprocess
            import tempfile

            if not shutil.which("ffmpeg"):
                log.debug("youtube_frames_no_ffmpeg")
                return []

            # Try yt-dlp first, then youtube-dl
            ytdl = shutil.which("yt-dlp") or shutil.which("youtube-dl")
            if not ytdl:
                log.debug("youtube_frames_no_ytdl")
                return []

            tmp_dir = Path(tempfile.mkdtemp(prefix="cognithor_yt_"))
            video_path = tmp_dir / "video.mp4"

            # Download lowest quality video (small + fast)
            dl_result = subprocess.run(
                [
                    ytdl,
                    "-f",
                    "worst[ext=mp4]",
                    "--no-playlist",
                    "--max-filesize",
                    "50M",
                    "-o",
                    str(video_path),
                    f"https://www.youtube.com/watch?v={video_id}",
                ],
                capture_output=True,
                timeout=120,
            )
            if dl_result.returncode != 0 or not video_path.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return []

            # Extract frames at regular intervals using ffmpeg
            frames_dir = tmp_dir / "frames"
            frames_dir.mkdir()
            # Get video duration via ffprobe
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "csv=p=0",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            duration = float(probe.stdout.strip()) if probe.returncode == 0 else 60.0
            interval = max(duration / (max_frames + 1), 5.0)

            # Extract frames
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    str(video_path),
                    "-vf",
                    f"fps=1/{interval:.0f}",
                    "-frames:v",
                    str(max_frames),
                    "-q:v",
                    "2",
                    str(frames_dir / "frame_%03d.png"),
                ],
                capture_output=True,
                timeout=60,
            )

            # Collect frame paths, clean up video
            video_path.unlink(missing_ok=True)
            paths = sorted(frames_dir.glob("*.png"))[:max_frames]
            log.info("youtube_frames_extracted", video_id=video_id, frames=len(paths))
            return paths

        except Exception as exc:
            log.debug("youtube_frames_failed", video_id=video_id, error=str(exc))
            return []

    def _extract_page_images(self, content: bytes, suffix: str) -> list[Path]:
        """Extract image-heavy pages from a PDF as PNG files.

        Uses pypdf to detect pages with embedded images, then renders
        those pages via pdf2image (if available). Falls back gracefully
        if pdf2image or poppler is not installed.
        """
        if suffix != ".pdf":
            return []

        try:
            from pypdf import PdfReader
        except ImportError:
            return []

        import io
        import tempfile

        try:
            reader = PdfReader(io.BytesIO(content))
        except Exception:
            return []

        image_pages: list[int] = []
        for i, page in enumerate(reader.pages):
            resources = page.get("/Resources")
            if resources is None:
                continue
            xobjects = resources.get("/XObject")
            if xobjects and len(xobjects) > 0:
                image_pages.append(i)

        if not image_pages:
            return []

        # Render image-heavy pages as PNG
        try:
            from pdf2image import convert_from_bytes

            images = convert_from_bytes(
                content,
                first_page=min(image_pages) + 1,
                last_page=max(image_pages) + 1,
                dpi=150,
                fmt="png",
            )

            paths: list[Path] = []
            for _idx, img in enumerate(images):
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                img.save(tmp.name, "PNG")
                paths.append(Path(tmp.name))
                if len(paths) >= 10:  # Cap at 10 images
                    break

            return paths

        except ImportError:
            log.debug("pdf2image_not_available_for_vision")
            return []
        except Exception as exc:
            log.debug("pdf_image_extraction_failed", error=str(exc))
            return []

    def _ensure_worker(self) -> None:
        """Start the background worker if not running."""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def _worker_loop(self) -> None:
        """Process queued deep-learn items."""
        while not self._queue.empty:
            item = self._queue.dequeue()
            log.info(
                "deep_learn_worker_processing",
                source=item.source,
                queue_remaining=len(self._queue),
            )
            try:
                await self._deep_learn(item)
            except Exception as exc:
                log.warning("deep_learn_failed", source=item.source, error=str(exc))
                self._update_result(item.result_id, "failed", error=str(exc))

    async def _deep_learn(self, item: _QueueItem) -> None:
        """Run full KnowledgeBuilder pipeline on queued item."""
        if self._knowledge_builder is None:
            log.debug("deep_learn_skipped_no_builder", source=item.source)
            self._update_result(item.result_id, "skipped")
            return

        from cognithor.evolution.research_agent import FetchResult

        await self._notify("deep_learn_start", item.source, priority=item.priority.name)

        # Vision analysis for page images
        vision_text = ""
        if item.page_images:
            await self._notify("deep_learn_vision", item.source, images=len(item.page_images))
            for img_path in item.page_images:
                try:
                    desc = await self._extract_image_text(img_path)
                    if desc:
                        vision_text += f"\n[Image description: {desc}]\n"
                except Exception:
                    log.debug("ingest_vision_image_extraction_failed", exc_info=True)
                finally:
                    img_path.unlink(missing_ok=True)

        full_text = item.text + vision_text if vision_text else item.text

        source_name = item.source.split("://", 1)[-1] if "://" in item.source else item.source
        await self._notify("deep_learn_building", item.source)
        fetch_result = FetchResult(
            url=item.source,
            text=full_text,
            title=source_name,
            source_type="user_upload",
        )
        await self._knowledge_builder.build(fetch_result)
        self._update_result(item.result_id, "completed")
        await self._notify("deep_learn_complete", item.source, text_len=len(full_text))

        # Also queue for Evolution Loop idle processing (if available)
        if self._evolution_loop and hasattr(self._evolution_loop, "inject_user_material"):
            try:
                await self._evolution_loop.inject_user_material(
                    full_text,
                    source_name,
                    goal_slug="user_upload",
                )
            except Exception:
                log.debug("evolution_inject_failed", source=item.source)

    def _update_result(self, result_id: str, status: str, error: str = "") -> None:
        """Update deep_learn_status on a stored IngestResult."""
        for r in self._results:
            if r.id == result_id:
                r.deep_learn_status = status
                if error:
                    r.error = error
                break

    async def _extract_image_text(self, path: Path) -> str:
        """Extract text from image using vision LLM."""
        try:
            from cognithor.mcp.media import MediaPipeline

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
