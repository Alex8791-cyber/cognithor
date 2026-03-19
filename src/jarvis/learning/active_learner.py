"""Active Learner: Background knowledge acquisition during idle time.

Watches user-configurable directories and processes files when the
system is idle (no active chat or tool execution).  Builds on the
existing ``IngestPipeline`` for text extraction and memory indexing.

Features:
  - Configurable watch directories with opt-in/opt-out
  - File importance scoring (recency, access frequency)
  - Content-hash deduplication (never reprocesses the same file)
  - Configurable learning rate (files per hour)
  - Event callbacks when new knowledge is acquired

Integration:
  - Uses ``IngestPipeline`` / ``TextExtractor`` for extraction
  - Uses ``MemoryManager.index_text()`` for indexing
  - Emits events consumed by CuriosityEngine / Gateway
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jarvis.memory.ingest import SUPPORTED_EXTENSIONS, TextExtractor
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from jarvis.memory.manager import MemoryManager

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Default directories to watch — DISABLED by default for privacy.
#: Users must explicitly enable directories via the Learning Dashboard.
_DEFAULT_WATCH_DIRS: list[str] = []

#: Maximum file size for active learning (5 MB -- lighter than ingest).
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024

#: Default learning rate: files per hour.
DEFAULT_FILES_PER_HOUR = 10


@dataclass
class WatchDirectory:
    """A single directory to watch."""

    path: Path
    enabled: bool = True
    recursive: bool = True


@dataclass
class ActiveLearnerConfig:
    """Configuration for the active learner."""

    watch_dirs: list[WatchDirectory] = field(default_factory=list)
    files_per_hour: int = DEFAULT_FILES_PER_HOUR
    max_file_size_bytes: int = MAX_FILE_SIZE_BYTES
    poll_interval_seconds: float = 60.0
    idle_threshold_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not self.watch_dirs:
            home = Path.home()
            for name in _DEFAULT_WATCH_DIRS:
                d = home / name
                # Only add if the directory actually exists
                self.watch_dirs.append(WatchDirectory(path=d, enabled=d.exists()))


# ---------------------------------------------------------------------------
# Learned-file record
# ---------------------------------------------------------------------------


@dataclass
class LearnedFile:
    """Record of a file that has been processed."""

    path: str
    content_hash: str
    learned_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    chunks_created: int = 0
    text_length: int = 0


# ---------------------------------------------------------------------------
# File scoring
# ---------------------------------------------------------------------------


def _file_importance(path: Path) -> float:
    """Score a file's importance for learning (0.0 -- 1.0).

    Factors:
      - Recency: recently modified files score higher.
      - Extension: knowledge-rich formats (.md, .txt, .pdf) score higher.
    """
    score = 0.0

    try:
        stat = path.stat()
    except OSError:
        return 0.0

    # Recency: files modified in the last 7 days get full score,
    # decaying linearly to 0 at 365 days.
    age_days = (time.time() - stat.st_mtime) / 86400
    recency = max(0.0, 1.0 - age_days / 365)
    score += recency * 0.6

    # Extension bonus
    ext = path.suffix.lower()
    ext_scores: dict[str, float] = {
        ".md": 0.4,
        ".txt": 0.3,
        ".pdf": 0.35,
        ".docx": 0.3,
        ".html": 0.2,
        ".json": 0.15,
        ".csv": 0.15,
        ".xml": 0.1,
    }
    score += ext_scores.get(ext, 0.05)

    return min(1.0, score)


# ---------------------------------------------------------------------------
# Content hashing
# ---------------------------------------------------------------------------


def _content_hash(path: Path) -> str:
    """SHA-256 of the first 8 KB + file size for fast deduplication."""
    hasher = hashlib.sha256()
    hasher.update(str(path.stat().st_size).encode())
    with open(path, "rb") as f:
        hasher.update(f.read(8192))
    return hasher.hexdigest()[:20]


# ---------------------------------------------------------------------------
# Active Learner
# ---------------------------------------------------------------------------


class ActiveLearner:
    """Background learner that processes files during idle time.

    Usage::

        learner = ActiveLearner(config, memory_manager)
        learner.on_knowledge_acquired = my_callback
        await learner.start()
        # ... later ...
        learner.stop()
    """

    def __init__(
        self,
        config: ActiveLearnerConfig | None = None,
        memory: MemoryManager | None = None,
    ) -> None:
        self._config = config or ActiveLearnerConfig()
        self._memory = memory
        self._extractor = TextExtractor()

        # Deduplication: content_hash -> LearnedFile
        self._learned: dict[str, LearnedFile] = {}

        # Rate limiting
        self._files_this_hour: int = 0
        self._hour_start: float = time.monotonic()

        # Idle tracking
        self._last_activity: float = time.monotonic()

        # Background task
        self._running = False
        self._task: asyncio.Task[None] | None = None

        # Event callback
        self.on_knowledge_acquired: Callable[[LearnedFile], Any] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background learning loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.ensure_future(self._run_loop())
        log.info(
            "active_learner_started",
            watch_dirs=len([d for d in self._config.watch_dirs if d.enabled]),
            rate=self._config.files_per_hour,
        )

    def stop(self) -> None:
        """Stop the background learning loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None
        log.info("active_learner_stopped")

    def notify_activity(self) -> None:
        """Call this when user activity occurs (chat, tool execution).

        Resets the idle timer so the learner pauses during active use.
        """
        self._last_activity = time.monotonic()

    @property
    def is_idle(self) -> bool:
        """True when the system has been idle long enough to learn."""
        return (time.monotonic() - self._last_activity) >= self._config.idle_threshold_seconds

    # ------------------------------------------------------------------
    # Directory management
    # ------------------------------------------------------------------

    def add_directory(
        self,
        path: str | Path,
        *,
        enabled: bool = True,
        recursive: bool = True,
    ) -> None:
        """Add a directory to the watch list."""
        p = Path(path).resolve()
        # Avoid duplicates
        for d in self._config.watch_dirs:
            if d.path.resolve() == p:
                d.enabled = enabled
                return
        self._config.watch_dirs.append(WatchDirectory(path=p, enabled=enabled, recursive=recursive))

    def remove_directory(self, path: str | Path) -> bool:
        """Remove a directory from the watch list.  Returns ``True`` if found."""
        p = Path(path).resolve()
        before = len(self._config.watch_dirs)
        self._config.watch_dirs = [d for d in self._config.watch_dirs if d.path.resolve() != p]
        return len(self._config.watch_dirs) < before

    def set_directory_enabled(self, path: str | Path, enabled: bool) -> bool:
        """Enable or disable a watch directory."""
        p = Path(path).resolve()
        for d in self._config.watch_dirs:
            if d.path.resolve() == p:
                d.enabled = enabled
                return True
        return False

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        """Main background loop: scan, score, learn."""
        while self._running:
            try:
                if self.is_idle and self._rate_ok():
                    await self._learn_cycle()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("active_learner_error", error=str(exc))

            await asyncio.sleep(self._config.poll_interval_seconds)

    def _rate_ok(self) -> bool:
        """Check whether we are within the hourly rate limit."""
        now = time.monotonic()
        if now - self._hour_start >= 3600:
            self._files_this_hour = 0
            self._hour_start = now
        return self._files_this_hour < self._config.files_per_hour

    async def _learn_cycle(self) -> None:
        """Run one learning cycle: discover candidates, pick best, process."""
        candidates = self._discover_candidates()
        if not candidates:
            return

        # Score and sort
        scored = [(path, _file_importance(path)) for path in candidates]
        scored.sort(key=lambda t: t[1], reverse=True)

        # Process top candidate(s) within rate budget
        budget = self._config.files_per_hour - self._files_this_hour
        for path, _score in scored[:budget]:
            if not self._running:
                break
            try:
                await self._process_file(path)
            except Exception as exc:
                log.warning("active_learner_file_error", file=str(path), error=str(exc))

    def _discover_candidates(self) -> list[Path]:
        """Scan enabled watch directories for unprocessed files."""
        candidates: list[Path] = []

        for wd in self._config.watch_dirs:
            if not wd.enabled or not wd.path.exists():
                continue
            try:
                iterator = wd.path.rglob("*") if wd.recursive else wd.path.iterdir()
                for f in iterator:
                    if not f.is_file():
                        continue
                    if f.suffix.lower() not in SUPPORTED_EXTENSIONS:
                        continue
                    if f.stat().st_size > self._config.max_file_size_bytes:
                        continue
                    # Quick dedup check by path (full hash check later)
                    if str(f) in {rec.path for rec in self._learned.values()}:
                        continue
                    candidates.append(f)
            except (OSError, PermissionError) as exc:
                log.debug("active_learner_scan_skip", dir=str(wd.path), error=str(exc))

        return candidates

    async def _process_file(self, path: Path) -> None:
        """Extract text from *path*, index it, and record the result."""
        # Content-hash dedup
        try:
            chash = _content_hash(path)
        except OSError:
            return

        if chash in self._learned:
            return

        # Extract
        text = await self._extractor.extract(path)
        if not text or not text.strip():
            return

        # Index
        chunks_created = 0
        if self._memory is not None:
            source = f"active_learn://{path.name}"
            if hasattr(self._memory, "index_text"):
                chunks_created = self._memory.index_text(text, source)

        # Record
        record = LearnedFile(
            path=str(path),
            content_hash=chash,
            chunks_created=chunks_created,
            text_length=len(text),
        )
        self._learned[chash] = record
        self._files_this_hour += 1

        log.info(
            "active_learner_learned",
            file=path.name,
            chunks=chunks_created,
            text_len=len(text),
        )

        # Emit event
        if self.on_knowledge_acquired is not None:
            try:
                result = self.on_knowledge_acquired(record)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                log.warning("active_learner_callback_error", error=str(exc))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return active learner statistics."""
        return {
            "running": self._running,
            "is_idle": self.is_idle,
            "files_learned": len(self._learned),
            "files_this_hour": self._files_this_hour,
            "rate_limit": self._config.files_per_hour,
            "total_chunks": sum(r.chunks_created for r in self._learned.values()),
            "total_text_length": sum(r.text_length for r in self._learned.values()),
            "watch_dirs": [
                {
                    "path": str(d.path),
                    "enabled": d.enabled,
                    "exists": d.path.exists(),
                }
                for d in self._config.watch_dirs
            ],
        }

    @property
    def learned_files(self) -> list[LearnedFile]:
        """Return all learned file records."""
        return list(self._learned.values())
