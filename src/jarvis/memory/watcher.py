"""File Watcher · Auto-Reindexierung bei Dateiänderungen. [B§4]

Überwacht ~/.jarvis/memory/ und re-indexiert geänderte Dateien.
Nutzt watchdog für Filesystem-Events.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

logger = logging.getLogger("jarvis.memory.watcher")

if TYPE_CHECKING:
    from collections.abc import Callable

    pass


class MemoryFileHandler:
    """Verarbeitet Filesystem-Events für Memory-Dateien.

    Sammelt Änderungen und verarbeitet sie gebatcht.
    """

    def __init__(
        self,
        callback: Callable[[str], None],
        debounce_seconds: float = 2.0,
    ) -> None:
        """Initialisiert den DebounceState für Dateiänderungs-Tracking."""
        self._callback = callback
        self._debounce = debounce_seconds
        self._pending: dict[str, float] = {}  # path → last_event_time
        self._lock = threading.Lock()

    def on_file_changed(self, path: str) -> None:
        """Registriert eine Dateiänderung."""
        if not path.endswith(".md"):
            return

        with self._lock:
            self._pending[path] = time.time()

    def process_pending(self) -> list[str]:
        """Verarbeitet ausstehende Änderungen (nach Debounce).

        Returns:
            Liste der verarbeiteten Pfade.
        """
        now = time.time()
        processed: list[str] = []

        with self._lock:
            ready = {p: t for p, t in self._pending.items() if now - t >= self._debounce}
            for path in ready:
                del self._pending[path]

        for path in ready:
            try:
                self._callback(path)
                processed.append(path)
                logger.debug("Re-indexiert: %s", path)
            except Exception as e:
                logger.error("Re-Index fehlgeschlagen für %s: %s", path, e)

        return processed


class MemoryWatcher:
    """Überwacht das Memory-Verzeichnis auf Dateiänderungen.

    Verwendet einen einfachen Polling-Ansatz als Fallback
    wenn watchdog nicht verfügbar ist.
    """

    def __init__(
        self,
        memory_dir: str | Path,
        on_file_changed: Callable[[str], None],
        *,
        poll_interval: float = 5.0,
        debounce_seconds: float = 2.0,
    ) -> None:
        """Initialisiert den FileWatcher mit Memory-Manager und Watch-Pfaden."""
        self._dir = Path(memory_dir)
        self._handler = MemoryFileHandler(on_file_changed, debounce_seconds)
        self._poll_interval = poll_interval
        self._running = False
        self._thread: threading.Thread | None = None
        self._observer: Any = None
        self._file_mtimes: dict[str, float] = {}
        self._use_watchdog = False

    @property
    def is_running(self) -> bool:
        """Prüft ob der Watcher aktiv ist."""
        return self._running

    def start(self) -> None:
        """Startet den File-Watcher."""
        if self._running:
            return

        self._running = True

        # Versuche watchdog, Fallback auf Polling
        try:
            self._start_watchdog()
            self._use_watchdog = True
            logger.info("Memory-Watcher gestartet (watchdog)")
        except ImportError:
            self._start_polling()
            logger.info("Memory-Watcher gestartet (polling, interval=%.1fs)", self._poll_interval)

    def stop(self) -> None:
        """Stoppt den File-Watcher."""
        self._running = False
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._thread = None
        logger.info("Memory-Watcher gestoppt")

    def _start_watchdog(self) -> None:
        """Startet watchdog-basiertes Monitoring."""
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        watcher = self

        class Handler(FileSystemEventHandler):
            """Watchdog-Event-Handler für Dateiänderungen."""

            def on_modified(self, event: Any) -> None:
                """Reagiert auf Dateiänderungen."""
                if not event.is_directory:
                    watcher._handler.on_file_changed(event.src_path)

            def on_created(self, event: Any) -> None:
                """Reagiert auf neue Dateien."""
                if not event.is_directory:
                    watcher._handler.on_file_changed(event.src_path)

        observer = Observer()
        observer.schedule(Handler(), str(self._dir), recursive=True)
        observer.daemon = True
        observer.start()
        self._observer = observer

        # Background-Thread für Debounce-Processing
        def _process_loop() -> None:
            """Verarbeitet die Reindex-Queue periodisch."""
            while self._running:
                self._handler.process_pending()
                time.sleep(1.0)

        self._thread = threading.Thread(target=_process_loop, daemon=True)
        self._thread.start()

    def _start_polling(self) -> None:
        """Startet Polling-basiertes Monitoring."""
        # Initial Scan
        self._scan_files()

        def _poll_loop() -> None:
            """Polling-basiertes Fallback wenn watchdog nicht verfügbar."""
            while self._running:
                time.sleep(self._poll_interval)
                self._check_changes()
                self._handler.process_pending()

        self._thread = threading.Thread(target=_poll_loop, daemon=True)
        self._thread.start()

    def _scan_files(self) -> None:
        """Scannt alle .md Dateien und speichert ihre mtimes."""
        if not self._dir.exists():
            return
        self._file_mtimes = {}
        for f in self._dir.rglob("*.md"):
            with contextlib.suppress(OSError):
                self._file_mtimes[str(f)] = f.stat().st_mtime

    def _check_changes(self) -> None:
        """Prüft auf geänderte oder neue Dateien."""
        if not self._dir.exists():
            return

        current_files: dict[str, float] = {}
        for f in self._dir.rglob("*.md"):
            try:
                current_files[str(f)] = f.stat().st_mtime
            except OSError:
                continue

        # Geänderte oder neue Dateien
        for path, mtime in current_files.items():
            old_mtime = self._file_mtimes.get(path)
            if old_mtime is None or mtime > old_mtime:
                self._handler.on_file_changed(path)

        self._file_mtimes = current_files
