"""Tests für memory/watcher.py · File Watcher."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from jarvis.memory.watcher import MemoryFileHandler, MemoryWatcher

if TYPE_CHECKING:
    from pathlib import Path


class TestMemoryFileHandler:
    def test_ignores_non_markdown(self):
        calls: list[str] = []
        handler = MemoryFileHandler(lambda p: calls.append(p))
        handler.on_file_changed("test.py")
        handler.on_file_changed("data.json")
        processed = handler.process_pending()
        assert processed == []

    def test_processes_markdown(self):
        calls: list[str] = []
        handler = MemoryFileHandler(lambda p: calls.append(p), debounce_seconds=0.0)
        handler.on_file_changed("test.md")
        processed = handler.process_pending()
        assert len(processed) == 1
        assert "test.md" in calls

    def test_debounce(self):
        calls: list[str] = []
        handler = MemoryFileHandler(lambda p: calls.append(p), debounce_seconds=1.0)
        handler.on_file_changed("test.md")

        # Should not process yet (debounce not elapsed)
        processed = handler.process_pending()
        assert processed == []

    def test_deduplication(self):
        calls: list[str] = []
        handler = MemoryFileHandler(lambda p: calls.append(p), debounce_seconds=0.0)

        # Same file changed multiple times
        handler.on_file_changed("test.md")
        handler.on_file_changed("test.md")
        handler.on_file_changed("test.md")

        processed = handler.process_pending()
        assert len(processed) == 1  # Only one callback

    def test_multiple_files(self):
        calls: list[str] = []
        handler = MemoryFileHandler(lambda p: calls.append(p), debounce_seconds=0.0)

        handler.on_file_changed("a.md")
        handler.on_file_changed("b.md")

        processed = handler.process_pending()
        assert len(processed) == 2

    def test_callback_error_handled(self):
        def bad_callback(path: str) -> None:
            raise RuntimeError("Test error")

        handler = MemoryFileHandler(bad_callback, debounce_seconds=0.0)
        handler.on_file_changed("test.md")

        # Should not raise
        processed = handler.process_pending()
        assert processed == []  # Failed, not in processed


class TestMemoryWatcher:
    def test_create(self, tmp_path: Path):
        watcher = MemoryWatcher(tmp_path, lambda p: None)
        assert not watcher.is_running

    def test_start_stop_polling(self, tmp_path: Path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        watcher = MemoryWatcher(
            tmp_path,
            lambda p: None,
            poll_interval=0.1,
        )
        watcher.start()
        assert watcher.is_running
        time.sleep(0.2)
        watcher.stop()
        assert not watcher.is_running

    def test_detects_new_file(self, tmp_path: Path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        changed_files: list[str] = []

        watcher = MemoryWatcher(
            tmp_path,
            lambda p: changed_files.append(p),
            poll_interval=0.1,
            debounce_seconds=0.0,
        )
        watcher.start()

        # Create a new file
        time.sleep(0.3)
        (tmp_path / "new.md").write_text("Hello", encoding="utf-8")
        time.sleep(1.0)

        watcher.stop()
        assert len(changed_files) >= 1

    def test_detects_modified_file(self, tmp_path: Path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        test_file = tmp_path / "test.md"
        test_file.write_text("Version 1", encoding="utf-8")

        changed_files: list[str] = []
        watcher = MemoryWatcher(
            tmp_path,
            lambda p: changed_files.append(p),
            poll_interval=0.1,
            debounce_seconds=0.0,
        )
        watcher.start()
        time.sleep(0.3)

        # Modify the file (ensure mtime changes)
        time.sleep(0.1)
        test_file.write_text("Version 2", encoding="utf-8")
        time.sleep(1.0)

        watcher.stop()
        assert len(changed_files) >= 1
