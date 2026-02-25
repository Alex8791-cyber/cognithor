"""
Tests für jarvis.utils.logging – Structured Logging.

Testet:
  - Setup mit verschiedenen Konfigurationen
  - Logger-Erstellung
  - Context-Binding
  - JSON-Log-Output
  - File-Logging
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jarvis.utils.logging import (
    bind_context,
    clear_context,
    get_logger,
    setup_logging,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestLoggingSetup:
    def test_default_setup(self) -> None:
        """Logging initialisiert ohne Fehler."""
        setup_logging(level="INFO", console=True)
        log = get_logger("test")
        # Sollte nicht werfen
        log.info("test_event", key="value")

    def test_debug_level(self) -> None:
        setup_logging(level="DEBUG", console=True)
        log = get_logger("test.debug")
        log.debug("debug_event", detail="works")

    def test_json_mode(self) -> None:
        setup_logging(level="INFO", json_logs=True, console=True)
        log = get_logger("test.json")
        log.info("json_test", number=42)

    def test_file_logging(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        setup_logging(level="DEBUG", log_dir=log_dir, console=False)
        log = get_logger("test.file")
        log.info("file_event", path=str(tmp_path))

        # Log-Datei sollte existieren
        log_file = log_dir / "jarvis.jsonl"
        assert log_file.exists()


class TestContextBinding:
    def test_bind_and_clear(self) -> None:
        setup_logging(level="INFO", console=True)
        bind_context(session_id="test_123", channel="cli")
        log = get_logger("test.context")
        log.info("with_context")  # session_id + channel sollten dabei sein
        clear_context()
        log.info("without_context")  # Kein extra Kontext

    def test_multiple_bindings(self) -> None:
        setup_logging(level="INFO", console=True)
        bind_context(a="1")
        bind_context(b="2")
        clear_context()
