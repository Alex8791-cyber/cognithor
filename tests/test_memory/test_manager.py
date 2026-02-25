"""Tests für memory/manager.py · Zentrale Memory-API."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from jarvis.config import JarvisConfig
from jarvis.memory.manager import MemoryManager
from jarvis.models import MemoryTier

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def config(tmp_path: Path) -> JarvisConfig:
    return JarvisConfig(jarvis_home=tmp_path / ".jarvis")


@pytest.fixture
def manager(config: JarvisConfig) -> MemoryManager:
    return MemoryManager(config)


class TestMemoryManagerInit:
    def test_initialize(self, manager: MemoryManager, config: JarvisConfig):
        stats = manager.initialize_sync()
        assert stats["initialized"]
        assert config.memory_dir.exists()
        assert config.index_dir.exists()
        assert config.episodes_dir.exists()
        assert config.knowledge_dir.exists()
        assert config.procedures_dir.exists()
        assert config.core_memory_path.exists()

    def test_creates_default_core(self, manager: MemoryManager):
        manager.initialize_sync()
        content = manager.core.content
        assert "Identität" in content

    def test_loads_existing_core(self, manager: MemoryManager, config: JarvisConfig):
        config.core_memory_path.parent.mkdir(parents=True, exist_ok=True)
        config.core_memory_path.write_text("# Custom\nMy custom core\n", encoding="utf-8")
        manager.initialize_sync()
        assert "Custom" in manager.core.content

    def test_stats(self, manager: MemoryManager):
        manager.initialize_sync()
        stats = manager.stats()
        assert "chunks" in stats
        assert "entities" in stats
        assert "procedures" in stats
        assert "core_memory_loaded" in stats
        assert stats["core_memory_loaded"]


class TestMemoryManagerIndexing:
    def test_index_file(self, manager: MemoryManager, config: JarvisConfig):
        manager.initialize_sync()

        # Erstelle eine Test-Datei
        test_file = config.knowledge_dir / "test.md"
        test_file.write_text(
            "# Test\nDies ist ein Testdokument.\n\nMit mehreren Absätzen.", encoding="utf-8"
        )

        count = manager.index_file(test_file)
        assert count >= 1
        assert manager.index.count_chunks() >= 1

    def test_index_text(self, manager: MemoryManager):
        manager.initialize_sync()
        count = manager.index_text(
            "Kontakt Müller nutzt Cloud-Lizenz Nr. 123",
            "virtual/mueller.md",
            tier=MemoryTier.SEMANTIC,
        )
        assert count >= 1

    def test_reindex_all(self, manager: MemoryManager, config: JarvisConfig):
        manager.initialize_sync()

        # Erstelle Dateien in verschiedenen Tiers
        (config.knowledge_dir / "test.md").write_text("# Test\nSemantic content", encoding="utf-8")

        ep = manager.episodic
        from datetime import datetime

        ep.append_entry("Test", "Episode", timestamp=datetime(2026, 2, 21, 10, 0))

        counts = manager.reindex_all()
        assert counts["core"] >= 1
        assert counts.get("episodic", 0) >= 0
        assert counts.get("semantic", 0) >= 0

    def test_index_replaces_old(self, manager: MemoryManager, config: JarvisConfig):
        manager.initialize_sync()

        test_file = config.knowledge_dir / "test.md"
        test_file.write_text("Version 1", encoding="utf-8")
        manager.index_file(test_file)
        count1 = manager.index.count_chunks()

        test_file.write_text("Version 2 with more content", encoding="utf-8")
        manager.index_file(test_file)
        count2 = manager.index.count_chunks()

        # Should replace, not accumulate
        assert count2 == count1


class TestMemoryManagerSearch:
    def test_sync_search(self, manager: MemoryManager, config: JarvisConfig):
        manager.initialize_sync()
        manager.index_text(
            "Projektmanagement für Entwicklerteams",
            "test/bu.md",
        )
        results = manager.search_memory_sync("Projektmanagement")
        assert len(results) >= 1

    def test_sync_search_no_results(self, manager: MemoryManager):
        manager.initialize_sync()
        results = manager.search_memory_sync("xyznonsense")
        assert results == []


class TestMemoryManagerSession:
    def test_start_session(self, manager: MemoryManager):
        manager.initialize_sync()
        sid = manager.start_session("test-123")
        assert sid == "test-123"
        assert manager.working.memory.core_memory_text != ""

    def test_end_session_with_summary(self, manager: MemoryManager):
        manager.initialize_sync()
        manager.start_session()
        manager.end_session("Session war erfolgreich")

        content = manager.episodic.get_today()
        assert "Session-Ende" in content

    def test_end_session_no_summary(self, manager: MemoryManager):
        manager.initialize_sync()
        manager.start_session()
        manager.end_session()  # No summary = no episode entry


class TestMemoryManagerProperties:
    def test_all_properties(self, manager: MemoryManager):
        manager.initialize_sync()
        assert manager.core is not None
        assert manager.episodic is not None
        assert manager.semantic is not None
        assert manager.procedural is not None
        assert manager.working is not None
        assert manager.index is not None
        assert manager.search is not None
        assert manager.embeddings is not None

    def test_close_sync(self, manager: MemoryManager):
        manager.initialize_sync()
        manager.close_sync()  # Should not raise
