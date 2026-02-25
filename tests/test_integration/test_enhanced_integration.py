"""Integration-Tests: Enhanced Memory + Executor Gap Detection.

Testet die Verdrahtung aller neuen Komponenten:
  - MemoryManager: EnhancedSearch, GraphRanking, Multimodal, Compressor
  - Executor: GapDetector Integration
  - End-to-End: Query → EnhancedPipeline → GraphBoost → Results
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.config import JarvisConfig
from jarvis.core.executor import Executor
from jarvis.memory.enhanced_retrieval import EnhancedSearchPipeline, FrequencyTracker
from jarvis.memory.graph_ranking import GraphRanking
from jarvis.memory.multimodal import MultimodalMemory
from jarvis.models import (
    GateDecision,
    GateStatus,
    PlannedAction,
    ToolResult,
)
from jarvis.skills.generator import GapDetector, SkillGapType


# ============================================================================
# MemoryManager Integration
# ============================================================================


class TestMemoryManagerEnhancedProperties:
    """MemoryManager hat alle neuen Properties."""

    @pytest.fixture
    def manager(self, tmp_path: Path):
        """Erstellt einen MemoryManager mit tmp_path."""
        with (
            patch("jarvis.memory.manager.EmbeddingClient"),
            patch("jarvis.memory.manager.CoreMemory"),
        ):
            from jarvis.memory.manager import MemoryManager

            config = JarvisConfig(
                jarvis_dir=tmp_path / ".jarvis",
                memory_dir=tmp_path / ".jarvis" / "memory",
            )
            mm = MemoryManager(config)
            return mm

    def test_has_enhanced_search(self, manager) -> None:
        assert isinstance(manager.enhanced_search, EnhancedSearchPipeline)

    def test_has_graph_ranking(self, manager) -> None:
        assert isinstance(manager.graph_ranking, GraphRanking)

    def test_has_multimodal(self, manager) -> None:
        assert isinstance(manager.multimodal, MultimodalMemory)

    def test_has_frequency_tracker(self, manager) -> None:
        assert isinstance(manager.frequency_tracker, FrequencyTracker)

    def test_has_compressor(self, manager) -> None:
        from jarvis.memory.enhanced_retrieval import EpisodicCompressor

        assert isinstance(manager.compressor, EpisodicCompressor)

    def test_set_media_pipeline(self, manager) -> None:
        mock_pipeline = MagicMock()
        manager.set_media_pipeline(mock_pipeline)
        assert manager.multimodal._pipeline is mock_pipeline

    def test_stats_include_new_fields(self, manager) -> None:
        stats = manager.stats()
        assert "multimodal_assets" in stats
        assert "graph_ranking_computed" in stats
        assert "frequency_tracked_chunks" in stats
        assert stats["multimodal_assets"] == 0
        assert stats["graph_ranking_computed"] is False


# ============================================================================
# Executor + GapDetector Integration
# ============================================================================


class TestExecutorGapDetection:
    """Executor meldet Fehler an GapDetector."""

    @pytest.fixture
    def gap_detector(self) -> GapDetector:
        return GapDetector()

    @pytest.fixture
    def executor(self, gap_detector: GapDetector) -> Executor:
        config = MagicMock()
        mock_client = AsyncMock()
        exec_ = Executor(config, mcp_client=mock_client, gap_detector=gap_detector)
        return exec_

    @pytest.mark.asyncio
    async def test_non_retryable_error_reports_gap(
        self,
        executor: Executor,
        gap_detector: GapDetector,
    ) -> None:
        """Nicht-wiederholbare Fehler werden als Gap gemeldet."""
        executor._mcp_client.call_tool = AsyncMock(
            side_effect=PermissionError("Tool nicht verfügbar"),
        )

        result = await executor._execute_single("unknown_tool", {})
        assert result.is_error

        gaps = gap_detector.get_all_gaps()
        assert len(gaps) == 1
        assert gaps[0].tool_name == "unknown_tool"
        assert gaps[0].gap_type == SkillGapType.UNKNOWN_TOOL

    @pytest.mark.asyncio
    async def test_retries_exhausted_reports_repeated_failure(
        self,
        executor: Executor,
        gap_detector: GapDetector,
    ) -> None:
        """Nach erschöpften Retries wird REPEATED_FAILURE gemeldet."""
        executor._max_retries = 2
        executor._base_delay = 0.01
        executor._mcp_client.call_tool = AsyncMock(
            side_effect=TimeoutError("Timeout"),
        )

        result = await executor._execute_single("slow_tool", {})
        assert result.is_error

        gaps = gap_detector.get_all_gaps()
        assert len(gaps) == 1
        assert gaps[0].gap_type == SkillGapType.REPEATED_FAILURE

    @pytest.mark.asyncio
    async def test_successful_call_no_gap(
        self,
        executor: Executor,
        gap_detector: GapDetector,
    ) -> None:
        """Erfolgreiche Tool-Calls erzeugen keine Gaps."""
        mock_result = MagicMock()
        mock_result.content = "Erfolg"
        mock_result.is_error = False
        executor._mcp_client.call_tool = AsyncMock(return_value=mock_result)

        result = await executor._execute_single("working_tool", {})
        assert result.success

        assert gap_detector.gap_count == 0

    @pytest.mark.asyncio
    async def test_no_gap_detector_no_crash(self) -> None:
        """Executor ohne GapDetector crasht nicht bei Fehlern."""
        config = MagicMock()
        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(
            side_effect=PermissionError("Nope"),
        )
        exec_ = Executor(config, mcp_client=mock_client, gap_detector=None)

        result = await exec_._execute_single("tool", {})
        assert result.is_error


# ============================================================================
# Enhanced Search + Graph Ranking End-to-End
# ============================================================================


class TestEnhancedSearchWithGraphBoost:
    """End-to-End: Enhanced Search Pipeline + Graph Ranking Boost."""

    @pytest.fixture
    def manager_with_data(self, tmp_path: Path):
        with (
            patch("jarvis.memory.manager.EmbeddingClient"),
            patch("jarvis.memory.manager.CoreMemory"),
        ):
            from jarvis.memory.manager import MemoryManager

            config = JarvisConfig(
                jarvis_dir=tmp_path / ".jarvis",
                memory_dir=tmp_path / ".jarvis" / "memory",
            )
            mm = MemoryManager(config)

            mm.index_text(
                "WWK Berufsunfähigkeitsversicherung Premium-Tarif",
                "bu_wwk.md",
            )
            mm.index_text(
                "Allianz BU-Vergleich und Marktanalyse",
                "bu_allianz.md",
            )
            mm.index_text(
                "Jarvis Agent OS Architektur-Dokument",
                "jarvis_arch.md",
            )

            return mm

    @pytest.mark.asyncio
    async def test_search_memory_enhanced_mode(self, manager_with_data) -> None:
        results = await manager_with_data.search_memory(
            "BU Tarif",
            enhanced=True,
            top_k=3,
        )
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_memory_basic_mode(self, manager_with_data) -> None:
        results = await manager_with_data.search_memory(
            "BU Tarif",
            enhanced=False,
            top_k=3,
        )
        assert isinstance(results, list)

    def test_frequency_tracking_persists(self, manager_with_data) -> None:
        tracker = manager_with_data.frequency_tracker
        tracker.record_access("chunk_1")
        tracker.record_access("chunk_1")
        tracker.record_access("chunk_2")

        assert tracker.total_accesses == 3
        assert tracker.get_count("chunk_1") == 2

    def test_graph_ranking_computable(self, manager_with_data) -> None:
        ranking = manager_with_data.graph_ranking
        ranks = ranking.compute_pagerank()
        assert isinstance(ranks, dict)


# ============================================================================
# MemoryManager search_memory signature
# ============================================================================


class TestSearchMemorySignature:
    """search_memory unterstützt enhanced=True/False."""

    @pytest.fixture
    def manager(self, tmp_path: Path):
        with (
            patch("jarvis.memory.manager.EmbeddingClient"),
            patch("jarvis.memory.manager.CoreMemory"),
        ):
            from jarvis.memory.manager import MemoryManager

            config = JarvisConfig(
                jarvis_dir=tmp_path / ".jarvis",
                memory_dir=tmp_path / ".jarvis" / "memory",
            )
            return MemoryManager(config)

    @pytest.mark.asyncio
    async def test_default_is_enhanced(self, manager) -> None:
        results = await manager.search_memory("test query")
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_explicit_not_enhanced(self, manager) -> None:
        results = await manager.search_memory("test", enhanced=False)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_with_tier_filter(self, manager) -> None:
        from jarvis.models import MemoryTier

        results = await manager.search_memory("test", tier=MemoryTier.CORE)
        assert isinstance(results, list)
