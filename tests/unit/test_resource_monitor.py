"""Tests fuer ResourceMonitor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from jarvis.system.resource_monitor import ResourceMonitor, ResourceSnapshot


class TestResourceSnapshot:
    def test_default_not_busy(self):
        """Default Snapshot ist nicht busy."""
        snap = ResourceSnapshot()
        assert not snap.is_busy

    def test_to_dict(self):
        """to_dict enthaelt alle Felder."""
        snap = ResourceSnapshot(
            cpu_percent=42.5,
            ram_percent=60.0,
            ram_total_gb=32.0,
            ram_used_gb=19.2,
        )
        d = snap.to_dict()
        assert d["cpu_percent"] == 42.5
        assert d["ram_percent"] == 60.0
        assert d["ram_used_gb"] == 19.2
        assert d["ram_total_gb"] == 32.0
        assert "is_busy" in d
        assert "gpu_util_percent" in d
        assert "gpu_vram_used_gb" in d
        assert "gpu_vram_total_gb" in d

    def test_to_dict_rounds_values(self):
        """to_dict rundet auf eine Nachkommastelle."""
        snap = ResourceSnapshot(cpu_percent=42.567, ram_percent=60.123)
        d = snap.to_dict()
        assert d["cpu_percent"] == 42.6
        assert d["ram_percent"] == 60.1


class TestResourceMonitor:
    @pytest.fixture()
    def monitor(self):
        return ResourceMonitor(
            cpu_threshold=80.0,
            ram_threshold=90.0,
            gpu_threshold=80.0,
            cache_seconds=5.0,
        )

    def test_should_yield_no_snapshot(self, monitor):
        """Ohne Snapshot -> nicht yielden."""
        assert not monitor.should_yield()

    def test_last_snapshot_initially_none(self, monitor):
        """last_snapshot ist initial None."""
        assert monitor.last_snapshot is None

    @pytest.mark.asyncio
    async def test_sample_returns_snapshot(self, monitor):
        """sample() gibt ResourceSnapshot zurueck."""
        mock_psutil = MagicMock()
        mock_psutil.cpu_percent.return_value = 25.0
        mock_mem = MagicMock()
        mock_mem.total = 32 * 1024**3
        mock_mem.available = 20 * 1024**3
        mock_mem.percent = 37.5
        mock_psutil.virtual_memory.return_value = mock_mem

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            with patch(
                "asyncio.create_subprocess_exec",
                side_effect=FileNotFoundError,
            ):
                snap = await monitor.sample()

        assert isinstance(snap, ResourceSnapshot)
        assert snap.cpu_percent == 25.0
        assert snap.ram_percent == 37.5
        assert not snap.is_busy
        assert monitor.last_snapshot is snap

    @pytest.mark.asyncio
    async def test_sample_caching(self, monitor):
        """Zweiter sample() innerhalb cache_seconds gibt cached Ergebnis."""
        mock_psutil = MagicMock()
        mock_psutil.cpu_percent.return_value = 25.0
        mock_mem = MagicMock()
        mock_mem.total = 32 * 1024**3
        mock_mem.available = 20 * 1024**3
        mock_mem.percent = 37.5
        mock_psutil.virtual_memory.return_value = mock_mem

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            with patch(
                "asyncio.create_subprocess_exec",
                side_effect=FileNotFoundError,
            ):
                snap1 = await monitor.sample()
                snap2 = await monitor.sample()

        assert snap1 is snap2  # Same cached object

    @pytest.mark.asyncio
    async def test_sample_cache_expired(self):
        """Nach Ablauf von cache_seconds wird neu gesampled."""
        monitor = ResourceMonitor(cache_seconds=0.0)  # No caching

        mock_psutil = MagicMock()
        mock_psutil.cpu_percent.return_value = 25.0
        mock_mem = MagicMock()
        mock_mem.total = 32 * 1024**3
        mock_mem.available = 20 * 1024**3
        mock_mem.percent = 37.5
        mock_psutil.virtual_memory.return_value = mock_mem

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            with patch(
                "asyncio.create_subprocess_exec",
                side_effect=FileNotFoundError,
            ):
                snap1 = await monitor.sample()
                snap2 = await monitor.sample()

        # With cache_seconds=0, each call produces a new snapshot
        assert snap1 is not snap2

    @pytest.mark.asyncio
    async def test_sample_gpu_success(self, monitor):
        """GPU-Daten werden aus nvidia-smi gelesen."""
        mock_psutil = MagicMock()
        mock_psutil.cpu_percent.return_value = 10.0
        mock_mem = MagicMock()
        mock_mem.total = 32 * 1024**3
        mock_mem.available = 28 * 1024**3
        mock_mem.percent = 12.5
        mock_psutil.virtual_memory.return_value = mock_mem

        # Mock nvidia-smi returning valid data
        mock_proc = MagicMock()
        mock_proc.returncode = 0

        async def mock_communicate():
            return (b"45, 4096, 8192\n", b"")

        mock_proc.communicate = mock_communicate

        async def mock_subprocess(*args, **kwargs):
            return mock_proc

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            with patch(
                "asyncio.create_subprocess_exec",
                side_effect=mock_subprocess,
            ):
                snap = await monitor.sample()

        assert snap.gpu_util_percent == 45.0
        assert snap.gpu_vram_used_gb == 4.0
        assert snap.gpu_vram_total_gb == 8.0

    def test_is_busy_cpu(self, monitor):
        """CPU ueber Threshold -> busy."""
        snap = ResourceSnapshot(cpu_percent=85.0, ram_percent=50.0)
        assert monitor._is_busy(snap)

    def test_is_busy_ram(self, monitor):
        """RAM ueber Threshold -> busy."""
        snap = ResourceSnapshot(cpu_percent=30.0, ram_percent=95.0)
        assert monitor._is_busy(snap)

    def test_is_busy_gpu(self, monitor):
        """GPU ueber Threshold + VRAM vorhanden -> busy."""
        snap = ResourceSnapshot(
            cpu_percent=30.0,
            ram_percent=50.0,
            gpu_util_percent=90.0,
            gpu_vram_total_gb=8.0,
        )
        assert monitor._is_busy(snap)

    def test_not_busy_gpu_no_vram(self, monitor):
        """GPU ueber Threshold aber kein VRAM -> nicht busy (kein echtes GPU)."""
        snap = ResourceSnapshot(
            cpu_percent=30.0,
            ram_percent=50.0,
            gpu_util_percent=90.0,
            gpu_vram_total_gb=0.0,
        )
        assert not monitor._is_busy(snap)

    def test_not_busy_normal(self, monitor):
        """Alles unter Threshold -> nicht busy."""
        snap = ResourceSnapshot(
            cpu_percent=30.0,
            ram_percent=50.0,
            gpu_util_percent=20.0,
            gpu_vram_total_gb=8.0,
        )
        assert not monitor._is_busy(snap)

    def test_should_yield_after_busy_sample(self, monitor):
        """should_yield() True nach busy Snapshot."""
        snap = ResourceSnapshot(is_busy=True)
        monitor._last_snapshot = snap
        assert monitor.should_yield()

    def test_should_yield_false_after_idle_sample(self, monitor):
        """should_yield() False nach idle Snapshot."""
        snap = ResourceSnapshot(is_busy=False)
        monitor._last_snapshot = snap
        assert not monitor.should_yield()

    @pytest.mark.asyncio
    async def test_sample_psutil_missing(self, monitor):
        """psutil nicht installiert -> Snapshot mit Defaults."""
        with patch.dict("sys.modules", {"psutil": None}):
            with patch(
                "asyncio.create_subprocess_exec",
                side_effect=FileNotFoundError,
            ):
                snap = await monitor.sample()

        assert isinstance(snap, ResourceSnapshot)
        assert snap.cpu_percent == 0.0
        assert snap.ram_percent == 0.0
