"""Real-time system resource monitoring.

Samples CPU, RAM, and GPU usage for cooperative scheduling
of background tasks like the Evolution Loop.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from jarvis.utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["ResourceMonitor", "ResourceSnapshot"]


@dataclass
class ResourceSnapshot:
    """Point-in-time resource usage snapshot."""

    timestamp: float = field(default_factory=time.monotonic)
    cpu_percent: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    ram_percent: float = 0.0
    gpu_util_percent: float = 0.0  # 0 if no GPU or detection failed
    gpu_vram_used_gb: float = 0.0
    gpu_vram_total_gb: float = 0.0
    is_busy: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_percent": round(self.cpu_percent, 1),
            "ram_used_gb": round(self.ram_used_gb, 1),
            "ram_total_gb": round(self.ram_total_gb, 1),
            "ram_percent": round(self.ram_percent, 1),
            "gpu_util_percent": round(self.gpu_util_percent, 1),
            "gpu_vram_used_gb": round(self.gpu_vram_used_gb, 1),
            "gpu_vram_total_gb": round(self.gpu_vram_total_gb, 1),
            "is_busy": self.is_busy,
        }


class ResourceMonitor:
    """Monitors system resources for cooperative scheduling.

    Samples CPU/RAM via psutil. GPU via nvidia-smi (async subprocess).
    Caches results for a configurable interval to avoid overhead.
    """

    def __init__(
        self,
        cpu_threshold: float = 80.0,
        ram_threshold: float = 90.0,
        gpu_threshold: float = 80.0,
        cache_seconds: float = 5.0,
    ) -> None:
        self._cpu_threshold = cpu_threshold
        self._ram_threshold = ram_threshold
        self._gpu_threshold = gpu_threshold
        self._cache_seconds = cache_seconds
        self._last_snapshot: ResourceSnapshot | None = None
        self._last_sample_time: float = 0.0

    async def sample(self) -> ResourceSnapshot:
        """Take a resource usage snapshot.

        Returns cached snapshot if sampled within cache_seconds.
        """
        now = time.monotonic()
        if (
            self._last_snapshot is not None
            and (now - self._last_sample_time) < self._cache_seconds
        ):
            return self._last_snapshot

        snap = ResourceSnapshot(timestamp=now)

        # CPU + RAM via psutil
        try:
            import psutil

            snap.cpu_percent = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            snap.ram_total_gb = round(mem.total / (1024**3), 1)
            snap.ram_used_gb = round((mem.total - mem.available) / (1024**3), 1)
            snap.ram_percent = mem.percent
        except ImportError:
            log.debug("psutil_not_available")

        # GPU via nvidia-smi (async)
        await self._sample_gpu(snap)

        # Determine busy state
        snap.is_busy = self._is_busy(snap)

        self._last_snapshot = snap
        self._last_sample_time = now
        return snap

    async def _sample_gpu(self, snap: ResourceSnapshot) -> None:
        """Query nvidia-smi for GPU utilization asynchronously."""
        import sys

        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
            kwargs: dict[str, Any] = {
                "stdout": asyncio.subprocess.PIPE,
                "stderr": asyncio.subprocess.PIPE,
            }
            if sys.platform == "win32":
                kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
            proc = await asyncio.create_subprocess_exec(*cmd, **kwargs)
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            if proc.returncode == 0 and stdout:
                line = stdout.decode().strip().split("\n")[0]
                parts = line.split(",")
                if len(parts) >= 3:
                    snap.gpu_util_percent = float(parts[0].strip())
                    snap.gpu_vram_used_gb = round(float(parts[1].strip()) / 1024, 1)
                    snap.gpu_vram_total_gb = round(float(parts[2].strip()) / 1024, 1)
        except (FileNotFoundError, asyncio.TimeoutError, ValueError, OSError):
            pass  # No NVIDIA GPU or nvidia-smi not available

    def _is_busy(self, snap: ResourceSnapshot) -> bool:
        """Determine if system is too busy for background work."""
        if snap.cpu_percent > self._cpu_threshold:
            return True
        if snap.ram_percent > self._ram_threshold:
            return True
        if snap.gpu_util_percent > self._gpu_threshold and snap.gpu_vram_total_gb > 0:
            return True
        return False

    def should_yield(self) -> bool:
        """Check if background tasks should yield (based on last snapshot).

        Returns True if the last snapshot indicated the system is busy,
        or False if no snapshot has been taken yet (allow work).
        """
        if self._last_snapshot is None:
            return False  # No data yet, allow work
        return self._last_snapshot.is_busy

    @property
    def last_snapshot(self) -> ResourceSnapshot | None:
        """Most recent snapshot, or None if never sampled."""
        return self._last_snapshot
