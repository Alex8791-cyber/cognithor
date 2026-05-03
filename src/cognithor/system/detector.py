"""System Detector — comprehensive hardware and software profiling.

Detects CPU, RAM, GPU, disk, network, Ollama, LM Studio at startup.
Results cached to ~/.cognithor/system_profile.json.
"""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cognithor.utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["DetectionResult", "SystemDetector", "SystemProfile"]


@dataclass
class DetectionResult:
    """Result of a single detection target."""

    key: str
    value: str
    status: str  # "ok" | "warn" | "fail"
    raw_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "status": self.status,
            "raw_data": self.raw_data,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DetectionResult:
        return cls(
            key=d["key"], value=d["value"], status=d["status"], raw_data=d.get("raw_data", {})
        )


@dataclass
class SystemProfile:
    """Complete system profile with all detection results."""

    results: dict[str, DetectionResult] = field(default_factory=dict)
    detected_at: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )

    def get_tier(self) -> str:
        ram_gb = self.results.get("ram", DetectionResult("ram", "", "fail", {})).raw_data.get(
            "total_gb", 0
        )
        cores = self.results.get("cpu", DetectionResult("cpu", "", "fail", {})).raw_data.get(
            "physical_cores", 0
        )
        vram = self.results.get("gpu", DetectionResult("gpu", "", "fail", {})).raw_data.get(
            "vram_total_gb", 0
        )
        if ram_gb >= 64 and cores >= 16 and vram >= 48:
            return "enterprise"
        if vram >= 16 and ram_gb >= 32:
            return "power"
        if vram >= 8 and ram_gb >= 16:
            return "standard"
        return "minimal"

    def get_available_modes(self) -> list[str]:
        modes = []
        ollama = self.results.get("ollama", DetectionResult("ollama", "", "fail", {}))
        network = self.results.get("network", DetectionResult("network", "", "fail", {}))
        gpu = self.results.get("gpu", DetectionResult("gpu", "", "fail", {}))
        if ollama.raw_data.get("running") or gpu.raw_data.get("vram_total_gb", 0) >= 4:
            modes.append("offline")
        if network.raw_data.get("internet"):
            modes.append("online")
        if "offline" in modes and "online" in modes:
            modes.append("hybrid")
        if not modes:
            modes.append("offline")
        return modes

    def get_recommended_mode(self) -> str:
        modes = self.get_available_modes()
        ollama = self.results.get("ollama", DetectionResult("ollama", "", "fail", {}))
        gpu = self.results.get("gpu", DetectionResult("gpu", "", "fail", {}))
        if (
            "offline" in modes
            and ollama.raw_data.get("running")
            and gpu.raw_data.get("vram_total_gb", 0) >= 8
        ):
            return "offline"
        if "hybrid" in modes:
            return "hybrid"
        if "online" in modes:
            return "online"
        return "offline"

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected_at": self.detected_at,
            "tier": self.get_tier(),
            "recommended_mode": self.get_recommended_mode(),
            "available_modes": self.get_available_modes(),
            "results": {k: v.to_dict() for k, v in self.results.items()},
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> SystemProfile | None:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            profile = cls(detected_at=data.get("detected_at", ""))
            for k, v in data.get("results", {}).items():
                profile.results[k] = DetectionResult.from_dict(v)
            return profile
        except Exception:
            return None


class SystemDetector:
    """Detects hardware and software environment."""

    def detect_os(self) -> DetectionResult:
        import platform
        import sys

        data = {
            "os": platform.system().lower(),
            "version": platform.release(),
            "arch": platform.machine(),
            "python": platform.python_version(),
            "is_wsl": (
                "microsoft" in platform.release().lower() if sys.platform == "linux" else False
            ),
        }
        return DetectionResult(
            key="os",
            value=f"{platform.system()} {platform.release()}",
            status="ok",
            raw_data=data,
        )

    def detect_cpu(self) -> DetectionResult:
        import os
        import platform

        cores = os.cpu_count() or 1
        physical = cores // 2 if cores > 1 else 1
        try:
            import psutil

            physical = psutil.cpu_count(logical=False) or physical
            freq = psutil.cpu_freq()
            max_freq = freq.max if freq else 0
        except ImportError:
            max_freq = 0
        name = platform.processor() or platform.machine()
        status = "ok" if physical >= 4 else "warn" if physical >= 2 else "fail"
        data = {
            "model": name,
            "physical_cores": physical,
            "logical_cores": cores,
            "max_freq_mhz": max_freq,
        }
        return DetectionResult(
            key="cpu", value=f"{name} ({physical}C/{cores}T)", status=status, raw_data=data
        )

    def detect_ram(self) -> DetectionResult:
        total_gb = 0
        available_gb = 0
        percent = 0
        try:
            import psutil

            mem = psutil.virtual_memory()
            total_gb = round(mem.total / (1024**3), 1)
            available_gb = round(mem.available / (1024**3), 1)
            percent = mem.percent
        except ImportError:
            # Fallback: platform-specific
            import sys

            if sys.platform == "win32":
                try:
                    import ctypes

                    mem_kb = ctypes.c_ulonglong(0)
                    ctypes.windll.kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(mem_kb))
                    total_gb = round(mem_kb.value / (1024 * 1024), 1)
                except Exception:
                    log.debug("detector_win32_memory_query_failed", exc_info=True)
        status = "ok" if total_gb >= 16 else "warn" if total_gb >= 8 else "fail"
        data = {"total_gb": total_gb, "available_gb": available_gb, "percent_used": percent}
        return DetectionResult(key="ram", value=f"{total_gb} GB", status=status, raw_data=data)

    def detect_gpu(self) -> DetectionResult:
        import platform
        import subprocess
        import sys

        # Try nvidia-smi
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ]
            if sys.platform == "win32":
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            else:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(",")
                if len(parts) >= 4:
                    name = parts[0].strip()
                    vram_total = round(float(parts[1].strip()) / 1024, 1)
                    vram_free = round(float(parts[2].strip()) / 1024, 1)
                    driver = parts[3].strip()
                    status = "ok" if vram_total >= 8 else "warn" if vram_total >= 4 else "fail"
                    data = {
                        "vendor": "nvidia",
                        "model": name,
                        "vram_total_gb": vram_total,
                        "vram_free_gb": vram_free,
                        "driver": driver,
                    }
                    return DetectionResult(
                        key="gpu",
                        value=f"{name} ({vram_total} GB)",
                        status=status,
                        raw_data=data,
                    )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        # Check Apple Silicon
        if platform.system() == "Darwin" and "arm" in platform.machine().lower():
            data = {
                "vendor": "apple",
                "model": "Apple Silicon",
                "vram_total_gb": 0,
                "unified_memory": True,
            }
            return DetectionResult(
                key="gpu", value="Apple Silicon (unified)", status="warn", raw_data=data
            )
        return DetectionResult(
            key="gpu",
            value="No dedicated GPU",
            status="fail",
            raw_data={"vendor": "none", "vram_total_gb": 0},
        )

    def detect_disk(self) -> DetectionResult:
        import shutil

        home = Path.home() / ".cognithor"
        usage = shutil.disk_usage(str(home.parent))
        free_gb = round(usage.free / (1024**3), 1)
        total_gb = round(usage.total / (1024**3), 1)
        status = "ok" if free_gb >= 50 else "warn" if free_gb >= 10 else "fail"
        data = {"path": str(home.parent), "total_gb": total_gb, "free_gb": free_gb}
        return DetectionResult(key="disk", value=f"{free_gb} GB free", status=status, raw_data=data)

    def detect_network(self) -> DetectionResult:
        import urllib.request

        internet = False
        # Use reliable, fast endpoints for connectivity check
        for url in (
            "https://www.google.com/generate_204",
            "https://connectivitycheck.gstatic.com/generate_204",
            "https://1.1.1.1",
        ):
            try:
                urllib.request.urlopen(url, timeout=5)
                internet = True
                break
            except Exception:
                continue
        status = "ok" if internet else "fail"
        data = {"internet": internet}
        return DetectionResult(
            key="network",
            value="Connected" if internet else "No internet",
            status=status,
            raw_data=data,
        )

    def detect_ollama(self) -> DetectionResult:
        import json as json_mod
        import shutil
        import urllib.request

        # Check if running
        try:
            with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as resp:
                data = json_mod.loads(resp.read())
                models = [m.get("name", "") for m in data.get("models", [])]
                return DetectionResult(
                    key="ollama",
                    value=f"Running ({len(models)} models)",
                    status="ok",
                    raw_data={
                        "installed": True,
                        "running": True,
                        "models": models,
                        "endpoint": "http://localhost:11434",
                    },
                )
        except Exception:
            log.debug("detector_ollama_api_check_failed", exc_info=True)
        # Check if installed
        if shutil.which("ollama"):
            return DetectionResult(
                key="ollama",
                value="Installed (not running)",
                status="warn",
                raw_data={"installed": True, "running": False, "models": []},
            )
        return DetectionResult(
            key="ollama",
            value="Not installed",
            status="fail",
            raw_data={"installed": False, "running": False, "models": []},
        )

    def detect_lmstudio(self) -> DetectionResult:
        import json as json_mod
        import urllib.request

        try:
            with urllib.request.urlopen("http://localhost:1234/v1/models", timeout=3) as resp:
                data = json_mod.loads(resp.read())
                models = [m.get("id", "") for m in data.get("data", [])]
                return DetectionResult(
                    key="lmstudio",
                    value=f"Running ({len(models)} models)",
                    status="ok",
                    raw_data={"installed": True, "running": True, "models": models},
                )
        except Exception:
            log.debug("detector_lmstudio_api_check_failed", exc_info=True)
        return DetectionResult(
            key="lmstudio",
            value="Not available",
            status="fail",
            raw_data={"installed": False, "running": False, "models": []},
        )

    def run_full_scan(self) -> SystemProfile:
        profile = SystemProfile()
        for detect_fn in [
            self.detect_os,
            self.detect_cpu,
            self.detect_ram,
            self.detect_gpu,
            self.detect_disk,
            self.detect_network,
            self.detect_ollama,
            self.detect_lmstudio,
        ]:
            try:
                result = detect_fn()
                profile.results[result.key] = result
            except Exception as exc:
                log.debug("detection_failed", target=detect_fn.__name__, error=str(exc))
        return profile

    def run_quick_scan(self, cache_path: Path | None = None) -> SystemProfile:
        # Use cached results for stable items, re-scan volatile items
        cached = SystemProfile.load(cache_path) if cache_path else None
        profile = SystemProfile()
        # Stable: OS, CPU, RAM, disk — use cache if available
        for key in ("os", "cpu", "ram", "disk"):
            if cached and key in cached.results:
                profile.results[key] = cached.results[key]
            else:
                detect_fn = getattr(self, f"detect_{key}")
                profile.results[key] = detect_fn()
        # Volatile: GPU state, network, Ollama, LMStudio — always re-scan
        for key in ("gpu", "network", "ollama", "lmstudio"):
            detect_fn = getattr(self, f"detect_{key}")
            with contextlib.suppress(Exception):
                profile.results[key] = detect_fn()
        return profile
