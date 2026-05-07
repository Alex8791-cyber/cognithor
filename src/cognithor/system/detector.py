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


# ── Compute-Capability → Architecture lookup (extensible) ────────────────────
# Source: https://developer.nvidia.com/cuda-gpus
# Forward-Compat: unknown sm > 12.x maps to "blackwell_or_newer" baseline.
_CC_TO_ARCH: dict[str, str] = {
    "5.0": "maxwell",
    "5.2": "maxwell",
    "5.3": "maxwell",
    "6.0": "pascal",
    "6.1": "pascal",
    "6.2": "pascal",
    "7.0": "volta",
    "7.2": "volta",
    "7.5": "turing",
    "8.0": "ampere",
    "8.6": "ampere",
    "8.7": "ampere",
    "8.9": "ada",
    "9.0": "hopper",
    "10.0": "blackwell",
    "10.1": "blackwell",
    "12.0": "blackwell",
    "12.1": "blackwell",
}


def _cc_to_arch(cc: str) -> str:
    """Map compute_capability ('12.0', '8.9', …) to architecture name."""
    if not cc:
        return "unknown"
    cc = cc.strip()
    if cc in _CC_TO_ARCH:
        return _CC_TO_ARCH[cc]
    # Forward-compat: pick the closest known arch by major version
    try:
        major = int(cc.split(".")[0])
        if major >= 12:
            return "blackwell_or_newer"
        if major >= 9:
            return "hopper_or_newer"
        if major >= 8:
            return "ampere_or_newer"
        if major >= 7:
            return "volta_or_newer"
    except ValueError:
        pass
    return "unknown"


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
        total_gb: float = 0
        available_gb: float = 0
        percent: float = 0
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
                    log.debug("detector_win32_memory_query_failed")
        status = "ok" if total_gb >= 16 else "warn" if total_gb >= 8 else "fail"
        data = {"total_gb": total_gb, "available_gb": available_gb, "percent_used": percent}
        return DetectionResult(key="ram", value=f"{total_gb} GB", status=status, raw_data=data)

    def detect_gpu(self) -> DetectionResult:
        """Detect primary GPU + compute_capability + CUDA version + multi-GPU.

        Capability-relevant fields populated for L2 capability mapping:
        - vendor, model, vram_total_gb, vram_free_gb, driver
        - compute_capability (e.g. "12.0" Blackwell, "8.9" Ada, "8.6" Ampere)
        - cuda_version (driver-reported CUDA runtime)
        - architecture (derived from compute_capability)
        - all_gpus (list of dicts for multi-GPU systems)
        """
        import platform
        import subprocess
        import sys

        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version,compute_cap",
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
                # Multi-GPU: one line per GPU
                gpu_lines = [ln.strip() for ln in result.stdout.strip().splitlines() if ln.strip()]
                all_gpus: list[dict[str, Any]] = []
                for line in gpu_lines:
                    parts = [p.replace(",", ".").strip() for p in line.split(",")]
                    if len(parts) >= 5:
                        try:
                            vram_total = round(float(parts[1]) / 1024, 1)
                            vram_free = round(float(parts[2]) / 1024, 1)
                        except (ValueError, IndexError):
                            continue
                        cc = parts[4] if len(parts) > 4 else ""
                        arch = _cc_to_arch(cc)
                        all_gpus.append(
                            {
                                "vendor": "nvidia",
                                "model": parts[0],
                                "vram_total_gb": vram_total,
                                "vram_free_gb": vram_free,
                                "driver": parts[3],
                                "compute_capability": cc,
                                "architecture": arch,
                            }
                        )
                if all_gpus:
                    primary = all_gpus[0]
                    cuda_version = self._detect_cuda_runtime_version()
                    primary["cuda_version"] = cuda_version
                    name = primary["model"]
                    vram_total = primary["vram_total_gb"]
                    status = "ok" if vram_total >= 8 else "warn" if vram_total >= 4 else "fail"
                    return DetectionResult(
                        key="gpu",
                        value=f"{name} ({vram_total} GB, {primary['architecture']})",
                        status=status,
                        raw_data={
                            **primary,
                            "all_gpus": all_gpus,
                            "multi_gpu_count": len(all_gpus),
                        },
                    )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            log.debug("nvidia_smi_unavailable", error=str(exc))

        # Apple Silicon — Unified Memory
        if platform.system() == "Darwin" and "arm" in platform.machine().lower():
            data = {
                "vendor": "apple",
                "model": "Apple Silicon",
                "vram_total_gb": 0,  # Unified, not separately addressable
                "unified_memory": True,
                "compute_capability": None,
                "cuda_version": None,
                "architecture": "apple_silicon",
            }
            return DetectionResult(
                key="gpu", value="Apple Silicon (unified)", status="warn", raw_data=data
            )

        # AMD ROCm — falls rocminfo verfügbar, später durch detect_rocm angereichert
        return DetectionResult(
            key="gpu",
            value="No dedicated GPU",
            status="fail",
            raw_data={
                "vendor": "none",
                "vram_total_gb": 0,
                "compute_capability": None,
                "cuda_version": None,
                "architecture": None,
            },
        )

    def _detect_cuda_runtime_version(self) -> str | None:
        """Driver-reported CUDA runtime version.

        Source order:
        1. `nvidia-smi --query-gpu=cuda_version` (newer drivers; fails on older).
        2. Plain `nvidia-smi` header line "CUDA Version: X.Y".
        3. `nvcc --version` (only if CUDA-Toolkit installed).
        """
        import re
        import subprocess
        import sys

        kwargs: dict[str, Any] = {"capture_output": True, "text": True, "timeout": 10}
        if sys.platform == "win32":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        # Source 1: --query-gpu=cuda_version (works on cu13+ drivers)
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader"],
                **kwargs,
            )
            if r.returncode == 0 and r.stdout.strip():
                v = r.stdout.strip().splitlines()[0].replace(",", ".").strip()
                if re.fullmatch(r"\d+\.\d+", v):
                    return v
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Source 2: parse plain nvidia-smi header
        try:
            r = subprocess.run(["nvidia-smi"], **kwargs)
            if r.returncode == 0:
                m = re.search(r"CUDA Version:\s*([\d.]+)", r.stdout)
                if m:
                    return m.group(1)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Source 3: nvcc (CUDA-Toolkit, optional)
        try:
            r = subprocess.run(["nvcc", "--version"], **kwargs)
            if r.returncode == 0:
                m = re.search(r"release\s+([\d.]+)", r.stdout)
                if m:
                    return m.group(1)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return None

    def detect_docker(self) -> DetectionResult:
        """Detect Docker availability + running state. Required for vLLM-Container."""
        import subprocess
        import sys

        kwargs: dict[str, Any] = {"capture_output": True, "text": True, "timeout": 5}
        if sys.platform == "win32":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            ver = subprocess.run(["docker", "--version"], **kwargs)
            if ver.returncode != 0:
                return DetectionResult(
                    key="docker",
                    value="Docker not installed",
                    status="fail",
                    raw_data={"installed": False, "running": False},
                )
            info = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], **kwargs)
            running = info.returncode == 0 and bool(info.stdout.strip())
            compose = subprocess.run(["docker", "compose", "version"], **kwargs)
            data = {
                "installed": True,
                "running": running,
                "version": ver.stdout.strip().split()[-1] if ver.stdout.strip() else "",
                "server_version": info.stdout.strip() if running else "",
                "compose_available": compose.returncode == 0,
            }
            return DetectionResult(
                key="docker",
                value=f"Docker {data['version']} ({'running' if running else 'stopped'})",
                status="ok" if running else "warn",
                raw_data=data,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return DetectionResult(
                key="docker",
                value="Docker not installed",
                status="fail",
                raw_data={"installed": False, "running": False},
            )

    def detect_wsl2(self) -> DetectionResult:
        """Windows-only: detect WSL2 availability for vLLM-Container path."""
        import subprocess
        import sys

        if sys.platform != "win32":
            return DetectionResult(
                key="wsl2",
                value="N/A (not Windows)",
                status="ok",
                raw_data={"applicable": False},
            )
        try:
            result = subprocess.run(
                ["wsl", "--status"],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            # `wsl --status` exit-code 0 = WSL installed AND has default distro
            output = (result.stdout or "") + (result.stderr or "")
            installed = result.returncode == 0
            default_v2 = "version: 2" in output.lower() or "wsl 2" in output.lower()
            return DetectionResult(
                key="wsl2",
                value=f"WSL2 {'available' if installed else 'not installed'}",
                status="ok" if installed else "warn",
                raw_data={"applicable": True, "installed": installed, "default_v2": default_v2},
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return DetectionResult(
                key="wsl2",
                value="WSL not available",
                status="warn",
                raw_data={"applicable": True, "installed": False},
            )

    def detect_container(self) -> DetectionResult:
        """Detect if Cognithor itself is running inside a container."""
        in_container = Path("/.dockerenv").exists()
        runtime = None
        if in_container:
            runtime = "docker"
        else:
            cgroup = Path("/proc/1/cgroup")
            if cgroup.exists():
                try:
                    content = cgroup.read_text()
                    if "docker" in content:
                        in_container, runtime = True, "docker"
                    elif "containerd" in content:
                        in_container, runtime = True, "containerd"
                    elif "podman" in content:
                        in_container, runtime = True, "podman"
                except OSError:
                    pass
        return DetectionResult(
            key="container",
            value=f"in container ({runtime})" if in_container else "host",
            status="ok",
            raw_data={"in_container": in_container, "runtime": runtime},
        )

    def detect_rocm(self) -> DetectionResult:
        """AMD ROCm toolchain detection."""
        import subprocess
        import sys

        if sys.platform == "win32":
            return DetectionResult(
                key="rocm",
                value="N/A (Windows)",
                status="ok",
                raw_data={"available": False, "applicable": False},
            )
        for cmd in (["rocm-smi", "--showid"], ["rocminfo"]):
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and result.stdout.strip():
                    return DetectionResult(
                        key="rocm",
                        value="ROCm available",
                        status="ok",
                        raw_data={"available": True, "tool": cmd[0]},
                    )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        return DetectionResult(
            key="rocm",
            value="ROCm not detected",
            status="warn",
            raw_data={"available": False, "applicable": True},
        )

    def detect_vllm(self) -> DetectionResult:
        """vLLM availability — pip-installed or running container at :8000."""
        import subprocess
        import sys
        import urllib.error
        import urllib.request

        kwargs: dict[str, Any] = {"capture_output": True, "text": True, "timeout": 5}
        if sys.platform == "win32":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        pip_installed = False
        pip_version = ""
        try:
            result = subprocess.run(["pip", "show", "vllm"], **kwargs)
            if result.returncode == 0:
                pip_installed = True
                for line in result.stdout.splitlines():
                    if line.lower().startswith("version:"):
                        pip_version = line.split(":", 1)[1].strip()
                        break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Check running OpenAI-compatible server on default port
        server_running = False
        try:
            req = urllib.request.Request("http://127.0.0.1:8000/v1/models")
            with urllib.request.urlopen(req, timeout=2) as resp:
                server_running = resp.status == 200
        except (urllib.error.URLError, OSError, ValueError):
            pass

        status = "ok" if (pip_installed or server_running) else "warn"
        value = (
            f"vLLM {pip_version} (pip)"
            if pip_installed
            else "vLLM container (port 8000)"
            if server_running
            else "vLLM not detected"
        )
        return DetectionResult(
            key="vllm",
            value=value,
            status=status,
            raw_data={
                "pip_installed": pip_installed,
                "pip_version": pip_version,
                "server_running": server_running,
            },
        )

    def detect_huggingface(self) -> DetectionResult:
        """HuggingFace Hub reachability + token presence."""
        import os
        import urllib.error
        import urllib.request

        token_present = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
        reachable = False
        try:
            req = urllib.request.Request(
                "https://huggingface.co/api/models?limit=1",
                headers={"User-Agent": "cognithor-detect/1"},
            )
            with urllib.request.urlopen(req, timeout=4) as resp:
                reachable = resp.status == 200
        except (urllib.error.URLError, OSError, ValueError):
            pass

        return DetectionResult(
            key="huggingface",
            value=(
                "HF reachable + token"
                if reachable and token_present
                else "HF reachable"
                if reachable
                else "HF unreachable"
            ),
            status="ok" if reachable else "warn",
            raw_data={"reachable": reachable, "token_present": token_present},
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
            log.debug("detector_ollama_api_check_failed")
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
            log.debug("detector_lmstudio_api_check_failed")
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
            # Hardware-Aware-Runtime additions (L1)
            self.detect_docker,
            self.detect_wsl2,
            self.detect_container,
            self.detect_rocm,
            self.detect_vllm,
            self.detect_huggingface,
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
