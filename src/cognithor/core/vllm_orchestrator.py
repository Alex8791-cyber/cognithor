"""vLLM lifecycle orchestrator — wraps docker/nvidia-smi subprocesses.

Stateful manager: hardware detection, Docker readiness, image pull,
container start/stop, model recommendation. No Docker-SDK dependency —
pure `subprocess` calls.

See spec: docs/superpowers/specs/2026-04-22-vllm-opt-in-backend-design.md
"""

from __future__ import annotations

import collections
import dataclasses
import json as _json
import re
import socket
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from cognithor.core.llm_backend import VLLMDockerError, VLLMHardwareError, VLLMNotReadyError
from cognithor.utils.logging import get_logger

log = get_logger(__name__)

ProgressCallback = Callable[[dict[str, Any]], None] | None

Priority = Literal["premium", "standard", "fallback"]
Capability = Literal["vision", "text"]


@dataclass
class HardwareInfo:
    """NVIDIA GPU detection result."""

    gpu_name: str
    vram_gb: int
    compute_capability: tuple[int, int]

    @property
    def sm_string(self) -> str:
        """Returns the compute capability as 'major.minor' string."""
        return f"{self.compute_capability[0]}.{self.compute_capability[1]}"


@dataclass
class DockerInfo:
    """Docker Desktop readiness."""

    available: bool
    version: str = ""
    server_running: bool = False


@dataclass
class ContainerInfo:
    """A running/started vLLM container."""

    container_id: str
    port: int
    model: str


@dataclass
class ModelEntry:
    """One row from the model_registry.json vllm provider section."""

    id: str
    display_name: str
    base_model: str
    quantization: str
    vram_gb_min: int
    min_compute_capability: str
    min_vllm_version: str
    capability: Capability
    priority: Priority
    tested: bool
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelEntry:
        return cls(
            id=data["id"],
            display_name=data["display_name"],
            base_model=data["base_model"],
            quantization=data["quantization"],
            vram_gb_min=int(data["vram_gb_min"]),
            min_compute_capability=data["min_compute_capability"],
            min_vllm_version=data["min_vllm_version"],
            capability=data["capability"],
            priority=data["priority"],
            tested=bool(data["tested"]),
            notes=data.get("notes", ""),
        )

    @property
    def min_cc_tuple(self) -> tuple[int, int]:
        """Returns min_compute_capability as (major, minor) tuple."""
        parts = self.min_compute_capability.split(".")
        return (int(parts[0]), int(parts[1]))


@dataclass
class VLLMState:
    """Aggregate state snapshot for UI rendering."""

    hardware_ok: bool = False
    hardware_info: HardwareInfo | None = None
    docker_ok: bool = False
    docker_info: DockerInfo | None = None
    image_pulled: bool = False
    container_running: bool = False
    current_model: str | None = None
    last_error: str | None = None


class VLLMOrchestrator:
    """Stateful vLLM lifecycle manager. Methods added in later tasks."""

    _PRIORITY_ORDER: dict[str, int] = {"premium": 0, "standard": 1, "fallback": 2}
    _MAX_PORT_FALLBACKS = 10

    def __init__(
        self,
        *,
        docker_image: str = "vllm/vllm-openai:v0.19.1",
        port: int = 8000,
        hf_token: str = "",
        log_ring_size: int = 500,
    ) -> None:
        self.docker_image = docker_image
        self.port = port
        self._hf_token = hf_token
        self.state = VLLMState()
        self._log_ring: collections.deque[str] = collections.deque(maxlen=log_ring_size)

    def get_logs(self) -> list[str]:
        """Snapshot of the container-log ring buffer."""
        return list(self._log_ring)

    def status(self) -> VLLMState:
        """Return a snapshot copy of the current state (safe to mutate)."""
        return dataclasses.replace(self.state)

    def pull_image(
        self,
        tag: str,
        *,
        progress_callback: ProgressCallback = None,
    ) -> None:
        """Run ``docker pull`` streaming JSON progress to the callback.

        Raises:
            VLLMDockerError: if the pull exits non-zero.
        """
        cmd = ["docker", "pull", "--progress=auto", tag]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            for line in proc.stdout or []:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = _json.loads(line)
                except _json.JSONDecodeError:
                    event = {"status": line}
                if progress_callback is not None:
                    progress_callback(event)
        finally:
            proc.wait()

        if proc.returncode != 0:
            raise VLLMDockerError(
                f"docker pull {tag} failed with exit {proc.returncode}",
                recovery_hint="Check Docker Desktop is running and you have network access.",
            )

        self.state.image_pulled = True

    def check_hardware(self) -> HardwareInfo:
        """Detect NVIDIA GPU. Raises VLLMHardwareError on any failure."""
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        except FileNotFoundError as exc:
            raise VLLMHardwareError(
                "nvidia-smi not found — NVIDIA driver not installed?",
                recovery_hint="Install the NVIDIA GPU driver from nvidia.com.",
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise VLLMHardwareError(
                "nvidia-smi timed out",
                recovery_hint="Check GPU driver health.",
            ) from exc

        if result.returncode != 0:
            raise VLLMHardwareError(
                f"nvidia-smi failed: {result.stderr.strip() or 'unknown error'}",
            )

        first_line = result.stdout.strip().split("\n")[0] if result.stdout.strip() else ""
        if not first_line:
            raise VLLMHardwareError("No NVIDIA GPU detected")

        parts = [p.strip() for p in first_line.split(",")]
        if len(parts) < 3:
            raise VLLMHardwareError(f"Unexpected nvidia-smi output: {first_line!r}")

        gpu_name = parts[0]
        try:
            vram_mib = int(parts[1])
            cc_parts = parts[2].split(".")
            compute_capability = (int(cc_parts[0]), int(cc_parts[1]))
        except (ValueError, IndexError) as exc:
            raise VLLMHardwareError(f"Cannot parse nvidia-smi output: {first_line!r}") from exc

        info = HardwareInfo(
            gpu_name=gpu_name,
            vram_gb=round(vram_mib / 1024),
            compute_capability=compute_capability,
        )
        self.state.hardware_info = info
        self.state.hardware_ok = True
        return info

    def check_docker(self) -> DockerInfo:
        """Detect Docker Desktop. Never raises — returns DockerInfo with flags."""
        try:
            result = subprocess.run(
                ["docker", "version", "--format", "json"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except FileNotFoundError:
            info = DockerInfo(available=False)
            self.state.docker_ok = False
            self.state.docker_info = info
            return info
        except subprocess.TimeoutExpired:
            info = DockerInfo(available=True, server_running=False)
            self.state.docker_ok = False
            self.state.docker_info = info
            return info

        if result.returncode != 0:
            info = DockerInfo(available=True, server_running=False)
            self.state.docker_ok = False
            self.state.docker_info = info
            return info

        try:
            parsed = _json.loads(result.stdout)
        except _json.JSONDecodeError:
            info = DockerInfo(available=True, server_running=False)
            self.state.docker_ok = False
            self.state.docker_info = info
            return info

        server = parsed.get("Server")
        version = (server or parsed.get("Client", {})).get("Version", "")
        info = DockerInfo(
            available=True,
            version=version,
            server_running=server is not None,
        )
        self.state.docker_ok = info.server_running
        self.state.docker_info = info
        return info

    def filter_registry(
        self,
        hardware: HardwareInfo,
        registry: list[ModelEntry],
    ) -> list[ModelEntry]:
        """Returns entries that fit the detected GPU (VRAM + compute capability)."""
        return [
            m
            for m in registry
            if m.vram_gb_min <= hardware.vram_gb and m.min_cc_tuple <= hardware.compute_capability
        ]

    def recommend_model(
        self,
        hardware: HardwareInfo,
        registry: list[ModelEntry],
        *,
        prefer: Literal["vision", "text"] = "vision",
    ) -> ModelEntry | None:
        """Pick the best curated model for detected hardware.

        Ranking:
          1. Matches requested capability (vision/text)
          2. tested==True beats tested==False
          3. Higher priority (premium > standard > fallback)
          4. Tie-break: smaller vram_gb_min (more headroom for KV cache)
          5. Stable min_vllm_version before "pending"
        """
        candidates = [m for m in self.filter_registry(hardware, registry) if m.capability == prefer]
        if not candidates:
            return None

        def sort_key(m: ModelEntry) -> tuple[int, int, int, int]:
            return (
                0 if m.tested else 1,
                self._PRIORITY_ORDER[m.priority],
                m.vram_gb_min,
                0 if m.min_vllm_version != "pending" else 1,
            )

        candidates.sort(key=sort_key)
        return candidates[0]

    def _port_available(self, port: int) -> bool:
        """True if nothing is listening on localhost:port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            try:
                s.bind(("127.0.0.1", port))
                return True
            except OSError:
                return False

    def _wait_for_health(self, port: int, *, timeout: int = 120) -> bool:
        """Poll vLLM /health until 200 or timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with httpx.Client(timeout=2.0) as client:
                    r = client.get(f"http://localhost:{port}/health")
                    if r.status_code == 200:
                        return True
            except Exception:
                pass
            time.sleep(2.0)
        return False

    def start_container(
        self,
        model: str,
        *,
        health_timeout: int | None = None,
    ) -> ContainerInfo:
        """Start a vLLM container. Auto-falls-back self.port → self.port+N on port conflict."""
        port = self.port
        for offset in range(self._MAX_PORT_FALLBACKS):
            candidate = self.port + offset
            if self._port_available(candidate):
                port = candidate
                break
        else:
            raise VLLMNotReadyError(
                f"All ports {self.port}..{self.port + self._MAX_PORT_FALLBACKS - 1} are busy",
                recovery_hint="Stop other services or change config.vllm.port.",
            )

        cmd = [
            "docker",
            "run",
            "-d",
            "--gpus",
            "all",
            "-v",
            "cognithor-hf-cache:/root/.cache/huggingface",
            "-e",
            f"HF_TOKEN={self._hf_token}",
            "-p",
            f"{port}:8000",
            "--label",
            "cognithor.managed=true",
            self.docker_image,
            "--model",
            model,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise VLLMNotReadyError(
                f"docker run failed: {result.stderr.strip()}",
                recovery_hint="Check Docker Desktop logs.",
            )

        container_id = result.stdout.strip().split("\n")[-1][:12]

        timeout = health_timeout if health_timeout is not None else 120
        if not self._wait_for_health(port, timeout=timeout):
            raise VLLMNotReadyError(
                f"vLLM /health did not respond within {timeout}s",
                recovery_hint="Check `docker logs <id>` for model-loading errors.",
            )

        info = ContainerInfo(container_id=container_id, port=port, model=model)
        self.state.container_running = True
        self.state.current_model = model
        return info

    def stop_container(self) -> None:
        """Stop and remove the cognithor-managed vLLM container. Noop if none."""
        find = subprocess.run(
            ["docker", "ps", "-q", "--filter", "label=cognithor.managed=true"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        container_id = find.stdout.strip().split("\n")[0] if find.stdout.strip() else ""
        if not container_id:
            self.state.container_running = False
            return

        subprocess.run(["docker", "stop", container_id], capture_output=True, timeout=30)
        subprocess.run(["docker", "rm", container_id], capture_output=True, timeout=10)
        self.state.container_running = False
        self.state.current_model = None

    def reuse_existing(self) -> ContainerInfo | None:
        """If a cognithor-managed container is already running, return its info."""
        result = subprocess.run(
            [
                "docker",
                "ps",
                "--filter",
                "label=cognithor.managed=true",
                "--format",
                "json",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None

        first_line = result.stdout.strip().split("\n")[0]
        try:
            row = _json.loads(first_line)
        except _json.JSONDecodeError:
            return None

        container_id = row.get("ID", "").strip()
        ports = row.get("Ports", "")
        cmd = row.get("Command", "")

        port_match = re.search(r"0\.0\.0\.0:(\d+)->8000/tcp", ports)
        port = int(port_match.group(1)) if port_match else self.port

        model_match = re.search(r"--model\s+(\S+)", cmd)
        model = model_match.group(1) if model_match else ""

        if not container_id:
            return None

        info = ContainerInfo(container_id=container_id, port=port, model=model)
        self.state.container_running = True
        self.state.current_model = model
        return info
