"""vLLM lifecycle orchestrator — wraps docker/nvidia-smi subprocesses.

Stateful manager: hardware detection, Docker readiness, image pull,
container start/stop, model recommendation. No Docker-SDK dependency —
pure `subprocess` calls.

See spec: docs/superpowers/specs/2026-04-22-vllm-opt-in-backend-design.md
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Any, Literal

from cognithor.utils.logging import get_logger

log = get_logger(__name__)

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
