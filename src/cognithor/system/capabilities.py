"""Layer 2 — Capability-Mapping.

Pure function: HardwareProfile → Capabilities.

Capabilities are STABLE flags that L3+L4 use as constraint primitives.
Adding a new GPU architecture or new quantization variant means adding a
new flag here (rare); adding a new model means a YAML-PR (frequent,
zero-code).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from cognithor.system.detector import SystemProfile

__all__ = [
    "DISK_CLASSES",
    "RAM_CLASSES",
    "VRAM_CLASSES",
    "Capabilities",
    "map_to_capabilities",
]

VRAM_CLASSES = ("none", "tiny", "small", "medium", "large", "xlarge", "xxlarge")
RAM_CLASSES = ("low", "medium", "high", "extreme")
DISK_CLASSES = ("very_low", "low", "medium", "high")


@dataclass(frozen=True)
class Capabilities:
    """Stable abstract capability flags consumed by L3 (manifest match) + L4 (solver)."""

    schema_version: int = 2

    # ── Quantization runtime support (CUDA-side) ─────────────────────
    can_run_nvfp4: bool = False  # sm120+ + drv≥596 + cuda≥13
    can_run_fp8_marlin: bool = False  # sm89+ + cuda≥12
    can_run_fp8_native: bool = False  # sm89+ + cuda≥12.4
    can_run_gptq_int4: bool = False  # sm70+
    can_run_awq_int4: bool = False  # sm70+
    can_run_bnb_int8: bool = False  # sm70+

    # ── GGUF runtimes (Ollama / llama.cpp) ────────────────────────────
    can_run_gguf_cuda: bool = False  # NVIDIA + ≥4 GB VRAM
    can_run_gguf_metal: bool = False  # Apple Silicon
    can_run_gguf_rocm: bool = False  # AMD ROCm
    can_run_gguf_cpu: bool = True  # always

    # ── Backend availability ──────────────────────────────────────────
    can_run_vllm_container: bool = False  # docker-running + nvidia-runtime + GPU
    can_run_vllm_inprocess: bool = False  # pip vllm + GPU-cap
    can_run_ollama_native: bool = False  # ollama daemon up OR pip ollama-py + binary
    can_run_lmstudio: bool = False
    can_run_llama_cpp: bool = True  # always (CPU baseline)

    # ── Memory size classes ───────────────────────────────────────────
    vram_class: Literal["none", "tiny", "small", "medium", "large", "xlarge", "xxlarge"] = "none"
    aggregate_vram_class: Literal[
        "none", "tiny", "small", "medium", "large", "xlarge", "xxlarge"
    ] = "none"
    ram_class: Literal["low", "medium", "high", "extreme"] = "low"
    disk_class: Literal["very_low", "low", "medium", "high"] = "very_low"

    # ── Multi-GPU ─────────────────────────────────────────────────────
    has_multi_gpu_homogeneous: bool = False
    has_multi_gpu_heterogeneous: bool = False
    multi_gpu_count: int = 1

    # ── Network ───────────────────────────────────────────────────────
    has_internet: bool = False
    has_huggingface_access: bool = False
    is_offline_only: bool = True
    is_metered_connection: bool = False

    # ── Container / Sandbox ───────────────────────────────────────────
    is_in_container: bool = False
    can_reach_host_gpu: bool = True

    # ── Origin profile-hash for traceability ──────────────────────────
    profile_hash: str = ""

    # ── Compatibility check ──────────────────────────────────────────

    def satisfies(self, requirement: str) -> bool:
        """Check if a single tier-requirement string is met.

        Supports:
        - "can_run_nvfp4" (bool flag)
        - "vram_class>=large" (ordinal compare)
        - "ram_class>=high"
        - "disk_class>=medium"
        """
        if ">=" in requirement:
            field, value = (s.strip() for s in requirement.split(">=", 1))
            return _ordinal_satisfies(self, field, value, op=">=")
        if ">" in requirement:
            field, value = (s.strip() for s in requirement.split(">", 1))
            return _ordinal_satisfies(self, field, value, op=">")
        # Bool flag
        return bool(getattr(self, requirement, False))


def _ordinal_satisfies(caps: Capabilities, field: str, value: str, *, op: str) -> bool:
    have = getattr(caps, field, None)
    if have is None:
        return False
    classes = (
        VRAM_CLASSES
        if "vram" in field
        else RAM_CLASSES
        if field == "ram_class"
        else DISK_CLASSES
        if field == "disk_class"
        else None
    )
    if classes is None:
        return False
    try:
        idx_have = classes.index(have)
        idx_need = classes.index(value)
    except ValueError:
        return False
    if op == ">=":
        return idx_have >= idx_need
    return idx_have > idx_need


# ── VRAM / RAM / Disk classifiers ────────────────────────────────────────────


def _vram_to_class(gb: float) -> str:
    if gb < 1:
        return "none"
    if gb < 4:
        return "tiny"
    if gb < 8:
        return "small"
    if gb < 16:
        return "medium"
    if gb < 24:
        return "large"
    if gb < 48:
        return "xlarge"
    return "xxlarge"


def _ram_to_class(gb: float) -> str:
    if gb < 16:
        return "low"
    if gb < 32:
        return "medium"
    if gb < 64:
        return "high"
    return "extreme"


def _disk_to_class(gb: float) -> str:
    if gb < 30:
        return "very_low"
    if gb < 80:
        return "low"
    if gb < 200:
        return "medium"
    return "high"


# ── Compute-capability compares ──────────────────────────────────────────────


def _sm_at_least(cc: str, target: str) -> bool:
    try:
        ca, cb = (float(c.split(".")[0]) + float(c.split(".")[1]) / 10 for c in (cc, target))
        return ca >= cb
    except (ValueError, IndexError):
        return False


def _ver_at_least(v: str, target: str) -> bool:
    """Compare dotted versions (a.b.c…). Tolerates extra suffixes."""
    try:
        a = [int(p) for p in v.split(".")[:3] if p.isdigit()]
        b = [int(p) for p in target.split(".")[:3] if p.isdigit()]
        return a >= b
    except (ValueError, AttributeError):
        return False


# ── Main mapping ─────────────────────────────────────────────────────────────


def map_to_capabilities(profile: SystemProfile) -> Capabilities:
    """Pure function. Same input → same output. Deterministic."""
    gpu = profile.results.get("gpu")
    ram = profile.results.get("ram")
    disk = profile.results.get("disk")
    network = profile.results.get("network")
    docker = profile.results.get("docker")
    container = profile.results.get("container")
    ollama = profile.results.get("ollama")
    lmstudio = profile.results.get("lmstudio")
    vllm = profile.results.get("vllm")
    rocm = profile.results.get("rocm")
    huggingface = profile.results.get("huggingface")

    gpu_data = gpu.raw_data if gpu else {}
    vendor = gpu_data.get("vendor", "none")
    cc = gpu_data.get("compute_capability") or ""
    drv = gpu_data.get("driver") or ""
    cuda = gpu_data.get("cuda_version") or ""
    vram_gb = gpu_data.get("vram_total_gb", 0) or 0

    # Quantization-runtime flags (NVIDIA-side)
    can_nvfp4 = (
        vendor == "nvidia"
        and bool(cc)
        and _sm_at_least(cc, "12.0")
        and bool(drv)
        and _ver_at_least(drv, "596.0")
        and bool(cuda)
        and _ver_at_least(cuda, "13.0")
    )
    can_fp8_marlin = (
        vendor == "nvidia"
        and bool(cc)
        and _sm_at_least(cc, "8.9")
        and bool(cuda)
        and _ver_at_least(cuda, "12.0")
    )
    can_fp8_native = (
        vendor == "nvidia"
        and bool(cc)
        and _sm_at_least(cc, "8.9")
        and bool(cuda)
        and _ver_at_least(cuda, "12.4")
    )
    can_gptq = vendor == "nvidia" and bool(cc) and _sm_at_least(cc, "7.0")
    can_awq = can_gptq
    can_bnb = can_gptq

    # GGUF runtimes
    can_gguf_cuda = vendor == "nvidia" and vram_gb >= 4
    can_gguf_metal = vendor == "apple"
    can_gguf_rocm = bool(rocm and rocm.raw_data.get("available"))

    # Backends
    docker_running = bool(docker and docker.raw_data.get("running"))
    can_vllm_container = (
        docker_running
        and (can_nvfp4 or can_fp8_marlin or can_gguf_cuda)
        and not bool(container and container.raw_data.get("in_container") and vendor == "none")
    )
    can_vllm_inprocess = bool(vllm and vllm.raw_data.get("pip_installed")) and (
        can_nvfp4 or can_fp8_marlin or can_gguf_cuda
    )
    can_ollama_native = bool(
        (ollama and ollama.raw_data.get("running"))
        or vendor in ("nvidia", "apple")
        or can_gguf_cuda
        or can_gguf_metal
        or can_gguf_rocm
    )
    can_lmstudio = bool(lmstudio and lmstudio.raw_data.get("running"))

    # Memory classes
    vram_class = _vram_to_class(vram_gb)
    all_gpus = gpu_data.get("all_gpus") or []
    aggregate_vram = sum(g.get("vram_total_gb", 0) for g in all_gpus) if all_gpus else vram_gb
    aggregate_vram_class = _vram_to_class(aggregate_vram)

    ram_gb = ram.raw_data.get("total_gb", 0) if ram else 0
    ram_class = _ram_to_class(ram_gb)
    disk_gb = disk.raw_data.get("free_gb", 0) if disk else 0
    disk_class = _disk_to_class(disk_gb)

    # Multi-GPU
    multi_count = gpu_data.get("multi_gpu_count", 1) or 1
    homogeneous = (
        (multi_count > 1 and all(g.get("model") == all_gpus[0].get("model") for g in all_gpus))
        if all_gpus
        else False
    )
    heterogeneous = multi_count > 1 and not homogeneous

    # Network
    has_internet = bool(network and network.raw_data.get("internet"))
    has_hf = bool(huggingface and huggingface.raw_data.get("reachable"))

    # Container
    in_container = bool(container and container.raw_data.get("in_container"))

    return Capabilities(
        can_run_nvfp4=can_nvfp4,
        can_run_fp8_marlin=can_fp8_marlin,
        can_run_fp8_native=can_fp8_native,
        can_run_gptq_int4=can_gptq,
        can_run_awq_int4=can_awq,
        can_run_bnb_int8=can_bnb,
        can_run_gguf_cuda=can_gguf_cuda,
        can_run_gguf_metal=can_gguf_metal,
        can_run_gguf_rocm=can_gguf_rocm,
        can_run_gguf_cpu=True,
        can_run_vllm_container=can_vllm_container,
        can_run_vllm_inprocess=can_vllm_inprocess,
        can_run_ollama_native=can_ollama_native,
        can_run_lmstudio=can_lmstudio,
        can_run_llama_cpp=True,
        vram_class=vram_class,  # type: ignore[arg-type]
        aggregate_vram_class=aggregate_vram_class,  # type: ignore[arg-type]
        ram_class=ram_class,  # type: ignore[arg-type]
        disk_class=disk_class,  # type: ignore[arg-type]
        has_multi_gpu_homogeneous=homogeneous,
        has_multi_gpu_heterogeneous=heterogeneous,
        multi_gpu_count=multi_count,
        has_internet=has_internet,
        has_huggingface_access=has_hf,
        is_offline_only=not has_internet,
        is_metered_connection=False,  # OS-API integration TBD
        is_in_container=in_container,
        can_reach_host_gpu=not (in_container and vendor == "none"),
        profile_hash=_hash_profile(profile),
    )


def _hash_profile(profile: SystemProfile) -> str:
    """SHA-256 over capability-relevant fields. Excludes timestamps."""
    import hashlib
    import json

    relevant: dict[str, dict[str, Any]] = {}
    for key in ("os", "cpu", "ram", "gpu", "disk", "docker", "wsl2", "rocm", "container"):
        if key in profile.results:
            data = dict(profile.results[key].raw_data)
            # Strip volatile sub-fields
            data.pop("vram_free_gb", None)
            data.pop("free_gb", None)
            data.pop("free_mb", None)
            data.pop("available_mb", None)
            relevant[key] = data
    payload = json.dumps(relevant, sort_keys=True, default=str).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"
