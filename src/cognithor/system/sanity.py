"""Layer 1.5 — Cross-Validation of raw HardwareProfile output.

Each rule is a pure function that consumes a SystemProfile and either
PASS-es (no-op) or returns a SanityWarning describing the inconsistency.

Rules never crash the boot path; they only annotate the profile so L2
(capability mapping) can decide how to interpret unreliable data.
"""

from __future__ import annotations

from dataclasses import dataclass

from cognithor.system.detector import SystemProfile
from cognithor.utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["SANITY_RULES", "SanityWarning", "validate"]


@dataclass(frozen=True)
class SanityWarning:
    rule_id: str
    severity: str  # "info" | "warn" | "error"
    message: str
    affected_field: str | None = None


def _rule_vram_inside_class(p: SystemProfile) -> SanityWarning | None:
    gpu = p.results.get("gpu")
    if gpu is None:
        return None
    vram = gpu.raw_data.get("vram_total_gb", 0)
    if vram > 200:
        return SanityWarning(
            rule_id="vram_implausible_huge",
            severity="warn",
            message=f"VRAM reported as {vram} GB — likely virtualization-lie. Capping at 200 GB.",
            affected_field="gpu.vram_total_gb",
        )
    return None


def _rule_ram_vs_cpu_plausibility(p: SystemProfile) -> SanityWarning | None:
    cpu = p.results.get("cpu")
    ram = p.results.get("ram")
    if cpu is None or ram is None:
        return None
    cores = cpu.raw_data.get("physical_cores", 0)
    ram_gb = ram.raw_data.get("total_gb", 0)
    if cores > 64 and ram_gb < 16:
        return SanityWarning(
            rule_id="cpu_ram_implausible",
            severity="warn",
            message=(
                f"{cores} CPU cores with only {ram_gb} GB RAM — likely virt-lie or container limit."
            ),
            affected_field="cpu.physical_cores",
        )
    return None


def _rule_gpu_present_implies_driver(p: SystemProfile) -> SanityWarning | None:
    gpu = p.results.get("gpu")
    if gpu is None:
        return None
    if gpu.raw_data.get("vendor") == "nvidia" and not gpu.raw_data.get("driver"):
        return SanityWarning(
            rule_id="nvidia_gpu_without_driver",
            severity="error",
            message="NVIDIA GPU detected but no driver version — treating as disabled.",
            affected_field="gpu.driver",
        )
    return None


def _rule_compute_cap_consistent_with_arch(p: SystemProfile) -> SanityWarning | None:
    gpu = p.results.get("gpu")
    if gpu is None:
        return None
    cc = gpu.raw_data.get("compute_capability")
    arch = gpu.raw_data.get("architecture")
    if cc and not arch:
        return SanityWarning(
            rule_id="cc_without_arch_mapping",
            severity="info",
            message=(
                f"compute_capability {cc} did not map to a known architecture — "
                "using forward-compat baseline."
            ),
            affected_field="gpu.architecture",
        )
    return None


def _rule_nvfp4_requires_driver_floor(p: SystemProfile) -> SanityWarning | None:
    gpu = p.results.get("gpu")
    if gpu is None:
        return None
    cc = gpu.raw_data.get("compute_capability", "")
    drv = gpu.raw_data.get("driver", "")
    cuda = gpu.raw_data.get("cuda_version", "")
    if not (cc and cc.startswith("12.")):
        return None
    if drv and _ver_lt(drv, "596.0"):
        return SanityWarning(
            rule_id="nvfp4_driver_too_old",
            severity="warn",
            message=(
                f"NVFP4-capable GPU but driver {drv} < 596 — "
                "NVFP4 will be unavailable until driver upgrade."
            ),
            affected_field="gpu.driver",
        )
    if cuda and _ver_lt(cuda, "13.0"):
        return SanityWarning(
            rule_id="nvfp4_cuda_too_old",
            severity="warn",
            message=(
                f"NVFP4-capable GPU but CUDA {cuda} < 13.0 — "
                "NVFP4 will be unavailable until CUDA upgrade."
            ),
            affected_field="gpu.cuda_version",
        )
    return None


def _rule_docker_required_for_vllm_container(p: SystemProfile) -> SanityWarning | None:
    docker = p.results.get("docker")
    vllm = p.results.get("vllm")
    if docker is None or vllm is None:
        return None
    if not docker.raw_data.get("running") and not vllm.raw_data.get("pip_installed"):
        return SanityWarning(
            rule_id="vllm_path_unavailable",
            severity="info",
            message=(
                "Neither Docker nor pip-vllm available — "
                "vLLM-Tier will be filtered out by L4 solver."
            ),
            affected_field=None,
        )
    return None


def _rule_container_without_gpu_passthrough(p: SystemProfile) -> SanityWarning | None:
    container = p.results.get("container")
    gpu = p.results.get("gpu")
    if container is None:
        return None
    if container.raw_data.get("in_container") and gpu is not None:
        if gpu.raw_data.get("vendor") == "none":
            return SanityWarning(
                rule_id="container_no_gpu_passthrough",
                severity="info",
                message=(
                    "Running inside a container without GPU passthrough — "
                    "local Inference unavailable."
                ),
                affected_field=None,
            )
    return None


SANITY_RULES = (
    _rule_vram_inside_class,
    _rule_ram_vs_cpu_plausibility,
    _rule_gpu_present_implies_driver,
    _rule_compute_cap_consistent_with_arch,
    _rule_nvfp4_requires_driver_floor,
    _rule_docker_required_for_vllm_container,
    _rule_container_without_gpu_passthrough,
)


def validate(profile: SystemProfile) -> tuple[SanityWarning, ...]:
    """Run all sanity rules. Never raises."""
    warnings: list[SanityWarning] = []
    for rule in SANITY_RULES:
        try:
            w = rule(profile)
        except Exception as exc:
            log.debug("sanity_rule_error", rule=rule.__name__, error=str(exc))
            continue
        if w is not None:
            warnings.append(w)
            log.info(
                "sanity_warning",
                rule_id=w.rule_id,
                severity=w.severity,
                message=w.message,
            )
    return tuple(warnings)


def _ver_lt(a: str, b: str) -> bool:
    """True iff version `a` is strictly less than `b`. Tolerates extra suffix."""
    try:
        a_parts = [int(x) for x in a.split(".")[:3] if x.isdigit()]
        b_parts = [int(x) for x in b.split(".")[:3] if x.isdigit()]
        return a_parts < b_parts
    except (ValueError, AttributeError):
        return False
