"""REST API for the Hardware-Aware Runtime stack.

Endpoints (mounted under /api/system):
- GET  /profile             Full hardware profile
- GET  /capabilities        Derived capability flags
- GET  /recommendations     Top Pareto solutions for given objective
- POST /apply               Apply a chosen tier_id (with explicit confirmation)
- POST /refresh-manifest    Force online manifest refresh
- GET  /health              Drift status + components health
- POST /rollback            Restore most recent config backup
- GET  /backups             List available backups

The router is included into the main FastAPI app via
`app.include_router(system_router)` in `channels/api.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from cognithor.system.apply_engine import (
    ApplyError,
    apply_solution,
    list_backups,
    rollback_last,
)
from cognithor.system.capabilities import map_to_capabilities
from cognithor.system.detector import SystemDetector
from cognithor.system.drift_detector import DriftDetector
from cognithor.system.manifest_loader import ManifestLoader, ManifestRecalledError
from cognithor.system.perf_tracker import get_default_tracker
from cognithor.system.sanity import validate
from cognithor.system.solver import OBJECTIVE_PRESETS, solve
from cognithor.utils.logging import get_logger

log = get_logger(__name__)

system_router = APIRouter(prefix="/api/system", tags=["system"])


# ── Module-level cache (rebuilt per request — fast enough) ─────────────────


def _detect() -> tuple[Any, Any, tuple[Any, ...]]:
    detector = SystemDetector()
    profile = detector.run_full_scan()
    caps = map_to_capabilities(profile)
    sanity_warns = validate(profile)
    return profile, caps, sanity_warns


# ── Pydantic-Schemas ───────────────────────────────────────────────────────


class ProfileResponse(BaseModel):
    detected_at: str
    tier: str
    recommended_mode: str
    sanity_warnings: list[dict[str, Any]] = Field(default_factory=list)
    components: dict[str, dict[str, Any]] = Field(default_factory=dict)


class CapabilitiesResponse(BaseModel):
    can_run_nvfp4: bool
    can_run_fp8_marlin: bool
    can_run_fp8_native: bool
    can_run_gguf_cuda: bool
    can_run_gguf_metal: bool
    can_run_gguf_rocm: bool
    can_run_gguf_cpu: bool
    can_run_vllm_container: bool
    can_run_vllm_inprocess: bool
    can_run_ollama_native: bool
    can_run_lmstudio: bool
    vram_class: str
    aggregate_vram_class: str
    ram_class: str
    disk_class: str
    has_multi_gpu_homogeneous: bool
    multi_gpu_count: int
    has_internet: bool
    has_huggingface_access: bool
    is_offline_only: bool
    is_in_container: bool
    profile_hash: str


class SolutionResponse(BaseModel):
    tier_id: str
    display_name: str
    rationale_de: str
    rationale_en: str
    score: float
    score_breakdown: dict[str, float]
    blockers: list[str]
    warnings: list[str]
    is_immediately_runnable: bool
    estimated_first_response_s: float
    estimated_disk_gb: float
    estimated_setup_minutes: int
    estimated_cost_eur_per_month: float
    backend: str
    model_set: dict[str, str]
    rule_id: str


class RecommendationsResponse(BaseModel):
    manifest_version: str
    manifest_origin: str
    manifest_signature_verified: bool
    objective_preset: str
    capabilities_hash: str
    solutions: list[SolutionResponse]


class ApplyRequest(BaseModel):
    tier_id: str
    objective_preset: Literal["balanced", "quality", "speed", "privacy", "cost"] = "balanced"
    user_confirmed: bool = False


class ApplyResponse(BaseModel):
    success: bool
    selected_tier_id: str
    config_path: str
    backup_path: str | None
    sidecar_path: str
    initialized_marker_path: str
    capabilities_hash: str


class HealthResponse(BaseModel):
    initialized: bool
    current_tier: str | None
    manifest_version: str | None
    drift_detected: bool
    drift_components: list[str]
    sanity_warnings: list[dict[str, Any]]
    last_capabilities_hash: str | None
    current_capabilities_hash: str


# ── Endpoints ───────────────────────────────────────────────────────────────


@system_router.get("/profile", response_model=ProfileResponse)
async def get_profile() -> ProfileResponse:
    profile, _caps, sanity_warns = _detect()
    return ProfileResponse(
        detected_at=profile.detected_at,
        tier=profile.get_tier(),
        recommended_mode=profile.get_recommended_mode(),
        sanity_warnings=[
            {"rule_id": w.rule_id, "severity": w.severity, "message": w.message}
            for w in sanity_warns
        ],
        components={k: r.to_dict() for k, r in profile.results.items()},
    )


@system_router.get("/capabilities", response_model=CapabilitiesResponse)
async def get_capabilities() -> CapabilitiesResponse:
    _profile, caps, _sanity = _detect()
    return CapabilitiesResponse(
        can_run_nvfp4=caps.can_run_nvfp4,
        can_run_fp8_marlin=caps.can_run_fp8_marlin,
        can_run_fp8_native=caps.can_run_fp8_native,
        can_run_gguf_cuda=caps.can_run_gguf_cuda,
        can_run_gguf_metal=caps.can_run_gguf_metal,
        can_run_gguf_rocm=caps.can_run_gguf_rocm,
        can_run_gguf_cpu=caps.can_run_gguf_cpu,
        can_run_vllm_container=caps.can_run_vllm_container,
        can_run_vllm_inprocess=caps.can_run_vllm_inprocess,
        can_run_ollama_native=caps.can_run_ollama_native,
        can_run_lmstudio=caps.can_run_lmstudio,
        vram_class=caps.vram_class,
        aggregate_vram_class=caps.aggregate_vram_class,
        ram_class=caps.ram_class,
        disk_class=caps.disk_class,
        has_multi_gpu_homogeneous=caps.has_multi_gpu_homogeneous,
        multi_gpu_count=caps.multi_gpu_count,
        has_internet=caps.has_internet,
        has_huggingface_access=caps.has_huggingface_access,
        is_offline_only=caps.is_offline_only,
        is_in_container=caps.is_in_container,
        profile_hash=caps.profile_hash,
    )


@system_router.get("/recommendations", response_model=RecommendationsResponse)
async def get_recommendations(
    objective: Literal["balanced", "quality", "speed", "privacy", "cost"] = "balanced",
    max_solutions: int = 5,
) -> RecommendationsResponse:
    _profile, caps, _sanity = _detect()
    loader = ManifestLoader()
    try:
        manifest, source = loader.load(prefer_online=False)
    except ManifestRecalledError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    pricing = loader.load_pricing()
    sols = solve(
        manifest,
        caps,
        OBJECTIVE_PRESETS[objective],
        pricing=pricing,
        max_solutions=max_solutions,
    )

    response_sols: list[SolutionResponse] = []
    for s in sols:
        tier = next((t for t in manifest.tiers if t.id == s.tier_id), None)
        if tier is None:
            continue
        response_sols.append(
            SolutionResponse(
                tier_id=s.tier_id,
                display_name=tier.display_name,
                rationale_de=tier.rationale_de,
                rationale_en=tier.rationale_en,
                score=s.score,
                score_breakdown=dict(s.score_breakdown),
                blockers=list(s.blockers),
                warnings=list(s.warnings),
                is_immediately_runnable=s.is_immediately_runnable,
                estimated_first_response_s=s.estimated_first_response_s,
                estimated_disk_gb=s.estimated_disk_gb,
                estimated_setup_minutes=s.estimated_setup_minutes,
                estimated_cost_eur_per_month=s.estimated_cost_eur_per_month,
                backend=tier.backend,
                model_set={
                    "planner": tier.model_set.planner,
                    "executor": tier.model_set.executor,
                    "coder": tier.model_set.coder,
                    "embedding": tier.model_set.embedding,
                    "formulate": tier.model_set.formulate,
                    "fast_path_validator": tier.model_set.fast_path_validator,
                },
                rule_id=s.rule_id,
            )
        )

    return RecommendationsResponse(
        manifest_version=source.manifest_version,
        manifest_origin=source.origin,
        manifest_signature_verified=source.signature_verified,
        objective_preset=objective,
        capabilities_hash=caps.profile_hash,
        solutions=response_sols,
    )


@system_router.post("/apply", response_model=ApplyResponse)
async def post_apply(req: ApplyRequest) -> ApplyResponse:
    if not req.user_confirmed:
        raise HTTPException(status_code=400, detail="user_confirmed must be true to apply")
    _profile, caps, _sanity = _detect()
    loader = ManifestLoader()
    try:
        manifest, _source = loader.load(prefer_online=False)
    except ManifestRecalledError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    pricing = loader.load_pricing()
    sols = solve(
        manifest,
        caps,
        OBJECTIVE_PRESETS[req.objective_preset],
        pricing=pricing,
        max_solutions=10,
    )
    chosen = next((s for s in sols if s.tier_id == req.tier_id), None)
    if chosen is None:
        raise HTTPException(
            status_code=404,
            detail=f"tier_id '{req.tier_id}' not in current solutions",
        )
    if chosen.blockers:
        raise HTTPException(
            status_code=400,
            detail=f"tier '{req.tier_id}' has blockers: {list(chosen.blockers)}",
        )
    try:
        result = apply_solution(
            solution=chosen,
            manifest=manifest,
            capabilities=caps,
            objective=OBJECTIVE_PRESETS[req.objective_preset],
            user_confirmed=True,
        )
    except ApplyError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ApplyResponse(
        success=result.success,
        selected_tier_id=result.selected_tier_id,
        config_path=str(result.config_path),
        backup_path=str(result.backup_path) if result.backup_path else None,
        sidecar_path=str(result.config_path.parent / ".hardware_aware.json"),
        initialized_marker_path=str(result.initialized_marker_path),
        capabilities_hash=result.capabilities_hash,
    )


@system_router.post("/refresh-manifest")
async def post_refresh_manifest() -> dict[str, Any]:
    loader = ManifestLoader()
    try:
        manifest, source = loader.load(prefer_online=True, force_refresh=True)
    except ManifestRecalledError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "manifest_version": source.manifest_version,
        "origin": source.origin,
        "signature_verified": source.signature_verified,
        "tiers": len(manifest.tiers),
        "models": len(manifest.models),
    }


@system_router.get("/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    import json

    home = Path.home() / ".cognithor"
    marker = home / ".cognithor_initialized"
    sidecar = home / ".hardware_aware.json"

    initialized = marker.exists()
    current_tier = None
    manifest_version = None
    last_hash = None
    if initialized:
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
            current_tier = data.get("tier_id")
            manifest_version = data.get("manifest_version")
            last_hash = data.get("capabilities_hash")
        except (OSError, json.JSONDecodeError):
            pass
    if last_hash is None and sidecar.exists():
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            last_hash = data.get("system_profile_hash")
        except (OSError, json.JSONDecodeError):
            pass

    profile, caps, sanity_warns = _detect()

    # Use the stateful drift-detector (hysteresis + cooldowns)
    detector = DriftDetector()
    hw_drift = detector.check_hardware_drift(caps, last_hash)

    perf_drift = None
    if current_tier and manifest_version:
        try:
            loader = ManifestLoader()
            manifest_for_perf, _ = loader.load(prefer_online=False)
            perf_drift = detector.check_performance_drift(manifest_for_perf, current_tier)
        except Exception:
            perf_drift = None

    drift_detected = bool(
        (hw_drift.detected and not hw_drift.cooldown_active)
        or (perf_drift and perf_drift.detected and not perf_drift.cooldown_active)
    )
    drift_components: list[str] = list(hw_drift.components) if hw_drift.detected else []
    if perf_drift and perf_drift.detected:
        drift_components.extend(perf_drift.components)

    return HealthResponse(
        initialized=initialized,
        current_tier=current_tier,
        manifest_version=manifest_version,
        drift_detected=drift_detected,
        drift_components=drift_components,
        sanity_warnings=[
            {"rule_id": w.rule_id, "severity": w.severity, "message": w.message}
            for w in sanity_warns
        ],
        last_capabilities_hash=last_hash,
        current_capabilities_hash=caps.profile_hash,
    )


@system_router.get("/perf")
async def get_perf_summary() -> dict[str, Any]:
    """Per-model rolling performance summary (last 24 h)."""
    tracker = get_default_tracker()
    return {"window_s": 86400, "models": tracker.model_summary(window_s=86400)}


@system_router.post("/dismiss-hardware-drift")
async def post_dismiss_hardware_drift() -> dict[str, Any]:
    DriftDetector().dismiss_hardware_banner()
    return {"ok": True}


@system_router.post("/dismiss-performance-drift")
async def post_dismiss_performance_drift() -> dict[str, Any]:
    DriftDetector().dismiss_performance_banner()
    return {"ok": True}


@system_router.post("/rollback")
async def post_rollback() -> dict[str, Any]:
    restored = rollback_last()
    if restored is None:
        raise HTTPException(status_code=404, detail="No backups available")
    return {"restored_from": str(restored)}


@system_router.get("/backups")
async def get_backups() -> dict[str, Any]:
    return {"backups": [str(p) for p in list_backups()]}
