"""FastAPI router for LLM-backend management endpoints.

Exposes GET /api/backends, GET /api/backends/vllm/status and related routes
used by the Flutter "LLM Backends" settings screen. Separated from the
main APIChannel app so it can be included or tested independently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, FastAPI, Request

if TYPE_CHECKING:
    from cognithor.config import CognithorConfig
    from cognithor.core.vllm_orchestrator import VLLMOrchestrator


backends_router = APIRouter(prefix="/api/backends", tags=["backends"])

# Module-level orchestrator singleton. Reset across app builds by wiring
# through app.state.config → build_backends_app().
_orchestrator_cache: dict[int, VLLMOrchestrator] = {}


def _get_orchestrator(config: CognithorConfig) -> VLLMOrchestrator:
    """Lazy-initialized singleton keyed by config id. Same config → same orchestrator."""
    from cognithor.core.vllm_orchestrator import VLLMOrchestrator

    key = id(config)
    if key not in _orchestrator_cache:
        _orchestrator_cache[key] = VLLMOrchestrator(
            docker_image=config.vllm.docker_image,
            port=config.vllm.port,
            hf_token=config.huggingface_api_key,
        )
    return _orchestrator_cache[key]


@backends_router.get("")
async def list_backends(request: Request) -> dict:
    """Return every configured backend with its current readiness."""
    config: CognithorConfig = request.app.state.config
    backends = [
        {
            "name": "ollama",
            "enabled": config.llm_backend_type == "ollama",
            "status": "ready",
        }
    ]
    orch = _get_orchestrator(config)
    st = orch.status()
    if st.container_running:
        vllm_status = "ready"
    elif config.vllm.enabled:
        vllm_status = "configured"
    else:
        vllm_status = "disabled"
    backends.append(
        {
            "name": "vllm",
            "enabled": config.vllm.enabled,
            "status": vllm_status,
        }
    )
    return {"active": config.llm_backend_type, "backends": backends}


@backends_router.get("/vllm/status")
async def vllm_status(request: Request) -> dict:
    """Return the current VLLMState as JSON for the Flutter setup page."""
    config: CognithorConfig = request.app.state.config
    orch = _get_orchestrator(config)
    st = orch.status()
    hw = None
    if st.hardware_info:
        hw = {
            "gpu_name": st.hardware_info.gpu_name,
            "vram_gb": st.hardware_info.vram_gb,
            "compute_capability": st.hardware_info.sm_string,
        }
    return {
        "hardware_ok": st.hardware_ok,
        "hardware_info": hw,
        "docker_ok": st.docker_ok,
        "image_pulled": st.image_pulled,
        "container_running": st.container_running,
        "current_model": st.current_model,
        "last_error": st.last_error,
    }


def build_backends_app(*, config: CognithorConfig) -> FastAPI:
    """Minimal FastAPI app exposing just the backends router.

    Used by tests. In production the router is included directly in the
    APIChannel's main app.
    """
    app = FastAPI()
    app.state.config = config
    app.include_router(backends_router)
    return app
