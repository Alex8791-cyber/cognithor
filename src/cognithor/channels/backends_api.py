"""FastAPI router for LLM-backend management endpoints.

Exposes GET /api/backends, GET /api/backends/vllm/status and related routes
used by the Flutter "LLM Backends" settings screen. Separated from the
main APIChannel app so it can be included or tested independently.
"""

from __future__ import annotations

import asyncio
import json as _json
from typing import TYPE_CHECKING, Literal

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

if TYPE_CHECKING:
    from cognithor.config import CognithorConfig
    from cognithor.core.vllm_orchestrator import VLLMOrchestrator


backends_router = APIRouter(prefix="/api/backends", tags=["backends"])


class StartRequest(BaseModel):
    model: str


class SetActiveRequest(BaseModel):
    backend: Literal[
        "ollama",
        "openai",
        "anthropic",
        "gemini",
        "groq",
        "deepseek",
        "mistral",
        "together",
        "openrouter",
        "xai",
        "cerebras",
        "github",
        "bedrock",
        "huggingface",
        "moonshot",
        "lmstudio",
        "vllm",
        "llama_cpp",
        "claude-code",
    ]


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


@backends_router.post("/vllm/check-hardware")
async def check_hardware_endpoint(request: Request) -> dict:
    config: CognithorConfig = request.app.state.config
    orch = _get_orchestrator(config)
    try:
        info = orch.check_hardware()
    except Exception as exc:
        from cognithor.core.llm_backend import VLLMHardwareError

        if isinstance(exc, VLLMHardwareError):
            raise HTTPException(
                status_code=503,
                detail={
                    "message": str(exc),
                    "recovery_hint": exc.recovery_hint,
                },
            ) from exc
        raise HTTPException(status_code=500, detail={"message": str(exc)}) from exc
    return {
        "gpu_name": info.gpu_name,
        "vram_gb": info.vram_gb,
        "compute_capability": info.sm_string,
    }


@backends_router.post("/vllm/start")
async def vllm_start(request: Request, body: StartRequest) -> dict:
    config: CognithorConfig = request.app.state.config
    orch = _get_orchestrator(config)
    try:
        info = orch.start_container(body.model)
    except Exception as exc:
        from cognithor.core.llm_backend import VLLMNotReadyError

        if isinstance(exc, VLLMNotReadyError):
            raise HTTPException(
                status_code=503,
                detail={
                    "message": str(exc),
                    "recovery_hint": getattr(exc, "recovery_hint", ""),
                },
            ) from exc
        raise HTTPException(status_code=500, detail={"message": str(exc)}) from exc
    return {
        "container_id": info.container_id,
        "port": info.port,
        "model": info.model,
    }


@backends_router.post("/vllm/stop")
async def vllm_stop(request: Request) -> dict:
    config: CognithorConfig = request.app.state.config
    orch = _get_orchestrator(config)
    orch.stop_container()
    return {"status": "stopped"}


@backends_router.post("/active")
async def set_active_backend(request: Request, body: SetActiveRequest) -> dict:
    """Switch the active LLM backend and re-init UnifiedLLMClient."""
    gateway = request.app.state.gateway
    if gateway is None:
        raise HTTPException(
            status_code=503,
            detail={"message": "Gateway not wired — backend switching not available"},
        )
    gateway.rebuild_llm_client(body.backend)
    return {"active": body.backend}


@backends_router.get("/vllm/logs")
async def vllm_logs(request: Request) -> dict:
    config: CognithorConfig = request.app.state.config
    orch = _get_orchestrator(config)
    return {"lines": orch.get_logs()}


@backends_router.get("/vllm/available-models")
async def vllm_available_models(request: Request) -> dict:
    """Return the curated vLLM model registry filtered against detected hardware.

    Response shape:
        {
          "recommended_id": "<model id or null>",
          "models": [ {<ModelEntry fields>, "fits": bool}, ... ]
        }
    """
    import json as _json
    from pathlib import Path

    from cognithor.core.vllm_orchestrator import ModelEntry

    config: CognithorConfig = request.app.state.config
    orch = _get_orchestrator(config)

    registry_path = Path(__file__).resolve().parents[1] / "cli" / "model_registry.json"
    registry_data = _json.loads(registry_path.read_text(encoding="utf-8"))
    entries = [ModelEntry.from_dict(m) for m in registry_data["providers"]["vllm"]["models"]]

    hw = orch.state.hardware_info
    if hw is None:
        try:
            hw = orch.check_hardware()
        except Exception:
            hw = None

    recommended_id: str | None = None
    fits_ids: set[str] = set()
    if hw is not None:
        best = orch.recommend_model(hw, entries)
        recommended_id = best.id if best else None
        fits_ids = {m.id for m in orch.filter_registry(hw, entries)}

    return {
        "recommended_id": recommended_id,
        "models": [
            {
                "id": e.id,
                "display_name": e.display_name,
                "quantization": e.quantization,
                "vram_gb_min": e.vram_gb_min,
                "min_compute_capability": e.min_compute_capability,
                "priority": e.priority,
                "tested": e.tested,
                "notes": e.notes,
                "fits": e.id in fits_ids,
            }
            for e in entries
        ],
    }


@backends_router.post("/vllm/pull-image")
async def vllm_pull_image(request: Request) -> StreamingResponse:
    """Stream docker-pull progress to the client as SSE."""
    config: CognithorConfig = request.app.state.config
    orch = _get_orchestrator(config)

    queue: asyncio.Queue[dict | None] = asyncio.Queue()

    def progress_cb(event: dict) -> None:
        queue.put_nowait(event)

    async def worker() -> None:
        """Run the blocking pull_image in a thread, enqueue events, sentinel at end."""
        try:
            await asyncio.to_thread(
                orch.pull_image,
                config.vllm.docker_image,
                progress_callback=progress_cb,
            )
        finally:
            queue.put_nowait(None)

    async def event_stream():
        task = asyncio.create_task(worker())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield f"data: {_json.dumps(event)}\n\n"
        finally:
            await task

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def build_backends_app(
    *,
    config: CognithorConfig,
    gateway: object | None = None,
) -> FastAPI:
    """Minimal FastAPI app exposing just the backends router.

    Used by tests. In production the router is included directly in the
    APIChannel's main app.
    """
    app = FastAPI()
    app.state.config = config
    app.state.gateway = gateway
    app.include_router(backends_router)
    return app
