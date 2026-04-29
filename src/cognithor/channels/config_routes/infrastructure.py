"""Cognithor · Infrastructure + Portal + Backend routes.

Sub-Modul des `config_routes`-Pakets (siehe
`docs/superpowers/plans/2026-04-29-config-routes-split.md`). Bundle aus
drei Querschnitts-Helfern:

  - `_register_infrastructure_routes()` — Ecosystem-Health,
    Performance-Manager.
  - `_register_portal_routes()` — End-User-Portal.
  - `_register_backend_routes()` — LLM-Backend-Status / Switch
    (Ollama / vLLM / etc.).
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

try:
    from starlette.requests import Request
except ImportError:
    Request = Any  # type: ignore[assignment,misc]

try:
    from fastapi import HTTPException
except ImportError:
    try:
        from starlette.exceptions import HTTPException  # type: ignore[assignment]
    except ImportError:
        HTTPException = Exception  # type: ignore[assignment,misc]

from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.config_manager import ConfigManager

log = get_logger(__name__)


# ======================================================================
# Infrastructure routes (ecosystem, performance, portal)
# ======================================================================


def _register_infrastructure_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """Ecosystem control, performance manager."""

    # -- Ecosystem-Kontrolle (Phase 28) -----------------------------------

    @app.get("/api/v1/ecosystem/stats", dependencies=deps)
    async def ecosystem_stats() -> dict[str, Any]:
        """Ecosystem-Controller Statistiken."""
        ctrl = getattr(gateway, "_ecosystem_controller", None)
        if ctrl is None:
            return {"curator": {}, "fraud": {}}
        return ctrl.stats()

    @app.get("/api/v1/ecosystem/curator", dependencies=deps)
    async def ecosystem_curator() -> dict[str, Any]:
        """Kuration-Statistiken."""
        ctrl = getattr(gateway, "_ecosystem_controller", None)
        if ctrl is None:
            return {"total_reviews": 0}
        return ctrl.curator.stats()

    @app.get("/api/v1/ecosystem/fraud", dependencies=deps)
    async def ecosystem_fraud() -> dict[str, Any]:
        """Fraud-Detection Statistiken."""
        ctrl = getattr(gateway, "_ecosystem_controller", None)
        if ctrl is None:
            return {"total_signals": 0}
        return ctrl.fraud.stats()

    @app.get("/api/v1/ecosystem/training", dependencies=deps)
    async def ecosystem_training() -> dict[str, Any]:
        """Security-Training Status."""
        ctrl = getattr(gateway, "_ecosystem_controller", None)
        if ctrl is None:
            return {"total_modules": 0}
        return ctrl.trainer.stats()

    @app.get("/api/v1/ecosystem/trust", dependencies=deps)
    async def ecosystem_trust() -> dict[str, Any]:
        """Trust-Boundary Statistiken."""
        ctrl = getattr(gateway, "_ecosystem_controller", None)
        if ctrl is None:
            return {"total_boundaries": 0}
        return ctrl.trust.stats()

    # -- Performance-Manager (Phase 37) -----------------------------------

    @app.get("/api/v1/performance/health", dependencies=deps)
    async def perf_health() -> dict[str, Any]:
        """Performance Health-Status."""
        pm = getattr(gateway, "_perf_manager", None)
        if pm is None:
            return {"vector_store": {"entries": 0}}
        return pm.health()

    @app.get("/api/v1/performance/latency", dependencies=deps)
    async def perf_latency() -> dict[str, Any]:
        """Latenz-Statistiken."""
        pm = getattr(gateway, "_perf_manager", None)
        if pm is None:
            return {"total_samples": 0}
        return pm.latency.stats()

    @app.get("/api/v1/performance/resources", dependencies=deps)
    async def perf_resources() -> dict[str, Any]:
        """Ressourcen-Auslastung."""
        pm = getattr(gateway, "_perf_manager", None)
        if pm is None:
            return {"snapshots": 0}
        return pm.optimizer.stats()


# ======================================================================
# Portal routes (end-user portal)
# ======================================================================


def _register_portal_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """End-user portal, consent management."""

    @app.get("/api/v1/portal/stats", dependencies=deps)
    async def portal_stats() -> dict[str, Any]:
        """Endnutzer-Portal Statistiken."""
        up = getattr(gateway, "_user_portal", None)
        if up is None:
            return {"consents": {"total_users": 0}}
        return up.stats()

    @app.get("/api/v1/portal/consents", dependencies=deps)
    async def portal_consents() -> dict[str, Any]:
        """Consent-Management Status."""
        up = getattr(gateway, "_user_portal", None)
        if up is None:
            return {"total_users": 0}
        return up.consents.stats()


# ======================================================================
# Backend status / switch routes
# ======================================================================


def _register_backend_routes(
    app: Any,
    deps: list[Any],
    config_manager: ConfigManager,
    gateway: Any,
) -> None:
    """Endpoints for querying LLM backend availability and switching backends."""

    import shutil

    @app.get("/api/v1/backend/status", dependencies=deps)
    async def get_backend_status() -> dict[str, Any]:
        """Check which LLM backends are available and authenticated."""
        results: dict[str, Any] = {}

        # Claude Code CLI
        claude_path = shutil.which("claude")
        results["claude-code"] = {
            "installed": claude_path is not None,
            "path": claude_path or "",
            "authenticated": False,
            "models": [],
        }
        if claude_path:
            try:
                proc = await asyncio.create_subprocess_exec(
                    claude_path,
                    "--version",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
                if proc.returncode == 0:
                    results["claude-code"]["authenticated"] = True
                    results["claude-code"]["version"] = stdout.decode().strip()
                    results["claude-code"]["models"] = [
                        "opus",
                        "sonnet",
                        "haiku",
                    ]
            except Exception:
                log.debug("claude_code_provider_check_failed", exc_info=True)

        # Ollama
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                resp = await client.get("http://localhost:11434/api/tags", timeout=3)
                if resp.status_code == 200:
                    models = [m["name"] for m in resp.json().get("models", [])]
                    results["ollama"] = {
                        "installed": True,
                        "authenticated": True,
                        "models": models,
                    }
                else:
                    results["ollama"] = {
                        "installed": True,
                        "authenticated": False,
                        "models": [],
                    }
        except Exception:
            results["ollama"] = {
                "installed": False,
                "authenticated": False,
                "models": [],
            }

        # OpenAI (check if key is set)
        cfg = config_manager.config if config_manager else None
        if cfg:
            has_openai = bool(getattr(cfg, "openai_api_key", ""))
            results["openai"] = {
                "installed": True,
                "authenticated": has_openai,
                "models": [],
            }

            has_anthropic = bool(getattr(cfg, "anthropic_api_key", ""))
            results["anthropic"] = {
                "installed": True,
                "authenticated": has_anthropic,
                "models": [],
            }

            has_openrouter = bool(getattr(cfg, "openrouter_api_key", ""))
            results["openrouter"] = {
                "installed": True,
                "authenticated": has_openrouter,
                "models": [],
            }

        # Current backend
        current = getattr(cfg, "llm_backend_type", "ollama") if cfg else "ollama"

        return {"backends": results, "current": current}

    @app.post("/api/v1/backend/switch", dependencies=deps)
    async def switch_backend(request: Request) -> dict[str, Any]:
        """Switch the LLM backend type."""
        body = await request.json()
        new_backend = body.get("backend", "")

        # Derive valid backends from the config Literal type (single source of truth)
        from typing import get_args, get_type_hints

        from cognithor.config import CognithorConfig

        _hints = get_type_hints(CognithorConfig, include_extras=True)
        valid = list(get_args(_hints["llm_backend_type"]))
        if new_backend not in valid:
            raise HTTPException(400, f"Invalid backend: {new_backend}. Valid: {valid}")

        # Update config
        if config_manager:
            config_manager.config.llm_backend_type = new_backend
            # Save to config.yaml
            with contextlib.suppress(Exception):
                config_manager.save()

        return {
            "status": "switched",
            "backend": new_backend,
            "note": "Restart required for full effect",
        }
