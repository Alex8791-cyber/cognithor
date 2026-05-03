# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Cognithor · Sprint-23 — Context-Profile REST endpoints.

Exposes the :class:`ContextProfile` system from
``cognithor.core.model_router`` over HTTP so the Flutter UI can let
the user pick a profile and see what's currently active.

Endpoints (all under ``/api/v1/context_profile``):

* ``GET  /``  — return ``{"active": <name|None>, "available": [...]}``
* ``POST /``  — body ``{"profile": "deep" | null}``; ``null`` clears
  the active profile and falls back to the model defaults.

Profile-list metadata is sourced from
:data:`CONTEXT_PROFILES` so adding a profile in code automatically
shows up in the UI without an API change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    from fastapi import HTTPException
except ImportError:
    try:
        from starlette.exceptions import HTTPException  # type: ignore[assignment]
    except ImportError:
        HTTPException = Exception  # type: ignore[assignment,misc]

from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.config_manager import ConfigManager  # noqa: F401

log = get_logger(__name__)


def _profile_payload() -> dict[str, Any]:
    """Build the JSON-serialisable list of available profiles."""
    from cognithor.core.model_router import CONTEXT_PROFILES

    return {
        name: {
            "name": p.name,
            "num_ctx": p.num_ctx,
            "temperature": p.temperature,
            "top_p": p.top_p,
            "description": p.description,
        }
        for name, p in CONTEXT_PROFILES.items()
    }


def _register_context_profile_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """Hook the GET / POST endpoints onto the FastAPI app."""

    @app.get("/api/v1/context_profile", dependencies=deps)
    async def get_active_context_profile() -> dict[str, Any]:
        """Return the active profile name (or ``None``) and all available profiles."""
        active: str | None = None
        try:
            router = getattr(gateway, "_model_router", None) or getattr(
                gateway, "model_router", None
            )
            if router is not None and hasattr(router, "get_context_profile"):
                active = router.get_context_profile()
        except Exception:
            log.debug("context_profile_get_active_failed", exc_info=True)
        return {"active": active, "available": _profile_payload()}

    @app.post("/api/v1/context_profile", dependencies=deps)
    async def set_active_context_profile(payload: dict[str, Any]) -> dict[str, Any]:
        """Set or clear the active context profile.

        Body: ``{"profile": "deep"}`` → activate;
        ``{"profile": null}`` → clear (model defaults take over).
        """
        from cognithor.core.model_router import CONTEXT_PROFILES

        profile = payload.get("profile")
        router = getattr(gateway, "_model_router", None) or getattr(gateway, "model_router", None)
        if router is None:
            raise HTTPException(status_code=503, detail="model router not initialised")

        if profile is None:
            router.clear_context_profile()
            return {"active": None, "available": _profile_payload()}

        if not isinstance(profile, str) or profile not in CONTEXT_PROFILES:
            raise HTTPException(
                status_code=400,
                detail=(f"unknown profile {profile!r}. valid: {sorted(CONTEXT_PROFILES)}"),
            )

        router.set_context_profile(profile)
        return {"active": profile, "available": _profile_payload()}
