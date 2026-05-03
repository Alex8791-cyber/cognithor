# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Tests for Sprint-23 PR#J — ``/api/v1/context_profile`` REST endpoints."""

from __future__ import annotations

from typing import Any

import pytest

from cognithor.channels.config_routes.profile import (
    _profile_payload,
    _register_context_profile_routes,
)
from cognithor.config import CognithorConfig
from cognithor.core.model_router import (
    CONTEXT_PROFILES,
    ModelRouter,
    OllamaClient,
    _context_profile_var,
)


class _FakeApp:
    """FastAPI-shaped app stub that captures the route handlers."""

    def __init__(self) -> None:
        self.routes: dict[str, Any] = {}

    def _register(self, method: str, path: str):
        def decorator(fn: Any) -> Any:
            self.routes[f"{method} {path}"] = fn
            return fn

        return decorator

    def get(self, path: str, **_: Any) -> Any:
        return self._register("GET", path)

    def post(self, path: str, **_: Any) -> Any:
        return self._register("POST", path)


class _FakeGateway:
    def __init__(self, router: ModelRouter | None) -> None:
        self._model_router = router


@pytest.fixture()
def config(tmp_path) -> CognithorConfig:
    return CognithorConfig(cognithor_home=tmp_path)


@pytest.fixture()
def router(config: CognithorConfig) -> ModelRouter:
    return ModelRouter(config, OllamaClient(config))


@pytest.fixture(autouse=True)
def _reset() -> Any:
    _context_profile_var.set(None)
    yield
    _context_profile_var.set(None)


# ---------------------------------------------------------------------------
# _profile_payload
# ---------------------------------------------------------------------------


class TestProfilePayload:
    def test_payload_lists_every_profile(self) -> None:
        payload = _profile_payload()
        assert set(payload) == set(CONTEXT_PROFILES)

    def test_payload_carries_full_profile_record(self) -> None:
        payload = _profile_payload()
        for name, profile in CONTEXT_PROFILES.items():
            row = payload[name]
            assert row["name"] == profile.name
            assert row["num_ctx"] == profile.num_ctx
            assert row["temperature"] == profile.temperature
            assert row["top_p"] == profile.top_p
            assert row["description"] == profile.description


# ---------------------------------------------------------------------------
# Route registration + behaviour
# ---------------------------------------------------------------------------


class TestContextProfileRoutes:
    @pytest.mark.asyncio
    async def test_get_returns_active_none_and_full_catalog(self, router: ModelRouter) -> None:
        app = _FakeApp()
        _register_context_profile_routes(app, [], _FakeGateway(router))
        handler = app.routes["GET /api/v1/context_profile"]
        result = await handler()
        assert result["active"] is None
        assert set(result["available"]) == set(CONTEXT_PROFILES)

    @pytest.mark.asyncio
    async def test_get_reflects_active_profile_after_set(self, router: ModelRouter) -> None:
        app = _FakeApp()
        _register_context_profile_routes(app, [], _FakeGateway(router))
        router.set_context_profile("deep")
        handler = app.routes["GET /api/v1/context_profile"]
        result = await handler()
        assert result["active"] == "deep"

    @pytest.mark.asyncio
    async def test_post_activates_profile(self, router: ModelRouter) -> None:
        app = _FakeApp()
        _register_context_profile_routes(app, [], _FakeGateway(router))
        post = app.routes["POST /api/v1/context_profile"]
        result = await post({"profile": "arc_agi3"})
        assert result["active"] == "arc_agi3"
        assert router.get_context_profile() == "arc_agi3"

    @pytest.mark.asyncio
    async def test_post_with_null_clears_profile(self, router: ModelRouter) -> None:
        app = _FakeApp()
        _register_context_profile_routes(app, [], _FakeGateway(router))
        router.set_context_profile("quick")
        post = app.routes["POST /api/v1/context_profile"]
        result = await post({"profile": None})
        assert result["active"] is None
        assert router.get_context_profile() is None

    @pytest.mark.asyncio
    async def test_post_unknown_profile_400(self, router: ModelRouter) -> None:
        app = _FakeApp()
        _register_context_profile_routes(app, [], _FakeGateway(router))
        post = app.routes["POST /api/v1/context_profile"]
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc:
            await post({"profile": "invalid"})
        assert exc.value.status_code == 400
        assert "valid" in exc.value.detail.lower()

    @pytest.mark.asyncio
    async def test_post_without_router_503(self) -> None:
        app = _FakeApp()
        _register_context_profile_routes(app, [], _FakeGateway(None))
        post = app.routes["POST /api/v1/context_profile"]
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc:
            await post({"profile": "deep"})
        assert exc.value.status_code == 503
