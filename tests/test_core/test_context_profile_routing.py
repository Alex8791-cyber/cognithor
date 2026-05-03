"""Tests for Sprint-23 task-aware context profile routing.

Tests:
  - ContextProfile dataclass shape + frozen
  - CONTEXT_PROFILES registry contents (4 profiles, expected num_ctx)
  - set_context_profile / get_context_profile / clear_context_profile
  - Invalid profile name raises ValueError with available options listed
  - get_model_config() overlays the profile when set
  - Embedding model ignores the overlay
  - Profile + urgency compose orthogonally (urgency picks model, profile picks ctx)
  - ContextVar isolation across asyncio tasks (no cross-request leakage)
"""

from __future__ import annotations

import asyncio
import contextvars

import pytest

from cognithor.config import CognithorConfig
from cognithor.core.model_router import (
    CONTEXT_PROFILES,
    ContextProfile,
    ModelRouter,
    OllamaClient,
    _context_profile_var,
    _urgency_var,
)


@pytest.fixture()
def config(tmp_path) -> CognithorConfig:
    return CognithorConfig(cognithor_home=tmp_path)


@pytest.fixture()
def client(config: CognithorConfig) -> OllamaClient:
    return OllamaClient(config)


@pytest.fixture()
def router(config: CognithorConfig, client: OllamaClient) -> ModelRouter:
    return ModelRouter(config, client)


@pytest.fixture(autouse=True)
def _reset_context_state():
    """Reset both ContextVars between tests to avoid cross-test bleed."""
    _context_profile_var.set(None)
    _urgency_var.set(None)
    yield
    _context_profile_var.set(None)
    _urgency_var.set(None)


# ============================================================================
# ContextProfile dataclass
# ============================================================================


class TestContextProfile:
    def test_profile_is_frozen(self) -> None:
        p = ContextProfile(name="test", num_ctx=1024, temperature=0.5, top_p=0.9, description="d")
        with pytest.raises(AttributeError):
            p.num_ctx = 2048  # type: ignore[misc]

    def test_profile_fields(self) -> None:
        p = ContextProfile(
            name="x",
            num_ctx=8192,
            temperature=0.3,
            top_p=0.95,
            description="X profile",
        )
        assert p.name == "x"
        assert p.num_ctx == 8192
        assert p.temperature == 0.3
        assert p.top_p == 0.95
        assert p.description == "X profile"


# ============================================================================
# CONTEXT_PROFILES registry
# ============================================================================


class TestContextProfileRegistry:
    def test_all_four_profiles_exist(self) -> None:
        assert set(CONTEXT_PROFILES) == {"quick", "default", "deep", "arc_agi3"}

    def test_quick_window_is_8k(self) -> None:
        assert CONTEXT_PROFILES["quick"].num_ctx == 8192

    def test_default_window_is_32k(self) -> None:
        assert CONTEXT_PROFILES["default"].num_ctx == 32768

    def test_deep_window_is_64k(self) -> None:
        assert CONTEXT_PROFILES["deep"].num_ctx == 65536

    def test_arc_agi3_window_is_128k(self) -> None:
        # Sprint-22 side-quest probe verified Qwen3.6:27b decodes
        # correctly at 128 k under vLLM. Pin the value here so the
        # ARC-AGI-3 game-loop interaction inherits it.
        assert CONTEXT_PROFILES["arc_agi3"].num_ctx == 131072

    def test_window_sizes_are_strictly_monotonic(self) -> None:
        # quick < default < deep < arc_agi3
        order = ("quick", "default", "deep", "arc_agi3")
        widths = [CONTEXT_PROFILES[name].num_ctx for name in order]
        assert widths == sorted(widths)
        assert len(set(widths)) == len(widths)

    def test_arc_agi3_uses_low_temperature(self) -> None:
        # Game-loop interaction needs deterministic action selection,
        # so the profile must keep temperature below 0.5.
        assert CONTEXT_PROFILES["arc_agi3"].temperature <= 0.5

    def test_get_context_profile_spec_found(self) -> None:
        p = ModelRouter.get_context_profile_spec("quick")
        assert p is not None
        assert p.name == "quick"

    def test_get_context_profile_spec_unknown(self) -> None:
        assert ModelRouter.get_context_profile_spec("nonexistent") is None


# ============================================================================
# set_context_profile / get / clear
# ============================================================================


class TestContextProfileLifecycle:
    def test_default_is_none(self, router: ModelRouter) -> None:
        assert router.get_context_profile() is None

    def test_set_then_get(self, router: ModelRouter) -> None:
        router.set_context_profile("deep")
        assert router.get_context_profile() == "deep"

    def test_set_then_clear(self, router: ModelRouter) -> None:
        router.set_context_profile("arc_agi3")
        router.clear_context_profile()
        assert router.get_context_profile() is None

    def test_invalid_name_raises_with_valid_options(self, router: ModelRouter) -> None:
        with pytest.raises(ValueError, match="Unknown context profile"):
            router.set_context_profile("nuclear")

    def test_invalid_name_lists_available_options(self, router: ModelRouter) -> None:
        with pytest.raises(ValueError) as excinfo:
            router.set_context_profile("bogus")
        # Each registered profile must appear in the error message so
        # the caller knows what they can use instead.
        for valid in ("quick", "default", "deep", "arc_agi3"):
            assert valid in str(excinfo.value)


# ============================================================================
# get_model_config overlay
# ============================================================================


class TestContextProfileOverlay:
    def test_no_profile_uses_model_defaults(self, router: ModelRouter, config) -> None:
        # Without an active profile, the model's own defaults from
        # config flow through unchanged.
        cfg = router.get_model_config(config.models.planner.name)
        assert cfg["context_window"] == config.models.planner.context_window

    def test_quick_overlay_shrinks_ctx_to_8k(self, router: ModelRouter, config) -> None:
        router.set_context_profile("quick")
        cfg = router.get_model_config(config.models.planner.name)
        assert cfg["context_window"] == 8192
        assert cfg["temperature"] == 0.3

    def test_arc_agi3_overlay_grows_ctx_to_128k(self, router: ModelRouter, config) -> None:
        router.set_context_profile("arc_agi3")
        cfg = router.get_model_config(config.models.planner.name)
        assert cfg["context_window"] == 131072

    def test_unknown_model_still_gets_overlay(self, router: ModelRouter) -> None:
        # Unknown model names normally fall back to {temp 0.7, top_p 0.9,
        # ctx 32k}. The overlay must apply to that fallback too —
        # otherwise switching providers (vLLM, OpenAI, …) would silently
        # drop the profile.
        router.set_context_profile("deep")
        cfg = router.get_model_config("some-third-party-model:8b")
        assert cfg["context_window"] == 65536
        assert cfg["temperature"] == 0.8

    def test_embedding_model_ignores_overlay(self, router: ModelRouter, config) -> None:
        # Embedding models have no notion of "creative sampling" — the
        # profile's temperature / top_p are inappropriate. They must
        # keep the family defaults from config.
        router.set_context_profile("arc_agi3")
        cfg = router.get_model_config(config.models.embedding.name)
        assert cfg["temperature"] == config.models.embedding.temperature
        assert cfg["top_p"] == config.models.embedding.top_p


# ============================================================================
# Orthogonality with ConciergeProfile (urgency)
# ============================================================================


class TestProfileOrthogonality:
    def test_urgency_and_context_profile_compose(self, router: ModelRouter) -> None:
        # Urgency chooses the model, context profile chooses the
        # window+sampling. They are independent dimensions.
        router.set_urgency("asap")
        router.set_context_profile("quick")
        assert router.get_urgency() == "asap"
        assert router.get_context_profile() == "quick"

    def test_clearing_one_does_not_affect_the_other(self, router: ModelRouter) -> None:
        router.set_urgency("balanced")
        router.set_context_profile("deep")
        router.clear_context_profile()
        assert router.get_urgency() == "balanced"
        assert router.get_context_profile() is None


# ============================================================================
# ContextVar isolation under concurrent asyncio tasks
# ============================================================================


class TestContextVarIsolation:
    def test_two_tasks_get_independent_profiles(self, router: ModelRouter) -> None:
        """Concurrent asyncio.Tasks each see only their own profile.

        Without ContextVar isolation, request A setting ``arc_agi3``
        before yielding would leak into request B's get_model_config,
        ballooning their context window 16x and throwing GPU OOM. This
        test pins the isolation guarantee.
        """

        async def _task(profile_name: str, holder: dict[str, str | None]) -> None:
            ctx = contextvars.copy_context()

            def _inside() -> None:
                router.set_context_profile(profile_name)
                holder[profile_name] = router.get_context_profile()

            ctx.run(_inside)

        async def _drive() -> dict[str, str | None]:
            holder: dict[str, str | None] = {}
            await asyncio.gather(
                _task("quick", holder),
                _task("arc_agi3", holder),
                _task("deep", holder),
            )
            return holder

        result = asyncio.run(_drive())
        assert result == {"quick": "quick", "arc_agi3": "arc_agi3", "deep": "deep"}
        # The outer test ContextVar is untouched.
        assert router.get_context_profile() is None
