"""Tests for agent-specific model/temperature/top_p overrides in the Planner."""

import pytest
from unittest.mock import AsyncMock, MagicMock


class TestPlannerAgentOverrides:
    """Planner.plan() must accept and apply agent overrides."""

    @pytest.fixture
    def mock_planner(self):
        from jarvis.config import JarvisConfig
        from jarvis.core.model_router import ModelRouter
        from jarvis.core.planner import Planner

        config = JarvisConfig()
        mock_ollama = MagicMock()
        mock_ollama.chat = AsyncMock(
            return_value={
                "message": {"content": "Test response"},
            }
        )
        mock_router = MagicMock(spec=ModelRouter)
        mock_router.select_model.return_value = "qwen3:32b"
        mock_router.get_model_config.return_value = {
            "temperature": 0.7,
            "top_p": 0.9,
        }
        planner = Planner(config, ollama=mock_ollama, model_router=mock_router)

        async def _passthrough_cb(coro):
            return await coro

        planner._llm_circuit_breaker = MagicMock()
        planner._llm_circuit_breaker.call = AsyncMock(side_effect=_passthrough_cb)
        return planner

    @pytest.fixture
    def empty_wm(self):
        from jarvis.models import WorkingMemory

        return WorkingMemory()

    @pytest.mark.asyncio
    async def test_plan_default_model(self, mock_planner, empty_wm):
        await mock_planner.plan("hello", empty_wm, {})
        call_kwargs = mock_planner._ollama.chat.call_args.kwargs
        assert call_kwargs["model"] == "qwen3:32b"

    @pytest.mark.asyncio
    async def test_plan_model_override(self, mock_planner, empty_wm):
        await mock_planner.plan("hello", empty_wm, {}, model_override="qwen3-coder:30b")
        call_kwargs = mock_planner._ollama.chat.call_args.kwargs
        assert call_kwargs["model"] == "qwen3-coder:30b"

    @pytest.mark.asyncio
    async def test_plan_temperature_override(self, mock_planner, empty_wm):
        await mock_planner.plan("hello", empty_wm, {}, temperature_override=0.2)
        call_kwargs = mock_planner._ollama.chat.call_args.kwargs
        assert call_kwargs["temperature"] == 0.2

    @pytest.mark.asyncio
    async def test_plan_top_p_override(self, mock_planner, empty_wm):
        await mock_planner.plan("hello", empty_wm, {}, top_p_override=0.85)
        call_kwargs = mock_planner._ollama.chat.call_args.kwargs
        assert call_kwargs["top_p"] == 0.85

    @pytest.mark.asyncio
    async def test_plan_all_overrides(self, mock_planner, empty_wm):
        await mock_planner.plan(
            "hello",
            empty_wm,
            {},
            model_override="test:7b",
            temperature_override=0.1,
            top_p_override=0.7,
        )
        call_kwargs = mock_planner._ollama.chat.call_args.kwargs
        assert call_kwargs["model"] == "test:7b"
        assert call_kwargs["temperature"] == 0.1
        assert call_kwargs["top_p"] == 0.7

    @pytest.mark.asyncio
    async def test_plan_no_overrides_uses_config_defaults(self, mock_planner, empty_wm):
        await mock_planner.plan("hello", empty_wm, {})
        call_kwargs = mock_planner._ollama.chat.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7  # default from config
        assert call_kwargs["top_p"] == 0.9  # default

    @pytest.mark.asyncio
    async def test_replan_accepts_overrides(self, mock_planner, empty_wm):
        await mock_planner.replan(
            "original goal",
            [],
            empty_wm,
            {},
            model_override="fast:3b",
            temperature_override=0.3,
            top_p_override=0.8,
        )
        call_kwargs = mock_planner._ollama.chat.call_args.kwargs
        assert call_kwargs["model"] == "fast:3b"
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["top_p"] == 0.8
