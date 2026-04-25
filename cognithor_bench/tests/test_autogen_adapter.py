"""AutoGenAdapter — opt-in via [autogen] extra; ImportError-safe."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognithor_bench.adapters.autogen_adapter import _AUTOGEN_IMPORT_ERROR_HINT, AutoGenAdapter
from cognithor_bench.adapters.base import ScenarioInput


def test_autogen_adapter_name() -> None:
    a = AutoGenAdapter(model="ollama/qwen3:8b")
    assert a.name == "autogen"


@pytest.mark.asyncio
async def test_autogen_adapter_raises_when_import_missing(monkeypatch) -> None:
    """If autogen_agentchat is not installed, .run raises a helpful ImportError."""
    import builtins

    real_import = builtins.__import__

    def fail_import(name: str, *args, **kwargs):
        if name.startswith("autogen_agentchat"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fail_import)

    a = AutoGenAdapter(model="ollama/qwen3:8b")
    scenario = ScenarioInput(id="s1", task="2+2", expected="4", timeout_sec=10, requires=())
    with pytest.raises(ImportError) as exc:
        await a.run(scenario)
    assert _AUTOGEN_IMPORT_ERROR_HINT in str(exc.value)


@pytest.mark.asyncio
async def test_autogen_adapter_runs_when_import_succeeds() -> None:
    """When autogen_agentchat is importable, adapter runs and reports success."""
    fake_msg = MagicMock()
    fake_msg.content = "4"
    fake_result = MagicMock()
    fake_result.messages = [fake_msg]

    fake_agent = MagicMock()
    fake_agent.run = AsyncMock(return_value=fake_result)

    fake_agents_module = MagicMock()
    fake_agents_module.AssistantAgent = MagicMock(return_value=fake_agent)

    fake_models_module = MagicMock()
    fake_models_module.OpenAIChatCompletionClient = MagicMock(return_value=MagicMock())

    with patch.dict(
        sys.modules,
        {
            "autogen_agentchat.agents": fake_agents_module,
            "autogen_ext.models.openai": fake_models_module,
        },
    ):
        a = AutoGenAdapter(model="ollama/qwen3:8b")
        scenario = ScenarioInput(id="s1", task="2+2", expected="4", timeout_sec=10, requires=())
        result = await a.run(scenario)
        assert result.success is True
        assert "4" in result.output
