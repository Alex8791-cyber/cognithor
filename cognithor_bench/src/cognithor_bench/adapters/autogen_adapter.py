"""AutoGenAdapter — opt-in via [autogen] extra.

Imports `autogen_agentchat` LAZILY inside .run(). If the extra isn't installed,
the adapter raises a helpful ImportError pointing at the install command.
"""

from __future__ import annotations

import asyncio
import time

from cognithor_bench.adapters.base import ScenarioInput, ScenarioResult

_AUTOGEN_IMPORT_ERROR_HINT = (
    "AutoGenAdapter requires `pip install cognithor[autogen]` "
    "which installs both `autogen-agentchat==0.7.5` and "
    "`autogen-ext[openai]==0.7.5` "
    "(autogen-ext supplies OpenAIChatCompletionClient)."
)


class AutoGenAdapter:
    """Run a scenario through autogen-agentchat AssistantAgent."""

    name = "autogen"

    def __init__(self, *, model: str = "ollama/qwen3:8b") -> None:
        self.model = model

    async def run(self, scenario: ScenarioInput) -> ScenarioResult:
        start = time.perf_counter()
        try:
            try:
                from autogen_agentchat.agents import (
                    AssistantAgent,  # type: ignore[import-not-found]
                )
                from autogen_ext.models.openai import (
                    OpenAIChatCompletionClient,  # type: ignore[import-not-found]
                )
            except ImportError as e:
                raise ImportError(_AUTOGEN_IMPORT_ERROR_HINT) from e

            client = OpenAIChatCompletionClient(model=self.model)
            agent = AssistantAgent(
                name="bench-agent",
                model_client=client,
                description="Answers benchmark questions with one short string.",
            )
            result = await asyncio.wait_for(
                agent.run(task=scenario.task),
                timeout=scenario.timeout_sec,
            )
            messages = getattr(result, "messages", []) or []
            raw = str(messages[-1].content) if messages else ""
            success = scenario.expected.lower() in raw.lower()
            return ScenarioResult(
                id=scenario.id,
                output=raw,
                success=success,
                duration_sec=time.perf_counter() - start,
                error=None,
            )
        except ImportError:
            raise
        except Exception as exc:
            return ScenarioResult(
                id=scenario.id,
                output="",
                success=False,
                duration_sec=time.perf_counter() - start,
                error=f"{type(exc).__name__}: {exc}",
            )
