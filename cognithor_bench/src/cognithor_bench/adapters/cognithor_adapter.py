"""Default CognithorAdapter — wraps cognithor.crew.Crew."""

from __future__ import annotations

import asyncio
import time

from cognithor.crew import Crew, CrewAgent, CrewTask  # type: ignore[attr-defined]

from cognithor_bench.adapters.base import ScenarioInput, ScenarioResult


class CognithorAdapter:
    """Run a scenario through a single-agent, single-task Cognithor Crew."""

    name = "cognithor"

    def __init__(self, *, model: str = "ollama/qwen3:8b") -> None:
        self.model = model

    async def run(self, scenario: ScenarioInput) -> ScenarioResult:
        start = time.perf_counter()
        try:
            agent = CrewAgent(
                role="bench-agent",
                goal=f"Answer the user's task accurately: {scenario.task}",
                backstory="You answer benchmark questions with one short string.",
                llm=self.model,
                verbose=False,
            )
            task = CrewTask(
                description=scenario.task,
                expected_output=scenario.expected,
                agent=agent,
            )
            crew = Crew(agents=[agent], tasks=[task])

            output = await asyncio.wait_for(
                crew.kickoff_async({}),
                timeout=scenario.timeout_sec,
            )
            raw = str(getattr(output, "raw", "") or "")
            success = scenario.expected.lower() in raw.lower()
            return ScenarioResult(
                id=scenario.id,
                output=raw,
                success=success,
                duration_sec=time.perf_counter() - start,
                error=None,
            )
        except Exception as exc:
            return ScenarioResult(
                id=scenario.id,
                output="",
                success=False,
                duration_sec=time.perf_counter() - start,
                error=f"{type(exc).__name__}: {exc}",
            )
