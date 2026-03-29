"""ARC-AGI-3 Swarm Orchestrator.

Runs multiple ARC-AGI-3 game sessions in parallel using asyncio.Semaphore
and run_in_executor to keep sync agent code off the event loop.

Usage:
    orchestrator = ArcSwarmOrchestrator(max_parallel=4)
    await orchestrator.run_swarm(["game_a", "game_b", "game_c"])
    print(orchestrator.get_summary())
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

from jarvis.utils.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "SwarmResult",
    "ArcSwarmOrchestrator",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class SwarmResult:
    """Result of a single game run inside the swarm."""

    game_id: str
    score: float = 0.0
    levels_completed: int = 0
    total_steps: int = 0
    errors: list[str] = field(default_factory=list)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class ArcSwarmOrchestrator:
    """Runs multiple ARC-AGI-3 games in parallel.

    Args:
        max_parallel: Maximum number of concurrent game sessions.
        use_llm: Whether to enable the LLM planner for each agent.
        config: Optional JarvisConfig instance (reads arc sub-config).
    """

    def __init__(
        self,
        max_parallel: int = 4,
        use_llm: bool = True,
        config: Any = None,
    ) -> None:
        self._max_parallel = max_parallel
        self._use_llm = use_llm
        self._config = config
        self._results: list[SwarmResult] = []

        # Read ARC-specific settings from config if available
        arc_cfg = getattr(config, "arc", None) if config is not None else None
        self._max_steps: int = getattr(arc_cfg, "max_steps_per_level", 500) if arc_cfg else 500
        self._max_resets: int = getattr(arc_cfg, "max_resets_per_level", 5) if arc_cfg else 5
        self._llm_interval: int = getattr(arc_cfg, "llm_call_interval", 10) if arc_cfg else 10

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_swarm(self, game_ids: list[str]) -> list[SwarmResult]:
        """Run all game_ids concurrently, bounded by max_parallel.

        Args:
            game_ids: List of ARC-AGI-3 environment identifiers.

        Returns:
            List of SwarmResult, one per game.
        """
        self._results = []
        semaphore = asyncio.Semaphore(self._max_parallel)

        async def bounded(gid: str) -> None:
            async with semaphore:
                result = await self._run_single(gid)
                self._results.append(result)

        await asyncio.gather(*(bounded(gid) for gid in game_ids))
        return self._results

    def get_aggregate_score(self) -> float:
        """Return the mean score across all completed games.

        Returns 0.0 if no results are available.
        """
        if not self._results:
            return 0.0
        return sum(r.score for r in self._results) / len(self._results)

    def get_summary(self) -> str:
        """Return an ASCII-safe summary of all swarm results."""
        if not self._results:
            return "No swarm results available."

        total = len(self._results)
        ok_count = sum(1 for r in self._results if not r.errors)
        fail_count = total - ok_count
        mean_score = self.get_aggregate_score()
        total_steps = sum(r.total_steps for r in self._results)
        total_levels = sum(r.levels_completed for r in self._results)

        lines = [
            f"ARC Swarm Summary ({total} game(s)):",
            f"  Completed : {ok_count} [OK]  {fail_count} [FAIL]",
            f"  Mean score: {mean_score:.4f}",
            f"  Levels    : {total_levels}",
            f"  Steps     : {total_steps}",
            "",
        ]

        for r in sorted(self._results, key=lambda x: x.game_id):
            status = "[OK]" if not r.errors else "[FAIL]"
            lines.append(
                f"  {status} {r.game_id:<30} "
                f"score={r.score:.4f}  levels={r.levels_completed}  steps={r.total_steps}"
            )
            for err in r.errors:
                lines.append(f"         error: {err}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run_single(self, game_id: str) -> SwarmResult:
        """Run one game in a thread executor, return a SwarmResult."""
        result = SwarmResult(game_id=game_id)
        loop = asyncio.get_running_loop()

        def _sync_run() -> dict[str, Any]:
            try:
                from jarvis.arc.agent import CognithorArcAgent
            except ImportError as exc:
                raise RuntimeError(
                    f"CognithorArcAgent not importable: {exc}. Install jarvis[arc] dependencies."
                ) from exc

            agent = CognithorArcAgent(
                game_id=game_id,
                use_llm_planner=self._use_llm,
                llm_call_interval=self._llm_interval,
                max_steps_per_level=self._max_steps,
                max_resets_per_level=self._max_resets,
            )
            return agent.run()

        try:
            raw: dict[str, Any] = await loop.run_in_executor(None, _sync_run)
            result.score = float(raw.get("score", 0.0))
            result.levels_completed = int(raw.get("levels_completed", 0))
            result.total_steps = int(raw.get("total_steps", 0))
            log.info(
                "arc.swarm.game_done",
                game_id=game_id,
                score=result.score,
                levels=result.levels_completed,
            )
        except Exception as exc:
            err_msg = str(exc)
            result.errors.append(err_msg)
            log.error("arc.swarm.game_failed", game_id=game_id, error=err_msg)

        return result
