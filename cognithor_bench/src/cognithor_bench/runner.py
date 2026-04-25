"""BenchRunner — load JSONL scenarios + execute through an Adapter."""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from cognithor_bench.adapters.base import ScenarioInput

if TYPE_CHECKING:
    from pathlib import Path

    from cognithor_bench.adapters.base import Adapter, ScenarioResult


class BenchRunner:
    """Drives an Adapter across a JSONL scenario file."""

    def __init__(self, *, adapter: Adapter, seed: int | None = None) -> None:
        self.adapter = adapter
        self._rng = random.Random(seed)

    async def run_file(
        self,
        scenario_path: Path,
        *,
        repeat: int = 1,
        subsample: float = 1.0,
        output_dir: Path | None = None,
    ) -> list[ScenarioResult]:
        scenarios = self._load(scenario_path)
        if subsample < 1.0:
            n = max(1, round(len(scenarios) * subsample))
            scenarios = self._rng.sample(scenarios, n)

        results: list[ScenarioResult] = []
        for s in scenarios:
            for _ in range(repeat):
                results.append(await self.adapter.run(s))

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
            out_file = output_dir / f"{self.adapter.name}-{scenario_path.stem}-{stamp}.jsonl"
            out_file.write_text(
                "\n".join(r.model_dump_json() for r in results) + "\n",
                encoding="utf-8",
            )
        return results

    @staticmethod
    def _load(scenario_path: Path) -> list[ScenarioInput]:
        scenarios: list[ScenarioInput] = []
        for line in scenario_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            row.setdefault("requires", [])
            row["requires"] = tuple(row["requires"])
            scenarios.append(ScenarioInput(**row))
        return scenarios
