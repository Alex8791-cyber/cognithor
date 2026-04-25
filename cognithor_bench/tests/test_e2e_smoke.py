"""End-to-end smoke run — uses MockAdapter so CI never hits an LLM."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from cognithor_bench.cli import main


def test_full_smoke_run_with_mock_adapter(tmp_path: Path, capsys) -> None:
    smoke = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "cognithor_bench"
        / "scenarios"
        / "smoke_test.jsonl"
    )

    # Stub CognithorAdapter to deterministic outputs (no LLM call).
    with patch("cognithor_bench.cli.CognithorAdapter") as ca:
        instance = MagicMock()
        instance.name = "cognithor"

        async def _run(s):
            from cognithor_bench.adapters.base import ScenarioResult

            return ScenarioResult(
                id=s.id,
                output=s.expected,
                success=True,
                duration_sec=0.001,
                error=None,
            )

        instance.run = AsyncMock(side_effect=_run)
        ca.return_value = instance

        out_dir = tmp_path / "results"
        rc = main(["run", str(smoke), "--output-dir", str(out_dir)])
        assert rc == 0

        files = list(out_dir.glob("*.jsonl"))
        assert len(files) == 1
        rows = [
            json.loads(line)
            for line in files[0].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(rows) == 3
        assert all(r["success"] for r in rows)
