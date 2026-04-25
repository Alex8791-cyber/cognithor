"""End-to-end run with local Ollama. Marked slow — opt-in for CI."""

from __future__ import annotations

import os

import pytest


@pytest.mark.slow
@pytest.mark.asyncio
async def test_full_run_with_local_ollama() -> None:
    """Skipped by default. Run via `pytest -m slow examples/insurance-agent-pack/tests/`.

    Requires `OLLAMA_HOST=http://localhost:11434` and a running Ollama instance
    with `qwen3:8b` pulled. Verifies a complete end-to-end kickoff WITHOUT any
    external API calls.
    """
    if not os.environ.get("OLLAMA_HOST"):
        pytest.skip("OLLAMA_HOST not set; skipping local-inference test")

    from insurance_agent_pack.crew import build_team

    crew = build_team(model="ollama/qwen3:8b")
    result = await crew.kickoff_async(
        {
            "name": "Anon",
            "age": "40",
            "berufsstatus": "GGF",
            "bestehende_policen": "keine",
        }
    )

    raw = str(getattr(result, "raw", "") or "")
    assert raw, "expected a non-empty Pre-Beratungs-Report"
    # Should NOT contain any §34d-style binding recommendation
    forbidden = ["schließe", "empfehle ich konkret", "kaufe", "vermeide unbedingt"]
    for f in forbidden:
        assert f.lower() not in raw.lower(), f"report contained forbidden token: {f!r}"
