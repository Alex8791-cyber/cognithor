"""CLI — `run --interview` smoke test with mocked Crew."""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from insurance_agent_pack.cli import main


def test_help_exits_zero(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "insurance-agent-pack" in captured.out.lower() or "run" in captured.out


def test_run_without_subcommand_returns_nonzero() -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code != 0


def test_run_interview_kicks_off_crew(monkeypatch, capsys) -> None:
    fake_output = MagicMock()
    fake_output.raw = "## Pre-Beratungs-Report\n\n- Beobachtung 1"
    fake_output.tasks_outputs = []

    fake_crew = MagicMock()
    fake_crew.kickoff_async = AsyncMock(return_value=fake_output)

    # Pre-fill stdin to short-circuit the interview prompts
    monkeypatch.setattr("sys.stdin", io.StringIO("Alex\n45\nGGF\nkeine\nq\n"))

    with patch("insurance_agent_pack.cli.build_team", return_value=fake_crew):
        rc = main(["run", "--interview", "--model", "ollama/qwen3:8b"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "Pre-Beratungs-Report" in captured.out
