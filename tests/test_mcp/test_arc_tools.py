# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-12 — MCP arc_tools rewire tests.

Verifies that ``handle_arc_play`` and ``handle_arc_replay`` use the
new ``program_synthesis.arc_agi3`` stack (EpisodeRunner + ArcAuditTrail
JSONL) instead of the legacy ``cognithor.arc.agent`` / ``cognithor.arc.audit``
imports.

The tests don't drive a real arc_agi harness — they exercise the
input-validation + error paths and the JSONL-replay format.
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from cognithor.mcp.arc_tools import (
    handle_arc_play,
    handle_arc_replay,
    handle_arc_status,
)


class TestArcPlayValidation:
    @pytest.mark.asyncio
    async def test_missing_game_id(self) -> None:
        result = await handle_arc_play()
        assert "game_id" in result
        assert "required" in result.lower()

    @pytest.mark.asyncio
    async def test_empty_game_id(self) -> None:
        result = await handle_arc_play(game_id="")
        assert "required" in result.lower()

    @pytest.mark.asyncio
    async def test_arc_agi_missing_returns_graceful_error(self) -> None:
        # Block arc_agi import so EpisodeRunner._connect raises.
        with patch.dict(sys.modules, {"arc_agi": None}):
            result = await handle_arc_play(game_id="smoke", use_llm=False, max_steps=3)
        # Either the runner returns an ERROR result or the function
        # surfaces it as a string. Either way the response is non-empty
        # and references the game_id.
        assert "smoke" in result.lower() or "error" in result.lower()


class TestArcReplayJsonl:
    @pytest.mark.asyncio
    async def test_missing_game_id(self) -> None:
        result = await handle_arc_replay()
        assert "required" in result.lower()

    @pytest.mark.asyncio
    async def test_no_audit_file_returns_clear_message(self, tmp_path: Path) -> None:
        bogus = tmp_path / "does_not_exist.jsonl"
        result = await handle_arc_replay(game_id="ghost", audit_path=str(bogus))
        assert "No audit trail" in result or "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_reads_jsonl_format(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.jsonl"
        events: list[dict[str, Any]] = [
            {
                "event_type": "game_start",
                "game_id": "smoke",
                "step": 0,
                "action": None,
            },
            {
                "event_type": "step",
                "game_id": "smoke",
                "step": 1,
                "action": "ACTION1",
            },
            {
                "event_type": "game_end",
                "game_id": "smoke",
                "step": 2,
                "action": None,
                "score": 1.0,
            },
        ]
        path.write_text(
            "\n".join(json.dumps(e) for e in events),
            encoding="utf-8",
        )

        result = await handle_arc_replay(game_id="smoke", audit_path=str(path))
        assert "3 recorded event" in result
        assert "smoke" in result

    @pytest.mark.asyncio
    async def test_verbose_lists_events(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.jsonl"
        path.write_text(
            json.dumps({"event_type": "step", "step": 1, "action": "ACTION1"}) + "\n",
            encoding="utf-8",
        )
        result = await handle_arc_replay(game_id="smoke", audit_path=str(path), verbose=True)
        assert "ACTION1" in result
        assert "step=1" in result


class TestArcStatusUnchanged:
    """The status handler doesn't depend on any agent stack — quick sanity check."""

    @pytest.mark.asyncio
    async def test_no_active_sessions(self) -> None:
        # Clear any session state from prior tests.
        from cognithor.mcp.arc_tools import _active_sessions

        _active_sessions.clear()
        result = await handle_arc_status()
        assert "No active" in result
