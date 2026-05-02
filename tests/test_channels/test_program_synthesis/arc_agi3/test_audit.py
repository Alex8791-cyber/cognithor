# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-12 — ArcAuditTrail tests."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from cognithor.channels.program_synthesis.arc_agi3.audit import (
    ArcAuditEvent,
    ArcAuditTrail,
)
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestArcAuditTrailBasics:
    def test_run_id_is_16_hex_chars(self) -> None:
        trail = ArcAuditTrail(game_id="ls20")
        assert len(trail.run_id) == 16
        # hex
        int(trail.run_id, 16)

    def test_distinct_runs_get_distinct_ids(self) -> None:
        a = ArcAuditTrail(game_id="ls20")
        b = ArcAuditTrail(game_id="ls20")
        assert a.run_id != b.run_id

    def test_log_event_returns_hash(self) -> None:
        trail = ArcAuditTrail(game_id="ls20")
        h = trail.log_game_start()
        # SHA-256 hex = 64 chars
        assert len(h) == 64
        int(h, 16)

    def test_chain_links_events(self) -> None:
        trail = ArcAuditTrail(game_id="ls20")
        trail.log_game_start()
        h2 = trail.log_step(
            level=0, step=1, action="ACTION1", game_state="NOT_FINISHED", pixels_changed=0
        )
        h3 = trail.log_step(
            level=0, step=2, action="ACTION2", game_state="NOT_FINISHED", pixels_changed=4
        )
        assert h2 != h3
        assert len(trail.events) == 3


class TestIntegrity:
    def test_clean_chain_verifies(self) -> None:
        trail = ArcAuditTrail(game_id="ls20")
        trail.log_game_start()
        trail.log_step(
            level=0, step=1, action="ACTION1", game_state="NOT_FINISHED", pixels_changed=0
        )
        trail.log_step(
            level=0, step=2, action="ACTION2", game_state="NOT_FINISHED", pixels_changed=4
        )
        trail.log_game_end(final_score=0.5)
        assert trail.verify_integrity() is True

    def test_tamper_detected(self) -> None:
        trail = ArcAuditTrail(game_id="ls20")
        trail.log_game_start()
        trail.log_step(
            level=0, step=1, action="ACTION1", game_state="NOT_FINISHED", pixels_changed=0
        )
        trail.events[1].action = "TAMPERED"
        assert trail.verify_integrity() is False

    def test_empty_trail_verifies(self) -> None:
        trail = ArcAuditTrail(game_id="ls20")
        assert trail.verify_integrity() is True


class TestExport:
    def test_export_jsonl_roundtrip(self, tmp_path: Path) -> None:
        trail = ArcAuditTrail(game_id="ls20")
        trail.log_game_start()
        trail.log_step(
            level=0, step=1, action="ACTION1", game_state="NOT_FINISHED", pixels_changed=0
        )
        path = tmp_path / "audit.jsonl"
        trail.export_jsonl(str(path))
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["event_type"] == "game_start"
        assert first["game_id"] == "ls20"


class TestEventDataclass:
    def test_event_has_required_fields(self) -> None:
        event = ArcAuditEvent(
            timestamp=time.time(),
            event_type="step",
            game_id="ls20",
            level=0,
            step=1,
            action="ACTION1",
        )
        assert event.event_type == "step"
        assert event.action == "ACTION1"

    def test_optional_fields_default_none(self) -> None:
        event = ArcAuditEvent(
            timestamp=0.0,
            event_type="game_start",
            game_id="ls20",
            level=0,
            step=0,
        )
        assert event.error is None
        assert event.score is None
        assert event.metadata is None
