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

    def test_sprint15_telemetry_fields_default_none(self) -> None:
        event = ArcAuditEvent(
            timestamp=0.0,
            event_type="step",
            game_id="ls20",
            level=0,
            step=0,
        )
        assert event.llm_input_tokens is None
        assert event.llm_output_tokens is None
        assert event.llm_finish_reason is None
        assert event.mtp_acceptance_rate is None


class TestSprint15TelemetryLogging:
    def test_log_step_with_llm_kwargs(self) -> None:
        trail = ArcAuditTrail(game_id="ls20")
        trail.log_step(
            level=0,
            step=1,
            action="ACTION1",
            game_state="NOT_FINISHED",
            pixels_changed=2,
            llm_input_tokens=450,
            llm_output_tokens=120,
            llm_think_tokens=80,
            llm_finish_reason="length",
            llm_wall_clock_s=12.5,
        )
        ev = trail.events[0]
        assert ev.llm_input_tokens == 450
        assert ev.llm_output_tokens == 120
        assert ev.llm_finish_reason == "length"
        # Chain integrity holds with the new fields included.
        assert trail.verify_integrity() is True

    def test_log_step_with_mtp_kwargs(self) -> None:
        trail = ArcAuditTrail(game_id="ls20")
        trail.log_step(
            level=0,
            step=1,
            action="ACTION1",
            game_state="NOT_FINISHED",
            pixels_changed=2,
            mtp_drafts_proposed=12,
            mtp_drafts_accepted=9,
            mtp_acceptance_rate=0.75,
        )
        ev = trail.events[0]
        assert ev.mtp_drafts_proposed == 12
        assert ev.mtp_drafts_accepted == 9
        assert ev.mtp_acceptance_rate == 0.75


class TestHashlineSeal:
    def test_export_jsonl_seals_into_hashline(self, tmp_path: Path) -> None:
        trail = ArcAuditTrail(game_id="ls20")
        trail.log_game_start()
        trail.log_step(
            level=0, step=0, action="ACTION1", game_state="NOT_FINISHED", pixels_changed=1
        )
        trail.log_game_end(final_score=0.5)

        out_path = tmp_path / "audit.jsonl"
        seal_hash = trail.export_jsonl(
            str(out_path),
            seal_into_hashline=True,
            hashline_data_dir=tmp_path,
        )
        # Sealed → returned a SHA-256 hex string.
        assert seal_hash is not None
        assert len(seal_hash) == 64
        # Hashline audit file exists + the seal entry is in it.
        hashline_log = tmp_path / "hashline_audit.jsonl"
        assert hashline_log.exists()
        last_line = hashline_log.read_text(encoding="utf-8").strip().splitlines()[-1]
        import json as _json

        entry = _json.loads(last_line)
        assert entry["type"] == "arc_episode_export"
        assert entry["game_id"] == "ls20"
        assert entry["events_count"] == 3
        assert "jsonl_sha256" in entry

    def test_export_without_seal_returns_none(self, tmp_path: Path) -> None:
        trail = ArcAuditTrail(game_id="ls20")
        trail.log_game_start()
        out_path = tmp_path / "audit.jsonl"
        result = trail.export_jsonl(str(out_path))  # default seal_into_hashline=False
        assert result is None
