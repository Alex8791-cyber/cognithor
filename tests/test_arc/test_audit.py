"""Tests for ARC-AGI-3 audit trail with SHA-256 hash chain."""

from __future__ import annotations

import json

from jarvis.arc.audit import ArcAuditTrail, ArcAuditEvent  # noqa: F401


class TestHashChainIntegrity:
    def test_single_event(self):
        trail = ArcAuditTrail("ls20")
        trail.log_game_start()
        assert trail.verify_integrity() is True

    def test_multiple_events(self):
        trail = ArcAuditTrail("ls20")
        trail.log_game_start()
        trail.log_step(0, 1, "ACTION1", "NOT_FINISHED", 50)
        trail.log_step(0, 2, "ACTION3", "WIN", 100)
        trail.log_game_end(0.5)
        assert trail.verify_integrity() is True
        assert len(trail.events) == 4

    def test_tampering_detected(self):
        trail = ArcAuditTrail("ls20")
        trail.log_game_start()
        trail.log_step(0, 1, "ACTION1", "NOT_FINISHED", 50)
        # Tamper with an event
        trail.events[0].action = "TAMPERED"
        assert trail.verify_integrity() is False


class TestExport:
    def test_export_jsonl(self, tmp_path):
        trail = ArcAuditTrail("ls20")
        trail.log_game_start()
        trail.log_step(0, 1, "ACTION1", "NOT_FINISHED", 10)
        filepath = str(tmp_path / "test.jsonl")
        trail.export_jsonl(filepath)
        with open(filepath) as f:
            lines = f.readlines()
        assert len(lines) == 2
        # Verify each line is valid JSON
        for line in lines:
            data = json.loads(line)
            assert "event_type" in data
            assert "game_id" in data

    def test_export_empty(self, tmp_path):
        trail = ArcAuditTrail("ls20")
        filepath = str(tmp_path / "empty.jsonl")
        trail.export_jsonl(filepath)
        with open(filepath) as f:
            assert f.read() == ""


class TestRunId:
    def test_run_id_is_16_chars(self):
        trail = ArcAuditTrail("ls20")
        assert len(trail.run_id) == 16

    def test_different_trails_different_ids(self):
        t1 = ArcAuditTrail("ls20")
        t2 = ArcAuditTrail("ls20")
        # Very likely different due to timestamp
        # (theoretically could collide but astronomically unlikely)
        assert t1.run_id != t2.run_id


class TestConvenienceMethods:
    def test_log_game_start(self):
        trail = ArcAuditTrail("ls20", agent_version="test-v1")
        h = trail.log_game_start()
        assert isinstance(h, str)
        assert trail.events[0].event_type == "game_start"
        assert trail.events[0].metadata["agent_version"] == "test-v1"

    def test_log_game_end(self):
        trail = ArcAuditTrail("ls20")
        trail.log_game_end(0.42)
        assert trail.events[0].event_type == "game_end"
        assert trail.events[0].score == 0.42

    def test_log_step(self):
        trail = ArcAuditTrail("ls20")
        trail.log_step(level=1, step=5, action="ACTION2", game_state="WIN", pixels_changed=30)
        e = trail.events[0]
        assert e.level == 1
        assert e.step == 5
        assert e.action == "ACTION2"
        assert e.pixels_changed == 30
