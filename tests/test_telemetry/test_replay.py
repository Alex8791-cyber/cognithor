"""Tests for Deterministic Replay System — recorder and replay engine.

Covers: ExecutionEvent, ExecutionRecording, ExecutionRecorder,
ReplayEngine, ReplayDiff, ReplayResult, export/import.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from jarvis.telemetry.recorder import (
    EventType,
    ExecutionEvent,
    ExecutionRecorder,
    ExecutionRecording,
)
from jarvis.telemetry.replay import (
    DiffEntry,
    DiffType,
    ReplayDiff,
    ReplayEngine,
    ReplayResult,
    _diff_dicts,
)


# ============================================================================
# ExecutionEvent
# ============================================================================


class TestExecutionEvent:
    def test_defaults(self) -> None:
        e = ExecutionEvent(event_id="e1", event_type=EventType.TOOL_CALL)
        assert e.sequence == 0
        assert e.data == {}
        assert e.iteration == 0

    def test_round_trip(self) -> None:
        e = ExecutionEvent(
            event_id="e1",
            event_type=EventType.LLM_REQUEST,
            sequence=5,
            data={"prompt": "hello"},
            iteration=2,
            duration_ms=150.5,
        )
        d = e.to_dict()
        e2 = ExecutionEvent.from_dict(d)
        assert e2.event_id == "e1"
        assert e2.event_type == EventType.LLM_REQUEST
        assert e2.sequence == 5
        assert e2.data == {"prompt": "hello"}
        assert e2.iteration == 2
        assert e2.duration_ms == 150.5


# ============================================================================
# ExecutionRecording
# ============================================================================


class TestExecutionRecording:
    def _make_recording(self) -> ExecutionRecording:
        rec = ExecutionRecording(
            recording_id="rec1",
            session_id="sess1",
            agent_name="planner",
            model="qwen3:32b",
            started_at=100.0,
            finished_at=105.0,
            success=True,
        )
        rec.events = [
            ExecutionEvent(event_id="e0", event_type=EventType.USER_MESSAGE, sequence=0),
            ExecutionEvent(
                event_id="e1", event_type=EventType.LLM_REQUEST, sequence=1, iteration=1
            ),
            ExecutionEvent(
                event_id="e2", event_type=EventType.LLM_RESPONSE, sequence=2, iteration=1
            ),
            ExecutionEvent(event_id="e3", event_type=EventType.TOOL_CALL, sequence=3, iteration=1),
            ExecutionEvent(
                event_id="e4", event_type=EventType.TOOL_RESULT, sequence=4, iteration=1
            ),
            ExecutionEvent(
                event_id="e5", event_type=EventType.AGENT_RESPONSE, sequence=5, iteration=1
            ),
        ]
        return rec

    def test_duration_ms(self) -> None:
        rec = self._make_recording()
        assert rec.duration_ms == 5000.0

    def test_event_count(self) -> None:
        rec = self._make_recording()
        assert rec.event_count == 6

    def test_iteration_count(self) -> None:
        rec = self._make_recording()
        assert rec.iteration_count == 1

    def test_tool_calls(self) -> None:
        rec = self._make_recording()
        assert len(rec.tool_calls) == 1

    def test_llm_calls(self) -> None:
        rec = self._make_recording()
        assert len(rec.llm_calls) == 1

    def test_checksum_stable(self) -> None:
        rec = self._make_recording()
        c1 = rec.checksum
        c2 = rec.checksum
        assert c1 == c2
        assert len(c1) == 16

    def test_round_trip(self) -> None:
        rec = self._make_recording()
        d = rec.to_dict()
        rec2 = ExecutionRecording.from_dict(d)
        assert rec2.recording_id == "rec1"
        assert rec2.event_count == 6
        assert rec2.success is True

    def test_unfinished_duration(self) -> None:
        rec = ExecutionRecording(recording_id="r", started_at=100.0)
        assert rec.duration_ms == 0.0


# ============================================================================
# ExecutionRecorder
# ============================================================================


class TestExecutionRecorder:
    def test_start_recording(self) -> None:
        recorder = ExecutionRecorder()
        rec = recorder.start("session1", agent_name="planner")
        assert rec.recording_id
        assert rec.session_id == "session1"
        assert rec.agent_name == "planner"

    def test_finish_recording(self) -> None:
        recorder = ExecutionRecorder()
        rec = recorder.start("s1")
        finished = recorder.finish(rec.recording_id, success=True)
        assert finished is not None
        assert finished.success is True
        assert finished.finished_at > 0

    def test_finish_unknown(self) -> None:
        recorder = ExecutionRecorder()
        assert recorder.finish("unknown") is None

    def test_record_llm_request(self) -> None:
        recorder = ExecutionRecorder()
        rec = recorder.start("s1")
        event = recorder.record_llm_request(
            rec.recording_id, prompt="Hello", model="qwen3:32b", iteration=1
        )
        assert event is not None
        assert event.event_type == EventType.LLM_REQUEST
        assert event.data["model"] == "qwen3:32b"
        assert event.sequence == 0

    def test_record_llm_response(self) -> None:
        recorder = ExecutionRecorder()
        rec = recorder.start("s1")
        event = recorder.record_llm_response(
            rec.recording_id, response="Hi there!", tokens=50, duration_ms=200
        )
        assert event is not None
        assert event.data["tokens"] == 50

    def test_record_tool_call(self) -> None:
        recorder = ExecutionRecorder()
        rec = recorder.start("s1")
        event = recorder.record_tool_call(
            rec.recording_id, tool="web_search", params={"query": "test"}
        )
        assert event.data["tool"] == "web_search"

    def test_record_tool_result(self) -> None:
        recorder = ExecutionRecorder()
        rec = recorder.start("s1")
        event = recorder.record_tool_result(
            rec.recording_id, tool="web_search", result="Results here", duration_ms=150
        )
        assert event.data["is_error"] is False

    def test_record_plan(self) -> None:
        recorder = ExecutionRecorder()
        rec = recorder.start("s1")
        steps = [{"tool": "web_search", "params": {}}]
        event = recorder.record_plan(
            rec.recording_id, goal="Find info", steps=steps, confidence=0.9
        )
        assert event.data["step_count"] == 1
        assert event.data["confidence"] == 0.9

    def test_record_gate_decision(self) -> None:
        recorder = ExecutionRecorder()
        rec = recorder.start("s1")
        event = recorder.record_gate_decision(
            rec.recording_id, tool="exec_command", status="approve", risk_level="orange"
        )
        assert event.data["risk_level"] == "orange"

    def test_record_user_message(self) -> None:
        recorder = ExecutionRecorder()
        rec = recorder.start("s1")
        event = recorder.record_user_message(
            rec.recording_id, message="What is the weather?", channel="telegram"
        )
        assert event.data["channel"] == "telegram"

    def test_record_agent_response(self) -> None:
        recorder = ExecutionRecorder()
        rec = recorder.start("s1")
        event = recorder.record_agent_response(rec.recording_id, response="The weather is sunny")
        assert event.event_type == EventType.AGENT_RESPONSE

    def test_sequence_increments(self) -> None:
        recorder = ExecutionRecorder()
        rec = recorder.start("s1")
        e1 = recorder.record_event(rec.recording_id, EventType.CUSTOM)
        e2 = recorder.record_event(rec.recording_id, EventType.CUSTOM)
        e3 = recorder.record_event(rec.recording_id, EventType.CUSTOM)
        assert e1.sequence == 0
        assert e2.sequence == 1
        assert e3.sequence == 2

    def test_record_on_unknown_recording(self) -> None:
        recorder = ExecutionRecorder()
        assert recorder.record_event("unknown", EventType.CUSTOM) is None

    def test_list_recordings(self) -> None:
        recorder = ExecutionRecorder()
        recorder.start("s1")
        recorder.start("s2")
        assert len(recorder.list_recordings()) == 2

    def test_max_recordings_limit(self) -> None:
        recorder = ExecutionRecorder(max_recordings=3)
        for i in range(5):
            recorder.start(f"s{i}")
        assert len(recorder.list_recordings()) == 3

    def test_get_recording(self) -> None:
        recorder = ExecutionRecorder()
        rec = recorder.start("s1")
        assert recorder.get(rec.recording_id) is not None
        assert recorder.get("nonexistent") is None

    def test_full_lifecycle(self) -> None:
        recorder = ExecutionRecorder()
        rec = recorder.start("s1", agent_name="planner", model="qwen3:32b")
        recorder.record_user_message(rec.recording_id, message="Hello")
        recorder.record_llm_request(rec.recording_id, prompt="Plan for: Hello", iteration=1)
        recorder.record_llm_response(rec.recording_id, response="I'll help", iteration=1)
        recorder.record_plan(rec.recording_id, goal="help", steps=[], iteration=1)
        recorder.record_tool_call(rec.recording_id, tool="memory_search", iteration=1)
        recorder.record_tool_result(
            rec.recording_id, tool="memory_search", result="found", iteration=1
        )
        recorder.record_agent_response(rec.recording_id, response="Here's my answer")
        recorder.finish(rec.recording_id, success=True)

        final = recorder.get(rec.recording_id)
        assert final.event_count == 7
        assert final.success is True
        assert final.finished_at >= final.started_at

    def test_stats(self) -> None:
        recorder = ExecutionRecorder()
        rec = recorder.start("s1")
        recorder.record_event(rec.recording_id, EventType.CUSTOM)
        recorder.finish(rec.recording_id, success=True)
        s = recorder.stats()
        assert s["total_recordings"] == 1
        assert s["total_events"] == 1
        assert s["completed"] == 1
        assert s["successful"] == 1


# ============================================================================
# Export / Import
# ============================================================================


class TestExportImport:
    def test_export_jsonl(self, tmp_path: Path) -> None:
        recorder = ExecutionRecorder()
        rec = recorder.start("s1", agent_name="test")
        recorder.record_tool_call(rec.recording_id, tool="web_search")
        recorder.record_tool_result(rec.recording_id, tool="web_search", result="ok")
        recorder.finish(rec.recording_id, success=True)

        path = tmp_path / "test.jsonl"
        assert recorder.export_jsonl(rec.recording_id, path) is True
        assert path.exists()
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3  # header + 2 events

    def test_export_unknown_recording(self, tmp_path: Path) -> None:
        recorder = ExecutionRecorder()
        assert recorder.export_jsonl("unknown", tmp_path / "test.jsonl") is False

    def test_import_jsonl(self, tmp_path: Path) -> None:
        recorder = ExecutionRecorder()
        rec = recorder.start("s1", agent_name="planner", model="qwen3:32b")
        recorder.record_llm_request(rec.recording_id, prompt="test", model="qwen3:32b")
        recorder.record_llm_response(rec.recording_id, response="ok")
        recorder.finish(rec.recording_id, success=True)

        path = tmp_path / "export.jsonl"
        recorder.export_jsonl(rec.recording_id, path)

        # Import into new recorder
        recorder2 = ExecutionRecorder()
        imported = recorder2.import_jsonl(path)
        assert imported is not None
        assert imported.recording_id == rec.recording_id
        assert imported.event_count == 2
        assert imported.agent_name == "planner"

    def test_import_nonexistent_file(self, tmp_path: Path) -> None:
        recorder = ExecutionRecorder()
        assert recorder.import_jsonl(tmp_path / "nope.jsonl") is None

    def test_round_trip_preserves_data(self, tmp_path: Path) -> None:
        recorder = ExecutionRecorder()
        rec = recorder.start("s1", model="test-model", seed=42, temperature=0.7)
        recorder.record_tool_call(rec.recording_id, tool="exec", params={"cmd": "echo hi"})
        recorder.record_tool_result(rec.recording_id, tool="exec", result="hi", duration_ms=50)
        recorder.finish(rec.recording_id, success=True)

        path = tmp_path / "roundtrip.jsonl"
        recorder.export_jsonl(rec.recording_id, path)

        recorder2 = ExecutionRecorder()
        imported = recorder2.import_jsonl(path)
        assert imported.model == "test-model"
        assert imported.seed == 42
        assert imported.temperature == 0.7
        assert imported.event_count == 2


# ============================================================================
# ReplayEngine
# ============================================================================


class TestReplayEngine:
    def _make_recording(self) -> ExecutionRecording:
        rec = ExecutionRecording(
            recording_id="rec1",
            session_id="sess1",
            agent_name="planner",
            model="qwen3:32b",
            started_at=100.0,
            finished_at=105.0,
            success=True,
        )
        rec.events = [
            ExecutionEvent(
                event_id="rec1:0000",
                event_type=EventType.USER_MESSAGE,
                sequence=0,
                data={"message_preview": "Hello", "message_length": 5},
            ),
            ExecutionEvent(
                event_id="rec1:0001",
                event_type=EventType.LLM_REQUEST,
                sequence=1,
                data={"prompt_preview": "Plan for: Hello", "model": "qwen3:32b"},
                iteration=1,
            ),
            ExecutionEvent(
                event_id="rec1:0002",
                event_type=EventType.LLM_RESPONSE,
                sequence=2,
                data={"response_preview": "I'll help", "tokens": 50},
                iteration=1,
            ),
            ExecutionEvent(
                event_id="rec1:0003",
                event_type=EventType.TOOL_CALL,
                sequence=3,
                data={"tool": "web_search", "params": {"query": "test"}},
                iteration=1,
            ),
            ExecutionEvent(
                event_id="rec1:0004",
                event_type=EventType.TOOL_RESULT,
                sequence=4,
                data={"tool": "web_search", "result_preview": "Results"},
                iteration=1,
            ),
            ExecutionEvent(
                event_id="rec1:0005",
                event_type=EventType.AGENT_RESPONSE,
                sequence=5,
                data={"response_preview": "Here's what I found"},
                iteration=1,
            ),
        ]
        return rec

    def test_basic_replay(self) -> None:
        engine = ReplayEngine()
        rec = self._make_recording()
        result = engine.replay(rec)
        assert result.success is True
        assert result.event_count == 6
        assert result.deterministic is True

    def test_replay_match_rate(self) -> None:
        engine = ReplayEngine()
        rec = self._make_recording()
        result = engine.replay(rec)
        assert result.match_rate == 100.0

    def test_replay_with_override(self) -> None:
        engine = ReplayEngine()
        rec = self._make_recording()
        result = engine.replay(
            rec,
            overrides={
                "rec1:0002": {"data": {"response_preview": "Different answer"}},
            },
        )
        assert result.success is True
        assert result.deterministic is False  # Override changes data

    def test_replay_from_iteration(self) -> None:
        engine = ReplayEngine()
        rec = self._make_recording()
        result = engine.replay_from_iteration(rec, start_iteration=1)
        # Only events with iteration >= 1
        assert result.event_count == 5

    def test_get_llm_responses(self) -> None:
        engine = ReplayEngine()
        rec = self._make_recording()
        responses = engine.get_llm_responses(rec)
        assert len(responses) == 1
        assert responses[0].data["tokens"] == 50

    def test_get_tool_results(self) -> None:
        engine = ReplayEngine()
        rec = self._make_recording()
        results = engine.get_tool_results(rec)
        assert len(results) == 1
        assert results[0].data["tool"] == "web_search"

    def test_replay_results_tracked(self) -> None:
        engine = ReplayEngine()
        rec = self._make_recording()
        engine.replay(rec)
        engine.replay(rec)
        assert len(engine.results) == 2

    def test_stats(self) -> None:
        engine = ReplayEngine()
        rec = self._make_recording()
        engine.replay(rec)
        s = engine.stats()
        assert s["total_replays"] == 1
        assert s["deterministic"] == 1

    def test_override_by_short_id(self) -> None:
        engine = ReplayEngine()
        rec = self._make_recording()
        result = engine.replay(
            rec,
            overrides={
                "0002": {"data": {"response_preview": "Short ID override"}},
            },
        )
        # Find the overridden event
        overridden = [e for e in result.events if e.sequence == 2][0]
        assert overridden.data["response_preview"] == "Short ID override"


# ============================================================================
# ReplayResult
# ============================================================================


class TestReplayResult:
    def test_match_rate_all_match(self) -> None:
        r = ReplayResult(
            recording_id="r1",
            diffs=[
                DiffEntry(diff_type=DiffType.MATCH, event_type="t", sequence=0),
                DiffEntry(diff_type=DiffType.MATCH, event_type="t", sequence=1),
            ],
        )
        assert r.match_rate == 100.0
        assert r.deterministic is True  # All diffs are MATCH

    def test_match_rate_mixed(self) -> None:
        r = ReplayResult(
            recording_id="r1",
            diffs=[
                DiffEntry(diff_type=DiffType.MATCH, event_type="t", sequence=0),
                DiffEntry(diff_type=DiffType.MODIFIED, event_type="t", sequence=1),
            ],
        )
        assert r.match_rate == 50.0
        assert r.mismatch_count == 1

    def test_match_rate_empty(self) -> None:
        r = ReplayResult(recording_id="r1")
        assert r.match_rate == 100.0

    def test_to_dict(self) -> None:
        r = ReplayResult(recording_id="r1", success=True, deterministic=True)
        d = r.to_dict()
        assert d["recording_id"] == "r1"
        assert d["success"] is True


# ============================================================================
# ReplayDiff
# ============================================================================


class TestReplayDiff:
    def test_compare(self) -> None:
        rec = ExecutionRecording(recording_id="r1", started_at=100.0, finished_at=105.0)
        rec.events = [
            ExecutionEvent(
                event_id="e1",
                event_type=EventType.TOOL_CALL,
                sequence=0,
                data={"tool": "web_search"},
            ),
        ]
        result = ReplayResult(
            recording_id="r1",
            events=[
                ExecutionEvent(
                    event_id="re1",
                    event_type=EventType.TOOL_CALL,
                    sequence=0,
                    data={"tool": "web_search"},
                ),
            ],
            success=True,
            deterministic=True,
        )
        comparison = ReplayDiff.compare(rec, result)
        assert comparison["deterministic"] is True
        assert comparison["tool_calls"]["same_sequence"] is True

    def test_summary(self) -> None:
        diffs = [
            DiffEntry(diff_type=DiffType.MATCH, event_type="tool_call", sequence=0),
            DiffEntry(diff_type=DiffType.MODIFIED, event_type="llm_response", sequence=1),
            DiffEntry(diff_type=DiffType.ADDED, event_type="tool_call", sequence=2),
        ]
        s = ReplayDiff.summary(diffs)
        assert s["total_diffs"] == 3
        assert s["matches"] == 1
        assert s["mismatches"] == 2

    def test_diff_entry_to_dict(self) -> None:
        d = DiffEntry(
            diff_type=DiffType.MODIFIED,
            event_type="llm_response",
            sequence=1,
            field_name="response",
            description="Response changed",
        )
        data = d.to_dict()
        assert data["diff_type"] == "modified"
        assert data["description"] == "Response changed"


# ============================================================================
# Helper Functions
# ============================================================================


class TestDiffDicts:
    def test_same_dicts(self) -> None:
        assert _diff_dicts({"a": 1, "b": 2}, {"a": 1, "b": 2}) == []

    def test_different_values(self) -> None:
        result = _diff_dicts({"a": 1, "b": 2}, {"a": 1, "b": 3})
        assert result == ["b"]

    def test_missing_keys(self) -> None:
        result = _diff_dicts({"a": 1}, {"a": 1, "b": 2})
        assert result == ["b"]

    def test_empty_dicts(self) -> None:
        assert _diff_dicts({}, {}) == []


# ============================================================================
# Enum coverage
# ============================================================================


class TestEnums:
    def test_event_types(self) -> None:
        assert len(EventType) == 13
        assert EventType.LLM_REQUEST == "llm_request"
        assert EventType.TOOL_CALL == "tool_call"

    def test_diff_types(self) -> None:
        assert len(DiffType) == 5
        assert DiffType.MATCH == "match"
        assert DiffType.MODIFIED == "modified"
