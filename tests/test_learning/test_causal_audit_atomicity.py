"""Atomicity + audit-emission tests for ``CausalAnalyzer.record_sequence``.

Operational-Trust PR-A/3:
  - Empty ``tool_sequence`` emits ``causal_skipped_empty_sequence``
    BEFORE the early return (audit gap closed).
  - Successful record emits ``causal_sequence_recorded`` AND inserts
    the row -- both, atomic via ``with conn:``.
  - If ``audit_emit_callback`` raises, the DB INSERT is rolled back
    (row count unchanged) -- atomic-or-nothing.
  - ``tool_sequence`` is serialised canonically (sort_keys=True,
    default separators, ensure_ascii=False — same convention as
    ``AuditLogger._last_hash_for_file``); identical inputs produce
    bit-identical rows.
"""

from __future__ import annotations

import json

import pytest

from cognithor.learning.causal import CausalAnalyzer


class TestSkippedEmptySequenceAuditEvent:
    """Empty-sequence path -- the silent-skip gap is closed."""

    def test_empty_sequence_emits_skip_event(self) -> None:
        events: list[tuple[str, dict]] = []

        analyzer = CausalAnalyzer(
            audit_emit_callback=lambda event_type, payload: events.append((event_type, payload)),
        )
        try:
            analyzer.record_sequence(
                session_id="s1",
                tool_sequence=[],
                success_score=0.5,
                model_used="qwen3:8b",
            )
            assert len(events) == 1
            event_type, payload = events[0]
            assert event_type == "causal_skipped_empty_sequence"
            assert payload["session_id"] == "s1"
            assert payload["success_score"] == 0.5
            assert payload["model_used"] == "qwen3:8b"
            # No row written.
            assert analyzer.get_total_sequences() == 0
        finally:
            analyzer.close()

    def test_empty_sequence_no_callback_still_skips(self) -> None:
        analyzer = CausalAnalyzer()  # no callback
        try:
            analyzer.record_sequence("s1", [], 0.5)
            assert analyzer.get_total_sequences() == 0
        finally:
            analyzer.close()


class TestSuccessfulRecordAtomicity:
    """Success path -- INSERT + audit emit are atomic."""

    def test_record_emits_event_and_inserts_row(self) -> None:
        events: list[tuple[str, dict]] = []

        analyzer = CausalAnalyzer(
            audit_emit_callback=lambda event_type, payload: events.append((event_type, payload)),
        )
        try:
            analyzer.record_sequence(
                session_id="s1",
                tool_sequence=["read_file", "write_file"],
                success_score=0.9,
                model_used="qwen3:8b",
            )
            assert analyzer.get_total_sequences() == 1
            assert len(events) == 1
            event_type, payload = events[0]
            assert event_type == "causal_sequence_recorded"
            assert payload["session_id"] == "s1"
            assert payload["tool_sequence"] == ["read_file", "write_file"]
            assert payload["success_score"] == 0.9
            assert payload["model_used"] == "qwen3:8b"
            assert "timestamp" in payload
        finally:
            analyzer.close()

    def test_audit_callback_raises_rolls_back_insert(self) -> None:
        """If the audit emit raises, the DB INSERT is rolled back."""

        def failing_callback(_event_type: str, _payload: dict) -> None:
            raise RuntimeError("audit backend down")

        analyzer = CausalAnalyzer(audit_emit_callback=failing_callback)
        try:
            with pytest.raises(RuntimeError, match="audit backend down"):
                analyzer.record_sequence(
                    session_id="s1",
                    tool_sequence=["a", "b"],
                    success_score=0.5,
                )
            # Row count unchanged: rollback worked.
            assert analyzer.get_total_sequences() == 0
        finally:
            analyzer.close()

    def test_no_callback_path_still_inserts(self) -> None:
        """Backwards-compat: callback=None keeps legacy behaviour."""
        analyzer = CausalAnalyzer()  # no callback
        try:
            analyzer.record_sequence("s1", ["a", "b"], 0.5)
            assert analyzer.get_total_sequences() == 1
        finally:
            analyzer.close()


class TestCanonicalRowContent:
    """``tool_sequence`` is serialised canonically."""

    def test_identical_input_produces_identical_row(self) -> None:
        analyzer_a = CausalAnalyzer()
        analyzer_b = CausalAnalyzer()
        try:
            analyzer_a.record_sequence("s1", ["read", "write"], 0.5)
            analyzer_b.record_sequence("s1", ["read", "write"], 0.5)
            row_a = (
                analyzer_a._get_conn()
                .execute("SELECT tool_sequence FROM causal_sequences")
                .fetchone()
            )
            row_b = (
                analyzer_b._get_conn()
                .execute("SELECT tool_sequence FROM causal_sequences")
                .fetchone()
            )
            assert row_a["tool_sequence"] == row_b["tool_sequence"]
            # Canonical form matching _last_hash_for_file convention:
            # default Python separators (with whitespace), sort_keys.
            assert row_a["tool_sequence"] == '["read", "write"]'
        finally:
            analyzer_a.close()
            analyzer_b.close()

    def test_unicode_preserved_via_ensure_ascii_false(self) -> None:
        analyzer = CausalAnalyzer()
        try:
            analyzer.record_sequence("s1", ["café_tool"], 0.5)
            row = (
                analyzer._get_conn()
                .execute("SELECT tool_sequence FROM causal_sequences")
                .fetchone()
            )
            # Unicode preserved as UTF-8, not \uXXXX-escaped.
            assert "café_tool" in row["tool_sequence"]
            # Round-trips correctly.
            assert json.loads(row["tool_sequence"]) == ["café_tool"]
        finally:
            analyzer.close()


class TestSetAuditEmitCallback:
    """Late-binding via the setter (gateway boot order)."""

    def test_setter_late_binds_callback(self) -> None:
        events: list[tuple[str, dict]] = []
        analyzer = CausalAnalyzer()
        try:
            analyzer.set_audit_emit_callback(
                lambda event_type, payload: events.append((event_type, payload))
            )
            analyzer.record_sequence("s1", ["a"], 0.5)
            assert len(events) == 1
            assert events[0][0] == "causal_sequence_recorded"
        finally:
            analyzer.close()

    def test_setter_can_clear_callback(self) -> None:
        events: list[tuple[str, dict]] = []
        analyzer = CausalAnalyzer(
            audit_emit_callback=lambda event_type, payload: events.append((event_type, payload)),
        )
        try:
            analyzer.set_audit_emit_callback(None)
            analyzer.record_sequence("s1", ["a"], 0.5)
            assert events == []
        finally:
            analyzer.close()
