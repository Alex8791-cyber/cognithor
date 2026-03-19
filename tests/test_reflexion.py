"""Tests for the Reflexion-Based Error Learning system."""

from __future__ import annotations

import json
import time

import pytest

from jarvis.learning.reflexion import ReflexionEntry, ReflexionMemory


class TestRecordAndLookup:
    """Record an error and look it up by signature."""

    def test_record_and_lookup(self, tmp_path: object) -> None:
        mem = ReflexionMemory(data_dir=tmp_path)  # type: ignore[arg-type]
        entry = mem.record_error(
            tool_name="shell_exec",
            error_category="timeout",
            error_message="Command timed out after 30s",
            root_cause="Long-running process",
            prevention_rule="Set timeout to 60s for build commands",
            task_context="User asked to compile project",
        )
        assert entry.tool_name == "shell_exec"
        assert entry.error_category == "timeout"
        assert entry.recurrence_count == 1

        # Look it up
        found = mem.get_solution("shell_exec", "timeout", "Command timed out after 30s")
        assert found is not None
        assert found.entry_id == entry.entry_id
        assert found.prevention_rule == "Set timeout to 60s for build commands"

    def test_lookup_miss(self, tmp_path: object) -> None:
        mem = ReflexionMemory(data_dir=tmp_path)  # type: ignore[arg-type]
        found = mem.get_solution("shell_exec", "timeout", "some random error")
        assert found is None


class TestRecurrenceTracking:
    """Same error twice, count increments."""

    def test_recurrence_increments(self, tmp_path: object) -> None:
        mem = ReflexionMemory(data_dir=tmp_path)  # type: ignore[arg-type]
        e1 = mem.record_error(
            tool_name="web_search",
            error_category="rate_limit",
            error_message="429 Too Many Requests",
            root_cause="Rate limit hit",
            prevention_rule="Add 2s delay between searches",
        )
        assert e1.recurrence_count == 1

        e2 = mem.record_error(
            tool_name="web_search",
            error_category="rate_limit",
            error_message="429 Too Many Requests",
            root_cause="Rate limit hit",
            prevention_rule="Add 2s delay between searches",
        )
        assert e2.recurrence_count == 2
        assert e2.entry_id == e1.entry_id  # Same entry updated

    def test_recurrence_updates_solution(self, tmp_path: object) -> None:
        mem = ReflexionMemory(data_dir=tmp_path)  # type: ignore[arg-type]
        e1 = mem.record_error(
            tool_name="web_search",
            error_category="rate_limit",
            error_message="429 Too Many Requests",
            root_cause="Rate limit hit",
            prevention_rule="Add delay",
        )
        assert e1.solution == ""

        e2 = mem.record_error(
            tool_name="web_search",
            error_category="rate_limit",
            error_message="429 Too Many Requests",
            root_cause="Rate limit hit",
            prevention_rule="Add delay",
            solution="Use exponential backoff",
        )
        assert e2.solution == "Use exponential backoff"


class TestPreventionRules:
    """Get rules filtered by tool."""

    def test_get_all_rules(self, tmp_path: object) -> None:
        mem = ReflexionMemory(data_dir=tmp_path)  # type: ignore[arg-type]
        mem.record_error("shell_exec", "timeout", "err1", "cause", "rule1")
        mem.record_error("web_search", "rate_limit", "err2", "cause", "rule2")
        mem.record_error("file_read", "permission", "err3", "cause", "rule3")

        rules = mem.get_prevention_rules()
        assert len(rules) == 3

    def test_filter_by_tool(self, tmp_path: object) -> None:
        mem = ReflexionMemory(data_dir=tmp_path)  # type: ignore[arg-type]
        mem.record_error("shell_exec", "timeout", "err1", "cause", "rule1")
        mem.record_error("web_search", "rate_limit", "err2", "cause", "rule2")

        rules = mem.get_prevention_rules(tool_name="shell_exec")
        assert len(rules) == 1
        assert rules[0] == "rule1"

    def test_rejected_rules_excluded(self, tmp_path: object) -> None:
        mem = ReflexionMemory(data_dir=tmp_path)  # type: ignore[arg-type]
        entry = mem.record_error("shell_exec", "timeout", "err1", "cause", "rule1")
        mem.reject_rule(entry.error_signature)

        rules = mem.get_prevention_rules()
        assert len(rules) == 0

    def test_empty_prevention_rule_excluded(self, tmp_path: object) -> None:
        mem = ReflexionMemory(data_dir=tmp_path)  # type: ignore[arg-type]
        mem.record_error("shell_exec", "timeout", "err1", "cause", "")

        rules = mem.get_prevention_rules()
        assert len(rules) == 0


class TestAdoptReject:
    """Mark rules as adopted/rejected."""

    def test_adopt_rule(self, tmp_path: object) -> None:
        mem = ReflexionMemory(data_dir=tmp_path)  # type: ignore[arg-type]
        entry = mem.record_error("shell_exec", "timeout", "err1", "cause", "rule1")
        assert entry.status == "pending"

        result = mem.adopt_rule(entry.error_signature)
        assert result is True
        assert entry.status == "adopted"

    def test_reject_rule(self, tmp_path: object) -> None:
        mem = ReflexionMemory(data_dir=tmp_path)  # type: ignore[arg-type]
        entry = mem.record_error("shell_exec", "timeout", "err1", "cause", "rule1")

        result = mem.reject_rule(entry.error_signature)
        assert result is True
        assert entry.status == "rejected"

    def test_adopt_nonexistent(self, tmp_path: object) -> None:
        mem = ReflexionMemory(data_dir=tmp_path)  # type: ignore[arg-type]
        assert mem.adopt_rule("nonexistent") is False

    def test_reject_nonexistent(self, tmp_path: object) -> None:
        mem = ReflexionMemory(data_dir=tmp_path)  # type: ignore[arg-type]
        assert mem.reject_rule("nonexistent") is False


class TestPruneOld:
    """Old entries removed, recurring ones kept."""

    def test_prune_removes_old(self, tmp_path: object) -> None:
        mem = ReflexionMemory(data_dir=tmp_path)  # type: ignore[arg-type]
        entry = mem.record_error("shell_exec", "timeout", "err1", "cause", "rule1")
        # Backdate the entry to 100 days ago
        entry.timestamp = time.time() - 100 * 86400
        mem._rewrite_all()

        pruned = mem.prune_old(older_than_days=90)
        assert pruned == 1
        assert len(mem._all_entries) == 0

    def test_prune_keeps_recurring(self, tmp_path: object) -> None:
        mem = ReflexionMemory(data_dir=tmp_path)  # type: ignore[arg-type]
        entry = mem.record_error("shell_exec", "timeout", "err1", "cause", "rule1")
        entry.recurrence_count = 5
        entry.timestamp = time.time() - 100 * 86400
        mem._rewrite_all()

        pruned = mem.prune_old(older_than_days=90)
        assert pruned == 0
        assert len(mem._all_entries) == 1

    def test_prune_keeps_recent(self, tmp_path: object) -> None:
        mem = ReflexionMemory(data_dir=tmp_path)  # type: ignore[arg-type]
        mem.record_error("shell_exec", "timeout", "err1", "cause", "rule1")

        pruned = mem.prune_old(older_than_days=90)
        assert pruned == 0
        assert len(mem._all_entries) == 1


class TestSignatureNormalization:
    """Paths, UUIDs, timestamps stripped from signatures."""

    def test_paths_normalized(self) -> None:
        sig1 = ReflexionMemory.compute_signature(
            "file_read", "error", "Failed to read /home/user/file.txt"
        )
        sig2 = ReflexionMemory.compute_signature(
            "file_read", "error", "Failed to read /var/log/app.txt"
        )
        assert sig1 == sig2

    def test_uuids_normalized(self) -> None:
        sig1 = ReflexionMemory.compute_signature(
            "tool", "error", "Object 550e8400-e29b-41d4-a716-446655440000 not found"
        )
        sig2 = ReflexionMemory.compute_signature(
            "tool", "error", "Object a1b2c3d4-e5f6-7890-abcd-ef1234567890 not found"
        )
        assert sig1 == sig2

    def test_timestamps_normalized(self) -> None:
        sig1 = ReflexionMemory.compute_signature("tool", "error", "Error at 2025-01-15T10:30:00")
        sig2 = ReflexionMemory.compute_signature("tool", "error", "Error at 2026-03-20T14:00:00")
        assert sig1 == sig2

    def test_hex_addresses_normalized(self) -> None:
        sig1 = ReflexionMemory.compute_signature("tool", "error", "Object at 0xdeadbeef crashed")
        sig2 = ReflexionMemory.compute_signature("tool", "error", "Object at 0x1234abcd crashed")
        assert sig1 == sig2

    def test_different_errors_different_signatures(self) -> None:
        sig1 = ReflexionMemory.compute_signature("tool", "timeout", "timed out")
        sig2 = ReflexionMemory.compute_signature("tool", "permission", "access denied")
        assert sig1 != sig2


class TestStats:
    """Verify stats output."""

    def test_stats_empty(self, tmp_path: object) -> None:
        mem = ReflexionMemory(data_dir=tmp_path)  # type: ignore[arg-type]
        s = mem.stats()
        assert s["total_entries"] == 0
        assert s["unique_patterns"] == 0
        assert s["adopted_rules"] == 0
        assert s["rejected_rules"] == 0
        assert s["pending_rules"] == 0
        assert s["top_recurring"] == []

    def test_stats_populated(self, tmp_path: object) -> None:
        mem = ReflexionMemory(data_dir=tmp_path)  # type: ignore[arg-type]
        mem.record_error("shell_exec", "timeout", "err1", "cause", "rule1")
        e2 = mem.record_error("web_search", "rate_limit", "err2", "cause", "rule2")
        mem.adopt_rule(e2.error_signature)
        e3 = mem.record_error("file_read", "permission", "err3", "cause", "rule3")
        mem.reject_rule(e3.error_signature)

        s = mem.stats()
        assert s["total_entries"] == 3
        assert s["unique_patterns"] == 3
        assert s["adopted_rules"] == 1
        assert s["rejected_rules"] == 1
        assert s["pending_rules"] == 1
        assert len(s["top_recurring"]) == 3


class TestEmptyMemory:
    """get_solution returns None on empty."""

    def test_empty_solution(self, tmp_path: object) -> None:
        mem = ReflexionMemory(data_dir=tmp_path)  # type: ignore[arg-type]
        assert mem.get_solution("any_tool", "any_cat", "any_msg") is None

    def test_empty_prevention_rules(self, tmp_path: object) -> None:
        mem = ReflexionMemory(data_dir=tmp_path)  # type: ignore[arg-type]
        assert mem.get_prevention_rules() == []

    def test_empty_recent_errors(self, tmp_path: object) -> None:
        mem = ReflexionMemory(data_dir=tmp_path)  # type: ignore[arg-type]
        assert mem.get_recent_errors() == []

    def test_empty_recurring_errors(self, tmp_path: object) -> None:
        mem = ReflexionMemory(data_dir=tmp_path)  # type: ignore[arg-type]
        assert mem.get_recurring_errors() == []


class TestPersistence:
    """Save, create new instance, entries still there."""

    def test_persistence_across_instances(self, tmp_path: object) -> None:
        mem1 = ReflexionMemory(data_dir=tmp_path)  # type: ignore[arg-type]
        mem1.record_error(
            tool_name="shell_exec",
            error_category="timeout",
            error_message="Command timed out after 30s",
            root_cause="Long-running process",
            prevention_rule="Set timeout to 60s",
            task_context="compile project",
            solution="increase timeout",
        )
        mem1.record_error(
            tool_name="web_search",
            error_category="rate_limit",
            error_message="429 Too Many Requests",
            root_cause="Rate limit hit",
            prevention_rule="Add 2s delay",
        )

        # Create new instance from same directory
        mem2 = ReflexionMemory(data_dir=tmp_path)  # type: ignore[arg-type]
        assert len(mem2._all_entries) == 2
        assert len(mem2._entries) == 2

        found = mem2.get_solution("shell_exec", "timeout", "Command timed out after 30s")
        assert found is not None
        assert found.prevention_rule == "Set timeout to 60s"
        assert found.solution == "increase timeout"

    def test_persistence_with_recurrence(self, tmp_path: object) -> None:
        mem1 = ReflexionMemory(data_dir=tmp_path)  # type: ignore[arg-type]
        mem1.record_error("tool", "cat", "error msg", "cause", "rule")
        mem1.record_error("tool", "cat", "error msg", "cause", "rule")

        mem2 = ReflexionMemory(data_dir=tmp_path)  # type: ignore[arg-type]
        found = mem2.get_solution("tool", "cat", "error msg")
        assert found is not None
        assert found.recurrence_count == 2

    def test_persistence_with_status_change(self, tmp_path: object) -> None:
        mem1 = ReflexionMemory(data_dir=tmp_path)  # type: ignore[arg-type]
        entry = mem1.record_error("tool", "cat", "err", "cause", "rule")
        mem1.adopt_rule(entry.error_signature)

        mem2 = ReflexionMemory(data_dir=tmp_path)  # type: ignore[arg-type]
        found = mem2.get_solution("tool", "cat", "err")
        assert found is not None
        assert found.status == "adopted"


class TestRecentAndRecurring:
    """Test get_recent_errors and get_recurring_errors."""

    def test_recent_errors_ordered(self, tmp_path: object) -> None:
        mem = ReflexionMemory(data_dir=tmp_path)  # type: ignore[arg-type]
        mem.record_error("t1", "cat", "err1", "c", "r")
        mem.record_error("t2", "cat", "err2", "c", "r")
        mem.record_error("t3", "cat", "err3", "c", "r")

        recent = mem.get_recent_errors(limit=2)
        assert len(recent) == 2
        assert recent[0].tool_name == "t3"  # Most recent first

    def test_recurring_errors_threshold(self, tmp_path: object) -> None:
        mem = ReflexionMemory(data_dir=tmp_path)  # type: ignore[arg-type]
        for _ in range(5):
            mem.record_error("web_search", "rate_limit", "429", "cause", "rule")
        mem.record_error("shell_exec", "timeout", "timed out", "cause", "rule2")

        recurring = mem.get_recurring_errors(min_count=3)
        assert len(recurring) == 1
        assert recurring[0].tool_name == "web_search"
        assert recurring[0].recurrence_count == 5


class TestEdgeCases:
    """Edge cases and robustness."""

    def test_long_error_message_truncated(self, tmp_path: object) -> None:
        mem = ReflexionMemory(data_dir=tmp_path)  # type: ignore[arg-type]
        long_msg = "x" * 1000
        entry = mem.record_error("tool", "cat", long_msg, "cause", "rule")
        assert len(entry.error_message) == 500

    def test_long_task_context_truncated(self, tmp_path: object) -> None:
        mem = ReflexionMemory(data_dir=tmp_path)  # type: ignore[arg-type]
        long_ctx = "y" * 500
        entry = mem.record_error("tool", "cat", "err", "cause", "rule", task_context=long_ctx)
        assert len(entry.task_context) == 300

    def test_corrupted_jsonl_handled(self, tmp_path: object) -> None:
        """Corrupted JSONL file should not crash initialization."""
        jsonl_file = tmp_path / "reflexion.jsonl"  # type: ignore[operator]
        jsonl_file.write_text("not valid json\n", encoding="utf-8")
        mem = ReflexionMemory(data_dir=tmp_path)  # type: ignore[arg-type]
        assert len(mem._all_entries) == 0

    def test_max_entries_concept(self, tmp_path: object) -> None:
        """MAX_ENTRIES constant is defined."""
        assert ReflexionMemory.MAX_ENTRIES == 5000
