"""Tests für Runtime Monitor.

Testet:
  - PolicyRule Matching (Glob-Patterns, Parameter-Checks)
  - RateLimiter (Token-Bucket, Scopes)
  - RuntimeMonitor (Check-Pipeline, Default-Regeln, Custom-Regeln)
  - Security-Events (Logging, Filtering)
"""

from __future__ import annotations

import pytest

from jarvis.security.monitor import (
    PolicyRule,
    RateLimit,
    RateLimiter,
    RuntimeMonitor,
    SecurityEvent,
    Severity,
    Verdict,
)


# ============================================================================
# PolicyRule
# ============================================================================


class TestPolicyRule:
    def test_basic_creation(self) -> None:
        rule = PolicyRule(
            rule_id="test",
            description="Test-Regel",
            verdict=Verdict.BLOCK,
        )
        assert rule.enabled is True
        assert rule.verdict == Verdict.BLOCK

    def test_disabled_rule(self) -> None:
        rule = PolicyRule(rule_id="disabled", enabled=False)
        assert not rule.enabled


# ============================================================================
# RateLimiter
# ============================================================================


class TestRateLimiter:
    def test_under_limit(self) -> None:
        limiter = RateLimiter()
        limiter.add_limit(RateLimit("test", max_calls=10, window_seconds=60))

        allowed, _ = limiter.check("tool_a")
        assert allowed is True

    def test_over_limit(self) -> None:
        limiter = RateLimiter()
        limiter.add_limit(RateLimit("test", max_calls=3, window_seconds=60))

        for _ in range(3):
            limiter.record("tool_a")

        allowed, reason = limiter.check("tool_a")
        assert allowed is False
        assert "überschritten" in reason

    def test_per_tool_scope(self) -> None:
        limiter = RateLimiter()
        limiter.add_limit(RateLimit("per_tool", max_calls=2, window_seconds=60, scope="per_tool"))

        limiter.record("tool_a")
        limiter.record("tool_a")

        # tool_a überschritten
        allowed_a, _ = limiter.check("tool_a")
        assert allowed_a is False

        # tool_b noch frei
        allowed_b, _ = limiter.check("tool_b")
        assert allowed_b is True

    def test_per_agent_scope(self) -> None:
        limiter = RateLimiter()
        limiter.add_limit(
            RateLimit("per_agent", max_calls=2, window_seconds=60, scope="per_agent"),
        )

        limiter.record("tool", "agent_1")
        limiter.record("tool", "agent_1")

        allowed_1, _ = limiter.check("tool", "agent_1")
        assert allowed_1 is False

        allowed_2, _ = limiter.check("tool", "agent_2")
        assert allowed_2 is True


# ============================================================================
# RuntimeMonitor
# ============================================================================


class TestRuntimeMonitor:
    @pytest.fixture
    def monitor(self) -> RuntimeMonitor:
        return RuntimeMonitor(enable_defaults=True)

    def test_safe_call_allowed(self, monitor: RuntimeMonitor) -> None:
        event = monitor.check_tool_call(
            "memory_search",
            {"query": "BU-Tarif"},
        )
        assert event.verdict == Verdict.ALLOW
        assert not event.is_blocked

    def test_system_dir_blocked(self, monitor: RuntimeMonitor) -> None:
        event = monitor.check_tool_call(
            "file_write",
            {"path": "/etc/passwd"},
        )
        assert event.is_blocked
        assert event.rule_id == "no_system_dirs"
        assert event.severity == Severity.CRITICAL

    def test_proc_dir_blocked(self, monitor: RuntimeMonitor) -> None:
        event = monitor.check_tool_call(
            "file_read",
            {"file": "/proc/1/maps"},
        )
        assert event.is_blocked

    def test_credential_warning(self, monitor: RuntimeMonitor) -> None:
        event = monitor.check_tool_call(
            "send_message",
            {"content": "Mein password ist geheim"},
        )
        assert event.verdict == Verdict.WARN
        assert event.rule_id == "no_credential_leak"

    def test_long_param_blocked(self, monitor: RuntimeMonitor) -> None:
        event = monitor.check_tool_call(
            "tool",
            {"data": "x" * 60000},  # > 50KB
        )
        assert event.is_blocked
        assert event.rule_id == "param_length_limit"

    def test_custom_rule(self) -> None:
        monitor = RuntimeMonitor(enable_defaults=False)
        monitor.add_rule(PolicyRule(
            rule_id="no_delete",
            tool_pattern="file_delete",
            verdict=Verdict.BLOCK,
            severity=Severity.CRITICAL,
            forbidden_params={"path": ["/important"]},
        ))

        # Andere Tools OK
        event = monitor.check_tool_call("file_read", {"path": "/important/doc.txt"})
        assert event.verdict == Verdict.ALLOW

        # file_delete mit /important → blockiert
        event = monitor.check_tool_call("file_delete", {"path": "/important/doc.txt"})
        assert event.is_blocked

    def test_glob_pattern_matching(self) -> None:
        monitor = RuntimeMonitor(enable_defaults=False)
        monitor.add_rule(PolicyRule(
            rule_id="no_file_ops",
            tool_pattern="file_*",
            verdict=Verdict.BLOCK,
            severity=Severity.CRITICAL,
            forbidden_params={"path": ["/secret"]},
        ))

        event = monitor.check_tool_call("file_write", {"path": "/secret/data"})
        assert event.is_blocked

        event = monitor.check_tool_call("memory_search", {"path": "/secret/data"})
        assert event.verdict == Verdict.ALLOW

    def test_rate_limit_triggered(self) -> None:
        monitor = RuntimeMonitor(enable_defaults=False)
        monitor.add_rate_limit(
            RateLimit("test", max_calls=3, window_seconds=60),
        )

        for _ in range(3):
            monitor.check_tool_call("tool")

        event = monitor.check_tool_call("tool")
        assert event.verdict == Verdict.THROTTLE

    def test_remove_rule(self, monitor: RuntimeMonitor) -> None:
        assert monitor.remove_rule("no_system_dirs") is True
        assert monitor.remove_rule("nonexistent") is False

        # Jetzt sollte /etc erlaubt sein
        event = monitor.check_tool_call("file_read", {"path": "/etc/test"})
        assert not event.is_blocked

    def test_record_execution(self, monitor: RuntimeMonitor) -> None:
        monitor.record_execution("tool_a", agent_name="coder", success=True, duration_ms=42)
        events = monitor.get_events(category="execution")
        assert len(events) >= 1

    def test_get_blocked_events(self, monitor: RuntimeMonitor) -> None:
        monitor.check_tool_call("file_write", {"path": "/etc/passwd"})
        blocked = monitor.get_blocked_events()
        assert len(blocked) >= 1

    def test_filter_by_severity(self, monitor: RuntimeMonitor) -> None:
        monitor.check_tool_call("safe_tool", {})
        monitor.check_tool_call("file_write", {"path": "/etc/test"})

        critical = monitor.get_events(severity=Severity.CRITICAL)
        assert len(critical) >= 1

    def test_stats(self, monitor: RuntimeMonitor) -> None:
        monitor.check_tool_call("tool", {})
        monitor.check_tool_call("file_write", {"path": "/etc/test"})

        stats = monitor.stats()
        assert stats["total_checks"] == 2
        assert stats["total_blocks"] >= 1
        assert stats["active_rules"] >= 3

    def test_no_defaults(self) -> None:
        monitor = RuntimeMonitor(enable_defaults=False)
        # /etc ist jetzt erlaubt (keine Default-Regeln)
        event = monitor.check_tool_call("file_write", {"path": "/etc/passwd"})
        assert event.verdict == Verdict.ALLOW

    def test_agent_pattern_matching(self) -> None:
        monitor = RuntimeMonitor(enable_defaults=False)
        monitor.add_rule(PolicyRule(
            rule_id="restrict_coder",
            tool_pattern="*",
            agent_pattern="coder",
            verdict=Verdict.BLOCK,
            severity=Severity.CRITICAL,
            forbidden_params={"action": ["delete"]},
        ))

        # coder mit delete → blockiert
        event = monitor.check_tool_call("tool", {"action": "delete"}, agent_name="coder")
        assert event.is_blocked

        # researcher mit delete → OK (anderer Agent)
        event = monitor.check_tool_call("tool", {"action": "delete"}, agent_name="researcher")
        assert not event.is_blocked
