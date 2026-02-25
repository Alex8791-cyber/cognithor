"""Tests fuer TaskTelemetryCollector."""

import pytest
from jarvis.telemetry.task_telemetry import TaskTelemetryCollector


class TestTaskTelemetryCollector:
    def setup_method(self):
        self.collector = TaskTelemetryCollector()  # in-memory

    def teardown_method(self):
        self.collector.close()

    def test_record_and_count(self):
        self.collector.record_task("s1", True, 100.0, ["read_file"])
        self.collector.record_task("s2", False, 200.0, ["exec_command"], "TimeoutError", "timeout")
        assert self.collector.get_total_tasks() == 2

    def test_success_rate(self):
        self.collector.record_task("s1", True, 100.0)
        self.collector.record_task("s2", True, 100.0)
        self.collector.record_task("s3", False, 100.0)
        rate = self.collector.get_success_rate(window_hours=24)
        assert abs(rate - 2.0/3.0) < 0.01

    def test_success_rate_empty(self):
        rate = self.collector.get_success_rate()
        assert rate == 0.0

    def test_tool_latency_profile(self):
        self.collector.record_task("s1", True, 100.0, ["read_file"])
        self.collector.record_task("s2", True, 200.0, ["read_file"])
        profile = self.collector.get_tool_latency_profile()
        assert "read_file" in profile
        assert profile["read_file"]["avg"] == 150.0

    def test_hourly_stats(self):
        self.collector.record_task("s1", True, 100.0)
        stats = self.collector.get_hourly_stats(hours=1)
        assert len(stats) >= 1
        assert stats[0]["total"] >= 1

    def test_record_with_error(self):
        self.collector.record_task(
            "s1", False, 50.0, ["exec_command"],
            error_type="TimeoutError",
            error_message="Command timed out"
        )
        assert self.collector.get_total_tasks() == 1
        rate = self.collector.get_success_rate()
        assert rate == 0.0
