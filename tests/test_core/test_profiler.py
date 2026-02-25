"""Tests fuer TaskProfiler."""

import pytest
from jarvis.core.profiler import TaskProfiler
from jarvis.models import ToolProfile, TaskProfile, CapabilityProfile


class TestTaskProfiler:
    def setup_method(self):
        self.profiler = TaskProfiler()  # in-memory

    def teardown_method(self):
        self.profiler.close()

    def test_start_and_finish_task(self):
        self.profiler.start_task("s1", "Test task", "testing")
        self.profiler.finish_task("s1", success_score=0.8)
        profile = self.profiler.get_task_profile("testing")
        assert profile.task_count == 1
        assert profile.avg_score == 0.8

    def test_record_tool_call(self):
        self.profiler.record_tool_call("read_file", 50.0, True, session_id="s1")
        self.profiler.record_tool_call("read_file", 100.0, True, session_id="s1")
        self.profiler.record_tool_call("read_file", 200.0, False, "TimeoutError", session_id="s1")

        profile = self.profiler.get_tool_profile("read_file")
        assert profile.call_count == 3
        assert abs(profile.avg_latency_ms - (50+100+200)/3) < 0.01
        assert abs(profile.success_rate - 2/3) < 0.01
        assert profile.error_types.get("TimeoutError") == 1

    def test_tool_profile_empty(self):
        profile = self.profiler.get_tool_profile("nonexistent")
        assert profile.call_count == 0
        assert profile.avg_latency_ms == 0.0

    def test_task_profile_empty(self):
        profile = self.profiler.get_task_profile("nonexistent")
        assert profile.task_count == 0
        assert profile.avg_score == 0.0

    def test_capability_profile(self):
        # Record enough data for profiling
        for i in range(5):
            self.profiler.start_task(f"s{i}", f"task {i}", "general")
            self.profiler.record_tool_call("read_file", 50.0, True, session_id=f"s{i}")
            self.profiler.record_tool_call("write_file", 100.0, True, session_id=f"s{i}")
            self.profiler.finish_task(f"s{i}", success_score=0.9)

        cap = self.profiler.get_capability_profile()
        assert isinstance(cap, CapabilityProfile)
        assert cap.overall_success_rate == 1.0
        assert len(cap.tool_profiles) == 2

    def test_tool_tracking_in_task(self):
        self.profiler.start_task("s1", "test", "general")
        self.profiler.record_tool_call("read_file", 50.0, True, session_id="s1")
        self.profiler.record_tool_call("write_file", 100.0, True, session_id="s1")
        self.profiler.finish_task("s1", 0.9)

        profile = self.profiler.get_task_profile("general")
        assert profile.task_count == 1
        assert "read_file" in profile.common_tools or "write_file" in profile.common_tools

    def test_success_rate_threshold(self):
        for i in range(5):
            self.profiler.start_task(f"s{i}", f"task", "general")
            self.profiler.finish_task(f"s{i}", success_score=0.8)  # >= 0.6 = success

        profile = self.profiler.get_task_profile("general")
        assert profile.success_rate == 1.0

    def test_strengths_weaknesses(self):
        # Strong tool
        for i in range(5):
            self.profiler.record_tool_call("good_tool", 50.0, True, session_id=f"s{i}")
        # Weak tool
        for i in range(5):
            self.profiler.record_tool_call("bad_tool", 50.0, False, "Error", session_id=f"s{i}")

        cap = self.profiler.get_capability_profile()
        assert any("good_tool" in s for s in cap.strengths)
        assert any("bad_tool" in s for s in cap.weaknesses)
