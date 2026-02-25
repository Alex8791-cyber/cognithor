"""Tests fuer CheckpointManager."""

import pytest
from jarvis.core.checkpoint import CheckpointManager
from jarvis.models import KernelState, ToolResult


class TestCheckpointManager:
    def setup_method(self):
        self.mgr = CheckpointManager()

    def test_create_checkpoint(self):
        cp = self.mgr.create_checkpoint(
            session_id="s1",
            kernel_state=KernelState.EXECUTING,
        )
        assert cp.session_id == "s1"
        assert cp.kernel_state == KernelState.EXECUTING

    def test_restore_checkpoint(self):
        cp = self.mgr.create_checkpoint("s1", KernelState.PLANNING)
        restored = self.mgr.restore_checkpoint(cp.id)
        assert restored is not None
        assert restored.id == cp.id

    def test_restore_nonexistent(self):
        assert self.mgr.restore_checkpoint("nonexistent") is None

    def test_list_checkpoints(self):
        self.mgr.create_checkpoint("s1", KernelState.IDLE)
        self.mgr.create_checkpoint("s1", KernelState.PLANNING)
        self.mgr.create_checkpoint("s2", KernelState.EXECUTING)

        s1_cps = self.mgr.list_checkpoints("s1")
        assert len(s1_cps) == 2

        s2_cps = self.mgr.list_checkpoints("s2")
        assert len(s2_cps) == 1

    def test_get_latest(self):
        self.mgr.create_checkpoint("s1", KernelState.IDLE)
        cp2 = self.mgr.create_checkpoint("s1", KernelState.PLANNING)

        latest = self.mgr.get_latest("s1")
        assert latest is not None
        assert latest.id == cp2.id

    def test_get_latest_empty(self):
        assert self.mgr.get_latest("nonexistent") is None

    def test_clear_session(self):
        self.mgr.create_checkpoint("s1", KernelState.IDLE)
        self.mgr.clear_session("s1")
        assert self.mgr.list_checkpoints("s1") == []

    def test_total_checkpoints(self):
        self.mgr.create_checkpoint("s1", KernelState.IDLE)
        self.mgr.create_checkpoint("s2", KernelState.IDLE)
        assert self.mgr.total_checkpoints == 2

    def test_checkpoint_with_data(self):
        cp = self.mgr.create_checkpoint(
            "s1", KernelState.EXECUTING,
            working_memory_snapshot={"key": "value"},
            completed_nodes=["n1", "n2"],
            tool_results=[ToolResult(tool_name="test", content="ok")],
        )
        assert cp.working_memory_snapshot == {"key": "value"}
        assert cp.completed_nodes == ["n1", "n2"]
        assert len(cp.tool_results) == 1
