"""Tests für memory/working.py · Tier 5 Session-Kontext."""

from __future__ import annotations

from jarvis.config import MemoryConfig
from jarvis.memory.working import STATIC_BUDGET, WorkingMemoryManager
from jarvis.models import Message, ToolResult


def _msg(role: str = "user", content: str = "x" * 400) -> Message:
    """Erstellt eine Test-Nachricht (~100 Tokens)."""
    return Message(role=role, content=content)


class TestWorkingMemoryManager:
    def test_init(self):
        wm = WorkingMemoryManager()
        assert wm.max_tokens == 32768
        assert wm.current_chat_tokens == 0
        assert wm.usage_ratio == 0.0

    def test_new_session(self):
        wm = WorkingMemoryManager()
        wm.add_message(_msg())
        wm_obj = wm.new_session("test-session")
        assert wm_obj.session_id == "test-session"
        assert len(wm.memory.chat_history) == 0

    def test_add_message(self):
        wm = WorkingMemoryManager()
        wm.add_message(_msg("user", "Hello"))
        assert len(wm.memory.chat_history) == 1
        assert wm.current_chat_tokens > 0

    def test_add_tool_result(self):
        wm = WorkingMemoryManager()
        result = ToolResult(tool_name="test", output="ok", success=True)
        wm.add_tool_result(result)
        assert len(wm.memory.tool_results) == 1

    def test_set_core_memory(self):
        wm = WorkingMemoryManager()
        wm.set_core_memory("# Test\nHello")
        assert wm.memory.core_memory_text == "# Test\nHello"

    def test_available_chat_tokens(self):
        wm = WorkingMemoryManager(max_tokens=32768)
        avail = wm.available_chat_tokens
        assert avail == 32768 - STATIC_BUDGET
        assert avail > 0

    def test_usage_ratio_grows(self):
        wm = WorkingMemoryManager(max_tokens=10000)
        r0 = wm.usage_ratio
        for _ in range(20):
            wm.add_message(_msg())
        r1 = wm.usage_ratio
        assert r1 > r0

    def test_needs_compaction(self):
        config = MemoryConfig(compaction_threshold=0.8, compaction_keep_last_n=4)
        wm = WorkingMemoryManager(config=config, max_tokens=10000)
        assert not wm.needs_compaction

        # Fill up to trigger compaction
        for _ in range(100):
            wm.add_message(_msg(content="x" * 800))  # ~200 tokens each

        assert wm.needs_compaction

    def test_compact_removes_old(self):
        config = MemoryConfig(compaction_keep_last_n=4)
        wm = WorkingMemoryManager(config=config, max_tokens=50000)

        for i in range(10):
            wm.add_message(_msg(content=f"Nachricht {i}"))

        result = wm.compact()
        assert result.messages_removed == 6  # 10 - 4
        assert result.tokens_freed > 0
        assert len(wm.memory.chat_history) == 4

    def test_compact_nothing_to_remove(self):
        config = MemoryConfig(compaction_keep_last_n=10)
        wm = WorkingMemoryManager(config=config)

        for _i in range(3):
            wm.add_message(_msg())

        result = wm.compact()
        assert result.messages_removed == 0
        assert len(wm.memory.chat_history) == 3

    def test_get_removable_messages(self):
        config = MemoryConfig(compaction_keep_last_n=3)
        wm = WorkingMemoryManager(config=config)

        for i in range(7):
            wm.add_message(_msg(content=f"Msg {i}"))

        removable = wm.get_removable_messages()
        assert len(removable) == 4  # 7 - 3
        assert "Msg 0" in removable[0].content

    def test_inject_memories(self):
        wm = WorkingMemoryManager()
        wm.inject_memories([])  # Empty is fine
        assert wm.memory.injected_memories == []

    def test_inject_procedures_max_two(self):
        wm = WorkingMemoryManager()
        wm.inject_procedures(["Proc A", "Proc B", "Proc C"])
        assert len(wm.memory.injected_procedures) == 2  # Max 2

    def test_build_context_parts(self):
        wm = WorkingMemoryManager()
        wm.set_core_memory("# Core\nI am Jarvis")
        wm.inject_procedures(["Step 1: Do thing"])

        parts = wm.build_context_parts()
        assert "core_memory" in parts
        assert "Jarvis" in parts["core_memory"]
        assert "procedures" in parts

    def test_build_context_empty(self):
        wm = WorkingMemoryManager()
        parts = wm.build_context_parts()
        assert parts == {}

    def test_budget_report(self):
        wm = WorkingMemoryManager()
        wm.add_message(_msg())
        report = wm.build_budget_report()
        assert "Token-Budget" in report
        assert "Chat verfügbar" in report
        assert "Compaction nötig" in report

    def test_set_plan(self):
        from jarvis.models import ActionPlan, PlannedAction

        wm = WorkingMemoryManager()
        plan = ActionPlan(
            goal="Test",
            steps=[PlannedAction(tool="test", rationale="Step 1")],
        )
        wm.set_plan(plan)
        assert wm.memory.active_plan is not None
        assert wm.memory.active_plan.goal == "Test"

        wm.set_plan(None)
        assert wm.memory.active_plan is None
