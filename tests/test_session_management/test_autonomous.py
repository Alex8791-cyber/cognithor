"""Tests for autonomous task orchestrator."""
from __future__ import annotations

from jarvis.core.autonomous_orchestrator import AutonomousOrchestrator, AutonomousTask


def test_detect_recurring_daily():
    orch = AutonomousOrchestrator()
    assert orch.detect_recurring("Schicke mir taeglich einen Wetterbericht") == "daily"
    assert orch.detect_recurring("Send me a daily stock report") == "daily"


def test_detect_recurring_weekly():
    orch = AutonomousOrchestrator()
    assert orch.detect_recurring("Erstelle woechentlich einen Bericht") == "weekly"
    assert orch.detect_recurring("Send me a weekly summary") == "weekly"


def test_detect_recurring_hourly():
    orch = AutonomousOrchestrator()
    assert orch.detect_recurring("Monitor the prices hourly") == "hourly"
    assert orch.detect_recurring("Ueberwache die Seite stuendlich") == "hourly"


def test_detect_recurring_none():
    orch = AutonomousOrchestrator()
    assert orch.detect_recurring("Was ist die Hauptstadt von Frankreich?") == "none"


def test_detect_complexity_simple():
    orch = AutonomousOrchestrator()
    assert orch.detect_complexity("Wie wird das Wetter?") == "simple"


def test_detect_complexity_moderate():
    orch = AutonomousOrchestrator()
    # Single complex signal -> moderate
    assert orch.detect_complexity("Monitor the GPU prices on the web") == "moderate"


def test_detect_complexity_complex():
    orch = AutonomousOrchestrator()
    msg = "Recherchiere die aktuellen GPU-Preise und erstelle einen Vergleichsbericht mit Diagramm und ueberwache die Preise"
    assert orch.detect_complexity(msg) in ("complex", "moderate")


def test_detect_complexity_long_message():
    orch = AutonomousOrchestrator()
    # Messages longer than 150 chars are at least moderate
    msg = "a " * 80  # 160 chars
    assert orch.detect_complexity(msg) == "moderate"


def test_should_orchestrate_complex():
    orch = AutonomousOrchestrator()
    assert orch.should_orchestrate("Monitor Facebook Marketplace for cheap 5090s daily") is True


def test_should_orchestrate_simple():
    orch = AutonomousOrchestrator()
    assert orch.should_orchestrate("Hi") is False


def test_should_orchestrate_recurring_only():
    orch = AutonomousOrchestrator()
    # Short message but has recurring keyword
    assert orch.should_orchestrate("Daily report") is True


def test_create_task():
    orch = AutonomousOrchestrator()
    task = orch.create_task("Daily stock report", "session123")
    assert task.task_id.startswith("auto_")
    assert task.recurring == "daily"
    assert task.status == "pending"
    assert task.description == "Daily stock report"


def test_create_task_stored():
    orch = AutonomousOrchestrator()
    task = orch.create_task("Test task", "session456")
    assert task.task_id in orch._active_tasks


def test_orchestration_prompt_basic():
    orch = AutonomousOrchestrator()
    task = AutonomousTask(task_id="test1", description="Complex task")
    prompt = orch.get_orchestration_prompt(task)
    assert "Autonome Ausfuehrung" in prompt
    assert "Zerlege" in prompt
    assert "Pruefe" in prompt


def test_orchestration_prompt_includes_recurring():
    orch = AutonomousOrchestrator()
    task = AutonomousTask(
        task_id="test1", description="Daily report", recurring="daily"
    )
    prompt = orch.get_orchestration_prompt(task)
    assert "taeglich" in prompt
    assert "set_reminder" in prompt


def test_orchestration_prompt_recurring_weekly():
    orch = AutonomousOrchestrator()
    task = AutonomousTask(
        task_id="test1", description="Weekly report", recurring="weekly"
    )
    prompt = orch.get_orchestration_prompt(task)
    assert "woechentlich" in prompt


def test_orchestration_prompt_retry():
    orch = AutonomousOrchestrator()
    task = AutonomousTask(
        task_id="test2",
        description="Complex task",
        current_attempt=1,
        max_attempts=3,
    )
    prompt = orch.get_orchestration_prompt(task)
    assert "Versuch 2" in prompt
    assert "anderen Ansatz" in prompt


def test_evaluate_result_good():
    orch = AutonomousOrchestrator()
    task = AutonomousTask(task_id="t1", description="test")

    class FakeResult:
        success = True
        is_error = False
        tool_name = "search_and_read"

    score = orch.evaluate_result(task, "A long detailed response " * 20, [FakeResult()])
    assert score >= 0.7


def test_evaluate_result_poor():
    orch = AutonomousOrchestrator()
    task = AutonomousTask(task_id="t2", description="test")
    score = orch.evaluate_result(task, "short", [])
    assert score < 0.7


def test_evaluate_result_with_errors():
    orch = AutonomousOrchestrator()
    task = AutonomousTask(task_id="t3", description="test")

    class ErrorResult:
        success = False
        is_error = True
        tool_name = "web_search"

    score = orch.evaluate_result(task, "Some response text here", [ErrorResult()])
    assert score < 0.7


def test_evaluate_result_clamped():
    orch = AutonomousOrchestrator()
    task = AutonomousTask(task_id="t4", description="test")
    # Empty response, no tools -> should not go below 0.0
    score = orch.evaluate_result(task, "", [])
    assert score >= 0.0
    assert score <= 1.0


def test_get_active_tasks():
    orch = AutonomousOrchestrator()
    orch.create_task("Task 1", "s1")
    orch.create_task("Task 2 daily", "s2")
    tasks = orch.get_active_tasks()
    assert len(tasks) == 2
    assert all("task_id" in t for t in tasks)
    assert all("status" in t for t in tasks)


def test_get_active_tasks_empty():
    orch = AutonomousOrchestrator()
    assert orch.get_active_tasks() == []


def test_quality_threshold():
    assert AutonomousOrchestrator.QUALITY_THRESHOLD == 0.7
