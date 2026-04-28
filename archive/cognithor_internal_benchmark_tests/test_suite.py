"""Tests for Agent Benchmark Suite.

Covers: BenchmarkTask, BenchmarkResult, BenchmarkScorer, BenchmarkRunner,
BenchmarkReport, RegressionDetector, and built-in tasks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cognithor.benchmark.suite import (
    BUILTIN_TASKS,
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkScorer,
    BenchmarkTask,
    RegressionDetector,
    RegressionEntry,
    ResultStatus,
    TaskCategory,
    TaskDifficulty,
)

if TYPE_CHECKING:
    from pathlib import Path

# ============================================================================
# BenchmarkTask
# ============================================================================


class TestBenchmarkTask:
    def test_defaults(self) -> None:
        t = BenchmarkTask(task_id="t1", name="Test", description="", category=TaskCategory.RESEARCH)
        assert t.difficulty == TaskDifficulty.MEDIUM
        assert t.max_iterations == 5
        assert t.timeout_seconds == 120

    def test_round_trip(self) -> None:
        t = BenchmarkTask(
            task_id="t1",
            name="Fact Check",
            description="Check a fact",
            category=TaskCategory.RESEARCH,
            difficulty=TaskDifficulty.HARD,
            expected_tools=["web_search"],
            expected_keywords=["Berlin"],
            tags=["search"],
        )
        d = t.to_dict()
        t2 = BenchmarkTask.from_dict(d)
        assert t2.task_id == "t1"
        assert t2.category == TaskCategory.RESEARCH
        assert t2.difficulty == TaskDifficulty.HARD
        assert t2.expected_tools == ["web_search"]

    def test_to_dict_keys(self) -> None:
        t = BenchmarkTask(task_id="t", name="n", description="d", category=TaskCategory.TOOL_USE)
        d = t.to_dict()
        assert "task_id" in d
        assert "category" in d
        assert d["category"] == "tool_use"


# ============================================================================
# BenchmarkResult
# ============================================================================


class TestBenchmarkResult:
    def test_passed(self) -> None:
        r = BenchmarkResult(task_id="t1", status=ResultStatus.PASSED, score=0.9)
        assert r.passed is True
        assert r.partial is False

    def test_partial(self) -> None:
        r = BenchmarkResult(task_id="t1", status=ResultStatus.PARTIAL, score=0.5)
        assert r.passed is False
        assert r.partial is True

    def test_failed(self) -> None:
        r = BenchmarkResult(task_id="t1", status=ResultStatus.FAILED, score=0.1)
        assert r.passed is False

    def test_round_trip(self) -> None:
        r = BenchmarkResult(
            task_id="t1",
            status=ResultStatus.PASSED,
            score=0.85,
            tools_used=["web_search"],
            iterations=2,
            tokens_used=500,
            duration_ms=1500.0,
        )
        d = r.to_dict()
        r2 = BenchmarkResult.from_dict(d)
        assert r2.task_id == "t1"
        assert r2.status == ResultStatus.PASSED
        assert r2.score == 0.85


# ============================================================================
# Built-in Tasks
# ============================================================================


class TestBuiltinTasks:
    def test_task_count(self) -> None:
        assert len(BUILTIN_TASKS) >= 14

    def test_all_categories_covered(self) -> None:
        cats = {t.category for t in BUILTIN_TASKS}
        assert TaskCategory.RESEARCH in cats
        assert TaskCategory.AUTOMATION in cats
        assert TaskCategory.KNOWLEDGE in cats
        assert TaskCategory.POLICY in cats
        assert TaskCategory.REASONING in cats
        assert TaskCategory.TOOL_USE in cats

    def test_unique_task_ids(self) -> None:
        ids = [t.task_id for t in BUILTIN_TASKS]
        assert len(ids) == len(set(ids))

    def test_all_have_user_messages(self) -> None:
        for task in BUILTIN_TASKS:
            assert task.user_message, f"Task {task.task_id} missing user_message"

    def test_difficulty_distribution(self) -> None:
        diffs = {t.difficulty for t in BUILTIN_TASKS}
        assert TaskDifficulty.EASY in diffs
        assert TaskDifficulty.MEDIUM in diffs
        assert TaskDifficulty.HARD in diffs


# ============================================================================
# BenchmarkScorer
# ============================================================================


class TestBenchmarkScorer:
    def _task(self, **kw: Any) -> BenchmarkTask:
        defaults = {
            "task_id": "t",
            "name": "Test",
            "description": "",
            "category": TaskCategory.RESEARCH,
        }
        return BenchmarkTask(**(defaults | kw))

    def test_perfect_score(self) -> None:
        task = self._task(expected_keywords=["Berlin"], expected_tools=["web_search"])
        result = BenchmarkResult(
            task_id="t",
            status=ResultStatus.PASSED,
            response="Die Antwort ist Berlin",
            tools_used=["web_search"],
            iterations=1,
            duration_ms=1000,
        )
        score = BenchmarkScorer.score_result(task, result)
        assert score > 0.8  # High score

    def test_zero_score_on_error(self) -> None:
        task = self._task()
        result = BenchmarkResult(task_id="t", status=ResultStatus.ERROR)
        assert BenchmarkScorer.score_result(task, result) == 0.0

    def test_zero_score_on_timeout(self) -> None:
        task = self._task()
        result = BenchmarkResult(task_id="t", status=ResultStatus.TIMEOUT)
        assert BenchmarkScorer.score_result(task, result) == 0.0

    def test_partial_keyword_match(self) -> None:
        task = self._task(expected_keywords=["Berlin", "1961", "1989"])
        result = BenchmarkResult(
            task_id="t",
            status=ResultStatus.PARTIAL,
            response="Berlin wurde 1961 geteilt",
            tools_used=[],
        )
        score = BenchmarkScorer.score_result(task, result)
        assert 0.0 < score < 1.0

    def test_no_keywords_full_marks(self) -> None:
        task = self._task(expected_keywords=[])
        result = BenchmarkResult(task_id="t", status=ResultStatus.PASSED)
        score = BenchmarkScorer.score_result(task, result)
        assert score > 0.5  # Should get full keyword marks

    def test_determine_status_passed(self) -> None:
        task = self._task(expected_keywords=["Berlin"], expected_tools=["web_search"])
        status = BenchmarkScorer.determine_status(
            task, response="Berlin ist die Hauptstadt", tools_used=["web_search"]
        )
        assert status == ResultStatus.PASSED

    def test_determine_status_failed(self) -> None:
        task = self._task(expected_keywords=["Berlin"], expected_tools=["web_search"])
        status = BenchmarkScorer.determine_status(
            task, response="Keine Ahnung", tools_used=["read_file"]
        )
        assert status == ResultStatus.FAILED

    def test_determine_status_partial(self) -> None:
        task = self._task(expected_keywords=["Berlin", "1961"], expected_tools=["web_search"])
        status = BenchmarkScorer.determine_status(
            task, response="Berlin ist schön", tools_used=["web_search"]
        )
        assert status == ResultStatus.PARTIAL


# ============================================================================
# BenchmarkRunner
# ============================================================================


class TestBenchmarkRunner:
    def _mock_executor(self, task: BenchmarkTask) -> BenchmarkResult:
        """Simple mock executor that always passes."""
        return BenchmarkResult(
            task_id=task.task_id,
            status=ResultStatus.PASSED,
            response=" ".join(task.expected_keywords) if task.expected_keywords else "Done",
            tools_used=task.expected_tools,
            iterations=1,
            tokens_used=100,
            duration_ms=50.0,
        )

    def test_run_all(self) -> None:
        runner = BenchmarkRunner(executor=self._mock_executor)
        results = runner.run()
        assert len(results) >= 14
        assert all(r.score > 0 for r in results)

    def test_run_filtered_category(self) -> None:
        runner = BenchmarkRunner(executor=self._mock_executor)
        results = runner.run(categories=[TaskCategory.RESEARCH])
        assert all(
            any(
                t.task_id == r.task_id and t.category == TaskCategory.RESEARCH for t in runner.tasks
            )
            for r in results
        )
        assert len(results) >= 2

    def test_run_filtered_difficulty(self) -> None:
        runner = BenchmarkRunner(executor=self._mock_executor)
        results = runner.run(difficulty=TaskDifficulty.EASY)
        assert len(results) >= 3

    def test_run_filtered_tags(self) -> None:
        runner = BenchmarkRunner(executor=self._mock_executor)
        results = runner.run(tags=["search"])
        assert len(results) >= 1

    def test_run_single(self) -> None:
        runner = BenchmarkRunner(executor=self._mock_executor)
        result = runner.run_single("research-01")
        assert result is not None
        assert result.task_id == "research-01"

    def test_run_single_unknown(self) -> None:
        runner = BenchmarkRunner(executor=self._mock_executor)
        assert runner.run_single("nonexistent") is None

    def test_add_custom_task(self) -> None:
        runner = BenchmarkRunner(tasks=[], executor=self._mock_executor)
        custom = BenchmarkTask(
            task_id="custom-01",
            name="Custom",
            description="Custom task",
            category=TaskCategory.REASONING,
            expected_keywords=["42"],
        )
        runner.add_task(custom)
        assert len(runner.tasks) == 1
        results = runner.run()
        assert len(results) == 1

    def test_no_executor_skips(self) -> None:
        runner = BenchmarkRunner(tasks=[BUILTIN_TASKS[0]])
        results = runner.run()
        assert len(results) == 1
        assert results[0].status == ResultStatus.SKIPPED

    def test_executor_exception_handled(self) -> None:
        def bad_executor(task: BenchmarkTask) -> BenchmarkResult:
            raise RuntimeError("Executor crashed")

        runner = BenchmarkRunner(tasks=[BUILTIN_TASKS[0]], executor=bad_executor)
        results = runner.run()
        assert results[0].status == ResultStatus.ERROR
        assert "crashed" in results[0].error

    def test_summary(self) -> None:
        runner = BenchmarkRunner(executor=self._mock_executor)
        runner.run()
        s = runner.summary()
        assert "run_id" in s
        assert s["total"] >= 14
        assert s["pass_rate"] > 0
        assert "by_category" in s

    def test_summary_empty(self) -> None:
        runner = BenchmarkRunner(tasks=[])
        s = runner.summary()
        assert s["total"] == 0


# ============================================================================
# BenchmarkReport
# ============================================================================


class TestBenchmarkReport:
    def _run_benchmark(self) -> BenchmarkRunner:
        def executor(task: BenchmarkTask) -> BenchmarkResult:
            return BenchmarkResult(
                task_id=task.task_id,
                status=ResultStatus.PASSED,
                response=" ".join(task.expected_keywords),
                tools_used=task.expected_tools,
                tokens_used=100,
            )

        runner = BenchmarkRunner(tasks=BUILTIN_TASKS[:3], executor=executor)
        runner.run()
        return runner

    def test_to_json(self) -> None:
        runner = self._run_benchmark()
        report = BenchmarkReport.to_json(runner, version="1.0.0")
        assert report["version"] == "1.0.0"
        assert "summary" in report
        assert "results" in report
        assert len(report["results"]) == 3

    def test_to_markdown(self) -> None:
        runner = self._run_benchmark()
        md = BenchmarkReport.to_markdown(runner, version="1.0.0")
        assert "# Cognithor Benchmark Report" in md
        assert "Summary" in md
        assert "Task Results" in md
        assert "v1.0.0" in md

    def test_save_json(self, tmp_path: Path) -> None:
        runner = self._run_benchmark()
        path = tmp_path / "report.json"
        BenchmarkReport.save_json(runner, path, version="1.0.0")
        assert path.exists()
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["version"] == "1.0.0"

    def test_save_markdown(self, tmp_path: Path) -> None:
        runner = self._run_benchmark()
        path = tmp_path / "report.md"
        BenchmarkReport.save_markdown(runner, path, version="1.0.0")
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "Benchmark Report" in content


# ============================================================================
# RegressionDetector
# ============================================================================


class TestRegressionDetector:
    def test_no_regressions(self) -> None:
        detector = RegressionDetector()
        old = [
            BenchmarkResult(task_id="t1", status=ResultStatus.PASSED, score=0.8, duration_ms=100)
        ]
        new = [
            BenchmarkResult(task_id="t1", status=ResultStatus.PASSED, score=0.85, duration_ms=95)
        ]
        entries = detector.compare(old, new)
        assert not detector.has_regressions(entries)

    def test_score_regression(self) -> None:
        detector = RegressionDetector(score_threshold=0.1)
        old = [BenchmarkResult(task_id="t1", status=ResultStatus.PASSED, score=0.9)]
        new = [BenchmarkResult(task_id="t1", status=ResultStatus.PASSED, score=0.7)]
        entries = detector.compare(old, new)
        score_entries = [e for e in entries if e.metric == "score"]
        assert len(score_entries) == 1
        assert score_entries[0].direction == "regression"

    def test_score_improvement(self) -> None:
        detector = RegressionDetector(score_threshold=0.1)
        old = [BenchmarkResult(task_id="t1", status=ResultStatus.PASSED, score=0.5)]
        new = [BenchmarkResult(task_id="t1", status=ResultStatus.PASSED, score=0.9)]
        entries = detector.compare(old, new)
        score_entries = [e for e in entries if e.metric == "score"]
        assert len(score_entries) == 1
        assert score_entries[0].direction == "improvement"

    def test_duration_regression(self) -> None:
        detector = RegressionDetector(duration_threshold=0.5)
        old = [BenchmarkResult(task_id="t1", status=ResultStatus.PASSED, duration_ms=100)]
        new = [BenchmarkResult(task_id="t1", status=ResultStatus.PASSED, duration_ms=200)]
        entries = detector.compare(old, new)
        dur_entries = [e for e in entries if e.metric == "duration_ms"]
        assert len(dur_entries) == 1
        assert dur_entries[0].direction == "regression"

    def test_status_regression(self) -> None:
        detector = RegressionDetector()
        old = [BenchmarkResult(task_id="t1", status=ResultStatus.PASSED, score=0.9)]
        new = [BenchmarkResult(task_id="t1", status=ResultStatus.FAILED, score=0.1)]
        entries = detector.compare(old, new)
        status_entries = [e for e in entries if e.metric == "status"]
        assert len(status_entries) == 1
        assert status_entries[0].direction == "regression"

    def test_status_improvement(self) -> None:
        detector = RegressionDetector()
        old = [BenchmarkResult(task_id="t1", status=ResultStatus.FAILED, score=0.1)]
        new = [BenchmarkResult(task_id="t1", status=ResultStatus.PASSED, score=0.9)]
        entries = detector.compare(old, new)
        assert any(e.metric == "status" and e.direction == "improvement" for e in entries)

    def test_skip_mismatched_tasks(self) -> None:
        detector = RegressionDetector()
        old = [BenchmarkResult(task_id="t1", status=ResultStatus.PASSED, score=0.9)]
        new = [BenchmarkResult(task_id="t2", status=ResultStatus.PASSED, score=0.9)]
        entries = detector.compare(old, new)
        assert entries == []

    def test_has_regressions(self) -> None:
        entries = [
            RegressionEntry(task_id="t1", metric="score", direction="improvement"),
            RegressionEntry(task_id="t2", metric="score", direction="regression"),
        ]
        detector = RegressionDetector()
        assert detector.has_regressions(entries) is True

    def test_summary(self) -> None:
        entries = [
            RegressionEntry(task_id="t1", metric="score", direction="regression"),
            RegressionEntry(task_id="t2", metric="score", direction="improvement"),
            RegressionEntry(task_id="t3", metric="duration_ms", direction="regression"),
        ]
        detector = RegressionDetector()
        s = detector.summary(entries)
        assert s["total_changes"] == 3
        assert s["regressions"] == 2
        assert s["improvements"] == 1
        assert "t1" in s["regression_tasks"]

    def test_regression_entry_to_dict(self) -> None:
        e = RegressionEntry(
            task_id="t1", metric="score", old_value=0.9, new_value=0.7, direction="regression"
        )
        d = e.to_dict()
        assert d["direction"] == "regression"


# ============================================================================
# Enum Coverage
# ============================================================================


class TestEnums:
    def test_task_category_values(self) -> None:
        assert len(TaskCategory) == 7

    def test_difficulty_values(self) -> None:
        assert len(TaskDifficulty) == 3

    def test_result_status_values(self) -> None:
        assert len(ResultStatus) == 6
        assert ResultStatus.PASSED == "passed"
        assert ResultStatus.TIMEOUT == "timeout"
