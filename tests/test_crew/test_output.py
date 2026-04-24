import pytest
from pydantic import ValidationError

from cognithor.crew.output import CrewOutput, TaskOutput, TokenUsageDict


class TestTokenUsageDict:
    def test_typed_keys(self):
        usage: TokenUsageDict = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        assert usage["total_tokens"] == 120

    def test_missing_key_raises_at_runtime_on_strict_access(self):
        from cognithor.crew.output import empty_token_usage

        usage = empty_token_usage()
        assert usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


class TestTaskOutput:
    def test_minimal(self):
        out = TaskOutput(task_id="t1", agent_role="writer", raw="hello")
        assert out.task_id == "t1"
        assert out.raw == "hello"
        assert out.duration_ms == 0.0
        assert out.token_usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def test_structured_output(self):
        out = TaskOutput(
            task_id="t1",
            agent_role="analyst",
            raw='{"foo": 1}',
            structured={"foo": 1},
        )
        assert out.structured == {"foo": 1}

    def test_frozen_after_construction(self):
        out = TaskOutput(task_id="t1", agent_role="x", raw="y")
        with pytest.raises(ValidationError):
            out.raw = "mutated"  # type: ignore[misc]


class TestCrewOutput:
    def test_aggregates_tasks(self):
        t1 = TaskOutput(
            task_id="t1",
            agent_role="analyst",
            raw="A",
            token_usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        t2 = TaskOutput(
            task_id="t2",
            agent_role="writer",
            raw="B",
            token_usage={"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
        )
        out = CrewOutput(raw="B", tasks_output=[t1, t2], trace_id="trace-xyz")
        assert out.raw == "B"
        assert len(out.tasks_output) == 2
        assert out.token_usage == {"prompt_tokens": 30, "completion_tokens": 13, "total_tokens": 43}
        assert out.trace_id == "trace-xyz"

    def test_trace_id_required(self):
        with pytest.raises(ValidationError):
            CrewOutput(raw="x", tasks_output=[])  # trace_id omitted

    def test_aggregates_total_tokens_directly_not_recomputed(self):
        # total_tokens must be summed from per-task totals, NOT recomputed
        # from prompt + completion. Providers that report cached/reasoning
        # tokens as a separate bucket can set total_tokens > prompt + completion
        # on an individual TaskOutput; the aggregate must preserve that.
        t = TaskOutput(
            task_id="t1",
            agent_role="analyst",
            raw="X",
            token_usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 42},
        )
        out = CrewOutput(raw="X", tasks_output=[t], trace_id="tid")
        assert out.token_usage["total_tokens"] == 42  # NOT 15
        assert out.token_usage["prompt_tokens"] == 10
        assert out.token_usage["completion_tokens"] == 5
