"""Tests for QualityAssessor — coverage check + LLM self-examination."""

from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest

from jarvis.evolution.models import QualityQuestion, SubGoal


@dataclass
class _MockToolResult:
    content: str = ""
    is_error: bool = False


_QUESTIONS_JSON = json.dumps(
    {
        "questions": [
            {
                "question": "Wie lang ist die Widerrufsfrist nach VVG?",
                "expected_answer": "14 Tage",
            },
            {
                "question": "Welches Gesetz regelt den Versicherungsvertrag?",
                "expected_answer": "Das Versicherungsvertragsgesetz (VVG)",
            },
        ]
    }
)

_GRADE_PASS_JSON = json.dumps({"score": 0.9, "correct": True})
_GRADE_FAIL_JSON = json.dumps({"score": 0.2, "correct": False})

_VAULT_ANSWER = "VVG \u00a77 regelt die Widerrufsfrist von 14 Tagen."


def _make_mcp() -> AsyncMock:
    mcp = AsyncMock()
    mcp.call_tool = AsyncMock(return_value=_MockToolResult(content=_VAULT_ANSWER))
    return mcp


def _make_llm_fn(responses: list[str]) -> AsyncMock:
    """Return an async callable that yields *responses* in order, cycling."""
    fn = AsyncMock(side_effect=list(responses))
    return fn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCoverageCheck:
    def test_coverage_check_passes(self):
        from jarvis.evolution.quality_assessor import QualityAssessor

        sg = SubGoal(
            title="VVG Grundlagen",
            description="Grundlagen des VVG",
            chunks_created=30,
            entities_created=10,
            vault_entries=10,
            sources_fetched=10,
        )
        assessor = QualityAssessor(mcp_client=AsyncMock(), llm_fn=AsyncMock())
        score = assessor.check_coverage(sg)
        assert score >= 0.9

    def test_coverage_check_fails(self):
        from jarvis.evolution.quality_assessor import QualityAssessor

        sg = SubGoal(
            title="VVG Grundlagen",
            description="Grundlagen des VVG",
            chunks_created=2,
            entities_created=0,
            vault_entries=0,
            sources_fetched=0,
        )
        assessor = QualityAssessor(mcp_client=AsyncMock(), llm_fn=AsyncMock())
        score = assessor.check_coverage(sg)
        assert score < 0.7


class TestGenerateQuestions:
    async def test_generate_questions(self):
        from jarvis.evolution.quality_assessor import QualityAssessor

        llm_fn = _make_llm_fn([_QUESTIONS_JSON])
        assessor = QualityAssessor(mcp_client=AsyncMock(), llm_fn=llm_fn)
        questions = await assessor.generate_questions("VVG Grundlagen", count=2)

        assert len(questions) == 2
        assert all(isinstance(q, QualityQuestion) for q in questions)
        assert questions[0].question == "Wie lang ist die Widerrufsfrist nach VVG?"
        assert questions[0].expected_answer == "14 Tage"
        llm_fn.assert_called_once()


class TestAnswerQuestion:
    async def test_answer_question_uses_memory(self):
        from jarvis.evolution.quality_assessor import QualityAssessor

        mcp = _make_mcp()
        assessor = QualityAssessor(mcp_client=mcp, llm_fn=AsyncMock())
        q = QualityQuestion(
            question="Wie lang ist die Widerrufsfrist?",
            expected_answer="14 Tage",
        )
        result = await assessor.answer_question(q)

        assert result.actual_answer is not None
        assert _VAULT_ANSWER in result.actual_answer
        # Must have called vault_search or search_memory, not web tools
        tool_names = [call.args[0] for call in mcp.call_tool.call_args_list]
        assert any(t in ("vault_search", "search_memory") for t in tool_names), (
            f"Expected vault_search or search_memory calls, got {tool_names}"
        )


class TestGradeQuestion:
    async def test_grade_question(self):
        from jarvis.evolution.quality_assessor import QualityAssessor

        llm_fn = _make_llm_fn([_GRADE_PASS_JSON])
        assessor = QualityAssessor(mcp_client=AsyncMock(), llm_fn=llm_fn)
        q = QualityQuestion(
            question="Wie lang ist die Widerrufsfrist?",
            expected_answer="14 Tage",
            actual_answer="Die Widerrufsfrist betraegt 14 Tage nach VVG.",
        )
        result = await assessor.grade_question(q)

        assert result.score == 0.9
        assert result.passed is True
        llm_fn.assert_called_once()


class TestFullQualityTest:
    async def test_full_quality_test(self):
        from jarvis.evolution.quality_assessor import QualityAssessor

        mcp = _make_mcp()
        # LLM calls: 1x generate_questions, then 2x grade_question
        llm_fn = _make_llm_fn([_QUESTIONS_JSON, _GRADE_PASS_JSON, _GRADE_PASS_JSON])
        assessor = QualityAssessor(
            mcp_client=mcp,
            llm_fn=llm_fn,
            coverage_threshold=0.7,
            quality_threshold=0.5,
        )
        sg = SubGoal(
            title="VVG Grundlagen",
            description="Grundlagen des VVG",
            chunks_created=30,
            entities_created=10,
            vault_entries=10,
            sources_fetched=10,
        )
        result = await assessor.run_quality_test(sg, "vvg-grundlagen")

        assert "coverage_score" in result
        assert "quality_score" in result
        assert "questions" in result
        assert "passed" in result
        assert result["coverage_score"] >= 0.9
        assert result["quality_score"] > 0
        assert len(result["questions"]) == 2

    async def test_quality_test_skips_when_coverage_fails(self):
        from jarvis.evolution.quality_assessor import QualityAssessor

        llm_fn = AsyncMock()
        assessor = QualityAssessor(
            mcp_client=AsyncMock(),
            llm_fn=llm_fn,
            coverage_threshold=0.7,
        )
        sg = SubGoal(
            title="VVG Grundlagen",
            description="Grundlagen des VVG",
            chunks_created=2,
            entities_created=0,
            vault_entries=0,
            sources_fetched=0,
        )
        result = await assessor.run_quality_test(sg, "vvg-grundlagen")

        assert result["quality_score"] == 0
        assert result["passed"] is False
        llm_fn.assert_not_called()
