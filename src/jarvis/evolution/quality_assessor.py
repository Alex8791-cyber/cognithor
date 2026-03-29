"""QualityAssessor — coverage check + LLM self-examination for Phase 5C."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Coroutine, Optional

from jarvis.evolution.models import QualityQuestion, SubGoal

__all__ = ["QualityAssessor"]

logger = logging.getLogger(__name__)

# Coverage thresholds per metric — a SubGoal must reach ALL of these
# before the LLM self-exam triggers. These represent meaningful depth
# for a single sub-topic (e.g. "VVG Grundlagen", not the entire plan).
_COVERAGE_THRESHOLDS = {
    "vault_entries": 8,
    "chunks_created": 25,
    "entities_created": 8,
    "sources_fetched": 8,
}


class QualityAssessor:
    """Assess learning quality via coverage metrics and LLM-generated exam."""

    def __init__(
        self,
        mcp_client: Any,
        llm_fn: Callable[[str], Coroutine[Any, Any, str]],
        coverage_threshold: float = 0.7,
        quality_threshold: float = 0.8,
    ) -> None:
        self._mcp = mcp_client
        self._llm = llm_fn
        self._coverage_threshold = coverage_threshold
        self._quality_threshold = quality_threshold

    # ------------------------------------------------------------------
    # Coverage
    # ------------------------------------------------------------------

    def check_coverage(self, subgoal: SubGoal) -> float:
        """Return ratio of coverage checks that pass (0.0 .. 1.0)."""
        checks = [
            subgoal.vault_entries >= _COVERAGE_THRESHOLDS["vault_entries"],
            subgoal.chunks_created >= _COVERAGE_THRESHOLDS["chunks_created"],
            subgoal.entities_created >= _COVERAGE_THRESHOLDS["entities_created"],
            subgoal.sources_fetched >= _COVERAGE_THRESHOLDS["sources_fetched"],
        ]
        return sum(checks) / len(checks)

    # ------------------------------------------------------------------
    # Question generation
    # ------------------------------------------------------------------

    async def generate_questions(self, topic: str, count: int = 5) -> list[QualityQuestion]:
        """Ask the LLM to produce *count* exam questions about *topic*."""
        import re

        prompt = (
            f"Erstelle genau {count} Pruefungsfragen zum Thema: {topic}\n"
            "Antworte NUR mit validem JSON:\n"
            '{"questions": [{"question": "...", "expected_answer": "..."}]}\n'
            "Kein anderer Text."
        )
        try:
            raw = await self._llm(prompt)
            # Extract JSON from response (LLM might wrap in ```json blocks)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                logger.warning("quality_generate_no_json topic=%s", topic[:40])
                return []
            data = json.loads(match.group())
            return [
                QualityQuestion(
                    question=q.get("question", ""),
                    expected_answer=q.get("expected_answer", ""),
                )
                for q in data.get("questions", [])
                if q.get("question")
            ]
        except Exception:
            logger.debug("quality_generate_questions_failed", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Answering (from memory, not web)
    # ------------------------------------------------------------------

    async def answer_question(self, q: QualityQuestion) -> QualityQuestion:
        """Search vault + memory for an answer; fill *actual_answer*.

        Truncates and cleans the raw search output so the grader sees
        actual content, not metadata/scores/source paths.
        """
        parts: list[str] = []

        # Extract key terms from the question for better search precision
        # Strip common question words to get the actual topic
        search_query = q.question
        for noise in (
            "Welche",
            "Welcher",
            "Welches",
            "Was ist",
            "Was sind",
            "Wie wird",
            "Wie werden",
            "Wodurch",
            "Worin",
            "im deutschen",
            "im Versicherungsrecht",
            "grundlegend",
            "wesentlich",
        ):
            search_query = search_query.replace(noise, "")
        search_query = search_query.strip().strip("?").strip()
        # Use both: specific terms AND expected answer keywords
        if q.expected_answer:
            # Add first few words of expected answer as search boost
            answer_hint = " ".join(q.expected_answer.split()[:6])
            search_query = f"{search_query} {answer_hint}"

        vault_result = await self._mcp.call_tool(
            "vault_search", {"query": search_query[:200], "limit": 5}
        )
        if vault_result.content:
            cleaned = self._clean_search_output(vault_result.content)
            if cleaned and "Keine Notizen gefunden" not in cleaned:
                parts.append(cleaned)

        memory_result = await self._mcp.call_tool(
            "search_memory", {"query": search_query[:200], "top_k": 5}
        )
        if memory_result.content:
            cleaned = self._clean_search_output(memory_result.content)
            if cleaned and "Keine Notizen gefunden" not in cleaned:
                parts.append(cleaned)

        combined = "\n".join(parts) if parts else ""
        # Cap at 1500 chars — enough for grading, not a wall of text
        q.actual_answer = combined[:1500] if combined else ""
        return q

    @staticmethod
    def _clean_search_output(raw: str) -> str:
        """Strip metadata lines (Score:, Tier:, Quelle:) from search output."""
        import re

        lines = raw.split("\n")
        cleaned: list[str] = []
        for line in lines:
            # Skip metadata lines
            if re.match(r"\s*\*\*\[\d+\]\*\*\s+Score:", line):
                continue
            if line.strip().startswith("Score:") or line.strip().startswith("Tier:"):
                continue
            if "· Tier:" in line or "· Quelle:" in line:
                continue
            if line.strip().startswith("###") and "Ergebnis" in line:
                continue
            # Keep actual content
            stripped = line.strip()
            if stripped:
                cleaned.append(stripped)
        return " ".join(cleaned)[:800]

    # ------------------------------------------------------------------
    # Grading
    # ------------------------------------------------------------------

    async def grade_question(self, q: QualityQuestion) -> QualityQuestion:
        """Let the LLM grade *actual_answer* against *expected_answer*."""
        import re

        if not q.actual_answer or q.actual_answer.strip() == "":
            q.score = 0.0
            q.passed = False
            return q
        prompt = (
            "Bewerte ob die gegebene Antwort die erwartete Antwort INHALTLICH abdeckt.\n"
            "Die Antwort muss NICHT woertlich identisch sein — sie stammt aus einer Wissensdatenbank.\n"
            "Wenn die Kernaussage der erwarteten Antwort in der gegebenen Antwort enthalten ist, "
            "gilt sie als korrekt.\n\n"
            f"Frage: {q.question}\n"
            f"Erwartete Kernaussage: {q.expected_answer}\n"
            f"Gegebene Antwort: {q.actual_answer[:800]}\n\n"
            "Bewertung:\n"
            "- score 0.8-1.0: Kernaussage ist klar enthalten\n"
            "- score 0.5-0.7: Teilweise enthalten oder umschrieben\n"
            "- score 0.1-0.4: Nur entfernt verwandt\n"
            "- score 0.0: Komplett falsch oder keine relevante Information\n\n"
            'Antworte NUR mit JSON: {"score": 0.0-1.0, "correct": true/false}\n'
        )
        try:
            raw = await self._llm(prompt)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
                q.score = float(data.get("score", 0.0))
                q.passed = bool(data.get("correct", False))
            else:
                q.score = 0.0
                q.passed = False
        except Exception:
            logger.debug("quality_grade_failed", exc_info=True)
            q.score = 0.0
            q.passed = False
        return q

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    async def run_quality_test(self, subgoal: SubGoal, goal_slug: str) -> dict[str, Any]:
        """Run the complete quality-test pipeline and return a result dict."""
        coverage_score = self.check_coverage(subgoal)

        if coverage_score < self._coverage_threshold:
            logger.info(
                "Coverage too low (%.2f < %.2f) — skipping quality test for %s",
                coverage_score,
                self._coverage_threshold,
                goal_slug,
            )
            return {
                "coverage_score": coverage_score,
                "quality_score": 0,
                "passed": False,
                "questions": [],
                "failed_questions": [],
            }

        questions = await self.generate_questions(subgoal.title)

        for q in questions:
            await self.answer_question(q)
            await self.grade_question(q)

        # Only score questions where we actually found content to grade
        answered = [q for q in questions if q.actual_answer and q.actual_answer.strip()]
        scores = [q.score for q in answered if q.score is not None]
        quality_score = sum(scores) / len(scores) if scores else 0.0
        failed = [q for q in answered if not q.passed]

        # Pass if quality threshold met on answered questions
        # At least 2 questions must be answerable for a valid test
        valid_test = len(answered) >= 2
        passed = valid_test and quality_score >= self._quality_threshold

        logger.info(
            "quality_test_result total=%d answered=%d score=%.2f passed=%s",
            len(questions),
            len(answered),
            quality_score,
            passed,
        )

        return {
            "coverage_score": coverage_score,
            "quality_score": quality_score,
            "passed": passed,
            "questions": questions,
            "failed_questions": failed,
        }
