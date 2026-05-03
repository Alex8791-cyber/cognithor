"""Sprint-25 — Planner-Refinement-Loop after PSE synthesis.

The PSE channel returns a deterministic program for a set of input/output
examples; ``cognithor.mcp.pse_tools.handle_pse_synthesize`` exposes that
to the rest of the system. By itself the synthesis result is a black
box — the program passes the demos and any held-out examples by
construction, but downstream consumers (Planner, channels, callers)
have no signal about whether the program *makes sense* for the task
or whether it is likely to overfit on edge cases the demos missed.

Sprint-25 closes that loop: after the engine returns a program, we run
a single LLM-refinement pass that reviews the program against the
natural-language task description and returns a structured verdict.

Hybrid pipeline shape::

    LLM (Planner picks PSE)
         │
         ▼
    PSE synthesis (deterministic, replayable)
         │  ← :func:`refine_pse_program` lives here
         ▼
    LLM refinement (this module)
         │
         ▼
    Final response (Planner formulates with verdict in hand)

The refinement is **side-effect-free** and **idempotent**. It does not
mutate the synthesis result, does not call back into the engine, and
does not re-run the program. The only resource it touches is the
provided ``llm_fn`` — callers that pass a deterministic stub get
deterministic verdicts.

Failure modes are handled defensively: an LLM that returns garbage,
a parse error, an exception in ``llm_fn`` — all collapse to a
``VERDICT_NEUTRAL`` outcome so the wrapping tool always returns a
usable shape.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

log = get_logger(__name__)


__all__ = [
    "VERDICT_ACCEPT",
    "VERDICT_CAUTION",
    "VERDICT_NEUTRAL",
    "VERDICT_REJECT",
    "RefinementVerdict",
    "build_refinement_prompt",
    "parse_refinement_response",
    "refine_pse_program",
]


# Verdict literals — exposed as constants so callers can compare without
# leaking the literal-typing into their signatures.
Verdict = Literal["accept", "caution", "reject", "neutral"]
VERDICT_ACCEPT: Verdict = "accept"
VERDICT_CAUTION: Verdict = "caution"
VERDICT_REJECT: Verdict = "reject"
VERDICT_NEUTRAL: Verdict = "neutral"  # parse-failure / no-llm fallback

Confidence = Literal["high", "medium", "low"]

_VALID_VERDICTS: frozenset[str] = frozenset({"accept", "caution", "reject"})
_VALID_CONFIDENCE: frozenset[str] = frozenset({"high", "medium", "low"})

# Hard cap on the LLM critique we keep, so a runaway response never
# inflates the planner's working memory.
_MAX_EXPLANATION_CHARS = 800

# Hard cap on prompt size — protects against pathological program
# strings or huge example payloads. The planner already operates inside
# the active context profile (Sprint-23/24), but the refinement prompt
# is one-shot so we keep it tight.
_MAX_PROGRAM_CHARS = 2_000
_MAX_TASK_DESC_CHARS = 1_000


@dataclass(frozen=True)
class RefinementVerdict:
    """The structured outcome of an LLM-refinement pass.

    Attributes:
        verdict: One of ``accept`` / ``caution`` / ``reject`` /
            ``neutral`` (the latter is the no-llm-or-parse-failure
            fallback so callers always get a valid shape).
        explanation: Free-form 2–3 sentence critique. Never longer than
            ``_MAX_EXPLANATION_CHARS``.
        confidence: ``high`` / ``medium`` / ``low``.
        raw_response: The unparsed LLM response. Useful for diagnostics
            when ``verdict == "neutral"`` but rarely needed at runtime.
    """

    verdict: Verdict
    explanation: str
    confidence: Confidence
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the shape consumed by the MCP tool wrapper."""
        return {
            "verdict": self.verdict,
            "explanation": self.explanation,
            "confidence": self.confidence,
        }


def build_refinement_prompt(
    *,
    program: str,
    task_description: str,
    n_examples: int,
    n_held_out: int,
    score: float,
) -> str:
    """Render the LLM-refinement prompt deterministically.

    Kept pure (no IO, no llm calls) so the test suite can pin the exact
    string the LLM sees. Sprint-26+ may rewrite the prompt body, but the
    function shape stays stable.
    """
    program_truncated = (program or "")[:_MAX_PROGRAM_CHARS]
    if len(program or "") > _MAX_PROGRAM_CHARS:
        program_truncated += "\n[... truncated ...]"

    desc_truncated = (task_description or "").strip()[:_MAX_TASK_DESC_CHARS]
    if not desc_truncated:
        desc_truncated = "(no task description provided)"

    return (
        "You are reviewing a program that the Cognithor PSE has synthesised "
        "from input/output examples. The program passes all training examples "
        f"({n_examples}) and {n_held_out} held-out anti-overfit example(s).\n"
        "\n"
        "## Task description (natural language)\n"
        f"{desc_truncated}\n"
        "\n"
        "## Synthesised program (DSL source)\n"
        "```\n"
        f"{program_truncated}\n"
        "```\n"
        "\n"
        "## Synthesis score\n"
        f"{score:.3f}  (1.0 = perfect match on examples + held-out)\n"
        "\n"
        "## Your task\n"
        "Critique this program in 2-3 sentences. Specifically:\n"
        "- Does the program look semantically correct for the described task?\n"
        "- Are there obvious edge cases the examples might not cover?\n"
        "- Does the program look minimal, or does it carry unnecessary "
        "    complexity (a sign of demo-overfit)?\n"
        "\n"
        'Reply with **only** a JSON object: {"verdict": "accept" | "caution" '
        '| "reject", "explanation": "<2-3 sentences>", "confidence": '
        '"high" | "medium" | "low"}. No prose around the JSON.'
    )


def parse_refinement_response(raw: str) -> RefinementVerdict:
    """Extract a :class:`RefinementVerdict` from a free-form LLM reply.

    Tolerant of:
      * Pure JSON (``{"verdict": ..., "explanation": ..., "confidence": ...}``)
      * Markdown-fenced JSON (```` ```json {...} ``` ```` )
      * Free-form prose with a JSON object somewhere inside it.

    Falls back to :data:`VERDICT_NEUTRAL` on any parse failure or
    out-of-range value, never raises.
    """
    if not raw or not raw.strip():
        return RefinementVerdict(
            verdict=VERDICT_NEUTRAL,
            explanation="empty LLM response",
            confidence="low",
            raw_response=raw or "",
        )

    cleaned = raw.strip()

    # Strip a leading ```json / ``` fence if present.
    fence = re.search(r"```(?:json)?\s*\n?(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()

    # If the response still isn't valid JSON, find the first balanced
    # object-literal and try that.
    candidate = cleaned
    if not candidate.startswith("{"):
        m = re.search(r"\{.*\}", candidate, re.DOTALL)
        if m:
            candidate = m.group(0)

    try:
        data = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        log.debug("pse_refinement_parse_failed", raw=raw[:200])
        return RefinementVerdict(
            verdict=VERDICT_NEUTRAL,
            explanation="LLM response was not valid JSON",
            confidence="low",
            raw_response=raw[:_MAX_EXPLANATION_CHARS],
        )

    if not isinstance(data, dict):
        return RefinementVerdict(
            verdict=VERDICT_NEUTRAL,
            explanation="LLM response was not a JSON object",
            confidence="low",
            raw_response=raw[:_MAX_EXPLANATION_CHARS],
        )

    raw_verdict = str(data.get("verdict", "")).strip().lower()
    if raw_verdict not in _VALID_VERDICTS:
        return RefinementVerdict(
            verdict=VERDICT_NEUTRAL,
            explanation=f"LLM verdict {raw_verdict!r} not in {sorted(_VALID_VERDICTS)}",
            confidence="low",
            raw_response=raw[:_MAX_EXPLANATION_CHARS],
        )

    raw_confidence = str(data.get("confidence", "")).strip().lower()
    confidence: Confidence = "low"
    if raw_confidence in _VALID_CONFIDENCE:
        # Narrow to the Literal type so mypy is happy.
        if raw_confidence == "high":
            confidence = "high"
        elif raw_confidence == "medium":
            confidence = "medium"
        else:
            confidence = "low"

    explanation = str(data.get("explanation", "")).strip()[:_MAX_EXPLANATION_CHARS]
    if not explanation:
        explanation = "(LLM provided no explanation)"

    # raw_verdict is now one of accept/caution/reject — narrow accordingly.
    verdict: Verdict
    if raw_verdict == "accept":
        verdict = VERDICT_ACCEPT
    elif raw_verdict == "caution":
        verdict = VERDICT_CAUTION
    else:
        verdict = VERDICT_REJECT

    return RefinementVerdict(
        verdict=verdict,
        explanation=explanation,
        confidence=confidence,
        raw_response=raw[:_MAX_EXPLANATION_CHARS],
    )


async def refine_pse_program(
    *,
    program: str,
    task_description: str,
    n_examples: int,
    n_held_out: int,
    score: float,
    llm_fn: Callable[[str], Awaitable[str]] | None,
) -> RefinementVerdict:
    """Run a single LLM-refinement pass on a synthesised PSE program.

    Returns :data:`VERDICT_NEUTRAL` when ``llm_fn`` is None, when the LLM
    raises, or when the LLM reply fails to parse — in every case the
    pipeline keeps moving without a hard error.
    """
    if llm_fn is None:
        return RefinementVerdict(
            verdict=VERDICT_NEUTRAL,
            explanation="no llm_fn wired (refinement disabled)",
            confidence="low",
        )
    if not program or not program.strip():
        return RefinementVerdict(
            verdict=VERDICT_NEUTRAL,
            explanation="no program to refine",
            confidence="low",
        )

    prompt = build_refinement_prompt(
        program=program,
        task_description=task_description,
        n_examples=n_examples,
        n_held_out=n_held_out,
        score=score,
    )

    try:
        raw_response = await llm_fn(prompt)
    except Exception:
        log.debug("pse_refinement_llm_failed", exc_info=True)
        return RefinementVerdict(
            verdict=VERDICT_NEUTRAL,
            explanation="llm_fn raised during refinement",
            confidence="low",
        )

    return parse_refinement_response(raw_response)
