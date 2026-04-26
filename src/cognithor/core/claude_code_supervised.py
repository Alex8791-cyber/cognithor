"""Supervised Claude Code driver.

Drives the ``claude`` CLI in ``--output-format stream-json`` mode, parses the
NDJSON event stream, records every tool call for the Observer, and hands
turn-end outcomes back to a goal-evaluation callback so the caller can
decide whether to send a follow-up prompt or finish.

This is the autonomous-loop companion to the hook bridge in
``cognithor.gateway.claude_code_hooks``. The hook bridge gates tool calls
*during* a Claude Code session (interactive in VS Code or headless). This
class additionally *starts* sessions, *monitors* them, and *re-prompts*
when a goal is not yet reached -- replacing the human at the keyboard.

Scope limits by design
----------------------
- We do not fork the Anthropic SDK. We spawn the same ``claude`` CLI the
  user already uses and parse its event stream.
- We do not re-implement Gatekeeper decisions here. Cognithor's
  ``~/.claude/settings.json`` HTTP hooks (installed by
  ``contrib/claude-code-bridge/install.py``) fire *inside the subprocess*
  too, so the Gatekeeper still gates every tool call in supervised mode.
- Budget enforcement is hard (max_turns, max_duration, max_cost_usd) --
  we would rather stop a runaway loop than hit an API bill surprise.

Event stream reference
----------------------
Claude Code emits NDJSON with these relevant envelope types:
- ``system`` (subtype ``init``): opening frame with session_id + tools.
- ``assistant``: message with text or ``tool_use`` blocks.
- ``user``: tool results (``tool_result`` blocks).
- ``result`` (subtype ``success`` | ``error_max_turns`` | ...): final frame
  carrying total cost + terminal text response.

We treat ``result`` as turn-end and feed the accumulated tool results into
the Observer. The caller's goal-evaluator then decides re-prompt vs halt.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from cognithor.core.llm_backend import (
    ChatResponse,
    EmbedResponse,
    LLMBackend,
    LLMBackendError,
    LLMBackendType,
)
from cognithor.models import ToolResult
from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.core.observer import ObserverAudit


log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────


GoalVerdict = Literal["done", "continue", "abort"]


@dataclass(frozen=True)
class GoalEvaluation:
    """Outcome of a goal-evaluation round.

    ``verdict``:
        - ``done``: goal reached, stop the loop and return ``final_text``.
        - ``continue``: send ``next_prompt`` and keep looping.
        - ``abort``: unrecoverable, stop with an error.
    """

    verdict: GoalVerdict
    next_prompt: str = ""
    reason: str = ""


@dataclass
class TurnResult:
    """One subprocess invocation of ``claude -p``."""

    turn: int
    prompt: str
    assistant_text: str = ""
    tool_results: list[ToolResult] = field(default_factory=list)
    cost_usd: float = 0.0
    duration_ms: int = 0
    is_error: bool = False
    error: str | None = None
    session_id: str = ""
    raw_events: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SupervisorResult:
    """Outcome of the full supervised run."""

    verdict: GoalVerdict  # "done" if goal evaluator said so, "abort" otherwise
    final_text: str
    reason: str
    turns: list[TurnResult]
    total_cost_usd: float
    total_duration_ms: int


# A goal evaluator receives the original user intent plus the per-turn
# history accumulated so far and returns whether the supervisor should
# stop, send another prompt, or abort. Receiving the intent makes it
# possible to plug in Planner-style judges that compare turn outcomes
# against the goal explicitly.
GoalEvaluator = Callable[[str, list[TurnResult]], Awaitable[GoalEvaluation]]


# ─────────────────────────────────────────────────────────────────────────────
# Supervisor
# ─────────────────────────────────────────────────────────────────────────────


class ClaudeCodeSupervisor:
    """Outer-loop driver for Claude Code in stream-json mode."""

    def __init__(
        self,
        *,
        model: str = "sonnet",
        claude_path: str | None = None,
        observer: ObserverAudit | None = None,
        goal_evaluator: GoalEvaluator | None = None,
        max_turns: int = 8,
        max_duration_seconds: int = 1800,
        max_cost_usd: float = 5.0,
        per_turn_timeout_seconds: int = 600,
        working_directory: str | None = None,
        extra_cli_args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self._model = model
        self._claude_path = claude_path or shutil.which("claude") or "claude"
        self._observer = observer
        self._goal_evaluator = goal_evaluator
        self._max_turns = max(1, max_turns)
        self._max_duration_seconds = max_duration_seconds
        self._max_cost_usd = max_cost_usd
        self._per_turn_timeout_seconds = per_turn_timeout_seconds
        self._working_directory = working_directory
        self._extra_cli_args = list(extra_cli_args or [])
        self._env = env

    # ── Public API ───────────────────────────────────────────────────────

    async def run(self, user_intent: str) -> SupervisorResult:
        """Drive Claude Code until the goal is reached or a budget is hit."""
        start = time.monotonic()
        turns: list[TurnResult] = []
        next_prompt = user_intent
        session_resume_id: str | None = None

        for turn_idx in range(1, self._max_turns + 1):
            if time.monotonic() - start > self._max_duration_seconds:
                return self._finish(
                    turns=turns,
                    verdict="abort",
                    reason=f"max_duration_seconds ({self._max_duration_seconds}) exceeded",
                    final_text=_last_text(turns),
                    start=start,
                )

            total_cost = sum(t.cost_usd for t in turns)
            if total_cost >= self._max_cost_usd:
                return self._finish(
                    turns=turns,
                    verdict="abort",
                    reason=f"max_cost_usd ({self._max_cost_usd:.2f}) exceeded at ${total_cost:.4f}",
                    final_text=_last_text(turns),
                    start=start,
                )

            log.info(
                "claude_code_supervisor_turn_start",
                turn=turn_idx,
                prompt_preview=next_prompt[:120],
                resume_id=session_resume_id,
            )

            turn = await self._run_turn(
                turn_idx=turn_idx,
                prompt=next_prompt,
                resume_session_id=session_resume_id,
            )
            turns.append(turn)
            session_resume_id = turn.session_id or session_resume_id

            if turn.is_error:
                return self._finish(
                    turns=turns,
                    verdict="abort",
                    reason=f"turn {turn_idx} errored: {turn.error}",
                    final_text=turn.assistant_text,
                    start=start,
                )

            evaluation = await self._evaluate(user_intent=user_intent, turns=turns)

            log.info(
                "claude_code_supervisor_turn_end",
                turn=turn_idx,
                verdict=evaluation.verdict,
                cost_usd=turn.cost_usd,
                tool_count=len(turn.tool_results),
                reason=evaluation.reason[:160] if evaluation.reason else None,
            )

            if evaluation.verdict == "done":
                return self._finish(
                    turns=turns,
                    verdict="done",
                    reason=evaluation.reason or "goal reached",
                    final_text=turn.assistant_text,
                    start=start,
                )
            if evaluation.verdict == "abort":
                return self._finish(
                    turns=turns,
                    verdict="abort",
                    reason=evaluation.reason or "aborted by goal evaluator",
                    final_text=turn.assistant_text,
                    start=start,
                )

            next_prompt = evaluation.next_prompt or _default_followup_prompt(turn)

        return self._finish(
            turns=turns,
            verdict="abort",
            reason=f"max_turns ({self._max_turns}) exceeded",
            final_text=_last_text(turns),
            start=start,
        )

    # ── Internals ────────────────────────────────────────────────────────

    async def _run_turn(
        self,
        *,
        turn_idx: int,
        prompt: str,
        resume_session_id: str | None,
    ) -> TurnResult:
        cmd = [
            self._claude_path,
            "-p",
            "--model",
            self._model,
            "--output-format",
            "stream-json",
            "--input-format",
            "stream-json",
            "--verbose",
        ]
        if resume_session_id:
            cmd += ["--resume", resume_session_id]
        cmd += self._extra_cli_args

        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._working_directory,
                env=self._env,
            )
        except FileNotFoundError:
            return TurnResult(
                turn=turn_idx,
                prompt=prompt,
                is_error=True,
                error=(
                    f"claude CLI not found at {self._claude_path!r}. "
                    "Install from https://docs.anthropic.com/claude-code"
                ),
            )

        assert proc.stdin is not None
        assert proc.stdout is not None

        user_frame = {
            "type": "user",
            "message": {"role": "user", "content": prompt},
        }
        proc.stdin.write((json.dumps(user_frame) + "\n").encode("utf-8"))
        await proc.stdin.drain()
        proc.stdin.close()

        turn = TurnResult(turn=turn_idx, prompt=prompt)
        # Maps tool_use_id -> index in turn.tool_results so tool_result blocks
        # can be paired even when several tool_use blocks ran in parallel.
        pending: dict[str, int] = {}

        try:
            async for event in _read_ndjson(
                proc.stdout, timeout_seconds=self._per_turn_timeout_seconds
            ):
                turn.raw_events.append(event)
                self._absorb_event(turn, event, pending=pending)
        except TimeoutError:
            turn.is_error = True
            turn.error = f"turn exceeded per_turn_timeout_seconds={self._per_turn_timeout_seconds}"
            with contextlib.suppress(ProcessLookupError):
                proc.kill()

        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except TimeoutError:
            proc.kill()
            await proc.wait()

        if proc.returncode not in (0, None) and not turn.is_error:
            stderr = b""
            if proc.stderr is not None:
                try:
                    stderr = await proc.stderr.read()
                except Exception:
                    stderr = b""
            turn.is_error = True
            turn.error = f"claude exit {proc.returncode}: {stderr.decode(errors='replace')[:400]}"

        turn.duration_ms = int((time.monotonic() - start) * 1000)
        return turn

    def _absorb_event(
        self,
        turn: TurnResult,
        event: dict[str, Any],
        *,
        pending: dict[str, int],
    ) -> None:
        etype = event.get("type")
        if etype == "system" and event.get("subtype") == "init":
            turn.session_id = event.get("session_id", turn.session_id) or turn.session_id
            return

        if etype == "assistant":
            msg = event.get("message") or {}
            for block in _iter_content(msg):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    # Only the final assistant text of the turn is user-facing;
                    # we keep overwriting so the last text wins.
                    turn.assistant_text = block["text"]
                elif block.get("type") == "tool_use":
                    # Pair with tool_result when it arrives; we pre-seed a
                    # placeholder so ordering is preserved and remember its
                    # index by tool_use_id for parallel-call correctness.
                    tool_use_id = str(block.get("id") or "")
                    turn.tool_results.append(
                        ToolResult(
                            tool_name=str(block.get("name", "")),
                            content="",
                            is_error=False,
                        )
                    )
                    if tool_use_id:
                        pending[tool_use_id] = len(turn.tool_results) - 1
            return

        if etype == "user":
            # tool_result blocks come back inside user-role messages.
            msg = event.get("message") or {}
            for block in _iter_content(msg):
                if block.get("type") != "tool_result":
                    continue
                tool_use_id = str(block.get("tool_use_id") or "")
                content = _tool_result_text(block.get("content"))
                is_error = bool(block.get("is_error"))

                idx = pending.pop(tool_use_id, None)
                if idx is not None and 0 <= idx < len(turn.tool_results):
                    placeholder = turn.tool_results[idx]
                    turn.tool_results[idx] = ToolResult(
                        tool_name=placeholder.tool_name,
                        content=content,
                        is_error=is_error,
                        error_message=content if is_error else None,
                    )
                else:
                    # Result without a known tool_use predecessor (e.g. id
                    # missing / out-of-order): append a best-effort record.
                    turn.tool_results.append(
                        ToolResult(
                            tool_name=tool_use_id or "unknown",
                            content=content,
                            is_error=is_error,
                            error_message=content if is_error else None,
                        )
                    )
            return

        if etype == "result":
            subtype = event.get("subtype", "")
            cost = event.get("total_cost_usd")
            if isinstance(cost, int | float):
                turn.cost_usd = float(cost)
            if subtype not in ("success",):
                turn.is_error = True
                turn.error = event.get("result") or subtype
            elif isinstance(event.get("result"), str):
                turn.assistant_text = event["result"]
            return

    async def _evaluate(self, *, user_intent: str, turns: list[TurnResult]) -> GoalEvaluation:
        if self._goal_evaluator is not None:
            try:
                return await self._goal_evaluator(user_intent, turns)
            except Exception as exc:
                log.warning("claude_code_supervisor_evaluator_failed", error=str(exc))
                return GoalEvaluation(verdict="abort", reason=f"evaluator raised: {exc!s}")

        # Default evaluator: one turn is one shot. The caller is expected
        # to supply their own evaluator (Planner / Observer-driven) for
        # anything multi-step.
        last = turns[-1]
        if self._observer is not None and last.tool_results:
            try:
                audit = await self._observer.audit(
                    user_message=user_intent,
                    response=last.assistant_text or "",
                    tool_results=last.tool_results,
                    session_id=last.session_id or "supervised",
                )
            except Exception:
                log.debug("claude_code_supervisor_observer_failed", exc_info=True)
                return GoalEvaluation(verdict="done", reason="observer_unavailable")
            if not audit.overall_passed and audit.retry_strategy in (
                "response_regen",
                "pge_reloop",
            ):
                failed = [d for d in audit.dimensions.values() if not d.passed]
                reasons = "; ".join(f"{d.name}: {d.reason}" for d in failed[:3])
                followup = (
                    "The previous attempt was flagged by the Observer audit: "
                    f"{reasons}. Address the specific issues and retry."
                )
                return GoalEvaluation(verdict="continue", next_prompt=followup, reason=reasons)

        return GoalEvaluation(verdict="done", reason="single-turn completion")

    def _finish(
        self,
        *,
        turns: list[TurnResult],
        verdict: GoalVerdict,
        reason: str,
        final_text: str,
        start: float,
    ) -> SupervisorResult:
        return SupervisorResult(
            verdict=verdict,
            final_text=final_text,
            reason=reason,
            turns=turns,
            total_cost_usd=sum(t.cost_usd for t in turns),
            total_duration_ms=int((time.monotonic() - start) * 1000),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


async def _read_ndjson(
    stream: asyncio.StreamReader, *, timeout_seconds: int
) -> AsyncIterator[dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError()
        try:
            line = await asyncio.wait_for(stream.readline(), timeout=remaining)
        except TimeoutError:
            raise
        if not line:
            return
        text = line.decode("utf-8", errors="replace").strip()
        if not text:
            continue
        try:
            yield json.loads(text)
        except json.JSONDecodeError:
            log.debug("claude_code_supervisor_ndjson_skip", line_preview=text[:160])
            continue


def _iter_content(message: Any) -> list[dict[str, Any]]:
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return []


def _tool_result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif isinstance(block.get("content"), str):
                parts.append(block["content"])
        return "\n".join(parts)
    if content is None:
        return ""
    try:
        return json.dumps(content, default=str)
    except Exception:
        return str(content)


def _last_text(turns: list[TurnResult]) -> str:
    for turn in reversed(turns):
        if turn.assistant_text:
            return turn.assistant_text
    return ""


def _default_followup_prompt(turn: TurnResult) -> str:
    if turn.tool_results:
        errs = [r for r in turn.tool_results if r.is_error]
        if errs:
            snippet = (errs[0].error_message or errs[0].content or "")[:240]
            return (
                f"The last turn produced an error: {snippet}\nFix it and continue toward the goal."
            )
    return "Continue toward the goal. If you believe the goal is reached, state 'DONE'."


__all__ = [
    "ClaudeCodeSupervisedBackend",
    "ClaudeCodeSupervisor",
    "GoalEvaluation",
    "GoalEvaluator",
    "PlannerGoalEvaluator",
    "SupervisorResult",
    "TurnResult",
]


# ─────────────────────────────────────────────────────────────────────────────
# Planner-driven goal evaluator
#
# Lightweight LLM judge that compares the goal against accumulated turn
# outcomes and returns done/continue/abort with a follow-up prompt. Does
# not invoke the full Cognithor Planner (which expects WorkingMemory and
# tool schemas) -- instead it makes a focused chat call to the same LLM
# client the Planner uses.
# ─────────────────────────────────────────────────────────────────────────────


_GOAL_JUDGE_SYSTEM = (
    "You are a goal-completion judge for an autonomous coding agent driving "
    "Claude Code. Read the user goal and the agent's progress, then decide "
    "whether the goal is reached, whether one more attempt would help, or "
    "whether the agent is stuck.\n\n"
    "Respond with EXACTLY one JSON object on a single line, no prose, no "
    "code fences, with this shape:\n"
    '  {"verdict": "done"|"continue"|"abort", '
    '"next_prompt": "<used only when verdict=continue>", '
    '"reason": "<one short sentence>"}\n\n'
    "Decision rules:\n"
    "- done: the user goal is fully and visibly satisfied by the latest "
    "assistant message and the tool history. Prefer 'done' when in doubt and "
    "the agent's last message claims completion AND the tool errors are zero.\n"
    "- continue: there is real progress but the goal is not yet met. Provide "
    "a focused next_prompt that names the specific next step. Do NOT just "
    "say 'continue'. Be concrete.\n"
    "- abort: the agent is looping, repeating the same tool with the same "
    "error, or has clearly diverged from the goal. Recovery via re-prompting "
    "is unlikely.\n"
)


def _format_turn_summary(turn: TurnResult, *, max_chars: int = 600) -> str:
    parts = [
        f"turn {turn.turn} (cost ${turn.cost_usd:.4f}, {turn.duration_ms}ms"
        f"{', ERROR' if turn.is_error else ''}):",
    ]
    if turn.tool_results:
        parts.append(f"  tools: {len(turn.tool_results)}")
        for tr in turn.tool_results[:6]:
            content = (tr.error_message or tr.content or "").strip().replace("\n", " ")
            tag = "ERR" if tr.is_error else "ok"
            parts.append(f"    - [{tag}] {tr.tool_name}: {content[:100]}")
        if len(turn.tool_results) > 6:
            parts.append(f"    ... (+{len(turn.tool_results) - 6} more)")
    if turn.assistant_text:
        parts.append(f"  text: {turn.assistant_text.strip()[:max_chars]}")
    if turn.error:
        parts.append(f"  error: {turn.error[:300]}")
    return "\n".join(parts)


class PlannerGoalEvaluator:
    """Goal evaluator that asks an LLM whether the supervisor should stop.

    Plug-compatible with ``ClaudeCodeSupervisor.goal_evaluator``: instances
    are awaitable callables matching the
    ``Callable[[str, list[TurnResult]], Awaitable[GoalEvaluation]]``
    signature.

    The judge is intentionally model-agnostic and accepts any LLM client
    that exposes ``async chat(model, messages, **kwargs) -> ChatResponse``
    (so Cognithor's Ollama / Anthropic / Claude-Code / vLLM backends all
    work). On any LLM failure the evaluator returns ``verdict="continue"``
    with a generic follow-up prompt -- failing the loop closed (towards
    "done") would silently mask real progress problems.

    Args:
        llm: Async chat client. Must accept (model, messages, **kwargs)
            and return a ``ChatResponse``-shaped object with a ``content``
            attribute.
        model: Model name to pass to the chat call.
        max_history_turns: Cap on how many earlier turns to summarize.
        on_failure: What to return if the LLM call or JSON parse fails.
            Defaults to a benign "continue" verdict.
    """

    def __init__(
        self,
        *,
        llm: Any,
        model: str,
        max_history_turns: int = 6,
        on_failure: GoalEvaluation | None = None,
    ) -> None:
        self._llm = llm
        self._model = model
        self._max_history_turns = max(1, max_history_turns)
        self._on_failure = on_failure or GoalEvaluation(
            verdict="continue",
            next_prompt=(
                "The goal-completion judge could not produce a verdict. "
                "Continue toward the goal and explicitly state 'DONE' when "
                "you believe it is reached."
            ),
            reason="judge_unavailable",
        )

    async def __call__(self, user_intent: str, turns: list[TurnResult]) -> GoalEvaluation:
        if not turns:
            return GoalEvaluation(verdict="continue", reason="no turns yet")

        history = turns[-self._max_history_turns :]
        history_block = "\n".join(_format_turn_summary(t) for t in history)
        prompt_user = (
            f"USER GOAL:\n{user_intent}\n\n"
            f"PROGRESS ({len(history)} of {len(turns)} turns shown, "
            f"oldest first):\n{history_block}\n\n"
            "Return the JSON verdict now."
        )

        try:
            response = await self._llm.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": _GOAL_JUDGE_SYSTEM},
                    {"role": "user", "content": prompt_user},
                ],
                temperature=0.2,
                top_p=0.9,
                format_json=True,
            )
        except TypeError:
            # Older clients without format_json kwarg.
            try:
                response = await self._llm.chat(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": _GOAL_JUDGE_SYSTEM},
                        {"role": "user", "content": prompt_user},
                    ],
                    temperature=0.2,
                    top_p=0.9,
                )
            except Exception as exc:
                log.warning("planner_goal_evaluator_llm_error", error=str(exc))
                return self._on_failure
        except Exception as exc:
            log.warning("planner_goal_evaluator_llm_error", error=str(exc))
            return self._on_failure

        content = getattr(response, "content", "") or ""
        parsed = _extract_json_object(content)
        if parsed is None:
            log.warning("planner_goal_evaluator_parse_failed", content_preview=content[:200])
            return self._on_failure

        verdict = str(parsed.get("verdict", "")).strip().lower()
        if verdict not in ("done", "continue", "abort"):
            log.warning("planner_goal_evaluator_bad_verdict", got=verdict)
            return self._on_failure

        next_prompt = str(parsed.get("next_prompt", "")).strip()
        reason = str(parsed.get("reason", "")).strip()

        log.info(
            "planner_goal_evaluator_verdict",
            verdict=verdict,
            turns=len(turns),
            reason=reason[:160],
        )
        return GoalEvaluation(
            verdict=verdict,  # type: ignore[arg-type]
            next_prompt=next_prompt,
            reason=reason,
        )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extract a single top-level JSON object from LLM output.

    Tolerates a trailing newline, a leading prose preamble, or accidental
    code fences. Returns ``None`` if nothing parseable is found.
    """
    if not text:
        return None
    # Strip a single set of ```json ... ``` fences if present.
    stripped = text.strip()
    if stripped.startswith("```"):
        first_nl = stripped.find("\n")
        if first_nl > 0:
            stripped = stripped[first_nl + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        stripped = stripped.strip()
    # Try direct parse first.
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Fallback: find the first balanced {...} block.
    start = stripped.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(stripped)):
        ch = stripped[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    return None
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Backend wrapper -- exposes the supervisor as a regular LLMBackend so it
# can be selected via LLMBackendType.CLAUDE_CODE_SUPERVISED.
# ─────────────────────────────────────────────────────────────────────────────


class ClaudeCodeSupervisedBackend(LLMBackend):
    """LLMBackend adapter around ``ClaudeCodeSupervisor``.

    Behaves like ``ClaudeCodeBackend`` for the caller (chat + list_models +
    is_available + close) but each ``chat()`` call drives one or more
    supervised turns. ``max_turns=1`` makes it functionally identical to
    the plain Claude Code backend; higher values + a goal evaluator turn it
    into an autonomous loop.

    Streaming is supported on a coarse granularity: each turn's final
    assistant text is yielded as a single chunk, prefixed with a separator
    on multi-turn runs. Token-by-token streaming would require a different
    transport than ``stream-json`` and is intentionally out of scope here.
    """

    def __init__(
        self,
        *,
        model: str = "sonnet",
        claude_path: str | None = None,
        observer: ObserverAudit | None = None,
        goal_evaluator: GoalEvaluator | None = None,
        max_turns: int = 1,
        max_duration_seconds: int = 1800,
        max_cost_usd: float = 5.0,
        per_turn_timeout_seconds: int = 600,
        working_directory: str | None = None,
        extra_cli_args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self._model = model
        self._claude_path = claude_path or shutil.which("claude") or "claude"
        self._observer = observer
        self._goal_evaluator = goal_evaluator
        self._max_turns = max(1, max_turns)
        self._max_duration_seconds = max_duration_seconds
        self._max_cost_usd = max_cost_usd
        self._per_turn_timeout_seconds = per_turn_timeout_seconds
        self._working_directory = working_directory
        self._extra_cli_args = list(extra_cli_args or [])
        self._env = env

    @property
    def backend_type(self) -> LLMBackendType:
        return LLMBackendType.CLAUDE_CODE_SUPERVISED

    def _build_supervisor(self, model: str) -> ClaudeCodeSupervisor:
        return ClaudeCodeSupervisor(
            model=model or self._model,
            claude_path=self._claude_path,
            observer=self._observer,
            goal_evaluator=self._goal_evaluator,
            max_turns=self._max_turns,
            max_duration_seconds=self._max_duration_seconds,
            max_cost_usd=self._max_cost_usd,
            per_turn_timeout_seconds=self._per_turn_timeout_seconds,
            working_directory=self._working_directory,
            extra_cli_args=self._extra_cli_args,
            env=self._env,
        )

    @staticmethod
    def _flatten_messages(messages: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for msg in messages:
            role = str(msg.get("role", "user"))
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            if not isinstance(content, str):
                content = str(content)
            if role == "system":
                parts.append(f"[Context]: {content}")
            elif role == "assistant":
                parts.append(f"[Previous response]: {content}")
            else:
                parts.append(content)
        return "\n\n".join(p for p in parts if p)

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        format_json: bool = False,
    ) -> ChatResponse:
        prompt = self._flatten_messages(messages)
        supervisor = self._build_supervisor(model)
        result = await supervisor.run(prompt)
        if result.verdict == "abort" and not result.final_text:
            raise LLMBackendError(
                f"Supervised Claude Code aborted: {result.reason}",
            )
        return ChatResponse(
            content=result.final_text,
            model=(model or self._model),
            usage=None,
            raw={
                "verdict": result.verdict,
                "reason": result.reason,
                "turns": [
                    {
                        "turn": t.turn,
                        "cost_usd": t.cost_usd,
                        "duration_ms": t.duration_ms,
                        "tool_count": len(t.tool_results),
                        "is_error": t.is_error,
                    }
                    for t in result.turns
                ],
                "total_cost_usd": result.total_cost_usd,
                "total_duration_ms": result.total_duration_ms,
            },
        )

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> AsyncIterator[str]:
        # Coarse-grained streaming: yield each turn's final text. Stays
        # within the LLMBackend contract without requiring token-level
        # parsing of stream-json content blocks.
        result = await self.chat(model, messages, temperature=temperature, top_p=top_p)
        yield result.content

    async def embed(self, model: str, text: str) -> EmbedResponse:
        raise LLMBackendError(
            "Claude Code (supervised) does not support embeddings. "
            "Use Ollama or OpenAI for embedding fallback.",
        )

    async def is_available(self) -> bool:
        try:
            proc = await asyncio.create_subprocess_exec(
                self._claude_path,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            return proc.returncode == 0
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        return ["opus", "sonnet", "haiku"]

    async def close(self) -> None:
        return None
