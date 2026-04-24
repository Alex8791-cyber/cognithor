"""Compiler translates Crew definitions into ordered execution steps that
route through the existing Planner/Gatekeeper pipeline.

The compiler itself is a pure function; the `execute_task` helper is where
the actual PGE integration happens (Task 11 — now LIVE). Every task runs
through :meth:`cognithor.core.planner.Planner.formulate_response`, which in
turn drives the Gatekeeper + Executor — the Crew-Layer never bypasses PGE.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import uuid as _uuid
from pathlib import Path
from typing import Any

from cognithor.crew.agent import CrewAgent
from cognithor.crew.errors import GuardrailFailure
from cognithor.crew.guardrails.base import GuardrailResult
from cognithor.crew.guardrails.function_guardrail import FunctionGuardrail
from cognithor.crew.guardrails.string_guardrail import StringGuardrail
from cognithor.crew.output import CrewOutput, TaskOutput, TokenUsageDict
from cognithor.crew.process import CrewProcess
from cognithor.crew.task import CrewTask
from cognithor.crew.tool_resolver import resolve_tools
from cognithor.models import ToolResult, WorkingMemory
from cognithor.security.pii_redactor import PIIRedactor

log = logging.getLogger(__name__)

# Module-level singleton - redactor is stateless; instantiating per-call
# would re-compile regex for every audit event.
_CREW_PII_REDACTOR = PIIRedactor()


def _scrub_audit_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``fields`` with string values passed through the
    PII redactor. Non-string values (ints, floats, bools, dicts) pass through
    untouched. Lists of strings are element-wise redacted; deeper nesting
    falls through as-is (audit-chain fields are flat by convention).
    """
    cleaned: dict[str, Any] = {}
    for key, value in fields.items():
        if isinstance(value, str):
            sanitized, _matches = _CREW_PII_REDACTOR.redact(value)
            cleaned[key] = sanitized
        elif isinstance(value, list) and value and all(isinstance(v, str) for v in value):
            cleaned[key] = [_CREW_PII_REDACTOR.redact(v)[0] for v in value]
        else:
            cleaned[key] = value
    return cleaned


# Audit helper - wraps cognithor.security.audit.AuditTrail.record_event()
# as a single module-local callable so tests can patch it cleanly.
_AUDIT_TRAIL_UNINITIALIZED = object()  # distinct from None = "init ran, permanently disabled"
_audit_trail: Any = _AUDIT_TRAIL_UNINITIALIZED
_audit_lock = threading.Lock()


def _get_audit_trail() -> Any:
    """Lazy-build a process-wide AuditTrail under the Cognithor audit log path.

    Uses a dedicated sentinel so a transient init failure doesn't retry on
    every audit event: once ``_audit_trail`` is set (to an AuditTrail OR to
    ``None``), future calls short-circuit. ``None`` means "init ran, failed,
    don't try again" — without the sentinel the ``is not None`` guard would
    let every audit event re-run ``load_config()`` + ``AuditTrail()``.
    """
    global _audit_trail
    with _audit_lock:
        if _audit_trail is not _AUDIT_TRAIL_UNINITIALIZED:
            return _audit_trail
        try:
            from cognithor.config import load_config
            from cognithor.security.audit import AuditTrail

            cfg = load_config()
            log_dir = Path(cfg.cognithor_home) / "logs"
            _audit_trail = AuditTrail(log_dir=log_dir)
        except Exception as exc:
            log.warning(
                "AuditTrail init failed; crew audit events will be no-ops for this process",
                exc_info=exc,
            )
            _audit_trail = None  # cached permanent no-op (NOT the uninitialized sentinel)
        return _audit_trail


def append_audit(event: str, **fields: Any) -> None:
    """Emit a Crew-Layer audit event via the Hashline-Guard chain.

    Falls back to a no-op when AuditTrail cannot be built (e.g. standalone
    test without ~/.cognithor/ present). Test code monkey-patches this
    callable directly rather than the AuditTrail inside it.
    """
    trail = _get_audit_trail()
    if trail is None:
        return
    session_id = fields.pop("trace_id", "crew")
    scrubbed = _scrub_audit_fields(fields)  # spec 8.2 / R4-I8 - PII before persist
    try:
        trail.record_event(session_id=session_id, event_type=event, details=scrubbed)
    except Exception as exc:
        # Spec 11.5: audit failures must be SURFACED, not silently swallowed.
        log.warning(
            "crew_audit_record_failed - Hashline-Guard chain may be incomplete",
            extra={"event": event, "session_id": session_id},
            exc_info=exc,
        )
        try:
            from cognithor.telemetry.metrics import MetricsProvider

            MetricsProvider.get_instance().counter(
                "cognithor_crew_audit_record_failures_total",
                1,
                labels={"reason": type(exc).__name__},
            )
        except (ImportError, AttributeError):
            pass


def order_tasks_sequential(tasks: list[CrewTask]) -> list[CrewTask]:
    """Sequential process: keep the declaration order."""
    return list(tasks)


def _build_user_message(
    task: CrewTask,
    inputs: dict[str, Any] | None,
) -> str:
    """Render the Crew task as a single user message.

    System-level framing (role, goal, backstory) is owned by the Planner via
    its own SYSTEM_PROMPT — the Crew-Layer intentionally does NOT inject its
    own system prompt, to avoid duplicating Cognithor's identity framing.
    We fold role/goal/backstory into the user message so the Planner still
    sees who it's acting as, but with a single source of truth for identity.
    """
    parts: list[str] = []
    parts.append(f"[Crew role: {task.agent.role}] goal: {task.agent.goal}")
    if task.agent.backstory:
        parts.append(f"Background: {task.agent.backstory}")
    parts.append("")
    desc = task.description
    if inputs:
        for k, v in inputs.items():
            desc = desc.replace("{" + str(k) + "}", str(v))
    parts.append(desc)
    parts.append(f"\nExpected output: {task.expected_output}")
    return "\n".join(parts)


def _read_token_usage(planner: Any) -> TokenUsageDict | None:
    """Pull the last-call token count from the planner's cost tracker.

    Uses the additive ``last_call()`` helper on CostTracker (Step 5). Probe
    is duck-typed so we gracefully degrade against older CostTracker builds,
    embedded Planners that disabled cost tracking, or test doubles that
    haven't configured a tracker.
    """
    tracker = getattr(planner, "_cost_tracker", None)
    if tracker is None:
        return None
    last = getattr(tracker, "last_call", None)
    if not callable(last):
        return None
    try:
        record = last()
    except Exception:
        return None
    if record is None:
        return None
    input_tokens = int(getattr(record, "input_tokens", 0))
    output_tokens = int(getattr(record, "output_tokens", 0))
    return TokenUsageDict(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


def _is_already_guardrail(g: Any) -> bool:
    """Duck-type check: Guardrails (FunctionGuardrail, StringGuardrail, chain-wrapper)
    have a ``__call__`` AND either a ``_rule`` attribute (StringGuardrail), a
    ``_fn`` attribute (FunctionGuardrail), or are builtin closures. Anything
    else the user passes is treated as a raw callable and wrapped.
    """
    return hasattr(g, "_rule") or hasattr(g, "_fn") or getattr(g, "_is_guardrail", False)


def _normalize_guardrail(g: Any, *, ollama_client: Any, model: str) -> Any:
    """Normalize whatever the user stuck into ``CrewTask.guardrail`` into a
    callable.

    - ``None`` -> None
    - ``str`` -> :class:`StringGuardrail`
    - already a Guardrail -> returned as-is
    - any other callable -> wrapped in :class:`FunctionGuardrail` for exception safety
    """
    if g is None:
        return None
    if isinstance(g, str):
        return StringGuardrail(g, llm_client=ollama_client, model=model)
    if _is_already_guardrail(g):
        return g
    if callable(g):
        return FunctionGuardrail(g)
    return g


async def _call_guardrail(guardrail: Any, out: TaskOutput) -> GuardrailResult:
    """Invoke a guardrail - may be sync or async. Awaits coroutine returns."""
    result = guardrail(out)
    if inspect.iscoroutine(result):
        result = await result
    return result


def execute_task(
    task: CrewTask,
    *,
    context: list[TaskOutput],
    inputs: dict[str, Any] | None,
    registry: Any,
    planner: Any | None = None,
) -> TaskOutput:
    """Synchronous wrapper around :func:`execute_task_async`.

    Refuses to run from inside a running event loop — same guard as
    :meth:`cognithor.crew.crew.Crew.kickoff` — because ``asyncio.run`` cannot
    be called from an already-running loop. (See NI2 in the Round 3 review.)
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass  # no running loop — safe to use asyncio.run
    else:
        raise RuntimeError(
            "execute_task() cannot be called from a running event loop. "
            "Use `await execute_task_async(...)` instead."
        )
    return asyncio.run(
        execute_task_async(
            task,
            context=context,
            inputs=inputs,
            registry=registry,
            planner=planner,
        )
    )


def _warn_if_hierarchical_is_stubbed(process: CrewProcess) -> None:
    """PR 1 → Task 10 bridge guard.

    HIERARCHICAL currently falls through to `compiler_hierarchical.order_tasks_hierarchical`,
    which is a declaration-order stub until Task 10 lands the real manager-LLM
    routing. A user running a HIERARCHICAL crew today gets sequential semantics
    silently — warn them so the divergence from spec §1.3 is visible.
    Removed in Task 10 when the real router lands.
    """
    import warnings

    if process is CrewProcess.HIERARCHICAL:
        warnings.warn(
            "CrewProcess.HIERARCHICAL currently uses a declaration-order stub "
            "(Task 10 lands manager-LLM routing). Tasks will run in the order "
            "you declared them, not as chosen by a manager LLM. Upgrade to "
            "cognithor>=0.93.0 for real hierarchical routing.",
            UserWarning,
            # Chain: warn -> _warn_if_hierarchical_is_stubbed ->
            #        compile_and_run_{sync,async} -> kickoff{,_async} -> USER
            stacklevel=4,
        )


def compile_and_run_sync(
    agents: list[CrewAgent],
    tasks: list[CrewTask],
    process: CrewProcess,
    inputs: dict[str, Any] | None,
    registry: Any,
    planner: Any | None = None,
    manager_llm: str | None = None,
) -> CrewOutput:
    """Synchronous compiler + runner.

    Sequential: straight linear order. Hierarchical: Task 10.

    ``planner`` is optional in Task 8/9 (stub `execute_task` doesn't need it);
    Task 11 wires a real Planner and starts passing it down.
    ``manager_llm`` forwards the Crew's manager model to
    :func:`order_tasks_hierarchical`; coerced to str when the caller passes
    an ``LLMConfig`` dict.
    """
    _warn_if_hierarchical_is_stubbed(process)
    if process is CrewProcess.SEQUENTIAL:
        ordered = order_tasks_sequential(tasks)
    else:
        from cognithor.crew.compiler_hierarchical import order_tasks_hierarchical

        ordered = order_tasks_hierarchical(tasks, agents, manager_llm=manager_llm)

    trace_id = _uuid.uuid4().hex
    outputs: list[TaskOutput] = []
    append_audit(
        "crew_kickoff_started",
        trace_id=trace_id,
        n_tasks=len(ordered),
        process=process.value,
    )
    for t in ordered:
        # Note: ``execute_task`` is the sync trampoline — it asyncio.run()s
        # execute_task_async which re-derives its own trace_id if omitted.
        # Passing the kickoff-level trace_id isn't plumbed through the sync
        # wrapper because the crew always reaches here via kickoff_async now;
        # this function is kept purely for backward compat.
        append_audit(
            "crew_task_started",
            trace_id=trace_id,
            task_id=t.task_id,
            agent_role=t.agent.role,
        )
        try:
            out = execute_task(
                t,
                context=outputs,
                inputs=inputs,
                registry=registry,
                planner=planner,
            )
        except Exception as exc:
            append_audit(
                "crew_task_failed",
                trace_id=trace_id,
                task_id=t.task_id,
                reason=type(exc).__name__,
            )
            append_audit(
                "crew_kickoff_failed",
                trace_id=trace_id,
                reason=type(exc).__name__,
            )
            raise
        append_audit(
            "crew_task_completed",
            trace_id=trace_id,
            task_id=t.task_id,
            duration_ms=out.duration_ms,
            tokens=out.token_usage.get("total_tokens", 0),
        )
        outputs.append(out)
    append_audit(
        "crew_kickoff_completed",
        trace_id=trace_id,
        n_tasks=len(outputs),
    )
    return CrewOutput(raw=outputs[-1].raw, tasks_output=outputs, trace_id=trace_id)


async def execute_task_async(
    task: CrewTask,
    *,
    context: list[TaskOutput],
    inputs: dict[str, Any] | None,
    registry: Any,
    planner: Any,
    trace_id: str | None = None,
) -> TaskOutput:
    """Route one task through the Planner (which internally drives
    Gatekeeper + Executor), then run any attached guardrail with
    retry-with-feedback. Raises :class:`GuardrailFailure` after
    ``task.max_retries`` retries.

    Spec §1.6: the Crew-Layer must NOT bypass the Planner. Every task builds
    a proper ``WorkingMemory`` + ``ToolResult`` list and calls
    ``Planner.formulate_response(user_message, results, working_memory)``.

    ``trace_id`` is plumbed from the kickoff so every in-kickoff tool result /
    chat turn / audit event buckets under one audit session and concurrent
    kickoffs stay isolated. If omitted (standalone call sites), a fresh UUID
    is minted.
    """
    import time

    # Resolve tools up-front so the error is raised before any LLM call —
    # cheap local-DB check, saves a Planner round-trip on bad configs.
    resolve_tools(task.agent.tools, registry=registry)
    resolve_tools(task.tools, registry=registry)

    user_message = _build_user_message(task, inputs)

    prior_results: list[ToolResult] = [
        ToolResult(
            tool_name=f"crew_context__{prior.agent_role}",
            content=prior.raw,
            is_error=False,
        )
        for prior in context
    ]

    session_id = trace_id or _uuid.uuid4().hex
    working_memory = WorkingMemory(session_id=session_id)

    t0 = time.perf_counter()
    envelope = await planner.formulate_response(
        user_message,
        prior_results,
        working_memory,
    )
    duration_ms = (time.perf_counter() - t0) * 1000.0

    raw = getattr(envelope, "content", "") or ""
    usage = _read_token_usage(planner) or TokenUsageDict(
        prompt_tokens=0, completion_tokens=0, total_tokens=0
    )

    # Guardrail evaluation with retry-with-feedback.
    ollama_client = getattr(planner, "_ollama", None)
    guardrail_model = task.agent.llm or "ollama/qwen3:8b"
    # Only coerce str llm into guardrail_model; dict LLMConfig falls back to default.
    if not isinstance(guardrail_model, str):
        guardrail_model = "ollama/qwen3:8b"
    guardrail = _normalize_guardrail(
        task.guardrail, ollama_client=ollama_client, model=guardrail_model
    )

    attempts = 0
    verdict = "skipped"
    while True:
        out = TaskOutput(
            task_id=task.task_id,
            agent_role=task.agent.role,
            raw=raw,
            duration_ms=duration_ms,
            token_usage=usage,
        )
        if guardrail is None:
            verdict = "skipped"
            break
        result = await _call_guardrail(guardrail, out)
        append_audit(
            "crew_guardrail_check",
            trace_id=trace_id,  # parent correlation - links verdict to kickoff
            task_id=task.task_id,
            verdict="pass" if result.passed else "fail",
            retry_count=attempts,
            pii_detected=result.pii_detected,
            feedback=result.feedback,
        )
        if result.passed:
            verdict = "pass"
            break
        attempts += 1
        if attempts > task.max_retries:
            raise GuardrailFailure(
                task_id=task.task_id,
                guardrail_name=type(guardrail).__name__,
                attempts=attempts,
                reason=result.feedback or "(no feedback)",
            )
        # Retry: re-invoke Planner with a retry-nudge synthesized as an extra
        # ToolResult carrying the feedback. ``crew:retry_feedback`` prefix
        # prevents Gatekeeper / audit scanners from mistaking it for a real
        # tool call. (R4-I3)
        retry_context = [
            *prior_results,
            ToolResult(
                tool_name="crew:retry_feedback",
                content=(
                    f"Vorheriger Versuch wurde abgelehnt. "
                    f"Feedback: {result.feedback}. "
                    "Bitte erneut versuchen und die Kritik einarbeiten."
                ),
                is_error=False,
            ),
        ]
        t0 = time.perf_counter()
        envelope = await planner.formulate_response(user_message, retry_context, working_memory)
        duration_ms = (time.perf_counter() - t0) * 1000.0
        raw = getattr(envelope, "content", "") or ""
        usage = _read_token_usage(planner) or TokenUsageDict(
            prompt_tokens=0, completion_tokens=0, total_tokens=0
        )

    return out.model_copy(update={"guardrail_verdict": verdict})


async def compile_and_run_async(
    agents: list[CrewAgent],
    tasks: list[CrewTask],
    process: CrewProcess,
    inputs: dict[str, Any] | None,
    registry: Any,
    planner: Any | None = None,
    manager_llm: str | None = None,
) -> CrewOutput:
    """Async compiler + runner with parallel fan-out for async_execution=True tasks.

    Consecutive tasks marked `async_execution=True` that don't depend on each
    other are gathered and run concurrently via `asyncio.gather`. Everything
    else falls back to sequential await.

    ``planner`` is optional in Task 8/9 (stub `execute_task_async` doesn't need
    it); Task 11 wires a real Planner and starts passing it down. Keeping the
    default here prevents Task 11 from cascading a breaking signature change
    across all fan-out call sites.
    ``manager_llm`` forwards the Crew's manager model to
    :func:`order_tasks_hierarchical`.
    """
    _warn_if_hierarchical_is_stubbed(process)
    if process is CrewProcess.SEQUENTIAL:
        ordered = order_tasks_sequential(tasks)
    else:
        from cognithor.crew.compiler_hierarchical import order_tasks_hierarchical

        ordered = order_tasks_hierarchical(tasks, agents, manager_llm=manager_llm)

    trace_id = _uuid.uuid4().hex
    outputs: list[TaskOutput] = []
    append_audit(
        "crew_kickoff_started",
        trace_id=trace_id,
        n_tasks=len(ordered),
        process=process.value,
    )
    i = 0
    while i < len(ordered):
        # Collect a fan-out group: consecutive tasks with async_execution=True
        # and no dependency on each other. The anchor itself must also be
        # async_execution=True — otherwise a sync task would be swept into the
        # gather pool and run in parallel with its neighbours.
        group = [ordered[i]]
        j = i + 1
        if ordered[i].async_execution:
            while j < len(ordered) and ordered[j].async_execution:
                # Only group if the later task doesn't depend on earlier group members
                deps = {t.task_id for t in ordered[j].context}
                if deps.isdisjoint({t.task_id for t in group}):
                    group.append(ordered[j])
                    j += 1
                else:
                    break
        if len(group) == 1:
            append_audit(
                "crew_task_started",
                trace_id=trace_id,
                task_id=group[0].task_id,
                agent_role=group[0].agent.role,
            )
            try:
                out = await execute_task_async(
                    group[0],
                    context=outputs,
                    inputs=inputs,
                    registry=registry,
                    planner=planner,
                    trace_id=trace_id,
                )
            except Exception as exc:
                # Audit-chain integrity: emit failure events so every
                # crew_task_started has a matching terminal event.
                append_audit(
                    "crew_task_failed",
                    trace_id=trace_id,
                    task_id=group[0].task_id,
                    reason=type(exc).__name__,
                )
                append_audit(
                    "crew_kickoff_failed",
                    trace_id=trace_id,
                    reason=type(exc).__name__,
                )
                raise
            append_audit(
                "crew_task_completed",
                trace_id=trace_id,
                task_id=group[0].task_id,
                duration_ms=out.duration_ms,
                tokens=out.token_usage.get("total_tokens", 0),
            )
            outputs.append(out)
        else:
            for t in group:
                append_audit(
                    "crew_task_started",
                    trace_id=trace_id,
                    task_id=t.task_id,
                    agent_role=t.agent.role,
                )
            try:
                parallel_outs = await asyncio.gather(
                    *[
                        execute_task_async(
                            t,
                            context=outputs,
                            inputs=inputs,
                            registry=registry,
                            planner=planner,
                            trace_id=trace_id,
                        )
                        for t in group
                    ]
                )
            except Exception as exc:
                # Any failure in the fan-out cancels the gather. We don't
                # know which individual task(s) failed from here, so emit a
                # terminal event for every started task + the kickoff to keep
                # the Hashline-Guard chain balanced.
                for t in group:
                    append_audit(
                        "crew_task_failed",
                        trace_id=trace_id,
                        task_id=t.task_id,
                        reason=type(exc).__name__,
                    )
                append_audit(
                    "crew_kickoff_failed",
                    trace_id=trace_id,
                    reason=type(exc).__name__,
                )
                raise
            for t, out in zip(group, parallel_outs, strict=True):
                append_audit(
                    "crew_task_completed",
                    trace_id=trace_id,
                    task_id=t.task_id,
                    duration_ms=out.duration_ms,
                    tokens=out.token_usage.get("total_tokens", 0),
                )
            outputs.extend(parallel_outs)
        i = j if len(group) > 1 else i + 1
    append_audit(
        "crew_kickoff_completed",
        trace_id=trace_id,
        n_tasks=len(outputs),
    )
    return CrewOutput(raw=outputs[-1].raw, tasks_output=outputs, trace_id=trace_id)
