"""Compiler translates Crew definitions into ordered execution steps that
route through the existing Planner/Gatekeeper pipeline.

The compiler itself is a pure function; the `execute_task` helper is where
the actual PGE integration happens (Task 11). For the happy path in Task 8
we only need ordered traversal.
"""

from __future__ import annotations

import asyncio
import uuid as _uuid
from typing import Any

from cognithor.crew.agent import CrewAgent
from cognithor.crew.output import CrewOutput, TaskOutput
from cognithor.crew.process import CrewProcess
from cognithor.crew.task import CrewTask


def order_tasks_sequential(tasks: list[CrewTask]) -> list[CrewTask]:
    """Sequential process: keep the declaration order."""
    return list(tasks)


def execute_task(
    task: CrewTask,
    *,
    context: list[TaskOutput],
    inputs: dict[str, Any] | None,
    registry: Any,
    planner: Any | None = None,
) -> TaskOutput:
    """Route one task through the PGE pipeline.

    Stub for Task 8 — the real PGE wiring lands in Task 11. The stub raises
    NotImplementedError so that the unit test at Task 8 is forced to patch
    this function (the test does). Integration happens in Task 11 where the
    patch target becomes a real call site."""
    raise NotImplementedError(
        "execute_task is stubbed in Task 8; real PGE wiring arrives in Task 11. "
        "Tests must patch 'cognithor.crew.compiler.execute_task' until then."
    )


# Guardrails land in Feature 4 (PR 2). Between PR 1 (this file) shipping and
# PR 2 landing on the user's install, a CrewTask with `guardrail=<anything>`
# would silently do nothing — the user gets no safety they expected. Guard
# against that foot-gun by probing the guardrails module at import time and
# emitting a UserWarning if a task declares a guardrail on a version that
# can't execute it. Removed in Task 21 when the real apply path lands.
try:
    from cognithor.crew.guardrails import base as _guardrails_base  # noqa: F401

    _guardrails_available = True
except ImportError:
    _guardrails_available = False


def _warn_if_guardrail_silently_ignored(task: CrewTask) -> None:
    """PR 1 → PR 2 bridge guard. Removed in Task 21."""
    import warnings

    if task.guardrail is not None and not _guardrails_available:
        warnings.warn(
            f"CrewTask '{task.task_id}' has a guardrail but "
            "cognithor.crew.guardrails is not available in this release. "
            "The guardrail will be IGNORED. Upgrade to cognithor>=0.93.0 "
            "(or install via `pip install cognithor[all]`) to enable guardrails.",
            UserWarning,
            # Chain: warn -> _warn_if_guardrail_silently_ignored ->
            #        compile_and_run_sync -> Crew.kickoff -> USER
            stacklevel=4,
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
) -> CrewOutput:
    """Synchronous compiler + runner.

    Sequential: straight linear order. Hierarchical: Task 10.

    ``planner`` is optional in Task 8/9 (stub `execute_task` doesn't need it);
    Task 11 wires a real Planner and starts passing it down.
    """
    _warn_if_hierarchical_is_stubbed(process)
    if process is CrewProcess.SEQUENTIAL:
        ordered = order_tasks_sequential(tasks)
    else:
        from cognithor.crew.compiler_hierarchical import order_tasks_hierarchical

        ordered = order_tasks_hierarchical(tasks, agents)

    trace_id = _uuid.uuid4().hex
    outputs: list[TaskOutput] = []
    for t in ordered:
        _warn_if_guardrail_silently_ignored(t)  # PR 1 → PR 2 bridge guard
        out = execute_task(
            t,
            context=outputs,
            inputs=inputs,
            registry=registry,
            planner=planner,
        )
        outputs.append(out)
    return CrewOutput(raw=outputs[-1].raw, tasks_output=outputs, trace_id=trace_id)


async def execute_task_async(
    task: CrewTask,
    *,
    context: list[TaskOutput],
    inputs: dict[str, Any] | None,
    registry: Any,
    planner: Any | None = None,
) -> TaskOutput:
    """Async counterpart of execute_task. Real PGE wiring in Task 11."""
    raise NotImplementedError(
        "execute_task_async is stubbed in Task 9; real PGE wiring arrives in Task 11."
    )


async def compile_and_run_async(
    agents: list[CrewAgent],
    tasks: list[CrewTask],
    process: CrewProcess,
    inputs: dict[str, Any] | None,
    registry: Any,
    planner: Any | None = None,
) -> CrewOutput:
    """Async compiler + runner with parallel fan-out for async_execution=True tasks.

    Consecutive tasks marked `async_execution=True` that don't depend on each
    other are gathered and run concurrently via `asyncio.gather`. Everything
    else falls back to sequential await.

    ``planner`` is optional in Task 8/9 (stub `execute_task_async` doesn't need
    it); Task 11 wires a real Planner and starts passing it down. Keeping the
    default here prevents Task 11 from cascading a breaking signature change
    across all fan-out call sites.
    """
    _warn_if_hierarchical_is_stubbed(process)
    if process is CrewProcess.SEQUENTIAL:
        ordered = order_tasks_sequential(tasks)
    else:
        from cognithor.crew.compiler_hierarchical import order_tasks_hierarchical

        ordered = order_tasks_hierarchical(tasks, agents)

    trace_id = _uuid.uuid4().hex
    outputs: list[TaskOutput] = []
    # PR 1 → PR 2 bridge: warn on any silently-ignored guardrail before entering
    # the fan-out loop (single pass; warnings filter dedupes by call site).
    for t in ordered:
        _warn_if_guardrail_silently_ignored(t)
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
            out = await execute_task_async(
                group[0],
                context=outputs,
                inputs=inputs,
                registry=registry,
                planner=planner,
            )
            outputs.append(out)
        else:
            parallel_outs = await asyncio.gather(
                *[
                    execute_task_async(
                        t,
                        context=outputs,
                        inputs=inputs,
                        registry=registry,
                        planner=planner,
                    )
                    for t in group
                ]
            )
            outputs.extend(parallel_outs)
        i = j if len(group) > 1 else i + 1
    return CrewOutput(raw=outputs[-1].raw, tasks_output=outputs, trace_id=trace_id)
