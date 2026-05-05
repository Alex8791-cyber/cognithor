"""Agent-to-agent delegation — extracted from Gateway.

Runs a delegated task on behalf of one agent inside the persona,
workspace, sandbox, and tool restrictions of a different agent. The
result text flows back to the calling agent as a string. Forks a
provenance-tracked sub-session and uses a fresh ``WorkingMemory`` so
the delegated agent's context does not leak into the parent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from cognithor.models import (
    GateStatus,
    Message,
    MessageRole,
    SessionContext,
    WorkingMemory,
)
from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.gateway.gateway import Gateway

log = get_logger(__name__)


async def execute_delegation(
    gw: Gateway,
    from_agent: str,
    to_agent: str,
    task: str,
    session: SessionContext,
    parent_wm: WorkingMemory,
) -> str:
    """Execute a true agent-to-agent delegation.

    The delegated agent runs with:
      - its own system prompt
      - its own (isolated) workspace
      - its own sandbox config
      - its own tool restrictions
      - the task as the user message

    Args:
        gw: Gateway holding the agent_router, planner, executor, etc.
        from_agent: Delegating agent's name.
        to_agent: Target agent's name.
        task: The delegated task text.
        session: Current session.
        parent_wm: Parent agent's working memory (provenance only).

    Returns:
        The delegated agent's response text. On failure / block /
        absent router, returns a short German status message instead.
    """
    if not gw._agent_router:
        return f"Agent router unavailable. Delegation to {to_agent} failed."

    # Build + validate delegation
    delegation = gw._agent_router.create_delegation(from_agent, to_agent, task)
    if delegation is None:
        return (
            f"Delegation from {from_agent} to {to_agent} not allowed. I'll handle the task myself."
        )

    target = delegation.target_profile
    if not target:
        return f"Agent {to_agent} not found."

    log.info(
        "delegation_executing",
        from_=from_agent,
        to=to_agent,
        task=task[:200],
        depth=delegation.depth,
    )

    # Broadcast delegation status to frontend
    try:
        channel_name = session.channel or "webui"
        status_cb = gw._make_status_callback(channel_name, session.session_id)
        await status_cb(
            "working",
            f"Delegation: {from_agent} -> {to_agent}: {task[:100]}",
        )
    except Exception:
        log.debug("delegation_status_broadcast_failed", exc_info=True)

    # Forked session for provenance tracking
    sub_session = SessionContext(
        user_id=session.user_id,
        channel=session.channel,
        agent_name=to_agent,
        parent_session_id=session.session_id,
        fork_reason=f"delegated from {from_agent}: {task[:200]}",
    )
    if gw._session_store:
        try:
            gw._session_store.save_session(sub_session)
        except Exception:
            log.debug("delegation_session_save_skipped", exc_info=True)

    # Separate working memory for the delegated agent
    sub_wm = WorkingMemory(session_id=sub_session.session_id)

    # Inject target agent's system prompt
    if target.system_prompt:
        sub_wm.add_message(
            Message(
                role=MessageRole.SYSTEM,
                content=target.system_prompt,
            )
        )

    # Task as user message
    sub_wm.add_message(
        Message(
            role=MessageRole.USER,
            content=task,
        )
    )

    # Resolve target agent's workspace
    target_workspace = gw._agent_router.resolve_agent_workspace(
        to_agent,
        gw._config.workspace_dir,
    )

    # Filter tool schemas for target agent
    tool_schemas = gw._mcp_client.get_tool_schemas() if gw._mcp_client else {}
    if target.has_tool_restrictions:
        tool_schemas = target.filter_tools(tool_schemas)

    if gw._planner is None:
        raise RuntimeError("Planner nicht initialisiert -- Delegation nicht möglich")

    # Agent-specific LLM overrides for the delegation target
    _del_model = target.preferred_model or None
    _del_temp = target.temperature
    _del_top_p = getattr(target, "top_p", None)

    plan = await gw._planner.plan(
        user_message=task,
        working_memory=sub_wm,
        tool_schemas=tool_schemas,
        model_override=_del_model,
        temperature_override=_del_temp,
        top_p_override=_del_top_p,
    )

    # Direct response?
    if not plan.has_actions and plan.direct_response:
        delegation.result = plan.direct_response
        delegation.success = True
        return cast("str", plan.direct_response)

    if not plan.has_actions:
        no_plan_msg = "Kein Plan erstellt."
        delegation.result = no_plan_msg
        delegation.success = False
        return no_plan_msg

    # Gatekeeper
    if gw._gatekeeper is None:
        raise RuntimeError("Gatekeeper nicht initialisiert -- Delegation nicht möglich")
    decisions = gw._gatekeeper.evaluate_plan(plan.steps, session)

    # APPROVE/BLOCK decisions are not actionable in delegations (no HITL)
    blocked = [d for d in decisions if d.status in (GateStatus.APPROVE, GateStatus.BLOCK)]
    if blocked:
        reasons = "; ".join(d.reason for d in blocked[:3])
        blocked_msg = f"Delegation blockiert: {reasons}"
        delegation.result = blocked_msg
        delegation.success = False
        return blocked_msg

    # Executor with target agent's context
    assert gw._executor is not None
    gw._executor.set_agent_context(
        workspace_dir=str(target_workspace),
        sandbox_overrides=target.get_sandbox_config(),
        agent_name=target.name,
        session_id=session.session_id,
    )

    try:
        results = await gw._executor.execute(plan.steps, decisions)
    finally:
        gw._executor.clear_agent_context()

    # Formulate result
    if any(r.success for r in results):
        _envelope = await gw._planner.formulate_response(
            user_message=task,
            results=results,
            working_memory=sub_wm,
        )
        response = _envelope.content
        delegation.result = response
        delegation.success = True
    else:
        delegation.result = "Delegation failed: no successful actions."
        delegation.success = False

    log.info(
        "delegation_complete",
        from_=from_agent,
        to=to_agent,
        success=delegation.success,
        result_len=len(delegation.result or ""),
    )

    return delegation.result or ""
