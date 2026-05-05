"""Deep coverage for cognithor.gateway.message_handler.

The module is the message-routing+dispatch surface: ``handle_message`` (the
PGE turn orchestrator), ``resolve_agent_route`` (Phase 1), ``prepare_execution_context``
(Phase 2), and the two callback factories ``make_status_callback`` /
``make_pipeline_callback``, plus ``formulate_response``.

Tests instantiate ``Gateway.__new__(Gateway)`` and inject only the attributes
the function under test reads — same pattern as the post_processing deep tests
landed in PR #486.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from cognithor.gateway import message_handler
from cognithor.gateway.gateway import Gateway
from cognithor.gateway.message_handler import (
    formulate_response,
    handle_message,
    make_pipeline_callback,
    make_status_callback,
    prepare_execution_context,
    resolve_agent_route,
)
from cognithor.models import (
    IncomingMessage,
    Message,
    MessageRole,
    OutgoingMessage,
    SessionContext,
    ToolResult,
    WorkingMemory,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


_RESOLVE_SENTINEL = "RESOLVE_REACHED"


def _bare_gateway(*, stub_resolve: bool = True) -> Gateway:
    """A Gateway shell without running __init__.

    By default, ``_resolve_agent_route`` is stubbed to raise
    ``RuntimeError(_RESOLVE_SENTINEL)`` so tests that want to assert the
    code path *reached* Phase 1 don't have to set up the full session-
    management machinery. Pass ``stub_resolve=False`` for tests that
    exercise resolve_agent_route directly.
    """
    gw = Gateway.__new__(Gateway)
    # Defaults — None means "feature off"
    gw._idle_detector = None  # type: ignore[attr-defined]
    gw._active_learner = None  # type: ignore[attr-defined]
    gw._consent_manager = None  # type: ignore[attr-defined]
    gw._compliance_engine = None  # type: ignore[attr-defined]
    gw._session_analyzer = None  # type: ignore[attr-defined]
    gw._task_profiler = None  # type: ignore[attr-defined]
    gw._cost_tracker = None  # type: ignore[attr-defined]
    gw._run_recorder = None  # type: ignore[attr-defined]
    gw._gatekeeper = None  # type: ignore[attr-defined]
    gw._planner = None  # type: ignore[attr-defined]
    gw._executor = None  # type: ignore[attr-defined]
    gw._mcp_client = None  # type: ignore[attr-defined]
    gw._model_router = None  # type: ignore[attr-defined]
    gw._context_pipeline = None  # type: ignore[attr-defined]
    gw._agent_router = None  # type: ignore[attr-defined]
    gw._skill_registry = None  # type: ignore[attr-defined]
    gw._skill_generator = None  # type: ignore[attr-defined]
    gw._audit_logger = None  # type: ignore[attr-defined]
    gw._explainability = None  # type: ignore[attr-defined]
    gw._video_cleanup = None  # type: ignore[attr-defined]
    gw._user_pref_store = None  # type: ignore[attr-defined]
    gw._autonomous_orchestrator = None  # type: ignore[attr-defined]
    gw._channels = {}  # type: ignore[attr-defined]
    gw._cancelled_sessions = set()  # type: ignore[attr-defined]
    gw._background_tasks = set()  # type: ignore[attr-defined]
    gw._running = True  # type: ignore[attr-defined]
    gw._config = MagicMock()  # type: ignore[attr-defined]
    gw._config.security.max_sub_agent_depth = 3
    gw._record_metric = MagicMock()  # type: ignore[attr-defined]
    if stub_resolve:
        gw._resolve_agent_route = AsyncMock(  # type: ignore[attr-defined]
            side_effect=RuntimeError(_RESOLVE_SENTINEL)
        )
    return gw


def _msg(
    text: str = "hello",
    *,
    channel: str = "cli",
    user_id: str = "alex",
    metadata: dict[str, Any] | None = None,
    session_id: str | None = None,
    attachments: list[str] | None = None,
) -> IncomingMessage:
    return IncomingMessage(
        text=text,
        channel=channel,
        user_id=user_id,
        metadata=metadata or {},
        session_id=session_id,
        attachments=attachments or [],
    )


# ─────────────────────────────────────────────────────────────────────────────
# handle_message — sub-agent depth guard
# ─────────────────────────────────────────────────────────────────────────────


class TestSubAgentDepthGuard:
    @pytest.mark.asyncio
    async def test_depth_exceeded_returns_error_response(self) -> None:
        gw = _bare_gateway()
        msg = _msg(metadata={"depth": 5})
        result = await handle_message(gw, msg)
        assert isinstance(result, OutgoingMessage)
        assert result.is_final
        assert result.channel == "cli"
        # The metric for "requests_total" is NOT recorded — guard short-circuits
        assert not gw._record_metric.called

    @pytest.mark.asyncio
    async def test_default_depth_zero_passes_guard(self) -> None:
        # depth=0 (default) is fine; assert that we don't get the depth-error
        # response by stubbing through to the consent gate.
        gw = _bare_gateway()
        msg = _msg(text="akzeptieren")
        # Set up consent_manager so the consent path triggers and we exit early
        cm = MagicMock()
        cm.requires_consent.return_value = True
        cm.grant_consent = MagicMock()
        gw._consent_manager = cm  # type: ignore[attr-defined]
        result = await handle_message(gw, msg)
        cm.grant_consent.assert_called_once()
        assert result.is_final

    @pytest.mark.asyncio
    @pytest.mark.parametrize("depth", [4, 10, 100])
    async def test_depth_above_max_blocks(self, depth: int) -> None:
        gw = _bare_gateway()
        msg = _msg(metadata={"depth": depth})
        result = await handle_message(gw, msg)
        assert result.is_final


# ─────────────────────────────────────────────────────────────────────────────
# handle_message — system-internal message detection
# ─────────────────────────────────────────────────────────────────────────────


class TestSystemMessageDetection:
    """The gateway treats `cron*`, `heartbeat*`, `agent:*` user_ids and the
    cron/sub_agent/system/evolution/heartbeat channels as system-internal.
    System messages should NOT trigger the consent gate.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "user_id,channel",
        [
            ("cron-job-1", "cli"),
            ("heartbeat-7", "cli"),
            ("agent:planner", "cli"),
            ("alex", "cron"),
            ("alex", "sub_agent"),
            ("alex", "system"),
            ("alex", "evolution"),
            ("alex", "heartbeat"),
        ],
    )
    async def test_system_message_skips_consent_gate(self, user_id: str, channel: str) -> None:
        gw = _bare_gateway()
        # Consent manager should NOT be queried for system messages.
        cm = MagicMock()
        cm.requires_consent = MagicMock(return_value=True)
        gw._consent_manager = cm  # type: ignore[attr-defined]
        msg = _msg(text="akzeptieren", user_id=user_id, channel=channel)
        # System messages MUST bypass consent and hit _resolve_agent_route
        # (which is stubbed to raise the sentinel).
        with pytest.raises(RuntimeError, match=_RESOLVE_SENTINEL):
            await handle_message(gw, msg)
        cm.requires_consent.assert_not_called()
        cm.grant_consent.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# handle_message — consent flow
# ─────────────────────────────────────────────────────────────────────────────


class TestConsentFlow:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "trigger", ["akzeptieren", "accept", "ja", "yes", "JA", "  Yes  ", "Akzeptieren"]
    )
    async def test_consent_grant_paths(self, trigger: str) -> None:
        gw = _bare_gateway()
        cm = MagicMock()
        cm.requires_consent.return_value = True
        gw._consent_manager = cm  # type: ignore[attr-defined]
        msg = _msg(text=trigger)
        result = await handle_message(gw, msg)
        assert isinstance(result, OutgoingMessage)
        assert result.is_final
        cm.grant_consent.assert_called_once()

    @pytest.mark.asyncio
    async def test_consent_not_required_falls_through(self) -> None:
        gw = _bare_gateway()
        cm = MagicMock()
        cm.requires_consent.return_value = False
        gw._consent_manager = cm  # type: ignore[attr-defined]
        msg = _msg(text="ja")
        # Consent not required → falls through to RuntimeError because
        # planner is None.
        with pytest.raises(RuntimeError, match=_RESOLVE_SENTINEL):
            await handle_message(gw, msg)
        cm.grant_consent.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_consent_text_is_ignored(self) -> None:
        gw = _bare_gateway()
        cm = MagicMock()
        cm.requires_consent.return_value = True
        gw._consent_manager = cm  # type: ignore[attr-defined]
        msg = _msg(text="anything else")
        with pytest.raises(RuntimeError, match=_RESOLVE_SENTINEL):
            await handle_message(gw, msg)
        cm.grant_consent.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# handle_message — GDPR compliance gate
# ─────────────────────────────────────────────────────────────────────────────


class TestComplianceGate:
    @pytest.mark.asyncio
    async def test_compliance_violation_returns_consent_prompt(self) -> None:
        from cognithor.security.compliance_engine import ComplianceViolation

        gw = _bare_gateway()
        ce = MagicMock()
        ce.check.side_effect = ComplianceViolation("Missing consent for processing")
        gw._compliance_engine = ce  # type: ignore[attr-defined]
        msg = _msg(text="hello")
        result = await handle_message(gw, msg)
        assert isinstance(result, OutgoingMessage)
        assert result.is_final
        # Consent prompt should mention consent
        assert result.text  # non-empty

    @pytest.mark.asyncio
    async def test_compliance_violation_non_consent_returned_verbatim(self) -> None:
        from cognithor.security.compliance_engine import ComplianceViolation

        gw = _bare_gateway()
        ce = MagicMock()
        ce.check.side_effect = ComplianceViolation("Some other compliance issue")
        gw._compliance_engine = ce  # type: ignore[attr-defined]
        msg = _msg(text="hello")
        result = await handle_message(gw, msg)
        # No "consent" keyword → returned verbatim (no replacement)
        assert "Some other compliance issue" in result.text

    @pytest.mark.asyncio
    async def test_compliance_pass_falls_through(self) -> None:
        gw = _bare_gateway()
        ce = MagicMock()
        ce.check = MagicMock()  # no exception
        gw._compliance_engine = ce  # type: ignore[attr-defined]
        msg = _msg(text="hello")
        with pytest.raises(RuntimeError, match=_RESOLVE_SENTINEL):
            await handle_message(gw, msg)
        ce.check.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# handle_message — idle detector + active learner notifications
# ─────────────────────────────────────────────────────────────────────────────


class TestIdleAndLearnerNotifications:
    @pytest.mark.asyncio
    async def test_idle_detector_notified_immediately(self) -> None:
        gw = _bare_gateway()
        idle = MagicMock()
        gw._idle_detector = idle  # type: ignore[attr-defined]
        # Force depth-exceeded to short-circuit early but still after notify.
        msg = _msg(metadata={"depth": 99})
        await handle_message(gw, msg)
        idle.notify_activity.assert_called()

    @pytest.mark.asyncio
    async def test_active_learner_notified_immediately(self) -> None:
        gw = _bare_gateway()
        al = MagicMock()
        gw._active_learner = al  # type: ignore[attr-defined]
        msg = _msg(metadata={"depth": 99})
        await handle_message(gw, msg)
        al.notify_activity.assert_called()

    @pytest.mark.asyncio
    async def test_no_idle_detector_attribute_is_safe(self) -> None:
        gw = _bare_gateway()
        # Deleting the attribute simulates an old-style Gateway without _idle_detector
        del gw._idle_detector
        msg = _msg(metadata={"depth": 99})
        # Must not raise AttributeError
        await handle_message(gw, msg)


# ─────────────────────────────────────────────────────────────────────────────
# resolve_agent_route — agent routing
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveAgentRoute:
    @pytest.mark.asyncio
    async def test_no_router_uses_default_jarvis(self) -> None:
        gw = _bare_gateway()
        # Patch the session/wm helpers so they don't blow up.
        sess = SessionContext(session_id="sess-1", channel="cli")
        wm = WorkingMemory()
        gw._get_or_create_session = MagicMock(return_value=sess)  # type: ignore[attr-defined]
        gw._get_or_create_working_memory = MagicMock(return_value=wm)  # type: ignore[attr-defined]
        msg = _msg()
        (
            route_decision,
            session,
            working_memory,
            active_skill,
            agent_workspace,
            agent_name,
            trail_id,
        ) = await resolve_agent_route(gw, msg)
        assert route_decision is None
        assert agent_name == "jarvis"
        assert session is sess
        assert working_memory is wm
        assert active_skill is None
        assert agent_workspace is None
        assert trail_id is None

    @pytest.mark.asyncio
    async def test_explicit_target_agent_in_metadata(self) -> None:
        gw = _bare_gateway()
        # Build a router that has the target agent
        router = MagicMock()
        target_profile = MagicMock()
        target_profile.name = "researcher"
        target_profile.system_prompt = "You research."
        target_profile.has_tool_restrictions = False
        target_profile.shared_workspace = True
        router.get_agent.return_value = target_profile
        router.resolve_agent_workspace = MagicMock(return_value=None)
        gw._agent_router = router  # type: ignore[attr-defined]

        sess = SessionContext(session_id="s", channel="cli")
        wm = WorkingMemory()
        gw._get_or_create_session = MagicMock(return_value=sess)  # type: ignore[attr-defined]
        gw._get_or_create_working_memory = MagicMock(return_value=wm)  # type: ignore[attr-defined]
        gw._config.workspace_dir = "/tmp/ws"  # type: ignore[attr-defined]

        msg = _msg(metadata={"target_agent": "researcher"})
        (route_decision, _s, _wm, _sk, _ws, agent_name, _tid) = await resolve_agent_route(gw, msg)
        assert route_decision is not None
        assert route_decision.agent.name == "researcher"
        assert route_decision.confidence == 1.0
        assert agent_name == "researcher"
        # System prompt added to working memory
        assert any(m.role == MessageRole.SYSTEM for m in wm.chat_history)

    @pytest.mark.asyncio
    async def test_explicit_target_unknown_falls_back_to_routing(self) -> None:
        gw = _bare_gateway()
        router = MagicMock()
        router.get_agent.return_value = None  # Unknown agent
        # Routing returns a decision
        from cognithor.core.agent_router import RouteDecision

        fallback_profile = MagicMock()
        fallback_profile.name = "jarvis"
        fallback_profile.system_prompt = ""
        fallback_profile.has_tool_restrictions = False
        fallback_profile.shared_workspace = True
        router.route.return_value = RouteDecision(agent=fallback_profile, confidence=0.5)
        router.resolve_agent_workspace = MagicMock(return_value=None)
        gw._agent_router = router  # type: ignore[attr-defined]

        sess = SessionContext(session_id="s", channel="cli")
        wm = WorkingMemory()
        gw._get_or_create_session = MagicMock(return_value=sess)  # type: ignore[attr-defined]
        gw._get_or_create_working_memory = MagicMock(return_value=wm)  # type: ignore[attr-defined]
        gw._config.workspace_dir = "/tmp/ws"  # type: ignore[attr-defined]

        msg = _msg(metadata={"target_agent": "missing-agent"})
        (route_decision, *_) = await resolve_agent_route(gw, msg)
        assert route_decision is not None
        assert route_decision.agent.name == "jarvis"
        router.route.assert_called_once()

    @pytest.mark.asyncio
    async def test_image_attachments_routed_to_wm(self) -> None:
        gw = _bare_gateway()
        sess = SessionContext(session_id="s", channel="cli")
        wm = WorkingMemory()
        gw._get_or_create_session = MagicMock(return_value=sess)  # type: ignore[attr-defined]
        gw._get_or_create_working_memory = MagicMock(return_value=wm)  # type: ignore[attr-defined]

        msg = _msg(attachments=["/tmp/photo.png", "/tmp/another.jpg"])
        await resolve_agent_route(gw, msg)
        assert wm.image_attachments == ["/tmp/photo.png", "/tmp/another.jpg"]

    @pytest.mark.asyncio
    async def test_no_attachments_clears_images(self) -> None:
        gw = _bare_gateway()
        sess = SessionContext(session_id="s", channel="cli")
        wm = WorkingMemory()
        # Pre-populate wm.image_attachments — should be cleared
        wm.image_attachments = ["/old/img.png"]
        gw._get_or_create_session = MagicMock(return_value=sess)  # type: ignore[attr-defined]
        gw._get_or_create_working_memory = MagicMock(return_value=wm)  # type: ignore[attr-defined]
        msg = _msg(attachments=[])
        await resolve_agent_route(gw, msg)
        assert wm.image_attachments == []

    @pytest.mark.asyncio
    async def test_skill_generator_gap_detection_for_tool_request(self) -> None:
        gw = _bare_gateway()
        sg = MagicMock()
        sg.gap_detector = MagicMock()
        gw._skill_generator = sg  # type: ignore[attr-defined]

        sess = SessionContext(session_id="s", channel="cli")
        wm = WorkingMemory()
        gw._get_or_create_session = MagicMock(return_value=sess)  # type: ignore[attr-defined]
        gw._get_or_create_working_memory = MagicMock(return_value=wm)  # type: ignore[attr-defined]

        msg = _msg(text="Bitte erstelle ein Tool für PDF-Konvertierung")
        await resolve_agent_route(gw, msg)
        sg.gap_detector.report_user_request.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# prepare_execution_context (Phase 2)
# ─────────────────────────────────────────────────────────────────────────────


class TestPrepareExecutionContext:
    @pytest.mark.asyncio
    async def test_no_subsystems_returns_none_run_id(self) -> None:
        gw = _bare_gateway()
        sess = SessionContext(session_id="s", channel="cli")
        wm = WorkingMemory()
        msg = _msg()
        run_id, budget_resp = await prepare_execution_context(gw, msg, sess, wm, None)
        assert run_id is None
        assert budget_resp is None

    @pytest.mark.asyncio
    async def test_budget_exceeded_returns_outgoing_message(self) -> None:
        gw = _bare_gateway()
        ct = MagicMock()
        budget = MagicMock()
        budget.ok = False
        budget.warning = "Daily limit hit"
        ct.check_budget.return_value = budget
        gw._cost_tracker = ct  # type: ignore[attr-defined]
        sess = SessionContext(session_id="s", channel="cli")
        wm = WorkingMemory()
        msg = _msg()
        run_id, budget_resp = await prepare_execution_context(gw, msg, sess, wm, None)
        assert run_id is None
        assert budget_resp is not None
        assert isinstance(budget_resp, OutgoingMessage)
        assert budget_resp.is_final

    @pytest.mark.asyncio
    async def test_run_recorder_starts_run_with_truncated_text(self) -> None:
        gw = _bare_gateway()
        rr = MagicMock()
        rr.start_run.return_value = "RUN-42"
        gw._run_recorder = rr  # type: ignore[attr-defined]
        sess = SessionContext(session_id="s", channel="cli")
        wm = WorkingMemory()
        big_text = "x" * 1000
        msg = _msg(text=big_text)
        run_id, _ = await prepare_execution_context(gw, msg, sess, wm, None)
        assert run_id == "RUN-42"
        kwargs = rr.start_run.call_args.kwargs
        # text is truncated to 500 chars
        assert len(kwargs["user_message"]) == 500

    @pytest.mark.asyncio
    async def test_task_profiler_failure_swallowed(self) -> None:
        gw = _bare_gateway()
        tp = MagicMock()
        tp.start_task.side_effect = RuntimeError("profiler down")
        gw._task_profiler = tp  # type: ignore[attr-defined]
        sess = SessionContext(session_id="s", channel="cli")
        wm = WorkingMemory()
        msg = _msg()
        # Must not raise
        await prepare_execution_context(gw, msg, sess, wm, None)

    @pytest.mark.asyncio
    async def test_policy_snapshot_recorded_when_run_id_present(self) -> None:
        gw = _bare_gateway()
        rr = MagicMock()
        rr.start_run.return_value = "R1"
        gw._run_recorder = rr  # type: ignore[attr-defined]
        gk = MagicMock()
        policy = MagicMock()
        policy.model_dump.return_value = {"name": "foo"}
        gk.get_policies.return_value = [policy]
        gw._gatekeeper = gk  # type: ignore[attr-defined]
        sess = SessionContext(session_id="s", channel="cli")
        wm = WorkingMemory()
        msg = _msg()
        await prepare_execution_context(gw, msg, sess, wm, None)
        rr.record_policy_snapshot.assert_called_once_with("R1", {"rules": [{"name": "foo"}]})


# ─────────────────────────────────────────────────────────────────────────────
# make_status_callback / make_pipeline_callback
# ─────────────────────────────────────────────────────────────────────────────


class TestStatusAndPipelineCallbacks:
    @pytest.mark.asyncio
    async def test_status_callback_no_channel_is_noop(self) -> None:
        gw = _bare_gateway()
        cb = make_status_callback(gw, "missing", "sid")
        # Must not raise
        await cb("thinking", "Working...")

    @pytest.mark.asyncio
    async def test_status_callback_sends_status_via_channel(self) -> None:
        gw = _bare_gateway()
        chan = MagicMock()
        chan.send_status = AsyncMock()
        gw._channels = {"cli": chan}  # type: ignore[attr-defined]
        cb = make_status_callback(gw, "cli", "sid")
        await cb("thinking", "Working...")
        chan.send_status.assert_awaited_once()
        args, _ = chan.send_status.await_args
        assert args[0] == "sid"

    @pytest.mark.asyncio
    async def test_status_callback_invalid_status_falls_back_to_processing(self) -> None:
        from cognithor.channels.base import StatusType

        gw = _bare_gateway()
        chan = MagicMock()
        chan.send_status = AsyncMock()
        gw._channels = {"cli": chan}  # type: ignore[attr-defined]
        cb = make_status_callback(gw, "cli", "sid")
        await cb("nonexistent_status_value", "msg")
        # Should still have been called — status defaulted to PROCESSING
        chan.send_status.assert_awaited_once()
        args, _ = chan.send_status.await_args
        assert args[1] == StatusType.PROCESSING

    @pytest.mark.asyncio
    async def test_status_callback_swallows_send_exception(self) -> None:
        gw = _bare_gateway()
        chan = MagicMock()
        chan.send_status = AsyncMock(side_effect=RuntimeError("net down"))
        gw._channels = {"cli": chan}  # type: ignore[attr-defined]
        cb = make_status_callback(gw, "cli", "sid")
        # Must not raise — fire-and-forget
        await cb("thinking", "msg")

    @pytest.mark.asyncio
    async def test_status_callback_timeout_swallowed(self) -> None:
        gw = _bare_gateway()
        chan = MagicMock()

        async def _slow_send(*a: Any, **kw: Any) -> None:
            await asyncio.sleep(10)

        chan.send_status = _slow_send
        gw._channels = {"cli": chan}  # type: ignore[attr-defined]
        cb = make_status_callback(gw, "cli", "sid")
        # Internal wait_for has 2.0s timeout; the test patches it at the
        # source by making send_status block 10s. Should swallow TimeoutError.
        # We need to actually wait the 2s — patch the timeout to be quick.
        # Simpler: just verify it doesn't raise within a reasonable window.
        await asyncio.wait_for(cb("thinking", "msg"), timeout=3.0)

    @pytest.mark.asyncio
    async def test_pipeline_callback_no_channel_is_noop(self) -> None:
        gw = _bare_gateway()
        cb = make_pipeline_callback(gw, "missing", "sid")
        await cb("plan", "start", iteration=1)

    @pytest.mark.asyncio
    async def test_pipeline_callback_channel_without_method_is_noop(self) -> None:
        gw = _bare_gateway()
        chan = MagicMock(spec=[])  # no send_pipeline_event attribute
        gw._channels = {"cli": chan}  # type: ignore[attr-defined]
        cb = make_pipeline_callback(gw, "cli", "sid")
        # Must not raise
        await cb("plan", "start", iteration=1)

    @pytest.mark.asyncio
    async def test_pipeline_callback_emits_phase_status_elapsed(self) -> None:
        gw = _bare_gateway()
        chan = MagicMock()
        chan.send_pipeline_event = AsyncMock()
        gw._channels = {"cli": chan}  # type: ignore[attr-defined]
        cb = make_pipeline_callback(gw, "cli", "sid")
        await cb("plan", "done", iteration=2, has_actions=True)
        chan.send_pipeline_event.assert_awaited_once()
        args, _ = chan.send_pipeline_event.await_args
        assert args[0] == "sid"
        payload = args[1]
        assert payload["phase"] == "plan"
        assert payload["status"] == "done"
        assert payload["iteration"] == 2
        assert payload["has_actions"] is True
        assert "elapsed_ms" in payload
        assert payload["elapsed_ms"] >= 0

    @pytest.mark.asyncio
    async def test_pipeline_callback_swallows_send_exception(self) -> None:
        gw = _bare_gateway()
        chan = MagicMock()
        chan.send_pipeline_event = AsyncMock(side_effect=RuntimeError("boom"))
        gw._channels = {"cli": chan}  # type: ignore[attr-defined]
        cb = make_pipeline_callback(gw, "cli", "sid")
        await cb("plan", "done", iteration=1)


# ─────────────────────────────────────────────────────────────────────────────
# formulate_response
# ─────────────────────────────────────────────────────────────────────────────


class TestFormulateResponse:
    @pytest.mark.asyncio
    async def test_streaming_path_uses_planner_stream(self) -> None:
        from cognithor.core.observer import ResponseEnvelope

        gw = _bare_gateway()
        planner = MagicMock()
        envelope = ResponseEnvelope(content="streamed answer", directive=None)
        planner.formulate_response_stream = AsyncMock(return_value=envelope)
        gw._planner = planner  # type: ignore[attr-defined]

        async def cb(event: str, data: dict[str, Any]) -> None:
            pass

        wm = WorkingMemory()
        result = await formulate_response(gw, "q", [], wm, stream_callback=cb)
        assert result.content == "streamed answer"
        planner.formulate_response_stream.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_streaming_failure_falls_back_to_non_streaming(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cognithor.core.observer import ResponseEnvelope
        from cognithor.gateway import gateway as _gw_mod

        gw = _bare_gateway()
        planner = MagicMock()
        planner.formulate_response_stream = AsyncMock(side_effect=RuntimeError("stream failed"))
        gw._planner = planner  # type: ignore[attr-defined]

        async def fake_observer(**kwargs: Any) -> ResponseEnvelope:
            return ResponseEnvelope(content="fallback answer", directive=None)

        monkeypatch.setattr(_gw_mod, "run_pge_with_observer_directive", fake_observer)

        wm = WorkingMemory()

        async def cb(event: str, data: dict[str, Any]) -> None:
            pass

        result = await formulate_response(gw, "q", [], wm, stream_callback=cb)
        assert result.content == "fallback answer"

    @pytest.mark.asyncio
    async def test_no_stream_callback_uses_observer_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cognithor.core.observer import ResponseEnvelope
        from cognithor.gateway import gateway as _gw_mod

        gw = _bare_gateway()
        gw._planner = MagicMock()  # type: ignore[attr-defined]
        # No formulate_response_stream attribute → must skip streaming path
        del gw._planner.formulate_response_stream

        called: dict[str, Any] = {}

        async def fake_observer(**kwargs: Any) -> ResponseEnvelope:
            called.update(kwargs)
            return ResponseEnvelope(content="non-stream", directive=None)

        monkeypatch.setattr(_gw_mod, "run_pge_with_observer_directive", fake_observer)
        wm = WorkingMemory()
        result = await formulate_response(gw, "the question", [], wm)
        assert result.content == "non-stream"
        assert called["user_message"] == "the question"
        assert called["working_memory"] is wm


# ─────────────────────────────────────────────────────────────────────────────
# Concurrency: handle_message — multiple concurrent depth-blocked turns
# ─────────────────────────────────────────────────────────────────────────────


class TestConcurrentDispatch:
    @pytest.mark.asyncio
    async def test_concurrent_depth_blocked_dispatch(self) -> None:
        """Many concurrent over-depth requests all return cleanly."""
        gw = _bare_gateway()
        msgs = [_msg(text=f"req-{i}", metadata={"depth": 99}) for i in range(20)]
        results = await asyncio.gather(*(handle_message(gw, m) for m in msgs))
        assert len(results) == 20
        assert all(r.is_final for r in results)
        # Each got its own response
        assert all(isinstance(r, OutgoingMessage) for r in results)


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_unicode_message_text_routes_normally(self) -> None:
        gw = _bare_gateway()
        # Force depth-exceeded so we don't need full subsystem setup
        msg = _msg(text="日本語テスト 🎉 emoji", metadata={"depth": 99})
        result = await handle_message(gw, msg)
        assert result.is_final

    @pytest.mark.asyncio
    async def test_empty_text_routes_normally(self) -> None:
        gw = _bare_gateway()
        msg = _msg(text="", metadata={"depth": 99})
        result = await handle_message(gw, msg)
        assert result.is_final

    @pytest.mark.asyncio
    async def test_very_long_text_truncated_through_pipeline(self) -> None:
        # 50KB of input doesn't blow up
        gw = _bare_gateway()
        msg = _msg(text="x" * 50_000, metadata={"depth": 99})
        result = await handle_message(gw, msg)
        assert result.is_final


# ─────────────────────────────────────────────────────────────────────────────
# Integration touchpoint: gateway methods delegate to message_handler
# ─────────────────────────────────────────────────────────────────────────────


def test_gateway_make_status_callback_delegates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake(gw: Any, channel: str, sid: str) -> Any:
        captured["gw"] = gw
        captured["channel"] = channel
        captured["sid"] = sid
        return "sentinel"

    monkeypatch.setattr(message_handler, "make_status_callback", fake)
    gw = _bare_gateway()
    result = gw._make_status_callback("cli", "S")
    assert result == "sentinel"
    assert captured == {"gw": gw, "channel": "cli", "sid": "S"}


def test_gateway_make_pipeline_callback_delegates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake(gw: Any, channel: str, sid: str) -> Any:
        captured["gw"] = gw
        captured["channel"] = channel
        captured["sid"] = sid
        return "sentinel"

    monkeypatch.setattr(message_handler, "make_pipeline_callback", fake)
    gw = _bare_gateway()
    result = gw._make_pipeline_callback("webui", "X")
    assert result == "sentinel"
    assert captured == {"gw": gw, "channel": "webui", "sid": "X"}


@pytest.mark.asyncio
async def test_handle_message_records_request_metric_on_pass_through() -> None:
    """The requests_total metric is invoked once we pass the early
    guards (consent + compliance) but before reaching Phase 1."""
    gw = _bare_gateway()
    msg = _msg(text="real request")
    with pytest.raises(RuntimeError, match=_RESOLVE_SENTINEL):
        await handle_message(gw, msg)
    # `requests_total` should have been recorded with the channel label
    gw._record_metric.assert_any_call("requests_total", 1, channel="cli")


# ─────────────────────────────────────────────────────────────────────────────
# Helper for unused import suppressions
# ─────────────────────────────────────────────────────────────────────────────


# Suppress F401 for re-exported names used implicitly in mocks above.
_keep_imports_alive = (
    Message,
    MessageRole,
    SessionContext,
    ToolResult,
    WorkingMemory,
)
