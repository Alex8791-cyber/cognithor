"""Claude Code hook bridge -- FastAPI router.

Exposes HTTP endpoints that Claude Code's native ``"type": "http"`` hook
system can POST to:

    POST /api/claude-hooks/pre-tool-use   -> Gatekeeper   -> allow|deny|ask
    POST /api/claude-hooks/post-tool-use  -> ToolHook + step check
    POST /api/claude-hooks/stop           -> Observer audit over the turn
    POST /api/claude-hooks/session-start  -> session bookkeeping
    POST /api/claude-hooks/session-end    -> session cleanup
    GET  /api/claude-hooks/health         -> bridge liveness

Claude Code sends the same JSON payload it would pass to a command hook's
stdin; we return the JSON the hook would print to stdout. Reference:
https://code.claude.com/docs/en/hooks -- in particular the
``hookSpecificOutput.permissionDecision`` contract for PreToolUse and the
``decision: "block"`` / ``additionalContext`` pattern for Post/Stop.

Design notes:
- Fail-open: if the Gatekeeper / Observer is unavailable or raises, we return
  "allow" with a reason rather than deadlocking the user's session.
- SessionContext is kept in a per-process LRU so that iteration counters
  persist across hook calls within the same Claude Code session.
- Per-step Observer audit is intentionally cheap (exit-code / error string
  heuristics). The full four-dimension Observer audit runs only on ``Stop``
  because it issues an LLM call and is too expensive per tool.
- ``GateStatus.APPROVE`` is routed through ``ApprovalManager`` when one is
  wired: the bridge creates a HITL request with a ``PAUSE_INDEFINITELY``
  escalation policy and awaits resolution within a short timeout
  (default 25s, under Claude Code's typical 30s hook timeout). Mapping:
  APPROVED→allow, REJECTED→deny. If the request is still PENDING (or in
  any other non-terminal state) when the wait elapses, the bridge maps
  to Claude Code's native ``"ask"`` prompt -- the user can answer in
  VSC right away, while the HITL request stays open in the background
  so a late Telegram/webhook reply is still recorded. Manager-side
  failures fall back to ``"ask"`` (best effort without a working
  notifier); wait-side failures deny for safety.
- Per-session decision cache: identical (tool, params) calls within the
  same Claude Code session re-use the prior Gatekeeper decision for a
  60s TTL. ``APPROVE`` decisions are NOT cached, so HITL prompts always
  fire on borderline calls.
"""

from __future__ import annotations

import hashlib
import json as _json
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Literal

from fastapi import APIRouter, FastAPI
from pydantic import BaseModel, ConfigDict, Field

from cognithor.models import (
    GateDecision,
    GateStatus,
    PlannedAction,
    SessionContext,
    ToolResult,
)
from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.config import CognithorConfig
    from cognithor.core.gatekeeper import Gatekeeper
    from cognithor.core.observer import ObserverAudit
    from cognithor.core.tool_hooks import ToolHookRunner
    from cognithor.hitl.manager import ApprovalManager

log = get_logger(__name__)


# Default HITL timeout (seconds). Kept under typical Claude Code hook
# timeouts (30s for HTTP) so the bridge response always wins. Override per
# router via ``hitl_timeout_seconds``.
DEFAULT_HITL_TIMEOUT_SECONDS = 25.0


# ─────────────────────────────────────────────────────────────────────────────
# Claude Code hook input schemas
#
# These mirror the stdin JSON documented at
# https://code.claude.com/docs/en/hooks. We accept unknown fields so that
# upstream additions (agent_id, new permission_mode values, etc.) do not
# break the bridge.
# ─────────────────────────────────────────────────────────────────────────────


class _CCHookBase(BaseModel):
    model_config = ConfigDict(extra="allow")

    session_id: str
    transcript_path: str = ""
    cwd: str = ""
    permission_mode: str = "default"
    hook_event_name: str


class CCPreToolUseInput(_CCHookBase):
    hook_event_name: Literal["PreToolUse"] = "PreToolUse"
    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)
    tool_use_id: str = ""


class CCPostToolUseInput(_CCHookBase):
    hook_event_name: Literal["PostToolUse"] = "PostToolUse"
    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)
    tool_response: Any = None
    tool_use_id: str = ""
    duration_ms: int = 0


class CCStopInput(_CCHookBase):
    hook_event_name: Literal["Stop"] = "Stop"


class CCSessionStartInput(_CCHookBase):
    hook_event_name: Literal["SessionStart"] = "SessionStart"
    source: str = "startup"
    model: str = ""


class CCSessionEndInput(_CCHookBase):
    hook_event_name: Literal["SessionEnd"] = "SessionEnd"
    reason: str = "other"


# ─────────────────────────────────────────────────────────────────────────────
# Session bookkeeping
#
# Claude Code's session_id is opaque UUID-like text. We map it to a Cognithor
# SessionContext so iteration counters, stalled-turn counts and _blocked_tools
# accumulate naturally. The dict is bounded so runaway session ids cannot
# leak memory on long-running daemons.
# ─────────────────────────────────────────────────────────────────────────────


_MAX_TRACKED_SESSIONS = 256
_DEFAULT_DECISION_CACHE_TTL_SECONDS = 60.0
_MAX_CACHED_DECISIONS_PER_SESSION = 128


def _hash_tool_call(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Stable hash of a tool name + its parameters for cache keying.

    Uses sorted-keys JSON so dict ordering does not produce different hashes
    for the same logical call. Non-JSON-serializable params fall back to
    ``str(...)`` rather than crashing -- a rare-tool cache miss is fine,
    a 500 from the bridge is not.
    """
    try:
        payload = _json.dumps({"t": tool_name, "i": tool_input}, sort_keys=True, default=str)
    except Exception:
        payload = f"{tool_name}::{tool_input!r}"
    return hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()


class _SessionTracker:
    """Per-process LRU of Claude-Code-session → Cognithor state."""

    def __init__(
        self,
        max_sessions: int = _MAX_TRACKED_SESSIONS,
        *,
        decision_cache_ttl_seconds: float = _DEFAULT_DECISION_CACHE_TTL_SECONDS,
    ) -> None:
        self._max = max_sessions
        self._contexts: OrderedDict[str, SessionContext] = OrderedDict()
        self._tool_log: dict[str, list[ToolResult]] = {}
        self._user_prompts: dict[str, str] = {}
        # Per-session LRU of (call_hash) → (decision, expires_at).
        self._decision_cache: dict[str, OrderedDict[str, tuple[GateDecision, float]]] = {}
        self._decision_cache_ttl = max(0.0, float(decision_cache_ttl_seconds))
        self._cache_hits = 0
        self._cache_misses = 0

    def get_or_create(self, cc_session_id: str, cwd: str = "") -> SessionContext:
        ctx = self._contexts.get(cc_session_id)
        if ctx is None:
            ctx = SessionContext(
                session_id=cc_session_id,
                user_id="claude-code",
                channel="claude-code-hook",
                agent_name="claude-code",
            )
            self._contexts[cc_session_id] = ctx
            self._tool_log[cc_session_id] = []
            self._user_prompts.setdefault(cc_session_id, "")
            self._decision_cache[cc_session_id] = OrderedDict()
            self._evict_if_needed()
        else:
            self._contexts.move_to_end(cc_session_id)
        return ctx

    def record_tool_result(self, cc_session_id: str, result: ToolResult) -> None:
        self._tool_log.setdefault(cc_session_id, []).append(result)
        # Keep only the last 64 results per session to bound memory.
        history = self._tool_log[cc_session_id]
        if len(history) > 64:
            self._tool_log[cc_session_id] = history[-64:]

    def tool_history(self, cc_session_id: str) -> list[ToolResult]:
        return list(self._tool_log.get(cc_session_id, []))

    def get_cached_decision(self, cc_session_id: str, call_hash: str) -> GateDecision | None:
        """Return a cached GateDecision if present and not expired."""
        if self._decision_cache_ttl <= 0:
            return None
        cache = self._decision_cache.get(cc_session_id)
        if cache is None:
            return None
        entry = cache.get(call_hash)
        if entry is None:
            self._cache_misses += 1
            return None
        decision, expires_at = entry
        if time.monotonic() >= expires_at:
            cache.pop(call_hash, None)
            self._cache_misses += 1
            return None
        cache.move_to_end(call_hash)
        self._cache_hits += 1
        return decision

    def cache_decision(
        self,
        cc_session_id: str,
        call_hash: str,
        decision: GateDecision,
    ) -> None:
        """Cache a Gatekeeper decision under (session, call_hash) for TTL.

        ``APPROVE`` is intentionally NOT cached -- HITL responses must be
        fresh per call so the user can revisit a borderline decision.
        """
        if self._decision_cache_ttl <= 0:
            return
        if decision.status == GateStatus.APPROVE:
            return
        cache = self._decision_cache.setdefault(cc_session_id, OrderedDict())
        cache[call_hash] = (decision, time.monotonic() + self._decision_cache_ttl)
        cache.move_to_end(call_hash)
        # Bound per-session entries.
        while len(cache) > _MAX_CACHED_DECISIONS_PER_SESSION:
            cache.popitem(last=False)

    def cache_stats(self) -> dict[str, int]:
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "tracked_sessions": len(self._contexts),
        }

    def drop(self, cc_session_id: str) -> None:
        self._contexts.pop(cc_session_id, None)
        self._tool_log.pop(cc_session_id, None)
        self._user_prompts.pop(cc_session_id, None)
        self._decision_cache.pop(cc_session_id, None)

    def _evict_if_needed(self) -> None:
        while len(self._contexts) > self._max:
            old_id, _ = self._contexts.popitem(last=False)
            self._tool_log.pop(old_id, None)
            self._user_prompts.pop(old_id, None)
            self._decision_cache.pop(old_id, None)
            log.debug("claude_code_hooks_session_evicted", session_id=old_id)


# ─────────────────────────────────────────────────────────────────────────────
# Step-level heuristic check (cheap, runs on every PostToolUse)
# ─────────────────────────────────────────────────────────────────────────────


def _detect_step_problem(
    tool_name: str,
    tool_input: dict[str, Any],
    tool_response: Any,
) -> str | None:
    """Return a short warning string if the tool result looks problematic.

    Heuristics only -- deliberately conservative. Full-quality auditing
    happens in the Observer on ``Stop``.
    """
    if tool_response is None:
        return None

    if isinstance(tool_response, dict):
        if tool_response.get("is_error") is True or tool_response.get("error"):
            err = tool_response.get("error") or tool_response.get("error_message") or ""
            return f"tool {tool_name!r} reported an error: {str(err)[:160]}"
        # Claude Code often stores the textual result under "content" or "stdout".
        text = (
            tool_response.get("content")
            or tool_response.get("stdout")
            or tool_response.get("output")
            or ""
        )
    elif isinstance(tool_response, str):
        text = tool_response
    else:
        return None

    if not isinstance(text, str) or not text:
        return None

    # Bash-style nonzero exit hint embedded in output.
    lowered = text.lower()
    if "command failed with exit code" in lowered or "exit code 1" in lowered:
        return f"tool {tool_name!r} returned a non-zero exit code"
    if tool_name == "Bash" and "permission denied" in lowered:
        return "shell reported 'permission denied' -- verify the path is writable"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Router factory
# ─────────────────────────────────────────────────────────────────────────────


def create_claude_code_hooks_router(
    *,
    gatekeeper: Gatekeeper | None,
    observer: ObserverAudit | None,
    hook_runner: ToolHookRunner | None = None,
    approval_manager: ApprovalManager | None = None,
    config: CognithorConfig | None = None,
    tracker: _SessionTracker | None = None,
    hitl_timeout_seconds: float = DEFAULT_HITL_TIMEOUT_SECONDS,
) -> APIRouter:
    """Build the Claude Code hook bridge router.

    All Cognithor collaborators are injected at creation time (same pattern as
    ``create_kanban_router`` / ``create_evolution_router``). Any of them may be
    ``None`` -- in that case the corresponding hook degrades to a pass-through.
    """
    router = APIRouter(prefix="/api/claude-hooks", tags=["claude-code-hooks"])
    session_tracker = tracker or _SessionTracker()

    @router.get("/health")
    async def health() -> dict[str, Any]:
        stats = session_tracker.cache_stats()
        return {
            "ok": True,
            "gatekeeper": gatekeeper is not None,
            "observer": observer is not None,
            "hook_runner": hook_runner is not None,
            "approval_manager": approval_manager is not None,
            "hitl_timeout_seconds": hitl_timeout_seconds,
            "tracked_sessions": stats["tracked_sessions"],
            "decision_cache_hits": stats["hits"],
            "decision_cache_misses": stats["misses"],
        }

    @router.post("/pre-tool-use")
    async def pre_tool_use(body: CCPreToolUseInput) -> dict[str, Any]:
        # ``bypassPermissions`` mode means the user has explicitly opted out
        # of guards for this session; defer to Claude Code's own handling.
        if body.permission_mode == "bypassPermissions":
            return _allow("bypassPermissions mode -- Cognithor deferring")

        if gatekeeper is None:
            return _allow("Cognithor Gatekeeper not wired -- allowing by default")

        session = session_tracker.get_or_create(body.session_id, body.cwd)
        # Cheap optimisation: identical (tool, params) inside the same session
        # within TTL re-uses the prior decision. Halves Gatekeeper load on
        # read-heavy turns where Claude does Read/Grep dozens of times.
        call_hash = _hash_tool_call(body.tool_name, body.tool_input)
        decision = session_tracker.get_cached_decision(body.session_id, call_hash)
        cache_hit = decision is not None

        if decision is None:
            action = PlannedAction(
                tool=body.tool_name,
                params=body.tool_input,
                rationale=f"Claude Code PreToolUse (session {body.session_id[:8]})",
            )

            try:
                decision = gatekeeper.evaluate(action, session)
            except Exception as exc:  # fail-open
                log.warning(
                    "claude_code_pre_tool_use_gatekeeper_error",
                    tool=body.tool_name,
                    error=str(exc),
                )
                return _allow(f"Gatekeeper raised -- failing open: {exc!s}")

            session_tracker.cache_decision(body.session_id, call_hash, decision)

        log.info(
            "claude_code_pre_tool_use",
            tool=body.tool_name,
            status=decision.status.value,
            risk=decision.risk_level.value,
            policy=decision.policy_name or None,
            session=body.session_id[:8],
            cache_hit=cache_hit,
        )

        if decision.is_allowed:
            cc_decision = "allow"
            reason_override: str | None = None
        elif decision.needs_approval:
            cc_decision, reason_override = await _route_approval(
                approval_manager=approval_manager,
                decision=decision,
                tool_name=body.tool_name,
                tool_input=body.tool_input,
                session_id=body.session_id,
                cwd=body.cwd,
                timeout_seconds=hitl_timeout_seconds,
            )
        else:
            cc_decision = "deny"
            reason_override = None

        reason = (
            reason_override
            or decision.reason
            or (f"Gatekeeper status={decision.status.value} risk={decision.risk_level.value}")
        )

        output: dict[str, Any] = {
            "hookEventName": "PreToolUse",
            "permissionDecision": cc_decision,
            "permissionDecisionReason": reason,
        }
        # Credential masking is surfaced to Claude Code as ``updatedInput``.
        if (
            decision.status == GateStatus.MASK
            and decision.masked_params is not None
            and decision.masked_params != body.tool_input
        ):
            output["updatedInput"] = decision.masked_params

        return {"hookSpecificOutput": output}

    @router.post("/post-tool-use")
    async def post_tool_use(body: CCPostToolUseInput) -> dict[str, Any]:
        # Forward to Cognithor's own PostToolUse hook chain (secret redaction,
        # audit log) so the two worlds share the same observability.
        if hook_runner is not None:
            try:
                response_text = (
                    body.tool_response
                    if isinstance(body.tool_response, str)
                    else _json.dumps(body.tool_response, default=str)
                )
                hook_runner.run_post_tool_use(
                    body.tool_name,
                    body.tool_input,
                    response_text,
                    body.duration_ms,
                )
            except Exception:
                log.debug("claude_code_post_tool_hook_runner_failed", exc_info=True)

        # Record into session tool history so Stop can audit the full turn.
        warning = _detect_step_problem(body.tool_name, body.tool_input, body.tool_response)
        is_error = warning is not None
        response_text = (
            body.tool_response
            if isinstance(body.tool_response, str)
            else _json.dumps(body.tool_response, default=str)[:4000]
        )
        result = ToolResult(
            tool_name=body.tool_name,
            content=response_text or "",
            is_error=is_error,
            error_message=warning,
            duration_ms=body.duration_ms,
        )
        session_tracker.record_tool_result(body.session_id, result)

        log.info(
            "claude_code_post_tool_use",
            tool=body.tool_name,
            duration_ms=body.duration_ms,
            warning=warning,
            session=body.session_id[:8],
        )

        if warning:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": f"[Cognithor step-check] {warning}",
                }
            }
        return {}

    @router.post("/stop")
    async def stop(body: CCStopInput) -> dict[str, Any]:
        history = session_tracker.tool_history(body.session_id)

        if observer is None or not history:
            session_tracker.drop(body.session_id)
            return {}

        # Best-effort turn audit. Observer is fail-open; if it returns a
        # non-pass verdict we surface ``additionalContext`` so Claude can see
        # the issue, but we do NOT block Stop unless tool_ignorance or
        # hallucination flag a retry strategy.
        try:
            audit = await observer.audit(
                user_message="<claude-code-turn>",
                response="<claude-code-final>",
                tool_results=history,
                session_id=body.session_id,
            )
        except Exception:
            log.debug("claude_code_stop_observer_failed", exc_info=True)
            session_tracker.drop(body.session_id)
            return {}

        log.info(
            "claude_code_stop_audit",
            passed=audit.overall_passed,
            strategy=audit.retry_strategy,
            session=body.session_id[:8],
            tool_count=len(history),
        )

        out: dict[str, Any] = {"hookSpecificOutput": {"hookEventName": "Stop"}}
        if not audit.overall_passed and audit.retry_strategy in (
            "response_regen",
            "pge_reloop",
        ):
            failed_reasons = [
                f"{dim.name}: {dim.reason}" for dim in audit.dimensions.values() if not dim.passed
            ]
            reason = "; ".join(failed_reasons[:3]) or "observer flagged issues"
            out["decision"] = "block"
            out["reason"] = f"[Cognithor Observer] {reason}"

        session_tracker.drop(body.session_id)
        return out

    @router.post("/session-start")
    async def session_start(body: CCSessionStartInput) -> dict[str, Any]:
        session_tracker.get_or_create(body.session_id, body.cwd)
        log.info(
            "claude_code_session_start",
            session=body.session_id[:8],
            source=body.source,
            model=body.model,
        )
        return {
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": (
                    "Cognithor is supervising this Claude Code session: "
                    "Gatekeeper gates tool calls, Observer audits on Stop."
                ),
            }
        }

    @router.post("/session-end")
    async def session_end(body: CCSessionEndInput) -> dict[str, Any]:
        session_tracker.drop(body.session_id)
        log.info(
            "claude_code_session_end",
            session=body.session_id[:8],
            reason=body.reason,
        )
        return {}

    return router


# ─────────────────────────────────────────────────────────────────────────────
# Standalone FastAPI app (for unit tests)
# ─────────────────────────────────────────────────────────────────────────────


def build_claude_code_hooks_app(
    *,
    gatekeeper: Gatekeeper | None = None,
    observer: ObserverAudit | None = None,
    hook_runner: ToolHookRunner | None = None,
    approval_manager: ApprovalManager | None = None,
    config: CognithorConfig | None = None,
    hitl_timeout_seconds: float = DEFAULT_HITL_TIMEOUT_SECONDS,
) -> FastAPI:
    """Minimal FastAPI app exposing just the claude-code-hooks router.

    Mirrors ``build_backends_app`` style -- intended for tests and for
    embedding in a dedicated hooks-only daemon if desired.
    """
    app = FastAPI(title="Cognithor Claude Code Hook Bridge")
    app.state.config = config
    app.state.gatekeeper = gatekeeper
    app.state.observer = observer
    app.state.hook_runner = hook_runner
    app.state.approval_manager = approval_manager
    app.include_router(
        create_claude_code_hooks_router(
            gatekeeper=gatekeeper,
            observer=observer,
            hook_runner=hook_runner,
            approval_manager=approval_manager,
            config=config,
            hitl_timeout_seconds=hitl_timeout_seconds,
        )
    )
    return app


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _allow(reason: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "permissionDecisionReason": reason,
        }
    }


async def _route_approval(
    *,
    approval_manager: ApprovalManager | None,
    decision: Any,
    tool_name: str,
    tool_input: dict[str, Any],
    session_id: str,
    cwd: str,
    timeout_seconds: float,
) -> tuple[str, str | None]:
    """Route a Gatekeeper APPROVE through HITL with a graceful fallback.

    Behaviour:
      * If no ``ApprovalManager`` is wired -> ``"ask"`` (Claude Code's
        native VSC prompt handles the user-in-the-loop).
      * Otherwise create an HITL request with ``PAUSE_INDEFINITELY``
        escalation. Wait up to ``timeout_seconds`` for resolution.
      * APPROVED -> ``"allow"`` (with reviewer comment).
      * REJECTED -> ``"deny"`` (with reviewer comment).
      * Still PENDING after the wait, or any unexpected status, ->
        ``"ask"``: Claude Code prompts the user inside VSC, the HITL
        request stays open in the background so a late Telegram answer
        is still recorded for audit/forensics.
      * Manager-side errors (create_request raise) -> ``"ask"`` (best
        we can do without a working notifier). Wait-side errors ->
        ``"deny"`` (treat as integrity failure of the supervision layer).

    Returns ``(cc_decision, reason_override_or_None)``.
    """
    if approval_manager is None:
        return "ask", None

    # Lazy import: HITL types are only needed when an approval is actually
    # requested, and avoids circulars at module load time.
    try:
        from cognithor.hitl.types import (
            ApprovalStatus,
            EscalationAction,
            EscalationPolicy,
            HITLConfig,
            HITLNodeKind,
            ReviewPriority,
        )
    except Exception:
        log.debug("claude_code_hitl_import_failed", exc_info=True)
        return "ask", None

    risk_value = getattr(getattr(decision, "risk_level", None), "value", "unknown")
    cfg = HITLConfig(
        node_kind=HITLNodeKind.APPROVAL,
        title=f"Claude Code: {tool_name}",
        description=(decision.reason or f"Gatekeeper risk={risk_value}")[:500],
        priority=(
            ReviewPriority.HIGH if risk_value in ("orange", "red") else ReviewPriority.NORMAL
        ),
        # PAUSE_INDEFINITELY keeps the request open after the hook timeout
        # so the user can still answer via Telegram/webhook later. The
        # bridge falls back to Claude Code's native "ask" prompt when the
        # initial wait elapses.
        escalation=EscalationPolicy(
            timeout_seconds=int(max(1.0, timeout_seconds)),
            action=EscalationAction.PAUSE_INDEFINITELY,
        ),
    )

    try:
        request = await approval_manager.create_request(
            execution_id=f"claude-code:{session_id[:12]}",
            graph_name="claude-code-hooks",
            node_name="pre-tool-use",
            config=cfg,
            context={
                "tool_name": tool_name,
                "tool_input": tool_input,
                "session_id": session_id,
                "cwd": cwd,
                "risk_level": risk_value,
                "policy": getattr(decision, "policy_name", "") or "",
                "gatekeeper_reason": decision.reason or "",
            },
        )
    except Exception as exc:
        log.warning("claude_code_hitl_create_failed", error=str(exc))
        return "ask", None

    try:
        task = await approval_manager.wait_for_resolution(
            request.request_id, timeout=timeout_seconds
        )
    except Exception as exc:
        log.warning("claude_code_hitl_wait_failed", error=str(exc))
        return "deny", f"HITL wait error -- denying for safety: {exc!s}"

    if task is None:
        return "deny", "HITL request not found -- denying for safety"

    status = task.request.status
    log.info(
        "claude_code_hitl_resolved",
        request=request.request_id,
        status=getattr(status, "value", str(status)),
        tool=tool_name,
        session=session_id[:8],
    )

    if status == ApprovalStatus.APPROVED:
        comment = ""
        if task.responses:
            comment = task.responses[-1].comment or ""
        reason = "HITL approved" + (f": {comment}" if comment else "")
        return "allow", reason
    if status == ApprovalStatus.REJECTED:
        comment = ""
        if task.responses:
            comment = task.responses[-1].comment or ""
        reason = "HITL rejected" + (f": {comment}" if comment else "")
        return "deny", reason
    # PENDING / ESCALATED / DELEGATED / CANCELED / TIMED_OUT -> hand the
    # decision back to Claude Code's native "ask" prompt. The HITL request
    # is intentionally NOT cancelled so a late Telegram/webhook reply
    # still gets recorded for audit.
    status_repr = getattr(status, "value", str(status))
    return "ask", (
        f"HITL still {status_repr} after {int(timeout_seconds)}s "
        f"(request {request.request_id}) -- handing off to Claude Code's prompt"
    )


__all__ = [
    "CCPostToolUseInput",
    "CCPreToolUseInput",
    "CCSessionEndInput",
    "CCSessionStartInput",
    "CCStopInput",
    "build_claude_code_hooks_app",
    "create_claude_code_hooks_router",
]
