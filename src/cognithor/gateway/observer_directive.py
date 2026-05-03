"""Gateway-side handling of Observer-issued PGE re-loop directives.

The Observer audit may return a ``PGEReloopDirective`` when it detects that a
tool call was missed (``tool_ignorance`` failure). The Gateway catches this
signal after ``Planner.formulate_response()`` and decides whether to:

- re-enter the full PGE loop with the Observer feedback injected, OR
- downgrade to response-regen only (when the directive is a duplicate of one
  already seen this session, or when the PGE iteration budget is exhausted).

This module is pure — no gateway state is touched here, only the per-session
state dict passed in by the caller.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from cognithor.core.observer import ResponseEnvelope

if TYPE_CHECKING:
    from cognithor.config import CognithorConfig
    from cognithor.core.observer import PGEReloopDirective


@dataclass(frozen=True)
class ObserverDirectiveDecision:
    """Outcome of handle_observer_directive()."""

    action: Literal["reenter_pge", "downgrade_to_regen"]
    planner_feedback: str  # empty when downgrading


def handle_observer_directive(
    *,
    directive: PGEReloopDirective,
    session_state: dict[str, Any],
    config: CognithorConfig,
) -> ObserverDirectiveDecision:
    """Decide how to act on an Observer-issued PGE directive.

    Returns ``reenter_pge`` when a fresh re-loop is warranted, or
    ``downgrade_to_regen`` when the directive is a duplicate or the PGE
    budget is exhausted (in which case Planner falls back to response
    regen).
    """
    hash_input = f"{directive.reason}|{directive.missing_data}"
    fb_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

    seen: set[str] = session_state.setdefault("seen_observer_feedback_hashes", set())
    pge_count: int = session_state.get("pge_iteration_count", 0)
    max_iter = config.security.max_iterations

    if fb_hash in seen or pge_count >= max_iter:
        return ObserverDirectiveDecision(action="downgrade_to_regen", planner_feedback="")

    seen.add(fb_hash)
    # Bounded memory: prune to last 50 when above 100.
    if len(seen) > 100:
        keep = list(seen)[-50:]
        session_state["seen_observer_feedback_hashes"] = set(keep)

    tools_text = ", ".join(directive.suggested_tools) or "(none)"
    feedback = (
        f"Observer detected tool_ignorance: missing data = {directive.missing_data}. "
        f"Suggested tools: {tools_text}. "
        "Re-plan the task and call the appropriate tools."
    )
    return ObserverDirectiveDecision(action="reenter_pge", planner_feedback=feedback)


async def run_pge_with_observer_directive(
    *,
    planner: Any,
    user_message: str,
    results: list[Any],
    working_memory: Any,
    session_state: dict[str, Any],
    config: CognithorConfig,
) -> ResponseEnvelope:
    """Drive the PGE loop with Observer-directive handling.

    Loops up to ``config.security.max_iterations`` times. Each iteration calls
    ``planner.formulate_response(...)``; if the returned envelope has a
    directive, the handler decides whether to re-enter PGE with feedback or
    downgrade (strip the directive and return the draft as-is).

    In this minimal wrapper, re-entry is simulated by prepending the Observer
    feedback to ``user_message`` and calling the Planner again. A real
    integration (Task 18) would also call ``Planner.plan()`` + Executor to
    refresh ``results``, but for now we delegate that to the Planner's own
    regen loop inside ``formulate_response``.
    """
    current_user_msg = user_message
    envelope: ResponseEnvelope | None = None

    for _ in range(config.security.max_iterations):
        envelope = await planner.formulate_response(
            user_message=current_user_msg,
            results=results,
            working_memory=working_memory,
        )
        session_state["pge_iteration_count"] = session_state.get("pge_iteration_count", 0) + 1

        if envelope.directive is None:  # type: ignore[union-attr]
            return envelope  # type: ignore[return-value]

        decision = handle_observer_directive(
            directive=envelope.directive,  # type: ignore[union-attr]
            session_state=session_state,
            config=config,
        )
        if decision.action == "downgrade_to_regen":
            # Strip the directive and deliver the envelope content as-is.
            return ResponseEnvelope(content=envelope.content, directive=None)  # type: ignore[union-attr]

        # reenter_pge: prepend the directive feedback into the next user message.
        current_user_msg = f"{user_message}\n\n[Observer feedback]\n{decision.planner_feedback}"

    # Safety: max iterations exhausted, return whatever the last envelope was
    # (directive stripped so the Gateway doesn't loop infinitely if the caller
    # decides to retry).
    if envelope is None:
        # This is unreachable when max_iterations >= 1 (Pydantic ge=1 guard),
        # but mypy needs the guard.
        return ResponseEnvelope(content="", directive=None)
    return ResponseEnvelope(content=envelope.content, directive=None)
