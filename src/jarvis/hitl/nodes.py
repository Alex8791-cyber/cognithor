"""HITL Nodes -- Graph-kompatible HITL-Handler (v20).

Erstellt Node-Handler für v18 GraphBuilder die HITL-Interaktion
auslösen. Jeder Handler:
  1. Erstellt ApprovalRequest über ApprovalManager
  2. Wartet auf Resolution (oder Timeout)
  3. Mergt Response-Daten in GraphState
  4. Gibt modifizierten State zurück

Usage mit GraphBuilder:
    from jarvis.hitl import create_approval_node, create_gate_node
    from jarvis.graph import GraphBuilder, END, NodeType

    manager = ApprovalManager()

    graph = (
        GraphBuilder("review_flow")
        .add_node("process", process_handler)
        .add_node("review", create_approval_node(
            manager,
            config=HITLConfig(
                title="Ergebnis prüfen",
                assignees=["supervisor"],
                priority=ReviewPriority.HIGH,
            ),
        ), node_type=NodeType.HITL)
        .add_node("finalize", finalize_handler)
        .chain("process", "review", "finalize", END)
        .build()
    )
"""

from __future__ import annotations

from typing import Any, Callable, Awaitable

from jarvis.hitl.types import (
    ApprovalRequest,
    ApprovalResponse,
    ApprovalStatus,
    HITLConfig,
    HITLNodeKind,
    ReviewPriority,
)
from jarvis.hitl.manager import ApprovalManager
from jarvis.graph.types import GraphState
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


# ── Approval Node ────────────────────────────────────────────────

def create_approval_node(
    manager: ApprovalManager,
    config: HITLConfig | None = None,
    *,
    title: str = "",
    assignees: list[str] | None = None,
    priority: ReviewPriority = ReviewPriority.NORMAL,
    timeout: float | None = None,
    context_keys: list[str] | None = None,
) -> Callable[[GraphState], Awaitable[GraphState]]:
    """Erstellt einen Approval-Node-Handler.

    Der Graph pausiert hier bis ein Reviewer genehmigt/ablehnt.
    Bei Ablehnung wird ein Error in den State geschrieben.

    State-Output:
        __hitl_status__: "approved" | "rejected" | "timed_out"
        __hitl_comment__: Reviewer-Kommentar
        __hitl_reviewer__: Name des Reviewers
        __hitl_request_id__: ID der Anfrage
    """
    from dataclasses import replace as _replace
    if config is not None:
        # Defensiv-Kopie um Mutation der Caller-Config zu verhindern
        cfg = _replace(config, assignees=list(config.assignees))
    else:
        cfg = HITLConfig(
            node_kind=HITLNodeKind.APPROVAL,
            title=title or "Approval Required",
            assignees=assignees or [],
            priority=priority,
            context_keys=context_keys or [],
        )

    async def handler(state: GraphState) -> GraphState:
        # Context aus State extrahieren
        context = _extract_context(state, cfg.context_keys)

        request = await manager.create_request(
            execution_id=state.get("__execution_id__", ""),
            graph_name=state.get("__graph_name__", ""),
            node_name=state.get("__current_node__", cfg.title),
            config=cfg,
            context=context,
        )

        state["__hitl_request_id__"] = request.request_id

        # Auto-approved (Gate)?
        if request.status == ApprovalStatus.APPROVED:
            state["__hitl_status__"] = "approved"
            state["__hitl_reviewer__"] = "__auto__"
            return state

        # Warten auf Resolution
        task = await manager.wait_for_resolution(
            request.request_id,
            timeout=timeout,
        )

        if task is None:
            state["__hitl_status__"] = "timed_out"
            return state

        request = task.request
        state["__hitl_status__"] = request.status.value

        # Letzte Response verarbeiten
        if task.responses:
            last = task.responses[-1]
            state["__hitl_comment__"] = last.comment
            state["__hitl_reviewer__"] = last.reviewer

            # Modifications in State mergen
            if last.modifications:
                state.update(last.modifications)

        # Bei Ablehnung: Error
        if request.status == ApprovalStatus.REJECTED:
            comment = task.responses[-1].comment if task.responses else ""
            raise ValueError(
                f"HITL rejected by {state.get('__hitl_reviewer__', '?')}: {comment}"
            )

        return state

    return handler


# ── Review Node ──────────────────────────────────────────────────

def create_review_node(
    manager: ApprovalManager,
    config: HITLConfig | None = None,
    *,
    title: str = "",
    instructions: str = "",
    assignees: list[str] | None = None,
    context_keys: list[str] | None = None,
    required_approvals: int = 1,
) -> Callable[[GraphState], Awaitable[GraphState]]:
    """Erstellt einen Review-Node-Handler.

    Wie Approval, aber mit Instruktionen und Multi-Approval-Support.
    """
    from dataclasses import replace as _replace
    if config is not None:
        cfg = _replace(config, assignees=list(config.assignees))
    else:
        cfg = HITLConfig(
            node_kind=HITLNodeKind.REVIEW,
            title=title or "Review Required",
            instructions=instructions,
            assignees=assignees or [],
            context_keys=context_keys or [],
            required_approvals=required_approvals,
        )
    cfg.node_kind = HITLNodeKind.REVIEW

    return create_approval_node(manager, cfg)


# ── Input Node ───────────────────────────────────────────────────

def create_input_node(
    manager: ApprovalManager,
    *,
    title: str = "Input Required",
    description: str = "",
    input_keys: list[str] | None = None,
    assignees: list[str] | None = None,
    timeout: float | None = None,
) -> Callable[[GraphState], Awaitable[GraphState]]:
    """Erstellt einen Input-Node-Handler.

    Pausiert den Graph und wartet auf menschliche Eingabe.
    Die Eingabe wird als `modifications` in den State gemergt.

    State-Output:
        __hitl_status__: "approved" (= Eingabe erfolgt)
        + alle Felder aus modifications
    """
    cfg = HITLConfig(
        node_kind=HITLNodeKind.INPUT,
        title=title,
        description=description,
        editable_keys=input_keys or [],
        assignees=assignees or [],
    )

    async def handler(state: GraphState) -> GraphState:
        context = _extract_context(state, cfg.context_keys)

        request = await manager.create_request(
            execution_id=state.get("__execution_id__", ""),
            graph_name=state.get("__graph_name__", ""),
            node_name=state.get("__current_node__", title),
            config=cfg,
            context=context,
        )

        state["__hitl_request_id__"] = request.request_id

        task = await manager.wait_for_resolution(
            request.request_id, timeout=timeout,
        )

        if task and task.responses:
            last = task.responses[-1]
            state["__hitl_status__"] = "approved"
            if last.modifications:
                state.update(last.modifications)
        else:
            state["__hitl_status__"] = "timed_out"

        return state

    return handler


# ── Gate Node ────────────────────────────────────────────────────

def create_gate_node(
    manager: ApprovalManager,
    *,
    title: str = "Safety Gate",
    check_fn: Callable[[dict[str, Any]], bool] | None = None,
    context_keys: list[str] | None = None,
    assignees: list[str] | None = None,
    timeout: float | None = None,
) -> Callable[[GraphState], Awaitable[GraphState]]:
    """Erstellt einen Gate-Node-Handler.

    Prüft zuerst automatisch via check_fn:
    - True → Gate passiert automatisch (kein HITL nötig)
    - False → HITL-Approval erforderlich

    Ideal für bedingte Human-Review (z.B. nur bei hohem Risiko).
    """
    cfg = HITLConfig(
        node_kind=HITLNodeKind.GATE,
        title=title,
        assignees=assignees or [],
        context_keys=context_keys or [],
        auto_approve_fn=check_fn,
    )

    return create_approval_node(manager, cfg, timeout=timeout)


# ── Selection Node ───────────────────────────────────────────────

def create_selection_node(
    manager: ApprovalManager,
    *,
    title: str = "Selection Required",
    options: list[str],
    description: str = "",
    assignees: list[str] | None = None,
    timeout: float | None = None,
    context_keys: list[str] | None = None,
) -> Callable[[GraphState], Awaitable[GraphState]]:
    """Erstellt einen Selection-Node-Handler.

    Mensch wählt eine Option aus der vorgegebenen Liste.

    State-Output:
        __hitl_selection__: gewählte Option
    """
    cfg = HITLConfig(
        node_kind=HITLNodeKind.SELECTION,
        title=title,
        description=description,
        assignees=assignees or [],
        options=options,
        context_keys=context_keys or [],
    )

    async def handler(state: GraphState) -> GraphState:
        context = _extract_context(state, cfg.context_keys)

        request = await manager.create_request(
            execution_id=state.get("__execution_id__", ""),
            graph_name=state.get("__graph_name__", ""),
            node_name=state.get("__current_node__", title),
            config=cfg,
            context=context,
        )

        state["__hitl_request_id__"] = request.request_id

        task = await manager.wait_for_resolution(
            request.request_id, timeout=timeout,
        )

        if task and task.responses:
            last = task.responses[-1]
            state["__hitl_status__"] = "approved"
            state["__hitl_selection__"] = last.selected_option or ""
            if last.modifications:
                state.update(last.modifications)
        else:
            state["__hitl_status__"] = "timed_out"

        return state

    return handler


# ── Edit Node ────────────────────────────────────────────────────

def create_edit_node(
    manager: ApprovalManager,
    *,
    title: str = "Edit Required",
    editable_keys: list[str],
    assignees: list[str] | None = None,
    timeout: float | None = None,
    context_keys: list[str] | None = None,
) -> Callable[[GraphState], Awaitable[GraphState]]:
    """Erstellt einen Edit-Node-Handler.

    Mensch kann spezifische State-Felder editieren.
    Editierte Werte werden als modifications in den State gemergt.
    """
    cfg = HITLConfig(
        node_kind=HITLNodeKind.EDIT,
        title=title,
        editable_keys=editable_keys,
        assignees=assignees or [],
        context_keys=context_keys or [],
    )

    return create_approval_node(manager, cfg, timeout=timeout)


# ── Helper ───────────────────────────────────────────────────────

def _extract_context(state: GraphState, keys: list[str]) -> dict[str, Any]:
    """Extrahiert Kontext-Daten aus State für den Reviewer."""
    if not keys:
        # Alle nicht-internen Keys
        return {k: v for k, v in state.items() if not k.startswith("__")}
    return {k: state.get(k) for k in keys if k in state}
