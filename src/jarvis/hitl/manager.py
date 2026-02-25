"""Approval Manager — Zentrale HITL-Verwaltung (v20).

Verwaltet:
  - Approval-Queue: Erstellen, Abrufen, Auflösen
  - Timeout-Handling: Automatische Eskalation bei Ablauf
  - Delegation: Weiterleitung an andere Reviewer
  - Multi-Approval: Mehrere Genehmigungen pro Request
  - Resume-Koordination: Verbindung mit v18 Graph Engine
  - Telemetry: HITL-Metriken über v19

Usage:
    manager = ApprovalManager()
    request = manager.create_request(
        execution_id="exec-1", graph_name="etl",
        node_name="validate", config=HITLConfig(...)
    )
    # ... later, reviewer responds ...
    manager.respond(request.request_id, ApprovalResponse(decision=APPROVED))
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from jarvis.hitl.types import (
    ApprovalRequest,
    ApprovalResponse,
    ApprovalStatus,
    EscalationAction,
    EscalationPolicy,
    HITLConfig,
    HITLNodeKind,
    NotificationChannel,
    NotificationType,
    ReviewPriority,
    ReviewTask,
)
from jarvis.hitl.notifier import HITLNotifier
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class ApprovalManager:
    """Zentrale Verwaltung für HITL-Approval-Workflows."""

    def __init__(self, notifier: HITLNotifier | None = None,
                 max_pending: int = 500) -> None:
        self._notifier = notifier or HITLNotifier()
        self._tasks: dict[str, ReviewTask] = {}         # request_id → task
        self._by_execution: dict[str, list[str]] = {}   # execution_id → [request_ids]
        self._resolved_callbacks: dict[str, Any] = {}   # request_id → asyncio.Event
        self._max_pending = max_pending
        self._total_created = 0
        self._total_approved = 0
        self._total_rejected = 0
        self._total_escalated = 0
        self._total_timed_out = 0

    @property
    def notifier(self) -> HITLNotifier:
        return self._notifier

    # ── Create Request ───────────────────────────────────────────

    async def create_request(
        self,
        execution_id: str,
        graph_name: str,
        node_name: str,
        config: HITLConfig,
        context: dict[str, Any] | None = None,
        checkpoint_id: str = "",
    ) -> ApprovalRequest:
        """Erstellt eine neue Approval-Anfrage."""
        request = ApprovalRequest(
            execution_id=execution_id,
            graph_name=graph_name,
            node_name=node_name,
            config=config,
            context=context or {},
            checkpoint_id=checkpoint_id,
        )

        # Gate-Check: Auto-Approve wenn Bedingung erfüllt
        if config.node_kind == HITLNodeKind.GATE and config.auto_approve_fn:
            try:
                if config.auto_approve_fn(request.context):
                    request.status = ApprovalStatus.APPROVED
                    log.info("hitl_gate_auto_approved",
                             request_id=request.request_id, node=node_name)
                    self._total_approved += 1
                    return request
            except Exception as exc:
                log.warning("hitl_gate_check_error", error=str(exc))

        task = ReviewTask(request=request)
        self._tasks[request.request_id] = task

        # Execution-Index
        if execution_id not in self._by_execution:
            self._by_execution[execution_id] = []
        self._by_execution[execution_id].append(request.request_id)

        # Event für await
        self._resolved_callbacks[request.request_id] = asyncio.Event()

        self._total_created += 1

        # Benachrichtigung
        await self._notifier.notify_new_request(request)

        log.info("hitl_request_created",
                 request_id=request.request_id,
                 graph=graph_name, node=node_name,
                 kind=config.node_kind.value,
                 priority=config.priority.value)

        return request

    # ── Respond ──────────────────────────────────────────────────

    async def respond(self, request_id: str,
                      response: ApprovalResponse) -> ReviewTask | None:
        """Verarbeitet eine Reviewer-Antwort."""
        task = self._tasks.get(request_id)
        if task is None:
            log.warning("hitl_response_not_found", request_id=request_id)
            return None

        if task.request.is_resolved:
            log.warning("hitl_already_resolved", request_id=request_id)
            return task

        response.request_id = request_id
        task.responses.append(response)
        task.request.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        if response.decision == ApprovalStatus.REJECTED:
            task.request.status = ApprovalStatus.REJECTED
            self._total_rejected += 1
            log.info("hitl_rejected", request_id=request_id,
                     reviewer=response.reviewer)
        elif response.decision == ApprovalStatus.APPROVED:
            if task.is_fully_approved:
                task.request.status = ApprovalStatus.APPROVED
                self._total_approved += 1
                log.info("hitl_approved", request_id=request_id,
                         reviewer=response.reviewer)
            else:
                log.info("hitl_partial_approval", request_id=request_id,
                         approved=task.approval_count,
                         required=task.request.config.required_approvals)
                return task

        # Notify resolved
        await self._notifier.notify_resolved(task.request, response)

        # Signal event
        event = self._resolved_callbacks.get(request_id)
        if event:
            event.set()

        return task

    # ── Wait for Resolution ──────────────────────────────────────

    async def wait_for_resolution(self, request_id: str,
                                   timeout: float | None = None) -> ReviewTask | None:
        """Wartet bis ein Request aufgelöst wird.

        Args:
            request_id: ID der Anfrage
            timeout: Max Wartezeit in Sekunden (None = aus Config)

        Returns:
            ReviewTask oder None bei Timeout
        """
        task = self._tasks.get(request_id)
        if task is None:
            return None

        if task.request.is_resolved:
            return task

        if timeout is None:
            timeout = task.request.config.escalation.timeout_seconds

        event = self._resolved_callbacks.get(request_id)
        if event is None:
            return task

        try:
            await asyncio.wait_for(event.wait(), timeout=float(timeout))
            return self._tasks.get(request_id)
        except asyncio.TimeoutError:
            await self._handle_timeout(request_id)
            return self._tasks.get(request_id)

    # ── Timeout & Escalation ─────────────────────────────────────

    async def _handle_timeout(self, request_id: str) -> None:
        """Behandelt Timeout einer Approval-Anfrage."""
        task = self._tasks.get(request_id)
        if task is None or task.request.is_resolved:
            return

        policy = task.request.config.escalation
        task.request.escalation_count += 1

        if task.request.escalation_count > policy.max_escalations:
            task.request.status = ApprovalStatus.TIMED_OUT
            self._total_timed_out += 1
            log.warning("hitl_timed_out", request_id=request_id)
            event = self._resolved_callbacks.get(request_id)
            if event:
                event.set()
            return

        action = policy.action
        self._total_escalated += 1

        if action == EscalationAction.AUTO_APPROVE:
            response = ApprovalResponse(
                request_id=request_id,
                decision=ApprovalStatus.APPROVED,
                reviewer="__auto__",
                comment="Auto-approved after timeout",
            )
            await self.respond(request_id, response)

        elif action == EscalationAction.AUTO_REJECT:
            response = ApprovalResponse(
                request_id=request_id,
                decision=ApprovalStatus.REJECTED,
                reviewer="__auto__",
                comment="Auto-rejected after timeout",
            )
            await self.respond(request_id, response)

        elif action == EscalationAction.DELEGATE:
            task.request.status = ApprovalStatus.DELEGATED
            if policy.delegate_to and policy.delegate_to not in task.request.config.assignees:
                task.request.config.assignees.append(policy.delegate_to)
            await self._notifier.notify_escalated(task.request)
            task.request.status = ApprovalStatus.PENDING
            log.info("hitl_delegated", request_id=request_id,
                     delegate_to=policy.delegate_to)

        elif action == EscalationAction.NOTIFY_SUPERVISOR:
            await self._notifier.notify_escalated(task.request)
            log.info("hitl_escalated_supervisor", request_id=request_id)

        elif action == EscalationAction.PAUSE_INDEFINITELY:
            task.request.status = ApprovalStatus.ESCALATED
            log.info("hitl_paused_indefinitely", request_id=request_id)

    async def check_timeouts(self) -> int:
        """Prüft alle offenen Requests auf Timeout."""
        timed_out = 0
        for request_id in list(self._tasks):
            task = self._tasks[request_id]
            if task.request.is_pending and task.request.is_expired:
                await self._handle_timeout(request_id)
                timed_out += 1
        return timed_out

    # ── Delegation ───────────────────────────────────────────────

    async def delegate(self, request_id: str, delegate_to: str,
                       delegated_by: str = "") -> bool:
        """Delegiert eine Anfrage an einen anderen Reviewer."""
        task = self._tasks.get(request_id)
        if task is None or task.request.is_resolved:
            return False

        if delegate_to not in task.request.config.assignees:
            task.request.config.assignees.append(delegate_to)
        task.request.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        await self._notifier.notify_new_request(task.request)
        log.info("hitl_delegated_manually", request_id=request_id,
                 to=delegate_to, by=delegated_by)
        return True

    # ── Cancel ───────────────────────────────────────────────────

    async def cancel(self, request_id: str, reason: str = "") -> bool:
        """Storniert eine Approval-Anfrage."""
        task = self._tasks.get(request_id)
        if task is None or task.request.is_resolved:
            return False

        task.request.status = ApprovalStatus.CANCELED
        task.request.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        event = self._resolved_callbacks.get(request_id)
        if event:
            event.set()

        log.info("hitl_canceled", request_id=request_id, reason=reason)
        return True

    # ── Queries ──────────────────────────────────────────────────

    def get_task(self, request_id: str) -> ReviewTask | None:
        return self._tasks.get(request_id)

    def get_pending(self, *, assignee: str = "",
                    priority: ReviewPriority | None = None,
                    limit: int = 50) -> list[ReviewTask]:
        """Gibt offene Review-Tasks zurück."""
        results: list[ReviewTask] = []
        for task in self._tasks.values():
            if not task.request.is_pending:
                continue
            if assignee and assignee not in task.request.config.assignees:
                continue
            if priority and task.request.config.priority != priority:
                continue
            results.append(task)

        results.sort(key=lambda t: (
            -list(ReviewPriority).index(t.request.config.priority),
            t.request.created_at,
        ))
        return results[:limit]

    def get_by_execution(self, execution_id: str) -> list[ReviewTask]:
        """Gibt alle Tasks einer Execution zurück."""
        request_ids = self._by_execution.get(execution_id, [])
        return [self._tasks[rid] for rid in request_ids if rid in self._tasks]

    def get_history(self, *, limit: int = 50,
                    status: ApprovalStatus | None = None) -> list[ReviewTask]:
        """Gibt historische Tasks zurück."""
        results = list(self._tasks.values())
        if status:
            results = [t for t in results if t.request.status == status]
        results.sort(key=lambda t: t.request.updated_at, reverse=True)
        return results[:limit]

    # ── Cleanup ──────────────────────────────────────────────────

    def cleanup(self, max_age_days: int = 30) -> int:
        """Bereinigt alte aufgelöste Tasks."""
        cutoff = time.time() - (max_age_days * 86400)
        removed = 0
        for request_id in list(self._tasks):
            task = self._tasks[request_id]
            if not task.request.is_resolved:
                continue
            try:
                import calendar
                ts = calendar.timegm(time.strptime(
                    task.request.updated_at, "%Y-%m-%dT%H:%M:%SZ"))
                if ts < cutoff:
                    del self._tasks[request_id]
                    self._resolved_callbacks.pop(request_id, None)
                    removed += 1
            except (ValueError, OverflowError):
                pass
        return removed

    # ── Stats ────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        pending = sum(1 for t in self._tasks.values() if t.request.is_pending)
        return {
            "total_created": self._total_created,
            "pending": pending,
            "approved": self._total_approved,
            "rejected": self._total_rejected,
            "escalated": self._total_escalated,
            "timed_out": self._total_timed_out,
            "total_tasks": len(self._tasks),
            "notifier": self._notifier.stats(),
        }
