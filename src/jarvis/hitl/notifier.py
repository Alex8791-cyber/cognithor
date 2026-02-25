"""HITL Notifier — Multi-Channel-Benachrichtigung (v20).

Unterstützt:
  - In-App: Nachricht in internem Queue (für Dashboard/CLI)
  - Webhook: HTTP POST an externe URL
  - Callback: Registrierte Python-Callbacks
  - Log: Strukturiertes Logging
  - Email: Placeholder (konfigurierbar)

Alle Notifications sind async und fail-safe (nie den Workflow blockieren).
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any, Callable, Awaitable

from jarvis.hitl.types import (
    ApprovalRequest,
    ApprovalResponse,
    NotificationChannel,
    NotificationType,
)
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


# ── Notification Record ──────────────────────────────────────────

class NotificationRecord:
    """Record einer gesendeten Notification."""

    def __init__(self, channel_type: str, request_id: str,
                 message: str, success: bool, error: str = "") -> None:
        self.channel_type = channel_type
        self.request_id = request_id
        self.message = message
        self.success = success
        self.error = error
        self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "channel": self.channel_type,
            "request_id": self.request_id,
            "success": self.success,
            "timestamp": self.timestamp,
        }
        if self.error:
            d["error"] = self.error
        return d


# ── Notifier ─────────────────────────────────────────────────────

class HITLNotifier:
    """Multi-Channel-Notification-System für HITL-Requests."""

    def __init__(self, max_history: int = 500) -> None:
        self._callbacks: dict[str, Callable[..., Awaitable[None]]] = {}
        self._in_app_queue: deque[dict[str, Any]] = deque(maxlen=200)
        self._history: deque[NotificationRecord] = deque(maxlen=max_history)
        self._webhook_handler: Callable[..., Awaitable[bool]] | None = None
        self._total_sent = 0
        self._total_errors = 0

    # ── Registration ─────────────────────────────────────────────

    def register_callback(self, name: str,
                          handler: Callable[..., Awaitable[None]]) -> None:
        """Registriert einen benannten Callback."""
        self._callbacks[name] = handler

    def unregister_callback(self, name: str) -> None:
        self._callbacks.pop(name, None)

    def set_webhook_handler(self, handler: Callable[..., Awaitable[bool]]) -> None:
        """Setzt den HTTP-Webhook-Handler (für Tests mockbar)."""
        self._webhook_handler = handler

    # ── Send Notifications ───────────────────────────────────────

    async def notify_new_request(self, request: ApprovalRequest,
                                  channels: list[NotificationChannel] | None = None) -> int:
        """Benachrichtigt über eine neue Approval-Anfrage."""
        channels = channels or request.config.notifications
        if not channels:
            channels = [NotificationChannel(channel_type=NotificationType.LOG)]

        sent = 0
        for channel in channels:
            if not channel.enabled:
                continue
            message = self._render_message(channel, request, "new_request")
            success = await self._send(channel, request.request_id, message, request.to_dict())
            if success:
                sent += 1

        return sent

    async def notify_reminder(self, request: ApprovalRequest,
                               channels: list[NotificationChannel] | None = None) -> int:
        """Sendet Erinnerung für ausstehende Approval."""
        channels = channels or request.config.notifications
        if not channels:
            channels = [NotificationChannel(channel_type=NotificationType.LOG)]

        sent = 0
        for channel in channels:
            if not channel.enabled:
                continue
            message = self._render_message(channel, request, "reminder")
            success = await self._send(channel, request.request_id, message, request.to_dict())
            if success:
                sent += 1

        return sent

    async def notify_resolved(self, request: ApprovalRequest,
                               response: ApprovalResponse,
                               channels: list[NotificationChannel] | None = None) -> int:
        """Benachrichtigt über aufgelöste Approval."""
        channels = channels or request.config.notifications
        if not channels:
            channels = [NotificationChannel(channel_type=NotificationType.LOG)]

        sent = 0
        payload = {**request.to_dict(), "response": response.to_dict()}
        for channel in channels:
            if not channel.enabled:
                continue
            message = self._render_message(channel, request, "resolved",
                                            extra={"decision": response.decision.value})
            success = await self._send(channel, request.request_id, message, payload)
            if success:
                sent += 1

        return sent

    async def notify_escalated(self, request: ApprovalRequest,
                                channels: list[NotificationChannel] | None = None) -> int:
        """Benachrichtigt über Eskalation."""
        channels = channels or request.config.notifications
        if not channels:
            channels = [NotificationChannel(channel_type=NotificationType.LOG)]

        sent = 0
        for channel in channels:
            if not channel.enabled:
                continue
            message = self._render_message(channel, request, "escalated")
            success = await self._send(channel, request.request_id, message, request.to_dict())
            if success:
                sent += 1

        return sent

    # ── Internal Send ────────────────────────────────────────────

    async def _send(self, channel: NotificationChannel, request_id: str,
                    message: str, payload: dict[str, Any]) -> bool:
        """Sendet eine Notification über den angegebenen Kanal."""
        try:
            success = False

            if channel.channel_type == NotificationType.LOG:
                log.info("hitl_notification", request_id=request_id, message=message)
                success = True

            elif channel.channel_type == NotificationType.IN_APP:
                self._in_app_queue.append({
                    "request_id": request_id,
                    "message": message,
                    "payload": payload,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                })
                success = True

            elif channel.channel_type == NotificationType.CALLBACK:
                handler = self._callbacks.get(channel.endpoint)
                if handler:
                    await handler(request_id, message, payload)
                    success = True
                else:
                    log.warning("hitl_callback_not_found", endpoint=channel.endpoint)

            elif channel.channel_type == NotificationType.WEBHOOK:
                if self._webhook_handler:
                    success = await self._webhook_handler(
                        channel.endpoint, request_id, message, payload
                    )
                else:
                    log.debug("hitl_webhook_no_handler", endpoint=channel.endpoint)
                    success = True  # Graceful: Kein Handler = OK

            elif channel.channel_type == NotificationType.EMAIL:
                log.info("hitl_email_notification",
                         to=channel.endpoint, request_id=request_id)
                success = True  # Placeholder

            record = NotificationRecord(
                channel.channel_type.value, request_id, message, success,
            )
            self._history.append(record)
            self._total_sent += 1
            return success

        except Exception as exc:
            self._total_errors += 1
            record = NotificationRecord(
                channel.channel_type.value, request_id, message,
                False, str(exc),
            )
            self._history.append(record)
            log.warning("hitl_notification_error",
                        channel=channel.channel_type.value, error=str(exc))
            return False

    # ── Message Rendering ────────────────────────────────────────

    def _render_message(self, channel: NotificationChannel,
                        request: ApprovalRequest, event: str,
                        extra: dict[str, str] | None = None) -> str:
        """Rendert eine Notification-Nachricht."""
        if channel.template:
            msg = channel.template
            msg = msg.replace("{event}", event)
            msg = msg.replace("{title}", request.config.title or request.node_name)
            msg = msg.replace("{graph}", request.graph_name)
            msg = msg.replace("{node}", request.node_name)
            msg = msg.replace("{priority}", request.config.priority.value)
            msg = msg.replace("{request_id}", request.request_id)
            if extra:
                for k, v in extra.items():
                    msg = msg.replace("{" + k + "}", v)
            return msg

        # Default-Templates
        title = request.config.title or request.node_name
        templates = {
            "new_request": f"[HITL] Neue Anfrage: {title} ({request.config.priority.value})",
            "reminder": f"[HITL] Erinnerung: {title} wartet auf Bearbeitung",
            "resolved": f"[HITL] Erledigt: {title} — {extra.get('decision', '?') if extra else '?'}",
            "escalated": f"[HITL] Eskaliert: {title} (Eskalation #{request.escalation_count})",
        }
        return templates.get(event, f"[HITL] {event}: {title}")

    # ── In-App Queue ─────────────────────────────────────────────

    def get_pending_notifications(self, limit: int = 20) -> list[dict[str, Any]]:
        """Gibt ungelesene In-App-Notifications zurück."""
        return list(self._in_app_queue)[-limit:]

    def clear_in_app_queue(self) -> int:
        count = len(self._in_app_queue)
        self._in_app_queue.clear()
        return count

    # ── Stats ────────────────────────────────────────────────────

    def get_history(self, limit: int = 50) -> list[dict[str, Any]]:
        return [r.to_dict() for r in list(self._history)[-limit:]]

    def stats(self) -> dict[str, Any]:
        return {
            "total_sent": self._total_sent,
            "total_errors": self._total_errors,
            "callbacks_registered": len(self._callbacks),
            "in_app_pending": len(self._in_app_queue),
            "history_size": len(self._history),
        }
