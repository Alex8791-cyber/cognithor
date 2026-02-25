"""A2A Server -- RC v1.0 (Linux Foundation).

JSON-RPC 2.0 Server für eingehende Agent-zu-Agent Tasks.
Methoden nach RC v1.0:
  - message/send:           Nachricht senden / Task erstellen
  - message/stream:         Streaming-Nachricht (SSE-Events)
  - tasks/get:              Task-Status abfragen
  - tasks/list:             Tasks auflisten (mit Filtern)
  - tasks/cancel:           Task abbrechen
  - tasks/pushNotification/create:  Push-Config erstellen
  - tasks/pushNotification/get:     Push-Config abrufen
  - tasks/pushNotification/list:    Push-Configs auflisten
  - tasks/pushNotification/delete:  Push-Config löschen
  - agent/authenticatedExtendedCard: Authentifizierte Agent Card

OPTIONAL: Nur aktiv wenn A2A-Server-Modus konfiguriert ist.
"""

from __future__ import annotations

import asyncio
import hmac
import time
from typing import Any, AsyncIterator, Awaitable, Callable

from jarvis.a2a.types import (
    A2A_CONTENT_TYPE,
    A2A_PROTOCOL_VERSION,
    A2A_VERSION_HEADER,
    A2AAgentCard,
    A2AErrorCode,
    Artifact,
    Message,
    MessageRole,
    PushNotificationAuth,
    PushNotificationConfig,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

TaskHandler = Callable[[Task], Awaitable[Task]]
StreamHandler = Callable[[Task], AsyncIterator[TaskStatusUpdateEvent | TaskArtifactUpdateEvent]]


class A2AServerConfig:
    def __init__(
        self,
        enabled: bool = False,
        host: str = "127.0.0.1",
        port: int = 3002,
        agent_name: str = "Jarvis",
        agent_description: str = "",
        require_auth: bool = False,
        auth_token: str = "",
        max_tasks: int = 100,
        task_timeout_seconds: int = 3600,
        enable_streaming: bool = False,
        enable_push: bool = False,
    ) -> None:
        self.enabled = enabled
        self.host = host
        self.port = port
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.require_auth = require_auth
        self.auth_token = auth_token
        self.max_tasks = max_tasks
        self.task_timeout_seconds = task_timeout_seconds
        self.enable_streaming = enable_streaming
        self.enable_push = enable_push


class A2AServer:
    """A2A-Server: Empfängt Tasks von Remote-Agenten (RC v1.0)."""

    def __init__(self, config: A2AServerConfig | None = None) -> None:
        self._config = config or A2AServerConfig()
        self._tasks: dict[str, Task] = {}
        self._contexts: dict[str, list[str]] = {}  # contextId → [taskIds]
        self._push_configs: dict[str, PushNotificationConfig] = {}
        self._task_handler: TaskHandler | None = None
        self._stream_handler: StreamHandler | None = None
        self._agent_card: A2AAgentCard | None = None
        self._running = False
        self._start_time: float = 0
        self._request_count = 0
        self._tasks_completed = 0
        self._tasks_failed = 0

    # ── Handler Registration ─────────────────────────────────────

    def set_task_handler(self, handler: TaskHandler) -> None:
        self._task_handler = handler

    def set_stream_handler(self, handler: StreamHandler) -> None:
        self._stream_handler = handler

    def set_agent_card(self, card: A2AAgentCard) -> None:
        self._agent_card = card

    # ── JSON-RPC Dispatch (RC v1.0 method names) ─────────────────

    async def dispatch(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._request_count += 1
        params = params or {}

        method_map: dict[str, Callable[..., Awaitable[dict[str, Any]]]] = {
            # RC v1.0 method names
            "message/send": lambda: self._handle_message_send(params),
            "tasks/get": lambda: self._handle_tasks_get(params),
            "tasks/list": lambda: self._handle_tasks_list(params),
            "tasks/cancel": lambda: self._handle_tasks_cancel(params),
            "tasks/pushNotification/create": lambda: self._handle_push_create(params),
            "tasks/pushNotification/get": lambda: self._handle_push_get(params),
            "tasks/pushNotification/list": lambda: self._handle_push_list(params),
            "tasks/pushNotification/delete": lambda: self._handle_push_delete(params),
            "agent/authenticatedExtendedCard": lambda: self._handle_extended_card(params),
            # Legacy v0.3 aliases (backwards compat)
            "tasks/send": lambda: self._handle_message_send(params),
        }

        handler = method_map.get(method)
        if handler is None:
            return self._error(A2AErrorCode.METHOD_NOT_FOUND, f"Method not found: {method}")

        try:
            return await handler()
        except Exception as exc:
            log.error("a2a_dispatch_error", method=method, error=str(exc))
            return self._error(A2AErrorCode.INTERNAL_ERROR, str(exc))

    # ── JSON-RPC Processing ──────────────────────────────────────

    async def process_jsonrpc(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method = message.get("method", "")
        params = message.get("params")
        msg_id = message.get("id")

        if msg_id is None:
            await self.dispatch(method, params)
            return None

        result = await self.dispatch(method, params)
        if "error" in result:
            return {"jsonrpc": "2.0", "id": msg_id, "error": result["error"]}
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    async def handle_http_request(
        self, body: dict[str, Any], auth_token: str | None = None,
        client_version: str = "",
    ) -> dict[str, Any]:
        """HTTP-Request-Handler mit Version-Negotiation."""
        # Auth
        if self._config.require_auth:
            if not auth_token or not hmac.compare_digest(auth_token, self._config.auth_token):
                return {
                    "jsonrpc": "2.0", "id": body.get("id"),
                    "error": {"code": A2AErrorCode.UNAUTHORIZED, "message": "Unauthorized"},
                }

        # Version negotiation
        if client_version and client_version != A2A_PROTOCOL_VERSION:
            major_client = client_version.split(".")[0] if client_version else "0"
            major_server = A2A_PROTOCOL_VERSION.split(".")[0]
            if major_client != major_server:
                return {
                    "jsonrpc": "2.0", "id": body.get("id"),
                    "error": {
                        "code": A2AErrorCode.INCOMPATIBLE_VERSION,
                        "message": f"Incompatible version. Server: {A2A_PROTOCOL_VERSION}, Client: {client_version}",
                    },
                }

        result = await self.process_jsonrpc(body)
        return result or {"jsonrpc": "2.0", "id": None, "result": {}}

    # ── Streaming Handler ────────────────────────────────────────

    async def handle_stream_request(
        self, body: dict[str, Any], auth_token: str | None = None,
    ) -> AsyncIterator[str]:
        """Handles message/stream -- returns SSE events."""
        if self._config.require_auth:
            if not auth_token or not hmac.compare_digest(auth_token, self._config.auth_token):
                yield f"event: error\ndata: {{\"code\": {A2AErrorCode.UNAUTHORIZED}}}\n\n"
                return

        params = body.get("params", {})
        message_data = params.get("message", {})
        context_id = params.get("contextId", "")
        message = Message.from_dict(message_data) if message_data else None

        task = Task.create(context_id=context_id or None, message=message)
        self._tasks[task.id] = task
        self._track_context(task)

        # Initial status
        yield TaskStatusUpdateEvent(
            task_id=task.id, context_id=task.context_id,
            status=task.status,
        ).to_sse()

        # Execute via stream handler or fallback to regular handler
        if self._stream_handler:
            try:
                async for event in self._stream_handler(task):
                    yield event.to_sse()
            except Exception as exc:
                task.transition(TaskState.FAILED, Message(
                    role=MessageRole.AGENT, parts=[TextPart(text=str(exc))],
                ))
                yield TaskStatusUpdateEvent(
                    task_id=task.id, context_id=task.context_id,
                    status=task.status, final=True,
                ).to_sse()
        elif self._task_handler:
            task.transition(TaskState.WORKING)
            yield TaskStatusUpdateEvent(
                task_id=task.id, context_id=task.context_id,
                status=task.status,
            ).to_sse()

            try:
                result_task = await asyncio.wait_for(
                    self._task_handler(task),
                    timeout=self._config.task_timeout_seconds,
                )
                for artifact in result_task.artifacts:
                    yield TaskArtifactUpdateEvent(
                        task_id=task.id, context_id=task.context_id,
                        artifact=artifact, last_chunk=True,
                    ).to_sse()
                yield TaskStatusUpdateEvent(
                    task_id=task.id, context_id=task.context_id,
                    status=result_task.status, final=True,
                ).to_sse()
            except Exception as exc:
                task.transition(TaskState.FAILED, Message(
                    role=MessageRole.AGENT, parts=[TextPart(text=str(exc))],
                ))
                yield TaskStatusUpdateEvent(
                    task_id=task.id, context_id=task.context_id,
                    status=task.status, final=True,
                ).to_sse()

    # ── Method Handlers (RC v1.0) ────────────────────────────────

    async def _handle_message_send(self, params: dict[str, Any]) -> dict[str, Any]:
        """message/send -- erstellt Task oder sendet Nachricht an bestehenden."""
        task_id = params.get("id")
        context_id = params.get("contextId", "")
        message_data = params.get("message", {})
        metadata = params.get("metadata", {})
        config = params.get("configuration", {})

        message = Message.from_dict(message_data) if message_data else None

        # Bestehender Task
        if task_id and task_id in self._tasks:
            task = self._tasks[task_id]
            if task.is_complete:
                return self._error(A2AErrorCode.INVALID_REQUEST,
                                   f"Task '{task_id}' already in terminal state: {task.state.value}")
            if message:
                task.messages.append(message)
            if task.state == TaskState.INPUT_REQUIRED:
                task.transition(TaskState.WORKING, message)
            if self._task_handler:
                t = asyncio.create_task(self._execute_task(task))
                t.add_done_callback(self._handle_task_exception)
            return task.to_dict()

        # Neuer Task
        if len(self._tasks) >= self._config.max_tasks:
            return self._error(A2AErrorCode.INVALID_REQUEST, "Max tasks reached")

        task = Task.create(task_id=task_id, context_id=context_id or None, message=message)
        if metadata:
            task.metadata = metadata
        self._tasks[task.id] = task
        self._track_context(task)

        # History length from config
        accepted_history = config.get("acceptedOutputModes") or config.get("historyLength")

        log.info("a2a_task_created", task_id=task.id, context_id=task.context_id)

        if self._task_handler:
            t = asyncio.create_task(self._execute_task(task))
            t.add_done_callback(self._handle_task_exception)

        return task.to_dict()

    async def _handle_tasks_get(self, params: dict[str, Any]) -> dict[str, Any]:
        task_id = params.get("id", "")
        task = self._tasks.get(task_id)
        if task is None:
            return self._error(A2AErrorCode.TASK_NOT_FOUND, f"Task '{task_id}' not found")

        result = task.to_dict()
        history_length = params.get("historyLength", 0)
        if not history_length:
            result.pop("history", None)
        elif isinstance(history_length, int) and history_length > 0:
            result["history"] = result.get("history", [])[-history_length:]
        return result

    async def _handle_tasks_list(self, params: dict[str, Any]) -> dict[str, Any]:
        """tasks/list -- Tasks auflisten mit optionalen Filtern (RC v1.0)."""
        context_id = params.get("contextId")
        state_filter = params.get("state")
        limit = params.get("limit", 50)
        offset = params.get("offset", 0)

        tasks = list(self._tasks.values())

        # Filter by contextId
        if context_id:
            tasks = [t for t in tasks if t.context_id == context_id]

        # Filter by state
        if state_filter:
            if isinstance(state_filter, str):
                tasks = [t for t in tasks if t.state.value == state_filter]
            elif isinstance(state_filter, list):
                tasks = [t for t in tasks if t.state.value in state_filter]

        total = len(tasks)
        tasks = tasks[offset:offset + limit]

        return {
            "tasks": [t.to_dict() for t in tasks],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    async def _handle_tasks_cancel(self, params: dict[str, Any]) -> dict[str, Any]:
        task_id = params.get("id", "")
        task = self._tasks.get(task_id)
        if task is None:
            return self._error(A2AErrorCode.TASK_NOT_FOUND, f"Task '{task_id}' not found")
        if task.is_complete:
            return self._error(A2AErrorCode.TASK_NOT_CANCELABLE,
                               f"Task in terminal state: {task.state.value}")

        cancel_msg = Message(role=MessageRole.AGENT, parts=[TextPart(text="Task canceled.")])
        if not task.transition(TaskState.CANCELED, cancel_msg):
            return self._error(A2AErrorCode.TASK_NOT_CANCELABLE, "State transition not possible")

        log.info("a2a_task_canceled", task_id=task_id)
        return task.to_dict()

    # ── Push Notification CRUD (RC v1.0) ─────────────────────────

    async def _handle_push_create(self, params: dict[str, Any]) -> dict[str, Any]:
        if not self._config.enable_push:
            return self._error(A2AErrorCode.PUSH_NOT_SUPPORTED, "Push notifications disabled")

        task_id = params.get("taskId", "")
        url = params.get("url", "")
        if not task_id or not url:
            return self._error(A2AErrorCode.INVALID_PARAMS, "taskId and url required")

        # SSRF protection: only allow https URLs
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.scheme not in ("https", "http"):
            return self._error(A2AErrorCode.INVALID_PARAMS, "Push URL must use http(s) scheme")
        if not parsed.hostname:
            return self._error(A2AErrorCode.INVALID_PARAMS, "Push URL must have a valid hostname")

        if task_id not in self._tasks:
            return self._error(A2AErrorCode.TASK_NOT_FOUND, f"Task '{task_id}' not found")

        auth_data = params.get("authentication")
        auth = None
        if auth_data:
            auth = PushNotificationAuth(
                type=auth_data.get("type", "bearer"),
                credentials=auth_data.get("credentials", ""),
            )

        config = PushNotificationConfig(
            task_id=task_id, url=url, authentication=auth,
            metadata=params.get("metadata", {}),
        )
        self._push_configs[config.config_id] = config
        log.info("a2a_push_config_created", config_id=config.config_id, task_id=task_id)
        return config.to_dict()

    async def _handle_push_get(self, params: dict[str, Any]) -> dict[str, Any]:
        config_id = params.get("configId", "")
        config = self._push_configs.get(config_id)
        if config is None:
            return self._error(A2AErrorCode.INVALID_PARAMS, f"Config '{config_id}' not found")
        return config.to_dict()

    async def _handle_push_list(self, params: dict[str, Any]) -> dict[str, Any]:
        task_id = params.get("taskId")
        configs = list(self._push_configs.values())
        if task_id:
            configs = [c for c in configs if c.task_id == task_id]
        return {"configs": [c.to_dict() for c in configs]}

    async def _handle_push_delete(self, params: dict[str, Any]) -> dict[str, Any]:
        config_id = params.get("configId", "")
        if config_id in self._push_configs:
            del self._push_configs[config_id]
            log.info("a2a_push_config_deleted", config_id=config_id)
            return {"deleted": True, "configId": config_id}
        return self._error(A2AErrorCode.INVALID_PARAMS, f"Config '{config_id}' not found")

    # ── Extended Card (RC v1.0) ──────────────────────────────────

    async def _handle_extended_card(self, params: dict[str, Any]) -> dict[str, Any]:
        card = self.get_agent_card()
        card["authenticated"] = True
        card["activeTaskCount"] = sum(1 for t in self._tasks.values() if t.is_active)
        card["totalTaskCount"] = len(self._tasks)
        return card

    # ── Task Execution ───────────────────────────────────────────

    def _handle_task_exception(self, task: asyncio.Task[None]) -> None:
        """Callback for fire-and-forget tasks -- logs unhandled exceptions."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            log.error("a2a_task_background_error", error=str(exc), exc_type=type(exc).__name__)

    async def _execute_task(self, task: Task) -> None:
        if not self._task_handler:
            task.transition(TaskState.FAILED, Message(
                role=MessageRole.AGENT, parts=[TextPart(text="No task handler registered.")],
            ))
            self._tasks_failed += 1
            return

        try:
            if task.state == TaskState.SUBMITTED:
                task.transition(TaskState.WORKING)

            result_task = await asyncio.wait_for(
                self._task_handler(task),
                timeout=self._config.task_timeout_seconds,
            )

            if result_task.state == TaskState.COMPLETED:
                self._tasks_completed += 1
            elif result_task.state == TaskState.FAILED:
                self._tasks_failed += 1

            # Push notification
            await self._send_push_notification(task)

        except asyncio.TimeoutError:
            task.transition(TaskState.FAILED, Message(
                role=MessageRole.AGENT,
                parts=[TextPart(text=f"Timeout after {self._config.task_timeout_seconds}s")],
            ))
            self._tasks_failed += 1
        except Exception as exc:
            task.transition(TaskState.FAILED, Message(
                role=MessageRole.AGENT, parts=[TextPart(text=f"Error: {exc}")],
            ))
            self._tasks_failed += 1

    async def _send_push_notification(self, task: Task) -> None:
        """Sends push notification for task state changes."""
        configs = [c for c in self._push_configs.values() if c.task_id == task.id]
        for config in configs:
            try:
                import httpx
                headers: dict[str, str] = {"Content-Type": A2A_CONTENT_TYPE}
                if config.authentication and config.authentication.credentials:
                    headers["Authorization"] = f"Bearer {config.authentication.credentials}"

                payload = {
                    "taskId": task.id, "contextId": task.context_id,
                    "status": task.status.to_dict(),
                }
                async with httpx.AsyncClient(timeout=10.0) as client:
                    await client.post(config.url, json=payload, headers=headers)
            except Exception as exc:
                log.warning("a2a_push_failed", config_id=config.config_id, error=str(exc))

    # ── Context Tracking ─────────────────────────────────────────

    def _track_context(self, task: Task) -> None:
        if task.context_id not in self._contexts:
            self._contexts[task.context_id] = []
        if task.id not in self._contexts[task.context_id]:
            self._contexts[task.context_id].append(task.id)

    # ── Agent Card ───────────────────────────────────────────────

    def get_agent_card(self) -> dict[str, Any]:
        if self._agent_card:
            return self._agent_card.to_dict()
        card = A2AAgentCard(
            name=self._config.agent_name,
            description=self._config.agent_description or "Jarvis AI Agent",
            url=f"http://{self._config.host}:{self._config.port}",
        )
        return card.to_dict()

    # ── Lifecycle ────────────────────────────────────────────────

    async def start(self) -> None:
        if not self._config.enabled:
            log.info("a2a_server_disabled")
            return
        self._running = True
        self._start_time = time.time()
        log.info("a2a_server_started", host=self._config.host, port=self._config.port)

    async def stop(self) -> None:
        self._running = False
        log.info("a2a_server_stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def active_tasks(self) -> list[Task]:
        return [t for t in self._tasks.values() if t.is_active]

    def cleanup_completed(self, max_age_seconds: int = 3600) -> int:
        import calendar
        now = time.time()
        to_remove = []
        for tid, task in self._tasks.items():
            if task.is_complete:
                try:
                    ts = calendar.timegm(time.strptime(task.status.timestamp, "%Y-%m-%dT%H:%M:%SZ"))
                    if now - ts > max_age_seconds:
                        to_remove.append(tid)
                except (ValueError, OverflowError):
                    to_remove.append(tid)
        for tid in to_remove:
            del self._tasks[tid]
        return len(to_remove)

    def stats(self) -> dict[str, Any]:
        uptime = time.time() - self._start_time if self._start_time else 0
        tasks = list(self._tasks.values())
        state_counts: dict[str, int] = {}
        for task in tasks:
            s = task.state.value
            state_counts[s] = state_counts.get(s, 0) + 1
        return {
            "enabled": self._config.enabled, "running": self._running,
            "protocol_version": A2A_PROTOCOL_VERSION,
            "uptime_seconds": round(uptime, 1),
            "total_requests": self._request_count,
            "total_tasks": len(tasks),
            "tasks_completed": self._tasks_completed,
            "tasks_failed": self._tasks_failed,
            "active_tasks": sum(1 for t in tasks if t.is_active),
            "contexts": len(self._contexts),
            "push_configs": len(self._push_configs),
            "tasks_by_state": state_counts,
        }

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _error(code: int, message: str) -> dict[str, Any]:
        return {"error": {"code": code, "message": message}}
