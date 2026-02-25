"""A2A Client -- RC v1.0 (Linux Foundation).

Sendet Tasks an Remote-Agenten über JSON-RPC 2.0 / HTTP.
Unterstützt Discovery, contextId, streaming, version negotiation.
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

from jarvis.a2a.types import (
    A2A_CONTENT_TYPE,
    A2A_PROTOCOL_VERSION,
    A2A_VERSION_HEADER,
    A2AAgentCard,
    Artifact,
    Message,
    MessageRole,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
    part_from_dict,
)
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class RemoteAgent:
    """Ein bekannter Remote-Agent."""

    def __init__(self, endpoint: str, card: A2AAgentCard | None = None,
                 auth_token: str = "") -> None:
        self.endpoint = endpoint.rstrip("/")
        self.card = card
        self.auth_token = auth_token
        self.discovered_at: str = ""
        self.last_contact: str = ""
        self.request_count: int = 0
        self.error_count: int = 0

    @property
    def name(self) -> str:
        return self.card.name if self.card else self.endpoint

    @property
    def a2a_url(self) -> str:
        return f"{self.endpoint}/a2a"

    @property
    def card_url(self) -> str:
        return f"{self.endpoint}/.well-known/agent.json"

    @property
    def supports_streaming(self) -> bool:
        if self.card and self.card.capabilities:
            return self.card.capabilities.streaming
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint, "name": self.name,
            "discovered_at": self.discovered_at, "last_contact": self.last_contact,
            "request_count": self.request_count, "error_count": self.error_count,
            "has_card": self.card is not None,
            "supports_streaming": self.supports_streaming,
            "skills": [s.id for s in self.card.skills] if self.card and self.card.skills else [],
        }


class A2AClient:
    """A2A-Client: Sendet Tasks an Remote-Agenten (RC v1.0)."""

    def __init__(self) -> None:
        self._remotes: dict[str, RemoteAgent] = {}
        self._request_id_counter = 0
        self._total_requests = 0

    # ── Remote Agent Management ──────────────────────────────────

    def register_remote(self, endpoint: str, auth_token: str = "",
                        card: A2AAgentCard | None = None) -> RemoteAgent:
        agent = RemoteAgent(endpoint=endpoint, card=card, auth_token=auth_token)
        agent.discovered_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._remotes[agent.endpoint] = agent
        log.info("a2a_remote_registered", endpoint=endpoint, name=agent.name)
        return agent

    def unregister_remote(self, endpoint: str) -> bool:
        endpoint = endpoint.rstrip("/")
        if endpoint in self._remotes:
            del self._remotes[endpoint]
            return True
        return False

    def get_remote(self, endpoint: str) -> RemoteAgent | None:
        return self._remotes.get(endpoint.rstrip("/"))

    def list_remotes(self) -> list[RemoteAgent]:
        return list(self._remotes.values())

    def find_by_skill(self, skill_id: str) -> list[RemoteAgent]:
        return [a for a in self._remotes.values()
                if a.card and a.card.skills and any(s.id == skill_id for s in a.card.skills)]

    def find_by_tag(self, tag: str) -> list[RemoteAgent]:
        return [a for a in self._remotes.values()
                if a.card and tag in a.card.tags]

    # ── Discovery ────────────────────────────────────────────────

    async def discover(self, endpoint: str) -> A2AAgentCard | None:
        endpoint = endpoint.rstrip("/")
        card_url = f"{endpoint}/.well-known/agent.json"
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(card_url)
                response.raise_for_status()
                data = response.json()

            card = A2AAgentCard.from_dict(data)
            if endpoint in self._remotes:
                self._remotes[endpoint].card = card
                self._remotes[endpoint].last_contact = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            else:
                self.register_remote(endpoint, card=card)

            log.info("a2a_agent_discovered", endpoint=endpoint, name=card.name,
                     skills=len(card.skills), version=card.protocol_version)
            return card
        except ImportError:
            log.warning("a2a_httpx_not_installed")
            return None
        except Exception as exc:
            log.error("a2a_discovery_failed", endpoint=endpoint, error=str(exc))
            return None

    # ── Task Operations (RC v1.0) ────────────────────────────────

    async def send_message(
        self, endpoint: str, text: str = "", message: Message | None = None,
        task_id: str | None = None, context_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Task | None:
        """message/send -- Nachricht senden / Task erstellen (RC v1.0)."""
        endpoint = endpoint.rstrip("/")
        if message is None:
            message = Message(role=MessageRole.USER, parts=[TextPart(text=text)])

        params: dict[str, Any] = {"message": message.to_dict()}
        if task_id:
            params["id"] = task_id
        if context_id:
            params["contextId"] = context_id
        if metadata:
            params["metadata"] = metadata

        result = await self._jsonrpc_call(endpoint, "message/send", params)
        return self._parse_task_response(result) if result else None

    # Alias for backwards compat
    async def send_task(self, endpoint: str, text: str = "", **kwargs: Any) -> Task | None:
        return await self.send_message(endpoint, text=text, **kwargs)

    async def get_task(self, endpoint: str, task_id: str,
                       history_length: int = 0) -> Task | None:
        params: dict[str, Any] = {"id": task_id}
        if history_length > 0:
            params["historyLength"] = history_length
        result = await self._jsonrpc_call(endpoint.rstrip("/"), "tasks/get", params)
        return self._parse_task_response(result) if result else None

    async def list_tasks(self, endpoint: str, context_id: str | None = None,
                         state: str | None = None, limit: int = 50) -> list[Task]:
        """tasks/list -- Tasks auflisten (RC v1.0)."""
        params: dict[str, Any] = {"limit": limit}
        if context_id:
            params["contextId"] = context_id
        if state:
            params["state"] = state

        result = await self._jsonrpc_call(endpoint.rstrip("/"), "tasks/list", params)
        if result is None:
            return []

        tasks = []
        for t in result.get("tasks", []):
            parsed = self._parse_task_response(t)
            if parsed:
                tasks.append(parsed)
        return tasks

    async def cancel_task(self, endpoint: str, task_id: str) -> Task | None:
        result = await self._jsonrpc_call(endpoint.rstrip("/"), "tasks/cancel", {"id": task_id})
        return self._parse_task_response(result) if result else None

    async def continue_task(self, endpoint: str, task_id: str, text: str) -> Task | None:
        """Sendet Nachricht an bestehenden Task (Multi-Turn / INPUT_REQUIRED)."""
        return await self.send_message(endpoint=endpoint, text=text, task_id=task_id)

    # ── Streaming (RC v1.0) ──────────────────────────────────────

    async def send_message_stream(
        self, endpoint: str, text: str = "", message: Message | None = None,
        context_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """message/stream -- Streaming-Nachricht mit SSE-Events."""
        endpoint = endpoint.rstrip("/")
        if message is None:
            message = Message(role=MessageRole.USER, parts=[TextPart(text=text)])

        params: dict[str, Any] = {"message": message.to_dict()}
        if context_id:
            params["contextId"] = context_id

        body = {
            "jsonrpc": "2.0", "id": self._next_id(),
            "method": "message/stream", "params": params,
        }

        remote = self._remotes.get(endpoint)
        headers = self._build_headers(remote)
        headers["Accept"] = "text/event-stream"

        try:
            import httpx
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST", f"{endpoint}/a2a/stream",
                    json=body, headers=headers,
                ) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                yield data
                            except json.JSONDecodeError:
                                pass
        except ImportError:
            log.warning("a2a_httpx_not_installed")
        except Exception as exc:
            log.error("a2a_stream_failed", endpoint=endpoint, error=str(exc))

    # ── Local Dispatch ───────────────────────────────────────────

    async def send_task_local(self, server: Any, text: str = "",
                              message: Message | None = None,
                              task_id: str | None = None,
                              context_id: str | None = None) -> Task | None:
        if message is None:
            message = Message(role=MessageRole.USER, parts=[TextPart(text=text)])

        params: dict[str, Any] = {"message": message.to_dict()}
        if task_id:
            params["id"] = task_id
        if context_id:
            params["contextId"] = context_id

        result = await server.dispatch("message/send", params)
        if "error" in result:
            log.error("a2a_local_error", error=result["error"])
            return None
        return self._parse_task_response(result)

    # ── JSON-RPC Transport ───────────────────────────────────────

    def _next_id(self) -> int:
        self._request_id_counter += 1
        return self._request_id_counter

    def _build_headers(self, remote: RemoteAgent | None = None) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": A2A_CONTENT_TYPE,
            A2A_VERSION_HEADER: A2A_PROTOCOL_VERSION,
        }
        if remote and remote.auth_token:
            headers["Authorization"] = f"Bearer {remote.auth_token}"
        return headers

    async def _jsonrpc_call(self, endpoint: str, method: str,
                            params: dict[str, Any]) -> dict[str, Any] | None:
        self._total_requests += 1
        url = f"{endpoint}/a2a"
        body = {
            "jsonrpc": "2.0", "id": self._next_id(),
            "method": method, "params": params,
        }

        remote = self._remotes.get(endpoint)
        headers = self._build_headers(remote)

        try:
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=body, headers=headers)
                response.raise_for_status()
                data = response.json()

            if remote:
                remote.last_contact = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                remote.request_count += 1

            if "error" in data:
                log.error("a2a_rpc_error", method=method, endpoint=endpoint, error=data["error"])
                if remote:
                    remote.error_count += 1
                return None
            return data.get("result", {})

        except ImportError:
            log.warning("a2a_httpx_not_installed")
            return None
        except Exception as exc:
            log.error("a2a_rpc_failed", method=method, endpoint=endpoint, error=str(exc))
            if remote:
                remote.error_count += 1
            return None

    # ── Helpers ───────────────────────────────────────────────────

    def _parse_task_response(self, data: dict[str, Any]) -> Task | None:
        try:
            task_id = data.get("id", "")
            context_id = data.get("contextId", "")
            status_data = data.get("status", {})
            state = TaskState(status_data.get("state", "submitted"))

            status_msg = None
            if "message" in status_data:
                status_msg = Message.from_dict(status_data["message"])

            task = Task(
                id=task_id, context_id=context_id,
                status=TaskStatus(state=state, message=status_msg,
                                  timestamp=status_data.get("timestamp", "")),
                metadata=data.get("metadata", {}),
            )
            for m in data.get("messages", []):
                task.messages.append(Message.from_dict(m))
            for a in data.get("artifacts", []):
                task.artifacts.append(Artifact.from_dict(a))
            return task
        except Exception as exc:
            log.error("a2a_parse_error", error=str(exc))
            return None

    def stats(self) -> dict[str, Any]:
        return {
            "remote_agents": len(self._remotes),
            "total_requests": self._total_requests,
            "agents": [a.to_dict() for a in self._remotes.values()],
        }
