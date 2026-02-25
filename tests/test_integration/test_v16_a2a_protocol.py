"""Tests: A2A Protocol RC v1.0.

Tests für alle v16-Module: Types, Server, Client, Integration.
Kein Netzwerk, kein externes LLM — alles lokal testbar.
"""

import asyncio
import json
import pytest
from unittest.mock import MagicMock, AsyncMock

from jarvis.a2a.types import (
    A2A_CONTENT_TYPE,
    A2A_PROTOCOL_VERSION,
    A2A_VERSION_HEADER,
    A2AAgentCapabilities,
    A2AAgentCard,
    A2AErrorCode,
    A2AInterface,
    A2AProvider,
    A2ASecurityScheme,
    A2ASkill,
    Artifact,
    DataPart,
    FilePart,
    Message,
    MessageRole,
    Part,
    PartType,
    PushNotificationAuth,
    PushNotificationConfig,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
    VALID_TRANSITIONS,
    is_valid_transition,
    part_from_dict,
)
from jarvis.a2a.server import A2AServer, A2AServerConfig
from jarvis.a2a.client import A2AClient, RemoteAgent


# ============================================================================
# Types Tests
# ============================================================================


class TestParts:
    def test_text_part(self):
        p = TextPart(text="Hello")
        d = p.to_dict()
        assert d["type"] == "text"
        assert d["text"] == "Hello"

    def test_text_part_with_metadata(self):
        p = TextPart(text="Hi", metadata={"lang": "de"})
        d = p.to_dict()
        assert d["metadata"]["lang"] == "de"

    def test_file_part_uri(self):
        p = FilePart(name="doc.pdf", mime_type="application/pdf", uri="https://example.com/doc.pdf")
        d = p.to_dict()
        assert d["file"]["uri"] == "https://example.com/doc.pdf"
        assert "bytes" not in d["file"]

    def test_file_part_inline(self):
        p = FilePart(name="img.png", data="base64data==")
        d = p.to_dict()
        assert d["file"]["bytes"] == "base64data=="

    def test_data_part(self):
        p = DataPart(data={"key": "value"})
        d = p.to_dict()
        assert d["data"]["key"] == "value"

    def test_part_from_dict_text(self):
        p = part_from_dict({"type": "text", "text": "Hello"})
        assert isinstance(p, TextPart)
        assert p.text == "Hello"

    def test_part_from_dict_file(self):
        p = part_from_dict({"type": "file", "file": {"name": "test.pdf", "mimeType": "application/pdf"}})
        assert isinstance(p, FilePart)
        assert p.name == "test.pdf"

    def test_part_from_dict_data(self):
        p = part_from_dict({"type": "data", "data": {"x": 1}})
        assert isinstance(p, DataPart)
        assert p.data["x"] == 1

    def test_part_from_dict_fallback(self):
        p = part_from_dict({"type": "unknown", "value": 42})
        assert isinstance(p, TextPart)


class TestTaskState:
    def test_terminal_states(self):
        assert TaskState.COMPLETED.is_terminal
        assert TaskState.FAILED.is_terminal
        assert TaskState.CANCELED.is_terminal
        assert TaskState.REJECTED.is_terminal

    def test_active_states(self):
        assert TaskState.SUBMITTED.is_active
        assert TaskState.WORKING.is_active
        assert TaskState.INPUT_REQUIRED.is_active
        assert TaskState.AUTH_REQUIRED.is_active

    def test_valid_transitions(self):
        assert is_valid_transition(TaskState.SUBMITTED, TaskState.WORKING)
        assert is_valid_transition(TaskState.WORKING, TaskState.COMPLETED)
        assert is_valid_transition(TaskState.WORKING, TaskState.INPUT_REQUIRED)
        assert is_valid_transition(TaskState.INPUT_REQUIRED, TaskState.WORKING)

    def test_invalid_transitions(self):
        assert not is_valid_transition(TaskState.COMPLETED, TaskState.WORKING)
        assert not is_valid_transition(TaskState.FAILED, TaskState.SUBMITTED)

    def test_new_v1_states(self):
        assert is_valid_transition(TaskState.SUBMITTED, TaskState.REJECTED)
        assert is_valid_transition(TaskState.SUBMITTED, TaskState.AUTH_REQUIRED)
        assert is_valid_transition(TaskState.AUTH_REQUIRED, TaskState.WORKING)
        assert TaskState.REJECTED.is_terminal


class TestMessage:
    def test_basic_message(self):
        m = Message(role=MessageRole.USER, parts=[TextPart(text="Hello")])
        assert m.text == "Hello"
        assert m.message_id
        assert len(m.message_id) == 16

    def test_message_to_dict(self):
        m = Message(role=MessageRole.AGENT, parts=[TextPart(text="Hi")],
                    context_id="ctx-1", task_id="task-1")
        d = m.to_dict()
        assert d["role"] == "agent"
        assert d["messageId"]
        assert d["contextId"] == "ctx-1"
        assert d["taskId"] == "task-1"

    def test_message_from_dict(self):
        d = {"role": "user", "parts": [{"type": "text", "text": "Test"}],
             "messageId": "abc123", "contextId": "ctx-x"}
        m = Message.from_dict(d)
        assert m.role == MessageRole.USER
        assert m.text == "Test"
        assert m.message_id == "abc123"
        assert m.context_id == "ctx-x"

    def test_message_roundtrip(self):
        orig = Message(role=MessageRole.USER, parts=[TextPart(text="Round")],
                       metadata={"key": "val"})
        d = orig.to_dict()
        restored = Message.from_dict(d)
        assert restored.text == "Round"
        assert restored.metadata["key"] == "val"


class TestArtifact:
    def test_basic(self):
        a = Artifact(parts=[TextPart(text="Result")], name="output")
        d = a.to_dict()
        assert d["artifactId"]
        assert d["name"] == "output"

    def test_roundtrip(self):
        a = Artifact(parts=[DataPart(data={"x": 1})], name="data")
        d = a.to_dict()
        restored = Artifact.from_dict(d)
        assert restored.name == "data"
        assert restored.artifact_id == a.artifact_id


class TestTask:
    def test_create(self):
        t = Task.create(message=Message(role=MessageRole.USER, parts=[TextPart(text="Do X")]))
        assert t.id
        assert t.context_id
        assert t.state == TaskState.SUBMITTED
        assert len(t.messages) == 1

    def test_transition(self):
        t = Task.create()
        assert t.transition(TaskState.WORKING)
        assert t.state == TaskState.WORKING
        assert len(t.history) == 1

    def test_invalid_transition(self):
        t = Task.create()
        t.transition(TaskState.WORKING)
        t.transition(TaskState.COMPLETED)
        assert not t.transition(TaskState.WORKING)

    def test_context_id_preserved(self):
        t = Task.create(context_id="my-ctx")
        assert t.context_id == "my-ctx"
        d = t.to_dict()
        assert d["contextId"] == "my-ctx"

    def test_to_dict(self):
        t = Task.create(task_id="t1", context_id="c1")
        d = t.to_dict()
        assert d["id"] == "t1"
        assert d["contextId"] == "c1"
        assert d["status"]["state"] == "submitted"


class TestStreamingEvents:
    def test_status_update_event(self):
        e = TaskStatusUpdateEvent(
            task_id="t1", context_id="c1",
            status=TaskStatus(state=TaskState.WORKING), final=False,
        )
        d = e.to_dict()
        assert d["taskId"] == "t1"
        assert d["status"]["state"] == "working"
        assert not d["final"]

    def test_status_event_sse(self):
        e = TaskStatusUpdateEvent(
            task_id="t1", context_id="c1",
            status=TaskStatus(state=TaskState.COMPLETED), final=True,
        )
        sse = e.to_sse()
        assert sse.startswith("event: status\n")
        assert '"final": true' in sse

    def test_artifact_update_event(self):
        a = Artifact(parts=[TextPart(text="Result")])
        e = TaskArtifactUpdateEvent(
            task_id="t1", context_id="c1", artifact=a, last_chunk=True,
        )
        d = e.to_dict()
        assert d["lastChunk"]
        assert d["artifact"]["parts"][0]["text"] == "Result"

    def test_artifact_event_sse(self):
        a = Artifact(parts=[TextPart(text="Data")])
        e = TaskArtifactUpdateEvent(task_id="t1", context_id="c1", artifact=a)
        sse = e.to_sse()
        assert sse.startswith("event: artifact\n")


class TestPushNotificationConfig:
    def test_basic(self):
        c = PushNotificationConfig(task_id="t1", url="https://example.com/webhook")
        assert c.config_id
        d = c.to_dict()
        assert d["taskId"] == "t1"
        assert d["url"] == "https://example.com/webhook"

    def test_with_auth(self):
        auth = PushNotificationAuth(type="bearer", credentials="token123")
        c = PushNotificationConfig(task_id="t1", url="https://x.com/hook", authentication=auth)
        d = c.to_dict()
        assert d["authentication"]["credentials"] == "token123"


class TestAgentCard:
    def test_basic_card(self):
        card = A2AAgentCard(name="TestAgent", description="A test agent")
        d = card.to_dict()
        assert d["name"] == "TestAgent"
        assert d["protocolVersion"] == A2A_PROTOCOL_VERSION
        assert d["capabilities"]["streaming"] is False

    def test_card_with_skills(self):
        card = A2AAgentCard(
            skills=[A2ASkill(id="code", name="Coding", tags=["python"])],
        )
        d = card.to_dict()
        assert len(d["skills"]) == 1
        assert d["skills"][0]["id"] == "code"

    def test_card_with_interfaces(self):
        card = A2AAgentCard(
            interfaces=[
                A2AInterface(protocol="jsonrpc", url="http://localhost:3002/a2a"),
                A2AInterface(protocol="rest", url="http://localhost:3002/"),
            ],
        )
        d = card.to_dict()
        assert len(d["interfaces"]) == 2
        assert d["interfaces"][0]["protocol"] == "jsonrpc"

    def test_card_with_security(self):
        card = A2AAgentCard(
            security_schemes=[A2ASecurityScheme(type="http", scheme="bearer")],
        )
        d = card.to_dict()
        assert "securitySchemes" in d
        assert d["securitySchemes"]["scheme_0"]["scheme"] == "bearer"

    def test_card_roundtrip(self):
        orig = A2AAgentCard(
            name="Jarvis", description="AI Agent", version="16.0.0",
            capabilities=A2AAgentCapabilities(streaming=True, push_notifications=True),
            skills=[A2ASkill(id="web", name="Web")],
            interfaces=[A2AInterface(protocol="jsonrpc", url="/a2a")],
            tags=["test"],
        )
        d = orig.to_dict()
        restored = A2AAgentCard.from_dict(d)
        assert restored.name == "Jarvis"
        assert restored.capabilities.streaming
        assert restored.capabilities.push_notifications
        assert len(restored.skills) == 1
        assert len(restored.interfaces) == 1
        assert "test" in restored.tags

    def test_capabilities_typed(self):
        cap = A2AAgentCapabilities(streaming=True, push_notifications=False)
        d = cap.to_dict()
        assert d["streaming"] is True
        assert d["pushNotifications"] is False
        assert d["stateTransitionHistory"] is True


class TestErrorCodes:
    def test_standard_codes(self):
        assert A2AErrorCode.METHOD_NOT_FOUND == -32601
        assert A2AErrorCode.TASK_NOT_FOUND == -32001
        assert A2AErrorCode.INCOMPATIBLE_VERSION == -32005


class TestConstants:
    def test_protocol_version(self):
        assert A2A_PROTOCOL_VERSION == "1.0"

    def test_content_type(self):
        assert A2A_CONTENT_TYPE == "application/a2a+json"

    def test_version_header(self):
        assert A2A_VERSION_HEADER == "A2A-Version"


# ============================================================================
# Server Tests
# ============================================================================


class TestA2AServerConfig:
    def test_defaults(self):
        c = A2AServerConfig()
        assert not c.enabled
        assert c.port == 3002
        assert not c.enable_streaming
        assert not c.enable_push

    def test_custom(self):
        c = A2AServerConfig(enabled=True, port=9000, enable_streaming=True, enable_push=True)
        assert c.enabled
        assert c.port == 9000


class TestA2AServer:
    @pytest.fixture
    def server(self):
        return A2AServer(A2AServerConfig(enabled=True))

    @pytest.fixture
    def server_with_handler(self):
        s = A2AServer(A2AServerConfig(enabled=True))
        async def handler(task):
            task.artifacts.append(Artifact(parts=[TextPart(text="Done")]))
            task.transition(TaskState.COMPLETED, Message(
                role=MessageRole.AGENT, parts=[TextPart(text="Done")],
            ))
            return task
        s.set_task_handler(handler)
        return s

    @pytest.mark.asyncio
    async def test_message_send_new_task(self, server_with_handler):
        result = await server_with_handler.dispatch("message/send", {
            "message": {"role": "user", "parts": [{"type": "text", "text": "Hello"}]},
        })
        assert "id" in result
        assert "contextId" in result
        assert result["status"]["state"] == "submitted"

    @pytest.mark.asyncio
    async def test_message_send_with_context_id(self, server_with_handler):
        result = await server_with_handler.dispatch("message/send", {
            "contextId": "my-context",
            "message": {"role": "user", "parts": [{"type": "text", "text": "Hi"}]},
        })
        assert result["contextId"] == "my-context"

    @pytest.mark.asyncio
    async def test_message_send_existing_task(self, server_with_handler):
        r1 = await server_with_handler.dispatch("message/send", {
            "message": {"role": "user", "parts": [{"type": "text", "text": "Start"}]},
        })
        task_id = r1["id"]
        task = server_with_handler.get_task(task_id)
        task.transition(TaskState.WORKING)
        task.transition(TaskState.INPUT_REQUIRED)
        r2 = await server_with_handler.dispatch("message/send", {
            "id": task_id,
            "message": {"role": "user", "parts": [{"type": "text", "text": "More info"}]},
        })
        assert r2["id"] == task_id

    @pytest.mark.asyncio
    async def test_message_send_max_tasks(self, server):
        server._config.max_tasks = 1
        await server.dispatch("message/send", {
            "message": {"role": "user", "parts": [{"type": "text", "text": "First"}]},
        })
        result = await server.dispatch("message/send", {
            "message": {"role": "user", "parts": [{"type": "text", "text": "Second"}]},
        })
        assert "error" in result

    @pytest.mark.asyncio
    async def test_legacy_tasks_send(self, server):
        result = await server.dispatch("tasks/send", {
            "message": {"role": "user", "parts": [{"type": "text", "text": "Legacy"}]},
        })
        assert "id" in result

    @pytest.mark.asyncio
    async def test_tasks_get(self, server):
        r = await server.dispatch("message/send", {
            "message": {"role": "user", "parts": [{"type": "text", "text": "Test"}]},
        })
        result = await server.dispatch("tasks/get", {"id": r["id"]})
        assert result["id"] == r["id"]

    @pytest.mark.asyncio
    async def test_tasks_get_not_found(self, server):
        result = await server.dispatch("tasks/get", {"id": "nonexistent"})
        assert result["error"]["code"] == A2AErrorCode.TASK_NOT_FOUND

    @pytest.mark.asyncio
    async def test_tasks_get_with_history(self, server):
        r = await server.dispatch("message/send", {
            "message": {"role": "user", "parts": [{"type": "text", "text": "X"}]},
        })
        task = server.get_task(r["id"])
        task.transition(TaskState.WORKING)
        task.transition(TaskState.COMPLETED)
        result = await server.dispatch("tasks/get", {"id": r["id"], "historyLength": 10})
        assert "history" in result
        assert len(result["history"]) == 2

    @pytest.mark.asyncio
    async def test_tasks_list_empty(self, server):
        result = await server.dispatch("tasks/list", {})
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_tasks_list_with_filter(self, server):
        await server.dispatch("message/send", {
            "contextId": "ctx-A",
            "message": {"role": "user", "parts": [{"type": "text", "text": "A"}]},
        })
        await server.dispatch("message/send", {
            "contextId": "ctx-B",
            "message": {"role": "user", "parts": [{"type": "text", "text": "B"}]},
        })
        result = await server.dispatch("tasks/list", {"contextId": "ctx-A"})
        assert result["total"] == 1

    @pytest.mark.asyncio
    async def test_tasks_list_state_filter(self, server):
        r = await server.dispatch("message/send", {
            "message": {"role": "user", "parts": [{"type": "text", "text": "X"}]},
        })
        task = server.get_task(r["id"])
        task.transition(TaskState.WORKING)
        task.transition(TaskState.COMPLETED)
        result = await server.dispatch("tasks/list", {"state": "completed"})
        assert result["total"] == 1

    @pytest.mark.asyncio
    async def test_tasks_list_pagination(self, server):
        for i in range(5):
            await server.dispatch("message/send", {
                "message": {"role": "user", "parts": [{"type": "text", "text": f"Task {i}"}]},
            })
        result = await server.dispatch("tasks/list", {"limit": 2, "offset": 0})
        assert len(result["tasks"]) == 2
        assert result["total"] == 5

    @pytest.mark.asyncio
    async def test_tasks_cancel(self, server):
        r = await server.dispatch("message/send", {
            "message": {"role": "user", "parts": [{"type": "text", "text": "Cancel me"}]},
        })
        result = await server.dispatch("tasks/cancel", {"id": r["id"]})
        assert result["status"]["state"] == "canceled"

    @pytest.mark.asyncio
    async def test_tasks_cancel_not_found(self, server):
        result = await server.dispatch("tasks/cancel", {"id": "nope"})
        assert result["error"]["code"] == A2AErrorCode.TASK_NOT_FOUND

    @pytest.mark.asyncio
    async def test_tasks_cancel_terminal(self, server):
        r = await server.dispatch("message/send", {
            "message": {"role": "user", "parts": [{"type": "text", "text": "X"}]},
        })
        task = server.get_task(r["id"])
        task.transition(TaskState.WORKING)
        task.transition(TaskState.COMPLETED)
        result = await server.dispatch("tasks/cancel", {"id": r["id"]})
        assert result["error"]["code"] == A2AErrorCode.TASK_NOT_CANCELABLE

    @pytest.mark.asyncio
    async def test_push_disabled(self, server):
        result = await server.dispatch("tasks/pushNotification/create", {
            "taskId": "t1", "url": "https://example.com/hook",
        })
        assert result["error"]["code"] == A2AErrorCode.PUSH_NOT_SUPPORTED

    @pytest.mark.asyncio
    async def test_push_crud(self):
        s = A2AServer(A2AServerConfig(enabled=True, enable_push=True))
        r = await s.dispatch("message/send", {
            "message": {"role": "user", "parts": [{"type": "text", "text": "X"}]},
        })
        task_id = r["id"]
        result = await s.dispatch("tasks/pushNotification/create", {
            "taskId": task_id, "url": "https://example.com/hook",
            "authentication": {"type": "bearer", "credentials": "tok"},
        })
        assert "configId" in result
        config_id = result["configId"]

        result = await s.dispatch("tasks/pushNotification/get", {"configId": config_id})
        assert result["url"] == "https://example.com/hook"

        result = await s.dispatch("tasks/pushNotification/list", {"taskId": task_id})
        assert len(result["configs"]) == 1

        result = await s.dispatch("tasks/pushNotification/delete", {"configId": config_id})
        assert result["deleted"]

        result = await s.dispatch("tasks/pushNotification/list", {})
        assert len(result["configs"]) == 0

    @pytest.mark.asyncio
    async def test_push_create_task_not_found(self):
        s = A2AServer(A2AServerConfig(enabled=True, enable_push=True))
        result = await s.dispatch("tasks/pushNotification/create", {
            "taskId": "nonexistent", "url": "https://x.com/hook",
        })
        assert result["error"]["code"] == A2AErrorCode.TASK_NOT_FOUND

    @pytest.mark.asyncio
    async def test_extended_card(self, server):
        result = await server.dispatch("agent/authenticatedExtendedCard", {})
        assert result["authenticated"]
        assert "activeTaskCount" in result

    @pytest.mark.asyncio
    async def test_version_negotiation_compatible(self, server):
        body = {"jsonrpc": "2.0", "id": 1, "method": "tasks/get", "params": {"id": "x"}}
        result = await server.handle_http_request(body, client_version="1.0")
        assert result.get("error", {}).get("code") != A2AErrorCode.INCOMPATIBLE_VERSION

    @pytest.mark.asyncio
    async def test_version_negotiation_incompatible(self, server):
        body = {"jsonrpc": "2.0", "id": 1, "method": "tasks/get", "params": {"id": "x"}}
        result = await server.handle_http_request(body, client_version="2.0")
        assert result["error"]["code"] == A2AErrorCode.INCOMPATIBLE_VERSION

    @pytest.mark.asyncio
    async def test_process_jsonrpc_request(self, server):
        result = await server.process_jsonrpc({
            "jsonrpc": "2.0", "id": 1, "method": "message/send",
            "params": {"message": {"role": "user", "parts": [{"type": "text", "text": "Hi"}]}},
        })
        assert result["jsonrpc"] == "2.0"
        assert result["id"] == 1
        assert "result" in result

    @pytest.mark.asyncio
    async def test_process_jsonrpc_notification(self, server):
        result = await server.process_jsonrpc({
            "jsonrpc": "2.0", "method": "message/send",
            "params": {"message": {"role": "user", "parts": [{"type": "text", "text": "Hi"}]}},
        })
        assert result is None

    @pytest.mark.asyncio
    async def test_method_not_found(self, server):
        result = await server.dispatch("nonexistent/method", {})
        assert result["error"]["code"] == A2AErrorCode.METHOD_NOT_FOUND

    @pytest.mark.asyncio
    async def test_auth_required(self):
        s = A2AServer(A2AServerConfig(enabled=True, require_auth=True, auth_token="secret"))
        body = {"jsonrpc": "2.0", "id": 1, "method": "tasks/get", "params": {"id": "x"}}
        result = await s.handle_http_request(body, auth_token=None)
        assert result["error"]["code"] == A2AErrorCode.UNAUTHORIZED
        result = await s.handle_http_request(body, auth_token="wrong")
        assert result["error"]["code"] == A2AErrorCode.UNAUTHORIZED
        result = await s.handle_http_request(body, auth_token="secret")
        assert result.get("error", {}).get("code") != A2AErrorCode.UNAUTHORIZED

    def test_agent_card_default(self):
        s = A2AServer(A2AServerConfig(enabled=True, agent_name="TestBot"))
        card = s.get_agent_card()
        assert card["name"] == "TestBot"
        assert card["protocolVersion"] == A2A_PROTOCOL_VERSION

    def test_agent_card_custom(self):
        s = A2AServer()
        custom = A2AAgentCard(name="Custom", skills=[A2ASkill(id="x", name="X")])
        s.set_agent_card(custom)
        card = s.get_agent_card()
        assert card["name"] == "Custom"
        assert len(card["skills"]) == 1

    @pytest.mark.asyncio
    async def test_lifecycle(self):
        s = A2AServer(A2AServerConfig(enabled=True))
        await s.start()
        assert s.is_running
        await s.stop()
        assert not s.is_running

    @pytest.mark.asyncio
    async def test_lifecycle_disabled(self):
        s = A2AServer(A2AServerConfig(enabled=False))
        await s.start()
        assert not s.is_running

    @pytest.mark.asyncio
    async def test_stats(self, server):
        await server.start()
        stats = server.stats()
        assert stats["enabled"]
        assert stats["running"]
        assert stats["protocol_version"] == "1.0"
        assert "contexts" in stats
        assert "push_configs" in stats

    @pytest.mark.asyncio
    async def test_context_tracking(self, server):
        await server.dispatch("message/send", {
            "contextId": "shared-ctx",
            "message": {"role": "user", "parts": [{"type": "text", "text": "A"}]},
        })
        await server.dispatch("message/send", {
            "contextId": "shared-ctx",
            "message": {"role": "user", "parts": [{"type": "text", "text": "B"}]},
        })
        assert "shared-ctx" in server._contexts
        assert len(server._contexts["shared-ctx"]) == 2


# ============================================================================
# Client Tests
# ============================================================================


class TestRemoteAgent:
    def test_basic(self):
        agent = RemoteAgent("http://agent2:3002")
        assert agent.endpoint == "http://agent2:3002"
        assert agent.a2a_url == "http://agent2:3002/a2a"

    def test_with_card(self):
        card = A2AAgentCard(name="Agent2", capabilities=A2AAgentCapabilities(streaming=True))
        agent = RemoteAgent("http://agent2:3002", card=card)
        assert agent.name == "Agent2"
        assert agent.supports_streaming

    def test_to_dict(self):
        agent = RemoteAgent("http://agent2:3002")
        d = agent.to_dict()
        assert d["endpoint"] == "http://agent2:3002"
        assert "supports_streaming" in d


class TestA2AClient:
    def test_register_remote(self):
        client = A2AClient()
        agent = client.register_remote("http://agent2:3002", auth_token="tok")
        assert len(client.list_remotes()) == 1
        assert agent.auth_token == "tok"

    def test_unregister_remote(self):
        client = A2AClient()
        client.register_remote("http://agent2:3002")
        assert client.unregister_remote("http://agent2:3002")
        assert len(client.list_remotes()) == 0

    def test_find_by_skill(self):
        client = A2AClient()
        card = A2AAgentCard(skills=[A2ASkill(id="coding", name="Coding")])
        client.register_remote("http://coder:3002", card=card)
        client.register_remote("http://writer:3002")
        results = client.find_by_skill("coding")
        assert len(results) == 1

    def test_find_by_tag(self):
        client = A2AClient()
        card = A2AAgentCard(tags=["insurance"])
        client.register_remote("http://ins:3002", card=card)
        results = client.find_by_tag("insurance")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_send_task_local(self):
        server = A2AServer(A2AServerConfig(enabled=True))
        client = A2AClient()
        task = await client.send_task_local(server, text="Hello local")
        assert task is not None
        assert task.state == TaskState.SUBMITTED
        assert task.context_id

    @pytest.mark.asyncio
    async def test_send_task_local_with_context(self):
        server = A2AServer(A2AServerConfig(enabled=True))
        client = A2AClient()
        task = await client.send_task_local(server, text="Hi", context_id="ctx-test")
        assert task is not None
        assert task.context_id == "ctx-test"

    def test_stats(self):
        client = A2AClient()
        client.register_remote("http://x:3002")
        stats = client.stats()
        assert stats["remote_agents"] == 1

    def test_build_headers(self):
        client = A2AClient()
        headers = client._build_headers()
        assert headers["Content-Type"] == A2A_CONTENT_TYPE
        assert headers[A2A_VERSION_HEADER] == A2A_PROTOCOL_VERSION

    def test_build_headers_with_auth(self):
        client = A2AClient()
        agent = RemoteAgent("http://x:3002", auth_token="mytoken")
        headers = client._build_headers(agent)
        assert headers["Authorization"] == "Bearer mytoken"


# ============================================================================
# Integration Tests (Client ↔ Server in-process)
# ============================================================================


class TestA2AIntegration:
    @pytest.mark.asyncio
    async def test_full_task_lifecycle(self):
        async def handler(task):
            task.artifacts.append(Artifact(parts=[TextPart(text="Result")]))
            task.transition(TaskState.COMPLETED, Message(
                role=MessageRole.AGENT, parts=[TextPart(text="Done")],
            ))
            return task

        server = A2AServer(A2AServerConfig(enabled=True))
        server.set_task_handler(handler)
        client = A2AClient()

        task = await client.send_task_local(server, text="Analyze this")
        assert task is not None
        assert task.state == TaskState.SUBMITTED
        await asyncio.sleep(0.1)

        result = await server.dispatch("tasks/get", {"id": task.id})
        assert result["status"]["state"] == "completed"
        assert len(result.get("artifacts", [])) == 1

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self):
        call_count = 0
        async def handler(task):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                task.transition(TaskState.INPUT_REQUIRED, Message(
                    role=MessageRole.AGENT, parts=[TextPart(text="Which product?")],
                ))
            else:
                task.artifacts.append(Artifact(parts=[TextPart(text="BU recommended")]))
                task.transition(TaskState.COMPLETED)
            return task

        server = A2AServer(A2AServerConfig(enabled=True))
        server.set_task_handler(handler)
        client = A2AClient()

        task = await client.send_task_local(server, text="Insurance advice",
                                             context_id="conversation-1")
        assert task is not None
        await asyncio.sleep(0.1)

        result = await server.dispatch("tasks/get", {"id": task.id})
        assert result["status"]["state"] == "input-required"

        task2 = await client.send_task_local(server, text="BU insurance", task_id=task.id)
        await asyncio.sleep(0.1)

        result = await server.dispatch("tasks/get", {"id": task.id})
        assert result["status"]["state"] == "completed"

    @pytest.mark.asyncio
    async def test_tasks_list_integration(self):
        server = A2AServer(A2AServerConfig(enabled=True))
        client = A2AClient()
        await client.send_task_local(server, text="A", context_id="ctx-list")
        await client.send_task_local(server, text="B", context_id="ctx-list")
        await client.send_task_local(server, text="C", context_id="other-ctx")

        result = await server.dispatch("tasks/list", {})
        assert result["total"] == 3
        result = await server.dispatch("tasks/list", {"contextId": "ctx-list"})
        assert result["total"] == 2

    @pytest.mark.asyncio
    async def test_rejected_task(self):
        async def handler(task):
            task.transition(TaskState.REJECTED, Message(
                role=MessageRole.AGENT, parts=[TextPart(text="Cannot process.")],
            ))
            return task

        server = A2AServer(A2AServerConfig(enabled=True))
        server.set_task_handler(handler)
        result = await server.dispatch("message/send", {
            "message": {"role": "user", "parts": [{"type": "text", "text": "Invalid"}]},
        })
        await asyncio.sleep(0.1)
        task = await server.dispatch("tasks/get", {"id": result["id"]})
        assert task["status"]["state"] == "rejected"


# ============================================================================
# HTTP Handler Tests
# ============================================================================


class TestA2AHTTPHandler:
    """Tests für den A2A HTTP-Handler (ohne FastAPI)."""

    def _make_adapter(self):
        from jarvis.a2a.adapter import A2AAdapter

        config = MagicMock()
        config.mcp_config_file = MagicMock()
        config.mcp_config_file.exists.return_value = False
        config.owner_name = "test"
        return A2AAdapter(config)

    def test_handler_init(self) -> None:
        from jarvis.a2a.http_handler import A2AHTTPHandler

        adapter = self._make_adapter()
        handler = A2AHTTPHandler(adapter)
        assert handler.adapter is adapter

    def test_extract_auth_token(self) -> None:
        from jarvis.a2a.http_handler import A2AHTTPHandler

        handler = A2AHTTPHandler(self._make_adapter())
        assert handler._extract_token("Bearer tok123") == "tok123"
        assert handler._extract_token("") is None
        assert handler._extract_token("Basic xxx") is None
        assert handler._extract_token("BearerNOSPACE") is None

    def test_response_headers(self) -> None:
        from jarvis.a2a.http_handler import A2AHTTPHandler

        handler = A2AHTTPHandler(self._make_adapter())
        headers = handler._response_headers()
        assert headers["Content-Type"] == "application/a2a+json"
        assert "A2A-Version" in headers
        assert headers["A2A-Version"] == "1.0"

    @pytest.mark.asyncio
    async def test_handle_agent_card(self) -> None:
        from jarvis.a2a.http_handler import A2AHTTPHandler

        handler = A2AHTTPHandler(self._make_adapter())
        card = await handler.handle_agent_card()
        assert "error" in card  # Not initialized

    @pytest.mark.asyncio
    async def test_handle_health_disabled(self) -> None:
        from jarvis.a2a.http_handler import A2AHTTPHandler

        handler = A2AHTTPHandler(self._make_adapter())
        result = await handler.handle_health()
        assert result["status"] == "disabled"
        assert result["enabled"] is False
        assert result["protocol_version"] == "1.0"


# ============================================================================
# Adapter Detail Tests
# ============================================================================


class TestA2AAdapterDetails:
    """Erweiterte Adapter-Tests."""

    def _make_adapter(self):
        from jarvis.a2a.adapter import A2AAdapter

        config = MagicMock()
        config.mcp_config_file = MagicMock()
        config.mcp_config_file.exists.return_value = False
        config.owner_name = "TestUser"
        return A2AAdapter(config)

    def test_adapter_disabled_by_default(self) -> None:
        adapter = self._make_adapter()
        assert adapter.setup() is False
        assert adapter.enabled is False
        assert adapter.server is None
        assert adapter.client is None

    def test_adapter_stats_disabled(self) -> None:
        adapter = self._make_adapter()
        stats = adapter.stats()
        assert stats["enabled"] is False
        assert stats["protocol_version"] == "1.0"

    def test_get_agent_card_uninitialized(self) -> None:
        adapter = self._make_adapter()
        card = adapter.get_agent_card()
        assert "error" in card

    @pytest.mark.asyncio
    async def test_handle_request_no_server(self) -> None:
        adapter = self._make_adapter()
        result = await adapter.handle_a2a_request(
            {"jsonrpc": "2.0", "id": 1, "method": "message/send"},
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_delegate_no_client(self) -> None:
        adapter = self._make_adapter()
        task = await adapter.delegate_task("http://x:3002", "Hello")
        assert task is None

    @pytest.mark.asyncio
    async def test_discover_no_client(self) -> None:
        adapter = self._make_adapter()
        card = await adapter.discover_remote("http://x:3002")
        assert card is None

    @pytest.mark.asyncio
    async def test_start_stop_disabled(self) -> None:
        adapter = self._make_adapter()
        await adapter.start()  # No-op
        await adapter.stop()   # No-op
