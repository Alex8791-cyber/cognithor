"""Tests for cognithor.channels.webui Trace-UI WebSocket additions.

These tests use the in-memory TraceBus + a mock WebSocket session to
verify subscribe/unsubscribe/cleanup behaviour without actually starting
a uvicorn server.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


def test_webui_module_imports() -> None:
    """Sanity: webui module imports without error after our additions."""
    import cognithor.channels.webui  # noqa: F401


@pytest.mark.asyncio
async def test_trace_subscriber_state_creates_handles_dict() -> None:
    """A WebSocket session that opts into trace events gets a per-session handle dict."""
    from cognithor.channels.webui import TraceSubscriberState

    state = TraceSubscriberState()
    assert state.lifecycle_handle is None
    assert state.topic_handles == {}


@pytest.mark.asyncio
async def test_trace_subscriber_state_clear_unsubscribes_everything() -> None:
    """clear_all() should unsubscribe lifecycle + every topic from the bus."""
    from cognithor.channels.webui import TraceSubscriberState
    from cognithor.crew.trace_bus import TraceBus, get_trace_bus

    bus = get_trace_bus()
    state = TraceSubscriberState()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=10)

    state.lifecycle_handle = bus.subscribe_lifecycle(queue)
    state.topic_handles["trace-x"] = bus.subscribe("trace-x", queue)
    assert len(bus._subscribers) == 2  # noqa: SLF001 — internal check

    state.clear_all(bus)
    assert state.lifecycle_handle is None
    assert state.topic_handles == {}


@pytest.mark.asyncio
async def test_handle_lifecycle_subscribe_for_owner_creates_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    from cognithor.channels.webui import (
        TraceSubscriberState,
        handle_trace_subscribe_message,
    )
    from cognithor.crew.trace_bus import get_trace_bus

    monkeypatch.setenv("COGNITHOR_OWNER_USER_ID", "owner")
    bus = get_trace_bus()
    state = TraceSubscriberState()
    sender = AsyncMock()

    err = await handle_trace_subscribe_message(
        message={"type": "crew_lifecycle_subscribe"},
        state=state,
        bus=bus,
        user_id="owner",
        sender=sender,
    )
    assert err is None
    assert state.lifecycle_handle is not None
    state.clear_all(bus)


@pytest.mark.asyncio
async def test_handle_topic_subscribe_creates_topic_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    from cognithor.channels.webui import (
        TraceSubscriberState,
        handle_trace_subscribe_message,
    )
    from cognithor.crew.trace_bus import get_trace_bus

    monkeypatch.setenv("COGNITHOR_OWNER_USER_ID", "owner")
    bus = get_trace_bus()
    state = TraceSubscriberState()
    sender = AsyncMock()

    err = await handle_trace_subscribe_message(
        message={"type": "crew_subscribe", "trace_id": "abc"},
        state=state,
        bus=bus,
        user_id="owner",
        sender=sender,
    )
    assert err is None
    assert "abc" in state.topic_handles
    state.clear_all(bus)


@pytest.mark.asyncio
async def test_handle_topic_unsubscribe_removes_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    from cognithor.channels.webui import (
        TraceSubscriberState,
        handle_trace_subscribe_message,
    )
    from cognithor.crew.trace_bus import get_trace_bus

    monkeypatch.setenv("COGNITHOR_OWNER_USER_ID", "owner")
    bus = get_trace_bus()
    state = TraceSubscriberState()
    sender = AsyncMock()

    await handle_trace_subscribe_message(
        message={"type": "crew_subscribe", "trace_id": "xyz"},
        state=state,
        bus=bus,
        user_id="owner",
        sender=sender,
    )
    assert "xyz" in state.topic_handles

    err = await handle_trace_subscribe_message(
        message={"type": "crew_unsubscribe", "trace_id": "xyz"},
        state=state,
        bus=bus,
        user_id="owner",
        sender=sender,
    )
    assert err is None
    assert "xyz" not in state.topic_handles


@pytest.mark.asyncio
async def test_handle_subscribe_rejects_non_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    from cognithor.channels.webui import (
        TraceSubscriberState,
        handle_trace_subscribe_message,
    )
    from cognithor.crew.trace_bus import get_trace_bus

    monkeypatch.setenv("COGNITHOR_OWNER_USER_ID", "real-owner")
    bus = get_trace_bus()
    state = TraceSubscriberState()
    sender = AsyncMock()

    err = await handle_trace_subscribe_message(
        message={"type": "crew_subscribe", "trace_id": "abc"},
        state=state,
        bus=bus,
        user_id="guest",
        sender=sender,
    )
    assert err == "owner_only"
    assert state.topic_handles == {}
    sender.assert_called_once()
    sent = sender.call_args.args[0]
    assert sent["type"] == "error"
    assert sent["code"] == "owner_only"


@pytest.mark.asyncio
async def test_handle_unknown_subscribe_type_returns_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from cognithor.channels.webui import (
        TraceSubscriberState,
        handle_trace_subscribe_message,
    )
    from cognithor.crew.trace_bus import get_trace_bus

    monkeypatch.setenv("COGNITHOR_OWNER_USER_ID", "owner")
    bus = get_trace_bus()
    state = TraceSubscriberState()
    sender = AsyncMock()

    err = await handle_trace_subscribe_message(
        message={"type": "crew_unknown_action"},
        state=state,
        bus=bus,
        user_id="owner",
        sender=sender,
    )
    assert err == "unknown_message_type"
