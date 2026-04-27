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
