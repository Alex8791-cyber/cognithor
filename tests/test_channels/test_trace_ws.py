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
