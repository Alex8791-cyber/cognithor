"""Integration test: compiler.append_audit() must publish to TraceBus."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cognithor.crew.compiler import append_audit
from cognithor.crew.trace_bus import TraceBus


@pytest.mark.asyncio
async def test_append_audit_publishes_to_trace_bus() -> None:
    """append_audit() must invoke get_trace_bus().publish(record)."""
    fake_bus = MagicMock(spec=TraceBus)
    with patch("cognithor.crew.compiler._get_audit_trail", return_value=MagicMock()):
        with patch("cognithor.crew.compiler.get_trace_bus", return_value=fake_bus):
            append_audit(
                "crew_kickoff_started",
                trace_id="test-trace-1",
                n_tasks=4,
                process="SEQUENTIAL",
            )
    fake_bus.publish.assert_called_once()
    record = fake_bus.publish.call_args.args[0]
    assert (
        record.get("event_type") == "crew_kickoff_started"
        or record.get("event") == "crew_kickoff_started"
    )
    assert record.get("trace_id") == "test-trace-1" or record.get("session_id") == "test-trace-1"


@pytest.mark.asyncio
async def test_append_audit_publishes_even_when_audit_trail_is_none() -> None:
    """Even when AuditTrail is unavailable, the bus should still see events."""
    fake_bus = MagicMock(spec=TraceBus)
    with patch("cognithor.crew.compiler._get_audit_trail", return_value=None):
        with patch("cognithor.crew.compiler.get_trace_bus", return_value=fake_bus):
            append_audit("crew_kickoff_started", trace_id="t-2", n_tasks=1, process="SEQUENTIAL")
    fake_bus.publish.assert_called_once()
