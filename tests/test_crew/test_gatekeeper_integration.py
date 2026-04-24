"""Task 12 - Gatekeeper integration lock.

The Planner already invokes the Gatekeeper internally when it plans a tool
call (``cognithor.core.gatekeeper.classify()`` returns a ``RiskLevel``). The
Crew-Layer does NOT bypass that path - it merely exposes it. This test
pins the contract: if the Planner raises because Gatekeeper classified a
tool as RED, the exception must surface through ``compile_and_run_async``
unmodified, NOT be swallowed.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from cognithor.crew import CrewAgent, CrewTask
from cognithor.crew.compiler import compile_and_run_async
from cognithor.crew.errors import CrewError
from cognithor.crew.process import CrewProcess


async def test_gatekeeper_red_tool_blocks_execution():
    """When an agent lists a tool that Gatekeeper classifies as RED, the
    task must fail-closed unless explicit approval is configured.
    """
    agent = CrewAgent(role="deleter", goal="delete", tools=["delete_all"])
    task = CrewTask(description="x", expected_output="y", agent=agent)

    mock_planner = MagicMock()
    # Simulate Planner raising when Gatekeeper denies the tool.
    mock_planner.formulate_response = AsyncMock(
        side_effect=CrewError("Gatekeeper RED: 'delete_all' blocked")
    )
    mock_registry = MagicMock()
    fake_tool = MagicMock()
    fake_tool.name = "delete_all"
    mock_registry.get_tools_for_role.return_value = [fake_tool]

    with pytest.raises(CrewError, match="Gatekeeper"):
        await compile_and_run_async(
            agents=[agent],
            tasks=[task],
            process=CrewProcess.SEQUENTIAL,
            inputs=None,
            registry=mock_registry,
            planner=mock_planner,
        )
