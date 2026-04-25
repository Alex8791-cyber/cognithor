"""AssistantAgent — autogen-agentchat==0.7.5 source-compat shim.

The 17-field signature mirrors:
  https://microsoft.github.io/autogen/stable//reference/python/autogen_agentchat.agents.html

Verified empirically via `inspect.signature(autogen_agentchat.agents.AssistantAgent.__init__)`
against `autogen-agentchat==0.7.5` — see tests/test_compat/test_autogen/test_signature_compat.py.

Cognithor's translation (no AutoGen code copied verbatim):
- name, system_message, description: stored, used to build the Cognithor
  CrewAgent's role/backstory/goal.
- model_client: wraps cognithor.core.model_router via OpenAIChatCompletionClient
  shim (see models/__init__.py).
- tools, workbench: bridged into MCP tool registry inside _bridge.py.
- run / run_stream: delegate to cognithor.crew.Crew(...).kickoff_async().
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from cognithor.compat.autogen._bridge import TaskResult


_DEFAULT_DESCRIPTION = "An agent that provides assistance with ability to use tools."
_DEFAULT_SYSTEM_MESSAGE = (
    "You are a helpful AI assistant. Solve tasks using your tools. "
    "Reply with TERMINATE when the task has been completed."
)


class AssistantAgent:
    """AutoGen-AgentChat-compatible AssistantAgent.

    Signature mirrors autogen_agentchat.agents.AssistantAgent (MIT licensed).
    Internally delegates to Cognithor's PGE Executor via cognithor.crew.

    Reference:
    https://microsoft.github.io/autogen/stable//reference/python/autogen_agentchat.agents.html
    """

    def __init__(
        self,
        name: str,
        model_client: Any,
        *,
        tools: Sequence[Any] | None = None,
        workbench: Any | None = None,
        handoffs: list[Any] | None = None,
        model_context: Any | None = None,
        description: str = _DEFAULT_DESCRIPTION,
        system_message: str | None = _DEFAULT_SYSTEM_MESSAGE,
        model_client_stream: bool = False,
        reflect_on_tool_use: bool | None = None,
        max_tool_iterations: int = 1,
        tool_call_summary_format: str = "{result}",
        tool_call_summary_formatter: Any | None = None,
        output_content_type: Any | None = None,
        output_content_type_format: str | None = None,
        memory: Sequence[Any] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.model_client = model_client
        self.tools = tools
        self.workbench = workbench
        self.handoffs = handoffs
        self.model_context = model_context
        self.description = description
        self.system_message = system_message
        self.model_client_stream = model_client_stream
        self.reflect_on_tool_use = reflect_on_tool_use
        self.max_tool_iterations = max_tool_iterations
        self.tool_call_summary_format = tool_call_summary_format
        self.tool_call_summary_formatter = tool_call_summary_formatter
        self.output_content_type = output_content_type
        self.output_content_type_format = output_content_type_format
        self.memory = memory
        self.metadata = metadata or {}

    async def run(self, *, task: str | Sequence[Any]) -> TaskResult:
        """Run a single 1-shot task. Maps to cognithor.crew.Crew.kickoff_async()."""
        from cognithor.compat.autogen._bridge import run_single_task

        return await run_single_task(self, task)

    def run_stream(self, *, task: str | Sequence[Any]) -> AsyncIterator[Any]:
        """Stream events from a single 1-shot task."""
        from cognithor.compat.autogen._bridge import stream_single_task

        return stream_single_task(self, task)
