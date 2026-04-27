"""Cognithor Agent SDK — build agents and tools in minimal code.

**Stability:** stable public API. Used at agent/tool **definition time**, not
runtime — the decorators construct dataclasses (`AgentDefinition`,
`ToolDefinition`, `HookDefinition`) that the rest of the system consumes.
Bump the major version when changing decorator signatures or definition shapes.

Quick Start::

    from cognithor.sdk import agent, tool, hook

    # Define a tool in 5 lines
    @tool(name="greet", description="Greet a user")
    async def greet(name: str = "World") -> str:
        return f"Hello, {name}!"

    # Define an agent in 10 lines
    @agent(
        name="greeter",
        description="A friendly greeting agent",
        tools=["greet"],
    )
    class GreeterAgent:
        async def on_message(self, message: str) -> str:
            return f"Received: {message}"

Architecture: §12.3 (Agent SDK)
"""

from cognithor.sdk.decorators import agent, hook, tool
from cognithor.sdk.definitions import AgentDefinition, HookDefinition, ToolDefinition
from cognithor.sdk.scaffold import scaffold_agent, scaffold_tool

__all__ = [
    "AgentDefinition",
    "HookDefinition",
    "ToolDefinition",
    "agent",
    "hook",
    "scaffold_agent",
    "scaffold_tool",
    "tool",
]
