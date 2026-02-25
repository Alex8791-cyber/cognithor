"""Jarvis MCP module — Client, Server, Resources, Prompts, Discovery, Bridge."""

from jarvis.mcp.client import JarvisMCPClient, ToolCallResult
from jarvis.mcp.server import (
    JarvisMCPServer,
    MCPServerConfig,
    MCPServerMode,
    MCPToolDef,
    MCPResource,
    MCPResourceTemplate,
    MCPPrompt,
    MCPPromptArgument,
)
from jarvis.mcp.bridge import MCPBridge
from jarvis.mcp.discovery import AgentCard, DiscoveryManager
from jarvis.mcp.resources import JarvisResourceProvider
from jarvis.mcp.prompts import JarvisPromptProvider

__all__ = [
    # Client (bestehend)
    "JarvisMCPClient",
    "ToolCallResult",
    # Server (v15 neu)
    "JarvisMCPServer",
    "MCPServerConfig",
    "MCPServerMode",
    "MCPToolDef",
    "MCPResource",
    "MCPResourceTemplate",
    "MCPPrompt",
    "MCPPromptArgument",
    # Bridge (v15 neu)
    "MCPBridge",
    # Discovery (v15 neu)
    "AgentCard",
    "DiscoveryManager",
    # Providers (v15 neu)
    "JarvisResourceProvider",
    "JarvisPromptProvider",
]
