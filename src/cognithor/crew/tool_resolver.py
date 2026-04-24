"""Resolve CrewAgent / CrewTask tool names against the MCP registry.

Wraps `cognithor.mcp.tool_registry_db.ToolRegistryDB` with the one helper
the Crew-Layer needs: 'give me every tool name'. The real registry groups
tools by role (planner/executor/browser/…) — we ask for role='all' to flatten.

Provides friendly 'did you mean' suggestions via difflib (stdlib, no new deps).
"""

from __future__ import annotations

import difflib
from typing import Any

from cognithor.crew.errors import ToolNotFoundError


def available_tool_names(registry: Any) -> list[str]:
    """Return every tool name known to the registry, flat.

    `registry` must be a `ToolRegistryDB` (or any duck-compatible object that
    exposes `get_tools_for_role(role: str) -> list[ToolInfo]` where each item
    has a `.name` attribute).
    """
    tools = registry.get_tools_for_role("all")
    return [t.name for t in tools]


def did_you_mean(name: str, candidates: list[str], cutoff: float = 0.6) -> str | None:
    """Return the closest match above cutoff, or None when nothing is close
    or when `name` is already in candidates.
    """
    if name in candidates:
        return None
    matches = difflib.get_close_matches(name, candidates, n=1, cutoff=cutoff)
    return matches[0] if matches else None


def resolve_tools(tool_names: list[str], *, registry: Any) -> list[str]:
    """Verify every tool name exists in the registry.

    Raises ToolNotFoundError on first unknown name, with a 'Meintest du ...?'
    suggestion when a close match exists.
    """
    available = available_tool_names(registry)
    for name in tool_names:
        if name in available:
            continue
        suggestion = did_you_mean(name, available)
        hint = f" Meintest du '{suggestion}'?" if suggestion else ""
        raise ToolNotFoundError(f"Tool '{name}' nicht in der Registry.{hint}")
    return tool_names
