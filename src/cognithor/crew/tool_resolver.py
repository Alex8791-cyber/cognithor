"""Resolve CrewAgent / CrewTask tool names against the MCP registry.

Wraps `cognithor.mcp.tool_registry_db.ToolRegistryDB` with the one helper
the Crew-Layer needs: 'give me every tool name'. The real registry groups
tools by role (planner/executor/browser/…) — we ask for role='all' to flatten.

Provides friendly 'did you mean' suggestions via difflib (stdlib, no new deps).

**Unwired by design until Task 11:** this module is imported and tested in
Task 7 but `resolve_tools` has no call site yet — `execute_task` (Task 11)
will call it to validate each agent's/task's tool list against the real
registry before routing through the Planner.
"""

from __future__ import annotations

import difflib
from typing import Any

from cognithor.crew.errors import ToolNotFoundError
from cognithor.i18n import t


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

    Raises ToolNotFoundError on first unknown name. When a close match is
    found, the error uses ``crew.errors.tool_suggestion`` (which includes
    the "did you mean" hint in the active locale). Otherwise it uses
    ``crew.errors.unknown_tool`` with the list of known tools.
    """
    available = available_tool_names(registry)
    for name in tool_names:
        if name in available:
            continue
        suggestion = did_you_mean(name, available)
        if suggestion is not None:
            msg = t("crew.errors.tool_suggestion", tool=name, suggestion=suggestion)
        else:
            msg = t(
                "crew.errors.unknown_tool",
                tool=name,
                known=", ".join(available) or "(none)",
            )
        raise ToolNotFoundError(msg)
    return tool_names
