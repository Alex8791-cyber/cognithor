"""Role behaviour registry for the Create / Operate / Live role system.

Agents have an explicit role that governs their capabilities:
- **orchestrator**: Extended Thinking ON, output NOT logged, can spawn agents
- **worker**: Extended Thinking OFF, everything logged, full MCP tool access
- **monitor**: Read-only tool access, logs everything, no spawning, no writing

Default role is ``"worker"`` for backward compatibility.
"""

from __future__ import annotations

from typing import Any, Literal

AgentRole = Literal["orchestrator", "worker", "monitor"]

ROLE_BEHAVIOURS: dict[AgentRole, dict[str, Any]] = {
    "orchestrator": {
        "extended_thinking": True,
        "log_output": False,
        "available_tool_tiers": ["reasoning", "delegation"],
        "can_spawn_agents": True,
    },
    "worker": {
        "extended_thinking": False,
        "log_output": True,
        "available_tool_tiers": ["execution", "memory", "search"],
        "can_spawn_agents": False,
    },
    "monitor": {
        "extended_thinking": False,
        "log_output": True,
        "available_tool_tiers": ["read_only"],
        "can_spawn_agents": False,
    },
}

# Tools a monitor is allowed to use (read-only subset)
MONITOR_ALLOWED_TOOLS: frozenset[str] = frozenset(
    {
        "read_file",
        "list_directory",
        "search_files",
        "find_in_files",
        "search_memory",
        "get_core_memory",
        "get_recent_episodes",
        "search_procedures",
        "memory_stats",
        "vault_search",
        "vault_read",
        "vault_list",
        "web_search",
        "search_and_read",
        "web_fetch",
        "web_news_search",
        "knowledge_synthesize",
        "knowledge_gaps",
        "knowledge_contradictions",
        "knowledge_timeline",
        "git_status",
        "git_diff",
        "git_log",
        "list_skills",
        "docker_ps",
        "docker_logs",
        "docker_inspect",
        "db_query",
        "db_schema",
        "email_read_inbox",
        "email_search",
        "email_summarize",
        "calendar_today",
        "calendar_upcoming",
        "calendar_check_availability",
        "list_reminders",
        "get_clipboard",
        "screenshot_desktop",
        "screenshot_region",
        "api_list",
        "get_entity",
        "analyze_document",
        "read_pdf",
        "read_docx",
        "read_ppt",
        "browser_extract",
        "browser_analyze",
        "browser_screenshot",
        "analyze_code",
    }
)


class AgentSpawnError(Exception):
    """Raised when an agent without spawn permission tries to spawn."""


class WriteBlockedError(Exception):
    """Raised when a monitor tries to use a write tool."""


def get_behaviour(role: AgentRole) -> dict[str, Any]:
    """Return the behaviour dict for the given role."""
    return ROLE_BEHAVIOURS[role]


def can_spawn(role: AgentRole) -> bool:
    """Return True if the role allows spawning sub-agents."""
    return bool(ROLE_BEHAVIOURS[role]["can_spawn_agents"])


def is_tool_allowed_for_role(tool_name: str, role: AgentRole) -> bool:
    """Check if a tool is allowed for the given role.

    - orchestrator and worker: all tools allowed (gatekeeper handles fine-grained)
    - monitor: only read-only tools
    """
    if role in ("orchestrator", "worker"):
        return True
    # monitor
    return tool_name in MONITOR_ALLOWED_TOOLS


def should_log_output(role: AgentRole) -> bool:
    """Return True if the role's output should be logged to conversation history."""
    return bool(ROLE_BEHAVIOURS[role]["log_output"])


def uses_extended_thinking(role: AgentRole) -> bool:
    """Return True if the role uses extended thinking."""
    return bool(ROLE_BEHAVIOURS[role]["extended_thinking"])
