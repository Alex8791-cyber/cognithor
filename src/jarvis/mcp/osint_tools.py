"""MCP bridge for HIM OSINT tools."""

from __future__ import annotations

import json
from typing import Any

from jarvis.osint.him_agent import HIMAgent
from jarvis.osint.models import HIMRequest, GDPRViolationError
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class OsintTools:
    def __init__(self, mcp_client: Any, config: Any = None) -> None:
        osint_cfg = getattr(config, "osint", None)
        self._agent = HIMAgent(
            mcp_client=mcp_client,
            github_token=getattr(osint_cfg, "github_token", "") if osint_cfg else "",
            collector_timeout=getattr(osint_cfg, "collector_timeout", 30) if osint_cfg else 30,
        )

    async def investigate_person(
        self,
        target_name: str,
        target_github: str = "",
        claims: str = "",
        depth: str = "standard",
        justification: str = "",
    ) -> str:
        return await self._run(
            target_name=target_name,
            target_github=target_github or None,
            claims=claims,
            target_type="person",
            depth=depth,
            justification=justification,
        )

    async def investigate_project(
        self,
        target_name: str,
        target_github: str = "",
        claims: str = "",
        justification: str = "",
    ) -> str:
        return await self._run(
            target_name=target_name,
            target_github=target_github or None,
            claims=claims,
            target_type="project",
            depth="standard",
            justification=justification,
        )

    async def investigate_org(
        self,
        target_name: str,
        claims: str = "",
        justification: str = "",
    ) -> str:
        return await self._run(
            target_name=target_name,
            target_github=None,
            claims=claims,
            target_type="org",
            depth="standard",
            justification=justification,
        )

    async def _run(
        self, *, target_name, target_github, claims, target_type, depth, justification
    ) -> str:
        try:
            claims_list = [c.strip() for c in claims.split(",") if c.strip()] if claims else []
            request = HIMRequest(
                target_name=target_name,
                target_github=target_github,
                claims=claims_list,
                target_type=target_type,
                depth=depth,
                requester_justification=justification or "MCP tool invocation",
            )
            report = await self._agent.run(request)
            from jarvis.osint.him_reporter import HIMReporter

            return (
                HIMReporter().render_quick(report) + "\n\n" + HIMReporter().render_markdown(report)
            )
        except GDPRViolationError as e:
            return f"GDPR VIOLATION: {e}"
        except Exception as e:
            log.debug("osint_tool_error", exc_info=True)
            return f"Investigation failed: {e}"


def register_osint_tools(mcp_client: Any, config: Any = None) -> OsintTools:
    """Register OSINT investigation MCP tools."""
    osint_cfg = getattr(config, "osint", None)
    if osint_cfg and not getattr(osint_cfg, "enabled", True):
        log.info("osint_tools_disabled")
        return None

    tools = OsintTools(mcp_client, config)

    mcp_client.register_builtin_handler(
        "investigate_person",
        tools.investigate_person,
        description="OSINT investigation of a person: collects evidence from GitHub, web, arXiv, cross-verifies claims, computes Trust Score (0-100)",
        input_schema={
            "type": "object",
            "properties": {
                "target_name": {
                    "type": "string",
                    "description": "Name or GitHub username of the person",
                },
                "target_github": {
                    "type": "string",
                    "description": "GitHub username (optional)",
                    "default": "",
                },
                "claims": {
                    "type": "string",
                    "description": "Comma-separated claims to verify",
                    "default": "",
                },
                "depth": {
                    "type": "string",
                    "enum": ["quick", "standard", "deep"],
                    "default": "standard",
                },
                "justification": {
                    "type": "string",
                    "description": "Why this investigation is needed (GDPR)",
                    "default": "",
                },
            },
            "required": ["target_name"],
        },
    )

    mcp_client.register_builtin_handler(
        "investigate_project",
        tools.investigate_project,
        description="OSINT investigation of a project: checks GitHub repos, web mentions, funding claims",
        input_schema={
            "type": "object",
            "properties": {
                "target_name": {"type": "string", "description": "Project name"},
                "target_github": {
                    "type": "string",
                    "description": "GitHub repo (optional)",
                    "default": "",
                },
                "claims": {
                    "type": "string",
                    "description": "Comma-separated claims to verify",
                    "default": "",
                },
                "justification": {
                    "type": "string",
                    "description": "Why this investigation is needed",
                    "default": "",
                },
            },
            "required": ["target_name"],
        },
    )

    mcp_client.register_builtin_handler(
        "investigate_org",
        tools.investigate_org,
        description="OSINT investigation of an organization: checks web presence, funding, team",
        input_schema={
            "type": "object",
            "properties": {
                "target_name": {"type": "string", "description": "Organization name"},
                "claims": {
                    "type": "string",
                    "description": "Comma-separated claims to verify",
                    "default": "",
                },
                "justification": {
                    "type": "string",
                    "description": "Why this investigation is needed",
                    "default": "",
                },
            },
            "required": ["target_name"],
        },
    )

    log.info(
        "osint_tools_registered",
        tools=["investigate_person", "investigate_project", "investigate_org"],
    )
    return tools
