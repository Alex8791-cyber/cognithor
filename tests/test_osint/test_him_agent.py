"""Tests for HIM Agent orchestrator."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from jarvis.osint.him_agent import HIMAgent
from jarvis.osint.models import HIMRequest, GDPRViolationError


@pytest.mark.asyncio
async def test_full_investigation_flow(terry_case):
    mock_mcp = AsyncMock()
    mock_result = MagicMock()
    mock_result.is_error = False
    mock_result.content = "Terry is building Agent Nexus, an A2A protocol layer."
    mock_mcp.call_tool = AsyncMock(return_value=mock_result)

    agent = HIMAgent(mcp_client=mock_mcp)
    request = HIMRequest(**terry_case["request"])

    with patch(
        "jarvis.osint.collectors.github.GitHubCollector._fetch_with_retry",
        new_callable=AsyncMock,
    ) as mock_gh:
        mock_gh.side_effect = [
            terry_case["mock_github_response"],
            [
                {
                    "name": "agent-nexus",
                    "description": "A2A protocol",
                    "stargazers_count": 8,
                    "updated_at": "2025-12-01T00:00:00Z",
                    "html_url": "https://github.com/x/y",
                    "fork": False,
                    "language": "Python",
                }
            ],
            [],  # orgs
        ]
        report = await agent.run(request)

    assert report is not None
    assert 0 <= report.trust_score.total <= 100
    assert report.report_signature


@pytest.mark.asyncio
async def test_gdpr_blocked_investigation():
    agent = HIMAgent()
    request = HIMRequest(
        target_name="Private Person",
        requester_justification="short",
    )
    with pytest.raises(GDPRViolationError):
        await agent.run(request)


@pytest.mark.asyncio
async def test_all_collectors_fail_graceful():
    agent = HIMAgent(mcp_client=None)
    request = HIMRequest(
        target_name="Nobody",
        requester_justification="Testing graceful degradation",
    )
    with (
        patch(
            "jarvis.osint.collectors.github.GitHubCollector._fetch_with_retry",
            new_callable=AsyncMock,
            side_effect=Exception("network down"),
        ),
        patch(
            "jarvis.osint.collectors.arxiv.ArxivCollector.collect",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        report = await agent.run(request)
    # With no evidence and no claims, score is very low but not necessarily 0
    # because transparency defaults to 100% when no claims exist.
    assert report.trust_score.total <= 15
    assert "No data" in report.summary or report.raw_evidence_count == 0
