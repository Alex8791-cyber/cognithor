"""HIM Agent — main orchestrator for OSINT investigations."""

from __future__ import annotations

import asyncio
from typing import Any

from jarvis.osint.collectors.arxiv import ArxivCollector
from jarvis.osint.collectors.crunchbase import CrunchbaseCollector
from jarvis.osint.collectors.github import GitHubCollector
from jarvis.osint.collectors.linkedin import LinkedInCollector
from jarvis.osint.collectors.scholar import ScholarCollector
from jarvis.osint.collectors.social import SocialCollector
from jarvis.osint.collectors.web import WebCollector
from jarvis.osint.evidence_aggregator import EvidenceAggregator
from jarvis.osint.gdpr_gatekeeper import GDPRGatekeeper
from jarvis.osint.him_reporter import HIMReporter
from jarvis.osint.models import (
    Evidence,
    Finding,
    GDPRViolationError,
    HIMReport,
    HIMRequest,
    VerificationStatus,
)
from jarvis.osint.trust_scorer import TrustScorer
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class HIMAgent:
    """Orchestrate OSINT investigations: GDPR -> Collect -> Aggregate -> Score -> Report."""

    def __init__(
        self,
        mcp_client: Any = None,
        github_token: str | None = None,
        collector_timeout: int = 30,
    ) -> None:
        self._mcp = mcp_client
        self._gdpr = GDPRGatekeeper()
        self._aggregator = EvidenceAggregator()
        self._scorer = TrustScorer()
        self._reporter = HIMReporter()
        self._collector_timeout = collector_timeout

        self._collectors = {
            "github": GitHubCollector(token=github_token),
            "web": WebCollector(mcp_client=mcp_client),
            "arxiv": ArxivCollector(),
            "scholar": ScholarCollector(),
            "linkedin": LinkedInCollector(),
            "crunchbase": CrunchbaseCollector(),
            "social": SocialCollector(),
        }

    async def run(self, request: HIMRequest) -> HIMReport:
        """Execute full investigation pipeline."""
        log.info("him_investigation_start", target=request.target_name[:30], depth=request.depth)

        # 1. GDPR check
        target_handle = request.target_github or request.target_name
        github_followers = 0
        try:
            gh = self._collectors["github"]
            if gh.is_available() and request.target_github:
                profile = await gh._fetch_with_retry(
                    f"{gh.BASE_URL}/users/{request.target_github}",
                    headers=gh._headers(),
                )
                github_followers = profile.get("followers", 0)
        except Exception:
            pass

        scope = self._gdpr.check(request, github_followers=github_followers)

        # Update web collector language from request
        web_collector = self._collectors.get("web")
        if web_collector and hasattr(web_collector, "_language"):
            web_collector._language = request.language

        # 2. Collect evidence in parallel
        all_evidence: list[Evidence] = []
        tasks = []
        for name, collector in self._collectors.items():
            if name not in scope.allowed_collectors:
                continue
            if not collector.is_available():
                continue
            tasks.append(self._collect_with_timeout(collector, target_handle, request.claims))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, list):
                    all_evidence.extend(result)
                elif isinstance(result, Exception):
                    log.debug("him_collector_exception", error=str(result)[:100])

        # 3. Aggregate + cross-verify
        claim_results = self._aggregator.aggregate(all_evidence, request.claims)

        # 4. Score
        trust_score = self._scorer.score(claim_results, all_evidence)

        # 5. Build findings and red flags
        findings: list[Finding] = []
        red_flags: list[str] = []

        for cr in claim_results:
            if cr.status == VerificationStatus.CONTRADICTED:
                red_flags.append(f"{cr.claim}: {cr.explanation}")
                findings.append(
                    Finding(
                        title=f"Contradicted: {cr.claim[:50]}",
                        description=cr.explanation,
                        severity="red_flag",
                        source=", ".join(cr.sources_used),
                    )
                )
            elif cr.status == VerificationStatus.PARTIAL:
                findings.append(
                    Finding(
                        title=f"Partially verified: {cr.claim[:50]}",
                        description=cr.explanation,
                        severity="warning",
                        source=", ".join(cr.sources_used),
                    )
                )
            elif cr.status == VerificationStatus.UNVERIFIED:
                red_flags.append(f"{cr.claim}: not verified by any source")

        # 6. Generate summary
        summary = self._generate_summary(request, trust_score, claim_results, all_evidence)
        recommendation = self._generate_recommendation(trust_score)

        # 7. Build report
        report = HIMReport(
            target=request.target_name,
            target_type=request.target_type,
            depth=request.depth,
            trust_score=trust_score,
            claims=claim_results,
            key_findings=findings,
            red_flags=red_flags,
            summary=summary,
            recommendation=recommendation,
            raw_evidence_count=len(all_evidence),
        )

        # 8. Sign report
        md_content = self._reporter.render_markdown(report)
        report.report_signature = self._reporter.sign_report(md_content)

        # 9. Save to vault
        await self._save_to_vault(report, md_content)

        log.info(
            "him_investigation_complete",
            target=request.target_name[:30],
            score=trust_score.total,
            label=trust_score.label,
            evidence=len(all_evidence),
        )
        return report

    async def _collect_with_timeout(
        self, collector: Any, target: str, claims: list[str]
    ) -> list[Evidence]:
        try:
            return await asyncio.wait_for(
                collector.collect(target, claims),
                timeout=self._collector_timeout,
            )
        except asyncio.TimeoutError:
            log.warning("him_collector_timeout", source=collector.source_name)
            return []

    async def _save_to_vault(self, report: HIMReport, md_content: str) -> None:
        if not self._mcp:
            return
        try:
            await self._mcp.call_tool(
                "vault_save",
                {
                    "title": f"HIM Report: {report.target}",
                    "content": md_content,
                    "tags": "osint, him, investigation",
                    "folder": "recherchen/osint",
                    "sources": "",
                },
            )
        except Exception:
            log.debug("him_vault_save_failed", exc_info=True)

    def _generate_summary(self, request, trust_score, claims, evidence) -> str:
        confirmed = sum(1 for c in claims if c.status == VerificationStatus.CONFIRMED)
        contradicted = sum(1 for c in claims if c.status == VerificationStatus.CONTRADICTED)
        if not evidence:
            return f"No data available for {request.target_name}."
        return (
            f"Investigation of '{request.target_name}' ({request.target_type}) "
            f"based on {len(evidence)} evidence items from {len(set(e.source_type for e in evidence))} sources. "
            f"Trust Score: {trust_score.total}/100 ({trust_score.label}). "
            f"{confirmed}/{len(claims)} claims confirmed"
            + (f", {contradicted} contradicted" if contradicted else "")
            + "."
        )

    def _generate_recommendation(self, trust_score) -> str:
        if trust_score.label == "high":
            return "Credentials appear credible. Proceed with normal engagement."
        if trust_score.label == "mixed":
            return "Some claims could not be fully verified. Request additional evidence before deep engagement."
        return (
            "Significant credibility concerns. Verify claims independently before any commitment."
        )
