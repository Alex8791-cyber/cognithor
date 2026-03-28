"""HIM Reporter — renders Markdown and JSON reports."""
from __future__ import annotations

import hashlib

from jarvis.osint.models import HIMReport, VerificationStatus


class HIMReporter:
    """Generate formatted reports from HIMReport objects."""

    def render_markdown(self, report: HIMReport) -> str:
        lines = [
            f"# HIM Report: {report.target} | Trust Score: {report.trust_score.total}/100",
            "",
            f"**Label:** {report.trust_score.label.upper()} | "
            f"**Type:** {report.target_type} | "
            f"**Depth:** {report.depth} | "
            f"**Evidence:** {report.raw_evidence_count} items",
            "",
            "## Summary",
            report.summary or "No summary available.",
            "",
            "## Claim Verification",
            "",
            "| Claim | Status | Confidence | Sources |",
            "|-------|--------|------------|---------|",
        ]
        for c in report.claims:
            status_icon = {
                VerificationStatus.CONFIRMED: "[OK]",
                VerificationStatus.PARTIAL: "[~]",
                VerificationStatus.UNVERIFIED: "[?]",
                VerificationStatus.CONTRADICTED: "[X]",
            }.get(c.status, "[-]")
            lines.append(
                f"| {c.claim[:60]} | {status_icon} {c.status.value} | "
                f"{c.confidence:.0%} | {', '.join(c.sources_used)} |"
            )

        if report.key_findings:
            lines.extend(["", "## Key Findings", ""])
            for f in report.key_findings:
                icon = {"info": "[i]", "warning": "[!]", "red_flag": "[!!]"}.get(f.severity, "-")
                lines.append(f"- {icon} **{f.title}**: {f.description} _(Source: {f.source})_")

        if report.red_flags:
            lines.extend(["", "## Red Flags", ""])
            for rf in report.red_flags:
                lines.append(f"- [!!] {rf}")

        lines.extend([
            "",
            "## Trust Score Breakdown",
            "",
            "| Dimension | Score | Weight |",
            "|-----------|-------|--------|",
            f"| Claim Accuracy | {report.trust_score.claim_accuracy:.0f} | 35% |",
            f"| Source Diversity | {report.trust_score.source_diversity:.0f} | 20% |",
            f"| Technical Substance | {report.trust_score.technical_substance:.0f} | 25% |",
            f"| Transparency | {report.trust_score.transparency:.0f} | 10% |",
            f"| Activity Recency | {report.trust_score.activity_recency:.0f} | 10% |",
            f"| **Total** | **{report.trust_score.total}** | |",
            "",
            "## Recommendation",
            report.recommendation or "No recommendation.",
            "",
            "---",
            f"*Generated: {report.generated_at.isoformat()} | "
            f"Report-ID: {report.report_id} | "
            f"Signature: {report.report_signature[:16]}...*",
        ])
        return "\n".join(lines)

    def render_json(self, report: HIMReport) -> str:
        return report.model_dump_json(indent=2)

    def render_quick(self, report: HIMReport) -> str:
        ts = report.trust_score
        status_line = f"Trust Score: {ts.total}/100 ({ts.label.upper()})"
        claims_line = ", ".join(
            f"{c.claim[:30]}={c.status.value}" for c in report.claims[:3]
        )
        flags = f"Red Flags: {len(report.red_flags)}" if report.red_flags else "No red flags"
        return f"{report.target} | {status_line} | Claims: [{claims_line}] | {flags}"

    @staticmethod
    def sign_report(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()
