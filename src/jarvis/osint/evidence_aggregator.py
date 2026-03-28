"""Evidence Aggregator — cross-verification and claim scoring."""
from __future__ import annotations

import re

from jarvis.osint.models import (
    ClaimResult,
    ClaimType,
    Evidence,
    VerificationStatus,
)
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

_CLAIM_KEYWORDS: dict[ClaimType, list[str]] = {
    ClaimType.EMPLOYMENT: ["works at", "employed", "position", "role at", "engineer at", "researcher at", "staff"],
    ClaimType.EDUCATION: ["phd", "doctorate", "studied", "graduate", "degree", "university", "stanford", "mit"],
    ClaimType.TECHNICAL: ["built", "created", "developed", "authored", "implemented", "designed", "architect"],
    ClaimType.FUNDING: ["backed", "funded", "raised", "seed", "series", "investor", "grant"],
    ClaimType.AFFILIATION: ["member", "affiliated", "associated", "collaborator", "partner", "advisor"],
    ClaimType.ACHIEVEMENT: ["award", "publication", "published", "prize", "won", "recognized"],
}

_SOURCE_PRIORITY: dict[str, float] = {
    "github": 0.9,
    "arxiv": 0.85,
    "scholar": 0.8,
    "crunchbase": 0.75,
    "web": 0.6,
    "linkedin": 0.4,
    "social": 0.3,
}

# Confidence caps by number of independent sources
_CONFIDENCE_CAPS = {1: 0.6, 2: 0.8}
_CONFIDENCE_CAP_3PLUS = 0.95
_SELF_REPORT_CAP = 0.4
_CONTRADICTION_PENALTY = 0.5


class EvidenceAggregator:
    """Cross-verify evidence and produce ClaimResults."""

    def classify_claim(self, claim: str) -> ClaimType:
        claim_lower = claim.lower()
        best_type = ClaimType.AFFILIATION
        best_count = 0
        for ct, keywords in _CLAIM_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in claim_lower)
            if matches > best_count:
                best_count = matches
                best_type = ct
        return best_type

    def aggregate(
        self, all_evidence: list[Evidence], claims: list[str]
    ) -> list[ClaimResult]:
        results: list[ClaimResult] = []
        for claim in claims:
            claim_type = self.classify_claim(claim)
            claim_lower = claim.lower()

            # Find relevant evidence for this claim
            relevant: list[Evidence] = []
            for ev in all_evidence:
                content_lower = ev.content.lower()
                claim_words = [w for w in claim_lower.split() if len(w) > 3]
                if any(w in content_lower for w in claim_words):
                    relevant.append(ev)

            if not relevant:
                results.append(ClaimResult(
                    claim=claim,
                    claim_type=claim_type,
                    status=VerificationStatus.UNVERIFIED,
                    confidence=0.0,
                    evidence=[],
                    sources_used=[],
                    explanation="No relevant evidence found",
                ))
                continue

            # Count independent source types
            source_types = set(ev.source_type for ev in relevant)
            n_sources = len(source_types)

            # Confidence cap by source count
            if n_sources >= 3:
                cap = _CONFIDENCE_CAP_3PLUS
            else:
                cap = _CONFIDENCE_CAPS.get(n_sources, 0.6)

            # Self-report only?
            self_report_only = source_types <= {"linkedin", "social"}
            if self_report_only:
                cap = _SELF_REPORT_CAP

            # Weighted confidence from evidence
            total_weight = 0.0
            weighted_conf = 0.0
            for ev in relevant:
                priority = _SOURCE_PRIORITY.get(ev.source_type, 0.5)
                weighted_conf += ev.confidence * priority
                total_weight += priority

            raw_confidence = weighted_conf / total_weight if total_weight else 0.0
            confidence = min(raw_confidence, cap)

            # Check for contradictions (negative signals)
            # Use word-boundary patterns to avoid false positives on "notable", "another" etc.
            has_contradiction = any(
                re.search(r'\bnot\b.{0,30}\b(member|found|affiliated|associated|employed|verified)\b', ev.content.lower())
                or re.search(r'\bno (mention|evidence|record|affiliation|membership|connection)\b', ev.content.lower())
                or re.search(r'\bnot in\b.{0,20}\borg', ev.content.lower())
                for ev in relevant
                if ev.source_type in ("github", "web")
            )
            if has_contradiction:
                confidence = max(0.0, confidence - _CONTRADICTION_PENALTY)

            # Determine status
            if has_contradiction and confidence < 0.3:
                status = VerificationStatus.CONTRADICTED
            elif confidence >= 0.7:
                status = VerificationStatus.CONFIRMED
            elif confidence >= 0.4:
                status = VerificationStatus.PARTIAL
            else:
                status = VerificationStatus.UNVERIFIED

            results.append(ClaimResult(
                claim=claim,
                claim_type=claim_type,
                status=status,
                confidence=round(confidence, 2),
                evidence=relevant,
                sources_used=list(source_types),
                explanation=self._explain(status, n_sources, has_contradiction, self_report_only),
            ))

        return results

    def _explain(
        self, status: VerificationStatus, n_sources: int,
        has_contradiction: bool, self_report: bool
    ) -> str:
        if status == VerificationStatus.CONFIRMED:
            return f"Confirmed by {n_sources} independent source(s)"
        if status == VerificationStatus.CONTRADICTED:
            return "Evidence contradicts this claim"
        if self_report:
            return "Only self-reported (LinkedIn/social), not independently verified"
        if status == VerificationStatus.PARTIAL:
            return f"Partially supported by {n_sources} source(s)"
        return "No sufficient evidence found"
