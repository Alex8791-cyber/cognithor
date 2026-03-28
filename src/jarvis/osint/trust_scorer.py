"""Trust Scorer — 5-dimension weighted scoring (0-100)."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

from jarvis.osint.models import (
    ClaimResult,
    Evidence,
    TrustScore,
    VerificationStatus,
)

_WEIGHTS = {
    "claim_accuracy": 0.35,
    "source_diversity": 0.20,
    "technical_substance": 0.25,
    "transparency": 0.10,
    "activity_recency": 0.10,
}

_STATUS_POINTS = {
    VerificationStatus.CONFIRMED: 100,
    VerificationStatus.PARTIAL: 50,
    VerificationStatus.UNVERIFIED: 20,
    VerificationStatus.CONTRADICTED: -50,
}

_DIVERSITY_MAP = {0: 0, 1: 30, 2: 55, 3: 75}


class TrustScorer:
    """Compute a 0-100 Trust Score from claims and evidence."""

    def score(
        self, claims: list[ClaimResult], all_evidence: list[Evidence]
    ) -> TrustScore:
        ca = self._claim_accuracy(claims)
        sd = self._source_diversity(all_evidence)
        ts = self._technical_substance(all_evidence)
        tr = self._transparency(claims)
        ar = self._activity_recency(all_evidence)

        total = (
            ca * _WEIGHTS["claim_accuracy"]
            + sd * _WEIGHTS["source_diversity"]
            + ts * _WEIGHTS["technical_substance"]
            + tr * _WEIGHTS["transparency"]
            + ar * _WEIGHTS["activity_recency"]
        )
        total_int = max(0, min(100, int(round(total))))

        return TrustScore(
            total=total_int,
            label=self._label(total_int),
            claim_accuracy=round(ca, 1),
            source_diversity=round(sd, 1),
            technical_substance=round(ts, 1),
            transparency=round(tr, 1),
            activity_recency=round(ar, 1),
        )

    def _label(self, score: int) -> str:
        if score >= 75:
            return "high"
        if score >= 40:
            return "mixed"
        return "low"

    def _claim_accuracy(self, claims: list[ClaimResult]) -> float:
        if not claims:
            return 0.0
        points = [_STATUS_POINTS.get(c.status, 0) for c in claims]
        avg = sum(points) / len(points)
        return max(0.0, min(100.0, avg))

    def _source_diversity(self, evidence: list[Evidence]) -> float:
        source_types = set(e.source_type for e in evidence if e.confidence > 0.3)
        n = len(source_types)
        if n >= 4:
            return min(100, 90 + (n - 4) * 5)
        return _DIVERSITY_MAP.get(n, 0)

    def _technical_substance(self, evidence: list[Evidence]) -> float:
        score = 0.0
        sources = {e.source_type for e in evidence}
        contents = " ".join(e.content.lower() for e in evidence)

        if "github" in sources and ("repo" in contents or "commit" in contents):
            score += 30
        if "arxiv" in sources or "scholar" in sources:
            score += 25
        if "github" in sources and "stars" in contents:
            score += 20
        if sources <= {"linkedin", "social", "web"}:
            if "linkedin" in sources:
                score = max(score, 5)
        return min(100.0, score)

    def _transparency(self, claims: list[ClaimResult]) -> float:
        if not claims:
            return 100.0
        transparent_count = 0
        for c in claims:
            if c.status == VerificationStatus.CONFIRMED:
                transparent_count += 1
            elif c.status == VerificationStatus.PARTIAL:
                # Partial = possible namedropping
                pass
            elif c.status == VerificationStatus.CONTRADICTED:
                return 0.0
        ratio = transparent_count / len(claims)
        if ratio >= 0.8:
            return 100.0
        if ratio >= 0.5:
            return 60.0
        return 30.0

    def _activity_recency(self, evidence: list[Evidence]) -> float:
        now = datetime.now(timezone.utc)
        most_recent = None
        for e in evidence:
            # Use collected_at as baseline
            if e.collected_at:
                ca = e.collected_at
                if most_recent is None or ca > most_recent:
                    most_recent = ca
            # Check content for date signals (may be more precise)
            if e.collected_at and e.collected_at > (most_recent or datetime.min.replace(tzinfo=timezone.utc)):
                content = e.content
                for date_str in _extract_date_hints(content):
                    try:
                        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                        if most_recent is None or dt > most_recent:
                            most_recent = dt
                    except (ValueError, TypeError):
                        pass
        if most_recent is None:
            return 10.0
        age = now - most_recent
        if age < timedelta(days=90):
            return 100.0
        if age < timedelta(days=365):
            return 70.0
        if age < timedelta(days=365 * 3):
            return 40.0
        return 10.0


def _extract_date_hints(content: str) -> list[str]:
    """Extract ISO-ish date strings from content."""
    import re
    return re.findall(r"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2})?", content)
