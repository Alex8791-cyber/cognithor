"""KnowledgeValidator — extracts claims, cross-references, tracks confidence."""

from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Coroutine

from jarvis.security.encrypted_db import encrypted_connect
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["KnowledgeClaim", "KnowledgeValidator"]

# Domain trust tiers — authoritative sources get higher initial confidence
_TRUST_TIERS: dict[str, float] = {
    # Tier 1: Official government / law sources (0.8 base confidence)
    ".gov.de": 0.8, ".bund.de": 0.8, "gesetze-im-internet.de": 0.8,
    "bundesfinanzministerium.de": 0.8, "dejure.org": 0.8,
    "bundesgerichtshof.de": 0.8, "bafin.de": 0.8,
    "deutsche-rentenversicherung.de": 0.8, "bundesbank.de": 0.8,
    # Tier 2: Established institutions / publishers (0.65)
    ".edu": 0.65, "wikipedia.org": 0.65, "haufe.de": 0.65,
    "beck-online.de": 0.65, "juris.de": 0.65, "stiftung-warentest.de": 0.65,
    "verbraucherzentrale.de": 0.65, "finanztip.de": 0.65,
    # Tier 3: General reference (0.5 default)
}


def _get_source_trust(url: str) -> float:
    """Return trust score (0.0-1.0) based on the source domain."""
    if not url:
        return 0.5
    url_lower = url.lower()
    for domain_pattern, trust in _TRUST_TIERS.items():
        if domain_pattern in url_lower:
            return trust
    return 0.5  # Unknown source = neutral

_EXTRACT_CLAIMS_PROMPT = """\
Extrahiere die 5 wichtigsten faktischen Aussagen (Claims) aus folgendem Text.
Jeder Claim muss eine ueberpruefbare Tatsachenbehauptung sein.

Text:
\"\"\"
{text}
\"\"\"

Antworte NUR mit JSON:
{{
  "claims": [
    {{"claim": "Die Widerrufsfrist nach §7 VVG betraegt 14 Tage.", "category": "law|fact|statistic|definition|opinion", "importance": "high|medium|low"}}
  ]
}}
"""

_VERIFY_CLAIM_PROMPT = """\
Pruefe ob der folgende Claim durch die gegebenen Quellentexte bestaetigt oder widerlegt wird.

Claim: "{claim}"

Quellentext:
\"\"\"
{source_text}
\"\"\"

Antworte NUR mit JSON:
{{
  "verdict": "confirmed|contradicted|unrelated|partially_confirmed",
  "evidence": "Kurze Begruendung (1-2 Saetze)",
  "confidence": 0.0-1.0
}}
"""


@dataclass
class KnowledgeClaim:
    """A factual claim extracted from research, tracked for validation."""

    id: str = ""
    claim: str = ""
    category: str = ""          # law | fact | statistic | definition | opinion
    importance: str = "medium"   # high | medium | low
    goal_slug: str = ""
    first_source: str = ""       # URL where first found
    sources_confirmed: int = 0
    sources_contradicted: int = 0
    sources_checked: int = 0
    confidence: float = 0.5      # 0.0-1.0, starts neutral
    status: str = "unverified"   # unverified | verified | disputed | debunked
    evidence_for: str = ""       # Concatenated supporting evidence
    evidence_against: str = ""   # Concatenated contradicting evidence
    created_at: str = ""
    last_checked: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "claim": self.claim,
            "category": self.category,
            "importance": self.importance,
            "goal_slug": self.goal_slug,
            "first_source": self.first_source,
            "sources_confirmed": self.sources_confirmed,
            "sources_contradicted": self.sources_contradicted,
            "sources_checked": self.sources_checked,
            "confidence": round(self.confidence, 2),
            "status": self.status,
            "evidence_for": self.evidence_for[:500],
            "evidence_against": self.evidence_against[:500],
            "created_at": self.created_at,
            "last_checked": self.last_checked,
        }


class KnowledgeValidator:
    """Extracts claims from text, cross-references them, tracks confidence.

    Flow:
    1. extract_claims(text) -> list of KnowledgeClaim
    2. verify_claim(claim, source_text) -> updates confidence
    3. challenge_weak_claims() -> finds low-confidence claims, searches for counter-evidence
    4. get_claims_table() -> summary of all claims with confidence scores
    """

    def __init__(
        self,
        db_path: Path | str,
        llm_fn: Callable[[str], Coroutine[Any, Any, str]] | None = None,
        mcp_client: Any = None,
    ) -> None:
        self._db_path = str(db_path)
        self._llm_fn = llm_fn
        self._mcp = mcp_client
        self._conn = encrypted_connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_claims (
                id TEXT PRIMARY KEY,
                claim TEXT NOT NULL,
                category TEXT DEFAULT '',
                importance TEXT DEFAULT 'medium',
                goal_slug TEXT DEFAULT '',
                first_source TEXT DEFAULT '',
                sources_confirmed INTEGER DEFAULT 0,
                sources_contradicted INTEGER DEFAULT 0,
                sources_checked INTEGER DEFAULT 0,
                confidence REAL DEFAULT 0.5,
                status TEXT DEFAULT 'unverified',
                evidence_for TEXT DEFAULT '',
                evidence_against TEXT DEFAULT '',
                created_at TEXT DEFAULT '',
                last_checked TEXT DEFAULT ''
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_claims_goal ON knowledge_claims(goal_slug)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_claims_confidence ON knowledge_claims(confidence)
        """)
        self._conn.commit()

    def _now(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _new_id(self) -> str:
        return uuid.uuid4().hex[:16]

    # -- Extract claims from text ----------------------------------------

    async def extract_claims(
        self, text: str, source_url: str = "", goal_slug: str = ""
    ) -> list[KnowledgeClaim]:
        """Extract factual claims from text via LLM."""
        if not self._llm_fn or not text.strip():
            return []
        try:
            prompt = _EXTRACT_CLAIMS_PROMPT.format(text=text[:3000])
            raw = await self._llm_fn(prompt)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return []
            data = json.loads(match.group())
            claims = []
            for c in data.get("claims", []):
                claim_text = c.get("claim", "").strip()
                if not claim_text or len(claim_text) < 10:
                    continue
                # Check for duplicate claims (fuzzy: same first 50 chars)
                existing = self._find_similar_claim(claim_text, goal_slug)
                if existing:
                    # Boost confidence of existing claim (confirmed by another source)
                    existing.sources_confirmed += 1
                    existing.sources_checked += 1
                    existing.confidence = min(1.0, existing.confidence + 0.1)
                    existing.last_checked = self._now()
                    if existing.sources_confirmed >= 2:
                        existing.status = "verified"
                    self._save_claim(existing)
                    claims.append(existing)
                else:
                    claim = KnowledgeClaim(
                        id=self._new_id(),
                        claim=claim_text,
                        category=c.get("category", "fact"),
                        importance=c.get("importance", "medium"),
                        goal_slug=goal_slug,
                        first_source=source_url,
                        sources_checked=1,
                        confidence=_get_source_trust(source_url),
                        status="unverified",
                        created_at=self._now(),
                        last_checked=self._now(),
                    )
                    self._save_claim(claim)
                    claims.append(claim)
            trust = _get_source_trust(source_url)
            log.info(
                "knowledge_claims_extracted",
                count=len(claims),
                source=source_url[:50],
                trust=trust,
            )
            return claims
        except Exception:
            log.debug("knowledge_claims_extraction_failed", exc_info=True)
            return []

    # -- Verify a claim against new text ---------------------------------

    async def verify_claim(
        self, claim: KnowledgeClaim, source_text: str
    ) -> KnowledgeClaim:
        """Check a claim against a new source text via LLM."""
        if not self._llm_fn:
            return claim
        try:
            prompt = _VERIFY_CLAIM_PROMPT.format(
                claim=claim.claim, source_text=source_text[:2000]
            )
            raw = await self._llm_fn(prompt)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return claim
            data = json.loads(match.group())
            verdict = data.get("verdict", "unrelated")
            evidence = data.get("evidence", "")

            claim.sources_checked += 1
            claim.last_checked = self._now()

            if verdict == "confirmed":
                claim.sources_confirmed += 1
                claim.confidence = min(1.0, claim.confidence + 0.15)
                claim.evidence_for += (
                    f"\n{evidence}" if claim.evidence_for else evidence
                )
                if claim.sources_confirmed >= 2:
                    claim.status = "verified"
            elif verdict == "contradicted":
                claim.sources_contradicted += 1
                claim.confidence = max(0.0, claim.confidence - 0.25)
                claim.evidence_against += (
                    f"\n{evidence}" if claim.evidence_against else evidence
                )
                if claim.sources_contradicted >= 2:
                    claim.status = "disputed"
                if claim.confidence < 0.2:
                    claim.status = "debunked"
            elif verdict == "partially_confirmed":
                claim.sources_confirmed += 1
                claim.confidence = min(1.0, claim.confidence + 0.05)

            self._save_claim(claim)
            return claim
        except Exception:
            log.debug("knowledge_verify_failed", exc_info=True)
            return claim

    # -- Challenge weak claims -------------------------------------------

    async def challenge_weak_claims(
        self, goal_slug: str, max_challenges: int = 3
    ) -> list[KnowledgeClaim]:
        """Find low-confidence claims and search for counter-evidence.

        Skips already-debunked claims — no point re-challenging them.
        """
        weak = self.get_claims(
            goal_slug=goal_slug, max_confidence=0.6, limit=max_challenges + 10
        )
        # Filter out debunked claims — they've already been proven false
        weak = [c for c in weak if c.status != "debunked"][:max_challenges]
        if not weak or not self._mcp:
            return weak

        challenged = []
        for claim in weak:
            try:
                result = await self._mcp.call_tool(
                    "search_and_read",
                    {
                        "query": claim.claim[:100],
                        "num_results": 2,
                        "language": "de",
                    },
                )
                if result and not result.is_error and result.content:
                    claim = await self.verify_claim(claim, result.content[:3000])
                    challenged.append(claim)
                    log.info(
                        "knowledge_claim_challenged",
                        claim=claim.claim[:50],
                        new_confidence=claim.confidence,
                        status=claim.status,
                    )
            except Exception:
                log.debug("knowledge_challenge_failed", exc_info=True)
        return challenged

    # -- Query claims table ----------------------------------------------

    def get_claims(
        self,
        goal_slug: str = "",
        status: str = "",
        max_confidence: float = 1.0,
        min_confidence: float = 0.0,
        limit: int = 50,
    ) -> list[KnowledgeClaim]:
        """Query the claims table with filters."""
        query = "SELECT * FROM knowledge_claims WHERE 1=1"
        params: list[Any] = []
        if goal_slug:
            query += " AND goal_slug = ?"
            params.append(goal_slug)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " AND confidence >= ? AND confidence <= ?"
        params.extend([min_confidence, max_confidence])
        query += " ORDER BY confidence ASC, importance DESC LIMIT ?"
        params.append(limit)

        cursor = self._conn.execute(query, params)
        columns = [d[0] for d in cursor.description]
        rows = cursor.fetchall()

        claims = []
        for row in rows:
            d = dict(zip(columns, row))
            claims.append(
                KnowledgeClaim(
                    **{
                        k: v
                        for k, v in d.items()
                        if k in KnowledgeClaim.__dataclass_fields__
                    }
                )
            )
        return claims

    def get_claims_summary(self, goal_slug: str = "") -> dict[str, Any]:
        """Get a summary of the claims table."""
        where = "WHERE goal_slug = ?" if goal_slug else ""
        params = [goal_slug] if goal_slug else []

        row = self._conn.execute(
            f"SELECT COUNT(*), AVG(confidence), "
            f"SUM(CASE WHEN status='verified' THEN 1 ELSE 0 END), "
            f"SUM(CASE WHEN status='disputed' THEN 1 ELSE 0 END), "
            f"SUM(CASE WHEN status='debunked' THEN 1 ELSE 0 END), "
            f"SUM(CASE WHEN status='unverified' THEN 1 ELSE 0 END) "
            f"FROM knowledge_claims {where}",
            params,
        ).fetchone()

        return {
            "total_claims": row[0] or 0,
            "avg_confidence": round(row[1] or 0.0, 2),
            "verified": row[2] or 0,
            "disputed": row[3] or 0,
            "debunked": row[4] or 0,
            "unverified": row[5] or 0,
        }

    # -- Internal helpers ------------------------------------------------

    def _find_similar_claim(
        self, claim_text: str, goal_slug: str
    ) -> KnowledgeClaim | None:
        """Find an existing claim that's similar (first 50 chars match)."""
        prefix = claim_text[:50]
        row = self._conn.execute(
            "SELECT * FROM knowledge_claims WHERE goal_slug = ? AND claim LIKE ? LIMIT 1",
            (goal_slug, f"{prefix}%"),
        ).fetchone()
        if not row:
            return None
        cursor = self._conn.execute("SELECT * FROM knowledge_claims LIMIT 0")
        columns = [d[0] for d in cursor.description]
        d = dict(zip(columns, row))
        return KnowledgeClaim(
            **{
                k: v
                for k, v in d.items()
                if k in KnowledgeClaim.__dataclass_fields__
            }
        )

    def _save_claim(self, claim: KnowledgeClaim) -> None:
        """Insert or update a claim in the database."""
        self._conn.execute(
            """INSERT OR REPLACE INTO knowledge_claims
               (id, claim, category, importance, goal_slug, first_source,
                sources_confirmed, sources_contradicted, sources_checked,
                confidence, status, evidence_for, evidence_against,
                created_at, last_checked)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                claim.id,
                claim.claim,
                claim.category,
                claim.importance,
                claim.goal_slug,
                claim.first_source,
                claim.sources_confirmed,
                claim.sources_contradicted,
                claim.sources_checked,
                claim.confidence,
                claim.status,
                claim.evidence_for[:2000],
                claim.evidence_against[:2000],
                claim.created_at,
                claim.last_checked,
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
