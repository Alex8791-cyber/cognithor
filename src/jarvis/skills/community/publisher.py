"""PublisherIdentity: GitHub-basierte Identitaet fuer Community-Skill-Autoren.

Trust-Level-System:
  - UNTRUSTED: Score < 20
  - LOW: Score 20-40, GitHub-Account existiert, > 30 Tage alt
  - MODERATE: Score 40-60
  - HIGH: Score 60-80, > 10 public Repos, E-Mail verifiziert
  - VERIFIED: Score 80-100, 3+ Skills mit 4+ Rating, 0 Recalls, 90+ Tage

Reputation:
  - Positives Review: +2, Negatives: -3
  - Abuse Report: -10, Recall: -25
  - Erfolgreiche Installation: +0.5
  - Security Scan bestanden: +5
  - Auto-Block bei Score < 10
  - Auto-Flag bei 3+ Abuse Reports

Bible reference: §14 (Marketplace Governance)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any

from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# HTTP-Timeout (Sekunden)
_HTTP_TIMEOUT_S = 15


# ============================================================================
# Trust-Level
# ============================================================================


class TrustLevel(Enum):
    """Vertrauensstufen fuer Publisher."""

    UNTRUSTED = "untrusted"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERIFIED = "verified"

    @classmethod
    def from_score(cls, score: float) -> TrustLevel:
        """Bestimmt TrustLevel aus dem Reputation-Score."""
        if score >= 80:
            return cls.VERIFIED
        if score >= 60:
            return cls.HIGH
        if score >= 40:
            return cls.MODERATE
        if score >= 20:
            return cls.LOW
        return cls.UNTRUSTED


# ============================================================================
# Publisher-Identity
# ============================================================================


@dataclass
class PublisherIdentity:
    """GitHub-basierte Publisher-Identitaet."""

    github_username: str
    github_id: int = 0
    display_name: str = ""
    verified: bool = False
    reputation_score: float = 50.0
    trust_level: TrustLevel = TrustLevel.LOW
    skills_published: int = 0
    abuse_reports: int = 0
    recalls: int = 0

    # GitHub-Profildetails (aus API)
    account_age_days: int = 0
    public_repos: int = 0
    email_verified: bool = False
    two_factor_enabled: bool = False

    def update_trust_level(self) -> None:
        """Aktualisiert den TrustLevel basierend auf Score und Profil."""
        base_level = TrustLevel.from_score(self.reputation_score)

        # Downgrade wenn Bedingungen nicht erfuellt
        if base_level == TrustLevel.VERIFIED:
            if (
                self.skills_published < 3
                or self.recalls > 0
                or self.account_age_days < 90
            ):
                base_level = TrustLevel.HIGH

        if base_level == TrustLevel.HIGH:
            if self.public_repos < 10 or not self.email_verified:
                base_level = TrustLevel.MODERATE

        if base_level == TrustLevel.LOW:
            if self.account_age_days < 30:
                base_level = TrustLevel.UNTRUSTED

        self.trust_level = base_level

    @property
    def is_blocked(self) -> bool:
        """Ob der Publisher automatisch geblockt ist."""
        return self.reputation_score < 10

    @property
    def is_flagged(self) -> bool:
        """Ob der Publisher geflaggt ist (3+ Abuse Reports)."""
        return self.abuse_reports >= 3

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert in ein serialisierbares Dict."""
        return {
            "github_username": self.github_username,
            "github_id": self.github_id,
            "display_name": self.display_name,
            "verified": self.verified,
            "reputation_score": self.reputation_score,
            "trust_level": self.trust_level.value,
            "skills_published": self.skills_published,
            "abuse_reports": self.abuse_reports,
            "recalls": self.recalls,
            "is_blocked": self.is_blocked,
            "is_flagged": self.is_flagged,
        }


# ============================================================================
# Publisher-Verifier
# ============================================================================


class PublisherVerifier:
    """Verifiziert GitHub-basierte Publisher-Identitaeten.

    Nutzt die GitHub API (oder publisher.json aus dem Registry-Repo)
    um Publisher-Profile zu laden und Trust-Level zu berechnen.

    Usage::

        verifier = PublisherVerifier()
        identity = await verifier.verify("github-username")
        if identity.trust_level.value == "verified":
            print("Vertrauenswuerdiger Publisher!")
    """

    def __init__(
        self,
        *,
        registry_url: str = "",
        marketplace_store: Any | None = None,
    ) -> None:
        self._registry_url = registry_url or (
            "https://raw.githubusercontent.com/cognithor/skill-registry/main"
        )
        self._marketplace_store = marketplace_store
        self._cache: dict[str, PublisherIdentity] = {}
        self._lock = asyncio.Lock()

    async def verify(self, github_username: str) -> PublisherIdentity:
        """Verifiziert einen Publisher anhand seines GitHub-Usernamens.

        1. Publisher-Profil aus Registry-Repo laden
        2. Optionaler GitHub-API-Check
        3. Trust-Level berechnen

        Returns:
            PublisherIdentity mit aktuellem Trust-Level.
        """
        async with self._lock:
            # Cache pruefen (innerhalb des Locks um TOCTOU zu vermeiden)
            if github_username in self._cache:
                return self._cache[github_username]

            identity = PublisherIdentity(github_username=github_username)

            # 1. Publisher-Profil aus Registry-Repo laden
            try:
                profile = await self._fetch_publisher_profile(github_username)
                if profile:
                    identity.reputation_score = profile.get("reputation_score", 50.0)
                    identity.skills_published = profile.get("skills_published", 0)
                    identity.abuse_reports = profile.get("abuse_reports", 0)
                    identity.recalls = profile.get("recalls", 0)
                    identity.verified = profile.get("verified", False)
                    identity.github_id = profile.get("github_id", 0)
                    identity.display_name = profile.get("display_name", github_username)
                    identity.account_age_days = profile.get("account_age_days", 0)
                    identity.public_repos = profile.get("public_repos", 0)
                    identity.email_verified = profile.get("email_verified", False)
            except Exception as exc:
                log.debug("publisher_profile_fetch_failed", user=github_username, error=str(exc))

            # 2. Lokale Daten aus MarketplaceStore
            if self._marketplace_store is not None:
                local = self._marketplace_store.get_publisher(github_username)
                if local:
                    # Lokale Reputation hat Vorrang (aktueller)
                    identity.reputation_score = local.get(
                        "reputation_score", identity.reputation_score,
                    )
                    identity.abuse_reports = local.get(
                        "abuse_reports", identity.abuse_reports,
                    )
                    identity.recalls = local.get("recalls", identity.recalls)

            # 3. Trust-Level berechnen
            identity.update_trust_level()

            # Cache
            self._cache[github_username] = identity

            log.info(
                "publisher_verified",
                user=github_username,
                trust_level=identity.trust_level.value,
                score=identity.reputation_score,
            )

            return identity

    def invalidate_cache(self, github_username: str = "") -> None:
        """Invalidiert den Publisher-Cache."""
        if github_username:
            self._cache.pop(github_username, None)
        else:
            self._cache.clear()

    async def _fetch_publisher_profile(
        self, github_username: str,
    ) -> dict[str, Any] | None:
        """Laedt ein Publisher-Profil aus dem Registry-Repo."""
        import json

        url = f"{self._registry_url}/publishers/{github_username}.json"
        try:
            text = await self._fetch_text(url)
            return json.loads(text)
        except Exception as exc:
            log.debug("publisher_profile_not_found", user=github_username, error=str(exc))
            return None

    async def _fetch_text(self, url: str) -> str:
        """Laedt Text von einer URL.

        Nutzt aiohttp wenn verfuegbar und funktional, sonst urllib-Fallback.
        """
        aiohttp_available = False
        try:
            import aiohttp
            aiohttp_available = True
        except ImportError:
            pass

        if aiohttp_available:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=_HTTP_TIMEOUT_S)) as resp:
                        resp.raise_for_status()
                        return await resp.text()
            except Exception as aio_exc:
                log.debug("aiohttp_fetch_failed_falling_back_to_urllib", url=url, error=str(aio_exc))

        # Fallback: urllib (synchron im Executor)
        import urllib.request

        def _sync() -> str:
            req = urllib.request.Request(url, headers={"User-Agent": "Jarvis-Publisher/1.0"})
            with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:
                return resp.read().decode("utf-8")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync)
