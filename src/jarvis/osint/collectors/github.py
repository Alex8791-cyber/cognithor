"""GitHub collector — uses REST API v3."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from jarvis.osint.collectors.base import BaseCollector, CollectorError
from jarvis.osint.models import Evidence
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class GitHubCollector(BaseCollector):
    source_name = "github"
    BASE_URL = "https://api.github.com"

    def __init__(self, token: str | None = None) -> None:
        self._token = token or os.environ.get("GITHUB_TOKEN", "")

    def is_available(self) -> bool:
        return True  # Works without token (60 req/h)

    def _headers(self) -> dict[str, str]:
        h = {"Accept": "application/vnd.github.v3+json"}
        if self._token:
            h["Authorization"] = f"token {self._token}"
        return h

    async def collect(self, target: str, claims: list[str]) -> list[Evidence]:
        evidence: list[Evidence] = []
        now = datetime.now(timezone.utc)
        try:
            profile = await self._fetch_with_retry(
                f"{self.BASE_URL}/users/{target}", headers=self._headers()
            )
            evidence.append(
                Evidence(
                    source="github_profile",
                    source_type="github",
                    content=(
                        f"User: {profile.get('login')} | "
                        f"Name: {profile.get('name')} | "
                        f"Bio: {profile.get('bio', '')} | "
                        f"Company: {profile.get('company', '')} | "
                        f"Repos: {profile.get('public_repos', 0)} | "
                        f"Followers: {profile.get('followers', 0)} | "
                        f"Following: {profile.get('following', 0)} | "
                        f"Created: {profile.get('created_at', '')} | "
                        f"Location: {profile.get('location', '')}"
                    ),
                    confidence=0.9,
                    collected_at=now,
                    url=profile.get("html_url", ""),
                )
            )

            repos = await self._fetch_with_retry(
                f"{self.BASE_URL}/users/{target}/repos?sort=updated&per_page=30",
                headers=self._headers(),
            )
            for repo in repos[:15]:
                if repo.get("fork"):
                    continue
                evidence.append(
                    Evidence(
                        source=f"github_repo:{repo['name']}",
                        source_type="github",
                        content=(
                            f"Repo: {repo['name']} | "
                            f"Description: {repo.get('description', '')} | "
                            f"Stars: {repo.get('stargazers_count', 0)} | "
                            f"Language: {repo.get('language', '')} | "
                            f"Updated: {repo.get('updated_at', '')}"
                        ),
                        confidence=0.85,
                        collected_at=now,
                        url=repo.get("html_url", ""),
                    )
                )

            orgs = await self._fetch_with_retry(
                f"{self.BASE_URL}/users/{target}/orgs",
                headers=self._headers(),
            )
            for org in orgs:
                evidence.append(
                    Evidence(
                        source=f"github_org:{org.get('login', '')}",
                        source_type="github",
                        content=f"Organization membership: {org.get('login', '')}",
                        confidence=0.9,
                        collected_at=now,
                        url=f"https://github.com/{org.get('login', '')}",
                    )
                )

            # Claim-specific: check if target has repo matching claim keywords
            for claim in claims:
                claim_lower = claim.lower()
                for repo in repos:
                    repo_name = (repo.get("name", "") or "").lower()
                    repo_desc = (repo.get("description", "") or "").lower()
                    if any(
                        w in repo_name or w in repo_desc for w in claim_lower.split() if len(w) > 3
                    ):
                        evidence.append(
                            Evidence(
                                source=f"github_claim_match:{repo['name']}",
                                source_type="github",
                                content=f"Claim '{claim}' may relate to repo '{repo['name']}': {repo.get('description', '')}",
                                confidence=0.7,
                                collected_at=now,
                                url=repo.get("html_url", ""),
                            )
                        )

        except CollectorError:
            log.warning("github_collector_failed", target=target[:30])
        except Exception:
            log.debug("github_collector_error", exc_info=True)
        return evidence
