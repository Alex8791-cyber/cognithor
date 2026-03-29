"""GDPR Gatekeeper — compliance check before investigations."""

from __future__ import annotations

from jarvis.osint.models import GDPRScope, GDPRViolationError, HIMRequest
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

_ALL_COLLECTORS = ["github", "web", "arxiv", "scholar", "linkedin", "crunchbase", "social"]
_PRIVATE_PERSON = ["github", "scholar", "arxiv", "web"]
_PROJECT = ["github", "crunchbase", "web", "arxiv"]
_ORG = ["crunchbase", "web", "github"]


class GDPRGatekeeper:
    """Check GDPR compliance before running investigations."""

    def check(
        self,
        request: HIMRequest,
        github_followers: int = 0,
        has_papers: bool = False,
        has_public_talks: bool = False,
    ) -> GDPRScope:
        if len(request.requester_justification.strip()) < 10:
            raise GDPRViolationError("requester_justification must be at least 10 characters")

        if request.target_type == "project":
            return GDPRScope(
                is_public_figure=False,
                allowed_collectors=list(_PROJECT),
                restrictions=["data_minimisation"],
            )

        if request.target_type == "org":
            return GDPRScope(
                is_public_figure=False,
                allowed_collectors=list(_ORG),
                restrictions=["data_minimisation"],
            )

        # Person checks
        is_public = github_followers >= 50 or has_papers or has_public_talks

        if not is_public and request.depth == "deep":
            raise GDPRViolationError("depth='deep' not allowed for private persons")

        allowed = list(_ALL_COLLECTORS) if is_public else list(_PRIVATE_PERSON)
        restrictions = ["data_minimisation"]
        if not is_public:
            restrictions.append("no_social_media")
            restrictions.append("no_deep_linkedin")

        log.info(
            "gdpr_check_passed",
            target=request.target_name[:30],
            is_public=is_public,
            collectors=len(allowed),
        )

        return GDPRScope(
            is_public_figure=is_public,
            allowed_collectors=allowed,
            restrictions=restrictions,
        )
