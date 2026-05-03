"""REST API for the Skill Marketplace.

Endpoints for browse, search, install, publish, and reviews.
Integrates into the FastAPI Control Center on port 8741
as an APIRouter under ``/api/v1/skills``.

Architecture reference: SS14 (Skills & Ecosystem)
"""

import sqlite3
from pathlib import Path
from typing import Any, cast

from cognithor.utils.logging import get_logger

log = get_logger(__name__)

# Lazily initialized store -- created on first request.
_store_holder: dict[str, Any] = {"store": None}


def _get_store() -> Any:
    """Return the global MarketplaceStore (lazy init)."""
    if _store_holder["store"] is None:
        from cognithor.skills.persistence import MarketplaceStore

        # Read path from config, fall back to default
        try:
            from cognithor.config import load_config

            cfg = load_config()
            db_path = getattr(cfg, "marketplace", None)
            if db_path and hasattr(db_path, "db_path") and db_path.db_path:
                store_path = Path(db_path.db_path)
            else:
                store_path = cfg.cognithor_home / "marketplace.db"
        except Exception:
            store_path = Path.home() / ".cognithor" / "marketplace.db"

        _store_holder["store"] = MarketplaceStore(store_path)
    return _store_holder["store"]


def set_store(store: Any) -> None:
    """Set the store manually (for tests)."""
    _store_holder["store"] = store


# ------------------------------------------------------------------
# Request/Response Models (module-level for FastAPI compatibility)
# ------------------------------------------------------------------

try:
    from pydantic import BaseModel as _BaseModel

    class ReviewRequest(_BaseModel):
        """Payload for a review submission."""

        rating: int
        comment: str = ""
        reviewer_id: str = "anonymous"

    class InstallRequest(_BaseModel):
        """Payload for an installation."""

        user_id: str = "default"
        version: str = ""

except ImportError:
    ReviewRequest = None  # type: ignore[assignment,misc]
    InstallRequest = None  # type: ignore[assignment,misc]


def _build_router() -> Any:
    """Create the FastAPI APIRouter with all marketplace endpoints."""
    try:
        from fastapi import APIRouter, HTTPException, Query
    except ImportError:
        # FastAPI not installed -- empty placeholder
        log.warning("fastapi_not_available_for_skills_api")
        return None

    router = APIRouter(prefix="/api/v1/skills", tags=["skills"])

    # ------------------------------------------------------------------
    # Search & Browse
    # ------------------------------------------------------------------

    @router.get("/search")
    async def search_skills(
        query: str = "",
        category: str = "",
        sort: str = "relevance",
        min_rating: float = 0.0,
        limit: int = Query(default=20, ge=1, le=100),
    ) -> dict[str, Any]:
        """Search the skill marketplace.

        Query parameters:
          - query: Full-text search
          - category: Category filter
          - sort: relevance | newest | rating | installs | popularity
          - min_rating: Minimum rating (0.0 - 5.0)
          - limit: Max results (1-100)
        """
        store = _get_store()
        results = store.search_listings(
            query=query,
            category=category,
            min_rating=min_rating,
            sort=sort,
            limit=limit,
        )
        return {"results": results, "count": len(results)}

    @router.get("/featured")
    async def get_featured(
        limit: int = Query(default=10, ge=1, le=50),
    ) -> dict[str, Any]:
        """Curated featured skills."""
        store = _get_store()
        return {"featured": store.get_featured(limit=limit)}

    @router.get("/trending")
    async def get_trending(
        days: int = Query(default=7, ge=1, le=90),
        limit: int = Query(default=10, ge=1, le=50),
    ) -> dict[str, Any]:
        """Trending skills of the last N days."""
        store = _get_store()
        return {"trending": store.get_trending(days=days, limit=limit)}

    @router.get("/categories")
    async def get_categories() -> dict[str, Any]:
        """All available categories with metadata."""
        from cognithor.skills.marketplace import CATEGORY_INFOS

        categories = []
        for cat, info in CATEGORY_INFOS.items():
            categories.append(
                {
                    "value": cat.value,
                    "display_name": info.display_name,
                    "icon": info.icon,
                    "description": info.description,
                }
            )
        return {"categories": categories}

    @router.get("/installed")
    async def list_installed(
        user_id: str = "default",
        limit: int = Query(default=50, ge=1, le=200),
    ) -> dict[str, Any]:
        """List of installed skills for a user."""
        store = _get_store()
        history = store.get_install_history(user_id=user_id, limit=limit)
        return {"installed": history, "count": len(history)}

    @router.get("/stats")
    async def get_marketplace_stats() -> dict[str, Any]:
        """Aggregated marketplace statistics."""
        store = _get_store()
        return cast("dict[str, Any]", store.get_stats())

    # ------------------------------------------------------------------
    # Skill Detail
    # ------------------------------------------------------------------

    @router.get("/{package_id}")
    async def get_skill_detail(package_id: str) -> dict[str, Any]:
        """Detail view of a single skill."""
        store = _get_store()
        listing = store.get_listing(package_id)
        if listing is None:
            raise HTTPException(status_code=404, detail="Skill not found")
        return cast("dict[str, Any]", listing)

    # ------------------------------------------------------------------
    # Install / Uninstall
    # ------------------------------------------------------------------

    @router.post("/{package_id}/install")
    async def install_skill(
        package_id: str,
        body: InstallRequest | None = None,
    ) -> dict[str, Any]:
        """Install a skill (records installation)."""
        store = _get_store()
        listing = store.get_listing(package_id)
        if listing is None:
            raise HTTPException(status_code=404, detail="Skill not found")

        user_id = body.user_id if body else "default"
        version = body.version if body else listing.get("version", "")

        store.increment_install_count(package_id)
        store.record_install(
            package_id=package_id,
            version=version,
            user_id=user_id,
        )
        log.info(
            "skill_installed",
            package_id=package_id,
            user_id=user_id,
            version=version,
        )
        return {"status": "installed", "package_id": package_id}

    @router.delete("/{package_id}")
    async def uninstall_skill(package_id: str) -> dict[str, Any]:
        """Uninstall a skill (marks as recalled)."""
        store = _get_store()
        listing = store.get_listing(package_id)
        if listing is None:
            raise HTTPException(status_code=404, detail="Skill not found")
        # Recall mechanism for uninstall
        store.recall_listing(package_id, reason="user_uninstall")
        return {"status": "uninstalled", "package_id": package_id}

    # ------------------------------------------------------------------
    # Reviews
    # ------------------------------------------------------------------

    @router.get("/{package_id}/reviews")
    async def get_reviews(
        package_id: str,
        limit: int = Query(default=20, ge=1, le=100),
    ) -> dict[str, Any]:
        """Reviews for a skill."""
        store = _get_store()
        reviews = store.get_reviews(package_id, limit=limit)
        avg = store.get_average_rating(package_id)
        return {"reviews": reviews, "count": len(reviews), "average_rating": avg}

    @router.post("/{package_id}/reviews")
    async def submit_review(
        package_id: str,
        body: ReviewRequest,
    ) -> dict[str, Any]:
        """Submit a review for a skill."""
        store = _get_store()
        listing = store.get_listing(package_id)
        if listing is None:
            raise HTTPException(status_code=404, detail="Skill not found")

        try:
            review_id = store.save_review(
                package_id=package_id,
                reviewer_id=body.reviewer_id,
                rating=body.rating,
                comment=body.comment,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except sqlite3.IntegrityError as exc:
            raise HTTPException(
                status_code=409,
                detail="You have already reviewed this skill",
            ) from exc

        return {
            "status": "created",
            "review_id": review_id,
            "package_id": package_id,
        }

    return router


def _build_community_router() -> Any:
    """Create the FastAPI APIRouter for community marketplace endpoints."""
    try:
        from fastapi import APIRouter, HTTPException, Query
    except ImportError:
        log.warning("fastapi_not_available_for_community_api")
        return None

    from pydantic import BaseModel as _BaseModel

    class CommunityInstallRequest(_BaseModel):
        """Payload for community skill installation."""

        user_id: str = "default"

    class ReportRequest(_BaseModel):
        """Payload for abuse report."""

        reporter: str = "anonymous"
        category: str = "other"
        description: str = ""
        evidence: str = ""

    class CommunityReviewRequest(_BaseModel):
        """Payload for community skill review."""

        rating: int
        comment: str = ""
        reviewer_id: str = "anonymous"

    cr = APIRouter(prefix="/api/v1/skills/community", tags=["community-skills"])

    # Lazily initialized CommunityRegistryClient
    _client_holder: dict[str, Any] = {"client": None}

    def _get_client() -> Any:
        if _client_holder["client"] is None:
            from cognithor.skills.community.client import CommunityRegistryClient

            try:
                from cognithor.config import load_config

                cfg = load_config()
                cm = getattr(cfg, "community_marketplace", None)
                registry_url = cm.registry_url if cm else ""
                community_dir = cfg.cognithor_home / "skills" / "community"
            except Exception:
                registry_url = ""
                community_dir = Path.home() / ".cognithor" / "skills" / "community"

            _client_holder["client"] = CommunityRegistryClient(
                community_dir=community_dir,
                registry_url=registry_url,
            )
        return _client_holder["client"]

    # ------------------------------------------------------------------
    # Search (static route — must come BEFORE /{name})
    # ------------------------------------------------------------------

    @cr.get("/search")
    async def search_community(
        query: str = "",
        category: str = "",
        limit: int = Query(default=20, ge=1, le=100),
    ) -> dict[str, Any]:
        """Search the community skill registry."""
        try:
            client = _get_client()
            results = await client.search(query=query, category=category, limit=limit)
        except Exception as exc:
            log.error("community_search_failed", error=str(exc))
            raise HTTPException(status_code=502, detail="Registry search failed") from exc
        return {
            "results": [
                {
                    "name": r.name,
                    "version": r.version,
                    "description": r.description,
                    "author_github": r.author_github,
                    "category": r.category,
                    "tools_required": r.tools_required,
                }
                for r in results
            ],
            "count": len(results),
        }

    # ------------------------------------------------------------------
    # Recalls (static route — must come BEFORE /{name})
    # ------------------------------------------------------------------

    @cr.get("/recalls")
    async def get_community_recalls() -> dict[str, Any]:
        """Active remote recalls."""
        try:
            store = _get_store()
            recalls = store.get_remote_recalls()
        except Exception as exc:
            log.error("community_recalls_fetch_failed", error=str(exc))
            raise HTTPException(status_code=500, detail="Could not load recalls") from exc
        return {"recalls": recalls, "count": len(recalls)}

    # ------------------------------------------------------------------
    # Publishers (static route — must come BEFORE /{name})
    # ------------------------------------------------------------------

    @cr.get("/publishers/{github}")
    async def get_publisher_profile(github: str) -> dict[str, Any]:
        """Publisher profile by GitHub username."""
        store = _get_store()
        publisher = store.get_publisher(github)
        if publisher is None:
            raise HTTPException(status_code=404, detail="Publisher not found")
        return cast("dict[str, Any]", publisher)

    # ------------------------------------------------------------------
    # Sync (static route — must come BEFORE /{name})
    # ------------------------------------------------------------------

    @cr.post("/sync")
    async def sync_registry() -> dict[str, Any]:
        """Manually synchronize registry."""
        try:
            from cognithor.skills.community.sync import RegistrySync

            sync = RegistrySync(marketplace_store=_get_store())
            result = await sync.sync_once()
        except Exception as exc:
            log.error("community_sync_failed", error=str(exc))
            raise HTTPException(status_code=502, detail="Registry sync failed") from exc
        return {
            "success": result.success,
            "registry_skills": result.registry_skills,
            "new_recalls": result.new_recalls,
            "deactivated_skills": result.deactivated_skills,
            "errors": result.errors,
            "sync_time_ms": round(result.sync_time * 1000),
        }

    # ------------------------------------------------------------------
    # Detail (catch-all /{name} — AFTER all static routes)
    # ------------------------------------------------------------------

    @cr.get("/{name}")
    async def get_community_skill(name: str) -> dict[str, Any]:
        """Detail view of a community skill."""
        try:
            client = _get_client()
            entry = await client.get_entry(name)
        except Exception as exc:
            log.error("community_skill_detail_failed", skill=name, error=str(exc))
            raise HTTPException(status_code=502, detail="Registry unreachable") from exc
        if entry is None:
            raise HTTPException(status_code=404, detail="Community skill not found")
        return {
            "name": entry.name,
            "version": entry.version,
            "description": entry.description,
            "author_github": entry.author_github,
            "category": entry.category,
            "tools_required": entry.tools_required,
            "content_hash": entry.content_hash,
            "recalled": entry.recalled,
        }

    # ------------------------------------------------------------------
    # Install / Uninstall
    # ------------------------------------------------------------------

    @cr.post("/{name}/install")
    async def install_community_skill(
        name: str,
        body: CommunityInstallRequest | None = None,
    ) -> dict[str, Any]:
        """Install a community skill."""
        try:
            client = _get_client()
            result = await client.install(name)
        except Exception as exc:
            log.error("community_install_failed", skill=name, error=str(exc))
            raise HTTPException(status_code=502, detail="Installation failed") from exc

        if not result.success:
            import json as _json

            raise HTTPException(
                status_code=400,
                detail=_json.dumps(
                    {
                        "errors": result.errors,
                        "warnings": result.warnings,
                    },
                    ensure_ascii=False,
                ),
            )

        # Track in MarketplaceStore
        try:
            store = _get_store()
            user_id = body.user_id if body else "default"
            store.record_install(package_id=name, version=result.version, user_id=user_id)
        except Exception as exc:
            log.warning("community_install_tracking_failed", skill=name, error=str(exc))

        return {
            "status": "installed",
            "skill_name": result.skill_name,
            "version": result.version,
            "install_path": result.install_path,
            "tools_required": result.tools_required,
            "warnings": result.warnings,
        }

    @cr.delete("/{name}")
    async def uninstall_community_skill(name: str) -> dict[str, Any]:
        """Uninstall a community skill."""
        client = _get_client()
        removed = await client.uninstall(name)
        if not removed:
            raise HTTPException(status_code=404, detail="Skill not installed")
        return {"status": "uninstalled", "skill_name": name}

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    @cr.post("/{name}/report")
    async def report_community_skill(name: str, body: ReportRequest) -> dict[str, Any]:
        """Report a community skill as abusive."""
        store = _get_store()
        # Track abuse in local store
        comment_parts = [f"[ABUSE-REPORT:{body.category}] {body.description}"]
        if body.evidence:
            comment_parts.append(f"[EVIDENCE] {body.evidence}")
        report_id = store.save_review(
            package_id=name,
            reviewer_id=f"report:{body.reporter}",
            rating=1,
            comment=" ".join(comment_parts),
        )
        log.warning(
            "community_skill_reported",
            skill=name,
            reporter=body.reporter,
            category=body.category,
            has_evidence=bool(body.evidence),
        )
        return {"status": "reported", "report_id": report_id}

    # ------------------------------------------------------------------
    # Reviews
    # ------------------------------------------------------------------

    @cr.post("/{name}/review")
    async def review_community_skill(
        name: str,
        body: CommunityReviewRequest,
    ) -> dict[str, Any]:
        """Submit a review for a community skill."""
        store = _get_store()
        try:
            review_id = store.save_review(
                package_id=name,
                reviewer_id=body.reviewer_id,
                rating=body.rating,
                comment=body.comment,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except sqlite3.IntegrityError as exc:
            raise HTTPException(status_code=409, detail="Already reviewed") from exc

        return {"status": "created", "review_id": review_id}

    return cr


# Create routers on import (None if FastAPI is missing)
router = _build_router()
community_router = _build_community_router()
