"""REST API fuer den Skill Marketplace.

Endpoints fuer Browse, Search, Install, Publish und Reviews.
Integriert sich in den FastAPI Control Center auf Port 8741
als APIRouter unter ``/api/v1/skills``.

Architektur-Bibel: SS14 (Skills & Ecosystem)
"""

import sqlite3
from pathlib import Path
from typing import Any, Optional

from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# Lazy-initialisierter Store -- wird beim ersten Request erzeugt.
_store_holder: dict[str, Any] = {"store": None}


def _get_store() -> Any:
    """Gibt den globalen MarketplaceStore zurueck (Lazy-Init)."""
    if _store_holder["store"] is None:
        from jarvis.skills.persistence import MarketplaceStore

        # Pfad aus Config lesen, Fallback auf Standard
        try:
            from jarvis.config import load_config
            cfg = load_config()
            db_path = getattr(cfg, "marketplace", None)
            if db_path and hasattr(db_path, "db_path") and db_path.db_path:
                store_path = Path(db_path.db_path)
            else:
                store_path = cfg.jarvis_home / "marketplace.db"
        except Exception:
            store_path = Path.home() / ".jarvis" / "marketplace.db"

        _store_holder["store"] = MarketplaceStore(store_path)
    return _store_holder["store"]


def set_store(store: Any) -> None:
    """Setzt den Store manuell (fuer Tests)."""
    _store_holder["store"] = store


# ------------------------------------------------------------------
# Request/Response Models (module-level fuer FastAPI-Kompatibilitaet)
# ------------------------------------------------------------------

try:
    from pydantic import BaseModel as _BaseModel

    class ReviewRequest(_BaseModel):
        """Payload fuer eine Review-Einreichung."""
        rating: int
        comment: str = ""
        reviewer_id: str = "anonymous"

    class InstallRequest(_BaseModel):
        """Payload fuer eine Installation."""
        user_id: str = "default"
        version: str = ""

except ImportError:
    ReviewRequest = None  # type: ignore[assignment,misc]
    InstallRequest = None  # type: ignore[assignment,misc]


def _build_router() -> Any:
    """Erstellt den FastAPI APIRouter mit allen Marketplace-Endpoints."""
    try:
        from fastapi import APIRouter, HTTPException, Query
    except ImportError:
        # FastAPI nicht installiert -- leerer Platzhalter
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
    ) -> dict:
        """Durchsucht den Skill-Marketplace.

        Query-Parameter:
          - query: Volltextsuche
          - category: Kategorie-Filter
          - sort: relevance | newest | rating | installs | popularity
          - min_rating: Mindestbewertung (0.0 - 5.0)
          - limit: Max Ergebnisse (1-100)
        """
        store = _get_store()
        results = store.search_listings(
            query=query, category=category,
            min_rating=min_rating, sort=sort, limit=limit,
        )
        return {"results": results, "count": len(results)}

    @router.get("/featured")
    async def get_featured(
        limit: int = Query(default=10, ge=1, le=50),
    ) -> dict:
        """Kuratierte Featured-Skills."""
        store = _get_store()
        return {"featured": store.get_featured(limit=limit)}

    @router.get("/trending")
    async def get_trending(
        days: int = Query(default=7, ge=1, le=90),
        limit: int = Query(default=10, ge=1, le=50),
    ) -> dict:
        """Trending-Skills der letzten N Tage."""
        store = _get_store()
        return {"trending": store.get_trending(days=days, limit=limit)}

    @router.get("/categories")
    async def get_categories() -> dict:
        """Alle verfuegbaren Kategorien mit Metadaten."""
        from jarvis.skills.marketplace import CATEGORY_INFOS
        categories = []
        for cat, info in CATEGORY_INFOS.items():
            categories.append({
                "value": cat.value,
                "display_name": info.display_name,
                "icon": info.icon,
                "description": info.description,
            })
        return {"categories": categories}

    @router.get("/installed")
    async def list_installed(
        user_id: str = "default",
        limit: int = Query(default=50, ge=1, le=200),
    ) -> dict:
        """Liste der installierten Skills eines Users."""
        store = _get_store()
        history = store.get_install_history(user_id=user_id, limit=limit)
        return {"installed": history, "count": len(history)}

    @router.get("/stats")
    async def get_marketplace_stats() -> dict:
        """Aggregierte Marketplace-Statistiken."""
        store = _get_store()
        return store.get_stats()

    # ------------------------------------------------------------------
    # Skill Detail
    # ------------------------------------------------------------------

    @router.get("/{package_id}")
    async def get_skill_detail(package_id: str) -> dict:
        """Detail-Ansicht eines einzelnen Skills."""
        store = _get_store()
        listing = store.get_listing(package_id)
        if listing is None:
            raise HTTPException(status_code=404, detail="Skill nicht gefunden")
        return listing

    # ------------------------------------------------------------------
    # Install / Uninstall
    # ------------------------------------------------------------------

    @router.post("/{package_id}/install")
    async def install_skill(
        package_id: str, body: Optional[InstallRequest] = None,
    ) -> dict:
        """Installiert einen Skill (zeichnet Installation auf)."""
        store = _get_store()
        listing = store.get_listing(package_id)
        if listing is None:
            raise HTTPException(status_code=404, detail="Skill nicht gefunden")

        user_id = body.user_id if body else "default"
        version = body.version if body else listing.get("version", "")

        store.increment_install_count(package_id)
        store.record_install(
            package_id=package_id, version=version, user_id=user_id,
        )
        log.info(
            "skill_installed",
            package_id=package_id, user_id=user_id, version=version,
        )
        return {"status": "installed", "package_id": package_id}

    @router.delete("/{package_id}")
    async def uninstall_skill(package_id: str) -> dict:
        """Deinstalliert einen Skill (markiert als recalled)."""
        store = _get_store()
        listing = store.get_listing(package_id)
        if listing is None:
            raise HTTPException(status_code=404, detail="Skill nicht gefunden")
        # Recall-Mechanismus fuer Uninstall
        store.recall_listing(package_id, reason="user_uninstall")
        return {"status": "uninstalled", "package_id": package_id}

    # ------------------------------------------------------------------
    # Reviews
    # ------------------------------------------------------------------

    @router.get("/{package_id}/reviews")
    async def get_reviews(
        package_id: str,
        limit: int = Query(default=20, ge=1, le=100),
    ) -> dict:
        """Reviews fuer einen Skill."""
        store = _get_store()
        reviews = store.get_reviews(package_id, limit=limit)
        avg = store.get_average_rating(package_id)
        return {"reviews": reviews, "count": len(reviews), "average_rating": avg}

    @router.post("/{package_id}/reviews")
    async def submit_review(
        package_id: str, body: ReviewRequest,
    ) -> dict:
        """Review fuer einen Skill einreichen."""
        store = _get_store()
        listing = store.get_listing(package_id)
        if listing is None:
            raise HTTPException(status_code=404, detail="Skill nicht gefunden")

        try:
            review_id = store.save_review(
                package_id=package_id,
                reviewer_id=body.reviewer_id,
                rating=body.rating,
                comment=body.comment,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except sqlite3.IntegrityError:
            raise HTTPException(
                status_code=409,
                detail="Du hast diesen Skill bereits bewertet",
            )

        return {
            "status": "created",
            "review_id": review_id,
            "package_id": package_id,
        }

    return router


def _build_community_router() -> Any:
    """Erstellt den FastAPI APIRouter fuer Community-Marketplace-Endpoints."""
    try:
        from fastapi import APIRouter, HTTPException, Query
    except ImportError:
        log.warning("fastapi_not_available_for_community_api")
        return None

    from pydantic import BaseModel as _BaseModel

    class CommunityInstallRequest(_BaseModel):
        """Payload fuer Community-Skill-Installation."""
        user_id: str = "default"

    class ReportRequest(_BaseModel):
        """Payload fuer Abuse-Report."""
        reporter: str = "anonymous"
        category: str = "other"
        description: str = ""
        evidence: str = ""

    class CommunityReviewRequest(_BaseModel):
        """Payload fuer Community-Skill-Review."""
        rating: int
        comment: str = ""
        reviewer_id: str = "anonymous"

    cr = APIRouter(prefix="/api/v1/skills/community", tags=["community-skills"])

    # Lazy-initialisierter CommunityRegistryClient
    _client_holder: dict[str, Any] = {"client": None}

    def _get_client() -> Any:
        if _client_holder["client"] is None:
            from jarvis.skills.community.client import CommunityRegistryClient
            try:
                from jarvis.config import load_config
                cfg = load_config()
                cm = getattr(cfg, "community_marketplace", None)
                registry_url = cm.registry_url if cm else ""
                community_dir = cfg.jarvis_home / "skills" / "community"
            except Exception:
                registry_url = ""
                community_dir = Path.home() / ".jarvis" / "skills" / "community"

            _client_holder["client"] = CommunityRegistryClient(
                community_dir=community_dir,
                registry_url=registry_url,
            )
        return _client_holder["client"]

    # ------------------------------------------------------------------
    # Search (statische Route — muss VOR /{name} stehen)
    # ------------------------------------------------------------------

    @cr.get("/search")
    async def search_community(
        query: str = "",
        category: str = "",
        limit: int = Query(default=20, ge=1, le=100),
    ) -> dict:
        """Durchsucht das Community-Skill-Registry."""
        try:
            client = _get_client()
            results = await client.search(query=query, category=category, limit=limit)
        except Exception as exc:
            log.error("community_search_failed", error=str(exc))
            raise HTTPException(status_code=502, detail="Registry-Suche fehlgeschlagen") from exc
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
    # Recalls (statische Route — muss VOR /{name} stehen)
    # ------------------------------------------------------------------

    @cr.get("/recalls")
    async def get_community_recalls() -> dict:
        """Aktive Remote-Recalls."""
        try:
            store = _get_store()
            recalls = store.get_remote_recalls()
        except Exception as exc:
            log.error("community_recalls_fetch_failed", error=str(exc))
            raise HTTPException(status_code=500, detail="Recalls konnten nicht geladen werden") from exc
        return {"recalls": recalls, "count": len(recalls)}

    # ------------------------------------------------------------------
    # Publishers (statische Route — muss VOR /{name} stehen)
    # ------------------------------------------------------------------

    @cr.get("/publishers/{github}")
    async def get_publisher_profile(github: str) -> dict:
        """Publisher-Profil anhand des GitHub-Usernamens."""
        store = _get_store()
        publisher = store.get_publisher(github)
        if publisher is None:
            raise HTTPException(status_code=404, detail="Publisher nicht gefunden")
        return publisher

    # ------------------------------------------------------------------
    # Sync (statische Route — muss VOR /{name} stehen)
    # ------------------------------------------------------------------

    @cr.post("/sync")
    async def sync_registry() -> dict:
        """Registry manuell synchronisieren."""
        try:
            from jarvis.skills.community.sync import RegistrySync
            sync = RegistrySync(marketplace_store=_get_store())
            result = await sync.sync_once()
        except Exception as exc:
            log.error("community_sync_failed", error=str(exc))
            raise HTTPException(status_code=502, detail="Registry-Sync fehlgeschlagen") from exc
        return {
            "success": result.success,
            "registry_skills": result.registry_skills,
            "new_recalls": result.new_recalls,
            "deactivated_skills": result.deactivated_skills,
            "errors": result.errors,
            "sync_time_ms": round(result.sync_time * 1000),
        }

    # ------------------------------------------------------------------
    # Detail (catch-all /{name} — NACH allen statischen Routen)
    # ------------------------------------------------------------------

    @cr.get("/{name}")
    async def get_community_skill(name: str) -> dict:
        """Detail-Ansicht eines Community-Skills."""
        try:
            client = _get_client()
            entry = await client.get_entry(name)
        except Exception as exc:
            log.error("community_skill_detail_failed", skill=name, error=str(exc))
            raise HTTPException(status_code=502, detail="Registry nicht erreichbar") from exc
        if entry is None:
            raise HTTPException(status_code=404, detail="Community-Skill nicht gefunden")
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
        name: str, body: CommunityInstallRequest | None = None,
    ) -> dict:
        """Installiert einen Community-Skill."""
        try:
            client = _get_client()
            result = await client.install(name)
        except Exception as exc:
            log.error("community_install_failed", skill=name, error=str(exc))
            raise HTTPException(status_code=502, detail="Installation fehlgeschlagen") from exc

        if not result.success:
            import json as _json
            raise HTTPException(status_code=400, detail=_json.dumps({
                "errors": result.errors,
                "warnings": result.warnings,
            }, ensure_ascii=False))

        # In MarketplaceStore tracken
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
    async def uninstall_community_skill(name: str) -> dict:
        """Deinstalliert einen Community-Skill."""
        client = _get_client()
        removed = await client.uninstall(name)
        if not removed:
            raise HTTPException(status_code=404, detail="Skill nicht installiert")
        return {"status": "uninstalled", "skill_name": name}

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    @cr.post("/{name}/report")
    async def report_community_skill(name: str, body: ReportRequest) -> dict:
        """Meldet einen Community-Skill als missbraeuchlich."""
        store = _get_store()
        # Abuse im lokalen Store tracken
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
        name: str, body: CommunityReviewRequest,
    ) -> dict:
        """Review fuer einen Community-Skill abgeben."""
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
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=409, detail="Bereits bewertet")

        return {"status": "created", "review_id": review_id}

    return cr


# Erstelle die Router beim Import (None falls FastAPI fehlt)
router = _build_router()
community_router = _build_community_router()
