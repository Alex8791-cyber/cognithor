"""Cognithor · Social / Reddit Lead Hunter routes.

Sub-Modul des `config_routes`-Pakets (siehe
`docs/superpowers/plans/2026-04-29-config-routes-split.md`). Enthaelt
`_register_social_routes()` — Endpoints fuer den LeadService und das
Reddit-Lead-Hunter-Pack (`/api/v1/leads/...`).
"""

from __future__ import annotations

from typing import Any

try:
    from starlette.requests import Request
except ImportError:
    Request = Any  # type: ignore[assignment,misc]

try:
    from fastapi import HTTPException
except ImportError:
    try:
        from starlette.exceptions import HTTPException  # type: ignore[assignment]
    except ImportError:
        HTTPException = Exception  # type: ignore[assignment,misc]

from cognithor.utils.logging import get_logger

log = get_logger(__name__)


# ======================================================================
# Social / Reddit Lead Hunter routes
# ======================================================================


def _register_social_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """REST endpoints for Reddit Lead Hunter."""

    def _get_service() -> Any:
        # Prefer the new source-agnostic LeadService; fall back to legacy alias.
        return (
            (
                getattr(gateway, "_leads_service", None)
                or getattr(gateway, "_reddit_lead_service", None)
            )
            if gateway
            else None
        )

    @app.get("/api/v1/leads/engine-status", dependencies=deps)
    async def leads_engine_status() -> dict[str, Any]:
        """Return which lead sources are enabled. Frontend uses this to gate the sidebar tab."""
        social_cfg = getattr(getattr(gateway, "_config", None), "social", None)
        if social_cfg is None:
            return {"enabled": False, "sources": {}}
        return {
            "enabled": bool(getattr(social_cfg, "leads_engine_enabled", False)),
            "sources": {
                "reddit": bool(getattr(social_cfg, "reddit_scan_enabled", False)),
                "hackernews": bool(getattr(social_cfg, "hn_enabled", False)),
                "discord": bool(getattr(social_cfg, "discord_scanner_enabled", False)),
                "rss": bool(getattr(social_cfg, "rss_enabled", False)),
            },
        }

    @app.get("/api/v1/packs/loaded", dependencies=deps)
    async def list_loaded_packs() -> dict[str, Any]:
        """Return currently loaded packs for Flutter tab gating."""
        loader = getattr(gateway, "_pack_loader", None)
        if loader is None:
            return {"packs": []}
        return {
            "packs": [
                {
                    "qualified_id": p.manifest.qualified_id,
                    "version": p.manifest.version,
                    "display_name": p.manifest.display_name,
                    "tools": p.manifest.tools,
                    "lead_sources": p.manifest.lead_sources,
                }
                for p in loader.loaded()
            ]
        }

    @app.get("/api/v1/leads/sources", dependencies=deps)
    async def list_lead_sources() -> dict[str, Any]:
        """Return registered LeadSource metadata.

        Feeds Flutter's LeadsScreen + locked-pack-card UX. An empty list
        means the backend has no lead sources and the sidebar tab should
        be hidden by the frontend.
        """
        svc = _get_service()
        if svc is None:
            return {"sources": []}
        # LeadService.list_sources() returns registered LeadSource instances.
        sources: list[dict[str, Any]] = []
        try:
            for source in svc.list_sources():
                sources.append(
                    {
                        "source_id": source.source_id,
                        "display_name": source.display_name,
                        "icon": getattr(source, "icon", ""),
                        "color": getattr(source, "color", ""),
                        "capabilities": sorted(getattr(source, "capabilities", [])),
                    }
                )
        except Exception:
            pass
        return {"sources": sources}

    @app.post("/api/v1/leads/scan/rss", dependencies=deps)
    async def scan_leads_rss(request: Request) -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead service not initialized", "status": 503}
        try:
            body = await request.json()
        except Exception:
            body = {}
        social_cfg = getattr(getattr(gateway, "_config", None), "social", None)
        feeds = body.get("feeds") or (
            list(getattr(social_cfg, "rss_feeds", [])) if social_cfg else []
        )
        min_score = int(
            body.get("min_score")
            or (getattr(social_cfg, "rss_min_score", 60) if social_cfg else 60)
        )
        if not feeds:
            return {"error": "No RSS feeds configured", "leads_found": 0, "posts_checked": 0}
        # Delegate to the rss-lead-hunter pack source (source_id="rss") if registered.
        try:
            result = await svc.scan(
                source_id="rss",
                min_score=min_score,
                config={"rss": {"feeds": feeds}},
                trigger="ui",
            )
            return {
                "id": result.id,
                "summary": result.summary(),
                "posts_checked": result.posts_checked,
                "leads_found": result.leads_found,
            }
        except ValueError:
            return {
                "error": "RSS source not registered — install rss-lead-hunter pack",
                "leads_found": 0,
                "posts_checked": 0,
            }

    @app.post("/api/v1/leads/scan", dependencies=deps)
    async def scan_leads(request: Request) -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead Service not initialized", "status": 503}
        try:
            body = await request.json()
        except Exception:
            body = {}
        min_score = int(body.get("min_score") or 0)
        source_id = body.get("source_id") or body.get("platform") or None
        social_cfg = getattr(getattr(gateway, "_config", None), "social", None)
        product = body.get("product") or getattr(social_cfg, "reddit_product_name", "") or ""
        result = await svc.scan(
            source_id=source_id,
            min_score=min_score,
            product=product,
            trigger="ui",
        )
        return {
            "id": result.id,
            "summary": result.summary(),
            "posts_checked": result.posts_checked,
            "leads_found": result.leads_found,
        }

    @app.get("/api/v1/leads", dependencies=deps)
    async def list_leads(
        status: str | None = None,
        min_score: int = 0,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead Service not initialized", "status": 503}
        from cognithor.leads.models import LeadStatus

        status_filter = None
        if status and status in [s.value for s in LeadStatus]:
            status_filter = LeadStatus(status)
        leads = svc.get_leads(status=status_filter, min_score=min_score, limit=limit, offset=offset)
        return {
            "leads": [l.to_dict() for l in leads],
            "count": len(leads),
        }

    @app.get("/api/v1/leads/stats", dependencies=deps)
    async def lead_stats() -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead Service not initialized", "status": 503}
        stats = svc.get_stats()
        history = svc.get_scan_history(limit=10)
        return {
            "stats": {
                "total": stats.total,
                "new": stats.new,
                "reviewed": stats.reviewed,
                "replied": stats.replied,
                "archived": stats.archived,
                "avg_score": stats.avg_score,
                "top_subreddits": stats.top_subreddits,
                "total_scans": stats.total_scans,
            },
            "recent_scans": history,
        }

    @app.post("/api/v1/leads/discover-subreddits", dependencies=deps)
    async def discover_subreddits(request: Request) -> dict[str, Any]:
        # Subreddit discovery is provided by the reddit-lead-hunter-pro pack.
        # Without the pack installed, this endpoint returns an empty suggestion list.
        return {
            "suggestions": [],
            "note": "Install reddit-lead-hunter-pro pack to enable subreddit discovery.",
        }

    @app.get("/api/v1/leads/templates", dependencies=deps)
    async def list_templates(subreddit: str = "") -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead Service not initialized", "status": 503}
        return {"templates": svc.get_templates(subreddit)}

    @app.post("/api/v1/leads/templates", dependencies=deps)
    async def create_template(request: Request) -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead Service not initialized", "status": 503}
        body = await request.json()
        tid = svc.create_template(
            name=body.get("name", ""),
            text=body.get("text", ""),
            subreddit=body.get("subreddit", ""),
            style=body.get("style", ""),
        )
        return {"id": tid, "status": "created"}

    @app.delete("/api/v1/leads/templates/{template_id}", dependencies=deps)
    async def delete_template(template_id: str) -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead Service not initialized", "status": 503}
        svc.delete_template(template_id)
        return {"status": "deleted"}

    @app.get("/api/v1/leads/{lead_id}", dependencies=deps)
    async def get_lead(lead_id: str) -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead Service not initialized", "status": 503}
        lead = svc.get_lead(lead_id)
        if lead is None:
            raise HTTPException(404, "Lead not found")
        return lead.to_dict()

    @app.patch("/api/v1/leads/{lead_id}", dependencies=deps)
    async def update_lead(lead_id: str, request: Request) -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead Service not initialized", "status": 503}
        body = await request.json()
        from cognithor.leads.models import LeadStatus

        status = LeadStatus(body["status"]) if "status" in body else None
        reply_final = body.get("reply_final")
        lead = svc.update_lead(lead_id, status=status, reply_final=reply_final)
        if lead is None:
            raise HTTPException(404, "Lead not found")
        return lead.to_dict()

    @app.post("/api/v1/leads/{lead_id}/reply", dependencies=deps)
    async def reply_to_lead(lead_id: str, request: Request) -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead Service not initialized", "status": 503}
        try:
            body = await request.json()
        except Exception:
            body = {}
        mode = body.get("mode", "clipboard")
        result = svc.post_reply(lead_id, mode=mode)
        return {
            "success": result.success,
            "mode": result.mode.value if hasattr(result.mode, "value") else result.mode,
            "error": result.error,
        }

    @app.post("/api/v1/leads/{lead_id}/refine", dependencies=deps)
    async def refine_lead(lead_id: str, request: Request) -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead Service not initialized", "status": 503}
        try:
            body = await request.json()
        except Exception:
            body = {}
        hint = body.get("hint", "")
        variants = body.get("variants", 0)
        result = await svc.refine_reply(lead_id, hint=hint, variants=variants)
        if result is None:
            raise HTTPException(404, "Lead not found")
        if isinstance(result, list):
            return {"variants": [{"text": r.text, "style": r.style} for r in result]}
        return {"text": result.text, "style": result.style, "changes": result.changes_summary}

    @app.get("/api/v1/leads/{lead_id}/performance", dependencies=deps)
    async def get_lead_performance(lead_id: str) -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead Service not initialized", "status": 503}
        perf = svc.get_performance(lead_id)
        if perf is None:
            return {"performance": None}

        def _engagement_score(upvotes: int, replies: int, author_replied: bool, tag: str) -> float:
            """Inline engagement score — simple weighted heuristic."""
            score = min(upvotes * 0.3 + replies * 0.5, 80.0)
            if author_replied:
                score += 15.0
            if tag == "good":
                score += 5.0
            elif tag == "bad":
                score -= 10.0
            return round(max(0.0, min(score, 100.0)), 1)

        perf["engagement_score"] = _engagement_score(
            perf.get("reply_upvotes", 0),
            perf.get("reply_replies", 0),
            bool(perf.get("author_replied", 0)),
            perf.get("feedback_tag", ""),
        )
        return {"performance": perf}

    @app.patch("/api/v1/leads/{lead_id}/feedback", dependencies=deps)
    async def set_lead_feedback(lead_id: str, request: Request) -> dict[str, Any]:
        svc = _get_service()
        if not svc:
            return {"error": "Lead Service not initialized", "status": 503}
        body = await request.json()
        svc.set_feedback(lead_id, tag=body.get("tag", ""), note=body.get("note", ""))
        return {"status": "ok"}
