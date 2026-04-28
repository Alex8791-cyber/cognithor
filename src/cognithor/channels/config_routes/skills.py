"""Cognithor · Skill management + Skill-Registry + Hermes routes.

Sub-Modul des `config_routes`-Pakets (siehe
`docs/superpowers/plans/2026-04-29-config-routes-split.md`). Bundle
aus drei thematisch verwandten `_register_*`-Helfern:

  - `_register_skill_routes()` — Skill-Marketplace, Procedural-Skills,
    P2P-Skills, Curated-Collections, Reflexion / Skill-Generator.
  - `_register_skill_registry_routes()` — Skill-Registry-Endpoints
    (`/api/v1/skill-registry/...`).
  - `_register_hermes_routes()` — agentskills.io / Hermes
    Kompatibilitaets-Layer (`/api/v1/skills/hermes/...`).
"""

from __future__ import annotations

import contextlib
from pathlib import Path
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
# Skill management routes
# ======================================================================


def _register_skill_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """Marketplace, updater, commands, skill-CLI, connectors, workflows,
    models, i18n, setup-wizard."""

    # -- Marketplace ------------------------------------------------------

    @app.get("/api/v1/marketplace/feed", dependencies=deps)
    async def marketplace_feed() -> dict[str, Any]:
        """Kuratierter Feed fuer die Startseite."""
        try:
            from cognithor.skills.marketplace import SkillMarketplace

            return SkillMarketplace().curated_feed()
        except Exception as exc:
            log.error("marketplace_feed_failed", error=str(exc))
            return {"error": "Marketplace-Feed nicht verfuegbar"}

    @app.get("/api/v1/marketplace/search", dependencies=deps)
    async def marketplace_search(
        q: str = "",
        category: str = "",
        verified_only: bool = False,
        sort_by: str = "relevance",
        max_results: int = 20,
    ) -> dict[str, Any]:
        """Durchsucht den Skill-Marktplatz."""
        try:
            from cognithor.skills.marketplace import SkillMarketplace

            mp = SkillMarketplace()
            results = mp.search(
                query=q,
                category=category,
                verified_only=verified_only,
                sort_by=sort_by,
                max_results=max_results,
            )
            return {"results": [r.to_dict() for r in results], "count": len(results)}
        except Exception as exc:
            log.error("marketplace_search_failed", error=str(exc))
            return {"error": "Marketplace-Suche fehlgeschlagen"}

    @app.get("/api/v1/marketplace/categories", dependencies=deps)
    async def marketplace_categories() -> dict[str, Any]:
        """Alle Skill-Kategorien mit Counts."""
        try:
            from cognithor.skills.marketplace import SkillMarketplace

            return {"categories": [c.to_dict() for c in SkillMarketplace().categories()]}
        except Exception as exc:
            log.error("marketplace_categories_failed", error=str(exc))
            return {"error": "Kategorien nicht verfuegbar"}

    @app.get("/api/v1/marketplace/featured", dependencies=deps)
    async def marketplace_featured(n: int = 10) -> dict[str, Any]:
        """Featured-Skills."""
        try:
            from cognithor.skills.marketplace import SkillMarketplace

            return {"featured": [s.to_dict() for s in SkillMarketplace().featured(n)]}
        except Exception as exc:
            log.error("marketplace_featured_failed", error=str(exc))
            return {"error": "Featured-Skills nicht verfuegbar"}

    @app.get("/api/v1/marketplace/trending", dependencies=deps)
    async def marketplace_trending(window: str = "24h", n: int = 10) -> dict[str, Any]:
        """Trending-Skills."""
        try:
            from cognithor.skills.marketplace import SkillMarketplace

            return {"trending": [s.to_dict() for s in SkillMarketplace().trending(max_results=n)]}
        except Exception as exc:
            log.error("marketplace_trending_failed", error=str(exc))
            return {"error": "Trending-Skills nicht verfuegbar"}

    @app.get("/api/v1/marketplace/stats", dependencies=deps)
    async def marketplace_stats() -> dict[str, Any]:
        """Marktplatz-Statistiken."""
        try:
            from cognithor.skills.marketplace import SkillMarketplace

            return SkillMarketplace().stats()
        except Exception as exc:
            log.error("marketplace_stats_failed", error=str(exc))
            return {"error": "Marketplace-Statistiken nicht verfuegbar"}

    # -- Skill-Updater ----------------------------------------------------

    @app.get("/api/v1/updater/stats", dependencies=deps)
    async def updater_stats() -> dict[str, Any]:
        """Skill-Updater-Statistiken."""
        try:
            from cognithor.skills.updater import SkillUpdater

            return SkillUpdater().stats()
        except Exception as exc:
            log.error("updater_stats_failed", error=str(exc))
            return {"error": "Updater-Statistiken nicht verfuegbar"}

    @app.get("/api/v1/updater/pending", dependencies=deps)
    async def updater_pending() -> dict[str, Any]:
        """Ausstehende Updates."""
        try:
            from cognithor.skills.updater import SkillUpdater

            u = SkillUpdater()
            return {"updates": [c.to_dict() for c in u.pending_updates()]}
        except Exception as exc:
            log.error("updater_pending_failed", error=str(exc))
            return {"error": "Ausstehende Updates nicht verfuegbar"}

    @app.get("/api/v1/updater/recalls", dependencies=deps)
    async def updater_recalls() -> dict[str, Any]:
        """Aktive Security-Recalls."""
        try:
            from cognithor.skills.updater import SkillUpdater

            u = SkillUpdater()
            return {"recalls": [r.to_dict() for r in u.active_recalls()]}
        except Exception as exc:
            log.error("updater_recalls_failed", error=str(exc))
            return {"error": "Recalls nicht verfuegbar"}

    @app.get("/api/v1/updater/history", dependencies=deps)
    async def updater_history(n: int = 20) -> dict[str, Any]:
        """Update-Historie."""
        try:
            from cognithor.skills.updater import SkillUpdater

            return {"history": SkillUpdater().update_history(n)}
        except Exception as exc:
            log.error("updater_history_failed", error=str(exc))
            return {"error": "Update-Historie nicht verfuegbar"}

    # -- Commands ---------------------------------------------------------

    @app.get("/api/v1/commands/list", dependencies=deps)
    async def list_commands() -> dict[str, Any]:
        """Alle registrierten Slash-Commands."""
        try:
            from cognithor.channels.commands import CommandRegistry

            reg = CommandRegistry()
            return {
                "commands": [c.to_dict() for c in reg.list_commands()],
                "count": reg.command_count,
            }
        except Exception as exc:
            log.error("commands_list_failed", error=str(exc))
            return {"error": "Commands konnten nicht geladen werden"}

    @app.get("/api/v1/commands/slack", dependencies=deps)
    async def commands_slack() -> dict[str, Any]:
        """Slack Slash-Command-Definitionen."""
        try:
            from cognithor.channels.commands import CommandRegistry

            return {"definitions": CommandRegistry().slack_definitions()}
        except Exception as exc:
            log.error("commands_slack_failed", error=str(exc))
            return {"error": "Slack-Commands nicht verfuegbar"}

    @app.get("/api/v1/commands/discord", dependencies=deps)
    async def commands_discord() -> dict[str, Any]:
        """Discord Application-Command-Definitionen."""
        try:
            from cognithor.channels.commands import CommandRegistry

            return {"definitions": CommandRegistry().discord_definitions()}
        except Exception as exc:
            log.error("commands_discord_failed", error=str(exc))
            return {"error": "Discord-Commands nicht verfuegbar"}

    # -- Connectors -------------------------------------------------------

    @app.get("/api/v1/connectors/list", dependencies=deps)
    async def list_connectors() -> dict[str, Any]:
        """Alle registrierten Konnektoren."""
        reg = getattr(gateway, "_connector_registry", None)
        if reg is None:
            return {"connectors": [], "count": 0}
        return {"connectors": reg.list_connectors(), "count": reg.connector_count}

    @app.get("/api/v1/connectors/stats", dependencies=deps)
    async def connector_stats() -> dict[str, Any]:
        """Konnektor-Statistiken."""
        reg = getattr(gateway, "_connector_registry", None)
        if reg is None:
            return {
                "total_connectors": 0,
                "connectors": [],
                "scope_guard": {"policies": 0, "violations": 0},
            }
        return reg.stats()

    # -- Workflows (categories + legacy start — main endpoints in _register_workflow_graph_routes)

    @app.get("/api/v1/workflows/templates/categories", dependencies=deps)
    async def workflow_categories() -> dict[str, Any]:
        """Workflow-Kategorien."""
        lib = getattr(gateway, "_template_library", None)
        if lib is None:
            return {"categories": []}
        return {"categories": lib.categories()}

    @app.post("/api/v1/workflows/start", dependencies=deps)
    async def workflow_start(request: Request) -> dict[str, Any]:
        """Workflow-Instanz starten (legacy endpoint)."""
        try:
            engine = getattr(gateway, "_workflow_engine", None)
            lib = getattr(gateway, "_template_library", None)
            if engine is None or lib is None:
                return {"error": "Workflow-Engine nicht verfügbar"}
            body = await request.json()
            template_id = body.get("template_id", "")
            template = lib.get(template_id)
            if not template:
                return {"error": f"Template nicht gefunden: {template_id}"}
            inst = engine.start(template, created_by=body.get("created_by", ""))
            return inst.to_dict()
        except Exception as exc:
            log.error("workflow_start_failed", error=str(exc))
            return {"error": "Workflow konnte nicht gestartet werden"}

    # -- Models -----------------------------------------------------------

    @app.get("/api/v1/models/list", dependencies=deps)
    async def model_list() -> dict[str, Any]:
        """Alle registrierten ML-Modelle."""
        reg = getattr(gateway, "_model_registry", None)
        if reg is None:
            return {"models": [], "count": 0}
        return {"models": reg.list_all(), "count": reg.model_count}

    @app.get("/api/v1/models/available", dependencies=deps)
    async def available_models() -> dict[str, Any]:
        """Listet alle in Ollama/Backend verfuegbaren Modelle auf."""
        router = getattr(gateway, "_model_router", None)
        if router is None:
            return {"models": [], "source": "none"}
        # Refresh the model list
        with contextlib.suppress(Exception):
            await router.initialize()
        models = sorted(router._available_models) if router._available_models else []
        # Also return currently configured models for reference
        cfg = getattr(gateway, "_config", None)
        if cfg is None:
            return {
                "models": models,
                "configured": {},
                "warnings": [],
                "source": "backend" if router._backend else "ollama",
            }
        configured = {
            "planner": cfg.models.planner.name,
            "executor": cfg.models.executor.name,
            "coder": cfg.models.coder.name,
            "embedding": cfg.models.embedding.name,
        }
        warnings = []
        for role, name in configured.items():
            if models and name not in models:
                warnings.append(
                    f"Modell '{name}' ({role}) ist nicht verfügbar. "
                    f"Installieren: ollama pull {name}"
                )
        return {
            "models": models,
            "configured": configured,
            "warnings": warnings,
            "source": "backend" if router._backend else "ollama",
        }

    @app.get("/api/v1/models/stats", dependencies=deps)
    async def model_stats() -> dict[str, Any]:
        """Model-Registry Statistiken."""
        reg = getattr(gateway, "_model_registry", None)
        if reg is None:
            return {"total_models": 0, "providers": [], "capabilities": [], "languages": []}
        return reg.stats()

    # -- i18n -------------------------------------------------------------

    @app.get("/api/v1/i18n/locales", dependencies=deps)
    async def i18n_locales() -> dict[str, Any]:
        """Verfuegbare Sprachen."""
        from cognithor.i18n import get_available_locales, get_locale

        return {"locales": get_available_locales(), "default": get_locale()}

    @app.get("/api/v1/i18n/translate/{key}", dependencies=deps)
    async def i18n_translate(key: str, locale: str = "") -> dict[str, Any]:
        """Einzelnen Key uebersetzen."""
        from cognithor.i18n import get_locale, set_locale, t

        if locale and locale != get_locale():
            prev = get_locale()
            set_locale(locale)
            result = t(key)
            set_locale(prev)
            return {"key": key, "translation": result}
        return {"key": key, "translation": t(key)}

    @app.get("/api/v1/i18n/stats", dependencies=deps)
    async def i18n_stats() -> dict[str, Any]:
        """i18n-Statistiken."""
        from cognithor.i18n import get_available_locales, get_locale

        locales = get_available_locales()
        return {"default_locale": get_locale(), "locale_count": len(locales), "locales": locales}

    # -- Skill-CLI (Phase 35) ---------------------------------------------

    @app.get("/api/v1/skill-cli/stats", dependencies=deps)
    async def skill_cli_stats() -> dict[str, Any]:
        """Skill-CLI Statistiken."""
        cli = getattr(gateway, "_skill_cli", None)
        if cli is None:
            return {"scaffolder": {"templates": 0}}
        return cli.stats()

    @app.get("/api/v1/skill-cli/templates", dependencies=deps)
    async def skill_cli_templates() -> dict[str, Any]:
        """Verfuegbare Skill-Templates."""
        cli = getattr(gateway, "_skill_cli", None)
        if cli is None:
            return {"templates": []}
        return {"templates": cli.scaffolder.available_templates()}

    @app.get("/api/v1/skill-cli/rewards", dependencies=deps)
    async def skill_cli_rewards() -> dict[str, Any]:
        """Reward-System Statistiken."""
        cli = getattr(gateway, "_skill_cli", None)
        if cli is None:
            return {"contributors": 0}
        return cli.rewards.stats()

    # -- Setup-Wizard (Phase 36) ------------------------------------------

    @app.get("/api/v1/setup/state", dependencies=deps)
    async def setup_state() -> dict[str, Any]:
        """Setup-Wizard Status."""
        wiz = getattr(gateway, "_setup_wizard", None)
        if wiz is None:
            return {"step": "unavailable"}
        return wiz.state.to_dict()

    @app.get("/api/v1/setup/stats", dependencies=deps)
    async def setup_stats() -> dict[str, Any]:
        """Setup-Wizard Statistiken."""
        wiz = getattr(gateway, "_setup_wizard", None)
        if wiz is None:
            return {"state": {}}
        return wiz.stats()


# ======================================================================
# Hermes / agentskills.io compatibility routes
# ======================================================================


def _register_skill_registry_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """Expose the built-in skill registry (not marketplace) via REST."""

    def _get_registry() -> Any:
        return getattr(gateway, "_skill_registry", None) if gateway else None

    @app.get("/api/v1/skill-registry/list", dependencies=deps)
    async def list_registry_skills() -> dict[str, Any]:
        reg = _get_registry()
        if not reg:
            return {"installed": [], "count": 0}
        skills = reg.list_all()
        return {
            "installed": [
                {
                    "name": s.name,
                    "slug": s.slug,
                    "description": s.description,
                    "category": s.category,
                    "enabled": s.enabled,
                    "source": getattr(s, "source", "builtin"),
                    "version": getattr(s, "version", "1.0.0"),
                    "author": getattr(s, "author", ""),
                    "total_uses": s.total_uses,
                    "success_rate": round(s.success_rate, 2) if s.total_uses > 0 else None,
                }
                for s in skills
            ],
            "count": len(skills),
        }

    @app.get("/api/v1/skill-registry/{slug}", dependencies=deps)
    async def get_skill_detail(slug: str) -> dict[str, Any]:
        """Get full skill detail including body."""
        reg = _get_registry()
        if not reg:
            raise HTTPException(404, "Skill registry not available")
        skill = reg.get(slug)
        if not skill:
            raise HTTPException(404, f"Skill '{slug}' not found")
        return {
            "name": skill.name,
            "slug": skill.slug,
            "description": skill.description,
            "category": skill.category,
            "trigger_keywords": skill.trigger_keywords,
            "tools_required": skill.tools_required,
            "priority": skill.priority,
            "enabled": skill.enabled,
            "model_preference": skill.model_preference,
            "agent": skill.agent,
            "body": skill.body,
            "source": getattr(skill, "source", "builtin"),
            "file_path": str(skill.file_path),
            "total_uses": skill.total_uses,
            "success_count": skill.success_count,
            "failure_count": skill.failure_count,
            "success_rate": round(skill.success_rate, 2) if skill.total_uses > 0 else None,
            "avg_score": skill.avg_score,
            "last_used": skill.last_used,
        }

    @app.post("/api/v1/skill-registry/create", dependencies=deps)
    async def create_skill(request: Request) -> dict[str, Any]:
        """Create a new skill from JSON body."""
        import re
        from pathlib import Path

        import yaml

        body = await request.json()
        name = body.get("name", "").strip()
        if not name:
            raise HTTPException(400, "Name is required")

        reg = _get_registry()
        if not reg:
            raise HTTPException(503, "Skill registry not available")

        # Generate slug from name
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        if not slug:
            raise HTTPException(400, "Invalid skill name")

        # Check for duplicate
        if reg.get(slug):
            raise HTTPException(409, f"Skill '{slug}' already exists")

        # Build skill file content
        frontmatter = {
            "name": name,
            "description": body.get("description", ""),
            "category": body.get("category", "general"),
            "trigger_keywords": body.get("trigger_keywords", []),
            "tools_required": body.get("tools_required", []),
            "priority": body.get("priority", 0),
            "enabled": body.get("enabled", True),
        }
        if body.get("model_preference"):
            frontmatter["model_preference"] = body["model_preference"]
        if body.get("agent"):
            frontmatter["agent"] = body["agent"]

        skill_body = body.get("body", "")
        front_yaml = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)
        content = f"---\n{front_yaml}---\n\n{skill_body}\n"

        # Determine save directory
        config = getattr(gateway, "_config", None)
        cognithor_home = Path(getattr(config, "cognithor_home", Path.home() / ".cognithor"))
        skills_dir = cognithor_home / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        file_path = skills_dir / f"{slug}.md"

        file_path.write_text(content, encoding="utf-8")

        # Reload registry
        reg.load_from_directories([skills_dir, Path("data/procedures")])

        return {"status": "created", "slug": slug, "file_path": str(file_path)}

    @app.put("/api/v1/skill-registry/{slug}", dependencies=deps)
    async def update_skill(slug: str, request: Request) -> dict[str, Any]:
        """Update an existing skill (metadata and/or body)."""
        import yaml

        reg = _get_registry()
        if not reg:
            raise HTTPException(503, "Skill registry not available")

        skill = reg.get(slug)
        if not skill:
            raise HTTPException(404, f"Skill '{slug}' not found")

        body = await request.json()

        # Read current file
        file_path = skill.file_path
        if not file_path.exists():
            raise HTTPException(500, "Skill file not found on disk")

        # Build updated frontmatter
        frontmatter = {
            "name": body.get("name", skill.name),
            "description": body.get("description", skill.description),
            "category": body.get("category", skill.category),
            "trigger_keywords": body.get("trigger_keywords", skill.trigger_keywords),
            "tools_required": body.get("tools_required", skill.tools_required),
            "priority": body.get("priority", skill.priority),
            "enabled": body.get("enabled", skill.enabled),
        }
        if skill.model_preference or body.get("model_preference"):
            frontmatter["model_preference"] = body.get("model_preference", skill.model_preference)
        if skill.agent or body.get("agent"):
            frontmatter["agent"] = body.get("agent", skill.agent)
        # Preserve stats
        frontmatter["success_count"] = skill.success_count
        frontmatter["failure_count"] = skill.failure_count
        frontmatter["total_uses"] = skill.total_uses
        frontmatter["avg_score"] = skill.avg_score
        if skill.last_used:
            frontmatter["last_used"] = skill.last_used

        skill_body = body.get("body", skill.body)
        front_yaml = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)
        content = f"---\n{front_yaml}---\n\n{skill_body}\n"

        file_path.write_text(content, encoding="utf-8")

        # Reload registry from the skill's parent directory
        config = getattr(gateway, "_config", None)
        cognithor_home = Path(getattr(config, "cognithor_home", Path.home() / ".cognithor"))
        reg.load_from_directories([cognithor_home / "skills", Path("data/procedures")])

        return {"status": "updated", "slug": slug}

    @app.delete("/api/v1/skill-registry/{slug}", dependencies=deps)
    async def delete_skill(slug: str) -> dict[str, Any]:
        """Delete a skill file from disk and reload registry."""
        reg = _get_registry()
        if not reg:
            raise HTTPException(503, "Skill registry not available")

        skill = reg.get(slug)
        if not skill:
            raise HTTPException(404, f"Skill '{slug}' not found")

        # Don't allow deleting built-in procedure skills
        file_path = skill.file_path
        if "data/procedures" in str(file_path):
            raise HTTPException(403, "Cannot delete built-in procedure skills")

        if file_path.exists():
            file_path.unlink()

        # Reload registry
        config = getattr(gateway, "_config", None)
        cognithor_home = Path(getattr(config, "cognithor_home", Path.home() / ".cognithor"))
        reg.load_from_directories([cognithor_home / "skills", Path("data/procedures")])

        return {"status": "deleted", "slug": slug}

    @app.put("/api/v1/skill-registry/{slug}/toggle", dependencies=deps)
    async def toggle_skill(slug: str) -> dict[str, Any]:
        """Enable or disable a skill."""
        reg = _get_registry()
        if not reg:
            raise HTTPException(503, "Skill registry not available")

        skill = reg.get(slug)
        if not skill:
            raise HTTPException(404, f"Skill '{slug}' not found")

        if skill.enabled:
            reg.disable(slug)
            return {"slug": slug, "enabled": False}
        else:
            reg.enable(slug)
            return {"slug": slug, "enabled": True}

    @app.get("/api/v1/skill-registry/{slug}/export", dependencies=deps)
    async def export_skill_md(slug: str) -> dict[str, Any]:
        """Export a skill in SKILL.md (agentskills.io) format."""
        reg = _get_registry()
        if not reg:
            raise HTTPException(503, "Skill registry not available")

        skill = reg.get(slug)
        if not skill:
            raise HTTPException(404, f"Skill '{slug}' not found")

        try:
            from cognithor.skills.hermes_compat import HermesCompatLayer

            skill_dict = {
                "name": skill.name,
                "description": skill.description,
                "tags": skill.trigger_keywords,
                "prompt": skill.body,
                "version": "1.0.0",
            }
            hermes = HermesCompatLayer.cognithor_to_hermes(skill_dict)
            content = HermesCompatLayer.to_skill_md(hermes)
            return {"slug": slug, "skill_md": content}
        except Exception as e:
            raise HTTPException(500, f"Export failed: {e}") from e


def _register_hermes_routes(
    app: Any,
    deps: list[Any],
    gateway: Any,
) -> None:
    """REST endpoints for Hermes SKILL.md import/export."""

    def _get_compat() -> Any:
        return getattr(gateway, "_hermes_compat", None) if gateway else None

    def _get_registry() -> Any:
        return getattr(gateway, "_skill_registry", None) if gateway else None

    @app.get("/api/v1/skills/hermes/export/{skill_name}", dependencies=deps)
    async def hermes_export_skill(skill_name: str) -> dict[str, Any]:
        """Export a Cognithor skill to SKILL.md format."""
        compat = _get_compat()
        if not compat:
            return {"error": "Hermes compatibility layer not initialized", "status": 503}

        registry = _get_registry()
        if not registry:
            return {"error": "Skill registry not initialized", "status": 503}

        skill = registry.get(skill_name)
        if not skill:
            return {"error": f"Skill '{skill_name}' not found", "status": 404}

        # Convert Cognithor Skill dataclass to dict for conversion
        skill_dict = {
            "name": skill.name,
            "description": skill.description,
            "tags": [],
            "prompt": skill.body,
            "version": "1.0.0",
            "author": "cognithor",
        }
        if skill.manifest:
            skill_dict["version"] = skill.manifest.version
            skill_dict["author"] = skill.manifest.author_github or "cognithor"

        hermes_skill = compat.cognithor_to_hermes(skill_dict)
        content = compat.to_skill_md(hermes_skill)
        return {"skill_name": skill_name, "skill_md": content}

    @app.post("/api/v1/skills/hermes/import", dependencies=deps)
    async def hermes_import_skill(request: Request) -> dict[str, Any]:
        """Import a SKILL.md into Cognithor."""
        compat = _get_compat()
        if not compat:
            return {"error": "Hermes compatibility layer not initialized", "status": 503}

        body = await request.json()
        content = body.get("content", "")
        if not content:
            return {"error": "Missing 'content' field with SKILL.md text", "status": 400}

        try:
            hermes_skill = compat.parse_skill_md(content)
        except ValueError as exc:
            return {"error": f"Invalid SKILL.md format: {exc}", "status": 400}

        cognithor_dict = compat.hermes_to_cognithor(hermes_skill)
        return {
            "imported": True,
            "skill": cognithor_dict,
        }
