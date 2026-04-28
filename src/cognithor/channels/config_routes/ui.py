"""Cognithor · UI-specific routes (Control Center frontend).

Sub-Modul des `config_routes`-Pakets (siehe
`docs/superpowers/plans/2026-04-29-config-routes-split.md`). Enthaelt
`_register_ui_routes()` — Frontend-Endpoints fuer das Flutter Control
Center: Control-Center-Status, Models-Discovery, Templates, Connectors,
Wizards-UI, Theme-Settings und weitere UI-Glue. Groesster einzelner
UI-Block (~970 LOC). Inkl. lokale `_load_yaml`/`_save_yaml` Helfer.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import yaml

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

if TYPE_CHECKING:
    from pathlib import Path

    from cognithor.config_manager import ConfigManager

log = get_logger(__name__)


# ======================================================================
# UI-specific routes (Control Center frontend)
# ======================================================================


def _register_ui_routes(
    app: Any,
    deps: list[Any],
    config_manager: ConfigManager,
    gateway: Any,
) -> None:
    """Endpoints consumed by the Cognithor Control Center React UI.

    Covers system lifecycle, agent/binding persistence, prompts,
    cron-jobs, MCP servers, and A2A configuration.
    """

    cognithor_home = config_manager.config.cognithor_home
    agents_path = cognithor_home / "agents.yaml"
    bindings_path = cognithor_home / "bindings.yaml"

    def _load_yaml(path: Path) -> Any:
        if not path.exists():
            return {}
        try:
            return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}

    def _save_yaml(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.dump(data, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )

    # -- 3.1: System Status -----------------------------------------------

    @app.get("/api/v1/system/status", dependencies=deps)
    async def ui_system_status() -> dict[str, Any]:
        """Returns running status for the UI status indicator."""
        return {
            "status": "running",
            "timestamp": time.time(),
            "config_version": config_manager.config.version,
            "owner": config_manager.config.owner_name,
        }

    # -- 3.2: System Start / Stop ----------------------------------------

    @app.post("/api/v1/system/start", dependencies=deps)
    async def ui_system_start() -> dict[str, Any]:
        """Reloads configuration (logical start)."""
        try:
            config_manager.reload()
            return {"status": "ok", "message": "System gestartet (Config neu geladen)"}
        except Exception as exc:
            log.error("system_start_failed", error=str(exc))
            return {"error": "System konnte nicht gestartet werden", "status": 500}

    @app.post("/api/v1/system/stop", dependencies=deps)
    async def ui_system_stop() -> dict[str, Any]:
        """Initiates graceful shutdown if gateway is available."""
        try:
            if gateway is not None and hasattr(gateway, "shutdown"):
                # Fire-and-forget; the shutdown coroutine completes the
                # process so the dangling task is intentional.
                asyncio.create_task(gateway.shutdown())  # noqa: RUF006
                return {"status": "ok", "message": "Shutdown eingeleitet"}
            return {"status": "ok", "message": "Kein Gateway — nur Config-Server aktiv"}
        except Exception as exc:
            log.error("system_stop_failed", error=str(exc))
            return {"error": "Shutdown fehlgeschlagen", "status": 500}

    # -- System Detector (hardware profiling) -----------------------------

    @app.get("/api/v1/system/profile", dependencies=deps)
    async def get_system_profile() -> dict[str, Any]:
        """Get hardware/software system profile."""
        profile = getattr(gateway, "_system_profile", None)
        if not profile:
            return {"error": "System profile not available"}
        return profile.to_dict()

    @app.post("/api/v1/system/rescan", dependencies=deps)
    async def rescan_system() -> dict[str, Any]:
        """Force a full system re-scan."""
        try:
            from cognithor.system.detector import SystemDetector

            detector = SystemDetector()
            profile = detector.run_full_scan()
            cache = config_manager.config.cognithor_home / "system_profile.json"
            profile.save(cache)
            if gateway:
                gateway._system_profile = profile
            return profile.to_dict()
        except Exception as exc:
            return {"error": str(exc)}

    # -- Per-Agent Budget + Resource Monitor (Phase 3) --------------------

    @app.get("/api/v1/budget/agents", dependencies=deps)
    async def get_agent_budgets() -> dict[str, Any]:
        """Per-agent cost breakdown and budget status."""
        tracker = getattr(gateway, "_cost_tracker", None)
        if not tracker:
            return {"agents": {}, "message": "Cost tracking not available"}
        costs_today = tracker.get_agent_costs(days=1)
        costs_week = tracker.get_agent_costs(days=7)
        costs_month = tracker.get_agent_costs(days=30)
        # Check budgets from config
        evo_config = getattr(config_manager.config, "evolution", None)
        agent_budgets = getattr(evo_config, "agent_budgets", {}) if evo_config else {}
        statuses = {}
        for agent_name in set(list(costs_today.keys()) + list(agent_budgets.keys())):
            limit = agent_budgets.get(agent_name, 0.0)
            status = tracker.check_agent_budget(agent_name, limit)
            statuses[agent_name] = {
                "daily_cost_usd": status.daily_cost_usd,
                "daily_limit_usd": status.daily_limit_usd,
                "ok": status.ok,
                "warning": status.warning,
            }
        return {
            "agents_today": costs_today,
            "agents_week": costs_week,
            "agents_month": costs_month,
            "budgets": statuses,
        }

    @app.get("/api/v1/system/resources", dependencies=deps)
    async def get_system_resources() -> dict[str, Any]:
        """Current system resource usage (CPU/RAM/GPU)."""
        monitor = getattr(gateway, "_resource_monitor", None)
        if not monitor:
            return {"error": "Resource monitor not available"}
        try:
            snap = await monitor.sample()
            return snap.to_dict()
        except Exception as exc:
            return {"error": str(exc)}

    @app.get("/api/v1/evolution/stats", dependencies=deps)
    async def get_evolution_stats() -> dict[str, Any]:
        """Evolution loop statistics including resource, budget, and checkpoint info."""
        loop = getattr(gateway, "_evolution_loop", None)
        if not loop:
            return {"enabled": False, "message": "Evolution loop not active"}
        stats = loop.stats()
        # Enrich with resume state if checkpoint store available
        store = getattr(gateway, "_checkpoint_store", None)
        if store:
            try:
                from cognithor.evolution.resume import EvolutionResumer

                resumer = EvolutionResumer(store)
                latest = resumer.get_latest_cycle_id()
                if latest is not None:
                    rs = resumer.get_resume_state(latest)
                    stats["resume"] = rs.to_dict()
                else:
                    stats["resume"] = None
            except Exception:
                stats["resume"] = None
        return stats

    @app.post("/api/v1/evolution/resume", dependencies=deps)
    async def resume_evolution_cycle() -> dict[str, Any]:
        """Manually resume the last interrupted evolution cycle."""
        loop = getattr(gateway, "_evolution_loop", None)
        store = getattr(gateway, "_checkpoint_store", None)
        if not loop:
            return {"error": "Evolution loop not active"}
        if not store:
            return {"error": "Checkpoint store not available"}
        try:
            from cognithor.evolution.resume import EvolutionResumer

            resumer = EvolutionResumer(store)
            latest = resumer.get_latest_cycle_id()
            if latest is None:
                return {"error": "No checkpoints found", "resumed": False}
            state = resumer.get_resume_state(latest)
            if state.is_complete:
                return {
                    "resumed": False,
                    "reason": "Cycle already complete",
                    "cycle_id": latest,
                }
            if not state.has_checkpoint:
                return {"resumed": False, "reason": "No checkpoint to resume from"}
            # Trigger a new cycle (the loop will pick up from where it left off)
            result = await loop.run_cycle()
            return {
                "resumed": True,
                "cycle_id": result.cycle_id,
                "steps_completed": result.steps_completed,
                "skill_created": result.skill_created,
            }
        except Exception as exc:
            return {"error": str(exc), "resumed": False}

    @app.get("/api/v1/evolution/goals", dependencies=deps)
    async def get_evolution_goals() -> dict[str, Any]:
        """Get user-defined learning goals as structured objects."""
        # Primary source: GoalManager (has real-time progress from Evolution Engine)
        evo_loop = getattr(gateway, "_evolution_loop", None)
        gm = getattr(evo_loop, "_goal_manager", None) if evo_loop else None
        if gm is not None:
            from dataclasses import asdict

            return {"goals": [asdict(g) for g in gm._goals.values()]}

        # Fallback: config.yaml (no live progress)
        evo = getattr(config_manager.config, "evolution", None)
        raw = getattr(evo, "learning_goals", []) if evo else []
        goals = []
        for i, g in enumerate(raw):
            if isinstance(g, str):
                goals.append(
                    {
                        "id": f"goal_{i}",
                        "title": g,
                        "description": "",
                        "status": "active",
                        "priority": 3,
                        "progress": 0.0,
                    }
                )
            elif isinstance(g, dict):
                g.setdefault("id", f"goal_{i}")
                g.setdefault("status", "active")
                g.setdefault("priority", 3)
                g.setdefault("progress", 0.0)
                g.setdefault("description", "")
                goals.append(g)
        return {"goals": goals}

    def _save_goals(goals: list[dict[str, Any]]) -> None:
        config_manager.update_section("evolution", {"learning_goals": goals})
        config_manager.save()
        loop = getattr(gateway, "_evolution_loop", None)
        if loop and loop._config:
            loop._config.learning_goals = [g.get("title", "") for g in goals if isinstance(g, dict)]

    @app.put("/api/v1/evolution/goals", dependencies=deps)
    async def set_evolution_goals(request: Request) -> dict[str, Any]:
        """Replace all learning goals."""
        try:
            body = await request.json()
            goals = body.get("goals", [])
            if not isinstance(goals, list):
                return {"error": "goals must be a list"}
            # Normalize strings to dicts
            normalized = []
            for i, g in enumerate(goals):
                if isinstance(g, str):
                    normalized.append(
                        {
                            "id": f"goal_{i}",
                            "title": g,
                            "description": "",
                            "status": "active",
                            "priority": 3,
                            "progress": 0.0,
                        }
                    )
                elif isinstance(g, dict):
                    normalized.append(g)
            _save_goals(normalized)
            return {"goals": normalized, "count": len(normalized)}
        except Exception as exc:
            return {"error": str(exc)}

    @app.post("/api/v1/evolution/goals", dependencies=deps)
    async def add_evolution_goal(request: Request) -> dict[str, Any]:
        """Add a single learning goal."""
        try:
            body = await request.json()
            title = body.get("title", "").strip()
            if not title:
                return {"error": "title is required"}
            # Load existing
            evo = getattr(config_manager.config, "evolution", None)
            raw = getattr(evo, "learning_goals", []) if evo else []
            existing = []
            for i, g in enumerate(raw):
                if isinstance(g, str):
                    existing.append(
                        {
                            "id": f"goal_{i}",
                            "title": g,
                            "description": "",
                            "status": "active",
                            "priority": 3,
                            "progress": 0.0,
                        }
                    )
                elif isinstance(g, dict):
                    existing.append(g)
            # Add new
            import uuid

            new_goal = {
                "id": uuid.uuid4().hex[:12],
                "title": title,
                "description": body.get("description", ""),
                "status": "active",
                "priority": body.get("priority", 3),
                "progress": 0.0,
            }
            existing.append(new_goal)
            _save_goals(existing)
            return new_goal
        except Exception as exc:
            import traceback

            traceback.print_exc()
            return {"error": str(exc)}

    @app.patch("/api/v1/evolution/goals/{goal_id}", dependencies=deps)
    async def update_evolution_goal(goal_id: str, request: Request) -> dict[str, Any]:
        """Update a single learning goal."""
        try:
            body = await request.json()
            evo = getattr(config_manager.config, "evolution", None)
            raw = getattr(evo, "learning_goals", []) if evo else []
            goals = []
            for i, g in enumerate(raw):
                if isinstance(g, str):
                    goals.append(
                        {
                            "id": f"goal_{i}",
                            "title": g,
                            "description": "",
                            "status": "active",
                            "priority": 3,
                            "progress": 0.0,
                        }
                    )
                elif isinstance(g, dict):
                    goals.append(g)
            updated = None
            for g in goals:
                if g.get("id") == goal_id:
                    for k, v in body.items():
                        g[k] = v
                    updated = g
                    break
            if updated is None:
                return {"error": "Goal not found"}
            _save_goals(goals)
            return updated
        except Exception as exc:
            return {"error": str(exc)}

    @app.delete("/api/v1/evolution/goals/{goal_id}", dependencies=deps)
    async def delete_evolution_goal(goal_id: str) -> dict[str, Any]:
        """Delete a single learning goal."""
        try:
            evo = getattr(config_manager.config, "evolution", None)
            raw = getattr(evo, "learning_goals", []) if evo else []
            goals = [g for g in raw if isinstance(g, dict)]
            filtered = [g for g in goals if g.get("id") != goal_id]
            if len(filtered) == len(goals):
                return {"error": "Goal not found"}
            _save_goals(filtered)
            return {"deleted": goal_id}
        except Exception as exc:
            return {"error": str(exc)}

    @app.get("/api/v1/evolution/journal", dependencies=deps)
    async def get_evolution_journal(days: int = 7) -> dict[str, Any]:
        """Get evolution journal from recent cycle results and vault entries."""
        try:
            from datetime import UTC, datetime, timedelta
            from pathlib import Path as _P

            lines: list[str] = []
            cutoff = datetime.now(UTC) - timedelta(days=days)

            # Source 1: Evolution vault entries
            vault_dir = _P(config_manager.config.cognithor_home) / "vault" / "wissen"
            if vault_dir.exists():
                for f in sorted(
                    vault_dir.glob("*.md"), key=lambda x: x.stat().st_mtime, reverse=True
                )[:30]:
                    mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=UTC)
                    if mtime < cutoff:
                        continue
                    title = f.stem.replace("-", " ")[:80]
                    size_kb = f.stat().st_size / 1024
                    lines.append(
                        f"[{mtime.strftime('%Y-%m-%d %H:%M')}]"
                        f" Researched: {title} ({size_kb:.1f} KB)"
                    )

            # Source 2: Evolution loop stats
            evo_loop = getattr(gateway, "_evolution_loop", None)
            if evo_loop:
                stats = evo_loop.stats()
                lines.insert(0, "## Evolution Engine Status")
                lines.insert(1, f"- Cycles today: {stats.get('cycles_today', 0)}")
                lines.insert(2, f"- Total cycles: {stats.get('total_cycles', 0)}")
                lines.insert(3, f"- Status: {'Running' if stats.get('running') else 'Stopped'}")
                lines.insert(4, "")

            # Source 3: Deep learner plan progress
            dl = getattr(gateway, "_deep_learner", None)
            if dl:
                plans = dl.list_plans()
                if plans:
                    lines.append("")
                    lines.append("## Learning Plans Progress")
                    for p in plans:
                        passed = sum(
                            1 for sg in p.sub_goals if sg.status in ("verified", "completed")
                        )
                        total = len(p.sub_goals)
                        lines.append(f"- {p.goal[:60]}: {passed}/{total} sub-goals")

            content = "\n".join(lines) if lines else ""
            return {"content": content}
        except Exception as exc:
            log.error("evolution_journal_failed", error=str(exc))
            return {"content": "", "error": str(exc)}

    @app.get("/api/v1/evolution/plans", dependencies=deps)
    async def list_evolution_plans() -> dict[str, Any]:
        """List all learning plans."""
        dl = getattr(gateway, "_deep_learner", None)
        if not dl:
            return {"plans": [], "message": "DeepLearner not available"}
        plans = dl.list_plans()
        return {"plans": [p.to_summary_dict() for p in plans]}

    @app.get("/api/v1/evolution/plans/{plan_id}", dependencies=deps)
    async def get_evolution_plan(plan_id: str) -> dict[str, Any]:
        """Get detailed plan with SubGoals."""
        dl = getattr(gateway, "_deep_learner", None)
        if not dl:
            return {"error": "DeepLearner not available"}
        plan = dl.get_plan(plan_id)
        if not plan:
            return {"error": "Plan not found"}
        return plan.to_dict()

    @app.post("/api/v1/evolution/plans", dependencies=deps)
    async def create_evolution_plan(request: Request) -> dict[str, Any]:
        """Create a new learning plan from a goal."""
        dl = getattr(gateway, "_deep_learner", None)
        if not dl:
            return {"error": "DeepLearner not available"}
        try:
            body = await request.json()
            goal = body.get("goal", "")
            if not goal:
                return {"error": "goal is required"}
            seeds_raw = body.get("seed_sources", [])
            seeds = []
            for s in seeds_raw:
                from cognithor.evolution.models import SeedSource

                seeds.append(
                    SeedSource(
                        content_type=s.get("content_type", "hint"),
                        value=s.get("value", ""),
                        title=s.get("title", ""),
                    )
                )
            plan = await dl.create_plan(goal, seed_sources=seeds if seeds else None)
            return plan.to_summary_dict()
        except Exception as exc:
            return {"error": str(exc)}

    @app.patch("/api/v1/evolution/plans/{plan_id}", dependencies=deps)
    async def update_evolution_plan(plan_id: str, request: Request) -> dict[str, Any]:
        """Update plan status (pause/resume/delete)."""
        dl = getattr(gateway, "_deep_learner", None)
        if not dl:
            return {"error": "DeepLearner not available"}
        try:
            body = await request.json()
            action = body.get("action", "")
            if action == "delete":
                dl.delete_plan(plan_id)
                return {"deleted": True}
            elif action in ("pause", "resume", "complete"):
                status_map = {"pause": "paused", "resume": "active", "complete": "completed"}
                ok = dl.update_plan_status(plan_id, status_map[action])
                return {"updated": ok, "status": status_map[action]}
            return {"error": f"Unknown action: {action}"}
        except Exception as exc:
            return {"error": str(exc)}

    @app.get("/api/v1/evolution/plans/{plan_id}/index", dependencies=deps)
    async def get_plan_index_stats(plan_id: str) -> dict[str, Any]:
        """Get per-goal index statistics."""
        dl = getattr(gateway, "_deep_learner", None)
        if not dl:
            return {"error": "DeepLearner not available"}
        plan = dl.get_plan(plan_id)
        if not plan:
            return {"error": "Plan not found"}
        try:
            from cognithor.evolution.goal_index import GoalScopedIndex

            index_base = dl._plans_dir.parent / "indexes"
            idx = GoalScopedIndex(goal_slug=plan.goal_slug, base_dir=index_base)
            stats = idx.stats()
            idx.close()
            return stats
        except Exception as exc:
            return {"error": str(exc)}

    @app.get("/api/v1/evolution/claims", dependencies=deps)
    async def get_evolution_claims(goal_slug: str = "") -> dict[str, Any]:
        """Get knowledge claims table with confidence scores."""
        dl = getattr(gateway, "_deep_learner", None)
        if not dl or not dl._knowledge_validator:
            return {"claims": [], "summary": {}, "message": "KnowledgeValidator not available"}
        summary = dl._knowledge_validator.get_claims_summary()
        claims = dl._knowledge_validator.get_claims(limit=100)
        return {
            "summary": summary,
            "claims": [c.to_dict() for c in claims],
        }

    # -- 3.4: POST /agents/{name} ----------------------------------------

    @app.post("/api/v1/agents/{name}", dependencies=deps)
    async def ui_upsert_agent(name: str, request: Request) -> dict[str, Any]:
        """Creates or updates an agent profile in agents.yaml."""
        try:
            from cognithor.gateway.config_api import AgentProfileDTO

            body = await request.json()
            body["name"] = name
            validated = AgentProfileDTO(**body).model_dump(exclude_unset=False)
            data = _load_yaml(agents_path)
            agents = data.get("agents", [])
            if not isinstance(agents, list):
                agents = []
            # Upsert by name
            found = False
            for i, a in enumerate(agents):
                if isinstance(a, dict) and a.get("name") == name:
                    agents[i] = validated
                    found = True
                    break
            if not found:
                agents.append(validated)
            data["agents"] = agents
            _save_yaml(agents_path, data)
            return {"status": "ok", "agent": name}
        except Exception as exc:
            log.error("agent_upsert_failed", agent=name, error=str(exc))
            return {"error": "Agent konnte nicht gespeichert werden", "status": 400}

    # -- 3.5: POST /bindings/{name} --------------------------------------

    @app.post("/api/v1/bindings/{name}", dependencies=deps)
    async def ui_upsert_binding(name: str, request: Request) -> dict[str, Any]:
        """Creates or updates a binding rule in bindings.yaml."""
        try:
            from cognithor.gateway.config_api import BindingRuleDTO

            body = await request.json()
            body["name"] = name
            validated = BindingRuleDTO(**body).model_dump(exclude_unset=False)
            data = _load_yaml(bindings_path)
            bindings = data.get("bindings", [])
            if not isinstance(bindings, list):
                bindings = []
            found = False
            for i, b in enumerate(bindings):
                if isinstance(b, dict) and b.get("name") == name:
                    bindings[i] = validated
                    found = True
                    break
            if not found:
                bindings.append(validated)
            data["bindings"] = bindings
            _save_yaml(bindings_path, data)
            return {"status": "ok", "binding": name}
        except Exception as exc:
            log.error("binding_upsert_failed", binding=name, error=str(exc))
            return {"error": "Binding konnte nicht gespeichert werden", "status": 400}

    # -- 3.6: Prompts GET / PUT ------------------------------------------

    @app.get("/api/v1/prompts", dependencies=deps)
    async def ui_get_prompts() -> dict[str, Any]:
        """Reads prompt/policy files for the Prompts & Policies page."""
        cfg = config_manager.config
        prompts_dir = cognithor_home / "prompts"
        result: dict[str, str] = {}

        # coreMd
        try:
            core_path = cfg.core_memory_file
            result["coreMd"] = core_path.read_text(encoding="utf-8") if core_path.exists() else ""
        except Exception:
            result["coreMd"] = ""

        # plannerSystem (.md bevorzugt, .txt als Migration-Fallback)
        try:
            from cognithor.core.planner import SYSTEM_PROMPT

            content = ""
            for fname in ("SYSTEM_PROMPT.md", "SYSTEM_PROMPT.txt"):
                sys_path = prompts_dir / fname
                if sys_path.exists():
                    content = sys_path.read_text(encoding="utf-8").strip()
                    if content:
                        break
            if not content:
                content = SYSTEM_PROMPT
            result["plannerSystem"] = content
        except Exception:
            result["plannerSystem"] = ""

        # replanPrompt
        try:
            from cognithor.core.planner import REPLAN_PROMPT

            content = ""
            for fname in ("REPLAN_PROMPT.md", "REPLAN_PROMPT.txt"):
                rp_path = prompts_dir / fname
                if rp_path.exists():
                    content = rp_path.read_text(encoding="utf-8").strip()
                    if content:
                        break
            if not content:
                content = REPLAN_PROMPT
            result["replanPrompt"] = content
        except Exception:
            result["replanPrompt"] = ""

        # escalationPrompt
        try:
            from cognithor.core.planner import ESCALATION_PROMPT

            content = ""
            for fname in ("ESCALATION_PROMPT.md", "ESCALATION_PROMPT.txt"):
                ep_path = prompts_dir / fname
                if ep_path.exists():
                    content = ep_path.read_text(encoding="utf-8").strip()
                    if content:
                        break
            if not content:
                content = ESCALATION_PROMPT
            result["escalationPrompt"] = content
        except Exception:
            result["escalationPrompt"] = ""

        # policyYaml
        try:
            policy_path = cfg.policies_dir / "default.yaml"
            content = policy_path.read_text(encoding="utf-8") if policy_path.exists() else ""
            result["policyYaml"] = content
        except Exception:
            result["policyYaml"] = ""

        # heartbeatMd
        try:
            hb_path = cognithor_home / cfg.heartbeat.checklist_file
            result["heartbeatMd"] = hb_path.read_text(encoding="utf-8") if hb_path.exists() else ""
        except Exception:
            result["heartbeatMd"] = ""

        return result

    @app.put("/api/v1/prompts", dependencies=deps)
    async def ui_put_prompts(request: Request) -> dict[str, Any]:
        """Writes prompt/policy files from the UI."""
        try:
            body = await request.json()
            cfg = config_manager.config
            prompts_dir = cognithor_home / "prompts"
            prompts_dir.mkdir(parents=True, exist_ok=True)
            written: list[str] = []

            if "coreMd" in body:
                cfg.core_memory_file.parent.mkdir(parents=True, exist_ok=True)
                cfg.core_memory_file.write_text(body["coreMd"], encoding="utf-8")
                written.append("coreMd")

            if "plannerSystem" in body:
                (prompts_dir / "SYSTEM_PROMPT.md").write_text(
                    body["plannerSystem"], encoding="utf-8"
                )
                written.append("plannerSystem")

            if "replanPrompt" in body:
                (prompts_dir / "REPLAN_PROMPT.md").write_text(
                    body["replanPrompt"], encoding="utf-8"
                )
                written.append("replanPrompt")

            if "escalationPrompt" in body:
                (prompts_dir / "ESCALATION_PROMPT.md").write_text(
                    body["escalationPrompt"], encoding="utf-8"
                )
                written.append("escalationPrompt")

            if "policyYaml" in body:
                cfg.policies_dir.mkdir(parents=True, exist_ok=True)
                (cfg.policies_dir / "default.yaml").write_text(body["policyYaml"], encoding="utf-8")
                written.append("policyYaml")

            if "heartbeatMd" in body:
                hb_path = cognithor_home / cfg.heartbeat.checklist_file
                hb_path.parent.mkdir(parents=True, exist_ok=True)
                hb_path.write_text(body["heartbeatMd"], encoding="utf-8")
                written.append("heartbeatMd")

            # Live-Reload: Gateway-Komponenten sofort aktualisieren
            if gateway is not None and hasattr(gateway, "reload_components") and written:
                reload_flags: dict[str, bool] = {}
                if any(k in written for k in ("plannerSystem", "replanPrompt", "escalationPrompt")):
                    reload_flags["prompts"] = True
                if "policyYaml" in written:
                    reload_flags["policies"] = True
                if "coreMd" in written:
                    reload_flags["core_memory"] = True
                if reload_flags:
                    gateway.reload_components(**reload_flags)

            return {"status": "ok", "written": written}
        except Exception as exc:
            log.error("prompts_put_failed", error=str(exc))
            return {"error": "Prompts konnten nicht gespeichert werden", "status": 400}

    # -- 3.7: Cron Jobs GET / PUT ----------------------------------------

    @app.get("/api/v1/cron-jobs", dependencies=deps)
    async def ui_get_cron_jobs() -> dict[str, Any]:
        """Reads cron jobs via JobStore."""
        try:
            from cognithor.cron.jobs import JobStore

            store = JobStore(config_manager.config.cron_config_file)
            jobs = store.load()
            return {
                "jobs": [
                    {
                        "name": j.name,
                        "schedule": j.schedule,
                        "prompt": j.prompt,
                        "channel": j.channel,
                        "model": j.model,
                        "enabled": j.enabled,
                        "agent": j.agent,
                    }
                    for j in jobs.values()
                ],
            }
        except Exception as exc:
            log.error("cron_jobs_get_failed", error=str(exc))
            return {"jobs": [], "error": "Cron-Jobs konnten nicht geladen werden"}

    @app.put("/api/v1/cron-jobs", dependencies=deps)
    async def ui_put_cron_jobs(request: Request) -> dict[str, Any]:
        """Writes cron jobs via JobStore."""
        try:
            from cognithor.cron.jobs import JobStore
            from cognithor.models import CronJob

            body = await request.json()
            store = JobStore(config_manager.config.cron_config_file)
            store.load()
            store.jobs = {}
            for j in body.get("jobs", []):
                if not isinstance(j, dict) or "name" not in j:
                    continue
                store.jobs[j["name"]] = CronJob(**j)
            store._save()
            return {"status": "ok", "count": len(store.jobs)}
        except Exception as exc:
            log.error("cron_jobs_put_failed", error=str(exc))
            return {"error": "Cron-Jobs konnten nicht gespeichert werden", "status": 400}

    # -- 3.7b: Cron-Jobs toggle + enriched list ---------------------------

    @app.patch("/api/v1/cron-jobs/{job_name}/toggle", dependencies=deps)
    async def ui_toggle_cron_job(job_name: str) -> dict[str, Any]:
        """Toggle a cron job enabled/disabled."""
        try:
            from cognithor.cron.jobs import JobStore

            store = JobStore(config_manager.config.cron_config_file)
            store.load()
            if job_name not in store.jobs:
                return {"error": f"Job '{job_name}' not found", "status": 404}
            new_enabled = not store.jobs[job_name].enabled
            store.toggle_job(job_name, new_enabled)
            # Also update the live scheduler if available
            gw = getattr(config_manager, "_gateway", None)
            cron_engine = getattr(gw, "_cron_engine", None) if gw else None
            if cron_engine and hasattr(cron_engine, "job_store"):
                cron_engine.job_store.jobs[job_name] = store.jobs[job_name]
            return {"name": job_name, "enabled": new_enabled}
        except Exception as exc:
            log.error("cron_job_toggle_failed", error=str(exc))
            return {"error": str(exc), "status": 500}

    @app.get("/api/v1/cron-jobs/enriched", dependencies=deps)
    async def ui_get_cron_jobs_enriched() -> dict[str, Any]:
        """Returns cron jobs with next_run times and last_run info."""
        try:
            from cognithor.cron.jobs import JobStore

            store = JobStore(config_manager.config.cron_config_file)
            jobs = store.load()

            # Try to get next_run from live engine
            gw = getattr(config_manager, "_gateway", None)
            cron_engine = getattr(gw, "_cron_engine", None) if gw else None
            next_runs: dict[str, Any] = {}
            if cron_engine:
                try:
                    next_runs = cron_engine.get_next_run_times()
                except Exception:
                    log.debug("cron_next_run_times_fetch_failed", exc_info=True)

            result = []
            for j in jobs.values():
                nr = next_runs.get(j.name)
                result.append(
                    {
                        "name": j.name,
                        "schedule": j.schedule,
                        "prompt": j.prompt,
                        "channel": j.channel,
                        "model": j.model,
                        "enabled": j.enabled,
                        "agent": j.agent,
                        "next_run": nr.isoformat() if nr else None,
                    }
                )
            return {"jobs": result}
        except Exception as exc:
            log.error("cron_jobs_enriched_failed", error=str(exc))
            return {"jobs": [], "error": str(exc)}

    # -- 3.8: MCP Servers GET / PUT --------------------------------------

    @app.get("/api/v1/mcp-servers", dependencies=deps)
    async def ui_get_mcp_servers() -> dict[str, Any]:
        """Reads MCP server config, flattened for the UI."""
        try:
            mcp_path = config_manager.config.mcp_config_file
            data = _load_yaml(mcp_path)
            sm = data.get("server_mode", {})
            servers_raw = data.get("servers", {})
            # servers can be dict (name→config) or list; normalize to dict
            if isinstance(servers_raw, list):
                servers_dict = {
                    s.get("name", f"server_{i}"): s
                    for i, s in enumerate(servers_raw)
                    if isinstance(s, dict)
                }
            elif isinstance(servers_raw, dict):
                servers_dict = servers_raw
            else:
                servers_dict = {}
            # Flatten server_mode fields into response + external_servers as dict
            result: dict[str, Any] = {
                "mode": sm.get("mode", "disabled"),
                "http_host": sm.get("http_host", "127.0.0.1"),
                "http_port": sm.get("http_port", 3001),
                "server_name": sm.get("server_name", "jarvis"),
                "require_auth": sm.get("require_auth", False),
                "auth_token": "***" if sm.get("auth_token") else "",
                "expose_tools": sm.get("expose_tools", True),
                "expose_resources": sm.get("expose_resources", True),
                "expose_prompts": sm.get("expose_prompts", False),
                "enable_sampling": sm.get("enable_sampling", False),
                "external_servers": servers_dict,
            }
            return result
        except Exception as exc:
            log.error("mcp_servers_load_failed", error=str(exc))
            return {
                "mode": "disabled",
                "external_servers": {},
                "error": "MCP-Server-Konfiguration konnte nicht geladen werden",
            }

    @app.put("/api/v1/mcp-servers", dependencies=deps)
    async def ui_put_mcp_servers(request: Request) -> dict[str, Any]:
        """Writes MCP config from flat UI format, preserving a2a and built_in_tools."""
        try:
            body = await request.json()
            mcp_path = config_manager.config.mcp_config_file
            data = _load_yaml(mcp_path)
            # Reconstruct server_mode from flat fields
            sm_keys = (
                "mode",
                "http_host",
                "http_port",
                "server_name",
                "require_auth",
                "auth_token",
                "expose_tools",
                "expose_resources",
                "expose_prompts",
                "enable_sampling",
            )
            sm = data.get("server_mode", {})
            for k in sm_keys:
                if k in body:
                    sm[k] = body[k]
            data["server_mode"] = sm
            # external_servers → servers
            if "external_servers" in body:
                data["servers"] = body["external_servers"]
            _save_yaml(mcp_path, data)
            return {"status": "ok"}
        except Exception as exc:
            log.error("mcp_servers_put_failed", error=str(exc))
            return {
                "error": "MCP-Server-Konfiguration konnte nicht gespeichert werden",
                "status": 400,
            }

    # -- 3.9: A2A GET / PUT ----------------------------------------------

    @app.get("/api/v1/a2a", dependencies=deps)
    async def ui_get_a2a() -> dict[str, Any]:
        """Reads a2a section from MCP config."""
        try:
            mcp_path = config_manager.config.mcp_config_file
            data = _load_yaml(mcp_path)
            a2a = data.get("a2a", {})
            # Provide sensible defaults
            return {
                "enabled": a2a.get("enabled", False),
                "host": a2a.get("host", "0.0.0.0"),
                "port": a2a.get("port", 8742),
                "agent_name": a2a.get("agent_name", "jarvis"),
                **{
                    k: v
                    for k, v in a2a.items()
                    if k not in ("enabled", "host", "port", "agent_name")
                },
            }
        except Exception as exc:
            log.error("a2a_get_failed", error=str(exc))
            return {"enabled": False, "error": "A2A-Konfiguration konnte nicht geladen werden"}

    @app.put("/api/v1/a2a", dependencies=deps)
    async def ui_put_a2a(request: Request) -> dict[str, Any]:
        """Writes a2a section, preserving other MCP config sections."""
        try:
            body = await request.json()
            mcp_path = config_manager.config.mcp_config_file
            data = _load_yaml(mcp_path)
            data["a2a"] = body
            _save_yaml(mcp_path, data)
            return {"status": "ok"}
        except Exception as exc:
            log.error("a2a_config_save_failed", error=str(exc))
            return {"error": "A2A-Konfiguration konnte nicht gespeichert werden", "status": 400}
