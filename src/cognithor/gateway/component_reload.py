"""Component-reload coordinator — extracted from Gateway.

Live-update path for prompts, policies, config, core_memory, and skills
that the UI / config-API can call without a full restart. Each section
is independent: failures are logged and the rest of the reload still
runs.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any

from cognithor.config import load_config
from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.gateway.gateway import Gateway

log = get_logger(__name__)


def reload_components(
    gw: Gateway,
    *,
    prompts: bool = False,
    policies: bool = False,
    config: bool = False,
    core_memory: bool = False,
    skills: bool = False,
) -> dict[str, Any]:
    """Coordinate live reload of selected gateway subsystems.

    Returns a dict ``{"reloaded": [...]}`` listing the sections that
    actually applied. Sections that fail mid-reload are logged but do
    not prevent later sections from running.
    """
    reloaded: list[str] = []

    if prompts and gw._planner:
        gw._planner.reload_prompts()
        reloaded.append("prompts")

    if policies and gw._gatekeeper:
        gw._gatekeeper.reload_policies()
        reloaded.append("policies")

    if core_memory:
        core_path = gw._config.core_memory_path
        if core_path.exists():
            try:
                text = core_path.read_text(encoding="utf-8")
                for wm in gw._working_memories.values():
                    wm.core_memory_text = text
                reloaded.append("core_memory")
            except Exception:
                log.debug("reload_core_memory_failed", exc_info=True)

    if skills and gw._skill_registry:
        try:
            skill_dirs = [
                gw._config.cognithor_home / "data" / "procedures",
                gw._config.cognithor_home / gw._config.plugins.skills_dir,
            ]
            gw._skill_registry.load_from_directories(skill_dirs)
            reloaded.append("skills")
        except Exception:
            log.warning("skills_reload_failed", exc_info=True)

    if config:
        # Reload config.yaml from disk
        try:
            new_config = load_config(gw._config.config_file)
            gw._config = new_config
        except Exception:
            log.debug("config_file_reload_failed", exc_info=True)
            new_config = gw._config

        # Live-update i18n locale from config
        try:
            from cognithor.i18n import set_locale

            _lang = os.environ.get("COGNITHOR_LANGUAGE") or new_config.language
            set_locale(_lang)
        except Exception:
            log.debug("i18n_locale_reload_failed", exc_info=True)

        # Live-update Executor runtime parameters
        if gw._executor and hasattr(gw._executor, "reload_config"):
            try:
                gw._executor.reload_config(new_config)
            except Exception:
                log.debug("executor_config_reload_failed", exc_info=True)

        # Live-update ModelRouter with new config + schedule model list refresh
        if gw._model_router and hasattr(gw._model_router, "_config"):
            try:
                gw._model_router._config = new_config
                # Schedule async re-initialization to refresh _available_models
                try:
                    loop = asyncio.get_running_loop()
                    _task = loop.create_task(gw._model_router.initialize())
                    gw._background_tasks.add(_task)
                    _task.add_done_callback(gw._background_tasks.discard)
                except RuntimeError:
                    pass  # no loop — model list refresh skipped
                log.info("model_router_config_reloaded")
            except Exception:
                log.debug("model_router_config_reload_failed", exc_info=True)

        # Recreate UnifiedLLMClient if backend type changed
        if gw._llm is not None:
            old_backend = getattr(gw._llm, "backend_type", "ollama")
            new_backend = new_config.llm_backend_type
            if old_backend != new_backend:
                try:
                    from cognithor.core.unified_llm import UnifiedLLMClient

                    old_llm = gw._llm
                    gw._llm = UnifiedLLMClient.create(new_config)
                    # Update references in Planner/Executor
                    if gw._planner and hasattr(gw._planner, "_ollama"):
                        gw._planner._ollama = gw._llm
                    if gw._executor and hasattr(gw._executor, "_ollama"):
                        gw._executor._ollama = gw._llm
                    # Close old client
                    try:
                        loop = asyncio.get_running_loop()
                        _task = loop.create_task(old_llm.close())
                        gw._background_tasks.add(_task)
                        _task.add_done_callback(gw._background_tasks.discard)
                    except RuntimeError:
                        pass
                    log.info(
                        "llm_backend_switched",
                        old=old_backend,
                        new=new_backend,
                    )
                except Exception:
                    log.warning("llm_backend_switch_failed", exc_info=True)

        # Live-update Planner with new config
        if gw._planner and hasattr(gw._planner, "_config"):
            try:
                gw._planner._config = new_config
            except Exception:
                log.debug("planner_config_reload_failed", exc_info=True)

        # Live-update WebTools runtime parameters
        web_tools = None
        if gw._mcp_client:
            handler = gw._mcp_client.get_handler("web_search")
            if handler is not None:
                web_tools = getattr(handler, "__self__", None)
        if web_tools and hasattr(web_tools, "reload_config"):
            try:
                web_tools.reload_config(new_config)
            except Exception:
                log.debug("web_tools_config_reload_failed", exc_info=True)

        # Live-update Gatekeeper tool toggles (disabled_tools list)
        if gw._gatekeeper and hasattr(gw._gatekeeper, "reload_disabled_tools"):
            try:
                gw._gatekeeper.reload_disabled_tools()
                reloaded.append("tool_toggles")
            except Exception:
                log.debug("gatekeeper_tool_toggles_reload_failed", exc_info=True)

        reloaded.append("config")

    log.info("gateway_components_reloaded", components=reloaded)
    return {"reloaded": reloaded}
