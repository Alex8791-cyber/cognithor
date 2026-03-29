"""Tools phase: MCP client, browser agent, graph engine, telemetry, HITL, A2A.

Attributes handled:
  _mcp_client, _mcp_bridge, _browser_agent, _graph_engine,
  _telemetry_hub, _hitl_manager, _a2a_adapter
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.gateway.phases import PhaseResult

log = get_logger(__name__)


def declare_tools_attrs(config: Any) -> PhaseResult:
    """Return default (None) values for all tool-related attributes."""
    return {
        "mcp_client": None,
        "mcp_bridge": None,
        "browser_agent": None,
        "graph_engine": None,
        "telemetry_hub": None,
        "hitl_manager": None,
        "a2a_adapter": None,
        "cost_tracker": None,
        "vault_tools": None,
    }


def _register_a2a_tools(mcp_client: Any, a2a_adapter: Any) -> None:
    """Register A2A delegation tools so the Planner can delegate to remote agents."""

    async def _list_remote_agents(**_kwargs: Any) -> str:
        """List all registered remote A2A agents."""
        if not a2a_adapter or not hasattr(a2a_adapter, "_client") or not a2a_adapter._client:
            return "A2A not available or no client configured."
        remotes = a2a_adapter._client.list_remotes()
        if not remotes:
            return "No remote agents registered. Use delegate_to_remote_agent with an endpoint URL."
        lines = []
        for r in remotes:
            name = r.card.name if r.card else "unknown"
            skills_str = ""
            if r.card and r.card.skills:
                skills_str = ", ".join(s.name for s in r.card.skills[:5])
            lines.append(f"- {name} ({r.endpoint}) skills: [{skills_str}]")
        return "\n".join(lines)

    async def _delegate_to_remote_agent(**kwargs: Any) -> str:
        """Send a task to a remote A2A agent and return the result."""
        endpoint = kwargs.get("endpoint", "")
        task_text = kwargs.get("task", "")
        if not endpoint or not task_text:
            return "Error: 'endpoint' and 'task' are required."
        if not a2a_adapter:
            return "Error: A2A adapter not available."

        # Auto-discover if not yet known
        if hasattr(a2a_adapter, "_client") and a2a_adapter._client:
            remote = a2a_adapter._client.get_remote(endpoint)
            if not remote:
                await a2a_adapter.discover_remote(endpoint)

        task = await a2a_adapter.delegate_task(endpoint=endpoint, text=task_text)
        if task is None:
            return f"Failed to delegate task to {endpoint}. Agent may be unreachable."

        # Extract result text from artifacts
        parts = []
        if task.artifacts:
            for art in task.artifacts:
                for p in art.parts:
                    if hasattr(p, "text"):
                        parts.append(p.text)
        if parts:
            return "\n".join(parts)
        return f"Task delegated (id={task.id}, state={task.state.value}). No text result yet."

    mcp_client.register_builtin_handler(
        "list_remote_agents",
        _list_remote_agents,
        description="List all registered remote A2A agents and their capabilities.",
        input_schema={"type": "object", "properties": {}},
    )

    mcp_client.register_builtin_handler(
        "delegate_to_remote_agent",
        _delegate_to_remote_agent,
        description=(
            "Delegate a task to a remote A2A agent. The agent processes the task "
            "and returns a result. Use list_remote_agents first to see available agents."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "endpoint": {
                    "type": "string",
                    "description": "The remote agent's base URL (e.g. http://192.168.1.10:3002)",
                },
                "task": {
                    "type": "string",
                    "description": "The task description to send to the remote agent",
                },
            },
            "required": ["endpoint", "task"],
        },
    )

    log.info("a2a_tools_registered", tools=["list_remote_agents", "delegate_to_remote_agent"])


async def init_tools(
    config: Any,
    mcp_client: Any,
    memory_manager: Any,
    interop: Any = None,
    handle_message: Any = None,
    gateway: Any = None,
) -> PhaseResult:
    """Initialize MCP tools, browser, graph engine, telemetry, HITL, A2A.

    Args:
        config: JarvisConfig instance.
        mcp_client: Already-created JarvisMCPClient (tools are registered on it).
        memory_manager: MemoryManager for memory tools and MCP bridge.
        interop: InteropProtocol instance (optional, for A2A).
        handle_message: Gateway.handle_message callback (optional, for A2A).

    Returns:
        PhaseResult with initialized tool subsystems.
    """
    from jarvis.mcp.bridge import MCPBridge
    from jarvis.mcp.chart_tools import register_chart_tools
    from jarvis.mcp.code_tools import register_code_tools
    from jarvis.mcp.database_tools import register_database_tools
    from jarvis.mcp.filesystem import register_fs_tools
    from jarvis.mcp.git_tools import register_git_tools
    from jarvis.mcp.memory_server import register_memory_tools
    from jarvis.mcp.search_tools import register_search_tools
    from jarvis.mcp.shell import register_shell_tools
    from jarvis.mcp.web import register_web_tools

    result: PhaseResult = {"mcp_client": mcp_client}

    # Register built-in MCP tools
    register_fs_tools(mcp_client, config)
    register_shell_tools(mcp_client, config)
    web_tools = register_web_tools(mcp_client, config)
    register_code_tools(mcp_client, config)
    register_git_tools(mcp_client, config)
    register_search_tools(mcp_client, config)
    register_database_tools(mcp_client, config)
    register_chart_tools(mcp_client, config)

    # ARC-AGI-3 Benchmark tools (optional — guarded by config.arc.enabled)
    if getattr(config, "arc", None) and getattr(config.arc, "enabled", False):
        try:
            from jarvis.mcp.arc_tools import register_arc_tools

            register_arc_tools(mcp_client)
            log.info("arc_tools_registered", tools=["arc_play", "arc_status", "arc_replay"])
        except Exception:
            log.debug("arc_tools_not_available", exc_info=True)
    else:
        log.debug("arc_tools_disabled_by_config")

    # Browser-Use v17: Autonomous browser automation (optional)
    browser_agent = None
    try:
        from jarvis.browser.tools import register_browser_use_tools

        # Vision-Analyzer erstellen wenn vision_model konfiguriert
        vision_analyzer = None
        vision_model = getattr(config, "vision_model", "")
        if vision_model:
            try:
                from jarvis.browser.vision import VisionAnalyzer, VisionConfig
                from jarvis.core.unified_llm import UnifiedLLMClient

                llm_for_vision = UnifiedLLMClient.create(config)
                vision_config = VisionConfig(
                    enabled=True,
                    model=vision_model,
                    backend_type=getattr(config, "llm_backend_type", "ollama"),
                )
                vision_analyzer = VisionAnalyzer(llm_for_vision, vision_config)
                log.info("vision_analyzer_created", model=vision_model)
            except Exception:
                log.debug("vision_analyzer_init_skipped", exc_info=True)

        browser_agent = register_browser_use_tools(mcp_client, vision_analyzer=vision_analyzer)
        log.info("browser_use_v17_registered")
    except Exception:
        log.debug("browser_use_v17_init_skipped", exc_info=True)
        # Fallback: Basic browser tools (v14)
        try:
            from jarvis.mcp.browser import register_browser_tools

            register_browser_tools(mcp_client)
        except Exception:
            log.warning("browser_tools_not_registered", exc_info=True)
    result["browser_agent"] = browser_agent

    # Graph Orchestrator v18: DAG-based workflow engine (optional)
    graph_engine = None
    try:
        from jarvis.graph.engine import GraphEngine
        from jarvis.graph.state import StateManager

        graph_engine = GraphEngine(state_manager=StateManager())
        log.info("graph_engine_v18_registered")
    except Exception:
        log.debug("graph_engine_not_available")
    result["graph_engine"] = graph_engine

    # OpenTelemetry v19: Distributed Tracing & Metrics (optional)
    telemetry_hub = None
    try:
        from jarvis.telemetry.instrumentation import TelemetryHub

        telemetry_hub = TelemetryHub(service_name="jarvis")
        log.info("telemetry_v19_registered")
    except Exception:
        log.debug("telemetry_not_available")
    result["telemetry_hub"] = telemetry_hub

    # Human-in-the-Loop v20: Approval workflows (optional)
    hitl_manager = None
    try:
        from jarvis.hitl.manager import ApprovalManager

        hitl_manager = ApprovalManager()
        log.info("hitl_v20_registered")
    except Exception:
        log.debug("hitl_not_available")
    result["hitl_manager"] = hitl_manager

    # Media-Tools (Audio/Image/Document processing)
    media_pipeline = None
    try:
        from jarvis.mcp.media import register_media_tools

        media_pipeline = register_media_tools(mcp_client, config)
    except Exception:
        log.warning("media_tools_not_registered")

    # Knowledge Vault (Obsidian-kompatible Notizen)
    vault_tools = None
    try:
        from jarvis.mcp.vault import register_vault_tools

        vault_tools = register_vault_tools(mcp_client, config)
        log.info("vault_tools_registered")
    except Exception:
        log.warning("vault_tools_not_registered")
    result["vault_tools"] = vault_tools

    # OSINT / Human Investigation Module (optional)
    try:
        from jarvis.mcp.osint_tools import register_osint_tools

        register_osint_tools(mcp_client, config)
    except Exception:
        log.debug("osint_tools_not_registered", exc_info=True)

    # LLM + Vault in MediaPipeline injizieren (fuer analyze_document)
    if media_pipeline is not None and hasattr(media_pipeline, "_set_llm_fn"):
        try:
            from jarvis.core.unified_llm import UnifiedLLMClient

            llm_client = UnifiedLLMClient.create(config)
            model_name = getattr(getattr(config, "models", None), "planner", None)
            model_name = getattr(model_name, "name", "") if model_name else ""

            async def _llm_for_analysis(prompt: str, model: str = "") -> str:
                resp = await llm_client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    model=model or model_name,
                )
                return resp.get("content", "") if isinstance(resp, dict) else str(resp)

            media_pipeline._set_llm_fn(_llm_for_analysis, model_name)
            log.info("media_llm_injected", model=model_name)
        except Exception:
            log.debug("media_llm_injection_skipped", exc_info=True)

    if (
        media_pipeline is not None
        and vault_tools is not None
        and hasattr(media_pipeline, "_set_vault")
    ):
        media_pipeline._set_vault(vault_tools)
        log.debug("media_vault_injected")

    # Memory tools
    memory_tools = register_memory_tools(mcp_client, memory_manager)

    # Knowledge Synthesis (orchestrates Memory + Vault + Web + LLM)
    synthesizer = None
    try:
        from jarvis.mcp.synthesis import register_synthesis_tools

        synthesizer = register_synthesis_tools(mcp_client, config)
        log.info("synthesis_tools_registered")
    except Exception:
        log.warning("synthesis_tools_not_registered", exc_info=True)

    # Inject dependencies into synthesizer
    if synthesizer is not None:
        # LLM (reuse the closure already created for media pipeline)
        if hasattr(synthesizer, "_set_llm_fn"):
            try:
                # Ensure LLM client + closure exist (may have been created above for media)
                if (
                    media_pipeline is not None
                    and hasattr(media_pipeline, "_llm_fn")
                    and media_pipeline._llm_fn is not None
                ):
                    # Reuse existing LLM function from media pipeline
                    synthesizer._set_llm_fn(media_pipeline._llm_fn, media_pipeline._llm_model)
                else:
                    from jarvis.core.unified_llm import UnifiedLLMClient

                    llm_client_synth = UnifiedLLMClient.create(config)
                    model_name_synth = getattr(getattr(config, "models", None), "planner", None)
                    model_name_synth = (
                        getattr(model_name_synth, "name", "") if model_name_synth else ""
                    )

                    async def _llm_for_synthesis(prompt: str, model: str = "") -> str:
                        resp = await llm_client_synth.chat(
                            messages=[{"role": "user", "content": prompt}],
                            model=model or model_name_synth,
                        )
                        return resp.get("content", "") if isinstance(resp, dict) else str(resp)

                    synthesizer._set_llm_fn(_llm_for_synthesis, model_name_synth)
                log.info("synthesis_llm_injected")
            except Exception:
                log.debug("synthesis_llm_injection_skipped", exc_info=True)

        # Memory tools
        if memory_tools is not None and hasattr(synthesizer, "_set_memory_tools"):
            synthesizer._set_memory_tools(memory_tools)
            log.debug("synthesis_memory_injected")

        # Vault tools
        if vault_tools is not None and hasattr(synthesizer, "_set_vault_tools"):
            synthesizer._set_vault_tools(vault_tools)
            log.debug("synthesis_vault_injected")

        # Web tools
        if web_tools is not None and hasattr(synthesizer, "_set_web_tools"):
            synthesizer._set_web_tools(web_tools)
            log.debug("synthesis_web_injected")

    # Email tools (optional)
    try:
        from jarvis.mcp.email_tools import register_email_tools

        register_email_tools(mcp_client, config)
    except Exception:
        log.debug("email_tools_not_registered")

    # Calendar tools (optional)
    try:
        from jarvis.mcp.calendar_tools import register_calendar_tools

        register_calendar_tools(mcp_client, config)
    except Exception:
        log.debug("calendar_tools_not_registered")

    # Docker tools (optional -- requires docker CLI)
    try:
        from jarvis.mcp.docker_tools import register_docker_tools

        register_docker_tools(mcp_client, config)
    except Exception:
        log.debug("docker_tools_not_registered", exc_info=True)

    # Remote Shell Tools (SSH)
    try:
        from jarvis.mcp.remote_shell import register_remote_shell_tools

        remote_cfg = getattr(config, "remote_shell", None)
        register_remote_shell_tools(
            mcp_client,
            config=remote_cfg._asdict() if remote_cfg else None,
        )
        log.info("remote_shell_tools_registered")
    except Exception:
        log.debug("remote_shell_skip", exc_info=True)

    # API Integration Hub
    try:
        from jarvis.mcp.api_hub import register_api_hub_tools

        register_api_hub_tools(mcp_client, config)
        log.info("api_hub_registered")
    except Exception:
        log.debug("api_hub_not_registered", exc_info=True)

        # MCP-Server mode (optional, only if enabled in config)
    mcp_bridge = None
    try:
        mcp_bridge = MCPBridge(config)
        if mcp_bridge.setup(mcp_client, memory_manager):
            log.info("mcp_server_mode_enabled")
        else:
            log.debug("mcp_server_mode_disabled")
    except Exception as exc:
        log.debug("mcp_bridge_not_available", reason=str(exc))
        mcp_bridge = None
    result["mcp_bridge"] = mcp_bridge

    # A2A Protocol (optional, only if enabled in config)
    a2a_adapter = None
    try:
        from jarvis.a2a.adapter import A2AAdapter

        a2a_adapter = A2AAdapter(config)
        if a2a_adapter.setup(interop, handle_message):
            log.info("a2a_protocol_enabled")
            _register_a2a_tools(mcp_client, a2a_adapter)
        else:
            log.debug("a2a_protocol_disabled")
    except Exception as exc:
        log.debug("a2a_adapter_not_available", reason=str(exc))
        a2a_adapter = None
    result["a2a_adapter"] = a2a_adapter

    # CostTracker (optional -- tracks LLM API costs)
    cost_tracker = None
    if getattr(config, "cost_tracking_enabled", False):
        try:
            from jarvis.telemetry.cost_tracker import CostTracker

            cost_db = str(config.db_path.with_name("memory_costs.db"))
            cost_tracker = CostTracker(
                db_path=cost_db,
                daily_budget=getattr(config, "daily_budget_usd", 0.0),
                monthly_budget=getattr(config, "monthly_budget_usd", 0.0),
            )
            log.info("cost_tracker_initialized", db=cost_db)
        except Exception:
            log.debug("cost_tracker_init_skipped", exc_info=True)
    result["cost_tracker"] = cost_tracker

    # Notification/Reminder tools
    notification_tools = None
    try:
        from jarvis.mcp.notification_tools import (
            register_notification_tools,
            restore_pending_reminders,
        )

        notification_tools = register_notification_tools(mcp_client, config)
        log.info("notification_tools_registered")
        # Restore pending reminders from DB (fire overdue, reschedule future)
        try:
            await restore_pending_reminders(notification_tools)
        except Exception:
            log.debug("notification_tools_restore_skipped", exc_info=True)
    except Exception:
        log.debug("notification_tools_not_registered", exc_info=True)

    # Desktop tools (clipboard, screenshot) — guarded by config.tools.desktop_tools_enabled
    if getattr(getattr(config, "tools", None), "desktop_tools_enabled", False):
        try:
            from jarvis.mcp.desktop_tools import register_desktop_tools

            register_desktop_tools(mcp_client, config)
            log.info("desktop_tools_registered")
        except Exception:
            log.debug("desktop_tools_not_registered", exc_info=True)
    else:
        log.info("desktop_tools_disabled_by_config")

    # Computer Use (screenshot + coordinate clicking) — guarded by config.tools.computer_use_enabled
    if getattr(getattr(config, "tools", None), "computer_use_enabled", False):
        try:
            from jarvis.mcp.computer_use import register_computer_use_tools

            vision = getattr(gateway, "_vision_analyzer", None) if gateway else None
            cu_tools = register_computer_use_tools(mcp_client, vision_analyzer=vision)
            if cu_tools:
                log.info("computer_use_tools_registered")
        except Exception:
            log.debug("computer_use_not_registered", exc_info=True)
    else:
        log.info("computer_use_disabled_by_config")

    # Background task tools (long-running shell commands)
    bg_manager = None
    try:
        from jarvis.mcp.background_tasks import register_background_tools

        _audit = getattr(gateway, "_audit_logger", None) if gateway else None
        bg_manager = register_background_tools(mcp_client, config, audit_logger=_audit)
        log.info("background_tools_registered")
    except Exception:
        log.debug("background_tools_not_registered", exc_info=True)
    result["bg_manager"] = bg_manager

    # Verified Web Lookup (multi-agent fact verification)
    try:
        from jarvis.mcp.verified_lookup import register_verified_lookup_tools

        verified_lookup = register_verified_lookup_tools(mcp_client, config)

        # Dependency Injection: WebTools
        if web_tools is not None:
            verified_lookup._set_web_tools(web_tools)

        # Dependency Injection: BrowserTool
        # browser_agent ist das v17-BrowserUse-Objekt; fuer verified_lookup
        # brauchen wir das v14-BrowserTool (navigate + extract_text)
        if browser_agent is not None and hasattr(browser_agent, "navigate"):
            # Prefer v17 BrowserAgent for richer extraction (tables, forms, JS)
            if hasattr(browser_agent, "extract_text"):
                verified_lookup._set_browser_agent(browser_agent)
                log.debug("verified_lookup_browser_v17_injected")
            else:
                verified_lookup._set_browser_tool(browser_agent)
        else:
            try:
                from jarvis.mcp.browser import BrowserTool

                _browser_for_lookup = BrowserTool(config)
                verified_lookup._set_browser_tool(_browser_for_lookup)
            except Exception:
                log.debug("verified_lookup_browser_skipped", exc_info=True)

        # Dependency Injection: LLM (gleicher Pattern wie Synthesizer)
        if synthesizer is not None and hasattr(synthesizer, "_llm_fn") and synthesizer._llm_fn:
            verified_lookup._set_llm_fn(synthesizer._llm_fn, synthesizer._llm_model)
        else:
            try:
                from jarvis.core.unified_llm import UnifiedLLMClient

                _llm_vl = UnifiedLLMClient.create(config)
                _model_vl = getattr(getattr(config, "models", None), "planner", None)
                _model_vl = getattr(_model_vl, "name", "") if _model_vl else ""

                async def _llm_for_verified(prompt: str, model: str = "") -> str:
                    resp = await _llm_vl.chat(
                        messages=[{"role": "user", "content": prompt}],
                        model=model or _model_vl,
                    )
                    return resp.get("content", "") if isinstance(resp, dict) else str(resp)

                verified_lookup._set_llm_fn(_llm_for_verified, _model_vl)
            except Exception:
                log.debug("verified_lookup_llm_skipped", exc_info=True)

        log.info("verified_lookup_registered")
    except Exception:
        log.debug("verified_lookup_not_registered", exc_info=True)

    # Deep Research v2 (Perplexity-style iterative search)
    try:
        from jarvis.mcp.deep_research_v2 import register_deep_research_v2
        from jarvis.mcp.web import WebTools

        _web_tools = None
        for tool_name, handler in result.items():
            if hasattr(handler, "__self__") and isinstance(handler.__self__, WebTools):
                _web_tools = handler.__self__
                break

        if _web_tools is None:
            # Fallback: create WebTools instance
            _web_tools = WebTools(config)

        _llm = getattr(gateway, "_llm", None) if gateway else None
        _model = getattr(config.models.planner, "name", "") if config else ""

        if register_deep_research_v2(mcp_client, _web_tools, _llm, _model):
            log.info("deep_research_v2_registered")
    except Exception:
        log.debug("deep_research_v2_not_registered", exc_info=True)

    return result
