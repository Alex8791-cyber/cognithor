"""Tools phase: MCP client, browser agent, graph engine, telemetry, HITL, A2A.

Attributes handled:
  _mcp_client, _mcp_bridge, _browser_agent, _graph_engine,
  _telemetry_hub, _hitl_manager, _a2a_adapter
"""

from __future__ import annotations

from typing import Any

from jarvis.gateway.phases import PhaseResult
from jarvis.utils.logging import get_logger

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
    }


async def init_tools(
    config: Any,
    mcp_client: Any,
    memory_manager: Any,
    interop: Any = None,
    handle_message: Any = None,
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
    from jarvis.mcp.filesystem import register_fs_tools
    from jarvis.mcp.memory_server import register_memory_tools
    from jarvis.mcp.shell import register_shell_tools
    from jarvis.mcp.web import register_web_tools

    result: PhaseResult = {"mcp_client": mcp_client}

    # Register built-in MCP tools
    register_fs_tools(mcp_client, config)
    register_shell_tools(mcp_client, config)
    register_web_tools(mcp_client, config)

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

        browser_agent = register_browser_use_tools(
            mcp_client, vision_analyzer=vision_analyzer
        )
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
    try:
        from jarvis.mcp.media import register_media_tools
        register_media_tools(mcp_client)
    except Exception:
        log.warning("media_tools_not_registered")

    # Memory tools
    register_memory_tools(mcp_client, memory_manager)

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
        else:
            log.debug("a2a_protocol_disabled")
    except Exception as exc:
        log.debug("a2a_adapter_not_available", reason=str(exc))
        a2a_adapter = None
    result["a2a_adapter"] = a2a_adapter

    # CostTracker (optional — tracks LLM API costs)
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

    return result
