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
        "vault_tools": None,
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
    from jarvis.mcp.code_tools import register_code_tools
    from jarvis.mcp.filesystem import register_fs_tools
    from jarvis.mcp.memory_server import register_memory_tools
    from jarvis.mcp.shell import register_shell_tools
    from jarvis.mcp.web import register_web_tools

    result: PhaseResult = {"mcp_client": mcp_client}

    # Register built-in MCP tools
    register_fs_tools(mcp_client, config)
    register_shell_tools(mcp_client, config)
    web_tools = register_web_tools(mcp_client, config)
    register_code_tools(mcp_client, config)

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

    # LLM + Vault in MediaPipeline injizieren (für analyze_document)
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

    if media_pipeline is not None and vault_tools is not None and hasattr(media_pipeline, "_set_vault"):
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
                if media_pipeline is not None and hasattr(media_pipeline, "_llm_fn") and media_pipeline._llm_fn is not None:
                    # Reuse existing LLM function from media pipeline
                    synthesizer._set_llm_fn(media_pipeline._llm_fn, media_pipeline._llm_model)
                else:
                    from jarvis.core.unified_llm import UnifiedLLMClient
                    llm_client_synth = UnifiedLLMClient.create(config)
                    model_name_synth = getattr(getattr(config, "models", None), "planner", None)
                    model_name_synth = getattr(model_name_synth, "name", "") if model_name_synth else ""

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

    return result
