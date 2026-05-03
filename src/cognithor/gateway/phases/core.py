"""Core phase: LLM clients, model router, session store.

Attributes handled:
  _ollama, _llm, _model_router, _session_store
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.gateway.phases import PhaseResult

log = get_logger(__name__)


def declare_core_attrs(config: Any) -> PhaseResult:
    """Return default (None) values for core attributes."""
    return {
        "ollama": None,
        "llm": None,
        "model_router": None,
        "session_store": None,
    }


async def init_core(config: Any) -> PhaseResult:
    """Initialize core LLM subsystems and session store.

    Returns a PhaseResult plus a special ``__llm_ok`` flag used by later phases.
    """
    from cognithor.core.model_router import ModelRouter
    from cognithor.core.unified_llm import UnifiedLLMClient
    from cognithor.gateway.session_store import SessionStore

    result: PhaseResult = {}

    # Session Store (SQLite persistence)
    session_db = config.sessions_dir / "sessions.db"
    session_store = SessionStore(session_db)
    existing = session_store.count_sessions()
    log.info("session_store_ready", db=str(session_db), active_sessions=existing)
    result["session_store"] = session_store

    # Unified LLM Client
    llm = UnifiedLLMClient.create(config)
    ollama = llm._ollama  # Legacy access (kann None sein bei API-Modus)
    llm_ok = await llm.is_available()

    if not llm_ok:
        if llm.backend_type == "ollama":
            log.warning(
                "llm_not_available",
                backend=llm.backend_type,
                url=config.ollama.base_url,
                message="Jarvis startet trotzdem, aber LLM-Funktionen sind eingeschraenkt.",
            )
        elif llm.backend_type == "lmstudio":
            log.warning(
                "llm_not_available",
                backend=llm.backend_type,
                url=config.lmstudio_base_url,
                message=(
                    "LM Studio nicht erreichbar. Laeuft der Server auf dem konfigurierten Port?"
                ),
            )
        else:
            log.warning(
                "llm_not_available",
                backend=llm.backend_type,
                message=f"LLM-Backend '{llm.backend_type}' nicht erreichbar. "
                f"Jarvis startet trotzdem, aber LLM-Funktionen sind eingeschraenkt.",
            )
    else:
        log.info("llm_backend_ready", backend=llm.backend_type)

    result["llm"] = llm
    result["ollama"] = ollama  # kann None sein

    # Model Router
    if llm._backend is not None:
        model_router = ModelRouter.from_backend(config, llm._backend)
    else:
        model_router = ModelRouter(config, llm._ollama)

    if llm_ok:
        await model_router.initialize()

    result["model_router"] = model_router

    # Extra metadata (not applied as attributes, consumed by gateway)
    result["__llm_ok"] = llm_ok

    return result
