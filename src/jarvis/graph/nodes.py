"""Built-in Node Handlers -- Vorgefertigte Nodes für Graph Orchestrator v18.

Stellt wiederverwendbare Handler bereit:
  - llm_node:       LLM-Aufruf mit Prompt-Template
  - tool_node:      MCP-Tool-Aufruf
  - transform_node: State-Transformation (map/filter/merge)
  - condition_node: Bedingungsprüfung (Router)
  - delay_node:     Wartezeit
  - log_node:       Logging/Debug
  - accumulate_node: Sammelt Ergebnisse aus parallelen Branches
  - gate_node:      Prüft ob Bedingung erfüllt ist (Gating)
  - retry_wrapper:  Wrapper für Retry-Logic

Alle Handlers haben die Signatur: async (GraphState) -> GraphState
Router-Handlers geben: async (GraphState) -> str
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Awaitable

from jarvis.graph.types import GraphState
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


# ── LLM Node ────────────────────────────────────────────────────

def llm_node(
    prompt_template: str,
    *,
    input_key: str = "messages",
    output_key: str = "response",
    model: str = "",
    llm_handler: Callable | None = None,
) -> Callable[[GraphState], Awaitable[GraphState]]:
    """Erstellt einen LLM-Aufruf-Handler.

    Args:
        prompt_template: Template mit {variable}-Platzhaltern
        input_key: State-Key für Input
        output_key: State-Key für Output
        model: LLM-Modell (optional)
        llm_handler: Custom LLM-Aufruf-Funktion
    """
    async def handler(state: GraphState) -> GraphState:
        # Template rendern
        prompt = prompt_template
        for key in state.keys():
            placeholder = "{" + key + "}"
            if placeholder in prompt:
                prompt = prompt.replace(placeholder, str(state[key]))

        if llm_handler:
            response = await llm_handler(prompt, model=model)
        else:
            # Fallback: Prompt als Response (für Tests ohne LLM)
            response = f"[LLM would process: {prompt[:200]}]"

        state[output_key] = response
        return state

    return handler


# ── Tool Node ────────────────────────────────────────────────────

def tool_node(
    tool_name: str,
    *,
    params_key: str = "tool_params",
    result_key: str = "tool_result",
    tool_executor: Callable | None = None,
) -> Callable[[GraphState], Awaitable[GraphState]]:
    """Erstellt einen MCP-Tool-Aufruf-Handler.

    Args:
        tool_name: Name des Tools
        params_key: State-Key für Tool-Parameter
        result_key: State-Key für Tool-Ergebnis
        tool_executor: Custom Tool-Executor
    """
    async def handler(state: GraphState) -> GraphState:
        params = state.get(params_key, {})

        if tool_executor:
            result = await tool_executor(tool_name, params)
        else:
            result = {"tool": tool_name, "params": params, "status": "simulated"}

        state[result_key] = result
        return state

    return handler


# ── Transform Node ───────────────────────────────────────────────

def transform_node(
    transform_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> Callable[[GraphState], Awaitable[GraphState]]:
    """Erstellt einen State-Transformations-Handler.

    Args:
        transform_fn: Sync-Funktion die dict→dict transformiert
    """
    async def handler(state: GraphState) -> GraphState:
        data = state.to_dict()
        transformed = transform_fn(data)
        state.update(transformed)
        return state

    return handler


# ── Condition Node (Router) ──────────────────────────────────────

def condition_node(
    condition_fn: Callable[[dict[str, Any]], str],
) -> Callable[[GraphState], Awaitable[str]]:
    """Erstellt einen Bedingungs-Router.

    Args:
        condition_fn: Funktion die dict→str (Edge-Name) zurückgibt
    """
    async def handler(state: GraphState) -> str:
        data = state.to_dict()
        return condition_fn(data)

    return handler


def threshold_router(
    key: str,
    threshold: float,
    above: str = "above",
    below: str = "below",
) -> Callable[[GraphState], Awaitable[str]]:
    """Router der basierend auf einem Schwellenwert entscheidet."""
    async def handler(state: GraphState) -> str:
        value = state.get(key, 0)
        try:
            return above if float(value) >= threshold else below
        except (TypeError, ValueError):
            return below

    return handler


def key_router(
    key: str,
    mapping: dict[str, str] | None = None,
    *,
    default: str = "__default__",
) -> Callable[[GraphState], Awaitable[str]]:
    """Router der basierend auf einem State-Wert routet.

    Args:
        key: State-Key dessen Wert geprüft wird
        mapping: Optionales Mapping {value: edge_name}
        default: Fallback wenn kein Match
    """
    async def handler(state: GraphState) -> str:
        value = str(state.get(key, ""))
        if mapping:
            return mapping.get(value, default)
        return value or default

    return handler


# ── Delay Node ───────────────────────────────────────────────────

def delay_node(seconds: float) -> Callable[[GraphState], Awaitable[GraphState]]:
    """Wartet eine bestimmte Zeit."""
    async def handler(state: GraphState) -> GraphState:
        await asyncio.sleep(seconds)
        return state

    return handler


# ── Log Node ─────────────────────────────────────────────────────

def log_node(
    message: str = "",
    *,
    log_keys: list[str] | None = None,
    state_key: str = "__log__",
) -> Callable[[GraphState], Awaitable[GraphState]]:
    """Loggt State-Informationen."""
    async def handler(state: GraphState) -> GraphState:
        info: dict[str, Any] = {"timestamp": time.strftime("%H:%M:%S", time.gmtime())}
        if message:
            info["message"] = message
        if log_keys:
            for k in log_keys:
                info[k] = state.get(k, "N/A")
        else:
            info["state_keys"] = list(state.keys())

        # Log-History im State aufbauen
        history = state.get(state_key, [])
        if not isinstance(history, list):
            history = []
        history.append(info)
        state[state_key] = history

        log.debug("graph_log_node", **info)
        return state

    return handler


# ── Accumulate Node ──────────────────────────────────────────────

def accumulate_node(
    source_keys: list[str],
    target_key: str = "accumulated",
) -> Callable[[GraphState], Awaitable[GraphState]]:
    """Sammelt Werte aus mehreren State-Keys in einer Liste."""
    async def handler(state: GraphState) -> GraphState:
        accumulated = []
        for key in source_keys:
            value = state.get(key)
            if value is not None:
                accumulated.append({"key": key, "value": value})
        state[target_key] = accumulated
        return state

    return handler


# ── Gate Node ────────────────────────────────────────────────────

def gate_node(
    check_fn: Callable[[dict[str, Any]], bool],
    *,
    error_message: str = "Gate check failed",
) -> Callable[[GraphState], Awaitable[GraphState]]:
    """Prüft eine Bedingung und wirft Exception wenn nicht erfüllt."""
    async def handler(state: GraphState) -> GraphState:
        data = state.to_dict()
        if not check_fn(data):
            raise ValueError(error_message)
        return state

    return handler


# ── Counter Node ─────────────────────────────────────────────────

def counter_node(
    key: str = "iteration",
    increment: int = 1,
) -> Callable[[GraphState], Awaitable[GraphState]]:
    """Inkrementiert einen Zähler im State."""
    async def handler(state: GraphState) -> GraphState:
        current = state.get(key, 0)
        state[key] = current + increment
        return state

    return handler


# ── Set Value Node ───────────────────────────────────────────────

def set_value_node(**values: Any) -> Callable[[GraphState], Awaitable[GraphState]]:
    """Setzt fixe Werte im State."""
    async def handler(state: GraphState) -> GraphState:
        for k, v in values.items():
            state[k] = v
        return state

    return handler


# ── Merge Node ───────────────────────────────────────────────────

def merge_node(
    merge_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> Callable[[GraphState], Awaitable[GraphState]]:
    """Merge-Punkt für parallele Branches.

    Kann optional eine Merge-Funktion anwenden.
    """
    async def handler(state: GraphState) -> GraphState:
        if merge_fn:
            merged = merge_fn(state.to_dict())
            state.update(merged)
        return state

    return handler
