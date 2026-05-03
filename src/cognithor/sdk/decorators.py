"""SDK decorators — @tool, @agent, @hook for declarative registration.

Usage::

    @tool(name="add", description="Add two numbers")
    async def add(a: int, b: int) -> int:
        return a + b

    @agent(name="calculator", tools=["add"])
    class CalcAgent:
        async def on_message(self, msg: str) -> str:
            return "Calculated!"

    @hook("on_error")
    async def log_error(error: Exception) -> None:
        print(f"Error: {error}")
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from cognithor.sdk.definitions import AgentDefinition, HookDefinition, HookEvent, ToolDefinition
from cognithor.sdk.registry import SDKRegistry

if TYPE_CHECKING:
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

_registry = SDKRegistry()


def get_registry() -> SDKRegistry:
    """Get the global SDK registry."""
    return _registry


# ---------------------------------------------------------------------------
# @tool decorator
# ---------------------------------------------------------------------------


def tool(
    name: str | None = None,
    description: str = "",
    *,
    risk_level: str = "green",
    requires_network: bool = False,
    idempotent: bool = False,
    read_only: bool = False,
    version: str = "0.1.0",
) -> Callable[..., Any]:
    """Register a function as an SDK tool.

    Can be used with or without arguments::

        @tool(name="greet", description="Greet someone")
        async def greet(name: str) -> str:
            return f"Hello, {name}!"

        @tool
        async def simple_tool() -> str:
            return "done"
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or func.__name__
        schema = _infer_schema(func)

        defn = ToolDefinition(
            name=tool_name,
            description=description or func.__doc__ or "",
            handler=func,
            input_schema=schema,
            risk_level=risk_level,
            requires_network=requires_network,
            idempotent=idempotent,
            read_only=read_only,
            version=version,
        )
        _registry.register_tool(defn)
        func._sdk_tool = defn  # type: ignore[attr-defined]  # Attach metadata
        return func

    # Handle @tool without parentheses
    if callable(name):
        func = name
        name = None
        return decorator(func)

    return decorator


# ---------------------------------------------------------------------------
# @agent decorator
# ---------------------------------------------------------------------------


def agent(
    name: str | None = None,
    description: str = "",
    *,
    tools: list[str] | None = None,
    system_prompt: str = "",
    trigger_keywords: list[str] | None = None,
    can_delegate_to: list[str] | None = None,
    max_iterations: int = 5,
    timeout_seconds: int = 300,
    version: str = "0.1.0",
) -> Callable[..., Any]:
    """Register a class as an SDK agent.

    Example::

        @agent(name="researcher", tools=["web_search"])
        class ResearchAgent:
            async def on_message(self, message: str) -> str:
                return "Researching..."
    """

    def decorator(cls: type) -> type:
        agent_name = name or cls.__name__.lower()
        defn = AgentDefinition(
            name=agent_name,
            description=description or cls.__doc__ or "",
            version=version,
            tools=tools or [],
            system_prompt=system_prompt,
            trigger_keywords=trigger_keywords or [],
            can_delegate_to=can_delegate_to or [],
            max_iterations=max_iterations,
            timeout_seconds=timeout_seconds,
            cls=cls,
        )
        _registry.register_agent(defn)
        cls._sdk_agent = defn  # type: ignore[attr-defined]  # Attach metadata
        return cls

    return decorator


# ---------------------------------------------------------------------------
# @hook decorator
# ---------------------------------------------------------------------------


def hook(
    event: str | HookEvent,
    *,
    priority: int = 0,
    description: str = "",
) -> Callable[..., Any]:
    """Register a function as a lifecycle hook.

    Example::

        @hook("on_error")
        async def handle_error(error: Exception) -> None:
            log.error(f"Agent error: {error}")
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        hook_event = HookEvent(event) if isinstance(event, str) else event
        defn = HookDefinition(
            event=hook_event,
            handler=func,
            priority=priority,
            description=description or func.__doc__ or "",
        )
        _registry.register_hook(defn)
        func._sdk_hook = defn  # type: ignore[attr-defined]
        return func

    return decorator


# ---------------------------------------------------------------------------
# Schema inference
# ---------------------------------------------------------------------------


def _infer_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """Infer JSON Schema from function signature type hints."""
    sig = inspect.signature(func)

    # Resolve annotations (handles PEP 563 stringified annotations)
    hints: dict[str, Any] = {}
    try:
        hints = {
            k: v for k, v in inspect.get_annotations(func, eval_str=True).items() if k != "return"
        }
    except Exception:
        # Fallback: try raw annotations
        raw = getattr(func, "__annotations__", {})
        str_type_map = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list[Any],
            "dict": dict[str, Any],
        }
        for k, v in raw.items():
            if k == "return":
                continue
            if isinstance(v, str):
                hints[k] = str_type_map.get(v, str)
            else:
                hints[k] = v

    properties: dict[str, Any] = {}
    required: list[str] = []

    type_map: dict[type, str] = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        hint = hints.get(param_name)
        json_type = type_map.get(hint, "string") if hint else "string"
        properties[param_name] = {"type": json_type}

        if param.default is inspect.Parameter.empty:
            required.append(param_name)
        else:
            properties[param_name]["default"] = param.default

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    return schema
