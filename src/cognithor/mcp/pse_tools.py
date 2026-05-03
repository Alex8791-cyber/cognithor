"""MCP Tools for the Cognithor Program Synthesis Engine (PSE).

Sprint-22 Track A: exposes the previously-isolated PSE channel as a
generic MCP tool so the Planner / Gatekeeper / Executor / any other
channel can route demo-based "find the program that maps these inputs
to these outputs" queries at the PSE engine instead of asking the LLM
to free-form-guess.

Tools:
  - ``pse_synthesize``   : given examples (input → output pairs), return a
                           synthesized program + verifier trace.
  - ``pse_is_synthesizable``: classifier — returns whether the task is
                           PSE-routable (cheap, no engine boot).
  - ``pse_status``       : returns the loaded engine's metadata
                           (DSL_VERSION, PSE_VERSION, primitive count).

The tools wrap :class:`ProgramSynthesisChannel` with one shared instance
per process — the engine + cache is heavy to construct so reuse pays off
across calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cognithor.utils.logging import get_logger

if TYPE_CHECKING:
    from cognithor.channels.program_synthesis.integration.pge_adapter import (
        ProgramSynthesisChannel,
    )

log = get_logger(__name__)

__all__ = [
    "handle_pse_is_synthesizable",
    "handle_pse_status",
    "handle_pse_synthesize",
    "register_pse_tools",
]


# ---------------------------------------------------------------------------
# Lazy-init singleton
# ---------------------------------------------------------------------------

_channel_singleton: ProgramSynthesisChannel | None = None


def _get_channel() -> ProgramSynthesisChannel:
    """Return the process-wide PSE channel; construct on first call.

    The channel ships its own cache + sandbox + engine so we want exactly
    one instance per process (cache persistence + warm sandbox subprocess
    pool). Callers MUST treat the returned channel as read-only — concurrent
    callers serialise on the channel's internal locks.
    """
    global _channel_singleton
    if _channel_singleton is None:
        from cognithor.channels.program_synthesis.integration.pge_adapter import (
            ProgramSynthesisChannel,
        )

        _channel_singleton = ProgramSynthesisChannel(actor="mcp_tool@cognithor")
        log.info("pse.channel_initialised")
    return _channel_singleton


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------


def _coerce_grid(raw: Any) -> Any:
    """Best-effort convert a JSON list-of-lists-of-ints into an int8 grid.

    Returns the numpy array on success; raises :class:`ValueError` with
    a structured diagnostic message on schema mismatch so the MCP caller
    gets a useful error instead of a numpy traceback.
    """
    import numpy as np

    if not isinstance(raw, list) or not raw:
        raise ValueError("grid must be a non-empty list")
    if not all(isinstance(row, list) and row for row in raw):
        raise ValueError("grid must be a list of non-empty rows")
    width = len(raw[0])
    if not all(len(row) == width for row in raw):
        raise ValueError("grid rows must have uniform width")
    if not all(isinstance(v, int) for row in raw for v in row):
        raise ValueError("grid cells must be ints")
    return np.array(raw, dtype=np.int8)


def _examples_from_json(payload: Any) -> tuple[tuple[Any, Any], ...]:
    """Parse a JSON ``examples`` list into the channel's tuple-of-Examples.

    Accepts ``[{"input": [[...]], "output": [[...]]}, ...]``.  Empty
    lists, missing keys, or mis-shaped grids raise :class:`ValueError`
    so the MCP wrapper turns them into a structured error response.
    """
    if not isinstance(payload, list):
        raise ValueError("'examples' must be a list")
    if len(payload) < 2:
        raise ValueError(
            "'examples' needs at least 2 entries (single-demo tasks are too under-specified)"
        )
    parsed: list[tuple[Any, Any]] = []
    for i, ex in enumerate(payload):
        if not isinstance(ex, dict):
            raise ValueError(f"example {i} must be a dict")
        if "input" not in ex or "output" not in ex:
            raise ValueError(f"example {i} needs both 'input' and 'output'")
        try:
            inp = _coerce_grid(ex["input"])
            out = _coerce_grid(ex["output"])
        except ValueError as exc:
            raise ValueError(f"example {i}: {exc}") from exc
        parsed.append((inp, out))
    return tuple(parsed)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def handle_pse_is_synthesizable(**kwargs: Any) -> str:
    """Cheap classifier: return whether the task is PSE-routable.

    Pure-function call — no engine boot. Useful for the Planner to
    pre-filter before paying the synthesise call cost.
    """
    examples = kwargs.get("examples")
    if examples is None:
        return "Error: 'examples' is required."
    try:
        from cognithor.channels.program_synthesis.integration.pge_adapter import (
            is_synthesizable,
        )
    except ImportError as exc:
        return f"Error: PSE module not available ({exc})."
    routable = is_synthesizable({"examples": examples})
    return "yes" if routable else "no"


async def handle_pse_status(**kwargs: Any) -> str:
    """Report PSE engine metadata for diagnostics."""
    del kwargs  # no inputs
    try:
        from cognithor.channels.program_synthesis import DSL_VERSION, PSE_VERSION
    except ImportError as exc:
        return f"Error: PSE module not available ({exc})."

    primitive_count = "unknown"
    try:
        from cognithor.channels.program_synthesis.dsl import primitives as _prims

        # The exact attribute varies by DSL family; enumerate ALL_PRIMITIVES
        # if exposed, otherwise fall through with "unknown" so the tool
        # stays diagnostically useful even if the inventory shape changes.
        all_prims = getattr(_prims, "ALL_PRIMITIVES", None)
        if all_prims is not None:
            primitive_count = str(len(all_prims))
    except Exception:  # pragma: no cover — defensive
        pass

    return f"PSE_VERSION={PSE_VERSION}, DSL_VERSION={DSL_VERSION}, primitives={primitive_count}"


async def handle_pse_synthesize(**kwargs: Any) -> dict[str, Any]:
    """Synthesise a program from input/output examples.

    Inputs::

        examples: list[{"input": <2D int list>, "output": <2D int list>}]
        budget   (optional dict):
            max_depth: int            (default 4)
            max_candidates: int       (default 50_000)
            wall_clock_seconds: float (default 30.0)
            cache_lookup: bool        (default True)

    Returns a structured dict::

        {
            "status": "success" | "partial" | "no_solution" | ...,
            "program": <serialised program> or null,
            "score": float,
            "confidence": float,
            "cost_seconds": float,
            "cost_candidates": int,
            "cache_hit": bool,
        }
    """
    examples_raw = kwargs.get("examples")
    if examples_raw is None:
        return {"error": "'examples' is required"}
    try:
        examples = _examples_from_json(examples_raw)
    except ValueError as exc:
        return {"error": str(exc)}

    budget_kwargs = kwargs.get("budget") or {}
    if not isinstance(budget_kwargs, dict):
        return {"error": "'budget' must be a dict"}

    try:
        from cognithor.channels.program_synthesis.core.types import (
            Budget,
            TaskSpec,
        )
        from cognithor.channels.program_synthesis.integration.pge_adapter import (
            SynthesisRequest,
        )
    except ImportError as exc:
        return {"error": f"PSE module not available ({exc})"}

    spec = TaskSpec(examples=examples)
    try:
        budget = Budget(
            max_depth=int(budget_kwargs.get("max_depth", 4)),
            max_candidates=int(budget_kwargs.get("max_candidates", 50_000)),
            wall_clock_seconds=float(budget_kwargs.get("wall_clock_seconds", 30.0)),
            cache_lookup=bool(budget_kwargs.get("cache_lookup", True)),
        )
    except (TypeError, ValueError) as exc:
        return {"error": f"invalid budget: {exc}"}

    log.info(
        "pse_synthesize.start",
        n_examples=len(examples),
        max_depth=budget.max_depth,
        wall_clock_seconds=budget.wall_clock_seconds,
    )

    channel = _get_channel()
    request = SynthesisRequest(spec=spec, budget=budget)

    import asyncio

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, channel.synthesize, request)

    log.info(
        "pse_synthesize.done",
        status=result.status.value,
        cost_seconds=result.cost_seconds,
        cache_hit=result.cache_hit,
    )

    return {
        "status": result.status.value,
        "program": str(result.program) if result.program is not None else None,
        "score": float(result.score),
        "confidence": float(result.confidence),
        "cost_seconds": float(result.cost_seconds),
        "cost_candidates": int(result.cost_candidates),
        "cache_hit": bool(result.cache_hit),
    }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_pse_tools(mcp_client: Any) -> None:
    """Register all three PSE tools with the MCP client.

    Mirrors :func:`cognithor.mcp.arc_tools.register_arc_tools` shape so the
    gateway tool-phase wiring is uniform.
    """
    register = getattr(mcp_client, "register_tool", None)
    if not callable(register):
        log.warning("pse_tools.mcp_client_missing_register_tool")
        return

    register(
        name="pse_is_synthesizable",
        description=(
            "Classifier: returns 'yes' / 'no' for whether the given "
            "examples are routable to the Cognithor Program Synthesis Engine. "
            "Cheap — no engine boot. Use before pse_synthesize."
        ),
        handler=handle_pse_is_synthesizable,
        schema={
            "type": "object",
            "properties": {
                "examples": {
                    "type": "array",
                    "description": "List of {input, output} demo pairs",
                },
            },
            "required": ["examples"],
        },
    )
    register(
        name="pse_status",
        description="Return PSE engine version + primitive count.",
        handler=handle_pse_status,
        schema={"type": "object", "properties": {}},
    )
    register(
        name="pse_synthesize",
        description=(
            "Synthesize a deterministic program from input/output examples "
            "using the Cognithor PSE (enumerative search over a typed DSL "
            "with NumPy fast-path + cache). Replayable, no LLM hallucination."
        ),
        handler=handle_pse_synthesize,
        schema={
            "type": "object",
            "properties": {
                "examples": {
                    "type": "array",
                    "description": (
                        "List of {input, output} demo pairs where each "
                        "input / output is a 2-D list of ints (rows × cols)."
                    ),
                },
                "budget": {
                    "type": "object",
                    "description": (
                        "Optional compute budget: max_depth, max_candidates, "
                        "wall_clock_seconds, cache_lookup."
                    ),
                },
            },
            "required": ["examples"],
        },
    )
    log.info(
        "pse_tools_registered",
        tools=["pse_is_synthesizable", "pse_status", "pse_synthesize"],
    )
