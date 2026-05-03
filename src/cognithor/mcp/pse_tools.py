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
        held_out (optional list, same shape as examples): held-out
                  validation pairs the verifier MUST pass before a
                  program is returned. Anti-overfit gate. If absent
                  AND ``auto_held_out=True`` AND ≥3 examples given,
                  the LAST example is auto-promoted into ``held_out``.
        auto_held_out (optional bool, default True): smart-split toggle.
        budget   (optional dict):
            max_depth: int            (default 4)
            max_candidates: int       (default 50_000)
            wall_clock_seconds: float (default 30.0)
            cache_lookup: bool        (default True)
            auto_escalate: bool       (default False) — on
                ``BUDGET_EXCEEDED`` retry once with depth+1 + 2× the
                candidate cap, capped by remaining wall_clock_seconds.

    Returns a structured dict::

        {
            "status": "success" | "partial" | "no_solution" | ...,
            "program": <serialised program> or null,
            "score": float,
            "confidence": float,
            "cost_seconds": float,
            "cost_candidates": int,
            "cache_hit": bool,
            "held_out_examples": int,           # how many anti-overfit
                                                #  pairs were used
            "escalations": int,                 # 0 = first try worked
        }
    """
    examples_raw = kwargs.get("examples")
    if examples_raw is None:
        return {"error": "'examples' is required"}
    try:
        examples = _examples_from_json(examples_raw)
    except ValueError as exc:
        return {"error": str(exc)}

    # Sprint-22 A.4 — held_out wiring against demo overfit. The smoke
    # showed an example where the engine accepted ``rotate180`` for a
    # color-swap demo because the demos were rotation-symmetric and
    # there was no held-out validation to refute the spurious match.
    held_out_raw = kwargs.get("held_out")
    held_out: tuple[Any, ...] = ()
    if held_out_raw is not None:
        try:
            held_out = _examples_from_json(held_out_raw)
        except ValueError as exc:
            return {"error": f"held_out: {exc}"}
    elif bool(kwargs.get("auto_held_out", True)) and len(examples) >= 3:
        # Auto-promote the LAST example as the held-out pair so callers
        # who pass ≥3 demos get anti-overfit for free. Below 3 we keep
        # all examples in the demo set — the engine needs at least 2 to
        # constrain the search.
        held_out = (examples[-1],)
        examples = examples[:-1]

    budget_kwargs = kwargs.get("budget") or {}
    if not isinstance(budget_kwargs, dict):
        return {"error": "'budget' must be a dict"}

    try:
        from cognithor.channels.program_synthesis.core.types import (
            Budget,
            SynthesisStatus,
            TaskSpec,
        )
        from cognithor.channels.program_synthesis.integration.pge_adapter import (
            SynthesisRequest,
        )
    except ImportError as exc:
        return {"error": f"PSE module not available ({exc})"}

    spec = TaskSpec(examples=examples, held_out=held_out)
    try:
        max_depth = int(budget_kwargs.get("max_depth", 4))
        max_candidates = int(budget_kwargs.get("max_candidates", 50_000))
        wall_clock_seconds = float(budget_kwargs.get("wall_clock_seconds", 30.0))
        cache_lookup = bool(budget_kwargs.get("cache_lookup", True))
        auto_escalate = bool(budget_kwargs.get("auto_escalate", False))
    except (TypeError, ValueError) as exc:
        return {"error": f"invalid budget: {exc}"}

    channel = _get_channel()

    import asyncio
    import time

    loop = asyncio.get_running_loop()

    # Sprint-22 A.5 — adaptive budget. On BUDGET_EXCEEDED retry once
    # with depth+1 + 2× candidate cap, gated by remaining wall-clock.
    # Two retries max so a runaway task can't burn forever.
    MAX_ESCALATIONS = 2
    escalations = 0
    total_started = time.monotonic()
    total_cost_seconds = 0.0
    total_candidates = 0
    cache_hit_first = False
    last_result: Any = None

    while True:
        remaining = max(0.0, wall_clock_seconds - (time.monotonic() - total_started))
        if remaining <= 0.0:
            # User-cap exhausted; surface the latest result if we have one.
            break
        attempt_budget = Budget(
            max_depth=max_depth,
            max_candidates=max_candidates,
            wall_clock_seconds=remaining,
            cache_lookup=cache_lookup if escalations == 0 else False,
        )
        log.info(
            "pse_synthesize.start",
            n_examples=len(examples),
            n_held_out=len(held_out),
            max_depth=max_depth,
            max_candidates=max_candidates,
            wall_clock_remaining=remaining,
            attempt=escalations + 1,
        )
        request = SynthesisRequest(spec=spec, budget=attempt_budget)
        last_result = await loop.run_in_executor(None, channel.synthesize, request)
        if escalations == 0:
            cache_hit_first = bool(last_result.cache_hit)
        total_cost_seconds += float(last_result.cost_seconds)
        total_candidates += int(last_result.cost_candidates)

        if not auto_escalate or last_result.status != SynthesisStatus.BUDGET_EXCEEDED:
            break
        if escalations >= MAX_ESCALATIONS:
            break
        escalations += 1
        max_depth += 1
        max_candidates *= 2

    if last_result is None:
        return {
            "status": "error",
            "error": "wall_clock_seconds exhausted before any attempt could run",
        }

    log.info(
        "pse_synthesize.done",
        status=last_result.status.value,
        cost_seconds=total_cost_seconds,
        cache_hit=cache_hit_first,
        escalations=escalations,
    )

    return {
        "status": last_result.status.value,
        "program": str(last_result.program) if last_result.program is not None else None,
        "score": float(last_result.score),
        "confidence": float(last_result.confidence),
        "cost_seconds": float(total_cost_seconds),
        "cost_candidates": int(total_candidates),
        "cache_hit": cache_hit_first,
        "held_out_examples": len(held_out),
        "escalations": int(escalations),
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
            "with NumPy fast-path + cache). Replayable, no LLM hallucination. "
            "Anti-overfit gate via held_out (auto-split when ≥3 examples). "
            "Optional auto-escalation on BUDGET_EXCEEDED."
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
                "held_out": {
                    "type": "array",
                    "description": (
                        "Optional anti-overfit validation pairs (same shape "
                        "as examples). When absent and ≥3 examples are given, "
                        "the last example is auto-promoted into held_out "
                        "(disable via auto_held_out=False)."
                    ),
                },
                "auto_held_out": {
                    "type": "boolean",
                    "description": "Default true — smart-split last example when held_out absent.",
                },
                "budget": {
                    "type": "object",
                    "description": (
                        "Optional compute budget: max_depth, max_candidates, "
                        "wall_clock_seconds, cache_lookup, auto_escalate."
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
