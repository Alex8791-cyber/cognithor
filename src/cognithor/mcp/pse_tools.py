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
    from collections.abc import Awaitable, Callable

    from cognithor.channels.program_synthesis.integration.pge_adapter import (
        ProgramSynthesisChannel,
    )

log = get_logger(__name__)

__all__ = [
    "handle_pse_is_synthesizable",
    "handle_pse_status",
    "handle_pse_synthesize",
    "handle_pse_synthesize_refined",
    "register_pse_tools",
]


# ---------------------------------------------------------------------------
# Sprint-25 — Planner-Refinement-Loop
#
# The new ``pse_synthesize_refined`` tool wraps :func:`handle_pse_synthesize`
# with a single LLM-refinement pass. It needs an ``llm_fn`` that the gateway
# passes in at registration time. We store it in a module-level holder so
# ``handle_pse_synthesize_refined`` (which the MCP client calls without
# context) can pick it up. ``None`` means the gateway didn't have an LLM
# wired yet — refinement degrades to a neutral-verdict no-op.
# ---------------------------------------------------------------------------

_refinement_llm_fn: Callable[[str], Awaitable[str]] | None = None


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

    Sprint-22: the cache is now JSONL-persistent at
    ``~/.cognithor/pse_cache.jsonl`` so cross-session synthesise calls
    benefit from prior runs. Persistence failures are non-fatal — the
    channel falls back to the in-memory cache.
    """
    global _channel_singleton
    if _channel_singleton is None:
        from cognithor.channels.program_synthesis.integration.pge_adapter import (
            ProgramSynthesisChannel,
        )
        from cognithor.channels.program_synthesis.integration.tactical_memory import (
            PSECache,
        )

        cache: PSECache | None = None
        try:
            from pathlib import Path

            cache_path = Path.home() / ".cognithor" / "pse_cache.jsonl"
            cache = PSECache(persistence_path=str(cache_path))
            log.info(
                "pse.cache_persistent",
                path=str(cache_path),
                entries=len(cache),
            )
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("pse.cache_persistence_disabled", error=str(exc))

        _channel_singleton = ProgramSynthesisChannel(
            actor="mcp_tool@cognithor",
            cache=cache,
        )
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


def _coerce_value(raw: Any) -> Any:
    """Sprint-22 — generic Input-Typing boundary.

    Accepts the shapes any of Cognithor's DSL families understand:

    * ``list[list[int]]`` → ``np.ndarray`` (grid family — ARC-DSL)
    * ``str`` → ``str`` (string family — FlashFill-style)
    * ``int`` → ``int`` (number family — arithmetic + bridge ops)
    * ``list[str]`` → ``list[str]`` (string-list family — split / join)
    * ``list[int]`` → ``list[int]`` (int-list family — sum / max / sort)

    Booleans are explicitly rejected: Python's ``bool`` is a subclass
    of ``int`` so an unguarded ``isinstance(raw, int)`` check would
    accept ``True`` / ``False`` and the search engine would happily
    type-check them as ``Int``, which is never what the caller meant.

    Empty lists are rejected per-example because the type tag (str-
    list vs int-list) cannot be inferred — the homogeneity check
    would then have to guess. Callers should send at least one
    populated demo so the type is unambiguous.

    Non-fitting payloads raise :class:`ValueError` so the MCP layer
    returns a structured error rather than crashing the engine.
    """
    if isinstance(raw, str):
        return raw
    if isinstance(raw, bool):
        raise ValueError(
            f"unsupported input type {type(raw).__name__} — "
            "expected 2-D int list (grid), str, int, list[str], or list[int]"
        )
    if isinstance(raw, int):
        return raw
    if isinstance(raw, list):
        if not raw:
            raise ValueError(
                "empty list rejected — type (StringList vs IntList) "
                "cannot be inferred from an empty payload"
            )
        head = raw[0]
        if isinstance(head, list):
            return _coerce_grid(raw)
        if isinstance(head, str):
            if not all(isinstance(v, str) for v in raw):
                raise ValueError("string list must contain only str elements")
            return raw
        if isinstance(head, int) and not isinstance(head, bool):
            if not all(isinstance(v, int) and not isinstance(v, bool) for v in raw):
                raise ValueError("int list must contain only non-bool int elements")
            return raw
    raise ValueError(
        f"unsupported input type {type(raw).__name__} — "
        "expected 2-D int list (grid), str, int, list[str], or list[int]"
    )


def _examples_from_json(payload: Any) -> tuple[tuple[Any, Any], ...]:
    """Parse a JSON ``examples`` list into the channel's tuple-of-Examples.

    Accepts ``[{"input": <value>, "output": <value>}, ...]`` where each
    ``<value>`` is either a 2-D int list (grid) or a string. Empty
    lists, missing keys, or unrepresentable types raise
    :class:`ValueError` so the MCP wrapper turns them into structured
    error responses.

    Sprint-22: a single ``examples`` payload is required to be
    homogeneous — all inputs the same family, all outputs the same
    family. Mixed payloads are rejected up-front so the search engine
    sees a coherent type-tagged signature.
    """
    if not isinstance(payload, list):
        raise ValueError("'examples' must be a list")
    if len(payload) < 2:
        raise ValueError(
            "'examples' needs at least 2 entries (single-demo tasks are too under-specified)"
        )
    parsed: list[tuple[Any, Any]] = []
    families: set[str] = set()
    for i, ex in enumerate(payload):
        if not isinstance(ex, dict):
            raise ValueError(f"example {i} must be a dict")
        if "input" not in ex or "output" not in ex:
            raise ValueError(f"example {i} needs both 'input' and 'output'")
        try:
            inp = _coerce_value(ex["input"])
            out = _coerce_value(ex["output"])
        except ValueError as exc:
            raise ValueError(f"example {i}: {exc}") from exc
        families.add(type(inp).__name__)
        families.add(type(out).__name__)
        parsed.append((inp, out))
    # Heterogeneous families would always lose at the type-filter; the
    # explicit error makes the diagnostic obvious instead of silently
    # returning ``no_solution``. Sprint-22: ``str`` / ``int`` / ``list``
    # mix freely because the families ship explicit bridge primitives
    # (``int_to_string`` / ``string_to_int`` / ``string_length`` /
    # ``string_list_length`` / ``int_list_sum``); only the Grid
    # family is structurally disjoint from the text-shaped families.
    if "ndarray" in families and len(families - {"ndarray"}) > 0:
        raise ValueError(
            "'examples' must not mix grids with text-shaped values (strings, ints, or lists)"
        )
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
# Sprint-25 — pse_synthesize_refined
# ---------------------------------------------------------------------------


async def handle_pse_synthesize_refined(**kwargs: Any) -> dict[str, Any]:
    """Synthesise a program AND run a single LLM-refinement pass over it.

    Same input shape as :func:`handle_pse_synthesize` plus:

      * ``task_description`` (str): natural-language description the LLM
        uses to critique the synthesised program. When empty, the
        refinement still runs but the prompt explicitly notes the
        absence — the LLM's verdict is naturally less confident.

    Returns the same structured shape as ``pse_synthesize`` plus a
    ``refinement`` field carrying the LLM verdict::

        {
            ...same as pse_synthesize...,
            "refinement": {
                "verdict": "accept" | "caution" | "reject" | "neutral",
                "explanation": str,
                "confidence": "high" | "medium" | "low",
            } | None,
        }

    ``refinement`` is ``None`` when synthesis itself didn't return a
    program (no successful program → nothing to refine). Otherwise the
    refinement always runs; if no ``llm_fn`` is wired in, it returns a
    neutral verdict with reason ``"no llm_fn wired (refinement
    disabled)"``.

    Failure-tolerant: a refinement parse error or an exception inside
    ``llm_fn`` collapses to a neutral verdict — the synthesis result
    itself is **never** discarded by a refinement failure.
    """
    task_description = str(kwargs.pop("task_description", "") or "")

    synthesis = await handle_pse_synthesize(**kwargs)

    # Bail early if synthesis itself errored or returned nothing to refine.
    program = synthesis.get("program")
    if not isinstance(program, str) or not program.strip():
        synthesis["refinement"] = None
        return synthesis

    try:
        from cognithor.core.pse_refinement import refine_pse_program
    except ImportError as exc:
        log.warning("pse_refinement_unavailable", error=str(exc))
        synthesis["refinement"] = None
        return synthesis

    n_examples = (
        len(kwargs.get("examples") or [])  # post-pop: examples is still in kwargs
    )
    # ``handle_pse_synthesize`` may auto-promote one of the examples into
    # held_out. Use the canonical counts from the synthesis result so the
    # refinement prompt reflects what the engine actually saw.
    n_held_out = int(synthesis.get("held_out_examples", 0) or 0)
    n_examples = max(0, n_examples - n_held_out) if n_held_out else n_examples

    score = float(synthesis.get("score", 0.0) or 0.0)

    verdict = await refine_pse_program(
        program=program,
        task_description=task_description,
        n_examples=n_examples,
        n_held_out=n_held_out,
        score=score,
        llm_fn=_refinement_llm_fn,
    )
    log.info(
        "pse_synthesize_refined.done",
        synthesis_status=synthesis.get("status"),
        verdict=verdict.verdict,
        confidence=verdict.confidence,
    )
    synthesis["refinement"] = verdict.to_dict()
    return synthesis


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_pse_tools(
    mcp_client: Any,
    *,
    llm_fn: Callable[[str], Awaitable[str]] | None = None,
) -> None:
    """Register all PSE tools with the MCP client.

    Mirrors :func:`cognithor.mcp.arc_tools.register_arc_tools` shape so the
    gateway tool-phase wiring is uniform.

    Args:
        mcp_client: The MCP client whose ``register_tool`` method takes
            ``(name, description, handler, schema)``.
        llm_fn: Optional async callable ``str -> str`` used by the
            Sprint-25 refinement tool. When None, ``pse_synthesize_refined``
            still registers but the refinement step degrades to a neutral
            verdict (synthesis result stays intact).
    """
    global _refinement_llm_fn
    _refinement_llm_fn = llm_fn
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
    register(
        name="pse_synthesize_refined",
        description=(
            "Sprint-25: Hybrid-Pipeline (LLM → Synthese → LLM-Refinement). "
            "Synthesises a program from input/output examples (same engine "
            "as pse_synthesize), then runs ONE LLM-refinement pass over the "
            "result and returns a structured verdict (accept | caution | "
            "reject) with explanation + confidence. Useful when a downstream "
            "consumer wants a sanity-check that the synthesised program "
            "matches the natural-language task — not just the demos."
        ),
        handler=handle_pse_synthesize_refined,
        schema={
            "type": "object",
            "properties": {
                "examples": {
                    "type": "array",
                    "description": "Same shape as pse_synthesize.examples.",
                },
                "held_out": {
                    "type": "array",
                    "description": "Same as pse_synthesize.held_out.",
                },
                "auto_held_out": {
                    "type": "boolean",
                    "description": "Same as pse_synthesize.auto_held_out.",
                },
                "budget": {
                    "type": "object",
                    "description": "Same as pse_synthesize.budget.",
                },
                "task_description": {
                    "type": "string",
                    "description": (
                        "Natural-language description of what the program "
                        "should do. Used by the LLM-refinement pass to "
                        "judge whether the synthesised program is "
                        "semantically right (not just demo-matching)."
                    ),
                },
            },
            "required": ["examples"],
        },
    )
    log.info(
        "pse_tools_registered",
        tools=[
            "pse_is_synthesizable",
            "pse_status",
            "pse_synthesize",
            "pse_synthesize_refined",
        ],
        refinement_llm_wired=llm_fn is not None,
    )
