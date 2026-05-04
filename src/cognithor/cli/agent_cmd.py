"""CLI dispatcher for ``cognithor agent run --plan FILE.json --stream``.

Sprint-27 PR-B. Wires the streaming Producer (PR-A) into the
existing ``__main__.py`` argparse surface. Kept thin — the real
work lives in :mod:`cognithor.streaming.runner`. This module
exists so the argparse handler in ``__main__.py`` does not have
to import asyncio / streaming primitives directly.

Exit codes:

* ``0`` — run_complete emitted successfully
* ``1`` — run_error (Gatekeeper block, tool exception, plan-file
  validation failure, ...)
* ``2`` — bad command-line arguments
* ``130`` — KeyboardInterrupt (run_cancelled emitted)
"""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

from cognithor.streaming import EventEmitter, JsonlSink
from cognithor.streaming.runner import parse_plan_file, run_plan

if TYPE_CHECKING:
    from pathlib import Path


def cmd_run(
    *,
    plan_path: Path,
    stream: bool = True,
    out: Path | None = None,
) -> int:
    """Run the streaming agent CLI.

    Parameters
    ----------
    plan_path:
        Path to the JSON plan-file. See
        :func:`cognithor.streaming.runner.parse_plan_file` for the
        accepted shape.
    stream:
        Currently always required (for parity with the
        plan-doc spec that called the flag ``--stream``). When
        ``False``, the function returns 2 — callers must opt in
        explicitly.
    out:
        Optional output path. Defaults to ``sys.stdout`` so the
        VS-Code extension can subprocess-stream straight from the
        pipe.
    """

    if not stream:
        print(
            "cognithor agent run currently requires --stream",
            file=sys.stderr,
        )
        return 2

    try:
        plan = parse_plan_file(plan_path)
    except (ValueError, OSError) as exc:
        print(f"cognithor agent run: bad plan file: {exc}", file=sys.stderr)
        return 2

    sink = JsonlSink.to_path(out) if out is not None else JsonlSink()
    emitter = EventEmitter()
    emitter.add_sink(sink)

    return asyncio.run(_run(plan, emitter))


async def _run(plan: object, emitter: EventEmitter) -> int:
    await emitter.start()
    try:
        # plan is a _ParsedPlan; runner accepts the unstructured
        # type to avoid circular imports through the cognithor.cli
        # layer.
        return await run_plan(plan, emitter)  # type: ignore[arg-type]
    finally:
        await emitter.stop()
