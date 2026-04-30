# Cognithor PSE — Hello-World Tutorial

This tutorial walks through synthesising and explaining one ARC-DSL
program end-to-end with the **Cognithor Program Synthesis Engine
(PSE)**. It is the spec **D12** acceptance criterion: at least one
solved task ships with a complete, replay-able trace that a contributor
can reproduce verbatim.

The task is the trivial 90°-rotation case (`pse-phase1-spec §24.A.1`).
It is intentionally small: depth-1 program, two enumerated candidates,
sub-millisecond search time. Once you have run it, the larger §24.A.2
through §24.A.4 examples follow the same recipe.

## Prerequisites

* `pip install cognithor` (or run from a checked-out repo with
  `pip install -e .`).
* The PSE channel is auto-registered — no extra setup, no Ollama, no
  network. The synthesis is purely symbolic.
* On Windows, the default sandbox strategy is `wsl2-worker`. The CLI
  falls back to `windows-research` if WSL2 is unavailable; the
  end-to-end behaviour is identical for this tutorial.

## Step 1 — write the task file

Save the following as `hello_world.json`:

```json
{
  "examples": [
    {"input": [[1, 2], [3, 4]],         "output": [[3, 1], [4, 2]]},
    {"input": [[5, 6], [7, 8]],         "output": [[7, 5], [8, 6]]},
    {"input": [[1, 1, 2], [3, 4, 5], [6, 7, 8]],
     "output": [[6, 3, 1], [7, 4, 1], [8, 5, 2]]}
  ],
  "budget": {"max_depth": 2, "wall_clock_seconds": 30.0}
}
```

Three demo pairs, all transformed by the same rule: rotate the grid
90° clockwise. PSE's heuristic `is_synthesizable` requires ≥ 2 demos
(see `integration/pge_adapter.py`); three is the minimum that gives
the verifier a held-out check.

## Step 2 — run the channel

```sh
cognithor pse run hello_world.json
```

Verbatim output (captured by the regression test that backs this
tutorial — see *Reproducing this output* below):

```text
status       : success
score        : 1.00
confidence   : 1.00
cost_seconds : 0.000
candidates   : 2
cache_hit    : False

program: rotate90(input)

# PSE Solution Trace
# Program hash: sha256:31dc973dce6c26fa12a7c3d2f72e5d3f9716018a30dc6dee421385b08ab4c671
# PSE version: 1.2.0-draft
# DSL version: 1.2.0
# Search time: 0.000s, 2 candidates

Step 1: step1 = rotate90(input)
        # → Grid (2, 2)

# Final: Grid (2, 2)
```

That is the full Phase-1 contract: **status**, **program source**,
**trace**, **program hash**, **search cost**.

## Step 3 — read the trace

The trace block is the K9 hard-gate output. Every line maps to one
node in the synthesised program tree:

| Field | Meaning |
|---|---|
| `Program hash` | SHA-256 of the canonical program form. Identical inputs → identical hash → cache hit on rerun. |
| `PSE version` / `DSL version` | Pinned in `core/version.py`. The trace is invalid against a different DSL version. |
| `Search time` | Wall-clock + total candidates the enumerator looked at (after observational-equivalence pruning). |
| `Step N: <var> = <primitive>(...)` | One bottom-up evaluation step. The comment shows the inferred output type and shape. |
| `Final: ...` | The output type/shape of the program's root node. |

For this task there is exactly one step because `rotate90` is a
single-arity primitive that fits in depth 1. Larger programs (see
spec §24.A.2 through §24.A.4) emit one step per internal node.

## Step 4 — replay the program

The trace is replay-able by construction (K10 hard-gate). To verify
it from Python:

```python
from cognithor.channels.program_synthesis.trace import (
    build_trace, replay_program,
)
from cognithor.channels.program_synthesis.integration.pge_adapter import (
    ProgramSynthesisChannel, SynthesisRequest,
)
from cognithor.channels.program_synthesis.core.types import Budget, TaskSpec
import numpy as np

spec = TaskSpec(examples=(
    (np.array([[1, 2], [3, 4]],   dtype=np.int8),
     np.array([[3, 1], [4, 2]],   dtype=np.int8)),
    (np.array([[5, 6], [7, 8]],   dtype=np.int8),
     np.array([[7, 5], [8, 6]],   dtype=np.int8)),
))
result = ProgramSynthesisChannel().synthesize(
    SynthesisRequest(spec=spec, budget=Budget(max_depth=2)),
)
assert result.status.value == "success"
replay = replay_program(
    result.program,
    spec.examples[0][0],
    spec.examples[0][1],
)
assert replay.identical
assert replay.duration_ms < 100  # K10 P95 budget
```

The cache-stored form is the program *source string*, not the cached
output, so replay always re-executes the live DSL — that is what makes
K10 honest.

## Step 5 — explore the DSL

The full primitive catalog is auto-generated and committed at
[`dsl_reference.md`](./dsl_reference.md). To inspect a single one
from the CLI:

```sh
cognithor pse dsl describe rotate90
```

```text
name        : rotate90
arity       : 1
signature   : ('Grid',) -> Grid
cost        : 1.0
description : Rotate the grid 90° clockwise.
examples:
  [[1,2],[3,4]]  →  [[3,1],[4,2]]
```

Every primitive's `cost` was decided by the deterministic Auto-Tuner
(D18 hard-gate, `dsl/auto_tuner.py`) — there is no ML in the loop.

## Reproducing this output

The exact command and trace block in this tutorial are pinned by
`tests/test_channels/test_program_synthesis/unit/test_hello_world_tutorial.py`.
That test runs the CLI against the JSON above and asserts the result
status, program source, search-cost line, and the trace shape. If
you change the DSL or the formatter and the tutorial output drifts,
that test fails — the gate is the same drift-test pattern used for
`dsl_reference.md`.

## What to read next

* **Architecture diagram** — [`architecture.md`](./architecture.md).
* **Channel overview** — [`overview.md`](./overview.md).
* **Full DSL catalog** — [`dsl_reference.md`](./dsl_reference.md).
* **Original spec** — `docs/superpowers/specs/2026-04-29-pse-phase1-spec-v1.2.md`.
