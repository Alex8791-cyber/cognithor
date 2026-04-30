# Cognithor PSE — Channel Overview

The **Program Synthesis Engine (PSE)** is a Cognithor channel that
synthesizes deterministic, replay-able **programs** instead of
free-form LLM answers. Given a task with input/output demo pairs (an
ARC-AGI-3 task, in Phase 1), the channel searches a typed DSL for a
program that maps every demo input to its expected output, then ships
the program plus a step-by-step trace as the response.

## Why programs

Three concrete properties LLM-only ARC solvers can't match:

1. **Deterministic.** The same program against the same input always
   yields the same output (K10 hard gate; verified by replay tests).
2. **Replay-able.** A solved program re-executes in P95 ≤ 100 ms on
   any solver host (K10 hard gate). The cache stores the program
   source rather than the cached output, so re-execution always works.
3. **Explainable.** Every solved task ships a human-readable
   pseudo-code trace with intermediate values (K9 hard gate). The
   trace is the spec's answer to "why did the model do that?".

## Where it lives

```
src/cognithor/channels/program_synthesis/
├── core/         types, exceptions, version constants
├── dsl/          primitives + Predicate + Lambda + auto-tuner
├── search/       enumerative bottom-up + observational equivalence
├── verify/       five-stage pipeline (syntax → type → demo → property → held-out)
├── sandbox/      strategy router (Linux / WSL2 / Windows-research)
├── integration/  PGE adapter, cache, capability tokens, SGN bridge,
│                 numpy fast-path
├── observability/ counters + histograms + Hashline-style audit trail
├── trace/        K9/K10 trace builder + replay verifier
└── cli/          `cognithor pse <subcommand>`
```

## Phase-1 catalog (current)

* **56 base primitives** — geometric, color, size/scale, spatial,
  object detection, mask/logic, construction, color constants.
* **5 higher-order primitives** — `map_objects`, `filter_objects`,
  `align_to`, `sort_objects`, `branch`.
* **13 predicate constructors** — `color_eq`, `color_in`, `size_eq`,
  `size_gt`, `size_lt`, `is_rectangle`, `is_square`, `is_largest_in`,
  `is_smallest_in`, `touches_border`, plus combinators
  `not` / `and` / `or`.
* **4 lambda constructors** — `identity_lambda`, `recolor_lambda`,
  `shift_lambda`, `branch_lambda`.

Auto-generated complete catalog: [`dsl_reference.md`](./dsl_reference.md).
Benchmark methodology + latest run: [`benchmarks.md`](./benchmarks.md).
Hello-World walkthrough: [`tutorial.md`](./tutorial.md).

## Public API

```python
from cognithor.channels.program_synthesis import (
    ProgramSynthesisChannel, SynthesisRequest,
    Budget, TaskSpec, SynthesisResult,
)

channel = ProgramSynthesisChannel()
result = channel.synthesize(SynthesisRequest(spec, Budget(max_depth=4)))
```

The channel is dependency-injection-friendly — every collaborator
(`PSECache`, `NumpySolverBridge`, `StateGraphBridge`, the sandbox
strategy, the search engine, a `Registry`, an `AuditTrail`) can be
swapped for tests.

## CLI

```sh
cognithor pse dsl list                  # all primitives
cognithor pse dsl describe rotate90     # one primitive
cognithor pse dsl reference --output X  # auto-gen this catalog
cognithor pse sandbox doctor            # platform-detected strategy
cognithor pse run task.json             # E2E synthesis + trace
```

## Spec hard-gates

| Gate | Status |
|---|---|
| **K9** Trace-Vollständigkeit (every solved task has a trace) | ✅ Phase 1 |
| **K10** Replay-Reproduzierbarkeit (P95 ≤ 100 ms, byte-identical) | ✅ Phase 1 |
| **K4** 100 % adversarial-cases blocked | ✅ active layers; 12 subprocess cases scaffolded |
| **D17** WSL2-Default under Windows | ✅ Strategy router |
| **D18** Auto-Tuner Pflichtlauf | ✅ Deterministic, no-ML |
| **D3** Test coverage ≥ 90 % on new code | ✅ 95 % on `channels/program_synthesis/` |
| **D8** CLI works | ✅ Subset shipped |
| **D9** Docs (overview + architecture + dsl_reference + tutorial + benchmarks) | ✅ All five present |
| **D10** Hashline audit trail | ✅ Chain-hash verifier |
| **D11** Telemetry counters | ✅ In-process Registry |
| **D12** Hello-World task documented with full trace | ✅ `tutorial.md` + drift gate |

## What's NOT in Phase 1 (Phase 2 / 3 / 4)

* No LLM prior — search is rein symbolisch.
* No MCTS — bottom-up enumeration only.
* No Library Learning / DreamCoder.
* No CEGIS.
* No additional DSLs (only ARC-DSL).

See `docs/superpowers/specs/2026-04-29-pse-phase1-spec-v1.2.md` for
the full spec, including the seven Phase-2 open questions.
