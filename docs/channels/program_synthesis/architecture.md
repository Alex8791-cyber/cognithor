# Cognithor PSE — Architecture

## Data flow (happy path)

```
SynthesisRequest (spec, budget, sgn_hints?)
          │
          ▼
┌──────────────────────────────────────┐
│ ProgramSynthesisChannel.synthesize() │
└──────┬───────────────────────────────┘
       │
       ▼  apply SGN hints if any
TaskSpec (annotated)
       │
       ▼  hash spec + budget + dsl_version
Cache lookup ──── hit? ───── return cached result
       │ miss
       ▼
NumPy fast-path ──── solved? ── stamp + cache + return
       │ no
       ▼
EnumerativeSearch.search()
       │
       │ depth=0..max_depth:
       │   for each primitive p of arity > 0:
       │     for each typed-arg-tuple in bank^arity:
       │       candidate = Program(p, args)
       │       if pruner.admit(candidate, output_type): bank.append
       │       if candidate satisfies all demos: return SUCCESS
       │
       ▼
Verifier (5-stage)         ← syntax → type → demo → property → held-out
       │
       ▼
SynthesisResult (status, program, score, confidence, trace, ...)
       │
       ▼
Cache write (if status cacheable)
       │
       ▼  always emits
Telemetry counters + AuditTrail entry (chain-hash)
       │
       ▼
return SynthesisResult
```

## Layers

| Layer | What it does | Phase-1 entry |
|---|---|---|
| **`core/`** | Frozen data types (`TaskSpec`, `Budget`, `SynthesisResult`), exception hierarchy, version constants. | — |
| **`dsl/`** | Primitive registry + 56 base primitives + 5 higher-order primitives + closed Predicate / Lambda systems + deterministic Cost-Auto-Tuner. | `REGISTRY`, `auto_tune` |
| **`search/`** | Bottom-up `EnumerativeSearch`, `ObservationalEquivalencePruner`, `Executor` protocol + `InProcessExecutor`. | `EnumerativeSearch().search(spec, budget)` |
| **`verify/`** | 5-stage `Verifier` pipeline (syntax → type → demo → property → held-out). | `Verifier().verify(program, spec)` |
| **`sandbox/`** | Strategy router: Linux native / WSL2 worker / Windows research-mode. Capability allow-list per strategy. | `select_sandbox_strategy()` |
| **`integration/`** | `ProgramSynthesisChannel` (orchestrator), `PSECache` (Tactical Memory shape), `NumpySolverBridge` (fast-path), `StateGraphBridge` (cost multipliers), capability constants. | `ProgramSynthesisChannel().synthesize(...)` |
| **`observability/`** | Counters + Histograms + AuditTrail (Hashline-shape SHA-256 chain). | `Registry`, `AuditTrail` |
| **`trace/`** | K9 trace builder, K10 replay verifier. | `build_trace`, `replay_program` |
| **`cli/`** | `cognithor pse <subcommand>` argparse front-end. | `python -m cognithor.channels.program_synthesis.cli` |

## Key invariants

* **Pure DSL primitives.** No in-place mutation, no shared state. Each
  call returns a fresh array / Object. Tested per primitive.
* **Closed predicate / lambda set.** No free Python lambdas. The
  search engine constructs predicates and lambdas only from the
  registered constructors (10 + 3 + 3 + 1 = 17 buildable nodes).
* **Sub-tiefe ≤ 1 for `branch`.** Phase 1 forbids `branch(branch(...))`
  to keep the search tractable; enforced both at construction time
  (the `branch` primitive) and at evaluation time (the `branch_lambda`
  evaluator).
* **Deterministic everything.** Same inputs → same outputs. Cache keys
  are byte-stable across processes (canonical JSON + SHA-256). Sort /
  filter / fingerprint all break ties by discovery order.
* **Sandbox is policy boundary.** The Strategy is the same `Executor`
  the search engine uses, so production-grade isolation slots in
  without changing call sites. The actual subprocess + AST whitelist
  + setrlimit machinery is a planned follow-up that swaps `execute()`
  on the strategy class.

## Capability gates

| Capability | Holder | Notes |
|---|---|---|
| `pse:synthesize` | Planner | Allowed on every strategy. |
| `pse:synthesize:production` | Planner | Linux/WSL2 only — research mode strips this. |
| `pse:execute` | Executor | Run the synthesized program. |
| `pse:cache:read` / `pse:cache:write` | Channel | Tactical-memory cache. |
| `pse:dsl:extend` | Admin | Register a new primitive at runtime. |
| `pse:dsl:tune` | Admin | Run the deterministic Cost-Auto-Tuner. |

## What's been validated end-to-end

* `output = rotate90(input)` — synthesized at depth 1 in < 100 ms.
* `output = rotate180(input)` — depth-2 search picks `rotate180`
  directly (cheaper) over `rotate90 ∘ rotate90` via cost prior.
* Identity tasks (`output = input`) — short-circuit returns `InputRef`
  before the search starts.
* CLI `pse run` solves a 2-demo rotate90 task and emits the trace
  block (K9 verified).
* Cache round-trip: second call to the same `(spec, budget)` returns
  in microseconds with `cache_hit=True`.
