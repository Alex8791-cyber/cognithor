# Cognithor PSE — Benchmarks

This page is the spec **D5** / **K1** scaffold. It documents *how*
Phase-1 PSE is benchmarked against the legacy NumPy-solver baseline
and is the file the eval-suite writes its numbers into.

The actual ARC-AGI-3 numbers (Solved@30s, Solved@5s, Median-Time, FP-Rate)
are produced by the eval-suite harness in
`tests/test_channels/test_program_synthesis/eval/test_arc_agi3_subset.py`
(spec §17.2) and committed into the *Latest run* table below. The
harness is marked `slow` and runs nightly, not on every CI push.

## Methodology (spec §18)

### Data

| Subset | Tasks | Use |
|---|---|---|
| `train` | 100 | Diversity-balanced — Geom 30 %, Color 30 %, Object 25 %, Mixed 15 %. Used during development. |
| `held_out` | 30 | Never seen during development. Final validation only. |

Task selection is recorded in `cognithor_bench/arc_agi3/manifest.json`
once the harness lands.

### Baseline

The current NumPy-solver (v0.78 "660× Speedup"-Solver) on identical
hardware, frozen as `baseline_v0.78.json` before any PSE search runs
on the train subset.

### Metrics

| Metric | Definition |
|---|---|
| **Solved@30s** | tasks solved within `wall_clock = 30 s` per task |
| **Solved@5s** | tasks solved within `wall_clock = 5 s` per task |
| **Median-Time-Solved** | median wall-clock over all *solved* tasks |
| **FP-Rate** | programs that pass *every* demo pair but fail the held-out check |

### Success threshold (K1)

```
Solved@30s_PSE  ≥  Solved@30s_baseline + 5
Solved@5s_PSE   ≥  Solved@5s_baseline           # no regression on easy tasks
```

### Reproducibility

* Fixed random seed (tie-break only — Phase-1 search is fully
  deterministic).
* DSL version recorded in every output row (currently `1.2.0`,
  pinned in `core/version.py`).
* Hardware fingerprint (CPU brand, core count, OS, Python) captured
  alongside the numbers.
* `make benchmark` is the single reproducible entry point — it boots
  the eval harness, runs both baseline and PSE, and updates the
  *Latest run* table below.

## Static catalog metrics

These numbers are derived from the live registry, not measured at
runtime — they document Phase-1's search-space surface and serve as
pre-conditions for the success threshold above.

| Metric | Value |
|---|---|
| Base primitives | **67** (56 base + 5 Phase-1.5 higher-order + 5 Sprint-8 object-level + 1 Sprint-10 fractal) |
| Predicate constructors | **13** (10 leaf + 3 combinators) |
| Lambda constructors | **4** (`identity`, `recolor`, `shift`, `branch`) |
| Higher-order primitives | **5** (`map_objects`, `filter_objects`, `align_to`, `sort_objects`, `branch`) |
| Default budget | `max_depth=4`, `wall_clock=30 s`, `max_candidates=50 000` |
| Cost-Tuner final pass | **D18 ✅** — see `dsl/auto_tuner.py` |

The auto-generated complete catalog with per-primitive cost is in
[`dsl_reference.md`](./dsl_reference.md).

## Latest run

> **Status: measured 2026-04-30 against the Phase-1 minimal fixture set**
> (8 train + 4 held-out tasks under `cognithor_bench/arc_agi3/`).
> The full 100/30 diverse curated set per spec §18.1 is a Phase-2
> expansion; the harness scales to it without code changes.

| Metric | Baseline (v0.78 NumPy solver, identity-only) | PSE Phase 1 | Δ |
|---|---|---|---|
| Solved@30s (train, n=8) | 1 | **8** | **+7** |
| Solved@5s  (train, n=8) | 1 | **8** | **+7** |
| Median-Time-Solved (train) | n/a | **~0.6 ms** | — |
| Solved@30s (held-out, n=4) | 0 | **4** | **+4** |
| Solved@5s  (held-out, n=4) | 0 | **4** | **+4** |
| FP-Rate (held-out) | n/a | **0 %** | — |
| **K1 threshold (train)** | — | ✅ +7 ≥ +5 | — |
| **D7 cache hit-rate on rerun** | — | ✅ 100 % | — |

Numbers reproduced by `tests/test_channels/test_program_synthesis/eval/
test_arc_agi3_subset.py::test_train_subset_meets_k1_threshold` (plus
the held-out + cache-hit-rate variants). Run on this commit's
`pse-1.2.0` / `dsl-1.2.0`. Hardware fingerprint + per-task breakdowns
from nightly CI will be written under
`cognithor_bench/arc_agi3/runs/<timestamp>/`.

## Microbenchmarks

In addition to the ARC-AGI-3 benchmark above, the test suite carries
microbenchmarks for the inner loops:

* `tests/test_channels/test_program_synthesis/perf/test_search_microbench.py`
  — depth-1 / depth-2 enumeration over a synthetic 10×10 grid.
* `tests/test_channels/test_program_synthesis/perf/test_replay_p95.py`
  — K10 replay budget gate: P95 ≤ 100 ms across the synthesised
  programs in the unit-test corpus.

These run on every CI push and protect against perf regressions
between releases. Microbench numbers are intentionally *not*
benchmarked against an external baseline — their job is "no
regression", not "beat NumPy".

## See also

* [`overview.md`](./overview.md) — channel intro + spec hard-gate
  status.
* [`architecture.md`](./architecture.md) — data-flow diagram.
* [`tutorial.md`](./tutorial.md) — Hello-World walk-through.
* [`dsl_reference.md`](./dsl_reference.md) — auto-generated primitive
  catalog.
* `docs/superpowers/specs/2026-04-29-pse-phase1-spec-v1.2.md` §18 —
  full benchmark plan.
