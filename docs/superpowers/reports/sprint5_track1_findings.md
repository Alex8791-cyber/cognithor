# Sprint-5 Track 1: A/B-Test on hard subset

**Date:** 2026-05-01
**Branch:** feat/pse-phase2-sprint5-ab-on-hard
**Builds on:** Sprint-3 Track 1 (#266) infrastructure + Sprint-4 (#267) ARC-AGI-3 corpus

## Question

Sprint-3's A/B test (#266) on the leak-free fixtures showed **0 % refined-rate** because the fixtures had no borderline-partials. Sprint-4 (#267) confirmed that the **hard subset** has 3 borderline-partials (#0202 + #0208 at score=0.5, plus #0203 already SUCCESS).

> Does Phase-2 wiring activate on those borderline-partials, and does it close them?

## Setup

```bash
# Phase-1-only baseline:
python -m cognithor.channels.program_synthesis.synthesis.benchmark_runner \
    --arc-corpus cognithor_bench/arc_agi3 \
    --arc-subset hard \
    --output .ci/sprint5_hard_phase1.json

# Phase-2-wired comparison:
python -m cognithor.channels.program_synthesis.synthesis.benchmark_runner \
    --phase2 \
    --arc-corpus cognithor_bench/arc_agi3 \
    --arc-subset hard \
    --output .ci/sprint5_hard_phase2.json \
    --baseline .ci/sprint5_hard_phase1.json
```

## Results

| Metric | Phase-1-only | Phase-2-wired | Δ |
|---|---|---|---|
| **success_rate** | 12.5 % | 12.5 % | **+0.0 PP** |
| **refined_rate** | 0 % | **25 %** | **+25 PP** |
| refinement_uplift_rate | n/a | 0 % | n/a |
| P50 latency | 81 ms | 82 ms | +1 ms |
| P95 latency | 127 ms | 127 ms | +0 ms |

## Per-task Phase-2 outcomes

| Task | Phase-1 | Phase-2 | Notes |
|---|---|---|---|
| 0201_count_objects_to_color | 0.0 NO_SOLUTION | 0.0 NO_SOLUTION | Refiner skipped (Phase-1 program=None) |
| **0202_largest_object_only** | **0.5 PARTIAL** | **0.5 refined_partial** | **Refiner activated, no improvement found** |
| 0203_fill_enclosed_areas | 1.0 SUCCESS | 1.0 SUCCESS | Refiner skipped (already winning) |
| 0204_grid_to_diagonal | 0.0 NO_SOLUTION | 0.0 NO_SOLUTION | Refiner skipped |
| 0205_repeat_pattern_to_size | 0.0 NO_SOLUTION | 0.0 NO_SOLUTION | Refiner skipped |
| 0206_remove_singletons | 0.0 NO_SOLUTION | 0.0 NO_SOLUTION | Refiner skipped |
| 0207_color_by_size | 0.0 NO_SOLUTION | 0.0 NO_SOLUTION | Refiner skipped |
| **0208_horizontal_then_vertical** | **0.5 PARTIAL** | **0.5 refined_partial** | **Refiner activated, no improvement found** |

## Findings

### Win: Phase-2 wiring activates on real partials

For the first time since Sprint-3, the Phase-2 RefinerEscalator entered Stage 2 dispatch on 2/8 tasks (the borderline-partials). Sprint-3 saw 0 % refined-rate; Sprint-5 sees 25 %. The wiring works as designed.

### Loss: Local-Edit alone doesn't close the gap

Both refined tasks ended in `refined_partial` (refiner ran but produced no improvement over the 0.5 baseline). Looking at why:

* **0202_largest_object_only** needs `largest_object` + masking — a 2-3 step chain. Local-Edit only mutates the existing program one token at a time; it can't synthesize the missing structure.
* **0208_horizontal_then_vertical** is a 3-step composition (rotate90 → mirror_horizontal → recolor 1→9). Phase-1 finds 1 of 2 demos via a 2-step program; Local-Edit's primitive substitution can't extend the chain to 3.

Both failures are **structural-search problems**, not single-token mutations. Local-Edit was designed for the latter.

### Sprint-6 directive (high confidence)

The two refined-but-unsolved tasks need:

1. **Symbolic-Repair-Advisor live** (`refiner/symbolic_repair.py` from Sprint-1 PR #252): currently stubbed in benchmark_runner's `_build_phase2_engine`. The advisor proposes structural modifications (insert subtree, replace with composition) — the kind of changes Local-Edit can't make. Wiring the existing `advise_repairs(...)` into the refiner pipeline is ~1 day's work.
2. **DSL primitives that close 0202**: `largest_component(grid)`, `mask_keep(grid, condition)`. The component-detection primitive alone might let Phase-1 solve 0202 directly without Phase-2 even needing to fire.

Recommended Sprint-6 split:
* **Track A**: Symbolic-Repair-Advisor live + re-run A/B on hard subset (target: refinement_uplift_rate ≥ 0.5)
* **Track B**: 5-10 object-level DSL primitives (target: hard success_rate from 12.5 % to ≥ 30 %)

## Reproducibility

The results above are deterministic on the committed corpus (bundle_hash `sha256:ba75f...`). To reproduce:

```bash
git checkout feat/pse-phase2-sprint5-ab-on-hard
python -m cognithor.channels.program_synthesis.synthesis.benchmark_runner \
    --arc-corpus cognithor_bench/arc_agi3 --arc-subset hard --output /tmp/p1.json
python -m cognithor.channels.program_synthesis.synthesis.benchmark_runner \
    --phase2 --arc-corpus cognithor_bench/arc_agi3 --arc-subset hard \
    --output /tmp/p2.json --baseline /tmp/p1.json
```

Expected: `success_rate=0.125 / refined_rate=0.0` (P1) and `success_rate=0.125 / refined_rate=0.25` (P2).
