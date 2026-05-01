# Sprint-9 Reality-Check on Real ARC-AGI Corpus

**Date:** 2026-05-01
**Branch:** feat/pse-phase2-sprint9-real-arc-corpus
**Builds on:** v0.96.0 (synthetic corpus 100 %)

## TL;DR — the honest number

Cognithor v0.96.0 on the **full public fchollet/ARC-AGI corpus** (Apache-2.0, 800 tasks):

| Subset | Tasks | Phase-1 alone | + Cascade depth-1 | Δ |
|---|---|---|---|---|
| **training** | 400 | **4.5 %** (18) | **4.75 %** (19) | +0.25 PP |
| **evaluation** | 400 | **0.5 %** (2) | 0.5 % (2) | 0 PP |
| **TOTAL** | 800 | **2.5 %** (20) | 2.625 % (21) | +0.125 PP |

The 100 % score on `cognithor_bench/arc_agi3` (synthetic 20-task corpus) **did not transfer** to real ARC-AGI. This is the reality-check the user asked for in the strategic-plan brainstorm.

## What Sprint-9 added

`cognithor_bench/arc_agi3_real/`:
- `tasks/training/*.json` — full **400 training tasks** from `fchollet/ARC-AGI`
- `tasks/evaluation/*.json` — full **400 evaluation tasks**
- `manifest.json` — `training` / `evaluation` subsets, source attribution, Apache-2.0
- `LICENSE` — bit-for-bit attribution to upstream

**No new code**. Sprint-9 reuses the existing `synthesis/arc_corpus.py` loader and `synthesis/arc_baseline_runner.py` / `synthesis/sprint7_cascade_runner.py` runners — both accept any `--corpus-root` so the same infrastructure works on the real corpus.

10 new tests verify the corpus structure, license attribution, subset disjointness, and palette compliance.

## Per-subset analysis

### Phase-1 EnumerativeSearch — what it does solve

18 training + 2 evaluation tasks (20/800 total). Looking at the 18 training successes:

| Task ID | Input shape | Output shape |
|---|---|---|
| 1cf80156 | 10×12 | 4×4 (= crop_bbox) |
| 1e0a9b12 | 4×4 | 4×4 |
| 3906de3d | 10×10 | 10×10 |
| 3c9b0459, 6150a2bd, 6f8cd79b, 74dd1130 | 3×3 | 3×3 |
| 67a3c6ac, 68b16354, 7468f01a, 9172f3a0, 9dfd6313, a416b8f3, b1948b0a, c59eb873, c8f0f002, d511f180, ed36ccf7 | small | similar |

**Pattern**: Phase-1 solves small-grid (≤ 12×12), shape-stable or shape-shrinking 1-2-step transformations — exactly the easy ARC subset.

### Phase-1 — what it cannot solve

- **107 tasks have ≥ 4 demos**: typical ARC tasks where multiple input/output pairs need a shared rule. Phase-1 finds programs that match each demo individually but no synthesis covers all.
- **138 tasks have shape changes** (102 shrink, 36 grow): variable output dimensions are hard for a depth-bounded enumerative search.
- **262 tasks have shape-stable outputs** but require object-level reasoning (filter, count, recolour-by-property). Phase-1's primitives don't compose to that without dedicated primitives.

### Cascade depth-1 — barely moves the needle

Closes **only 1 additional task** (training, +0.25 PP). On evaluation: **0 improvement**. Cascade adds 18 unary chains of length 1 + recolor variants — that's not enough new search surface to crack tasks Phase-1 misses.

### Why depth-2 cascade is impractical

Measured: depth-2 cascade on a single 10×10 task costs **23.6 seconds**. On 800 tasks that's ~5 hours. The combinatorial explosion (18² × 6 = 1944 candidates × 5 demos = ~10 000 verifications per task on grids 4× larger than the synthetic hard subset) makes depth-2 unusable as a default strategy.

## Honest assessment of v0.96.0

### What v0.96.0's "100 % on cognithor_bench/arc_agi3" actually means

- The 20 tasks were **synthetic, hand-crafted to demonstrate Sprint-by-Sprint progression**
- The 5 Sprint-8 DSL primitives (`tile_3x`, `remove_singletons`, etc.) **closed the synthetic tasks because the tasks were designed around them**
- Sprint-7's cascade closed the synthetic 0202 because the recolor cascade fitted that fixture's exact diff pattern
- This is not generalizable capability — it's curve-fitting to a small designed-from-the-inside corpus

### What v0.96.0's "2.5 % on real ARC" actually means

- Cognithor in its current state solves the **easiest 2-3 % of ARC-AGI tasks** out of the box
- This is comparable to 2017-era brute-force enumerators
- State-of-the-art ARC solvers (Hodel, icecuber, FunSearch) hit 25-40 % through specialised DSLs and search heuristics
- Top-tier human solvers hit 80-90 %; LLM-based approaches (Claude, GPT-4) hit 40-60 %

## What Sprint-10+ would need to close the gap

The Sprint-9 reality-check directly informs the next 6 months of work:

### Priority 1: Object-level primitives that match ARC's actual task distribution

Sprint-8 added 5 primitives that fit synthetic-hard. Real ARC needs:
- **Symmetry detection** (`is_symmetric`, `complete_symmetry`)
- **Anchor detection** (`find_object_at`, `align_to_grid_corner`)
- **Conditional fills** (`flood_fill_if`, `apply_to_each_object`)
- **Pattern recognition** (`detect_repeating_pattern`, `extract_grid_motif`)
- **Object-relational ops** (`pair_objects_by_property`, `move_object_to`)

Estimate: **30-50 new primitives** to reach 15-20 % on training.

### Priority 2: LLM-driven search guidance

Phase-1 enumerative search blindly explores. Real ARC needs LLM-prior guidance per-task:
- Activate `LLMPriorClient` (already shipped in Sprint-1) against a real backend
- Symbolic-Prior-Catalog (Sprint-1) with the 20 heuristics needs evaluation on real data
- DualPriorMixer (Sprint-1) against a vLLM/Qwen instance

Estimate: **10-15 PP score improvement** once LLM-prior is wired and tuned.

### Priority 3: Module B — MCTS controller in production wiring

Sprint-1 shipped `MCTSController` as a standalone module. Real ARC needs it as the search backend, not just a tested abstraction.

Estimate: **5-10 PP score improvement** through deeper search guided by LLM-prior.

### Priority 4: Library-Learning / Skill extraction

Real ARC has clusters of similar tasks. Solving 5 in a cluster should make subsequent solving cheaper. The skills system exists (`cognithor.skills`) but isn't wired into PSE.

Estimate: **5 PP improvement** + dramatic latency reduction on test-time tasks.

### Realistic 6-month target

Aggressive but achievable: **15-25 % on training, 8-15 % on evaluation** by Sprint-15. That puts Cognithor in the same ballpark as published 2024 solvers, not yet at GPT-4-class but a real research-quality system.

## Reproducibility

```bash
git checkout feat/pse-phase2-sprint9-real-arc-corpus

# Phase-1 baseline:
python -m cognithor.channels.program_synthesis.synthesis.arc_baseline_runner \
    --corpus-root cognithor_bench/arc_agi3_real \
    --subset training \
    --output baseline.json
# Expected: success_rate=0.045, n_tasks=400

# + Cascade depth-1:
python -m cognithor.channels.program_synthesis.synthesis.sprint7_cascade_runner \
    --corpus-root cognithor_bench/arc_agi3_real \
    --subset training \
    --output cascade.json --max-chain-depth 1
# Expected: success_rate=0.0475, refined_rate=0.035, refinement_uplift_rate~0.07
```

Persisted baselines (frozen, in `.ci/`):
- `arc_real_phase1_training.json` — Phase-1 only on 400 training
- `arc_real_phase1_evaluation.json` — Phase-1 only on 400 evaluation
- `arc_real_cascade_d1_training.json` — + Sprint-7 cascade depth-1
- `arc_real_cascade_d1_evaluation.json` — same on evaluation

## What this means for the v0.96.0 release

**v0.96.0 stays valid.** Its release notes are honest:
- "100 % on the committed cognithor_bench/arc_agi3 corpus" → true
- The corpus is committed, the score reproduces, the infrastructure works
- v0.96.0 does NOT claim to solve real ARC-AGI

The Sprint-9 finding **does not invalidate** the release — it adds a bigger, more honest data point that should drive the next 6 months of work.

## Sprint-10 directive (deduced)

Three weeks of work, in order:

1. **Object-level DSL extension** (~10 days): add the 30-50 primitives listed above. Should lift training score from 4.5 % to 12-15 %.
2. **LLM-Prior wiring** (~5 days): activate `LLMPriorClient` against a real vLLM/Qwen backend. Should lift another 5-10 PP.
3. **Honest re-measurement** on full corpus + commit baselines as `.ci/arc_real_v097_*.json` for release v0.97.0.

Sprint-9 deliverable: the **honest baseline number**. Sprint-10's job is to close the gap.
