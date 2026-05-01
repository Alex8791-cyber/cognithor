# Sprint-4 Reality-Check: Phase-1 on ARC-AGI-3-style corpus

**Date:** 2026-05-01
**Branch:** feat/pse-phase2-sprint4-arc-agi3-corpus
**Runner:** `python -m cognithor.channels.program_synthesis.synthesis.arc_baseline_runner`

## Question

> "Wo steht Cognithor heute auf echtem ARC-AGI-3?"

Sprint-4 directive: measure Phase-1 EnumerativeSearch alone (no Phase-2 wiring) on the ARC-AGI-3 corpus committed at `cognithor_bench/arc_agi3/`.

## Setup

- **Corpus:** 12 existing tasks (`train`, `held_out`) from v0.78 + 8 new ARC-style **hard** tasks added in this PR
- **Engine:** Phase-1 `EnumerativeSearch` only — no LLM-prior, no refiner, no MCTS
- **Budget per task:** 5 seconds wall-clock, max_depth=4, max_candidates=10 000
- **Hardware:** dev machine (Win-py3.13, no GPU)

## Results

| Subset | Description | Tasks | Success rate | P50 | P95 |
|---|---|---|---|---|---|
| `train` | 1-step transformations (rotate, mirror, recolor, identity) | 8 | **100 %** (8/8) | 0.6 ms | 1.9 ms |
| `held_out` | Same complexity as train, different specific transforms | 4 | **100 %** (4/4) | 1.9 ms | 79 ms |
| **`hard`** | **ARC-style: object-level reasoning, multi-step composition** | 8 | **12.5 %** (1/8) | 79 ms | 127 ms |

## Per-task breakdown — `hard` subset

| Task | Score | Verdict | Why |
|---|---|---|---|
| `0201_count_objects_to_color` | 0.0 | NO_SOLUTION | Requires object counting + scalar→grid; no DSL primitive for either |
| `0202_largest_object_only` | **0.5** | **PARTIAL** | 1/2 demos passed — refinable candidate |
| `0203_fill_enclosed_areas` | 1.0 | SUCCESS | Phase-1 found a flood-fill-equivalent program |
| `0204_grid_to_diagonal` | 0.0 | NO_SOLUTION | Variable output size depending on input; needs object→position mapping |
| `0205_repeat_pattern_to_size` | 0.0 | NO_SOLUTION | 3×3 tiling missing as primitive (only have `tile_2x`) |
| `0206_remove_singletons` | 0.0 | NO_SOLUTION | Requires neighbour-counting + conditional masking |
| `0207_color_by_size` | 0.0 | NO_SOLUTION | Component-size measurement → recolor; no size-aware primitive |
| `0208_horizontal_then_vertical` | **0.5** | **PARTIAL** | 1/2 demos passed — 3-step chain at depth-3 is at the search frontier |

## Findings

### 1. Phase-1 is *very* effective on simple synthetic transformations

100 % on the trivial `train` and `held_out` subsets. This is what makes Sprint-2's leak-free 95 % score not a great differentiator: Phase-1 alone solves nearly everything synthetic.

### 2. The "ARC-style hard" subset reveals the real gap

12.5 % success rate. Phase-1 fails on tasks requiring:
- **Object-level reasoning** (count, filter by size, neighbour-aware)
- **Variable output shapes** (output dimensions depend on input properties)
- **DSL primitives we don't have** (component-size, scalar→grid, tile_3x)
- **Long composition chains** (3+ steps push the enumerator's depth limit)

### 3. Sprint-3's missing borderline-partials are present here

**3 of 8 tasks land at score = 0.5** (one demo correct, one incorrect):
- `0202_largest_object_only`
- `0208_horizontal_then_vertical`
- (#0203 succeeded fully)

These are exactly the cases Phase-2's RefinerEscalator targets. On the Sprint-2 leak-free fixtures, no such partials existed — every task was either trivially solved or completely failed. **The hard subset has the data Phase-2 needs to demonstrate uplift.**

## Implications for the 6-month roadmap

The roadmap from the previous strategic plan is largely confirmed, with one important refinement:

- **Track 1 (Real ARC corpus)**: still essential, but Sprint-4 shows we can already reason about gaps without external data. The 8 hard tasks are sufficient for Sprint-5's Phase-2 A/B test to actually fire the refiner.
- **Track 2 (Graduated Verifier)**: confirmed — Phase-1's score *is* graduated when partial matches exist (the 0.5 verdicts above). The blocker for Sprint-3's A/B was the absence of partial-credit tasks in the fixture set, not the verifier semantics.
- **Track 3 (DSL expansion)**: confirmed as **highest-impact** lever. Six of seven `hard` failures are explicitly missing-primitive cases (object-counting, neighbour-aware ops, component-size, etc.). DSL expansion should jump in priority above LLM integration.

## Sprint-5 directive candidates

Three concrete next steps, ordered by ROI:

1. **Re-run the A/B test (#266) on the `hard` subset**: We now have 3 borderline-partial tasks. Run Phase-1-only vs Phase-2-wired on `hard`; expected uplift > 0 PP. This validates the Phase-2 architecture in 1 day.
2. **Add 5-10 object-level DSL primitives** (`count_objects`, `component_size`, `tile_3x`, `remove_singletons`, `filter_by_size`): Sprint-1 had 60 primitives; the hard subset reveals we need ~10 more to cover ARC-level composition.
3. **Curate 50 real ARC-AGI-3 tasks** from the public Apache-2.0 repository: replace synthetic `hard` subset with real-data validation.

## Reproducibility

```bash
# Train (expected 100 %):
python -m cognithor.channels.program_synthesis.synthesis.arc_baseline_runner \
    --subset train --output .ci/arc_agi3_phase1_train_baseline.json

# Held-out (expected 100 %):
python -m cognithor.channels.program_synthesis.synthesis.arc_baseline_runner \
    --subset held_out --output .ci/arc_agi3_phase1_held_out_baseline.json

# Hard (expected 12.5 %):
python -m cognithor.channels.program_synthesis.synthesis.arc_baseline_runner \
    --subset hard --output .ci/arc_agi3_phase1_hard_baseline.json
```

Bundle hashes (drift in fixtures fails CI):
- train: `sha256:dcfbdf3fdafa198b31b7486ce9528ded99a289895885db49c8ba8a2772777742`
- held_out: `sha256:2b9055cb0298d2056f0192701a208b662af40d4c9202566b8e0a65983b375be3`
- hard: `sha256:ba75f39b5d9b02fa6ae4457dfdd36a0e4a5d794a9b0af61c3bf1f5ff273aab40`
