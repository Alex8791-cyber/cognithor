# Sprint-6 Track A: Symbolic-Repair-Advisor live — first measurable Score-Lift

**Date:** 2026-05-01
**Branch:** feat/pse-phase2-sprint6-symbolic-repair-live
**Builds on:** Sprint-5 Track 1 (#268), Sprint-4 (#267) corpus

## Headline result

Cognithor's first **measurable Phase-2 Score-Lift** on real ARC-style tasks:

| Engine | hard subset success_rate | refined_rate | uplift_rate |
|---|---|---|---|
| Phase-1 only (Sprint-4 baseline) | 12.5 % (1/8) | 0 % | n/a |
| Phase-1 + WiredPhase2Engine local-edit (Sprint-5) | 12.5 % (1/8) | 25 % | **0 %** |
| **Phase-1 + Symbolic-Repair-Advisor (Sprint-6)** | **25 % (2/8)** | 12.5 % | **100 %** |

**+12.5 PP success-rate**, the first real Phase-2 contribution observed end-to-end.

## What closed task 0202_largest_object_only

Phase-1's enumerative search produced `mirror_vertical(input)` (score 0.5; right on demo 2 by coincidence, wrong on demo 1). The Symbolic-Repair-Advisor saw:

* `palette_actual = {0, 1, 2, 3}` (after mirror_vertical on demo 1's input)
* `palette_expected = {0, 1}`
* Diff: shape match, pixel diff = 6, colors {2, 3} introduced

The advisor returned `color_repair / hint=recolor`. The Sprint-6 candidate-builder then enumerated:

* Single-step replacement: `recolor(input, src, dst)` for every (src, dst) pair → all score 0.5
* Wrap on top of Phase-1: `recolor(mirror_vertical(input), src, dst)` → still 0.5 max
* **2-step cascade on InputRef base**: `recolor(recolor(input, 2, 0), 3, 0)` → **score 1.0** (both demos correct — the 2-step cascade removes the spurious colors that Phase-1's `mirror_vertical` couldn't fix)

The winner replaces Phase-1's structurally-wrong `mirror_vertical` base with a fresh `InputRef`, then cascades two recolors derived from the diff's `palette_actual − palette_expected` set.

## What didn't close task 0208_horizontal_then_vertical

Phase-1 produced `recolor(input, 1, 9)` (score 0.5 — coincidentally correct on demo 0, wrong on demo 2). The advisor on the failing demo saw:

* `shape_mismatch=True`, `pixel_count=0`, `colors_intro=[]`, `colors_miss=[]`

→ **No suggestion fired.** The advisor's R3 RotationRepair / R4 MirrorRepair require the *whole* expected/actual to be a rotation/flip; for this task only one demo's expected is a rotation+recolor cascade, the other is a single-step recolor. The structural cue isn't visible to the advisor's per-demo rules.

This is a real limitation: **the advisor is per-demo; some tasks need cross-demo reasoning**.

## Findings

### Win: Symbolic-Repair-Advisor closes a real ARC-style gap

Sprint-3's #266 showed 0 % refiner uplift because no borderline-partials existed. Sprint-5's #268 activated the refiner pipeline on real partials but Local-Edit alone couldn't fix them. Sprint-6 demonstrates that the **right** refiner stage (Symbolic-Repair-Advisor + structured candidate enumeration) does close the gap when the diff's signal is rich (palette diff with multiple introduced colors).

### Loss: Per-demo advisor misses cross-demo invariants

Tasks where one demo's transformation is a *subset* of another demo's full transformation (like 0208's "the second demo also rotates") fool the advisor — the failing demo's diff doesn't carry the rotation signal.

### Critical insight: the cascade strategy was the lever

The advisor produces hints (`recolor`, `swap_colors`); Sprint-6's candidate-builder grew them into a 2-step cascade with subset-enumeration over (src, dst) pairs. **Without cascade enumeration, refined_rate was still 0 %.** The advisor alone didn't move the needle; the cascade did.

This validates the architecture: **suggestions are seeds, not solutions**. The wrapper logic that lifts hints into 1- and 2-step structural candidates is where the Phase-2 ROI lives.

## Sprint-7 directive (deduced)

Three concrete next steps, ordered by ROI:

1. **Generalise cascade enumeration** beyond recolor: chains of 2-3 unary primitives (`rotate90 ∘ mirror_horizontal`, `transpose ∘ rotate180`) — one of these probably solves task 0208. Estimated +10-15 PP on hard.
2. **Cross-demo invariant detection** before calling the advisor: detect that all demos share a structural pre-shape (e.g. "all expected outputs are rotations of inputs") and use that to constrain candidates.
3. **Object-level DSL primitives** (Sprint-4's Track B): `largest_component`, `remove_singletons`, etc. — would let Phase-1 solve tasks 0202 + 0207 directly without needing Phase-2. Higher absolute lift but more work.

Recommended Sprint-7 split:
* **Track A (this PR's continuation)**: extend cascade enumeration to non-recolor unary chains. Target: hard success_rate from 25 % to ≥ 37.5 % (3/8).
* **Track B (parallel)**: object-level DSL primitives. Target: hard success_rate independently to ≥ 37.5 % via Phase-1 alone.

## Reproducibility

```bash
git checkout feat/pse-phase2-sprint6-symbolic-repair-live
python -m cognithor.channels.program_synthesis.synthesis.sprint6_symbolic_repair_runner \
    --output /tmp/s6.json --markdown /tmp/s6.md --wall-clock-budget-seconds 5.0
```

Expected output:
```json
{"engine": "phase1_plus_symbolic_repair", "n_tasks": 8, "success_rate": 0.25,
 "refined_rate": 0.125, "refinement_uplift_rate": 1.0, ...}
```

## Production-wiring note

This Sprint-6 PR is a **standalone experimental runner** that bypasses `WiredPhase2Engine`. It validates that the architecture *can* deliver uplift; Sprint-7 should refactor `WiredPhase2Engine` to wire the working strategies (cascade enumeration, advisor-driven candidate generation) as proper refiner stages. The architectural pattern is clear from this PR's results.
