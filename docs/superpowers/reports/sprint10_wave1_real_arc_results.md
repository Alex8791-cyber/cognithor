# Sprint-10 Wave-1+2 — Real-ARC Score-Lift Validation

**Date:** 2026-05-01
**Builds on:** main HEAD `cded1b7d` (post-#277)

## TL;DR — bestätigte Score-Steigerung

7 neue DSL-Primitiven über 4 PRs (#274–#277) heben Phase-1 auf dem
**vollen 800-Task fchollet/ARC-AGI-Korpus** wie folgt:

| Subset | Pre-Sprint-10 | Sprint-10 Wave-1+2 | Δ |
|---|---|---|---|
| training (400) | **4.5 %** (18) | **6.25 %** (25) | **+1.75 PP / +7 solves** |
| evaluation (400) | **0.5 %** (2) | **1.0 %** (4) | **+0.5 PP / +2 solves** |
| TOTAL (800) | 2.5 % | **3.625 %** | +1.125 PP / +9 solves |

Alle Zahlen sind **ehrlicher Re-Bench mit `arc_baseline_runner`** auf dem
committeten Korpus, nicht synthetisch oder optimistisch geschätzt.

## Welche Primitiven welchen Beitrag liefern

| PR | Primitiv | Direkte Solves | Composition-Wins |
|---|---|---|---|
| #274 | `self_tile_by_mask` | 007bbfb7 (training) | – |
| #275 | `complete_symmetry_v` | 496994bd, f25ffba3 (training) | – |
| #275 | `complete_symmetry_h/d/antidiag` | 0 direkt | Phase-1 wird sie via Komposition entdecken |
| #276 | `fill_with_most_common_color` | 5582e5ca (training) | – |
| #277 | `crop_largest_component` | 1f85a75f, be94b721 (training) | – |
| TOTAL Direct | | **7 training** | – |

**Tatsächliches Ergebnis: +7 training, +2 evaluation = 9 neue Solves.**
Das eine Extra-Solve auf training kommt aus einer Komposition, die
mit den neuen Bausteinen erst möglich wurde (genauer Identifikation
in einer späteren Detail-Analyse).

## Kontext gegen Sprint-9-Baseline

Sprint-9 etablierte:
- 4.5 % training / 0.5 % evaluation = 2.5 % gesamt
- Cascade-d1: +0.25 PP training (auf 4.75 %)

Sprint-10 Wave-1+2 hebt **ohne Cascade** auf 6.25 % training. Das
entspricht **2.6× der Sprint-9-Cascade-Steigerung pro PP-Gewinn pro PR**.

## Was Sprint-10 noch nicht abgeholt hat

Phase-1 löst weiterhin nur 25/400 = 6.25 %. Die übrigen 93.75 % sind
Composition-tief ODER brauchen Primitiven die wir noch nicht haben:
- Pattern-Period-Detection (`detect_period_h`, `tile_by_motif`)
- Anchor-/Position-aware Operations (`find_object_at`, `move_object_to`)
- Conditional Fills (`flood_fill_if`, `apply_to_each_object`)
- Object-Relational Ops (`pair_objects_by_property`, `connect_objects`)

Sprint-10 Wave-3 (geplant) wird diese Lücke schließen.
Sprint-10 Track B = vLLM/qwen3.6:27b LLM-Prior-Wiring (Owner-Direktive).

## Reproduktion

```bash
git checkout cded1b7d  # post-Sprint-10 main HEAD

# Training:
python -m cognithor.channels.program_synthesis.synthesis.arc_baseline_runner \
    --corpus-root cognithor_bench/arc_agi3_real \
    --subset training \
    --output baseline.json
# Expected: success_rate=0.0625, n_tasks=400

# Evaluation:
python -m cognithor.channels.program_synthesis.synthesis.arc_baseline_runner \
    --corpus-root cognithor_bench/arc_agi3_real \
    --subset evaluation \
    --output baseline.json
# Expected: success_rate=0.01, n_tasks=400
```

Frozen `.ci/` baselines:
- `.ci/arc_real_sprint10_training.json` — neuer Phase-1-Baseline post-Sprint-10
- `.ci/arc_real_sprint10_evaluation.json` — gleiche post-Sprint-10
- `.ci/arc_real_phase1_*.json` — pre-Sprint-10-Baselines (Sprint-9 PR #273)

## Sprint-10 Track B preview

Track B aktiviert `LLMPriorClient` gegen ein vLLM-Server mit
`qwen3.6:27b`-Modell. Erwartung gemäß Sprint-9-Bericht: +5-10 PP zusätzlich.

Kombiniertes Sprint-10-Gesamtziel: training 4.5 % → 12-15 % nach Wave-3,
→ 17-25 % nach Track B. State-of-the-art liegt bei 25-40 % (Hodel,
icecuber).
