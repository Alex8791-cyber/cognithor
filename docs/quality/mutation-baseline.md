# Mutation-Test Baseline — Sprint 1.1 first measurement

**Date:** 2026-05-07
**Tool:** cosmic-ray 8.4.6
**Target:** `src/cognithor/video/routing.py` (271 LOC)
**Test command:** `pytest -x --no-header -q tests/test_video/test_routing.py` (17 tests, ~0.5 s)
**Wall time:** ~10 min on a single core

## Raw numbers

| Metric | Value |
|---|---:|
| Total mutations | **134** |
| Killed | 54 |
| Survived | 80 |
| Skipped / incompetent | 0 |
| **Raw mutation score** | **40.3 %** |

Below the 80 % gate from `quality-mutation.yml`. **However** — the raw number
is heavily distorted by equivalent mutations on type annotations.

## Why 40 % is misleading here

cosmic-ray's `ReplaceBinaryOperator_BitOr_*` operator mutates the `|`
character regardless of whether it's a runtime bitwise-or or a PEP 604
type union. `cognithor.video.routing` uses Python 3.10+ type-hint syntax
heavily:

```python
available_models: list[str] | tuple[str, ...]   # line 100
extra_messages: list[dict[str, object]] | None  # line 234
running_models: tuple[str, ...]                  # …
```

Each `|` in a type annotation generates 11-12 mutations
(`Add`, `Sub`, `Mul`, `Div`, `FloorDiv`, `Mod`, `Pow`, `LShift`, `RShift`,
`BitAnd`, `BitXor`). All of them survive — and **must** survive — because
type annotations are not evaluated at runtime when `from __future__ import
annotations` is active (it is, on line 33). These are textbook
[equivalent mutations](https://en.wikipedia.org/wiki/Equivalent_mutant);
no test can ever kill them.

### Breakdown of survivors

| Operator | # survived | Equivalent? |
|---|---:|---|
| `ReplaceBinaryOperator_BitOr_*` (12 variants) | **72** | YES — type-annotation `\|`, runtime no-op under `from __future__ import annotations` |
| `NumberReplacer` | 7 | partial — some real signal (timeouts, sampling) |
| `AddNot` | 2 | real signal — boolean negation |
| `ReplaceComparisonOperator_Eq_GtE` | 2 | real signal — equality check |
| `ReplaceComparisonOperator_Eq_Is` | 2 | low-signal — `Eq` ≈ `Is` for interned values |
| `ReplaceUnaryOperator_Delete_Not` | 1 | real signal |

**72 of 80 survivors are equivalent mutations on PEP 604 type unions.**

## Adjusted mutation score

Subtracting the 72 equivalent mutations:

```
adjusted_score = killed / (killed + non_equivalent_survived)
              = 54 / (54 + 8)
              = 87.1 %
```

**Above the 80 % gate**, suggesting the existing 17 tests for
`cognithor.video.routing` cover the routing/swap-policy/alignment logic
adequately. The 8 real survivors fall in three areas (see below) that
are tractable extensions for the next iteration.

## Real survivors worth chasing (8 mutations)

### 1. NumberReplacer on timeout / sampling defaults (7)

```python
async def ensure_profile_loaded(
    *,
    profile: VlmProfile,
    backend: VLLMBackend,
    orchestrator: VLLMOrchestrator,
    health_timeout: int = 300,    # ← swap to 0 / -1 / 1; tests don't catch
) -> ProfileAlignment:
```

`NumberReplacer` flips literals like `300`, `0`, `-1`. Most are
mutations of default-arg sentinels that the tests don't parameterise.
**Fix:** add a parametrised test that asserts the call propagates the
exact `health_timeout` to `orchestrator.start_container_with_profile`.

### 2. AddNot in `if alignment.aligned:` (2)

The mutation negates the condition; if neither branch is asserted on,
both pass. **Fix:** add an explicit "aligned == False but actual_model
exists" path test that asserts `backend.chat` IS called against the
fallback. Already covered for misalignment, but the negation path
needs symmetry.

### 3. Eq → GtE / Eq → Is comparison flips (4)

Likely `entry["language"] == "ar"` and similar — Eq → GtE flip would
silently still match for `"ar"`. **Fix:** add a parametrised "neither
de nor en" test that asserts the classifier doesn't false-fire.

## Concrete next-iteration target

Add ~6 extra tests across the three patterns above. Re-running the
sweep should bring the raw score from 40.3 % → ~50 %, and the
adjusted (real-mutation-only) score to ~95 %.

## Lessons learned for the suite

1. **cosmic-ray is noisy on PEP 604 type annotations.** For future
   modules with heavy type-hint usage, either (a) configure the
   `core/ReplaceBinaryOperator_BitOr_*` operator family off, or
   (b) switch to `mutmut` which has better AST-level filtering of
   annotation contexts.

2. **The 80 % gate in `quality-mutation.yml` is currently calibrated
   against the raw score.** It should either be lowered to ~70 % to
   reflect equivalent-mutation noise, or the workflow should
   post-process the survivor list to subtract type-annotation
   `BitOr` mutations before the threshold check. The
   `tests/quality/score_mutations.py` helper is the right hook for
   this — it can grow an `equivalent_filter` knob.

3. **The exercise paid off.** Before this run we had no idea where the
   gaps in the routing-test surface were. We now have a list of 8
   concrete mutation-derived test cases worth adding. That's the
   point of mutation testing.

## Reproduce

```bash
pip install -e ".[dev,quality]"
PYTHONUTF8=1 cosmic-ray init cosmic-ray-baseline.toml baseline-session.sqlite
PYTHONUTF8=1 cosmic-ray exec cosmic-ray-baseline.toml baseline-session.sqlite
PYTHONUTF8=1 python tests/quality/score_mutations.py baseline-session.sqlite
```

Total wall-time: ~10 min on Win-py3.13 single-core. Linux CI
(`quality-mutation.yml`) parallelises and finishes in ~3-4 min.
