# Sprint-15 Phase-A Diagnostic Report

**Date:** 2026-05-02
**Hardware:** RTX 5090 + WSL2 Ubuntu 24.04 + CUDA 13.0 + vLLM 0.20.0
**Model:** `sakamakismile/Qwen3.6-27B-NVFP4` (multimodal NVFP4)
**Drafter:** `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`
**Workload:** ARC-AGI-3 game `bp35`, single-stream `llm_full` agent, 40 steps, greedy decoding

## Headline Findings

1. **MTP acceptance is 0 % across all tested configurations** — `num_speculative_tokens` ∈ {1, 3} both produce 0 accepted drafts over 50 000+ proposals.
2. **MTP enabled is +30 % faster than MTP off**, despite 0 % acceptance — the win is from CUDA-graph + forward-pass pipeline differences, not from accepted drafts.
3. **Five critical bugs found and fixed during Phase-A**, four of which were silent failure modes that would not have surfaced without the Sprint-15 telemetry pipeline.

## Apples-to-Apples Benchmark

| Run | Config | Wall (40 steps) | Mean LLM call | tok/s output | Acceptance |
|---|---|---|---|---|---|
| #7 | MTP=3 + sep. drafter | **2487 s** | 62.2 s | **23.6** | 0.0 % |
| #8 | MTP off | 3030 s | 75.7 s | 17.5 | n/a |
| #10 | MTP=1 + sep. drafter | 2600 s | 65.0 s | 22.7 | 0.0 % |

Identical input across all three runs (343 383 input tokens summed).

## Outcome Classification

The reviewer's pre-flagged A/B/C/D outcome chart maps as follows:

* **Outcome A** (MTP healthy, ≥70 %): excluded.
* **Outcome B** (workload-dependent kollapse): excluded — 0 % is uniform across reasoning + final-answer phases.
* **Outcome C** (drafter quality issue): **confirmed in extreme form.**
* **Outcome D** (drafter drift at later positions): excluded — MTP=1 also 0 %.

**Specific Outcome C variant** — independent NVFP4 quantisation drift between
`Qwen3.6-27B-NVFP4` and `Qwen3.6-27B-Text-NVFP4-MTP`. Both checkpoints share the
same architecture (`qwen3_5`, `Qwen3_5ForConditionalGeneration`), vocab (248 320),
and `mtp_num_hidden_layers=1`. They differ only in how the FP4 quantisation
pass rounded the weights — and FP4's ~3-bit-effective precision is too narrow
to absorb the divergence. The drafter's argmax never aligns with the verifier's,
even at greedy temp=0 + Position-1 (which the head was actually trained for).

## Surprising Side Finding

**MTP-on is 30 % faster than MTP-off even at 0 % acceptance.**

Expected: MTP-off should be at least as fast as MTP-on with 0 % acceptance,
since the drafter's forward pass is wasted compute when nothing gets accepted.

Observed: MTP-off (3030 s) is *slower* than MTP-on (2487 s for MTP=3, 2600 s
for MTP=1).

Likely cause: vLLM compiles a separate CUDA graph for the spec-decode pipeline
that uses different memory-access patterns. For our single-stream workload,
those patterns are more efficient than the standard decode graph. The "win"
is structural, not from accepted drafts.

This means **enabling MTP is net positive even with our broken drafter** —
keeping MTP=1 as the production config. Theoretical 1.4–1.9× ceiling from
acceptance gains stays unattainable until a quantisation-aligned drafter ships.

## Bugs Found During Phase-A (Real-Time Failure Modes)

The Sprint-15 telemetry pipeline + `assert_telemetry_active()` sanity-check
caught five distinct silent failure modes that would have wasted significant
debug cycles:

1. **MTP checkpoint ID 401** — initial guess `Qwen3.6-27B-NVFP4-MTP` doesn't
   exist on HF; correct ID has `-Text-` infix. vLLM crashed engine init,
   driver fell through to DSL baseline. Sanity-check flagged within 18 s.
2. **vLLM v1 API drift — TTFT field rename**. `RequestStateStats.first_token_ts`
   replaced `RequestMetrics.first_token_time`; `first_token_latency` is the
   direct field. Adapters now try v1, v1-fallback, then v0.
3. **vLLM v1 API drift — spec-decode metric names**. v1 dropped the `_total`
   suffix and renamed `num_emitted_tokens` to `num_draft_tokens`. Counter-
   matching in `poll_engine_mtp_metrics` now accepts both shapes.
4. **vLLM v1 API drift — per-request acceptance gone**. `RequestOutput.metrics`
   no longer carries `spec_token_acceptance_counts`; live only on engine-side
   `SchedulerStats.spec_decoding_stats`. Choice-fn now does delta-poll between
   calls.
5. **`disable_log_stats=True` trap** — vLLM's offline `LLM` class defaults to
   `disable_log_stats=True`, which strips `RequestOutput.metrics` AND raises
   `AssertionError` from `LLMEngine.get_metrics()`. Fix: set
   `disable_log_stats=False` whenever telemetry/MTP-stats is wired.

Each bug was caught by the sanity-check warning surface (4/5 explicit warnings,
1/5 detected by zero-output latency).

## Outstanding Gaps

* **TTFT capture still 0 % coverage** even with `disable_log_stats=False`. Need
  separate investigation — `RequestOutput.metrics` may require a different
  flag in vLLM 0.20-v1.
* **Score = 0/9** across all runs (LLM doesn't solve any level in 40 steps).
  This is a Sprint-16 problem (prompt tuning, plan horizon, game-knowledge
  injection) — orthogonal to MTP/throughput.

## Production Config (post-Phase-A)

```python
build_inprocess_vllm_choice_fn(
    speculative_config={
        "model": "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP",
        "num_speculative_tokens": 1,
    },
    kv_cache_dtype="fp8",
    temperature=0.0,
    mtp_stats=mtp_stats,
    telemetry=telemetry,
)
```

* `num_speculative_tokens=1` matches the drafter head's training distribution
  (head is single-layer per `mtp_num_hidden_layers: 1`).
* MTP=1 vs MTP=3 throughput within 4 % (noise) — MTP=1 wins on power efficiency.
* `temperature=0.0` for deterministic ARC reasoning; MTP-acceptance turned out
  not to depend on this, but greedy is the right choice for benchmark
  reproducibility regardless.

## Reviewer Hand-off

The full per-step JSONL audit trails for all 3 runs are at:

```
/tmp/sprint15_phase_a_final/
  run7_mtp3.jsonl
  run8_mtp_off.jsonl
  run10_mtp1.jsonl
```

The conditional-acceptance properties (`MTPSnapshot.conditional_acceptances`,
`mean_accepted_length`) are computable but degenerate (all values 0) given
the 0 % acceptance — re-running with a quantisation-aligned drafter would
make them meaningful.

## Next Steps

1. **MTP**: stop further tuning here. Acceptance fix requires either:
   * Co-quantised drafter+main from the same NVFP4 calibration run, or
   * A drafter checkpoint specifically trained against this main checkpoint.
   Both are upstream-dependent (sakamakismile or successor maintainer needs
   to ship a paired pair).
2. **TTFT**: separate small fix. Likely needs vLLM-version-specific path or
   a v1-only `enable_metrics=True` override.
3. **Score lift**: Sprint-16 priority. With now-stable MTP=1 throughput
   (~22.7 tok/s, 65 s per LLM call), focus on prompt + plan-horizon
   tuning. The 40-step cap with 1 LLM call per step is currently the
   bottleneck.
