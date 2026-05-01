# Sprint-10 Track B — vLLM / qwen3.6:27b Deployment Guide

**Date:** 2026-05-01
**Status:** wiring infrastructure-ready; runtime validation requires hardware

## TL;DR

Sprint-10 Track B wires the **production LLM-Prior over a vLLM
OpenAI-compat endpoint** into the PSE benchmark runner. The
`DualPriorMixer` combines an `LLMPriorClient(VLLMBackend)` with a
`UniformSymbolicPrior` and feeds the live α to the
`WiredPhase2Engine`'s search loop.

**The wiring is fully constructed and tested at module level.** What's
missing is a *running* vLLM server with `Qwen/Qwen3.6-27B-Instruct`
loaded — that requires a 32+ GB GPU (RTX 5090 / A100 / H100) and is
out of scope for the autonomous-mode session.

## CLI surface

```
python -m cognithor.channels.program_synthesis.synthesis.benchmark_runner \
    --arc-corpus cognithor_bench/arc_agi3_real \
    --arc-subset training \
    --phase2 \
    --llm-prior \
    --llm-base-url http://localhost:8000/v1 \
    --llm-model Qwen/Qwen3.6-27B-Instruct \
    --output sprint10_track_b_training.json
```

Flags introduced this PR:

- `--llm-prior` — opt-in switch. Without it, `--phase2` runs without an LLM (cold-start α, identical to pre-Track-B behaviour).
- `--llm-base-url` — vLLM endpoint URL. Default `http://localhost:8000/v1`.
- `--llm-model` — model name as registered with vLLM. Default `Qwen/Qwen3.6-27B-Instruct`.

## Server-side deployment

### Hardware

- **Recommended**: NVIDIA RTX 5090 (32 GB VRAM), Linux, CUDA 12+
- **Alternative**: A100 / H100 / multi-GPU
- **Model VRAM** at FP16: ~54 GB (won't fit single 5090). Use a quantised build:
  - **Q5_K_M (default)**: ~22 GB, strongly recommended
  - **Q4_K_M (fallback)**: ~17 GB, smaller VRAM headroom

### Install vLLM

```bash
pip install vllm==0.6.3
# or build from source for cutting-edge model support
```

### Launch the server

```bash
# Q5_K_M quantisation (preferred):
vllm serve Qwen/Qwen3.6-27B-Instruct \
    --quantization awq \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92

# Or Q4_K_M fallback for tighter VRAM:
vllm serve Qwen/Qwen3.6-27B-Instruct-AWQ \
    --quantization awq \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85
```

Verify:

```bash
curl http://localhost:8000/v1/models
# expected: list with Qwen/Qwen3.6-27B-Instruct
curl http://localhost:8000/health
# expected: 200 OK
```

## Two-stage prompting (spec §4.7)

The `LLMPriorClient` runs each task through:

1. **Stage 1 — free-form CoT** at `temperature=0.7` to explore which DSL primitives might apply.
2. **Stage 2 — constrained JSON** at `temperature=0.1` to extract a structured prior + α-entropy hint, with one retry on parse failure.

The resulting `LLMPrior.primitive_scores` weights the candidate enumeration in the search engine. The `α-entropy hint` and the symbolic prior's confidence are mixed by `mix_alpha()` per spec §4.5.

Per-call wall-clock cap: 8 s (`Phase2Config.llm_call_timeout_seconds`). On timeout / parse failure, the engine falls back to cold-start α (telemetry event `wired.alpha_fallback`).

## Expected score-lift

Sprint-9's reality-check estimated **+5-10 PP** from LLM-Prior wiring on the real fchollet/ARC-AGI training subset.

Sprint-10 DSL (Wave-1+2+3+4) measured: **4.5 % → 7.5 %** (+3.0 PP, 30/400 training).

Track B target: **7.5 % → 12-17 %** training (rough order-of-magnitude per Sprint-9 §"Sprint-10+ priorities" estimate).

## Validation flow once vLLM is running

```bash
# Step 1: Without LLM-Prior (cold-start α) — should match the
# committed .ci/arc_real_sprint10_*.json baselines.
python -m cognithor.channels.program_synthesis.synthesis.benchmark_runner \
    --arc-corpus cognithor_bench/arc_agi3_real \
    --arc-subset training \
    --phase2 \
    --output baseline_no_llm.json

# Step 2: With LLM-Prior — measure the lift.
python -m cognithor.channels.program_synthesis.synthesis.benchmark_runner \
    --arc-corpus cognithor_bench/arc_agi3_real \
    --arc-subset training \
    --phase2 \
    --llm-prior \
    --output baseline_with_llm.json

# Step 3: diff success_rate fields.
```

A real lift (+3 PP or more) on training validates the wiring; a flat
or negative result indicates the prompts need spec-§4.7-tuning
against the chosen quantisation.

## What this PR does NOT do

- **Run the actual vLLM server.** The Track B PR is wiring + flags; the deployment is operator-side.
- **Tune the prompts against quantisation drift.** Q4_K_M may degrade Stage-2 JSON parsing — measure first, then adjust if needed.
- **Validate end-to-end score lift.** That happens in Sprint-11 once a vLLM instance is provisioned, OR can be run manually with this PR's flags and the documented deployment above.

## Code references

- Wiring helper: `_build_dual_prior_stack(base_url, model_name)` in `synthesis/benchmark_runner.py`
- Engine integration: `_build_phase2_engine(dual_prior=...)` in same module
- LLM client: `phase2/llm_prior.py::LLMPriorClient` (Sprint-1)
- Mixer: `phase2/dual_prior.py::DualPriorMixer` (Sprint-1)
- Backend: `core/vllm_backend.py::VLLMBackend`
- Config defaults: `phase2/config.py::Phase2Config.llm_*`
- Wiring tests: `tests/test_channels/test_program_synthesis/synthesis/test_track_b_llm_prior_wiring.py`
