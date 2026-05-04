# Spike — Qwen3-VL on RTX 5090 (32 GB) for Cognithor's video-read pipeline

**Date:** 2026-05-04
**Sprint:** 27 — IDE-Integration (parallel VLM track)
**Owner:** Alexander Söllner
**Status:** Research output for VLM-1 (#132). Decisions land in VLM-2..VLM-4.

---

## TL;DR

The canonical Qwen3-VL release is the 32B variant (not 27B — owner's
"qwen3.6:27b" tag may be a custom Ollama alias or a distilled 27B
snapshot). For a single RTX 5090 (Blackwell, 32 GB VRAM) running
both **image** and **video** input through vLLM, the right pinning is:

> **`Qwen/Qwen3-VL-32B-Instruct-FP8`** (block-size-128 fine-grained FP8),
> launched with `--quantization fp8 --kv-cache-dtype fp8 --max-model-len 32768
> --gpu-memory-utilization 0.93 --enable-prefix-caching`.

This fits in ~31 GB with KV cache headroom for ~32k context, sustains
~190 tok/s text generation, and accepts both `image_url` and
`video_url` content types in the OpenAI-compatible chat schema.

NVFP4 quantization works on Blackwell *technically* (RTX 5090 has FP4
tensor-core instructions) but is more thinly tested for the
Qwen3-VL family than FP8. **FP8 is the default for VLM-2;** NVFP4
stays as an opt-in path with a feature flag.

## Model resolution

| Owner's tag | Resolves to (recommended pin) | Why |
|---|---|---|
| `qwen3.6:27b` | `Qwen/Qwen3-VL-32B-Instruct-FP8` | 32B is the canonical multimodal release; "27B" likely refers to a distilled `mconcat/Qwen3.5-27B-…-NVFP4` text-only variant that does NOT have native vision/video. |
| `qwen3.6:27b-thinking` | `Qwen/Qwen3-VL-32B-Thinking-FP8` | "Thinking" gives the same vision/video stack with explicit chain-of-thought channel. Use only when CoT is needed (more tokens, slightly slower). |

The "27b" alias used in earlier Sprint-10/11 memory probably referred
to the **text-only** Qwen3.5-distilled-27B-NVFP4 used for ARC
reasoning. Vision/video work needs the proper Qwen3-VL-32B series.

Worth confirming in the Cognithor `model_registry.json` whether
"qwen3.6:27b" is currently aliased to a vision-capable model or to
the text-only distilled variant — VLM-2 must explicitly pin the
multimodal SHA before rollout.

## Hardware envelope (RTX 5090)

- VRAM: 32 GB (GDDR7, 28 GT/s)
- Tensor cores: Blackwell sm_120, FP4 + FP8 + BF16 lanes
- Recommendation hierarchy: **FP8 > NVFP4 > AWQ-Int4 > GPTQ-Int4** for VLM-quality preservation per HF Qwen team's own benchmarks.

## Memory budget for Qwen3-VL-32B-Instruct-FP8 at 32k context

| Component | Approx. VRAM | Notes |
|---|---|---|
| Model weights (32B params × 1 byte FP8) | ~16 GB | Qwen FP8 docs: "fine-grained block-128, near-identical to BF16" |
| Vision encoder (ViT-L/14 in fp16) | ~0.6 GB | always loaded, small |
| KV cache (FP8, 32k ctx, batch=1) | ~10–12 GB | scales with concurrent requests; 32k is the default budget for video |
| Frame-feature cache (32 frames × 256 tok × FP16) | ~0.5 GB | per-request, freed after response |
| CUDA + Triton overhead | ~2 GB | constant |
| **Total at concurrency=1** | **~29–31 GB** | leaves ~1–3 GB headroom |

For concurrency > 1 the KV-cache budget eats the headroom — single-
user is safe; for multi-user opt down to `--max-model-len 16384`.

## Video token budget

vLLM's Qwen3-VL pipeline splits a video into N frames, each producing
~256 vision tokens at default patch size (`patch=14`, frame
resolution 448×448).

| Sampling | Frames | Vision tokens | Reserve from `max-model-len` |
|---|---|---|---|
| `fps=3` (short clips ≤ 30 s) | up to 90 | ~23 040 | 23k of 32k → 9k for prompt + reply |
| `num_frames=32` (long videos) | 32 | ~8 192 | 8k of 32k → 24k for prompt + reply (RECOMMENDED default) |
| `num_frames=64` | 64 | ~16 384 | 16k of 32k → 16k for prompt + reply (only with `max-model-len 32k+`) |

The Cognithor v0.92.7 video-input pipeline already established
`fps=3` for short / `num_frames=32` for long, dispatched via ffprobe
metadata. VLM-2 keeps that as default; VLM-3 adds the same dispatch
on the LLMBackend side so the same heuristic works whether the
caller is the v0.92.7 chat path or the new HyperFrames-feeding path.

## vLLM launch profile (Linux WSL2, Ubuntu-24.04, articall user — per memory)

```bash
# RECOMMENDED Qwen3-VL-32B-Instruct-FP8 default
source vllm-env/bin/activate
export HF_HOME=/home/articall/hf_cache
export VLLM_USE_V1=1                 # vLLM 0.6+ engine

vllm serve Qwen/Qwen3-VL-32B-Instruct-FP8 \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.93 \
  --max-model-len 32768 \
  --enable-prefix-caching \
  --num-speculative-tokens 1 \
  --port 8000 \
  --served-model-name qwen3-vl-32b-fp8 \
  --limit-mm-per-prompt '{"image":4,"video":1}'
```

Notes:

- `--limit-mm-per-prompt` caps per-request multi-modal items; matches
  the v0.92.7 contract (single video per turn, up to 4 images).
- `--enable-prefix-caching` reuses the KV cache for repeated
  prompts (common in agent loops).
- `--num-speculative-tokens 1` is the Sprint-16 finding that boots on
  RTX 5090 with the nightly cu130 vLLM image; higher values still
  crash on SM120 as of 2026-04.
- `--served-model-name qwen3-vl-32b-fp8` decouples the served name
  from the HF repo path so the `model_registry.json` alias survives
  a model swap.

## NVFP4 alternative (opt-in)

`Qwen3-VL-32B-Instruct-NVFP4` is published. NVFP4 weights ~12 GB +
FP8 KV cache + the rest gives ~26 GB total — extra 5 GB headroom
that can fund `--max-model-len 65536` for very long videos. But:

- vLLM's NVFP4 path requires the `cu130-nightly` image with the
  `FlashInferCutlassNvFp4LinearKernel` (per v0.92.7 spike notes).
- Throughput is ~10-15 % lower than FP8 on Qwen3-VL specifically
  (Qwen team's benchmark — NVFP4 saves memory not compute).
- Quality regression is < 1 % on standard VLM benchmarks.

VLM-2 ships an opt-in `--vlm-quant nvfp4` flag. Default stays fp8.

## NOT viable on RTX 5090 32 GB

* `Qwen3-VL-72B-Instruct-FP8` — needs ~40 GB; OOM.
* `Qwen3-VL-32B` BF16 (no quant) — needs ~64 GB; OOM.
* `Qwen3-VL-32B-Instruct-AWQ-Int4` — works but Qwen team benchmarks
  show 2-3 % VLM quality regression vs FP8. Use only if FP8 path
  fails on a future driver / kernel revision.

## Cost ledger (TRUST-9) — vision tokens

The existing `CostKind` StrEnum gets a new value:

```python
class CostKind(StrEnum):
    LLM_INPUT_TOKENS = "llm_input_tokens"
    LLM_OUTPUT_TOKENS = "llm_output_tokens"
    VISION_TOKENS = "vision_tokens"     # NEW — tracks image + video frame tokens separately
    EMBEDDING_TOKENS = "embedding_tokens"
    SUBPROCESS_RUNTIME_MS = "subprocess_runtime_ms"
    ...
```

VLM-3 emits `CostEntry(kind=VISION_TOKENS, value_micro_usd=…)` for
every image + video the agent sends; rate is 0 for local vLLM
(operator paid the GPU fixed cost) but the unit-tracking lets the
extension show "this run consumed X vision tokens" so users can see
per-run multimodal cost intuitively.

## Decision

* **Default model:** `Qwen/Qwen3-VL-32B-Instruct-FP8` (block-128 FP8).
* **Default launch:** `--quantization fp8 --kv-cache-dtype fp8
  --max-model-len 32768 --gpu-memory-utilization 0.93
  --enable-prefix-caching --num-speculative-tokens 1`.
* **Default frame sampling:** keep v0.92.7's `fps=3` for short /
  `num_frames=32` for long via ffprobe dispatch.
* **Opt-ins:** `--vlm-quant nvfp4` for memory-constrained / very-long-video
  use; `Qwen3-VL-32B-Thinking-FP8` for explicit-CoT use.
* **TRUST-9:** add `VISION_TOKENS` `CostKind`, wire it through
  `LLMBackend.chat()` for image_url + video_url payloads.

## Sources

- [Qwen/Qwen3-VL-32B-Instruct-FP8 model card — huggingface.co](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct-FP8)
- [Qwen/Qwen3-VL-32B-Thinking-FP8 model card — huggingface.co](https://huggingface.co/Qwen/Qwen3-VL-32B-Thinking-FP8)
- [Qwen3.5 & Qwen3.6 vLLM recipes — docs.vllm.ai](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
- [vllm-qwen3.5-nvfp4-sm120 — github.com/aliez-ren](https://github.com/aliez-ren/vllm-qwen3.5-nvfp4-sm120)
- [Optimizing Qwen3 Coder for RTX 5090 — cloudrift.ai](https://www.cloudrift.ai/blog/optimizing-qwen3-coder-rtx5090-pro6000)
- Cognithor v0.92.7 video-input spike findings (`docs/superpowers/spikes/2026-04-23-video-input-vllm-spike-findings.md`)
- Cognithor Sprint-16 vLLM defaults memo (PSE Phase-A validation)
