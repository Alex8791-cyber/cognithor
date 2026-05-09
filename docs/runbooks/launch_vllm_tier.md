# Launch vLLM Tier (Hardware-Aware Runtime)

`scripts/launch_vllm_tier.py` is the operator-side companion to `cognithor doctor --apply-recommendation` — once the apply-engine has written the sidecar (`~/.cognithor/.hardware_aware.json`) for a vLLM-backed tier, this script reads it and starts vLLM with the safe defaults discovered during the 2026-05-09 NVFP4 smoke-test.

## Why this script exists

vLLM 0.20 introduced six independent regression-traps for the NVFP4 + Blackwell + WSL2 combo. Each trap silently breaks an otherwise-correct tier-apply. The script bundles all six fixes; running `python -m vllm.entrypoints.openai.api_server …` directly without these is unlikely to produce a working server.

| # | Trap | Fix the script applies |
|---|---|---|
| 1 | HF Hub HEAD-request fails when a community-uploaded model lacks `preprocessor_config.json` upstream — even if the local snapshot is intact | `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`, plus auto-resolves the HF id to the cached snapshot path |
| 2 | flashinfer 0.6.x reads `nvcc --version`; if the system nvcc is < 12.9 it refuses sm_120 even though PyTorch is built against CUDA 13 | `--force-cuda-arch 12.0f` env override (opt-in flag) |
| 3 | `ninja` PATH-lookup hits a non-executable match in WSL `/mnt/c/...` interop dirs → `PermissionError` | PATH cleanup: prepends venv `bin/`, drops Windows-mnt entries |
| 4 | `--cpu-offload-gb` triggers a uvloop deadlock during the first incoming request on WSL2 + vLLM 0.20; the API server stays bound but stops accepting connections | Off by default; explicit `--cpu-offload N` to enable |
| 5 | `--num-speculative-tokens` was deprecated in vLLM 0.20+ | Reads `speculative_config` (dict) from sidecar `vllm_extras`, emits as `--speculative-config '{…}'` JSON |
| 6 | `--host 127.0.0.1` makes the server unreachable from the Windows host through WSL2 port-forward | Defaults to `0.0.0.0`; pass `--host 127.0.0.1` to keep it WSL-internal |

## Usage

```bash
# Resolve everything from the sidecar
python scripts/launch_vllm_tier.py

# Explicit model snapshot path + port
python scripts/launch_vllm_tier.py \
    --model ~/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B-FP8/snapshots/abc123 \
    --port 8000

# Blackwell + nvcc 12.0 (most current Ubuntu WSL stacks)
python scripts/launch_vllm_tier.py --force-cuda-arch 12.0f

# Dry-run: print the constructed command + env, exec nothing
python scripts/launch_vllm_tier.py --print-only

# Override the venv path (default ~/vllm-env)
python scripts/launch_vllm_tier.py --venv /opt/vllm-env

# Pass through additional vLLM CLI args
python scripts/launch_vllm_tier.py -- --tensor-parallel-size 2 --quantization fp8
```

## Sidecar fields the script reads

```jsonc
{
  "vllm": {
    "gpu_memory_utilization": 0.94,
    "enforce_eager": true,
    "max_model_len": 16384
  },
  "vllm_extras": {
    "speculative_config": {"num_speculative_tokens": 1},
    "enable_prefix_caching": true
  },
  "model_set_extras": {
    "planner": {"name": "Qwen/Qwen3.6-27B-FP8"}
  }
}
```

CLI flags override sidecar values. If both the sidecar and CLI lack a value, vLLM defaults apply.

## Smoke-test after launch

`scripts/smoke_vllm_backend.py` is the existing smoke-test runner. Once `launch_vllm_tier.py` reports `Application startup complete`:

```bash
# from another shell — uses the default localhost:8000
python scripts/smoke_vllm_backend.py
```

## When something still fails

- `OSError: Can't load image processor for 'org/repo'` — the model repo is missing `preprocessor_config.json` upstream. Copy it from a sibling repo (e.g. official Qwen FP8 release) into the local snapshot, then `mv` the corresponding `.no_exist/<rev>/preprocessor_config.json` marker out of the way.
- `RuntimeError: No supported CUDA architectures found for major versions [12]` — pass `--force-cuda-arch 12.0f`.
- `PermissionError: [Errno 13] Permission denied: 'ninja'` — script already cleans PATH; if it still fails, your venv `bin/ninja` may be missing or non-executable. `chmod +x ~/vllm-env/bin/ninja`.
- Server reaches `Application startup complete` but `curl` hangs — check the engine log for stalled JIT compilation; first-request fp4_gemm autotuning can take 1-3 min on RTX 5090. Increase HTTP client timeout.
- `--num-speculative-tokens unrecognized` — the sidecar still has the legacy field. Fix in the manifest YAML (`speculative_config: {num_speculative_tokens: N}`) and rerun `cognithor doctor --apply-recommendation` to refresh the sidecar.

## Related files

- `manifest/v2/tiers.yaml` — tier definitions; `enterprise-vllm-nvfp4-blackwell` is the NVFP4 tier this script was designed against.
- `src/cognithor/system/apply_engine.py` — emits the sidecar that this script reads.
- `scripts/smoke_vllm_backend.py` — companion smoke-test that hits `/v1/models` and `/v1/chat/completions`.
