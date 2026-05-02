"""Sprint-15 TTFT probe — inspect what vLLM 0.20-v1 actually attaches
to ``RequestOutput.metrics`` when ``disable_log_stats=False``.

Phase-A telemetry showed ttft_count=0/40 across runs that explicitly
set ``disable_log_stats=False``. Either the field isn't named what
we read, or ``metrics`` is None despite the flag, or v1 needs a
separate enable-token. This script answers that without burning
another 6-min Phase-A loop.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("HF_HOME", "/home/articall/hf_cache")

# Match Phase-A driver's CUDA env so flashinfer JIT can find SM 12.x
# kernels — without this the engine init crashes with
# "No supported CUDA architectures found for major versions [12]".
_CUDA_HOME = "/usr/local/cuda-13.0"
if os.path.isdir(_CUDA_HOME):
    os.environ.setdefault("CUDA_HOME", _CUDA_HOME)
    os.environ["PATH"] = f"{_CUDA_HOME}/bin:{os.environ.get('PATH', '')}"


def main() -> None:
    from vllm import LLM, SamplingParams

    # Match Phase-A exact config so flashinfer reuses cached JIT kernels.
    print("[probe] init engine — Phase-A config, disable_log_stats=False …")
    t0 = time.monotonic()
    llm = LLM(
        model="sakamakismile/Qwen3.6-27B-NVFP4",
        max_model_len=32768,
        gpu_memory_utilization=0.92,
        max_num_seqs=64,
        enforce_eager=False,
        dtype="auto",
        kv_cache_dtype="fp8",
        disable_log_stats=False,
    )
    print(f"[probe] engine init done in {time.monotonic() - t0:.1f}s")

    sampling = SamplingParams(temperature=0.0, max_tokens=64)
    print("[probe] running one chat call …")
    t1 = time.monotonic()
    outs = llm.chat(
        messages=[{"role": "user", "content": "Say hello in three words."}],
        sampling_params=sampling,
    )
    wall = time.monotonic() - t1
    print(f"[probe] call wall = {wall:.2f}s")

    req = outs[0]
    print(f"[probe] type(req) = {type(req).__name__}")
    print(f"[probe] req.metrics = {req.metrics!r}")
    if req.metrics is not None:
        print(f"[probe] type(req.metrics) = {type(req.metrics).__name__}")
        for attr in dir(req.metrics):
            if attr.startswith("_"):
                continue
            try:
                val = getattr(req.metrics, attr)
                if not callable(val):
                    print(f"  metrics.{attr} = {val!r}")
            except Exception as exc:
                print(f"  metrics.{attr} = <err: {exc}>")
    else:
        print("[probe] metrics is None — checking llm_engine state")
        eng = getattr(llm, "llm_engine", None)
        print(f"[probe] llm.llm_engine.log_stats = {getattr(eng, 'log_stats', None)}")
        op = getattr(eng, "output_processor", None)
        print(f"[probe] output_processor.log_stats = {getattr(op, 'log_stats', None)}")

    print(f"[probe] req.outputs[0].text = {req.outputs[0].text[:100]!r}")
    print(f"[probe] req.outputs[0].finish_reason = {req.outputs[0].finish_reason!r}")
    print(f"[probe] len(req.outputs[0].token_ids) = {len(req.outputs[0].token_ids)}")
    print(f"[probe] req.num_cached_tokens = {req.num_cached_tokens}")


if __name__ == "__main__":
    main()
