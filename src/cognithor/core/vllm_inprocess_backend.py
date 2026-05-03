# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""In-process vLLM backend — Sprint-12 (validated 2026-05-02 on RTX 5090).

Wraps ``vllm.LLM`` directly in the same Python process, **bypassing
HTTP entirely**. This was added after the WSL2 mirrored-networking
quirk made the OpenAI-compat HTTP server unreachable from inside the
same WSL distro: uvicorn binds the listen socket, but ``accept()``
deadlocks. The model + GPU + engine are healthy; only the network
layer is broken.

The in-process backend solves it by skipping HTTP completely:
``vllm.LLM(model=...)`` loads the engine in the host process, and
``llm.chat(...)`` calls it directly. Same compute, no network.

Defaults match Phase2Config (Sprint-12 Owner-validated):
- model: ``sakamakismile/Qwen3.6-27B-NVFP4`` (Blackwell native FP4)
- max_model_len: 32768, max_num_seqs: 64 (Mamba cache-block ceiling)
- gpu_memory_utilization: 0.92, enforce_eager: False (CUDA graphs)

Lazy import of ``vllm`` so this module loads cleanly on Windows / hosts
without a GPU. The engine only spins up on first ``chat()`` call.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any

from cognithor.core.llm_backend import (
    ChatResponse,
    EmbedResponse,
    LLMBackend,
    LLMBackendError,
    LLMBackendType,
)
from cognithor.utils.logging import get_logger

log = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class VLLMInProcessBackend(LLMBackend):
    """vLLM running in-process (no HTTP).

    Validated 2026-05-02 on RTX 5090 + WSL2 Ubuntu 24.04 + CUDA 13.0:
    50 tok/s output / 240 tok/s prefill at sustained 370-400 W.
    """

    def __init__(
        self,
        *,
        model: str = "sakamakismile/Qwen3.6-27B-NVFP4",
        max_model_len: int = 32768,
        max_num_seqs: int = 64,
        gpu_memory_utilization: float = 0.92,
        enforce_eager: bool = False,
        cuda_home: str = "/usr/local/cuda-13.0",
        dtype: str = "auto",
        hf_home: str | None = None,
    ) -> None:
        self._model_name = model
        self._engine_kwargs = {
            "model": model,
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enforce_eager": enforce_eager,
            "dtype": dtype,
        }
        # Set env vars before vllm imports torch/cuda. The cuda_home
        # path must be a real directory containing bin/nvcc; FlashInfer's
        # NVFP4 JIT compile breaks without it on Blackwell.
        if cuda_home and os.path.isdir(cuda_home):
            os.environ.setdefault("CUDA_HOME", cuda_home)
            os.environ["PATH"] = f"{cuda_home}/bin:{os.environ.get('PATH', '')}"
        if hf_home:
            os.environ.setdefault("HF_HOME", hf_home)
        self._llm: Any = None  # lazy-init on first chat()
        self._lock = asyncio.Lock()

    @property
    def backend_type(self) -> LLMBackendType:
        return LLMBackendType.VLLM_INPROCESS

    async def _ensure_engine(self) -> Any:
        if self._llm is not None:
            return self._llm
        async with self._lock:
            if self._llm is not None:
                return self._llm
            try:
                # Heavy import — keep it lazy so the module loads on
                # hosts without vLLM / GPU.
                from vllm import LLM  # type: ignore[import-not-found]
            except ImportError as exc:
                raise LLMBackendError(
                    "vllm is not installed. Run inside a venv with "
                    "`pip install vllm` (Linux + CUDA-capable GPU only)."
                ) from exc
            try:
                # vllm.LLM blocks while loading the model + warming up.
                # Run it in a worker thread to keep the asyncio loop
                # responsive (otherwise even health-pings stall).
                self._llm = await asyncio.to_thread(LLM, **self._engine_kwargs)
            except Exception as exc:
                raise LLMBackendError(
                    f"vLLM engine init failed: {type(exc).__name__}: {exc}"
                ) from exc
        return self._llm

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        format_json: bool = False,
        num_ctx: int | None = None,
    ) -> ChatResponse:
        del tools, format_json  # vLLM's chat method handles formatting natively
        # Sprint-23: in-process vLLM honours the engine-launch-time
        # ``max_model_len``. Per-request ``num_ctx`` is a soft hint at
        # this layer — logged for diagnostics, no change in behaviour.
        # Operators must launch this backend with a window large enough
        # for the largest profile they expect to use.
        if num_ctx is not None:
            log.debug("vllm_inprocess_num_ctx_hint", model=model, num_ctx=int(num_ctx))
        engine = await self._ensure_engine()
        # The engine is bound to a single model loaded at init time.
        # If the caller asks for a different model, fail loudly rather
        # than silently dispatching to the wrong weights.
        if model and model != self._model_name:
            raise LLMBackendError(
                f"VLLMInProcessBackend is loaded with {self._model_name!r}, "
                f"caller requested {model!r}. Construct a new backend per model."
            )
        from vllm import SamplingParams  # type: ignore[import-not-found]

        sampling = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=2048)
        outputs = await asyncio.to_thread(engine.chat, messages, sampling)
        if not outputs:
            raise LLMBackendError("vLLM returned empty output list")
        text = outputs[0].outputs[0].text if outputs[0].outputs else ""
        return ChatResponse(
            content=text, model=self._model_name, raw={"vllm_output": str(outputs[0])}
        )

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_ctx: int | None = None,
    ) -> AsyncIterator[str]:
        # Streaming is supported by vLLM but the in-process API
        # delivers complete outputs. Async-iterator over the final
        # text in chunks keeps the interface contract.
        response = await self.chat(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            num_ctx=num_ctx,
        )

        async def _gen() -> AsyncIterator[str]:
            yield response.content

        return _gen()

    async def embed(
        self,
        model: str,
        text: str,
    ) -> EmbedResponse:
        del model, text
        raise LLMBackendError(
            "VLLMInProcessBackend does not implement embeddings yet. "
            "Use a dedicated embedding backend (Ollama, OpenAI) for that."
        )

    async def is_available(self) -> bool:
        # The backend is "available" once vLLM imports cleanly. We do
        # NOT eagerly load the model here — that happens on first chat().
        try:
            import vllm  # type: ignore[import-not-found]  # noqa: F401

            return True
        except ImportError:
            return False

    async def list_models(self) -> list[str]:
        return [self._model_name]

    async def close(self) -> None:
        # vLLM engine cleanup happens at process exit. Explicit shutdown
        # would require holding a reference to the engine_core
        # subprocess; the default GC path is sufficient for our usage.
        self._llm = None


__all__ = ["VLLMInProcessBackend"]
