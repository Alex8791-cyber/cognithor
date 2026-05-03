#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Sprint-21 — operator-side smoke test for the central VLLMBackend.

Use this to verify a vLLM HTTP endpoint is reachable from a host
machine before relying on it from any cognithor module (PSE channel,
gateway, MCP tools, ...).

Default target: ``http://localhost:8000/v1`` (matches
``CognithorConfig.vllm_base_url`` default and a vanilla local
``vllm serve`` invocation). Override with ``--base-url`` or the
``VLLM_BASE_URL`` env var.

Smoke matrix:

* Phase 1 — connectivity: ``is_available()`` + ``/health`` ping
* Phase 2 — models list: ``list_models()`` returns at least one
* Phase 3 — text chat: simple ``"hi"`` prompt → non-empty response
* Phase 4 — multimodal chat: 1×1 PNG + ``"describe the image"`` prompt
  (only when ``--multimodal`` is set; some text-only models will 400)

Each phase prints PASS / FAIL with a one-line reason. Exit code 0
when all selected phases pass; non-zero otherwise — suitable for CI
gating once an HTTP endpoint is part of the test infra.

Usage::

    # default localhost vLLM
    python scripts/smoke_vllm_backend.py

    # remote / WSL endpoint
    python scripts/smoke_vllm_backend.py --base-url http://192.168.1.42:8000/v1

    # multimodal probe
    python scripts/smoke_vllm_backend.py --multimodal

    # specific model from list_models
    python scripts/smoke_vllm_backend.py --model sakamakismile/Qwen3.6-27B-NVFP4
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import sys
from typing import Any

# 1×1 transparent PNG (80 bytes, base64-decoded → header-valid PNG).
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def _emit(phase: str, status: str, msg: str) -> None:
    """Print a structured PASS/FAIL line with a phase marker."""
    marker = "[PASS]" if status == "pass" else "[FAIL]"
    print(f"{marker} phase={phase}: {msg}")


async def _phase_connectivity(backend: Any) -> bool:
    try:
        ok = await backend.is_available()
    except Exception as exc:
        _emit("connectivity", "fail", f"is_available() raised: {type(exc).__name__}: {exc}")
        return False
    if not ok:
        _emit("connectivity", "fail", "is_available() returned False — /health unreachable")
        return False
    _emit("connectivity", "pass", "is_available() True")
    return True


async def _phase_models(backend: Any, requested_model: str | None) -> tuple[bool, str | None]:
    try:
        models = await backend.list_models()
    except Exception as exc:
        _emit("models", "fail", f"list_models() raised: {type(exc).__name__}: {exc}")
        return False, None
    if not models:
        _emit("models", "fail", "list_models() returned []")
        return False, None
    chosen = requested_model or models[0]
    if requested_model and requested_model not in models:
        _emit(
            "models",
            "fail",
            f"requested model {requested_model!r} not in list_models()={models[:3]}...",
        )
        return False, None
    _emit("models", "pass", f"server reports {len(models)} model(s); using {chosen!r}")
    return True, chosen


async def _phase_text_chat(backend: Any, model: str) -> bool:
    try:
        response = await backend.chat(
            model=model,
            messages=[{"role": "user", "content": "Reply with a single word: hello."}],
            temperature=0.0,
        )
    except Exception as exc:
        _emit("text_chat", "fail", f"chat() raised: {type(exc).__name__}: {exc}")
        return False
    content = (getattr(response, "content", "") or "").strip()
    if not content:
        _emit("text_chat", "fail", "chat() returned empty content")
        return False
    snippet = content[:80].replace("\n", " ")
    _emit("text_chat", "pass", f'response: "{snippet}{"..." if len(content) > 80 else ""}"')
    return True


async def _phase_multimodal_chat(backend: Any, model: str) -> bool:
    image_data_uri = "data:image/png;base64," + _TINY_PNG_B64
    try:
        response = await backend.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_uri}},
                        {"type": "text", "text": "Describe this image in one sentence."},
                    ],
                }
            ],
            temperature=0.0,
        )
    except Exception as exc:
        _emit(
            "multimodal_chat",
            "fail",
            f"chat() raised: {type(exc).__name__}: {exc} (likely text-only model)",
        )
        return False
    content = (getattr(response, "content", "") or "").strip()
    if not content:
        _emit("multimodal_chat", "fail", "chat() returned empty content")
        return False
    snippet = content[:80].replace("\n", " ")
    _emit(
        "multimodal_chat",
        "pass",
        f'response: "{snippet}{"..." if len(content) > 80 else ""}"',
    )
    return True


async def run_smoke(args: argparse.Namespace) -> int:
    try:
        from cognithor.core.vllm_backend import VLLMBackend
    except ImportError as exc:
        print(f"FATAL: cognithor.core.vllm_backend not importable: {exc}")
        return 2

    backend = VLLMBackend(base_url=args.base_url)
    failures = 0
    try:
        connectivity_ok = await _phase_connectivity(backend)
        if not connectivity_ok:
            return 1  # all later phases pointless without connectivity

        models_ok, model = await _phase_models(backend, args.model)
        if not models_ok or model is None:
            failures += 1
        else:
            if not await _phase_text_chat(backend, model):
                failures += 1
            if args.multimodal and not await _phase_multimodal_chat(backend, model):
                failures += 1
    finally:
        with contextlib.suppress(Exception):
            await backend.close()
    return 0 if failures == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test the cognithor.core.vllm_backend.VLLMBackend "
        "against a running vLLM HTTP endpoint.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
        help="vLLM OpenAI-compatible base URL (default: $VLLM_BASE_URL or http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Specific model id to test (default: first model from list_models)",
    )
    parser.add_argument(
        "--multimodal",
        action="store_true",
        help="Also run the multimodal-image phase (text-only models will fail this)",
    )
    args = parser.parse_args()

    print(f"smoke_vllm_backend: target {args.base_url}")
    rc = asyncio.run(run_smoke(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
