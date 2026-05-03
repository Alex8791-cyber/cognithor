"""CAG cache builders — prefix-based and native llama.cpp."""

from __future__ import annotations

from typing import Any


def get_builder(backend: str) -> Any:
    """Factory: return the appropriate CacheBuilder for the backend."""
    if backend in ("auto", "prefix"):
        from cognithor.memory.cag.builders.prefix import PrefixCacheBuilder

        return PrefixCacheBuilder()
    elif backend == "llamacpp_native":
        from cognithor.memory.cag.builders.native import NativeLlamaCppBuilder

        return NativeLlamaCppBuilder()
    else:
        raise ValueError(
            f"Unknown CAG backend: {backend!r}. Use 'auto', 'prefix', or 'llamacpp_native'."
        )
