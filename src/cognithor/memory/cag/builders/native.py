from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cognithor.memory.cag.builders.base import CacheBuilder

if TYPE_CHECKING:
    from pathlib import Path

    from cognithor.memory.cag.models import CacheEntry


class NativeLlamaCppBuilder(CacheBuilder):
    """CAG builder that leverages llama.cpp native KV-cache state save/load."""

    async def is_available(self) -> bool:
        """Return True only if llama_cpp is importable."""
        try:
            import llama_cpp  # noqa: F401

            return True
        except ImportError:
            return False

    def supports_native_state(self) -> bool:
        return True

    async def prepare_prefix(self, entries: list[CacheEntry], model_id: str) -> str:
        """Build the same deterministic prefix text as PrefixCacheBuilder."""
        if not entries:
            return ""
        sorted_entries = sorted(entries, key=lambda e: e.cache_id)
        blocks = [f"[CAG:{e.cache_id}]\n{e.normalized_text}" for e in sorted_entries]
        return "\n\n".join(blocks)

    async def build_state(self, content: str, model_path: str, target_path: Path) -> Path:
        """Tokenize content, evaluate, and save KV-cache state to disk."""
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError("llama_cpp is required for native KV-cache state building") from exc

        llm = Llama(model_path=model_path, n_ctx=4096, verbose=False)
        tokens = llm.tokenize(content.encode("utf-8"))
        llm.eval(tokens)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        llm.save_state(str(target_path))
        return target_path

    async def load_state(self, state_path: Path, model_path: str) -> Any:
        """Load a previously saved KV-cache state."""
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError("llama_cpp is required for native KV-cache state loading") from exc

        llm = Llama(model_path=model_path, n_ctx=4096, verbose=False)
        llm.load_state(str(state_path))
        return llm
