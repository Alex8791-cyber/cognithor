"""Model registry with dynamic provider discovery and cached JSON fallback."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, cast

import httpx

logger = logging.getLogger(__name__)

_REGISTRY_PATH = Path(__file__).parent / "model_registry.json"
_CUSTOM_OPTION = "[ Custom... ]"


class ModelRegistry:
    """Discovers available LLM models from providers with a cached fallback."""

    def __init__(self, registry_path: Path | None = None) -> None:
        self._path = registry_path or _REGISTRY_PATH
        self._data: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        if self._data is None:
            with open(self._path, encoding="utf-8") as f:
                self._data = json.load(f)
        return self._data

    def _provider_block(self, provider: str) -> dict[str, Any] | None:
        data = self._load()
        return cast("dict[str, Any] | None", data.get("providers", {}).get(provider))

    # ------------------------------------------------------------------
    # Cached (offline) access
    # ------------------------------------------------------------------

    def get_cached_models(self, provider: str) -> list[str]:
        """Return cached model list for *provider*, always ending with Custom."""
        block = self._provider_block(provider)
        if block is None:
            return [_CUSTOM_OPTION]
        return [*list(block.get("models", [])), _CUSTOM_OPTION]

    # ------------------------------------------------------------------
    # Live discovery
    # ------------------------------------------------------------------

    async def discover_models(self, provider: str) -> list[str]:
        """Query the provider API for models; fall back to cached on error."""
        block = self._provider_block(provider)
        if block is None:
            return [_CUSTOM_OPTION]

        url: str | None = block.get("discovery_url")
        if not url:
            return self.get_cached_models(provider)

        headers: dict[str, str] = {}
        api_key_env = block.get("api_key_env")
        if api_key_env:
            api_key = os.environ.get(api_key_env, "")
            if api_key:
                if provider == "gemini":
                    # Gemini uses query-param auth; append to URL
                    sep = "&" if "?" in url else "?"
                    url = f"{url}{sep}key={api_key}"
                else:
                    headers["Authorization"] = f"Bearer {api_key}"

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                payload = resp.json()
                models = self._parse_response(provider, payload)
                if models:
                    return [*models, _CUSTOM_OPTION]
        except Exception:
            logger.debug("Live discovery failed for %s, falling back to cache", provider)

        return self.get_cached_models(provider)

    def discover_models_sync(self, provider: str) -> list[str]:
        """Synchronous wrapper around :meth:`discover_models`."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self.discover_models(provider)).result()
        return asyncio.run(self.discover_models(provider))

    # ------------------------------------------------------------------
    # Response parsing per provider
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(provider: str, payload: dict[str, Any]) -> list[str]:
        if provider == "ollama":
            return [m["name"] for m in payload.get("models", [])]
        if provider in ("openai", "lmstudio"):
            return [m["id"] for m in payload.get("data", [])]
        if provider == "gemini":
            out: list[str] = []
            for m in payload.get("models", []):
                methods = m.get("supportedGenerationMethods", [])
                if "generateContent" in methods:
                    name = m.get("name", "")
                    # Strip leading "models/" prefix
                    if name.startswith("models/"):
                        name = name[len("models/") :]
                    out.append(name)
            return out
        return []


# ------------------------------------------------------------------
# CLI entry point: python -m cognithor.cli.model_registry --update
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import datetime

    parser = argparse.ArgumentParser(description="Model registry updater")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Query all providers and refresh the cached JSON",
    )
    args = parser.parse_args()

    if args.update:
        registry = ModelRegistry()
        data = registry._load()
        for prov in list(data["providers"]):
            print(f"[*] Discovering models for {prov} ...")
            try:
                models = registry.discover_models_sync(prov)
                # Remove the Custom sentinel before saving
                models = [m for m in models if m != _CUSTOM_OPTION]
                if models:
                    data["providers"][prov]["models"] = models
                    print(f"    Found {len(models)} model(s)")
                else:
                    print("    No models found, keeping cached list")
            except Exception as exc:
                print(f"    Error: {exc}")
        data["updated"] = datetime.date.today().isoformat()
        with open(_REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print("[OK] Registry updated.")
    else:
        parser.print_help()
