"""Embedding client with provider strategy and cache. [B§4.7, B§12]

Generates embeddings via configurable provider (Ollama, OpenAI, Gemini, etc.).
Uses content hash as cache key → same text = no new API call.
"""

from __future__ import annotations

import logging
import math
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import httpx

if TYPE_CHECKING:
    from cognithor.config import CognithorConfig

logger = logging.getLogger("cognithor.memory.embeddings")

# Default: qwen3-embedding:0.6b via Ollama (higher accuracy, multilingual)
DEFAULT_MODEL = "qwen3-embedding:0.6b"
DEFAULT_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_DIMENSIONS = 1024


@dataclass
class EmbeddingResult:
    """Result of an embedding computation."""

    vector: list[float]
    model: str
    dimensions: int
    cached: bool = False


@dataclass
class EmbeddingStats:
    """Statistiken ueber Embedding-Operationen."""

    total_requests: int = 0
    cache_hits: int = 0
    api_calls: int = 0
    errors: int = 0

    @property
    def cache_hit_rate(self) -> float:
        """Compute the cache-hit rate (0.0--1.0)."""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests


# ============================================================================
# EmbeddingProvider ABC + Implementierungen
# ============================================================================


class EmbeddingProvider(ABC):
    """Abstrakte Basisklasse fuer Embedding-Provider.

    Kapselt die HTTP-Logik fuer verschiedene Embedding-APIs.
    Cache, LRU, and stats remain in EmbeddingClient.
    """

    @abstractmethod
    async def embed_single(self, model: str, text: str) -> list[float]:
        """Generate a single embedding vector."""
        ...

    @abstractmethod
    async def embed_batch_raw(self, model: str, texts: list[str]) -> list[list[float]]:
        """Erzeugt Embeddings fuer eine Liste von Texten."""
        ...

    async def close(self) -> None:
        """Schliesst den HTTP-Client (falls vorhanden)."""


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Embedding-Provider fuer Ollama (POST /api/embed)."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or getattr(self._client, "is_closed", False):
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                trust_env=False,
            )
        return self._client

    async def embed_single(self, model: str, text: str) -> list[float]:
        client = await self._get_client()
        resp = await client.post(
            "/api/embed",
            json={"model": model, "input": text},
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings", [])
        if not embeddings:
            raise ValueError("Keine Embeddings in Ollama-Antwort")
        return cast("list[float]", embeddings[0])

    async def embed_batch_raw(self, model: str, texts: list[str]) -> list[list[float]]:
        client = await self._get_client()
        resp = await client.post(
            "/api/embed",
            json={"model": model, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()
        return cast("list[list[float]]", data.get("embeddings", []))

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    """Embedding-Provider fuer OpenAI-kompatible APIs (POST /embeddings).

    Funktioniert mit: OpenAI, Mistral, GitHub Models, Bedrock, Together, etc.
    """

    def __init__(
        self, api_key: str, base_url: str = "https://api.openai.com/v1", timeout: float = 30.0
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or getattr(self._client, "is_closed", False):
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                headers={"Authorization": f"Bearer {self._api_key}"},
                trust_env=False,
            )
        return self._client

    async def embed_single(self, model: str, text: str) -> list[float]:
        client = await self._get_client()
        resp = await client.post(
            "/embeddings",
            json={"model": model, "input": text},
        )
        resp.raise_for_status()
        data = resp.json()
        return cast("list[float]", data.get("data", [{}])[0].get("embedding", []))

    async def embed_batch_raw(self, model: str, texts: list[str]) -> list[list[float]]:
        client = await self._get_client()
        resp = await client.post(
            "/embeddings",
            json={"model": model, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()
        # OpenAI gibt data[] als Liste zurueck, sortiert nach index
        items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
        return [item.get("embedding", []) for item in items]

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Embedding-Provider fuer Google Gemini (embedContent REST API)."""

    API_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, api_key: str, timeout: float = 30.0) -> None:
        self._api_key = api_key
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or getattr(self._client, "is_closed", False):
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                trust_env=False,
                headers={"x-goog-api-key": self._api_key},
            )
        return self._client

    async def embed_single(self, model: str, text: str) -> list[float]:
        client = await self._get_client()
        url = f"{self.API_URL}/models/{model}:embedContent"
        resp = await client.post(
            url,
            json={"content": {"parts": [{"text": text}]}},
        )
        resp.raise_for_status()
        data = resp.json()
        return cast("list[float]", data.get("embedding", {}).get("values", []))

    async def embed_batch_raw(self, model: str, texts: list[str]) -> list[list[float]]:
        client = await self._get_client()
        url = f"{self.API_URL}/models/{model}:batchEmbedContents"
        requests = [
            {"model": f"models/{model}", "content": {"parts": [{"text": t}]}} for t in texts
        ]
        resp = await client.post(url, json={"requests": requests})
        resp.raise_for_status()
        data = resp.json()
        return [emb.get("values", []) for emb in data.get("embeddings", [])]

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


class NullEmbeddingProvider(EmbeddingProvider):
    """Dummy-Provider fuer Backends ohne Embedding-API.

    Wirft NotImplementedError — die Vektor-Suche wird deaktiviert,
    BM25+Graph bleiben funktional (search.py faengt alle Exceptions ab).
    """

    def __init__(self, backend_name: str = "unknown") -> None:
        self._backend_name = backend_name

    async def embed_single(self, model: str, text: str) -> list[float]:
        raise NotImplementedError(
            f"Backend '{self._backend_name}' bietet keine Embedding-API. "
            f"Vektor-Suche deaktiviert, BM25+Graph weiterhin aktiv."
        )

    async def embed_batch_raw(self, model: str, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(
            f"Backend '{self._backend_name}' bietet keine Embedding-API. "
            f"Vektor-Suche deaktiviert, BM25+Graph weiterhin aktiv."
        )


# ============================================================================
# Factory
# ============================================================================

# Backends die den OpenAI-kompatiblen /embeddings-Endpoint unterstuetzen
_OPENAI_COMPAT_EMBEDDING_BACKENDS = {"openai", "mistral", "github", "bedrock"}

# Backends ohne eigene Embedding-API
_NO_EMBEDDING_BACKENDS = {
    "anthropic",
    "groq",
    "deepseek",
    "together",
    "openrouter",
    "xai",
    "cerebras",
    "huggingface",
    "moonshot",
}


def _get_api_key_and_url(config: CognithorConfig, backend: str) -> tuple[str, str]:
    """Return (api_key, base_url) for a backend type."""
    from cognithor.config import _PROVIDER_BASE_URLS

    key_map: dict[str, str] = {
        "openai": "openai_api_key",
        "mistral": "mistral_api_key",
        "github": "github_api_key",
        "bedrock": "bedrock_api_key",
        "groq": "groq_api_key",
        "deepseek": "deepseek_api_key",
        "together": "together_api_key",
        "openrouter": "openrouter_api_key",
        "xai": "xai_api_key",
        "cerebras": "cerebras_api_key",
        "huggingface": "huggingface_api_key",
        "moonshot": "moonshot_api_key",
    }

    api_key = getattr(config, key_map.get(backend, ""), "")

    if backend == "openai":
        base_url = getattr(config, "openai_base_url", "https://api.openai.com/v1")
    else:
        base_url = _PROVIDER_BASE_URLS.get(backend, "")

    return api_key, base_url


def create_embedding_provider(config: CognithorConfig) -> EmbeddingProvider:
    """Factory: Create the appropriate EmbeddingProvider based on the config.

    Returns:
        Konfigurierter EmbeddingProvider fuer das aktive Backend.
    """
    backend = config.llm_backend_type

    if backend == "ollama":
        return OllamaEmbeddingProvider(
            base_url=config.ollama.base_url,
            timeout=float(config.ollama.timeout_seconds),
        )

    if backend == "gemini":
        return GeminiEmbeddingProvider(
            api_key=config.gemini_api_key,
        )

    if backend in _OPENAI_COMPAT_EMBEDDING_BACKENDS:
        api_key, base_url = _get_api_key_and_url(config, backend)
        return OpenAICompatibleEmbeddingProvider(
            api_key=api_key,
            base_url=base_url,
        )

    # Backends without embedding API: fall back to OpenAI if key available
    if backend in _NO_EMBEDDING_BACKENDS:
        if config.openai_api_key:
            logger.info(
                "Backend '%s' hat keine Embedding-API, nutze OpenAI-Embeddings als Fallback",
                backend,
            )
            return OpenAICompatibleEmbeddingProvider(
                api_key=config.openai_api_key,
                base_url=getattr(config, "openai_base_url", "https://api.openai.com/v1"),
            )
        logger.warning(
            "Backend '%s' hat keine Embedding-API und kein OpenAI-Key gesetzt. "
            "Vektor-Suche deaktiviert, BM25+Graph weiterhin aktiv.",
            backend,
        )
        return NullEmbeddingProvider(backend_name=backend)

    # Unknown backend → Ollama-Fallback
    logger.warning("Unbekanntes Backend '%s', nutze Ollama-Embedding-Provider", backend)
    return OllamaEmbeddingProvider(
        base_url=config.ollama.base_url,
        timeout=float(config.ollama.timeout_seconds),
    )


# ============================================================================
# EmbeddingClient (Cache + Stats, delegiert an Provider)
# ============================================================================


class EmbeddingClient:
    """Async Embedding-Client mit In-Memory-Cache.

    Delegiert API-Calls an einen konfigurierbaren EmbeddingProvider.
    Cache basiert auf Content-Hash (SHA-256).
    Uses an LRU-bounded OrderedDict to prevent unbounded memory growth.
    """

    _MAX_CACHE_SIZE: int = 50_000

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        dimensions: int = DEFAULT_DIMENSIONS,
        timeout: float = 30.0,
        provider: EmbeddingProvider | None = None,
    ) -> None:
        """Initialisiert den Embedding-Client mit Modell und Cache.

        Args:
            model: Name des Embedding-Modells.
            base_url: Base-URL (nur fuer Rueckwaertskompatibilitaet,
                wird ignoriert wenn provider gesetzt).
            dimensions: Erwartete Vektor-Dimensionalitaet.
            timeout: Request-Timeout in Sekunden.
            provider: Optionaler EmbeddingProvider. Default: OllamaEmbeddingProvider.
        """
        self._model = model
        self._dimensions = dimensions
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._stats = EmbeddingStats()
        # Provider: explizit gesetzt oder Ollama-Default (Rueckwaertskompatibilitaet)
        self._provider = provider or OllamaEmbeddingProvider(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
        )

    @property
    def model(self) -> str:
        """Name of the embedding model."""
        return self._model

    @property
    def dimensions(self) -> int:
        """Dimensionalitaet der Embedding-Vektoren."""
        return self._dimensions

    @property
    def stats(self) -> EmbeddingStats:
        """Return the current cache statistics."""
        return self._stats

    def _cache_put(self, key: str, vector: list[float]) -> None:
        """Insert or update a single cache entry, evicting the oldest if full."""
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._cache[key] = vector
        else:
            self._cache[key] = vector
            while len(self._cache) > self._MAX_CACHE_SIZE:
                self._cache.popitem(last=False)  # evict oldest (FIFO/LRU)

    def load_cache(self, entries: dict[str, list[float]]) -> int:
        """Laedt Embeddings in den In-Memory-Cache.

        Args:
            entries: {content_hash: vector} Dict.

        Returns:
            Anzahl geladener Eintraege.
        """
        for key, vector in entries.items():
            self._cache_put(key, vector)
        return len(entries)

    def get_cached(self, content_hash: str) -> list[float] | None:
        """Prueft ob ein Embedding im Cache liegt.

        Promotes the entry to most-recently-used on access (LRU).
        """
        if content_hash in self._cache:
            self._cache.move_to_end(content_hash)
            return self._cache[content_hash]
        return None

    async def embed_text(self, text: str, content_hash: str = "") -> EmbeddingResult:
        """Generiert ein Embedding fuer einen Text.

        Args:
            text: The text to embed.
            content_hash: Optional Cache-Key.

        Returns:
            EmbeddingResult mit Vektor.
        """
        self._stats.total_requests += 1

        # Cache-Check (promote to most-recently-used on hit)
        if content_hash and content_hash in self._cache:
            self._stats.cache_hits += 1
            self._cache.move_to_end(content_hash)
            return EmbeddingResult(
                vector=self._cache[content_hash],
                model=self._model,
                dimensions=self._dimensions,
                cached=True,
            )

        # API-Call via Provider
        self._stats.api_calls += 1
        try:
            vector = await self._provider.embed_single(self._model, text)

            # Save to cache (bounded LRU)
            if content_hash:
                self._cache_put(content_hash, vector)

            return EmbeddingResult(
                vector=vector,
                model=self._model,
                dimensions=len(vector),
                cached=False,
            )

        except (httpx.HTTPError, ValueError, KeyError, NotImplementedError) as e:
            self._stats.errors += 1
            logger.error("Embedding-Fehler für '%s...': %s", text[:50], e)
            raise

    async def embed_batch(
        self,
        texts: list[str],
        content_hashes: list[str] | None = None,
        *,
        batch_size: int = 32,
    ) -> list[EmbeddingResult | None]:
        """Generiert Embeddings fuer mehrere Texte.

        Nutzt Cache wo moeglich, batcht API-Calls.

        Args:
            texts: Liste von Texten.
            content_hashes: Optional zugehoerige Cache-Keys.
            batch_size: Max Texte pro API-Call.

        Returns:
            Liste von EmbeddingResults (gleiche Reihenfolge wie Input).
            Eintraege koennen None sein wenn der API-Call fehlgeschlagen ist.
        """
        if content_hashes is None:
            content_hashes = [""] * len(texts)

        if len(texts) != len(content_hashes):
            raise ValueError("texts und content_hashes müssen gleich lang sein")

        results: list[EmbeddingResult | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        # Schritt 1: Cache-Hits sammeln (promote to most-recently-used)
        for i, (text, h) in enumerate(zip(texts, content_hashes, strict=False)):
            self._stats.total_requests += 1
            if h and h in self._cache:
                self._stats.cache_hits += 1
                self._cache.move_to_end(h)
                results[i] = EmbeddingResult(
                    vector=self._cache[h],
                    model=self._model,
                    dimensions=self._dimensions,
                    cached=True,
                )
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Schritt 2: Uncached in Batches embedden via Provider
        for batch_start in range(0, len(uncached_texts), batch_size):
            batch = uncached_texts[batch_start : batch_start + batch_size]
            batch_indices = uncached_indices[batch_start : batch_start + batch_size]

            try:
                self._stats.api_calls += 1
                embeddings = await self._provider.embed_batch_raw(self._model, batch)

                for _j, (idx, vec) in enumerate(zip(batch_indices, embeddings, strict=False)):
                    h = content_hashes[idx]
                    if h:
                        self._cache_put(h, vec)
                    results[idx] = EmbeddingResult(
                        vector=vec,
                        model=self._model,
                        dimensions=len(vec),
                        cached=False,
                    )

            except (httpx.HTTPError, ValueError, NotImplementedError) as e:
                self._stats.errors += 1
                logger.error("Batch-Embedding-Fehler: %s", e)
                # Fehlende Ergebnisse bleiben None (kein Zero-Vektor)

        # Log warning if some entries failed (remain None)
        failed = sum(1 for r in results if r is None)
        if failed:
            logger.warning(
                "embed_batch: %d/%d Embeddings fehlgeschlagen (None)",
                failed,
                len(texts),
            )
        return results

    async def close(self) -> None:
        """Schliesst den HTTP-Client."""
        await self._provider.close()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute the cosine similarity of two vectors.

    Returns:
        Value between -1.0 and 1.0. Higher = more similar.
    """
    if len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)
